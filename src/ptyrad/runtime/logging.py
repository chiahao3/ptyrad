import io
import logging
import os
import warnings
from typing import Literal

from ptyrad.utils.time import get_time

# A module-level variable to store our "Recording Engineer"
_active_manager = None

VERBOSITY_MAPPING = {'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL}

def report(message, verbosity: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'):
    """
    Dual purpose print / log, ensures output reaches the user even if they haven't initialized PtyRAD logging.
    
    This report function is used exclusively for diagnostic functions like print_system_info, print_gpu_info, etc.
    """
    target_level = VERBOSITY_MAPPING.get(verbosity, logging.INFO)
    logger = logging.getLogger('ptyrad')
    
    # 1. Emit to the logger hierarchy
    logger.log(level=target_level, msg=message)
    
    # 2. The Bulletproof Check:
    # We fallback to print() ONLY if:
    # a) There are no handlers at all (Newbie mode)
    # OR 
    # b) Handlers exist, but the current logger level is too high to show this message
    #    (e.g., user set level to ERROR, but we are reporting vital system INFO)
    
    has_handlers = logger.hasHandlers() or logging.getLogger().hasHandlers()
    is_enabled = logger.isEnabledFor(target_level)

    if not has_handlers or not is_enabled:
        # Fallback to standard print so the human actually sees the diagnostic
        print(message)

def get_logging_manager():
    """Access the manager from anywhere in the library."""
    return _active_manager

class RankZeroFilter(logging.Filter):
    def filter(self, record):
        import sys
        # Only check rank if torch is already loaded in memory
        if 'torch' in sys.modules:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        return True

class LoggingManager:
    def __init__(self, 
                 log_file='ptyrad_log.txt', 
                 log_dir='auto', 
                 prefix_time='datetime', 
                 prefix_date=None, 
                 prefix_jobid=0, 
                 append_to_file=True, 
                 show_timestamp=True,
                 show_config=True, 
                 verbosity: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'DEBUG', **kwargs):
        global _active_manager
        _active_manager = self # Register this instance as the active one
        self.logger = logging.getLogger('ptyrad')
        target_level = VERBOSITY_MAPPING.get(verbosity.upper(), logging.DEBUG)
        self.logger.setLevel(target_level)
        
        # Clear all existing handlers to re-instantiate the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.log_file       = log_file
        self.log_dir        = log_dir
        self.flush_file     = log_file is not None
        # Backward compatibility: if prefix_date is set (legacy), use it, else use prefix_time
        if prefix_date is not None:
            warnings.warn(
                "The 'prefix_date' argument is deprecated and will be removed by 2025 Aug."
                "Please use 'prefix_time' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.prefix_time = prefix_date
        else:
            self.prefix_time = prefix_time
        self.prefix_jobid   = prefix_jobid
        self.append_to_file = append_to_file
        self.show_timestamp = show_timestamp

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.addFilter(RankZeroFilter())
        self.console_handler.setLevel(target_level)
        formatter = logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s')
        self.console_handler.setFormatter(formatter)
        
        # Create a buffer for file logs
        self.log_buffer = io.StringIO()
        self.buffer_handler = logging.StreamHandler(self.log_buffer)
        self.buffer_handler.addFilter(RankZeroFilter())
        self.buffer_handler.setLevel(target_level)
        self.buffer_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.buffer_handler)
        
        # Print logger information, these will take effect immediately after attaching handlers
        if show_config:
            self._show_manager_config()

    def _show_manager_config(self):
        self.logger.info("### PtyRAD LoggingManager configuration ###")
        self.logger.info(f"log_file       = '{self.log_file}'. If log_file = None, no log file will be created.")
        self.logger.info(f"log_dir        = '{self.log_dir}'. If log_dir = 'auto', then log will be saved to `output_path` or 'logs/'.")
        self.logger.info(f"flush_file     = {self.flush_file}. Automatically set to True if `log_file is not None`")
        self.logger.info(f"prefix_time    = {self.prefix_time}. If true, preset strings ('date', 'time', 'datetime'), or a string of time format, a datetime str is prefixed to the `log_file`.")
        self.logger.info(f"prefix_jobid   = '{self.prefix_jobid}'. If not 0, it'll be prefixed to the log file. This is used for hypertune mode with multiple GPUs.")
        self.logger.info(f"append_to_file = {self.append_to_file}. If true, logs will be appended to the existing file. If false, the log file will be overwritten.")
        self.logger.info(f"show_timestamp = {self.show_timestamp}. If true, the printed information will contain a timestamp.")
        self.logger.info(' ')        

    def flush_to_file(self, log_dir=None, append_to_file=None):
        """
        Flushes buffered logs to a file based on user-defined file mode (append or write)
        """
        
        # Set log_dir
        if log_dir is None:
            if self.log_dir == 'auto':
                log_dir = 'logs'
            else:
                log_dir = self.log_dir

        # Set file_mode
        if append_to_file is None:
            append_to_file = self.append_to_file
        file_mode = 'a' if append_to_file else 'w'
        
        # Set file name
        log_file = self.log_file
        if self.prefix_jobid != 0:
            log_file = str(self.prefix_jobid).zfill(2) + '_' + log_file
        
        # Set prefix_time
        prefix_time = self.prefix_time
        if prefix_time is True or (isinstance(prefix_time, str) and prefix_time):
            time_str = get_time(prefix_time)
            log_file = f"{time_str}_{log_file}"
        
        show_timestamp = self.show_timestamp
        
        if self.flush_file:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)

            # Write the buffered logs to the specified file
            try:
                with open(log_file_path, file_mode, encoding="utf-8") as f:
                    f.write(self.log_buffer.getvalue())
            except UnicodeEncodeError as e:
                self.logger.warning(f"Failed to write log due to Unicode issue: {e}")
                with open(log_file_path, file_mode, encoding="ascii", errors="replace") as f:
                    f.write(self.log_buffer.getvalue())

            # Clear the buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

            # Set up a file handler for future logging to the file
            self.file_handler = logging.FileHandler(log_file_path, mode='a')  # Always append after initial flush
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s'))
            self.logger.addHandler(self.file_handler)
            self.logger.info(f"### Log file is flushed (created) as {log_file_path} ###")
        else:
            self.file_handler = None
            self.logger.warning(f"### Log file is not flushed (created) because log_file is set to {self.log_file} ###")
        self.logger.info(' ')
        
    def close(self):
        """Closes the file handler if it exists."""
        if self.file_handler is not None:
            self.file_handler.flush()
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None