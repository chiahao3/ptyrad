from typing import Union

def get_time(time_format: Union[bool, str, None] = 'date') -> str:
    """
    Returns a formatted timestamp string based on `time_format`.

    Args:
        time_format (bool or str): Controls the time formatting behavior.
            - True: Use default date format ("%Y%m%d").
            - False, None, or "": Disable timestamp and return an empty string.
            - "date": Use date format ("%Y%m%d").
            - "datetime": Use date and time format ("%Y%m%d_%H%M%S").
            - "time": Use time-only format ("%H%M%S").
            - Custom strftime format (e.g., "%Y-%m-%d %H:%M") is also supported.

    Returns:
        str: Formatted timestamp string, or an empty string if disabled.
    """
    from datetime import date, datetime

    if not time_format:
        return ""

    if isinstance(time_format, bool):
        fmt = "%Y%m%d"
    elif isinstance(time_format, str):
        presets = {
            "date": "%Y%m%d",
            "datetime": "%Y%m%d_%H%M%S",
            "time": "%H%M%S",
        }
        fmt = presets.get(time_format.lower(), time_format)
    else:
        raise TypeError(f"time_format must be a bool or str, got {type(time_format).__name__}")

    # Choose datetime vs date object depending on format
    try:
        if any(tok in fmt for tok in ("%H", "%M", "%S")):
            return datetime.now().strftime(fmt)
        else:
            return date.today().strftime(fmt)
    except ValueError as e:
        raise ValueError(f"Invalid time format string: {fmt!r}") from e

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"