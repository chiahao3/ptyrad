"""The PtyRAD entry point.

This allows `python -m ptyrad` to run the ptyrad package as a module, similar to pip, black, pytest, etc.
This also allows `accelerate launch -m ptyrad --args` without actually needing a script for multi GPU.
"""

from .cli.main import main

if __name__ == "__main__":
    main()