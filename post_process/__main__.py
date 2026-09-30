"""
__main__.py - Vstupní bod pro spuštění balíčku post_process z terminálu:
  python -m post_process [parametry]
"""

import sys

try:
    from .cli import main
except ImportError:
    from cli import main

if __name__ == "__main__":
    sys.exit(main())
