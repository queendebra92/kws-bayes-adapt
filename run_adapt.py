"""One-command entry point:  python run_adapt.py --musan-root ... --cache-dir ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kwsba.run_adapt import main  # noqa: E402

if __name__ == "__main__":
    main()
