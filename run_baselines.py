"""One-command entry point:  python run_baselines.py --sc-root ... --musan-root ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kwsba.run_baselines import main  # noqa: E402

if __name__ == "__main__":
    main()
