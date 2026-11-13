"""One-command entry point for the Stage 4 experiments:  python run_stage4.py --musan-root ... (see --help)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kwsba.run_stage4 import main  # noqa: E402

if __name__ == "__main__":
    main()
