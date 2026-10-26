"""One-command entry point:  python run_tcresnet.py --musan-root ... --cache-dir ... --train-set clean|aug"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kwsba.run_tcresnet import main  # noqa: E402

if __name__ == "__main__":
    main()
