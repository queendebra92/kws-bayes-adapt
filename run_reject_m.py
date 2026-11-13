import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from kwsba.run_reject_m import main  # noqa: E402

main()
