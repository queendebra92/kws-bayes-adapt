import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from kwsba.run_sensitivity_dn import main  # noqa: E402

main()
