"""Run the fused Stage-I Self-DACE++ model."""
from pathlib import Path
import sys

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent))
from inference import main

if __name__ == "__main__":
    main(STAGE_DIR, "small")

		



