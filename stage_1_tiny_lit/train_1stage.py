"""Train the Tiny Self-DACE++ illumination network (9 LAEC + 3 HASC)."""
from pathlib import Path
import sys

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent))
from stage1_training import main

if __name__ == "__main__":
    main(STAGE_DIR, "tiny")








