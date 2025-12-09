import os
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Please install fpdf2: pip install fpdf2")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent

# Colors
COLOR_PRIMARY = (0, 242, 234)
COLOR_SECONDARY = (30, 30, 40)
COLOR_TEXT = (60, 60, 60)
COLOR_RED = (255, 107, 107)
COLOR_TEAL = (78, 205, 196)
COLOR_YELLOW = (255, 230, 109)
COLOR_GREEN = (149, 225, 211)
