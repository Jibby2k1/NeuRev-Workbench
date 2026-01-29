# data_loader.py
"""Functions for loading video and ground truth data."""
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import tifffile
except ImportError:
    tifffile = None

def load_video(filepath: Path):
    """Loads a video from a TIFF file using memory mapping."""
    if not tifffile:
        raise ImportError("tifffile library is required to load video data.")
    logging.info(f"Loading video from: {filepath}")
    return tifffile.memmap(filepath)

def parse_ground_truth_csv(filepath: Optional[Path]) -> Dict:
    """Parses a ground truth CSV file into a dictionary."""
    if not pd:
        raise ImportError("pandas library is required to parse CSV data.")
    if not filepath or not filepath.exists():
        logging.warning(f"Ground truth file not specified or not found: {filepath}")
        return {}
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    
    gt_data = defaultdict(list)
    for _, row in df.iterrows():
        for frame_num in range(int(row['Start Frame']), int(row['End Frame']) + 1):
            gt_data[frame_num].append((float(row['X']), float(row['Y']), int(row['ID'])))
            
    logging.info(f"Loaded {len(df)} ground truth tracks from: {filepath}")
    return dict(gt_data)