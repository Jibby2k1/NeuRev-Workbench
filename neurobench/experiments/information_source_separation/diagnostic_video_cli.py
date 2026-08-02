"""Generate the source-separation diagnostic MP4 suite."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .calibration_config import CalibrationConfig
from .diagnostic_videos import generate_diagnostic_suite
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True); p.add_argument("--calibration-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args(argv)
    result=generate_diagnostic_suite(CalibrationConfig.load(a.config),calibration_root=a.calibration_root,output_dir=a.output_dir); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
