from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import Config, preflight, run

def main():
    p=argparse.ArgumentParser(description="Guarded weakly-supervised learnable-contrast experiment")
    p.add_argument("--config",required=True); p.add_argument("--preflight",action="store_true")
    p.add_argument("--artifact-dir",type=Path)
    a=p.parse_args(); c=Config.load(a.config)
    result=preflight(c,artifact_dir=a.artifact_dir) if a.preflight else run(c)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
