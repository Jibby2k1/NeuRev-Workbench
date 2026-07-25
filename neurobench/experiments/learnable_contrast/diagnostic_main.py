from __future__ import annotations
import argparse,json
from pathlib import Path
from .diagnostic import DiagnosticConfig,preflight,run
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--artifact-dir',type=Path)
a=p.parse_args();c=DiagnosticConfig.load(a.config);result=preflight(c,a.artifact_dir) if a.preflight else run(c);print(json.dumps(result,indent=2,sort_keys=True))
