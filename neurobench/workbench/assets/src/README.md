# Workbench JavaScript Sources

`neurobench/workbench/assets/workbench.js` is the served browser bundle, but it
is generated from the ordered source files in this directory. Edit these
numbered files, then rebuild and check the bundle:

```bash
python3 tools/build_workbench_assets.py
python3 tools/build_workbench_assets.py --check
```

The bundle is intentionally concatenated rather than ES-module based so the
existing browser globals and static-file workflow keep working. Keep numeric
prefixes stable so execution order is explicit in code review.

Use `--init-from-current` only for recovery or a fresh migration branch; normal
changes should edit the ordered source files directly.
