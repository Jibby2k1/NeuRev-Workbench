# Workbench JavaScript Sources

`neurobench/workbench/assets/workbench.js` is the served browser bundle, but it
is generated from the ordered source files in this directory. Production order
is declared explicitly in `bundle_sources.txt`; numeric filenames remain useful
for readability but are not the production build contract. Edit the numbered
files and the manifest together, then rebuild and check the bundle:

```bash
python3 tools/build_workbench_assets.py
python3 tools/build_workbench_assets.py --check
```

The bundle is intentionally concatenated rather than ES-module based so the
existing browser globals and static-file workflow keep working. The production
build fails if a manifest entry is missing or if an unlisted `.js` file appears
in this directory. This prevents a new module from being silently omitted or
included in the browser runtime.

Custom `--source-dir` directories retain the recovery/test behavior: their
`.js` files are bundled in sorted filename order and do not require a manifest.

Use `--init-from-current` only for recovery or a fresh migration branch; normal
changes should edit the declared source files directly. Recovery into a custom
source directory continues to use sorted fallback order. The builder rejects
`--init-from-current` when it targets this production directory before writing
anything; pass a separate `--source-dir` for recovery initialization.
