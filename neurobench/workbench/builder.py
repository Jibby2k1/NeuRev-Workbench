"""Workbench build helpers shared by CLI tools and future package entrypoints."""
from __future__ import annotations

import hashlib
import json
from importlib import resources
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from neurobench.annotations import migrate_annotations_v3
from neurobench.data.catalog import dataset_id_from_review
from neurobench.manifests import load_dataset_manifest, load_json, manifest_path
from neurobench.pipeline_catalog import catalog_as_dict


def architecture_runs_from_review(data: Mapping[str, Any], review_data_path: Path, dataset_id: str) -> dict[str, Any]:
    """Return a truthful empty run catalog for annotation-only review data.

    ``review_data.json`` and rendered browser frames do not prove that a
    scientific detection pipeline completed. The function remains public for
    compatibility with older callers, but deliberately does not infer a run
    from ROI, event, or frame counts.
    """
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "runs": [],
    }


def load_workbench_asset(name: str, fallback: str = "") -> str:
    """Load packaged workbench assets, using a provided fallback during migration."""
    try:
        asset = resources.files("neurobench.workbench").joinpath("assets", name)
        return asset.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ModuleNotFoundError):
        return fallback.strip()


def workbench_asset_version(
    *,
    css_text: str | None = None,
    js_text: str | None = None,
    html_text: str | None = None,
) -> str:
    """Return the stable version shared by built apps and status tooling."""
    css = load_workbench_asset("workbench.css") if css_text is None else css_text
    js = load_workbench_asset("workbench.js") if js_text is None else js_text
    html = load_workbench_asset("workbench.html") if html_text is None else html_text
    return hashlib.sha256((css + "\0" + js + "\0" + html).encode("utf-8")).hexdigest()[:12]


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _prepared_workbench_data(
    *,
    review_data_path: Path,
    dataset_id: str,
    dataset_manifest: Mapping[str, Any] | None,
    architecture_runs_path: Path | None,
    app_dir: Path | None,
) -> dict[str, Any]:
    """Load and augment review data without changing a source object or file."""

    data = _json_object(review_data_path, label="Review data")
    manifest_payload = dict(dataset_manifest or {})
    review_dataset = data.get("dataset") if isinstance(data.get("dataset"), Mapping) else {}
    data["dataset"] = {
        **review_dataset,
        **{key: value for key, value in manifest_payload.items() if not str(key).startswith("_")},
    }
    data["dataset"].setdefault("dataset_id", dataset_id)
    data["pipelineCatalog"] = catalog_as_dict()

    if architecture_runs_path is not None:
        if not architecture_runs_path.is_file():
            raise FileNotFoundError(f"Architecture-run catalog does not exist: {architecture_runs_path}")
        architecture_runs = _json_object(architecture_runs_path, label="Architecture-run catalog")
    else:
        installed_catalog = app_dir / "architecture_runs.json" if app_dir is not None else None
        embedded_catalog = data.get("architectureRuns")
        if installed_catalog is not None and installed_catalog.is_file():
            architecture_runs = _json_object(installed_catalog, label="Architecture-run catalog")
        elif isinstance(embedded_catalog, Mapping):
            architecture_runs = dict(embedded_catalog)
        else:
            architecture_runs = architecture_runs_from_review(data, review_data_path, dataset_id)
    data["architectureRuns"] = architecture_runs
    return data


def _render_assets_from_data(
    data: Mapping[str, Any],
    *,
    dataset_id: str,
    html_template: str | None,
    css_fallback: str,
    js_fallback: str,
) -> dict[str, bytes]:
    css_text = load_workbench_asset("workbench.css", css_fallback)
    js_text = load_workbench_asset("workbench.js", js_fallback)
    template = html_template.strip() if html_template is not None else load_workbench_asset("workbench.html")
    asset_version = workbench_asset_version(css_text=css_text, js_text=js_text, html_text=template)
    video = data.get("video") if isinstance(data.get("video"), Mapping) else {}
    html = template.format(
        dataset_id=dataset_id,
        frames=video.get("frames", 0),
        asset_version=asset_version,
        data_json=json.dumps(data, separators=(",", ":")).replace("</script>", "<\\/script>"),
    )
    return {
        "index.html": html.encode("utf-8"),
        "workbench.css": (css_text + "\n").encode("utf-8"),
        "workbench.js": (js_text + "\n").encode("utf-8"),
    }


def render_workbench_assets(
    *,
    review_data_path: str | Path,
    dataset_id: str,
    dataset_manifest: Mapping[str, Any] | None = None,
    architecture_runs_path: str | Path | None = None,
    app_dir: str | Path | None = None,
    html_template: str | None = None,
    css_fallback: str = "",
    js_fallback: str = "",
) -> dict[str, bytes]:
    """Render current HTML/CSS/JS entirely in memory without app writes.

    The returned mapping always has exactly ``index.html``, ``workbench.css``,
    and ``workbench.js`` keys. Existing review data and architecture-run
    catalogs are read only; annotations and all app artifacts remain untouched.
    """

    review_path = Path(review_data_path).expanduser().resolve()
    app_path = Path(app_dir).expanduser().resolve() if app_dir is not None else None
    architecture_source = (
        Path(architecture_runs_path).expanduser().resolve()
        if architecture_runs_path is not None
        else None
    )
    data = _prepared_workbench_data(
        review_data_path=review_path,
        dataset_id=dataset_id,
        dataset_manifest=dataset_manifest,
        architecture_runs_path=architecture_source,
        app_dir=app_path,
    )
    return _render_assets_from_data(
        data,
        dataset_id=dataset_id,
        html_template=html_template,
        css_fallback=css_fallback,
        js_fallback=js_fallback,
    )


_ASSET_META_PATTERN = re.compile(r"<meta\b[^>]*>", flags=re.IGNORECASE)
_META_ATTRIBUTE_PATTERN = re.compile(
    r"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*([\"'])(.*?)\2",
    flags=re.DOTALL,
)


def _installed_asset_marker(html_text: str) -> str:
    for match in _ASSET_META_PATTERN.finditer(html_text):
        attributes = {
            name.lower(): value
            for name, _, value in _META_ATTRIBUTE_PATTERN.findall(match.group(0))
        }
        if attributes.get("name", "").lower() == "neurobench-workbench-asset-version":
            value = attributes.get("content", "").lower()
            return value if re.fullmatch(r"[0-9a-f]{12}", value) else ""
    return ""


def workbench_asset_status(app_dir: str | Path) -> dict[str, Any]:
    """Inspect both the installed marker and installed CSS/JS bytes."""

    app_path = Path(app_dir).expanduser().resolve()
    installed_paths = {
        "html": app_path / "index.html",
        "css": app_path / "workbench.css",
        "js": app_path / "workbench.js",
    }
    installed_bytes = {
        key: path.read_bytes() if path.is_file() else b""
        for key, path in installed_paths.items()
    }
    try:
        installed_html = installed_bytes["html"].decode("utf-8")
    except UnicodeDecodeError:
        installed_html = ""
    marker = _installed_asset_marker(installed_html)
    packaged_css = (load_workbench_asset("workbench.css") + "\n").encode("utf-8")
    packaged_js = (load_workbench_asset("workbench.js") + "\n").encode("utf-8")
    packaged_version = workbench_asset_version()
    css_current = bool(installed_bytes["css"]) and installed_bytes["css"] == packaged_css
    js_current = bool(installed_bytes["js"]) and installed_bytes["js"] == packaged_js
    marker_current = bool(marker) and marker == packaged_version
    return {
        "installed_version": marker,
        "packaged_version": packaged_version,
        "current": marker_current and css_current and js_current,
        "marker_current": marker_current,
        "css_current": css_current,
        "js_current": js_current,
        "missing": [key for key, path in installed_paths.items() if not path.is_file()],
        "sha256": {
            "installed_css": hashlib.sha256(installed_bytes["css"]).hexdigest()
            if installed_bytes["css"]
            else "",
            "installed_js": hashlib.sha256(installed_bytes["js"]).hexdigest()
            if installed_bytes["js"]
            else "",
            "packaged_css": hashlib.sha256(packaged_css).hexdigest(),
            "packaged_js": hashlib.sha256(packaged_js).hexdigest(),
        },
    }


def resolve_build_inputs(
    *,
    app_dir: str | Path | None = None,
    review_data: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    architecture_runs: str | Path | None = None,
    default_app_dir: str | Path,
    default_review_data: str | Path,
    default_dataset_id: str,
) -> dict[str, Any]:
    """Resolve builder paths from direct args, a dataset manifest, and defaults."""
    manifest = load_dataset_manifest(dataset_manifest) if dataset_manifest else None
    resolved_app_dir = Path(app_dir) if app_dir is not None else None
    resolved_review_data = Path(review_data) if review_data is not None else None
    resolved_architecture_runs = Path(architecture_runs) if architecture_runs is not None else None
    if manifest:
        resolved_app_dir = resolved_app_dir or manifest_path(manifest, "app_dir")
        resolved_review_data = resolved_review_data or manifest_path(manifest, "review_data")
        manifest_architecture_runs = manifest_path(manifest, "architecture_runs")
        # Intake manifests declare the future output path before the catalog
        # exists. Only an explicitly supplied path should fail fast later.
        if resolved_architecture_runs is None and manifest_architecture_runs and manifest_architecture_runs.is_file():
            resolved_architecture_runs = manifest_architecture_runs
    review_data_path = (resolved_review_data or Path(default_review_data)).resolve()
    review_dataset_id = None
    if review_data_path.exists():
        review_payload = load_json(review_data_path)
        inferred_root = Path(resolved_app_dir).resolve() if resolved_app_dir else review_data_path.parent
        if inferred_root.name == "app":
            inferred_root = inferred_root.parent
        review_dataset_id = dataset_id_from_review(
            review_payload,
            fallback=inferred_root.name or default_dataset_id,
        )
    return {
        "dataset_manifest": manifest,
        "app_dir": (resolved_app_dir or Path(default_app_dir)).resolve(),
        "review_data_path": review_data_path,
        "architecture_runs_path": resolved_architecture_runs.resolve() if resolved_architecture_runs else None,
        "dataset_id": (manifest or {}).get("dataset_id") or review_dataset_id or default_dataset_id,
    }


def build_workbench(
    *,
    app_dir: str | Path,
    review_data_path: str | Path,
    dataset_id: str,
    html_template: str | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
    architecture_runs_path: str | Path | None = None,
    css_fallback: str = "",
    js_fallback: str = "",
    migrate_annotations: bool = False,
) -> dict[str, Path]:
    """Atomically publish a browser workbench and return generated paths.

    Existing annotations and an attached architecture-run catalog are
    byte-preserved by default. Annotation migration is an explicit operation,
    selected with ``migrate_annotations=True``.
    """

    app_path = Path(app_dir).expanduser().resolve()
    review_path = Path(review_data_path).expanduser().resolve()
    architecture_source = (
        Path(architecture_runs_path).expanduser().resolve()
        if architecture_runs_path is not None
        else None
    )
    if architecture_source is not None and not architecture_source.is_file():
        raise FileNotFoundError(f"Architecture-run catalog does not exist: {architecture_source}")
    if app_path.exists() and not app_path.is_dir():
        raise NotADirectoryError(f"Workbench app path is not a directory: {app_path}")

    architecture_output = app_path / "architecture_runs.json"
    paths = {
        "index": app_path / "index.html",
        "css": app_path / "workbench.css",
        "js": app_path / "workbench.js",
        "annotations": app_path / "annotations.json",
        "architecture_runs": architecture_output,
    }

    data = _prepared_workbench_data(
        review_data_path=review_path,
        dataset_id=dataset_id,
        dataset_manifest=dataset_manifest,
        architecture_runs_path=architecture_source,
        app_dir=app_path,
    )
    rendered = _render_assets_from_data(
        data,
        dataset_id=dataset_id,
        html_template=html_template,
        css_fallback=css_fallback,
        js_fallback=js_fallback,
    )

    publish: dict[str, bytes] = dict(rendered)
    if not architecture_output.exists():
        publish["architecture_runs.json"] = (
            json.dumps(data["architectureRuns"], indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    elif architecture_source is not None and architecture_source != architecture_output:
        publish["architecture_runs.json"] = architecture_source.read_bytes()

    annotations_path = paths["annotations"]
    if not annotations_path.exists():
        annotations = migrate_annotations_v3(None)
        publish["annotations.json"] = (
            json.dumps(annotations, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    elif migrate_annotations:
        annotations = migrate_annotations_v3(load_json(annotations_path))
        publish["annotations.json"] = (
            json.dumps(annotations, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    app_existed = app_path.is_dir()
    app_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{app_path.name}.build-",
            dir=app_path.parent,
        )
    )
    try:
        for name, body in publish.items():
            staged_path = staging / name
            with staged_path.open("wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())

        if not app_existed:
            os.replace(staging, app_path)
            staging = app_path
        else:
            # Dependencies are installed before index.html, which is the
            # publication commit point for a complete static-asset set.
            publication_order = (
                "architecture_runs.json",
                "annotations.json",
                "workbench.css",
                "workbench.js",
                "index.html",
            )
            for name in publication_order:
                staged_path = staging / name
                if staged_path.is_file():
                    os.replace(staged_path, app_path / name)
    finally:
        if staging != app_path and staging.exists():
            shutil.rmtree(staging)
    return paths
