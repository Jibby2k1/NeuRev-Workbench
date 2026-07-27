"""Data helpers for Neurobench."""

from neurobench.data.checksums import checksum_file, dataset_input_checksums, input_path_keys, sha256_path
from neurobench.data.catalog import dataset_record_for_app, discover_dataset_catalog, llm_catalog_context, query_dataset_catalog
from neurobench.data.intake import PUBLIC_DATASET_TEMPLATES, build_dataset_intake_manifest, dataset_intake_report, intake_checks
from neurobench.data.qc import compute_dataset_qc_from_manifest, compute_video_qc, render_dataset_qc_markdown
from neurobench.data.video_manifest import build_video_manifest
from neurobench.data.video import VideoChunk, VideoStore, as_video_store, load_video_array, open_video, video_metadata
from neurobench.data.imports import (
    IMPORT_STATES,
    MAX_IMPORT_RECORD_BYTES,
    MAX_NEUREV_JSON_BYTES,
    SUPPORTED_LABEL_SUFFIXES,
    SUPPORTED_NEUREV_SUFFIXES,
    SUPPORTED_VIDEO_SUFFIXES,
    atomic_write_import_record,
    checksum_file as import_checksum_file,
    dataset_manifest_from_import,
    infer_label_mapping,
    inspect_label_table,
    inspect_neurev_json,
    inspect_source,
    iter_label_rows,
    make_import_record,
    load_neurev_json,
    normalize_dataset_id,
    read_import_record,
    resolve_allowed_local_path,
    update_import_record,
)

__all__ = [
    "SyntheticDataset",
    "SyntheticEvent",
    "VideoChunk",
    "VideoStore",
    "PUBLIC_DATASET_TEMPLATES",
    "as_video_store",
    "build_dataset_intake_manifest",
    "build_video_manifest",
    "checksum_file",
    "compute_dataset_qc_from_manifest",
    "compute_video_qc",
    "dataset_input_checksums",
    "dataset_intake_report",
    "dataset_record_for_app",
    "discover_dataset_catalog",
    "generate_synthetic_calcium_dataset",
    "input_path_keys",
    "video_metadata",
    "load_video_array",
    "intake_checks",
    "llm_catalog_context",
    "open_video",
    "render_dataset_qc_markdown",
    "query_dataset_catalog",
    "sha256_path",
    "IMPORT_STATES",
    "MAX_IMPORT_RECORD_BYTES",
    "MAX_NEUREV_JSON_BYTES",
    "SUPPORTED_LABEL_SUFFIXES",
    "SUPPORTED_NEUREV_SUFFIXES",
    "SUPPORTED_VIDEO_SUFFIXES",
    "atomic_write_import_record",
    "dataset_manifest_from_import",
    "infer_label_mapping",
    "inspect_label_table",
    "inspect_neurev_json",
    "inspect_source",
    "import_checksum_file",
    "iter_label_rows",
    "make_import_record",
    "load_neurev_json",
    "normalize_dataset_id",
    "read_import_record",
    "resolve_allowed_local_path",
    "update_import_record",
]


def __getattr__(name: str):
    if name in {"SyntheticDataset", "SyntheticEvent", "generate_synthetic_calcium_dataset"}:
        from neurobench.data import synthetic

        return getattr(synthetic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
