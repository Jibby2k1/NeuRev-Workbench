"""Data helpers for Neurobench."""

from neurobench.data.checksums import checksum_file, dataset_input_checksums, input_path_keys, sha256_path
from neurobench.data.intake import PUBLIC_DATASET_TEMPLATES, build_dataset_intake_manifest, dataset_intake_report, intake_checks
from neurobench.data.qc import compute_dataset_qc_from_manifest, compute_video_qc, render_dataset_qc_markdown
from neurobench.data.video import VideoChunk, VideoStore, as_video_store, open_video

__all__ = [
    "SyntheticDataset",
    "SyntheticEvent",
    "VideoChunk",
    "VideoStore",
    "PUBLIC_DATASET_TEMPLATES",
    "as_video_store",
    "build_dataset_intake_manifest",
    "checksum_file",
    "compute_dataset_qc_from_manifest",
    "compute_video_qc",
    "dataset_input_checksums",
    "dataset_intake_report",
    "generate_synthetic_calcium_dataset",
    "input_path_keys",
    "intake_checks",
    "open_video",
    "render_dataset_qc_markdown",
    "sha256_path",
]


def __getattr__(name: str):
    if name in {"SyntheticDataset", "SyntheticEvent", "generate_synthetic_calcium_dataset"}:
        from neurobench.data import synthetic

        return getattr(synthetic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
