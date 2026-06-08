"""Memory-aware frame-first video access helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator

DEFAULT_MAX_EAGER_LOAD_BYTES = 1_000_000_000
ALLOW_LARGE_EAGER_LOADS_ENV = "NEUROBENCH_ALLOW_LARGE_EAGER_LOADS"
MAX_EAGER_LOAD_BYTES_ENV = "NEUROBENCH_MAX_EAGER_LOAD_BYTES"


@dataclass(frozen=True)
class VideoChunk:
    """One contiguous frame chunk from a frame-first video."""

    start_frame: int
    end_frame: int
    data: Any

    @property
    def frame_count(self) -> int:
        return int(self.end_frame - self.start_frame)


class VideoStore:
    """Small abstraction for frame-first videos backed by arrays or files.

    The first implementation intentionally supports in-memory arrays and NumPy
    ``.npy`` files. ``.npy`` paths are opened with ``mmap_mode='r'`` by default
    so callers can iterate frame chunks without reading the whole movie eagerly.
    """

    def __init__(
        self,
        array: Any,
        *,
        source_path: str | Path = "",
        storage_mode: str = "array",
    ) -> None:
        np = _load_numpy()
        self._array = array
        self.source_path = str(source_path)
        self.storage_mode = str(storage_mode)
        self.shape = tuple(int(value) for value in array.shape)
        if len(self.shape) < 3:
            raise ValueError(f"Expected a frame-first video with at least 3 dimensions, got shape {self.shape}.")
        self.dtype = np.dtype(array.dtype)

    @classmethod
    def from_array(cls, array: Any, *, source_path: str | Path = "") -> "VideoStore":
        """Create a store from an existing frame-first array-like object."""
        np = _load_numpy()
        raw = np.asarray(array)
        if raw.ndim < 3:
            return cls(raw, source_path=source_path, storage_mode="array")
        return cls(coerce_frame_first_video(raw, allow_single_frame=False), source_path=source_path, storage_mode="array")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mmap: bool = True,
        max_eager_bytes: int | None = None,
    ) -> "VideoStore":
        """Open a supported video path.

        NumPy ``.npy`` files are memory-mapped by default. TIFF files are loaded
        through ``tifffile`` when available; they are marked as eager arrays
        because memory mapping is not guaranteed across TIFF encodings.
        """
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Video path does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix == ".npy":
            np = _load_numpy()
            if not mmap:
                guard_eager_load_size(source, max_eager_bytes=max_eager_bytes)
            array = np.load(source, mmap_mode="r" if mmap else None)
            return cls(coerce_frame_first_video(array, allow_single_frame=False), source_path=source, storage_mode="npy_memmap" if mmap else "npy_array")
        if suffix in {".tif", ".tiff"}:
            try:
                import tifffile  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError("TIFF video access requires tifffile. Use .npy or install tifffile.") from exc
            guard_eager_load_size(source, max_eager_bytes=max_eager_bytes)
            return cls(coerce_frame_first_video(tifffile.imread(source), allow_single_frame=False), source_path=source, storage_mode="tiff_array")
        raise ValueError(f"Unsupported video format: {source.suffix}")

    @property
    def frame_count(self) -> int:
        return int(self.shape[0])

    @property
    def height(self) -> int:
        return int(self.shape[-2])

    @property
    def width(self) -> int:
        return int(self.shape[-1])

    @property
    def nbytes(self) -> int:
        return int(getattr(self._array, "nbytes", 0))

    def metadata(self) -> dict[str, Any]:
        """Return stable video metadata for manifests, QC, and reports."""
        return {
            "shape": [int(value) for value in self.shape],
            "frames": self.frame_count,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.dtype),
            "nbytes": self.nbytes,
            "source_path": self.source_path,
            "storage_mode": self.storage_mode,
        }

    def frame(self, index: int) -> Any:
        """Return one frame by zero-based index."""
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"Frame index {index} is outside [0, {self.frame_count}).")
        return self._array[index]

    def iter_chunks(self, chunk_size: int) -> Iterator[VideoChunk]:
        """Yield contiguous frame chunks with half-open frame bounds."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        for start in range(0, self.frame_count, int(chunk_size)):
            end = min(self.frame_count, start + int(chunk_size))
            yield VideoChunk(start_frame=start, end_frame=end, data=self._array[start:end])

    def as_array(self) -> Any:
        """Return the underlying array-like object."""
        return self._array

    def __array__(self, dtype: Any | None = None) -> Any:
        np = _load_numpy()
        return np.asarray(self._array, dtype=dtype)


def open_video(path: str | Path, *, mmap: bool = True, max_eager_bytes: int | None = None) -> VideoStore:
    """Open a supported video path as a ``VideoStore``."""
    return VideoStore.from_path(path, mmap=mmap, max_eager_bytes=max_eager_bytes)




def iter_video_chunks(
    path: str | Path,
    *,
    chunk_size: int = 64,
    mmap: bool = True,
    max_eager_bytes: int | None = None,
) -> Iterator[VideoChunk]:
    """Yield frame-first chunks from .npy or TIFF without eager full-video reads.

    NumPy files use mmap by default. Multi-page grayscale TIFF stacks are read
    page-by-page. TIFF layouts that cannot be addressed by page fall back to an
    eager read only after the normal eager-load size guard has approved it.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Video path does not exist: {source}")
    suffix = source.suffix.lower()
    np = _load_numpy()
    if suffix == ".npy":
        if not mmap:
            guard_eager_load_size(source, max_eager_bytes=max_eager_bytes)
        array = np.load(source, mmap_mode="r" if mmap else None)
        video = coerce_frame_first_video(array, allow_single_frame=False)
        for start in range(0, int(video.shape[0]), int(chunk_size)):
            end = min(int(video.shape[0]), start + int(chunk_size))
            yield VideoChunk(start_frame=start, end_frame=end, data=video[start:end])
        return
    if suffix in {".tif", ".tiff"}:
        yield from _iter_tiff_chunks(source, chunk_size=int(chunk_size), max_eager_bytes=max_eager_bytes)
        return
    raise ValueError(f"Unsupported video format: {source.suffix}. Expected .npy, .tif, or .tiff.")


def _iter_tiff_chunks(path: Path, *, chunk_size: int, max_eager_bytes: int | None) -> Iterator[VideoChunk]:
    np = _load_numpy()
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("TIFF video chunking requires tifffile.") from exc
    meta = video_metadata(path)
    frames = int(meta["frames"])
    height = int(meta["height"])
    width = int(meta["width"])
    with tifffile.TiffFile(path) as tif:
        if frames == 1:
            data = coerce_frame_first_video(tif.asarray())
            yield VideoChunk(start_frame=0, end_frame=1, data=data.astype(np.float32, copy=False))
            return
        page_count = len(tif.pages)
        if page_count >= frames and tuple(int(v) for v in tif.pages[0].shape) == (height, width):
            for start in range(0, frames, chunk_size):
                end = min(frames, start + chunk_size)
                chunk = np.stack([tif.pages[index].asarray() for index in range(start, end)], axis=0)
                yield VideoChunk(start_frame=start, end_frame=end, data=coerce_frame_first_video(chunk, allow_single_frame=False))
            return
        guard_eager_load_size(path, max_eager_bytes=max_eager_bytes)
        video = coerce_frame_first_video(tif.asarray(), allow_single_frame=False)
        for start in range(0, int(video.shape[0]), int(chunk_size)):
            end = min(int(video.shape[0]), start + int(chunk_size))
            yield VideoChunk(start_frame=start, end_frame=end, data=video[start:end])

def load_video_array(path: str | Path, *, mmap: bool = False, max_eager_bytes: int | None = None) -> Any:
    """Return a frame-first ``[T, H, W]`` video array from ``.npy`` or TIFF.

    Color videos are converted to grayscale by averaging the last channel. A
    single 2-D image is treated as a one-frame stack. Ambiguous higher-rank
    arrays fail with a readable error so model and registration stages do not
    silently consume channel-first or non-video data.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Video path does not exist: {source}")
    suffix = source.suffix.lower()
    np = _load_numpy()
    if suffix == ".npy":
        if not mmap:
            guard_eager_load_size(source, max_eager_bytes=max_eager_bytes)
        array = np.load(source, mmap_mode="r" if mmap else None)
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("TIFF video loading requires tifffile.") from exc
        guard_eager_load_size(source, max_eager_bytes=max_eager_bytes)
        array = tifffile.imread(source)
    else:
        raise ValueError(f"Unsupported video format: {source.suffix}. Expected .npy, .tif, or .tiff.")
    return coerce_frame_first_video(array)

def guard_eager_load_size(path: str | Path, *, max_eager_bytes: int | None = None) -> None:
    """Refuse accidental eager loads of very large videos unless overridden."""
    source = Path(path).expanduser()
    if _large_eager_loads_allowed():
        return
    limit = _max_eager_load_bytes(max_eager_bytes)
    size = int(source.stat().st_size)
    if size <= limit:
        return
    raise RuntimeError(
        f"Refusing to eagerly load {source} ({_format_bytes(size)}). "
        f"The safety limit is {_format_bytes(limit)} to avoid freezing the workstation. "
        f"Use chunked/memory-mapped processing or set {ALLOW_LARGE_EAGER_LOADS_ENV}=1 "
        "only after explicitly accepting the risk for this run."
    )


def coerce_frame_first_video(array: Any, *, allow_single_frame: bool = True) -> Any:
    """Normalize array shape to frame-first ``[T, H, W]``.

    Three-dimensional arrays are treated as existing frame-first videos. This
    preserves narrow movies whose width happens to be 3 or 4 pixels; color video
    conversion is reserved for explicit four-dimensional ``[T,H,W,C]`` stacks.
    """
    np = _load_numpy()
    arr = np.asarray(array)
    if arr.ndim == 2:
        if allow_single_frame:
            return arr.reshape((1, int(arr.shape[0]), int(arr.shape[1])))
        raise ValueError(f"Expected video array shape [T,H,W], [T,H,W,C], or [H,W,C]; got {arr.shape}.")
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] in {1, 3, 4}:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return np.mean(arr[..., :3], axis=-1)
    raise ValueError(
        f"Expected video array shape [T,H,W], [H,W], [T,H,W,C], or [H,W,C]; got {arr.shape}."
    )


def video_metadata(path: str | Path) -> dict[str, Any]:
    """Return lightweight metadata for a supported video path."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Video path does not exist: {source}")
    suffix = source.suffix.lower()
    np = _load_numpy()
    if suffix == ".npy":
        array = np.load(source, mmap_mode="r")
        video = coerce_frame_first_video(array)
        return {
            "shape": [int(v) for v in video.shape],
            "frames": int(video.shape[0]),
            "height": int(video.shape[1]),
            "width": int(video.shape[2]),
            "dtype": str(np.dtype(video.dtype)),
            "source_path": str(source),
            "storage_mode": "npy_memmap",
        }
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("TIFF video metadata requires tifffile.") from exc
        with tifffile.TiffFile(source) as tif:
            series = tif.series[0]
            shape = tuple(int(v) for v in series.shape)
            dtype = str(series.dtype)
        if len(shape) == 2:
            frames, height, width = 1, shape[0], shape[1]
        elif len(shape) == 3 and shape[-1] in {3, 4} and shape[0] > 4:
            frames, height, width = 1, shape[0], shape[1]
        elif len(shape) == 3:
            frames, height, width = shape[0], shape[-2], shape[-1]
        elif len(shape) == 4 and shape[-1] in {1, 3, 4}:
            frames, height, width = shape[0], shape[1], shape[2]
        else:
            video = load_video_array(source)
            frames, height, width = video.shape
            dtype = str(video.dtype)
        return {
            "shape": [int(frames), int(height), int(width)],
            "frames": int(frames),
            "height": int(height),
            "width": int(width),
            "dtype": dtype,
            "source_path": str(source),
            "storage_mode": "tiff_metadata",
        }
    raise ValueError(f"Unsupported video format: {source.suffix}. Expected .npy, .tif, or .tiff.")


def as_video_store(video: Any) -> VideoStore:
    """Return ``video`` as a ``VideoStore`` without copying when possible."""
    if isinstance(video, VideoStore):
        return video
    return VideoStore.from_array(video)


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for video access.") from exc
    return np


def _large_eager_loads_allowed() -> bool:
    return os.environ.get(ALLOW_LARGE_EAGER_LOADS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _max_eager_load_bytes(override: int | None) -> int:
    if override is not None:
        return int(override)
    raw = os.environ.get(MAX_EAGER_LOAD_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_EAGER_LOAD_BYTES
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{MAX_EAGER_LOAD_BYTES_ENV} must be an integer byte count.") from exc


def _format_bytes(value: int) -> str:
    amount = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{float(value):.1f} B"
