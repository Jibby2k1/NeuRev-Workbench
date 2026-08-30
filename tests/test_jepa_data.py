from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from neurobench.experiments.neuron_identifiability.contracts import stable_hash
from neurobench.experiments.neuron_identifiability.discovery import sha256_file
from neurobench.experiments.neuron_identifiability.jepa_data import (
    ClipRequest,
    EXPECTED_060126_FRAME_SHAPE,
    RecordingDescriptor,
    RecordingInventory,
    TemporalClipContract,
    assert_spon_excluded,
    build_jepa_data_manifest,
    deterministic_uniform_frame_indices,
    fit_training_robust_normalization,
    fixed_behavior_stratified_split,
    load_recording_inventory,
    open_recording_memmap,
    read_sequential_clip,
    sample_training_crop_plan,
    sample_validation_crop_plan,
    validate_060126_inventory,
)


TRIALS = (
    (1, "rest", 1531),
    (2, "left", 1568),
    (3, "right", 1584),
    (4, "rest", 1525),
    (5, "right", 1585),
    (6, "left", 1608),
    (7, "rest", 1691),
    (8, "left", 1592),
    (10, "rest", 1739),
    (12, "left", 1633),
    (15, "right", 1608),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _filename(number: int, behavior: str) -> str:
    return f"{number} {'resting' if behavior == 'rest' and number in {1, 4} else behavior}.tif"


def _metadata_inventory(tmp_path: Path) -> RecordingInventory:
    recordings = tuple(
        RecordingDescriptor(
            recording_id=f"060126_{number:02d}_{behavior}",
            dataset_id="060126",
            recording_number=number,
            behavior=behavior,
            uri=f"data://Inputs/060126/{_filename(number, behavior)}",
            resolved_path=tmp_path / _filename(number, behavior),
            sha256=_digest(f"recording-{number}"),
            shape=(frames, *EXPECTED_060126_FRAME_SHAPE),
            dtype="uint16",
        )
        for number, behavior, frames in TRIALS
    )
    return RecordingInventory(
        dataset_id="060126",
        recordings=recordings,
        descriptor_uri="repo://fixture_inventory.json",
        descriptor_sha256=_digest("fixture-inventory"),
        independence="untouched_by_current_spontaneous_burst_feature_selection",
    )


def _write_tiff(path: Path, video: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, video, photometric="minisblack", contiguous=True)


def _live_inventory(tmp_path: Path, videos: dict[int, np.ndarray]) -> RecordingInventory:
    recordings = []
    for number, behavior, _ in TRIALS:
        video = videos[number]
        path = tmp_path / "Inputs" / "060126" / _filename(number, behavior)
        _write_tiff(path, video)
        recordings.append(
            RecordingDescriptor(
                recording_id=f"060126_{number:02d}_{behavior}",
                dataset_id="060126",
                recording_number=number,
                behavior=behavior,
                uri=f"data://Inputs/060126/{path.name}",
                resolved_path=path,
                sha256=sha256_file(path),
                shape=tuple(video.shape),
                dtype=str(video.dtype),
            )
        )
    return RecordingInventory(
        dataset_id="060126",
        recordings=tuple(recordings),
        descriptor_uri="repo://synthetic_inventory.json",
        descriptor_sha256=_digest("synthetic-inventory"),
    )


def test_load_portable_hash_frozen_descriptor_and_verify_live_tiff(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    data_root = tmp_path / "frozen-data"
    video_path = data_root / "Inputs" / "060126" / "1 resting.tif"
    video = np.arange(8 * 12 * 16, dtype=np.uint16).reshape(8, 12, 16)
    _write_tiff(video_path, video)
    descriptor_path = repository / "inventory.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "dataset_id": "060126",
                "recordings": [
                    {
                        "recording_id": "060126_01_rest",
                        "behavior": "resting",
                        "portable_path": "data://Inputs/060126/1 resting.tif",
                        "sha256": sha256_file(video_path),
                        "shape_tyx": list(video.shape),
                        "dtype": "uint16",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = load_recording_inventory(
        descriptor_path,
        repository_root=repository,
        data_root=data_root,
        verify_live=True,
        verify_hashes=True,
    )

    recording = inventory.recordings[0]
    assert recording.uri == "data://Inputs/060126/1 resting.tif"
    assert recording.resolved_path == video_path.resolve()
    assert inventory.descriptor_uri == "repo://inventory.json"
    assert recording.pixel_size_um is None
    assert recording.frame_interval_s is None


def test_loader_rejects_changed_hash_when_verification_is_requested(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    data_root = tmp_path / "data"
    video_path = data_root / "Inputs" / "060126" / "1 resting.tif"
    video = np.zeros((8, 12, 16), dtype=np.uint16)
    _write_tiff(video_path, video)
    descriptor_path = repository / "inventory.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "portable_path": "data://Inputs/060126/1 resting.tif",
                        "sha256": "0" * 64,
                        "shape": list(video.shape),
                        "dtype": "uint16",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_recording_inventory(
            descriptor_path,
            repository_root=repository,
            data_root=data_root,
            verify_live=True,
            verify_hashes=True,
        )


def test_legacy_absolute_descriptor_keeps_runtime_path_but_serializes_portably(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    external = tmp_path / "external-authority"
    video_path = external / "Inputs" / "060126" / "1 resting.tif"
    video = np.zeros((8, 12, 16), dtype=np.uint16)
    _write_tiff(video_path, video)
    descriptor_path = repository / "legacy.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "path": str(video_path),
                        "sha256": sha256_file(video_path),
                        "shape": list(video.shape),
                        "dtype": "uint16",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    inventory = load_recording_inventory(
        descriptor_path,
        repository_root=repository,
        verify_live=True,
    )

    assert inventory.recordings[0].resolved_path == video_path.resolve()
    assert inventory.recordings[0].uri == "data://Inputs/060126/1 resting.tif"
    assert str(external) not in json.dumps(inventory.recordings[0].to_manifest())


def test_explicit_behavior_and_number_do_not_depend_on_filename_tokens(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    descriptor_path = repository / "inventory.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "recording_id": "060126_01_rest",
                        "recording_number": 1,
                        "behavior": "rest",
                        "portable_path": "data://Inputs/060126/frozen_source_a.tif",
                        "sha256": _digest("generic-name"),
                        "shape": [8, 12, 16],
                        "dtype": "uint16",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    inventory = load_recording_inventory(descriptor_path, repository_root=repository)

    assert inventory.recordings[0].recording_number == 1
    assert inventory.recordings[0].behavior == "rest"


def test_frozen_inventory_totals_and_missing_physical_metadata_are_explicit(tmp_path: Path) -> None:
    validation = validate_060126_inventory(_metadata_inventory(tmp_path))

    assert validation["observed"]["recording_count"] == 11
    assert validation["observed"]["total_frames"] == 17_664
    assert validation["observed"]["frame_shape_yx"] == [764, 1046]
    assert validation["physical_metadata"]["pixel_size_um"]["status"] == "unresolved"
    assert validation["physical_metadata"]["frame_interval_s"]["status"] == "unresolved"
    assert len(validation["physical_metadata"]["pixel_size_um"]["recordings_missing"]) == 11


def test_frozen_inventory_fails_closed_on_shape_or_total_drift(tmp_path: Path) -> None:
    inventory = _metadata_inventory(tmp_path)
    changed = replace(inventory.recordings[0], shape=(1530, *EXPECTED_060126_FRAME_SHAPE))
    drifted = replace(inventory, recordings=(changed, *inventory.recordings[1:]))

    with pytest.raises(ValueError, match="total_frames"):
        validate_060126_inventory(drifted)


def test_fixed_behavior_stratified_split_is_eight_by_three(tmp_path: Path) -> None:
    inventory = _metadata_inventory(tmp_path)
    split = fixed_behavior_stratified_split(inventory.recordings)
    manifest = split.to_manifest(inventory.recordings)

    assert len(split.train_ids) == 8
    assert len(split.validation_ids) == 3
    assert set(split.validation_ids) == {"060126_10_rest", "060126_12_left", "060126_15_right"}
    assert manifest["train_behavior_counts"] == {"left": 3, "rest": 3, "right": 2}
    assert manifest["validation_behavior_counts"] == {"left": 1, "rest": 1, "right": 1}
    assert manifest["split_sha256"] == stable_hash({key: value for key, value in manifest.items() if key != "split_sha256"})


def test_spon_source_is_explicitly_forbidden(tmp_path: Path) -> None:
    recording = _metadata_inventory(tmp_path).recordings[0]

    with pytest.raises(ValueError, match="Spon source"):
        assert_spon_excluded([replace(recording, dataset_id="spon_ca_burst")])
    with pytest.raises(ValueError, match="Spon source"):
        assert_spon_excluded([replace(recording, uri="data://Inputs/Spon Ca Burst/source.tif")])


def test_temporal_contract_and_reader_require_guarded_contiguous_clips(tmp_path: Path) -> None:
    video = np.arange(12 * 10 * 14, dtype=np.uint16).reshape(12, 10, 14)
    inventory = _live_inventory(tmp_path, {number: video + number for number, _, _ in TRIALS})
    descriptor = inventory.recordings[0]
    contract = TemporalClipContract(clip_frames=4, guard_frames=2)
    request = ClipRequest(
        recording_id=descriptor.recording_id,
        split="train",
        start_frame_zero=2,
        stop_frame_zero_exclusive=6,
        y_zero=1,
        x_zero=3,
        height=4,
        width=5,
        sampling_stratum="uniform",
        temporal_mad=0.0,
    )

    clip = read_sequential_clip(descriptor, request, contract)

    assert np.array_equal(clip, (video + 1)[2:6, 1:5, 3:8])
    assert not clip.flags.writeable
    assert not clip.flags.owndata
    assert not open_recording_memmap(descriptor).flags.writeable
    with pytest.raises(ValueError, match="contiguous"):
        contract.validate_indices([2, 3, 5, 6], descriptor.frame_count)
    with pytest.raises(ValueError, match="guard"):
        contract.validate(1, 5, descriptor.frame_count)


def test_memmap_accepts_big_endian_uint16_as_semantic_uint16(tmp_path: Path) -> None:
    path = tmp_path / "Inputs" / "060126" / "1 resting.tif"
    video = np.arange(8 * 6 * 10, dtype=">u2").reshape(8, 6, 10)
    _write_tiff(path, video)
    descriptor = RecordingDescriptor(
        recording_id="060126_01_rest",
        dataset_id="060126",
        recording_number=1,
        behavior="rest",
        uri="data://Inputs/060126/1 resting.tif",
        resolved_path=path,
        sha256=sha256_file(path),
        shape=tuple(video.shape),
        dtype="uint16",
    )

    movie = open_recording_memmap(descriptor)

    assert movie.dtype.str == ">u2"
    assert movie.dtype.name == descriptor.dtype == "uint16"
    assert int(movie[-1, -1, -1]) == int(video[-1, -1, -1])


def test_uniform_frame_indices_are_integer_deterministic_and_cover_endpoints() -> None:
    assert deterministic_uniform_frame_indices(10, 4) == (0, 3, 6, 9)
    assert deterministic_uniform_frame_indices(5, 20) == (0, 1, 2, 3, 4)
    assert deterministic_uniform_frame_indices(9, 1) == (4,)


def test_robust_normalization_is_fitted_on_training_recordings_only(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    videos: dict[int, np.ndarray] = {}
    validation_numbers = {10, 12, 15}
    for number, _, _ in TRIALS:
        location = 5000 if number in validation_numbers else 30 + number
        videos[number] = np.clip(
            location + rng.integers(-3, 4, size=(16, 16, 20)),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
    inventory = _live_inventory(tmp_path, videos)
    split = fixed_behavior_stratified_split(inventory.recordings)

    normalization = fit_training_robust_normalization(
        inventory.recordings,
        split,
        frames_per_recording=4,
        spatial_stride=2,
    )

    assert normalization.center < 100
    assert normalization.scale > 0
    assert set(normalization.fit_recording_ids) == set(split.train_ids)
    assert not set(normalization.fit_recording_ids) & set(split.validation_ids)
    assert all(indices == (0, 5, 10, 15) for _, indices in normalization.uniform_frame_indices)
    normalized = normalization.apply(videos[1][:2])
    assert normalized.dtype == np.float32


def test_crop_plan_is_deterministic_label_free_and_exactly_50_25_25(tmp_path: Path) -> None:
    videos: dict[int, np.ndarray] = {}
    for number, _, _ in TRIALS:
        video = np.full((48, 24, 96), 100 + number, dtype=np.uint16)
        video[:, :, 32:64] += (np.arange(48, dtype=np.uint16) % 3)[:, None, None]
        video[:, :, 64:] += ((np.arange(48) % 2) * 200).astype(np.uint16)[:, None, None]
        videos[number] = video
    inventory = _live_inventory(tmp_path, videos)
    split = fixed_behavior_stratified_split(inventory.recordings)
    contract = TemporalClipContract(clip_frames=8, guard_frames=2)

    first = sample_training_crop_plan(
        inventory.recordings,
        split,
        clip_count=12,
        seed=17,
        contract=contract,
        crop_shape=(8, 8),
        candidate_multiplier=6,
        score_temporal_stride=1,
        score_spatial_stride=2,
    )
    second = sample_training_crop_plan(
        inventory.recordings,
        split,
        clip_count=12,
        seed=17,
        contract=contract,
        crop_shape=(8, 8),
        candidate_multiplier=6,
        score_temporal_stride=1,
        score_spatial_stride=2,
    )

    assert first == second
    assert Counter(item.sampling_stratum for item in first) == {"uniform": 6, "high_mad": 3, "low_mad": 3}
    assert all(item.recording_id in split.train_ids and item.split == "train" for item in first)
    assert len({item.key() for item in first}) == len(first)
    high = np.median([item.temporal_mad for item in first if item.sampling_stratum == "high_mad"])
    low = np.median([item.temporal_mad for item in first if item.sampling_stratum == "low_mad"])
    assert high > low


def test_validation_crop_plan_is_deterministic_fixed_uniform_coverage(tmp_path: Path) -> None:
    videos = {
        number: np.full((20, 16, 24), 100 + number, dtype=np.uint16)
        for number, _, _ in TRIALS
    }
    inventory = _live_inventory(tmp_path, videos)
    split = fixed_behavior_stratified_split(inventory.recordings)
    contract = TemporalClipContract(clip_frames=6, guard_frames=2)

    first = sample_validation_crop_plan(
        inventory.recordings,
        split,
        clip_count=6,
        seed=29,
        contract=contract,
        crop_shape=(8, 8),
    )
    second = sample_validation_crop_plan(
        inventory.recordings,
        split,
        clip_count=6,
        seed=29,
        contract=contract,
        crop_shape=(8, 8),
    )

    assert first == second
    assert Counter(item.recording_id for item in first) == {
        "060126_10_rest": 2,
        "060126_12_left": 2,
        "060126_15_right": 2,
    }
    assert all(
        item.split == "validation" and item.sampling_stratum == "uniform" and item.temporal_mad == 0.0
        for item in first
    )

    manifest = build_jepa_data_manifest(
        inventory,
        split,
        temporal_contract=contract,
        validation_crop_plan=first,
        validate_frozen_060126=False,
    )
    lane = manifest["sampling"]["validation_uniform"]
    assert lane["selection"] == "fixed_uniform_coverage_only"
    assert lane["validation_statistics_fit"] is False
    assert lane["observed_count"] == 6


def test_manifest_is_portable_deterministic_and_declares_no_raw_copy(tmp_path: Path) -> None:
    inventory = _metadata_inventory(tmp_path)
    split = fixed_behavior_stratified_split(inventory.recordings)
    contract = TemporalClipContract(clip_frames=32, guard_frames=16)

    first = build_jepa_data_manifest(
        inventory,
        split,
        temporal_contract=contract,
    )
    second = build_jepa_data_manifest(
        inventory,
        split,
        temporal_contract=contract,
    )

    assert first == second
    assert first["source_access"] == {
        "reader": "tifffile.memmap",
        "mode": "read_only_sequential_clips",
        "raw_video_copied": False,
        "fallback_copy_allowed": False,
    }
    assert first["physical_metadata"]["pixel_size_um"]["status"] == "unresolved"
    assert first["physical_metadata"]["frame_interval_s"]["status"] == "unresolved"
    assert first["sampling"]["label_access"] == "forbidden_by_data_api"
    assert first["sampling"]["roi_access"] == "forbidden_by_data_api"
    assert first["sampling"]["burst_access"] == "forbidden_by_data_api"
    assert str(tmp_path) not in json.dumps(first)
    assert first["manifest_sha256"] == stable_hash(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
