from __future__ import annotations

import numpy as np
import pytest

from neurobench.experiments.soma_excitation.zones import (
    DarkSomaZoneConfig,
    detect_dark_soma_zones,
)


def _synthetic_baseline(
    shape: tuple[int, int] = (96, 96),
    centers: tuple[tuple[int, int], ...] = ((48, 48),),
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    image = np.full(shape, 100.0, dtype=np.float32)
    for y, x in centers:
        distance_squared = (yy - y) ** 2 + (xx - x) ** 2
        image[distance_squared <= 3**2] = 30.0
        image[(distance_squared >= 7**2) & (distance_squared <= 9**2)] = 145.0
    return image


def _config(**overrides: object) -> DarkSomaZoneConfig:
    values: dict[str, object] = {
        "inner_sigma": 1.0,
        "outer_sigma": 5.0,
        "z_threshold": 2.5,
        "min_distance": 6.0,
        "border": 10,
        "max_zones": 20,
        "core_radius": 4.0,
        "ring_inner_radius": 4.0,
        "ring_outer_radius": 10.0,
    }
    values.update(overrides)
    return DarkSomaZoneConfig(**values)


def test_detects_dark_core_and_builds_perisomatic_annulus() -> None:
    result = detect_dark_soma_zones(_synthetic_baseline(), _config())

    nearest = min(result.zones, key=lambda zone: (zone.y - 48) ** 2 + (zone.x - 48) ** 2)
    assert (nearest.y, nearest.x) == (48, 48)
    assert nearest.contrast > 0
    assert nearest.robust_z >= 2.5
    assert result.core_mask[48, 48]
    assert not result.ring_mask[48, 48]
    assert result.ring_mask[48, 56]  # Bright synthetic annulus.
    assert not result.core_mask[48, 56]
    assert nearest.core_pixel_count == 49
    assert nearest.ring_pixel_count > nearest.core_pixel_count


def test_greedy_nms_and_cap_are_deterministic() -> None:
    image = _synthetic_baseline(centers=((30, 30), (30, 34), (66, 66)))
    config = _config(min_distance=8.0, max_zones=2)

    first = detect_dark_soma_zones(image, config)
    second = detect_dark_soma_zones(image.copy(), config)

    first_centers = [(zone.y, zone.x) for zone in first.zones]
    assert len(first_centers) == 2
    assert (66, 66) in first_centers
    assert sum((y - 30) ** 2 + (x - 32) ** 2 <= 2**2 for y, x in first_centers) == 1
    assert all(
        (y1 - y2) ** 2 + (x1 - x2) ** 2 >= config.min_distance**2
        for index, (y1, x1) in enumerate(first_centers)
        for y2, x2 in first_centers[index + 1 :]
    )
    assert first_centers == [(zone.y, zone.x) for zone in second.zones]
    assert np.array_equal(first.core_mask, second.core_mask)
    assert first.metadata() == second.metadata()


def test_constant_input_has_no_candidates_and_empty_input_is_rejected() -> None:
    result = detect_dark_soma_zones(np.ones((40, 50), dtype=np.float32), _config())

    assert result.zones == ()
    assert not result.core_mask.any()
    assert not result.ring_mask.any()
    assert np.all(result.robust_z == 0)

    with pytest.raises(ValueError, match="must not be empty"):
        detect_dark_soma_zones(np.empty((0, 10), dtype=np.float32), _config())


def test_border_exclusion_and_saturation_annotation_do_not_suppress_by_default() -> None:
    image = _synthetic_baseline(centers=((5, 5), (48, 48)))
    image[42:55, 54:60] = 200.0
    config = _config(
        saturation_threshold=180.0,
        saturation_flag_fraction=0.01,
        saturation_penalty=0.0,
    )

    result = detect_dark_soma_zones(image, config)

    assert all(zone.y >= 10 and zone.x >= 10 for zone in result.zones)
    central = min(result.zones, key=lambda zone: (zone.y - 48) ** 2 + (zone.x - 48) ** 2)
    assert central.saturation_fraction > 0
    assert central.saturation_flagged
    assert central.selection_score == pytest.approx(central.robust_z)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"outer_sigma": 0.5}, "inner_sigma"),
        ({"ring_inner_radius": 3.0}, "ring_inner_radius"),
        ({"ring_outer_radius": 4.0}, "ring_outer_radius"),
        ({"max_zones": -1}, "max_zones"),
        ({"border": 1.5}, "border"),
        ({"saturation_flag_fraction": 1.1}, "saturation_flag_fraction"),
    ],
)
def test_config_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        detect_dark_soma_zones(np.ones((32, 32), dtype=np.float32), _config(**overrides))


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        detect_dark_soma_zones(np.ones((2, 3, 4), dtype=np.float32), _config())
    invalid = np.ones((32, 32), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        detect_dark_soma_zones(invalid, _config())
