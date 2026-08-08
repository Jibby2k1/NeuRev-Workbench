"""Validated contracts for revisioned single-reviewer annotations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from neurobench.validation.schemas import validate_dict


def _payload_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(payload))


@dataclass(frozen=True)
class AnnotationViewContract:
    """Coordinate and intensity contract for one selectable video representation."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnnotationViewContract":
        result = cls(_payload_copy(payload))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return _payload_copy(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "annotation_view_contract")
        transform = self.payload["source_to_view"]
        if transform["kind"] == "affine":
            matrix = transform["matrix_3x3"]
            if any(not math.isfinite(float(value)) for row in matrix for value in row):
                raise ValueError("source_to_view matrix values must be finite")
            if any(abs(float(actual) - expected) > 1e-12 for actual, expected in zip(matrix[2], (0.0, 0.0, 1.0))):
                raise ValueError("source_to_view affine matrix bottom row must be [0, 0, 1]")
            determinant = float(matrix[0][0]) * float(matrix[1][1]) - float(matrix[0][1]) * float(matrix[1][0])
            if abs(determinant) <= 1e-12:
                raise ValueError("source_to_view affine matrix must be invertible")

    def source_xy_to_view(self, x: float, y: float) -> tuple[float, float]:
        transform = self.payload["source_to_view"]
        if transform["kind"] == "identity":
            return float(x), float(y)
        matrix = transform["matrix_3x3"]
        return (
            float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2]),
            float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2]),
        )

    def view_xy_to_source(self, x: float, y: float) -> tuple[float, float]:
        transform = self.payload["source_to_view"]
        if transform["kind"] == "identity":
            return float(x), float(y)
        matrix = transform["matrix_3x3"]
        a, b, tx = (float(value) for value in matrix[0])
        c, d, ty = (float(value) for value in matrix[1])
        determinant = a * d - b * c
        shifted_x, shifted_y = float(x) - tx, float(y) - ty
        return (
            (d * shifted_x - b * shifted_y) / determinant,
            (-c * shifted_x + a * shifted_y) / determinant,
        )

    def source_frame_to_view_index(self, source_index: int) -> int:
        return int(source_index) + int(self.payload["frame_mapping"]["offset"])

    def view_frame_to_source_index(self, view_index: int) -> int:
        return int(view_index) - int(self.payload["frame_mapping"]["offset"])


@dataclass(frozen=True)
class AnnotationOperation:
    """One attributable, append-only change in an annotation draft."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnnotationOperation":
        result = cls(_payload_copy(payload))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return _payload_copy(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "annotation_operation")
        if any(not math.isfinite(float(value)) for value in self.payload["sourceXy"]):
            raise ValueError("sourceXy values must be finite")
        operation_type = self.payload["operationType"]
        if operation_type == "create" and self.payload["before"] is not None:
            raise ValueError("create operation before value must be null")
        if operation_type == "tombstone" and self.payload["after"] is not None:
            raise ValueError("tombstone operation after value must be null")
        if operation_type not in {"create", "tombstone"} and self.payload["before"] is None:
            raise ValueError(f"{operation_type} operation requires a before value")
        if operation_type != "tombstone" and self.payload["after"] is None:
            raise ValueError(f"{operation_type} operation requires an after value")
        after = self.payload["after"] or {}
        if operation_type == "link" and not str(after.get("linked_model_id") or ""):
            raise ValueError("link operation requires linked_model_id")
        if operation_type == "unlink" and str(after.get("linked_model_id") or ""):
            raise ValueError("unlink operation must clear linked_model_id")
        if operation_type == "promote":
            if not str(self.payload["before"].get("proposal_id") or ""):
                raise ValueError("promote operation requires proposal_id evidence")
            if str(after.get("promoted_from_model_id") or "") != str(self.payload["before"]["proposal_id"]):
                raise ValueError("promote operation must retain its model proposal ID")
        if operation_type == "edit-event-interval":
            intervals = after.get("event_intervals")
            if not isinstance(intervals, list):
                raise ValueError("edit-event-interval requires event_intervals")
            for interval in intervals:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in interval)
                    or interval[0] < 1
                    or interval[1] < interval[0]
                ):
                    raise ValueError("event intervals must be one-based inclusive [start, end] pairs")



@dataclass(frozen=True)
class AnnotationRevision:
    """Immutable metadata envelope for a draft or published revision root."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnnotationRevision":
        result = cls(_payload_copy(payload))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return _payload_copy(self.payload)

    def validate(self) -> None:
        validate_dict(self.payload, "annotation_revision")
        if self.payload["revisionToken"] != self.payload["operationCount"]:
            raise ValueError("revisionToken must equal operationCount in revision contract v1")
        if self.payload["parentRevisionId"] == self.payload["revisionId"]:
            raise ValueError("parentRevisionId must differ from revisionId")
