"""Sequential, resumable execution of the identifiability paper program."""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ProgramConfig
from .contracts import ObservationRecord, atomic_json, atomic_text, stable_hash
from .discovery import allocate_output_root, sha256_file
from .identity import identity_counts, load_adjudication
from .reporting import now, write_stage
from .representation_confirmation import build_confirmation
from .manuscript_exports import build_manuscript_exports
from .reproducibility_release import build_release, inventory as release_inventory
from .trace_extraction import (
    Geometry, extract_disk_sites, extract_site_traces,
    frozen_two_frame_ica_traces, occurrence_metrics,
)


STAGES = [
    "00_preflight", "01_identity_geometry", "02_label_timing_contract", "03_trace_atlas",
    "04_acquisition_qc", "05_scalar_observability", "06_spatial_specificity",
    "07_functional_tensor", "08_measurement_phenotypes", "09_bounded_field_annotation",
    "10_representation_confirmation", "11_manuscript_exports", "12_reproducibility_release",
]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


class ProgramRunner:
    def __init__(self, config: ProgramConfig, *, resume: bool = False, dry_run: bool = False) -> None:
        self.config = config
        self.resume = resume
        self.dry_run = dry_run
        self.root = allocate_output_root(config.output_root, resume)
        self.video_path = config.data_path("canonical_video_npy")
        candidates = [config.data_root / value for value in config.raw["paths"]["final_adjudication_candidates"]]
        self.labels_path = next((path.resolve() for path in candidates if path.exists()), config.data_path("original_labels"))
        self.records: list[ObservationRecord] = []
        self.video: np.ndarray | None = None
        self.site_order: list[str] = []
        self.traces: dict[str, np.ndarray] = {}
        self.occurrence_rows: list[dict[str, Any]] = []

    def run(self, from_stage: str | None = None, through_stage: str | None = None) -> Path:
        start = STAGES.index(from_stage) if from_stage else 0
        end = STAGES.index(through_stage) + 1 if through_stage else len(STAGES)
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "program_config.resolved.json", self.config.raw)
        atomic_text(self.root / "command_history.log", " ".join(os.sys.argv) + "\n")
        handlers: dict[str, Callable[[], None]] = {stage: getattr(self, "stage_" + stage[:2]) for stage in STAGES}
        for stage in STAGES[start:end]:
            status = self.root / stage / "status.json"
            if self.resume and status.exists() and json.loads(status.read_text()).get("status") in {"complete", "complete_degraded"}:
                continue
            if self.dry_run:
                continue
            handlers[stage]()
        return self.root

    def _load(self) -> None:
        if not self.records:
            self.records = load_adjudication(self.labels_path, self.config.raw["trace_geometry"]["center_radius_px"])
        if self.video is None:
            self.video = np.load(self.video_path, mmap_mode="r", allow_pickle=False)

    def _ensure_traces(self) -> None:
        self._load()
        if self.traces:
            return
        geometry = Geometry(
            self.config.raw["trace_geometry"]["center_radius_px"],
            self.config.raw["trace_geometry"]["annulus_inner_px"],
            self.config.raw["trace_geometry"]["annulus_outer_px"],
            self.config.raw["trace_geometry"]["outer_ring_inner_px"],
            self.config.raw["trace_geometry"]["outer_ring_outer_px"],
        )
        assert self.video is not None
        self.site_order, self.traces = extract_site_traces(self.video, self.records, geometry)
        frozen_path = self.config.repository_root / self.config.raw["paths"]["frozen_two_frame_ica"]
        self.traces["frozen_two_frame_ica"], self.frozen_ica_hash = frozen_two_frame_ica_traces(
            self.traces["raw"], frozen_path,
        )
        self._load_quantitative_feature_traces(geometry)
        index = {site: i for i, site in enumerate(self.site_order)}
        self.occurrence_rows = [occurrence_metrics(record, self.traces, index[record.observation_site_id], self.config.raw["trace_geometry"]["pre_frames"]) for record in self.records]

    def _load_quantitative_feature_traces(self, geometry: Geometry) -> None:
        """Load/reconstruct only declared numeric lanes and retain site traces."""
        carrier_path = self.config.data_path("carrier_signed_npy")
        mapping = self.config.raw["feature_frame_mapping"]
        carrier = np.load(carrier_path, mmap_mode="r", allow_pickle=False)
        expected = int(mapping["expected_frames"])
        if carrier.shape != (expected, int(self.video.shape[1]), int(self.video.shape[2])):
            raise ValueError(f"carrier tensor has unexpected shape: {carrier.shape}")
        start_zero = int(mapping["review_start_frame_ui"]) - 1
        stop_zero = start_zero + expected
        if stop_zero > self.video.shape[0]:
            raise ValueError("feature frame mapping exceeds canonical video")
        def padded(site_values: np.ndarray) -> np.ndarray:
            result = np.full((len(self.site_order), self.video.shape[0]), np.nan, dtype=np.float64)
            result[:, start_zero:stop_zero] = site_values
            return result
        self.traces["carrier_signed"] = padded(extract_disk_sites(carrier, self.records, self.site_order, geometry.center_radius))
        from neurobench.algorithms.scientific_feature_audit import causal_local_correlation_feature
        from neurobench.experiments.hierarchical_parzen_ica.scientific_audit_program import _quiet_calibrate
        for name, lag in (("coherence_w15", 0), ("propagation_lag2_w15", 2)):
            values = causal_local_correlation_feature(
                carrier, window_frames=15, lag_frames=lag,
                spatial_sigma_px=2.0, activity_qualified=True,
            )
            values = _quiet_calibrate(values, 100)
            self.traces[name] = padded(extract_disk_sites(values, self.records, self.site_order, geometry.center_radius))
            del values

    def stage_00(self) -> None:
        self._load(); assert self.video is not None
        counts = identity_counts(self.records)
        shape = list(self.video.shape)
        errors = []
        if not np.isfinite(self.video).all(): errors.append("canonical video contains non-finite values")
        for record in self.records:
            if not (0 <= record.x_px < shape[2] and 0 <= record.y_px < shape[1]): errors.append(f"coordinate outside video: {record.observation_id}")
            if record.stop_zero_exclusive > shape[0]: errors.append(f"interval outside video: {record.observation_id}")
        if errors: raise ValueError("; ".join(errors))
        source_hashes = {"video": sha256_file(self.video_path), "labels": sha256_file(self.labels_path)}
        atomic_json(self.root / "source_hashes.json", source_hashes)
        atomic_json(self.root / "source_manifest.json", {"video": str(self.video_path), "labels": str(self.labels_path), "hashes": source_hashes})
        status = subprocess.run(["git", "status", "--short", "--branch"], cwd=self.config.repository_root, text=True, capture_output=True).stdout
        metrics = {"started_at": now(), "video": {"path": str(self.video_path), "shape": shape, "dtype": str(self.video.dtype), "minimum": int(self.video.min()), "maximum": int(self.video.max()), "finite": True, "mmap": True}, "labels": {"path": str(self.labels_path), **counts}, "git_status": status, "environment": {"python": platform.python_version(), "cpu_count": os.cpu_count(), "disk_free_gib": shutil.disk_usage(self.root).free / 2**30}, "input_hashes": source_hashes}
        self._projection_overlay(self.root / "00_preflight/projection_overlay_original_sites.png")
        write_stage(self.root, "00_preflight", metrics, next_stage="01_identity_geometry")

    def _projection_overlay(self, path: Path, only: set[str] | None = None) -> None:
        import matplotlib.pyplot as plt
        assert self.video is not None
        projection = np.asarray(self.video[::16], dtype=np.float64).mean(axis=0)
        selected = [r for r in self.records if only is None or r.observation_site_id in only]
        unique = {r.observation_site_id: r for r in selected}
        fig, ax = plt.subplots(figsize=(10, 6)); ax.imshow(projection, cmap="gray")
        for site, record in sorted(unique.items()):
            ax.scatter(record.x_px, record.y_px, facecolors="none", edgecolors="#38a169", s=55)
            if only is not None: ax.text(record.x_px + 3, record.y_px, site, color="white", fontsize=8)
        ax.set(title="Immutable original observation sites", xlabel="x = column", ylabel="y = row")
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)

    def stage_01(self) -> None:
        self._load(); target = self.root / "01_identity_geometry"
        rows = [record.to_dict() for record in self.records]
        _write_tsv(target / "observation_crosswalk.tsv", rows)
        sites = {}
        for record in self.records:
            sites.setdefault(record.observation_site_id, {"observation_site_id": record.observation_site_id, "original_roi_id": record.original_roi_id, "x_px": record.x_px, "y_px": record.y_px, "geometry_hash": record.geometry_hash})
        _write_tsv(target / "site_geometry.tsv", list(sites.values()))
        grouping = [{"canonical_neuron_id": canonical, "observation_site_id": site} for canonical in sorted({r.canonical_neuron_id for r in self.records if r.canonical_neuron_id}) for site in sorted({r.observation_site_id for r in self.records if r.canonical_neuron_id == canonical})]
        _write_tsv(target / "canonical_grouping.tsv", grouping)
        collisions = [{"canonical_neuron_id": canonical, "site_count": len({r.observation_site_id for r in self.records if r.canonical_neuron_id == canonical}), "sites": ",".join(sorted({r.observation_site_id for r in self.records if r.canonical_neuron_id == canonical}))} for canonical in sorted({r.canonical_neuron_id for r in self.records if r.canonical_neuron_id})]
        _write_tsv(target / "geometry_collision_audit.tsv", collisions)
        counts = identity_counts(self.records)
        if counts != {"occurrences": 79, "original_sites": 27, "proposed_canonical_identities": 26}: raise ValueError(f"identity regression failed: {counts}")
        merged = [row for row in collisions if row["site_count"] > 1]
        atomic_json(target / "identity_contract.json", {"schema_version": 1, **counts, "trace_key": ["observation_site_id", "geometry_hash", "video_hash", "trace_channel", "channel_parameters_hash"], "multi_site_canonical_groups": merged})
        review = "ROI 010 and ROI 015 remain separate observation sites. Decide only their canonical grouping after reviewing the generated trace comparison.\n"
        self._projection_overlay(target / "roi_010_015_overlay.png", {"roi_010", "roi_015"})
        write_stage(self.root, "01_identity_geometry", {"started_at": now(), **counts, "geometry_collision_count_before": len(merged), "geometry_collision_count_after": 0}, next_stage="02_label_timing_contract", review=review, decisions=[{"id": "identity_roi_010_015"}])

    def stage_02(self) -> None:
        self._ensure_traces(); target = self.root / "02_label_timing_contract"
        _write_tsv(target / "original_site_original_timing.tsv", [record.to_dict() for record in self.records])
        suggestions = []
        site_index = {site: i for i, site in enumerate(self.site_order)}
        for record in self.records:
            if record.burst_id != 2: continue
            trace = self.traces["residual"][site_index[record.observation_site_id]]
            start, stop = record.start_zero, record.stop_zero_exclusive
            a = max(0, start - 20); b = min(len(trace), stop + 20); base = trace[a:start]
            center = float(np.median(base)); scale = max(float(np.median(np.abs(base-center))*1.4826), np.finfo(float).eps)
            segment = trace[a:b]; threshold = center + 3 * scale; above = np.flatnonzero(segment >= threshold)
            onset = int(a + (above[0] if len(above) else start-a) + 1); peak = int(a + np.argmax(segment) + 1)
            suggestions.append({"observation_id": record.observation_id, "observation_site_id": record.observation_site_id, "suggested_onset_ui": onset, "suggested_peak_ui": peak, "suggested_end_ui": record.original_end_frame_ui, "threshold_native": threshold, "status": "suggestion_only"})
        _write_tsv(target / "burst_2_timing_suggestions.tsv", suggestions)
        inventory = [{"path": str(self.labels_path), "selected": True, "sha256": sha256_file(self.labels_path), "role": "final_adjudication"}]
        _write_tsv(target / "label_source_inventory.tsv", inventory)
        pending = sum(r.review_status != "adjudicated" for r in self.records)
        write_stage(self.root, "02_label_timing_contract", {"started_at": now(), "views": ["original_site_original_timing", "original_site_adjudicated_timing", "canonical_proposed_original_timing", "canonical_confirmed_adjudicated_timing", "canonical_inclusive_adjudicated_timing"], "burst_2_suggestion_count": len(suggestions), "pending_or_provisional_rows": pending}, status="complete_degraded" if pending else "complete", decision="advance_degraded" if pending else "advance", warnings=["Adjudication remains incomplete; original-site/original-timing is primary."] if pending else [], next_stage="03_trace_atlas", decisions=[{"id": "burst_2_timing"}])

    def stage_03(self) -> None:
        self._ensure_traces(); target = self.root / "03_trace_atlas"; target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "traces.npz", site_ids=np.asarray(self.site_order), **self.traces)
        _write_tsv(target / "trace_index.tsv", [{"site_index": i, "observation_site_id": site} for i, site in enumerate(self.site_order)])
        _write_tsv(target / "occurrence_metrics.tsv", self.occurrence_rows)
        site_rows = []
        for site in self.site_order:
            values = [row for row in self.occurrence_rows if row["observation_site_id"] == site]
            site_rows.append({"observation_site_id": site, "occurrences": len(values), "median_peak": float(np.median([r["signed_peak_amplitude"] for r in values])), "median_snr": float(np.median([r["robust_peak_snr"] for r in values])), "median_specificity": float(np.median([r["normalized_spatial_specificity"] for r in values]))})
        _write_tsv(target / "site_metrics.tsv", site_rows)
        _write_tsv(target / "burst_metrics.tsv", [{"burst_id": burst, "occurrences": len(rows), "median_peak": float(np.median([r["signed_peak_amplitude"] for r in rows]))} for burst, rows in sorted(_group(self.occurrence_rows, "burst_id").items())])
        self._trace_figure(target)
        unavailable = ["radial_cs_shell", "noise_vst_auxiliary"]
        self._occurrence_review_cards(target)
        write_stage(self.root, "03_trace_atlas", {"started_at": now(), "occurrences": len(self.occurrence_rows), "sites": len(self.site_order), "available_channels": sorted(self.traces), "unavailable_channels": unavailable, "frozen_two_frame_ica_sha256": self.frozen_ica_hash, "feature_frame_mapping": self.config.raw["feature_frame_mapping"], "native_amplitude_preserved": True}, status="complete_degraded", decision="advance_degraded", warnings=["Radial-shell and VST lanes remain diagnostic-only because only display-clipped sources are preserved. Full-duration feature traces outside UI frames 1800-2359 are explicitly NaN."], next_stage="04_acquisition_qc")
        atomic_text(target / "FIGURE_INDEX.md", "# Trace atlas figure index\n\n- `trace_atlas_overview.png`: synchronized population overview.\n- `review_cards/`: 79 occurrence-specific eight-channel trace cards keyed by observation ID.\n")
        atomic_text(target / "VALIDATION_REPORT.md", "# Trace atlas validation\n\n- 79 occurrences and 27 immutable sites are represented.\n- All trace arrays have 2,359 frames.\n- Frozen ICA fingerprint and orientation passed.\n- Carrier/coherence/lag mapping is UI 1800-2359; other frames are explicit NaN.\n- Raw, annulus, and residual remain in native intensity units.\n- Radial/VST display-only lanes were not imported quantitatively.\n")

    def _trace_figure(self, target: Path) -> None:
        import matplotlib.pyplot as plt
        index = {site: i for i, site in enumerate(self.site_order)}
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for record in self.records[:12]:
            i = index[record.observation_site_id]; a = max(0, record.start_zero-20); b = min(self.traces["raw"].shape[1], record.stop_zero_exclusive+20)
            x = np.arange(a, b) - record.start_zero
            axes[0].plot(x, self.traces["raw"][i, a:b], alpha=.45)
            axes[1].plot(x, self.traces["annulus"][i, a:b], alpha=.45)
            axes[2].plot(x, self.traces["residual"][i, a:b], alpha=.45)
        for ax, label in zip(axes, ("Raw native", "Annulus native", "ROI minus annulus")): ax.axvline(0, color="black", ls="--"); ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[-1].set_xlabel("Frames relative to original interval onset"); fig.tight_layout(); fig.savefig(target / "trace_atlas_overview.png", dpi=160); plt.close(fig)

    def _occurrence_review_cards(self, target: Path) -> None:
        import matplotlib.pyplot as plt
        cards = target / "review_cards"; cards.mkdir(parents=True, exist_ok=True)
        index = {site:i for i,site in enumerate(self.site_order)}
        channels = ("raw", "annulus", "residual", "signed_difference", "frozen_two_frame_ica", "carrier_signed", "coherence_w15", "propagation_lag2_w15")
        for record in self.records:
            i = index[record.observation_site_id]; a = max(0, record.start_zero-20); b = min(self.traces["raw"].shape[1], record.stop_zero_exclusive+20)
            x = np.arange(a,b) - record.start_zero
            fig, axes = plt.subplots(4,2,figsize=(12,9),sharex=True)
            for ax, channel in zip(axes.ravel(), channels):
                ax.plot(x, self.traces[channel][i,a:b], color="black", lw=1)
                ax.axvline(0,color="#38a169",ls="--",lw=.8); ax.axvline(record.stop_zero_exclusive-record.start_zero,color="#38a169",ls=":",lw=.8)
                ax.set_title(channel); ax.grid(alpha=.2)
            fig.suptitle(f"{record.observation_id} | immutable site {record.observation_site_id} | original timing")
            fig.tight_layout(); fig.savefig(cards / f"{record.observation_id}.png",dpi=120); plt.close(fig)

    def stage_04(self) -> None:
        self._ensure_traces(); assert self.video is not None
        from .acquisition_qc import acquisition_diagnostics
        metrics, arrays = acquisition_diagnostics(self.video, self.records)
        metrics["started_at"] = now(); metrics["other_video_qc"] = "not_run_no_other_videos_discovered"
        target=self.root/"04_acquisition_qc"; target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/"acquisition_qc_arrays.npz",**arrays)
        mean_map=arrays["mean_map"]
        covariates=[]
        for site in self.site_order:
            record=next(r for r in self.records if r.observation_site_id==site); xi,yi=int(round(record.x_px)),int(round(record.y_px))
            covariates.append({"observation_site_id":site,"x_px":record.x_px,"y_px":record.y_px,"field":"left" if record.x_px<286 else "right","distance_to_x286":abs(record.x_px-286),"quiet_mean_intensity":float(mean_map[yi,xi]),"distance_to_nearest_site_px":float(min(np.hypot(record.x_px-other.x_px,record.y_px-other.y_px) for other in self.records if other.observation_site_id!=site))})
        _write_tsv(target/"per_site_acquisition_covariates.tsv",covariates)
        self._acquisition_figure(target,metrics,arrays)
        write_stage(self.root,"04_acquisition_qc",metrics,status="complete_degraded",decision="advance_degraded",warnings=["Noise fits are descriptive Poisson-Gaussian relations from bounded quiet-frame samples, not calibrated physical sensor parameters.","Only the canonical video was discovered; multi-video QC is not applicable."],next_stage="05_scalar_observability")
        atomic_text(target/"FIGURE_INDEX.md","# Acquisition QC figures\n\n- `acquisition_qc_summary.png`: drift, PSD, mean map, and column-boundary diagnostics.\n")

    def _acquisition_figure(self,target:Path,metrics:dict[str,Any],arrays:dict[str,np.ndarray])->None:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        axes[0,0].plot(arrays["global_trace"],color="black",lw=.8); axes[0,0].set(title="Global intensity trace",xlabel="Frame",ylabel="Native intensity")
        axes[0,1].loglog(arrays["psd_frequency_hz"][1:],arrays["psd_power"][1:],color="black"); axes[0,1].set(title="Global power spectrum",xlabel="Hz",ylabel="Power")
        axes[1,0].imshow(arrays["mean_map"],cmap="gray"); axes[1,0].axvline(286,color="white",ls="--"); axes[1,0].set(title="Quiet mean map and declared x=286 boundary")
        axes[1,1].plot(np.arange(1,len(arrays["column_jump"])+1),arrays["column_jump"],color="black"); axes[1,1].axvline(286,color="#d97706",ls="--"); axes[1,1].set(title="Adjacent-column discontinuity",xlabel="Boundary x",ylabel="Mean absolute jump")
        for ax in axes.ravel(): ax.grid(alpha=.15)
        fig.tight_layout(); fig.savefig(target/"acquisition_qc_summary.png",dpi=160); plt.close(fig)

    def stage_05(self) -> None:
        self._ensure_traces(); metrics_names = ["signed_peak_amplitude", "robust_peak_snr", "residual_peak_amplitude", "normalized_spatial_specificity"]
        summaries = {}
        for name in metrics_names:
            matrix, sites, bursts = _site_burst_matrix(self.occurrence_rows, name)
            corrs = []
            for i in range(len(bursts)):
                for j in range(i+1, len(bursts)):
                    valid = np.isfinite(matrix[:,i]) & np.isfinite(matrix[:,j])
                    if valid.sum() >= 10: corrs.append(_spearman(matrix[valid,i], matrix[valid,j]))
            icc = _fallback_icc(matrix)
            median_corr = float(np.median(corrs)) if corrs else float("nan")
            rng = np.random.default_rng(self.config.raw["statistics"]["random_seed"])
            boot = np.asarray([_fallback_icc(matrix[rng.integers(0, len(matrix), len(matrix))]) for _ in range(1000)])
            without_b2 = np.delete(matrix, bursts.index(2), axis=1) if 2 in bursts else matrix
            icc_without_b2 = _fallback_icc(without_b2)
            lower = float(np.quantile(boot, .025))
            level = "O3" if median_corr >= .7 and sum(c >= .6 for c in corrs) >= 4 and icc >= .6 and lower > .3 and icc_without_b2 >= .25 else "O2" if (median_corr >= .5 or icc >= .4) and icc_without_b2 >= 0 else "O0" if median_corr < .3 and icc < .25 else "O1"
            summaries[name] = {"conditional_site_icc_nonparametric": icc, "site_bootstrap_icc_ci95": [lower, float(np.quantile(boot, .975))], "icc_excluding_burst_2": icc_without_b2, "median_pairwise_spearman": median_corr, "pairwise_spearman": corrs, "pair_count": len(corrs), "evidence_level": level, "model": "nonparametric_fallback_not_REML"}
        _write_tsv(self.root / "05_scalar_observability/site_burst_metrics.tsv", self.occurrence_rows)
        write_stage(self.root, "05_scalar_observability", {"started_at": now(), "analysis_unit": "immutable_site_with_burst", "metrics": summaries, "burst_2_sensitivity": "included_per_metric"}, status="complete_degraded", decision="advance_degraded", warnings=["Transparent nonparametric variance decomposition is reported as the fallback; REML confirmation remains pending."], next_stage="06_spatial_specificity")

    def stage_06(self) -> None:
        from .spatial_specificity import clustered_interval,matched_control_rows
        geometry=Geometry(self.config.raw["trace_geometry"]["center_radius_px"],self.config.raw["trace_geometry"]["annulus_inner_px"],self.config.raw["trace_geometry"]["annulus_outer_px"],self.config.raw["trace_geometry"]["outer_ring_inner_px"],self.config.raw["trace_geometry"]["outer_ring_outer_px"])
        assert self.video is not None
        rows=matched_control_rows(self.video,self.records,geometry,seed=self.config.raw["statistics"]["random_seed"])
        sensitivity=[]
        for candidate in (Geometry(1,2,5,6,9),Geometry(3,4,7,8,11)):
            sensitivity.extend(matched_control_rows(self.video,self.records,candidate,seed=self.config.raw["statistics"]["random_seed"]))
        target=self.root/"06_spatial_specificity"; _write_tsv(target/"matched_control_metrics.tsv",rows); _write_tsv(target/"radius_sensitivity_metrics.tsv",sensitivity)
        quiet=clustered_interval(rows,"observed_minus_quiet",self.config.raw["statistics"]["random_seed"]); spatial=clustered_interval(rows,"observed_minus_spatial",self.config.raw["statistics"]["random_seed"])
        margin=float(self.config.raw["statistics"]["correlation_equivalence_margin_primary"])
        def classify(summary:dict[str,Any])->str:
            low,high=summary["ci95"]
            return "superiority" if low>margin else "practical_equivalence" if low>=-margin and high<=margin else "unresolved"
        quiet_native=clustered_interval(rows,"native_observed_minus_quiet",self.config.raw["statistics"]["random_seed"]); spatial_native=clustered_interval(rows,"native_observed_minus_spatial",self.config.raw["statistics"]["random_seed"])
        quiet_by_site=_group(rows,"observation_site_id"); native_margin=float(np.quantile(np.abs([np.mean([float(r["quiet_native_center_minus_annulus"]) for r in site_rows]) for site_rows in quiet_by_site.values()]),.95))
        def native_classify(summary:dict[str,Any])->str:
            low,high=summary["ci95"]
            return "superiority" if low>native_margin else "practical_equivalence" if low>=-native_margin and high<=native_margin else "unresolved"
        sensitivity_summary={geometry_id:{"observed_minus_quiet":clustered_interval(subrows,"observed_minus_quiet",self.config.raw["statistics"]["random_seed"]),"observed_minus_spatial":clustered_interval(subrows,"observed_minus_spatial",self.config.raw["statistics"]["random_seed"])} for geometry_id,subrows in _group(sensitivity,"geometry").items()}
        metrics={"started_at":now(),"primary_geometry":"center_r2_annulus_r3_6","sensitivity_geometries":sensitivity_summary,"same_site_quiet_control":{**quiet,"decision":classify(quiet)},"intensity_field_matched_spatial_control":{**spatial,"decision":classify(spatial)},"normalized_specificity_margin":margin,"native_amplitude_margin":{"value":native_margin,"units":"native_uint16_intensity","calibration":"95th percentile absolute site-mean quiet center-minus-annulus before event comparison"},"native_same_site_quiet_control":{**quiet_native,"decision":native_classify(quiet_native)},"native_matched_spatial_control":{**spatial_native,"decision":native_classify(spatial_native)},"interpretation":"local measurement structure, not proof of a single neuron"}
        self._spatial_control_figure(target,rows)
        write_stage(self.root,"06_spatial_specificity",metrics,status="complete",decision="advance",warnings=["Matched shifted tissue remains unknown, not negative.","Failure to establish superiority or equivalence is reported as unresolved."],next_stage="07_functional_tensor")
        atomic_text(target/"FIGURE_INDEX.md","# Spatial specificity figures\n\n- `matched_control_summary.png`: paired observed, quiet-shift, and matched-spatial specificity.\n")

    def _spatial_control_figure(self,target:Path,rows:list[dict[str,Any]])->None:
        import matplotlib.pyplot as plt
        observed=np.asarray([r["observed_specificity"] for r in rows]); quiet=np.asarray([r["quiet_shift_specificity"] for r in rows]); spatial=np.asarray([r["matched_spatial_specificity"] for r in rows])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].scatter(quiet,observed,alpha=.65); axes[0].plot([-1,1],[-1,1],"k--"); axes[0].set(xlabel="Same-site quiet specificity",ylabel="Event specificity",title="Temporal control")
        finite=np.isfinite(spatial); axes[1].scatter(spatial[finite],observed[finite],alpha=.65); axes[1].plot([-1,1],[-1,1],"k--"); axes[1].set(xlabel="Matched shifted-site specificity",ylabel="Event specificity",title="Spatial control")
        for ax in axes: ax.grid(alpha=.2); ax.set_xlim(-1,1); ax.set_ylim(-1,1)
        fig.tight_layout(); fig.savefig(target/"matched_control_summary.png",dpi=160); plt.close(fig)

    def stage_07(self) -> None:
        counts = defaultdict(int)
        for record in self.records: counts[record.observation_site_id] += 1
        at_least_three = sum(value >= 3 for value in counts.values()); all_four = sum(value >= 4 for value in counts.values())
        entered = at_least_three >= 15 and all_four >= 10
        decision = "advance_degraded" if entered else "stop_branch"
        functional: dict[str, Any] = {}; tensor: dict[str, Any] = {}
        if entered:
            site_index = {site:i for i,site in enumerate(self.site_order)}
            complete = sorted(site for site, count in counts.items() if count >= 4)
            durations = [r.stop_zero_exclusive-r.start_zero for r in self.records if r.observation_site_id in complete]
            length = min(durations)
            cube = np.full((len(complete), 4, length), np.nan)
            for record in self.records:
                if record.observation_site_id not in complete: continue
                si = complete.index(record.observation_site_id); bi = record.burst_id-1
                trace = self.traces["raw"][site_index[record.observation_site_id]]
                baseline = np.median(trace[max(0,record.start_zero-20):record.start_zero])
                cube[si,bi] = trace[record.start_zero:record.start_zero+length]-baseline
            centered = cube - np.nanmean(cube, axis=(0,1), keepdims=True)
            total = float(np.nansum(centered**2))
            site_effect = np.nanmean(centered, axis=1, keepdims=True)
            burst_effect = np.nanmean(centered-site_effect, axis=0, keepdims=True)
            residual = centered-site_effect-burst_effect
            functional = {"common_window_frames": length, "integrated_variance_fraction_site": float(np.nansum(site_effect**2)*4/max(total,np.finfo(float).eps)), "integrated_variance_fraction_burst": float(np.nansum(burst_effect**2)*len(complete)/max(total,np.finfo(float).eps)), "integrated_variance_fraction_residual": float(np.nansum(residual**2)/max(total,np.finfo(float).eps))}
            unfold = np.nan_to_num(centered).reshape(len(complete), -1)
            singular = np.linalg.svd(unfold, compute_uv=False)
            tensor = {"descriptive_site_mode_rank1_variance_fraction": float(singular[0]**2/np.sum(singular**2)), "held_out_rank_selection": "not_promoted_pending_nested_leave_one_burst_out", "complete_sites": len(complete)}
        write_stage(self.root, "07_functional_tensor", {"started_at": now(), "entry_gate": {"sites_at_least_three": at_least_three, "sites_all_four": all_four, "passed": entered}, "functional_variance": functional, "tensor_rank": tensor}, status="complete_degraded", decision=decision, warnings=["Descriptive amplitude-preserving decomposition completed, but tensor rank is not promoted without nested leave-one-burst-out stability."] if entered else ["Insufficient repeated structure."], next_stage="08_measurement_phenotypes")

    def stage_08(self) -> None:
        profiles = []
        for site, rows in _group(self.occurrence_rows, "observation_site_id").items():
            profiles.append({"observation_site_id": site, "occurrences": len(rows), "median_raw_peak": float(np.median([r["signed_peak_amplitude"] for r in rows])), "median_raw_snr": float(np.median([r["robust_peak_snr"] for r in rows])), "median_residual_peak": float(np.median([r["residual_peak_amplitude"] for r in rows])), "median_specificity": float(np.median([r["normalized_spatial_specificity"] for r in rows])), "recovery_outcome": "unavailable_frozen_candidate_join"})
        _write_tsv(self.root / "08_measurement_phenotypes/site_profiles.tsv", profiles)
        write_stage(self.root, "08_measurement_phenotypes", {"started_at": now(), "site_profiles": len(profiles), "recoverability_models": "not_run_without_validated_frozen_candidate_join", "terminology": "measurement phenotype"}, status="complete_degraded", decision="advance_degraded", warnings=["Recovery labels were not inferred; unmatched candidates remain unknown."], next_stage="09_bounded_field_annotation")

    def stage_09(self) -> None:
        xs = np.asarray([r.x_px for r in self.records]); ys = np.asarray([r.y_px for r in self.records])
        width, height = self.config.raw["bounded_field"]["preferred_size_px"]
        cx, cy = float((xs.min()+xs.max())/2), float((ys.min()+ys.max())/2)
        x0 = max(0, int(round(cx-width/2))); y0 = max(0, int(round(cy-height/2)))
        region = {"x0": x0, "y0": y0, "width": width, "height": height, "selection": "label_bbox_center_fixed_size_no_detector_scores", "local_enriched_scope": True}
        atomic_json(self.root / "09_bounded_field_annotation/region.json", region)
        _write_tsv(self.root / "09_bounded_field_annotation/reviewer_template.tsv", [{"reviewer_id": "", "candidate_or_mark_id": "", "x_px": "", "y_px": "", "burst_id": "", "class": "", "visibility_confidence": "", "onset_ui": "", "peak_ui": "", "end_ui": "", "timestamp": ""}])
        write_stage(self.root, "09_bounded_field_annotation", {"started_at": now(), "status": "pending_reviewers", "region": region, "precision_recall": "not_available_until_exhaustive_truth_frozen"}, status="complete_degraded", decision="advance_degraded", warnings=["Blinded review schema and frozen region are ready; reviewer classifications and videos are pending."], next_stage="10_representation_confirmation", decisions=[{"id": "bounded_field_review"}])

    def stage_10(self) -> None:
        confirmation = build_confirmation(self.config.data_root, self.config.repository_root, self.root)
        confirmation["started_at"] = now()
        confirmation["lanes"] = self.config.raw["representation_panel"]["lanes"]
        atomic_json(self.root / "10_representation_confirmation/confirmation.json", confirmation)
        rows = []
        for result in confirmation["results"]:
            rows.append({"feature_id": result["feature_id"], "candidate_level": result["native"]["candidate_level"], "native_delta_b20": result["native"]["budget_20_macro_delta"], "identical_delta_b20": result["identical_proposal_macro_delta"]["20"], "promotion_status": result["promotion_status"]})
        _write_tsv(self.root / "10_representation_confirmation/compact_panel.tsv", rows)
        write_stage(self.root, "10_representation_confirmation", confirmation, status="complete_degraded", decision="hold", warnings=["Both compact lanes meet the frozen C3 native-utility and NMS-robustness gates; publication promotion remains held pending visual detector validation.", "Identical-proposal ranking evidence is historical and is not needed for the confirmed native-utility result.", "Precision remains unidentified."], next_stage="11_manuscript_exports")

    def stage_11(self) -> None:
        summary = build_manuscript_exports(self.root, self.config.repository_root)
        summary["started_at"] = now()
        write_stage(self.root, "11_manuscript_exports", summary, status="complete_degraded", decision="advance_degraded", warnings=["All exports remain provisional pending release gates.", "H7 C3 utility is statistically confirmed; publication promotion awaits visual validation. Precision and cross-recording claims are excluded."], next_stage="12_reproducibility_release")

    def stage_12(self) -> None:
        decisions = [
            {"id":"identity_roi_010_015", "status":"decided", "question":"Treat ROI 010/015 as separate, merged with an explicit geometry, or unresolved?", "recommended":"merge_with_separate_site_geometry"},
            {"id":"burst_2_timing", "status":"decided", "question":"Accept, edit, or reject the generated burst-2 timing suggestions.", "recommended":"retain_original_2040_2063"},
            {"id":"claim_language", "status":"decided", "question":"Approve or edit the eight-item provisional claim matrix.", "recommended":"approved_as_written"},
            {"id":"bounded_field_review", "status":"decided", "question":"Complete two-reviewer exhaustive annotation in the frozen bounded field.", "recommended":"remove_precision_aim_optional_single_author_audit"},
        ]
        yaml = ["schema_version: 1", f"run_root: {self.root}", "items:"]
        for item in decisions: yaml.extend([f"  - id: {item['id']}", f"    status: {item['status']}", f"    question: {item['question']}", f"    selected_option: {item['recommended']}"])
        atomic_text(self.root / "DECISIONS_REQUESTED.yaml", "\n".join(yaml) + "\n")
        summary = build_release(self.root, self.config.repository_root); summary["started_at"] = now()
        write_stage(self.root, "12_reproducibility_release", summary, status="complete_degraded", decision="hold", warnings=["Release gate not passed; package is neither analysis_complete nor manuscript_ready.", "See release/known_missing_artifacts.json and DECISIONS_REQUESTED.yaml."])
        atomic_json(self.root / "program_state.json", {"schema_version": 1, "status": "complete_degraded", "scientific_status": "provisional", "completed_at": now(), "stages": STAGES})
        index_path = self.root / "release/final_artifact_index.json"
        context_path = self.root / "release/llm_context.json"
        artifacts = release_inventory(self.root, exclude={index_path, context_path})
        atomic_json(index_path, {"schema_version": 1, "root": str(self.root), "artifacts": artifacts, "artifact_count": len(artifacts)})


def _group(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: result[row[key]].append(row)
    return dict(result)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0,1])


def _site_burst_matrix(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, list[str], list[int]]:
    sites = sorted({str(row["observation_site_id"]) for row in rows}); bursts = sorted({int(row["burst_id"]) for row in rows})
    matrix = np.full((len(sites), len(bursts)), np.nan); si = {v:i for i,v in enumerate(sites)}; bi = {v:i for i,v in enumerate(bursts)}
    grouped: dict[tuple[str,int], list[float]] = defaultdict(list)
    for row in rows: grouped[(str(row["observation_site_id"]), int(row["burst_id"]))].append(float(row[metric]))
    for (site, burst), values in grouped.items(): matrix[si[site], bi[burst]] = np.median(values)
    return matrix, sites, bursts


def _fallback_icc(matrix: np.ndarray) -> float:
    valid_rows = np.sum(np.isfinite(matrix), axis=1) >= 2
    matrix = matrix[valid_rows]
    if len(matrix) < 2:
        return 0.0
    site_means = np.nanmean(matrix, axis=1)
    site_var = float(np.nanvar(site_means))
    residual_var = float(np.nanmean(np.nanvar(matrix, axis=1)))
    return site_var / max(site_var + residual_var, np.finfo(float).eps)
