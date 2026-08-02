"""Diagnostic MP4 suite for generated identifiability and Spon burst review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import _Mp4Writer, _probe_video
from neurobench.experiments.learnable_contrast import core as label_core

from .calibration_config import CalibrationConfig
from .gpu_screen import _execute_cuda_method
from .identifiability import make_identifiability_fixture
from .qualification import qualify_temporal_components
from .screen_runner import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb(values: np.ndarray, lo: float, hi: float, cmap: str = "gray") -> np.ndarray:
    import matplotlib
    unit = np.clip((np.asarray(values, dtype=np.float32)-lo)/max(hi-lo, 1e-8), 0, 1)
    return np.asarray(matplotlib.colormaps[cmap](unit)[..., :3]*255, dtype=np.uint8)


def _panel(values: np.ndarray, lo: float, hi: float, cmap: str, title: str, size: int = 240) -> Image.Image:
    image = Image.fromarray(_rgb(values, lo, hi, cmap)).resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size+20), "black")
    canvas.paste(image, (0, 20))
    ImageDraw.Draw(canvas).text((5, 4), title, fill="white", font=ImageFont.load_default())
    return canvas


def _generated_frame(
    observation: np.ndarray, truth: np.ndarray, component: np.ndarray, residual: np.ndarray,
    *, case: str, method: str, frame: int, probability: float, threshold: float,
    truth_identifiable: bool, reported_resolved: bool, limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    panels = [
        _panel(observation, *limits["observation"], "gray", "observed mixture"),
        _panel(truth, *limits["truth"], "magma", "true neural contribution"),
        _panel(component, *limits["component"], "coolwarm", "top recovered component"),
        _panel(residual, *limits["residual"], "coolwarm", "model residual"),
    ]
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, 315), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    verdict = "CORRECT" if truth_identifiable == reported_resolved else "ERROR"
    draw.text((8, 7), f"{method} | {case} | frame {frame+1} | truth={'identifiable' if truth_identifiable else 'unidentifiable'}", fill="white", font=font)
    draw.text((8, 26), f"P(identifiable)={probability:.4f} threshold={threshold:.4f} decision={'resolved' if reported_resolved else 'abstain'} | {verdict}", fill=(90,255,130) if verdict=="CORRECT" else (255,100,90), font=font)
    draw.text((8, 45), "Recovery panels are truth-known generated diagnostics; confidence uses no evaluation labels.", fill=(190,190,190), font=font)
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 55)); left += panel.width
    return np.asarray(canvas)


def _render_generated(
    config: CalibrationConfig, calibration_root: Path, output: Path,
    *, seed: int = 137, snr: float = 8.0, fps: float = 12.0,
) -> list[dict[str, Any]]:
    metrics = json.loads((calibration_root/"metrics.json").read_text())
    result_by_method = {row["method_id"]: row for row in metrics["method_results"]}
    records = []
    cases = list(config.evaluation["case_ids"])
    methods = {str(row["method_id"]): dict(row["parameters"]) for row in config.methods}
    for method_id, parameters in methods.items():
        result = result_by_method[method_id]
        threshold = float(result["model"]["selected"]["threshold"])
        predictions = result["evaluation_predictions"]
        for case in cases:
            prediction = next(row for row in predictions if row["case_id"]==case and int(row["seed"])==seed and float(row["snr"])==snr)
            fixture = make_identifiability_fixture(case, seed=seed, snr=snr)
            execution = _execute_cuda_method(fixture.observation, method_id, parameters, config.scientific_config, seed, str(config.resources["device"]))
            qualification = qualify_temporal_components(execution["spatial_maps"], execution["sources"], spatial_shape=fixture.observation.shape[1:])
            top = max(qualification["components"], key=lambda row: row["neural_evidence_score"])
            index, sign = int(top["component"]), int(top["orientation_sign"])
            spatial = sign*np.asarray(execution["spatial_maps"][:, index]).reshape(fixture.observation.shape[1:])
            temporal = sign*np.asarray(execution["sources"][index])
            component = temporal[:,None,None]*spatial[None,:,:]
            matrix = np.asarray(execution["spatial_maps"]) @ np.asarray(execution["sources"])
            if method_id != "pca_reference":
                matrix = matrix + fixture.observation.reshape(len(fixture.observation),-1).T.mean(axis=1,keepdims=True)
            reconstruction = matrix.T.reshape(fixture.observation.shape)
            residual = fixture.observation-reconstruction
            obs_lo, obs_hi = np.percentile(fixture.observation, [1,99.5])
            truth_hi = max(float(np.percentile(fixture.neural_signal,99.5)),1e-6)
            comp_hi = max(float(np.percentile(np.abs(component),99.5)),1e-6)
            residual_hi = max(float(np.percentile(np.abs(residual),99.5)),1e-6)
            limits = {"observation":(float(obs_lo),float(obs_hi)),"truth":(0.0,truth_hi),"component":(-comp_hi,comp_hi),"residual":(-residual_hi,residual_hi)}
            path = output/f"generated_{method_id}_{case}_seed{seed}_snr{snr:g}.mp4"
            start, stop = 48, 176
            first = _generated_frame(fixture.observation[start],fixture.neural_signal[start],component[start],residual[start],case=case,method=method_id,frame=start,probability=float(prediction["probability_identifiable"]),threshold=threshold,truth_identifiable=bool(fixture.identifiable),reported_resolved=bool(prediction["reported_resolved"]),limits=limits)
            writer = _Mp4Writer(path, first.shape[:2], fps)
            try:
                writer.write(first)
                for frame in range(start+1,stop):
                    writer.write(_generated_frame(fixture.observation[frame],fixture.neural_signal[frame],component[frame],residual[frame],case=case,method=method_id,frame=frame,probability=float(prediction["probability_identifiable"]),threshold=threshold,truth_identifiable=bool(fixture.identifiable),reported_resolved=bool(prediction["reported_resolved"]),limits=limits))
                writer.close()
            except Exception:
                writer.abort(); raise
            records.append({"kind":"generated_identifiability","method_id":method_id,"case_id":case,"seed":seed,"snr":snr,"truth_identifiable":bool(fixture.identifiable),"probability_identifiable":prediction["probability_identifiable"],"threshold":threshold,"reported_resolved":prediction["reported_resolved"],"path":path.name,"probe":_probe_video(path),"bytes":path.stat().st_size,"sha256":_sha256(path)})
    return records


def _draw_rois(image: Image.Image, rows: list[dict[str, Any]], crop: tuple[int,int,int,int]) -> None:
    draw=ImageDraw.Draw(image); font=ImageFont.load_default(); x0,y0,_,_=crop
    for row in rows:
        x=int(round(float(row["x_px"])))-x0; y=int(round(float(row["y_px"])))-y0
        draw.ellipse((x-5,y-5,x+5,y+5),outline=(70,235,255),width=2)
        draw.text((x+6,y-6),str(row["roi_identity"]).replace("roi_",""),fill=(70,235,255),font=font)


def _spon_frame(raw: np.ndarray, baseline: np.ndarray, previous: np.ndarray, *, rows: list[dict[str,Any]], crop: tuple[int,int,int,int], frame_ui: int, burst: int, raw_limits: tuple[float,float], change_hi: float, derivative_hi: float) -> np.ndarray:
    x0,y0,x1,y1=crop; sample=raw[y0:y1,x0:x1].astype(np.float32); base=baseline[y0:y1,x0:x1]; prior=previous[y0:y1,x0:x1].astype(np.float32)
    raw_rgb=_rgb(sample,*raw_limits,"gray"); pseudo=_rgb(sample,*raw_limits,"turbo")
    change=_rgb(np.maximum(sample-base,0),0,change_hi,"magma"); derivative=_rgb(np.maximum(sample-prior,0),0,derivative_hi,"viridis")
    panels=[]
    for values,title in ((raw_rgb,"raw + ROI IDs"),(pseudo,"pseudo-color intensity"),(change,"positive change vs pre-window"),(derivative,"positive frame derivative")):
        image=Image.fromarray(values); _draw_rois(image,rows,crop); canvas=Image.new("RGB",(image.width,image.height+20),"black"); canvas.paste(image,(0,20)); ImageDraw.Draw(canvas).text((5,4),title,fill="white",font=ImageFont.load_default()); panels.append(canvas)
    width=sum(p.width for p in panels); height=max(p.height for p in panels)+48; canvas=Image.new("RGB",(width+width%2,height+height%2),"black"); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    draw.text((8,7),f"Spon Ca Burst {burst} | UI frame {frame_ui} | UI indices one-based inclusive",fill="white",font=font)
    draw.text((8,25),"Cyan circles/IDs are sparse known-positive ROIs; no detector outcomes are shown. Unlabeled pixels remain unknown.",fill=(200,200,200),font=font)
    left=0
    for panel in panels: canvas.paste(panel,(left,48)); left+=panel.width
    return np.asarray(canvas)


def _render_spon(config: CalibrationConfig, output: Path, *, fps: float=8.0) -> list[dict[str,Any]]:
    source=np.load(config.scientific_config.source_video,mmap_mode="r",allow_pickle=False)
    labels=label_core.load_labels(Path("Inputs/Spon Ca Burst/labels/labels_normalized.tsv"))
    records=[]
    for burst in sorted({int(row["burst_id"]) for row in labels}):
        rows=[row for row in labels if int(row["burst_id"])==burst]
        labeled_start=min(int(row["start_frame_ui"]) for row in rows); labeled_end=max(int(row["end_frame_ui"]) for row in rows)
        start_ui=max(2,labeled_start-15); end_ui=min(len(source),labeled_end+5)
        xs=[float(row["x_px"]) for row in rows]; ys=[float(row["y_px"]) for row in rows]
        crop=(max(0,int(min(xs))-18),max(0,int(min(ys))-18),min(source.shape[2],int(max(xs))+19),min(source.shape[1],int(max(ys))+19))
        clip=np.asarray(source[start_ui-1:end_ui],dtype=np.float32)
        baseline=np.median(np.asarray(source[start_ui-1:labeled_start-1],dtype=np.float32),axis=0)
        sample=clip[:,crop[1]:crop[3],crop[0]:crop[2]]
        raw_limits=tuple(map(float,np.percentile(sample,[1,99.8])))
        change_hi=max(float(np.percentile(np.maximum(sample-baseline[crop[1]:crop[3],crop[0]:crop[2]],0),99.7)),1.0)
        derivative_hi=max(float(np.percentile(np.maximum(np.diff(sample,axis=0),0),99.7)),1.0)
        path=output/f"spon_burst_{burst:02d}_raw_pseudocolor_change_derivative.mp4"
        first=_spon_frame(clip[0],baseline,np.asarray(source[start_ui-2],dtype=np.float32),rows=rows,crop=crop,frame_ui=start_ui,burst=burst,raw_limits=raw_limits,change_hi=change_hi,derivative_hi=derivative_hi)
        writer=_Mp4Writer(path,first.shape[:2],fps)
        try:
            writer.write(first)
            for offset in range(1,len(clip)):
                writer.write(_spon_frame(clip[offset],baseline,clip[offset-1],rows=rows,crop=crop,frame_ui=start_ui+offset,burst=burst,raw_limits=raw_limits,change_hi=change_hi,derivative_hi=derivative_hi))
            writer.close()
        except Exception:
            writer.abort(); raise
        records.append({"kind":"spon_detector_blinded_burst_review","burst_id":burst,"path":path.name,"source_frames_ui_inclusive":[start_ui,end_ui],"labeled_frames_ui_inclusive":[labeled_start,labeled_end],"crop_xyxy":list(crop),"roi_ids":[str(row["roi_identity"]) for row in rows],"probe":_probe_video(path),"bytes":path.stat().st_size,"sha256":_sha256(path)})
    return records


def generate_diagnostic_suite(config: CalibrationConfig, *, calibration_root: Path, output_dir: Path) -> dict[str,Any]:
    output=output_dir.resolve(); partial=Path(str(output)+".partial")
    if output.exists() or partial.exists(): raise FileExistsError(f"diagnostic output collision: {output}")
    partial.mkdir(parents=True,exist_ok=False)
    generated=_render_generated(config,calibration_root.resolve(),partial)
    spon=_render_spon(config,partial)
    payload={"schema_version":1,"status":"diagnostic_video_suite_complete","experiment_id":config.experiment_id,"calibration_root":str(calibration_root.resolve()),"generated_videos":generated,"spon_videos":spon,"video_count":len(generated)+len(spon),"detector_blinded_spon_videos":True,"scientific_contract":"Generated videos expose truth-known recovery and identifiability decisions. Spon videos show raw/pseudo-color/change/derivative with sparse-positive ROI overlays and no detector outcomes."}
    _atomic_json(partial/"manifest.json",payload); partial.replace(output); return payload
