"""Truth-known diagnostic videos for frozen native-best semi-synthetic methods."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurobench.experiments.hierarchical_parzen_ica.missed_neuron_video import _Mp4Writer, _probe_video

from .config import InformationSeparationConfig
from .consensus import fit_multistart_consensus
from .conclusive_config import ConclusiveBatchConfig
from .diagnostic_videos import _panel
from .parzen_native import fit_spatial_stochastic_parzen_noisy_posterior
from .qualification import qualify_temporal_components
from .references import fit_dense_patch_fastica_wiener_reference
from .screen_runner import _atomic_json
from .semi_synthetic_v2 import make_real_background_fixture_v2


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _frame(observation,truth,recovered,residual,*,method,frame,metrics,limits):
    panels=[_panel(observation,*limits["observation"],"gray","observed real-background injection",size=240),
            _panel(truth,*limits["truth"],"magma","true injected neural contribution",size=240),
            _panel(recovered,*limits["recovered"],"coolwarm","native recovered reconstruction",size=240),
            _panel(residual,*limits["residual"],"coolwarm","observation minus reconstruction",size=240)]
    canvas=Image.new("RGB",(960,330),"black"); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
    draw.text((8,6),f"{method} | semi-synthetic overlap | frame {frame+1}",fill="white",font=font)
    draw.text((8,24),f"NMSE={metrics.get('neural_reconstruction_nmse',float('nan')):.3g} waveform={metrics.get('mean_waveform_correlation',float('nan')):.3f} IoU={metrics.get('mean_footprint_iou',float('nan')):.3f}",fill=(220,220,220),font=font)
    draw.text((8,42),"Truth is used only for evaluation. Native background is not decomposed truth.",fill=(180,180,180),font=font)
    left=0
    for panel in panels: canvas.paste(panel,(left,66)); left+=panel.width
    return np.asarray(canvas)


def _reconstruction(method,fixture,scientific,config,stage):
    method_id=method["method_id"]; p=method["parameters"]
    if method_id=="dense_patch_fastica_wiener_reference":
        return fit_dense_patch_fastica_wiener_reference(fixture.observation,quiet_frames=32,
            patch_size=int(p["patch_size"]),rank=int(p["rank"]),sample_count=512,seed=1804+int(p["rank"]),
            wiener_lambda_z=float(p["wiener_lambda_z"]))["signal"]
    if method_id=="spatial_noisy_parzen_infomax":
        return fit_spatial_stochastic_parzen_noisy_posterior(fixture.observation,quiet_frames=32,
            patch_size=int(p["patch_size"]),rank=int(p["rank"]),noise_scale=float(p["noise_scale"]),
            seed=1804+int(p["rank"]),device="cuda",sample_count=1024)["signal"]
    if method_id=="multistart_consensus":
        fit=fit_multistart_consensus(fixture.observation,base_method=p["base_method"],rank=int(p["rank"]),
            starts=int(p["starts"]),scientific_config=scientific,seed=1800,device=str(config.resources["gpu_device"]))
        qualification=qualify_temporal_components(fit["spatial_maps"],fit["sources"],spatial_shape=fixture.observation.shape[1:])
        selected=sorted(qualification["components"],key=lambda row:-row["neural_evidence_score"])[:3]
        reconstruction=np.zeros_like(fixture.observation,dtype=np.float64)
        for row in selected:
            index=int(row["component"])
            reconstruction += fit["sources"][index,:,None,None]*fit["spatial_maps"][:,index].reshape(fixture.observation.shape[1:])[None]
        return reconstruction.astype(np.float32)
    rows=[json.loads(path.read_text()) for path in (stage/"rows").glob("*.json")]
    target=next(row for row in rows if row["fixture"]=={"split":"evaluation","crop_index":0,
        "crop_origin_xy":[320,80],"morphology":"overlap","amplitude":1.0,"seed":20260921}
        and row["method_id"]==method_id and row["parameters"]==p)
    return np.load(stage/"native_artifacts"/target["fit_id"]/"result"/"neural_reconstruction.npy")


def generate(config: ConclusiveBatchConfig) -> dict[str,Any]:
    root=Path(str(config.output_root)+".partial"); stage=root/"stages/04_native_semi_synthetic"
    frozen=json.loads((stage/"frozen_native_methods.json").read_text())["selected"]
    metrics=json.loads((stage/"metrics.json").read_text())
    result_by_method={row["method_id"]:row for row in metrics["method_results"]}
    scientific=InformationSeparationConfig.load(config.scientific_config_path)
    fixture=make_real_background_fixture_v2(config.source_video,quiet_start_ui=1800,quiet_end_ui=1899,
        crop_origin_xy=(320,80),crop_size_px=32,amplitude=1.0,seed=20260921,morphology_case="overlap")
    output=root/"videos/native_semi_synthetic_v1"
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    records=[]
    for method in frozen:
        recovered=_reconstruction(method,fixture,scientific,config,stage)
        residual=fixture.observation-recovered
        abs_rec=max(float(np.percentile(np.abs(recovered),99.5)),1e-6)
        abs_res=max(float(np.percentile(np.abs(residual),99.5)),1e-6)
        limits={"observation":tuple(map(float,np.percentile(fixture.observation,[1,99.5]))),
                "truth":(0.0,max(float(np.percentile(fixture.injected_neural_signal,99.5)),1e-6)),
                "recovered":(-abs_rec,abs_rec),"residual":(-abs_res,abs_res)}
        row_metrics=result_by_method[method["method_id"]]
        path=output/f"native_{method['method_id']}_overlap_a1_seed20260921.mp4"
        first=_frame(fixture.observation[0],fixture.injected_neural_signal[0],recovered[0],residual[0],
            method=method["method_id"],frame=0,metrics={"neural_reconstruction_nmse":row_metrics["median_neural_reconstruction_nmse"]},limits=limits)
        writer=_Mp4Writer(path,first.shape[:2],8.0)
        try:
            writer.write(first)
            for index in range(1,len(recovered)):
                writer.write(_frame(fixture.observation[index],fixture.injected_neural_signal[index],recovered[index],residual[index],
                    method=method["method_id"],frame=index,metrics={"neural_reconstruction_nmse":row_metrics["median_neural_reconstruction_nmse"]},limits=limits))
            writer.close()
        except Exception: writer.abort(); raise
        records.append({"kind":"native_truth_known_semi_synthetic","method_id":method["method_id"],
            "parameters":method["parameters"],"path":path.name,"bytes":path.stat().st_size,
            "sha256":_sha256(path),"probe":_probe_video(path),"gate_passed":row_metrics["gate_passed"]})
    payload={"schema_version":1,"status":"native_diagnostic_videos_complete","videos":records,"video_count":len(records)}
    _atomic_json(output/"manifest.json",payload); return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);args=parser.parse_args(argv)
    print(json.dumps(generate(ConclusiveBatchConfig.load(args.config)),indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
