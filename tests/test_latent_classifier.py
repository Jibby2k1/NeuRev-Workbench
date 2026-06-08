from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class LatentClassifierTests(unittest.TestCase):
    def test_video_level_classifier_exports_confusion_matrix(self):
        from neurobench.dynamics.classifier import train_latent_classifier

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            codes=[]; vids=[]; labels=[]
            for i,label in enumerate(["neutral","left","right"]):
                for vid_index in range(2):
                    vid=f"{vid_index+1}_{label}"
                    for _ in range(3):
                        z=np.zeros(4, dtype=np.float32); z[i]=1.0
                        codes.append(z); vids.append(vid); labels.append(label)
            latent=root/"latent_codes.npz"
            np.savez(latent, latent_codes=np.stack(codes), frame_video_ids=np.asarray(vids), frame_labels=np.asarray(labels))
            run=train_latent_classifier(dataset={"array_path":"arrays.npz"}, autoencoder_run={"latent_codes_path":str(latent),"checkpoint_path":str(root/"dummy.pt")}, out_dir=root/"clf")
            predictions_exist = Path(run["per_video_predictions_path"]).is_file()
            prediction_rows = Path(run["per_video_predictions_path"]).read_text(encoding="utf-8").splitlines()[1:]

        self.assertEqual(np.asarray(run["confusion_matrix"]).shape, (3,3))
        self.assertTrue(predictions_exist)
        self.assertEqual(run["metrics"]["evaluation"], "stratified_kfold")
        self.assertEqual(run["metrics"]["fold_count"], 2)
        self.assertTrue(all(not row.startswith("resubstitution") for row in prediction_rows))
        for fold in run["extras"]["folds"]:
            self.assertFalse(set(fold["train_video_ids"]) & set(fold["test_video_ids"]))
