"""Latent-code classifier for filename-derived zebrafish labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neurobench.workbench.intermediates import normalize_array_frame, write_png_gray8

LABELS = ("neutral", "left", "right")


def train_latent_classifier(
    *,
    dataset: Mapping[str, Any],
    autoencoder_run: Mapping[str, Any],
    out_dir: str | Path,
    classifier: str = "logistic_regression",
    split_method: str = "stratified_kfold",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latent_path = Path(str(autoencoder_run.get("latent_codes_path") or Path(autoencoder_run["checkpoint_path"]).with_name("latent_codes.npz")))
    with np.load(latent_path, allow_pickle=False) as data:
        codes = data["latent_codes"].astype(np.float32)
        video_ids = data["frame_video_ids"].astype(str)
        labels = data["frame_labels"].astype(str)
    features, video_order, y = video_level_summaries(codes, video_ids, labels)
    y_idx = encode_labels(y)
    evaluation = evaluate_video_level_classifier(
        features,
        y_idx,
        video_order,
        classifier=classifier,
        split_method=split_method,
        dataset=dataset,
    )
    preds = evaluation["preds"]
    cm = confusion_matrix(y_idx, preds, len(LABELS))
    metrics = classification_metrics(cm)
    metrics["video_count"] = int(len(video_order))
    metrics["chance_accuracy"] = float(1.0 / max(len(set(y)), 1))
    metrics["majority_class_accuracy"] = float(max(Counter(y).values()) / max(len(y), 1))
    metrics["evaluation"] = evaluation["evaluation"]
    metrics["fold_count"] = int(len(evaluation["folds"]))
    pred_tsv = out / "per_video_predictions.tsv"
    lines = ["fold\tvideo_id\ttrue_label\tpredicted_label\tcorrect\n"]
    for fold_id, vid, truth, pred in zip(evaluation["fold_ids"], video_order, y, preds):
        pred_label = LABELS[int(pred)]
        lines.append(f"{fold_id}\t{vid}\t{truth}\t{pred_label}\t{1 if pred_label == truth else 0}\n")
    pred_tsv.write_text("".join(lines), encoding="utf-8")
    cm_png = out / "confusion_matrix.png"
    emb_png = out / "latent_embedding_2d.png"
    write_matrix_preview(cm_png, cm.astype(np.float32))
    write_embedding_preview(emb_png, features, y_idx)
    run = {
        "schema_version": 1,
        "run_id": out.name or "latent_classifier_v1",
        "label_set": list(LABELS),
        "feature_source": "encoder_latent_codes",
        "split_unit": "video",
        "split_method": split_method,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(cm_png),
        "per_video_predictions_path": str(pred_tsv),
        "embedding_preview_path": str(emb_png),
        "warnings": evaluation["warnings"],
        "extras": {
            "classifier": classifier,
            "source_dataset": dataset.get("array_path"),
            "folds": evaluation["folds"],
            "class_counts": {label: int(count) for label, count in Counter(y).items()},
        },
    }
    (out / "latent_classifier_run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run


def video_level_summaries(codes: np.ndarray, video_ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    feats = []
    vids = []
    labs = []
    for vid in sorted(set(video_ids.tolist())):
        mask = video_ids == vid
        z = codes[mask]
        feats.append(np.concatenate([z.mean(axis=0), z.std(axis=0)], axis=0))
        vids.append(str(vid))
        labs.append(str(labels[mask][0]))
    if not feats:
        raise ValueError("No latent codes were available for classification.")
    return np.stack(feats).astype(np.float32), vids, labs


def encode_labels(labels: Sequence[str]) -> np.ndarray:
    label_to_i = {label: i for i, label in enumerate(LABELS)}
    encoded = []
    for label in labels:
        if label not in label_to_i:
            raise ValueError(f"Unsupported latent classifier label '{label}'. Expected one of {LABELS}.")
        encoded.append(label_to_i[str(label)])
    return np.asarray(encoded, dtype=int)


def evaluate_video_level_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    video_order: Sequence[str],
    *,
    classifier: str,
    split_method: str,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict video labels using video-held-out folds whenever possible."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    if x.shape[0] != y.shape[0] or x.shape[0] != len(video_order):
        raise ValueError("Classifier features, labels, and video ids must have matching lengths.")
    warnings: list[str] = []
    fold_defs = _fold_definitions(y, video_order, split_method=split_method, dataset=dataset, warnings=warnings)
    evaluation_name = split_method
    if not fold_defs:
        warnings.append("not enough videos per class for held-out evaluation; using resubstitution smoke evaluation")
        fold_defs = [{"fold_id": "resubstitution", "train_indices": list(range(len(video_order))), "test_indices": list(range(len(video_order)))}]
        evaluation_name = "resubstitution_smoke"
    preds = np.full(y.shape, -1, dtype=int)
    probs = np.zeros((len(y), len(LABELS)), dtype=np.float32)
    fold_ids = [""] * len(y)
    folds: list[dict[str, Any]] = []
    for fold in fold_defs:
        test_idx = np.asarray(fold["test_indices"], dtype=int)
        train_idx = np.asarray(fold["train_indices"], dtype=int)
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        fold_pred, fold_prob, fold_warnings = _fit_predict_split(x[train_idx], y[train_idx], x[test_idx], classifier=classifier)
        warnings.extend(fold_warnings)
        preds[test_idx] = fold_pred
        probs[test_idx] = fold_prob
        for index in test_idx.tolist():
            fold_ids[index] = str(fold["fold_id"])
        folds.append(
            {
                "fold_id": str(fold["fold_id"]),
                "train_video_ids": [str(video_order[i]) for i in train_idx.tolist()],
                "test_video_ids": [str(video_order[i]) for i in test_idx.tolist()],
            }
        )
    if np.any(preds < 0):
        missing = np.where(preds < 0)[0]
        fallback_pred, fallback_prob, fold_warnings = _fit_predict_split(x, y, x[missing], classifier=classifier)
        warnings.extend(fold_warnings)
        preds[missing] = fallback_pred
        probs[missing] = fallback_prob
        for index in missing.tolist():
            fold_ids[index] = "fallback_resubstitution"
    return {
        "preds": preds.astype(int),
        "probs": probs.astype(np.float32),
        "fold_ids": fold_ids,
        "folds": folds,
        "warnings": _dedupe(warnings),
        "evaluation": evaluation_name,
    }


def _fold_definitions(
    y: np.ndarray,
    video_order: Sequence[str],
    *,
    split_method: str,
    dataset: Mapping[str, Any] | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    method = str(split_method or "stratified_kfold")
    n = int(y.shape[0])
    all_indices = set(range(n))
    if method in {"resubstitution", "train_on_all", "smoke"}:
        return [{"fold_id": "resubstitution", "train_indices": list(range(n)), "test_indices": list(range(n))}]
    if method in {"dataset_split", "train_val_test", "heldout"}:
        folds = _dataset_split_fold(video_order, dataset=dataset)
        if folds:
            return folds
        warnings.append("dataset split evaluation requested but dataset split ids were unavailable; falling back to stratified_kfold")
    if method == "leave_one_video_out":
        return [
            {"fold_id": f"video_{i + 1:03d}", "train_indices": sorted(all_indices - {i}), "test_indices": [i]}
            for i in range(n)
            if n > 1
        ]
    if method != "stratified_kfold":
        warnings.append(f"unsupported classifier split_method '{method}'; using stratified_kfold")
    return _stratified_kfold_indices(y)


def _dataset_split_fold(video_order: Sequence[str], *, dataset: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not dataset:
        return []
    splits = dict(dataset.get("splits") or {})
    train_ids = {str(v) for v in splits.get("train_video_ids") or []}
    test_ids = {str(v) for v in (splits.get("val_video_ids") or []) + (splits.get("test_video_ids") or [])}
    if not train_ids or not test_ids:
        return []
    train = [i for i, vid in enumerate(video_order) if str(vid) in train_ids]
    test = [i for i, vid in enumerate(video_order) if str(vid) in test_ids]
    if not train or not test:
        return []
    return [{"fold_id": "dataset_heldout", "train_indices": train, "test_indices": test}]


def _stratified_kfold_indices(y: np.ndarray, max_folds: int = 5) -> list[dict[str, Any]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(y.tolist()):
        groups[int(label)].append(int(index))
    if not groups:
        return []
    min_count = min(len(indices) for indices in groups.values())
    fold_count = min(int(max_folds), int(min_count))
    if fold_count < 2:
        return []
    test_by_fold: list[list[int]] = [[] for _ in range(fold_count)]
    for _label, indices in sorted(groups.items()):
        for offset, index in enumerate(indices):
            test_by_fold[offset % fold_count].append(index)
    all_indices = set(range(int(y.shape[0])))
    folds = []
    for fold_index, test_indices in enumerate(test_by_fold):
        test_set = set(test_indices)
        folds.append(
            {
                "fold_id": f"fold_{fold_index + 1}",
                "train_indices": sorted(all_indices - test_set),
                "test_indices": sorted(test_set),
            }
        )
    return folds


def _fit_predict_split(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, classifier: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    classes = sorted(set(int(v) for v in y_train.tolist()))
    if not classes:
        raise ValueError("Cannot train latent classifier with no training videos.")
    if len(classes) == 1:
        pred = np.full(x_test.shape[0], classes[0], dtype=int)
        probs = np.zeros((x_test.shape[0], len(LABELS)), dtype=np.float32)
        probs[:, classes[0]] = 1.0
        warnings.append("classifier fold had one training class; constant-class fallback used")
        return pred, probs, warnings
    if classifier == "logistic_regression":
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=500, multi_class="auto")
            model.fit(x_train, y_train)
            pred = model.predict(x_test).astype(int)
            raw_probs = model.predict_proba(x_test).astype(np.float32)
            probs = np.zeros((x_test.shape[0], len(LABELS)), dtype=np.float32)
            for col, label_index in enumerate(model.classes_.astype(int).tolist()):
                probs[:, label_index] = raw_probs[:, col]
            return pred, probs, warnings
        except Exception as exc:
            warnings.append(f"logistic regression fallback used: {exc}")
    elif classifier != "nearest_centroid":
        warnings.append(f"unsupported classifier '{classifier}'; nearest-centroid fallback used")
    centroids = []
    for class_index in classes:
        centroids.append(x_train[y_train == class_index].mean(axis=0))
    centroid_arr = np.stack(centroids).astype(np.float32)
    dist = ((x_test[:, None, :] - centroid_arr[None, :, :]) ** 2).sum(axis=2)
    class_arr = np.asarray(classes, dtype=int)
    pred = class_arr[np.argmin(dist, axis=1)]
    probs = np.zeros((x_test.shape[0], len(LABELS)), dtype=np.float32)
    probs[np.arange(x_test.shape[0]), pred] = 1.0
    return pred.astype(int), probs, warnings


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(cm: np.ndarray) -> dict[str, Any]:
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))
    recalls = []
    precisions = []
    f1s = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {"accuracy": acc, "balanced_accuracy": float(np.mean(recalls)), "macro_precision": float(np.mean(precisions)), "macro_recall": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s))}


def write_matrix_preview(path: Path, matrix: np.ndarray) -> None:
    img = np.kron(matrix, np.ones((24, 24), dtype=np.float32))
    write_png_gray8(path, int(img.shape[1]), int(img.shape[0]), normalize_array_frame(img))


def write_embedding_preview(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    x = features - features.mean(axis=0, keepdims=True)
    if x.shape[0] >= 2:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        coords = x @ vt[:2].T
    else:
        coords = np.zeros((x.shape[0], 2), dtype=np.float32)
    img = np.zeros((96, 96), dtype=np.float32)
    if coords.size:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pix = np.round((coords - mins) / span * 90 + 3).astype(int)
        for (px, py), lab in zip(pix, labels):
            img[int(py) % 96, int(px) % 96] = 0.35 + 0.25 * int(lab)
    write_png_gray8(path, 96, 96, normalize_array_frame(img))


def _dedupe(items: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(str(item))
            seen.add(item)
    return out
