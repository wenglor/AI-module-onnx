"""360-degree (direction-aware) rotated detection with Rotated RTMDet.

Importing this module switches mmrotate's ``rbox`` box type to the r360 version of
``projects/RR360`` (angles in [-pi, pi), no edge swap). It must therefore be imported
BEFORE ``mmrotate.utils.register_all_modules()``: mmrotate modules bind ``RotatedBoxes`` at
import time, and a later patch fails deep in the assigner with "Box type ... hasn't been
registered in box_types".

Heading convention: the heading is the direction of the box's +w axis, clockwise positive in
image coordinates (y down). Annotations are 4-point polygons with P1 -> P2 along +w.
"""

import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_MMROTATE_ROOT = Path(__file__).resolve().parents[1] / "mmrotate"
if str(_MMROTATE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MMROTATE_ROOT))  # `projects.RR360` is imported by absolute path

if "mmrotate.models" in sys.modules:
    raise ImportError(
        "utils.rotated360 must be imported before mmrotate.models / register_all_modules()"
    )

import mmrotate.structures  # noqa: E402
from projects.RR360.structures.bbox import RotatedBoxes  # noqa: E402

mmrotate.structures.bbox.RotatedBoxes = RotatedBoxes


def _flip_directed_(self, img_shape, direction="horizontal"):
    """RR360's ``flip_`` with the horizontal and vertical angle updates corrected.

    RR360 has them swapped: a horizontal flip maps the heading (cos t, sin t) to
    (-cos t, sin t), i.e. t -> pi - t, but RR360 computes -t (and pi - t for vertical).
    The diagonal flip (t -> t - pi) was already right.
    """
    assert direction in ("horizontal", "vertical", "diagonal")
    boxes = self.tensor
    if direction in ("horizontal", "diagonal"):
        boxes[..., 0] = img_shape[1] - boxes[..., 0]
    if direction in ("vertical", "diagonal"):
        boxes[..., 1] = img_shape[0] - boxes[..., 1]
    theta = {"horizontal": math.pi - boxes[..., 4], "vertical": -boxes[..., 4], "diagonal": boxes[..., 4] - math.pi}
    boxes[..., 4] = torch.remainder(theta[direction] + math.pi, 2 * math.pi) - math.pi


RotatedBoxes.flip_ = _flip_directed_

from mmcv.transforms import BaseTransform  # noqa: E402
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder  # noqa: E402
from mmengine.evaluator import BaseMetric  # noqa: E402
from mmrotate.registry import METRICS, TASK_UTILS, TRANSFORMS  # noqa: E402
from torchvision.ops import nms  # noqa: E402


def wrap_deg(a):
    """Wraps angles in degrees to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def oriented_corners(cx, cy, w, h, theta):
    """Corners P1..P4 of a box, in the box's own frame TL, TR, BR, BL; P1 -> P2 is +w."""
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-math.sin(theta), math.cos(theta)])
    c = np.array([cx, cy])
    return np.stack(
        [
            c - w / 2 * u - h / 2 * v,
            c + w / 2 * u - h / 2 * v,
            c + w / 2 * u + h / 2 * v,
            c - w / 2 * u + h / 2 * v,
        ]
    )


def poly4_to_rbox_directed(poly):
    """Ordered 4-point polygon -> [cx, cy, w, h, heading]; unlike cv2.minAreaRect it keeps
    the heading."""
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    cx, cy = p.mean(axis=0)
    e_w, e_h = p[1] - p[0], p[2] - p[1]
    return [
        float(cx),
        float(cy),
        float(np.linalg.norm(e_w)),
        float(np.linalg.norm(e_h)),
        float(math.atan2(e_w[1], e_w[0])),
    ]


@TASK_UTILS.register_module()
class PSC360Coder(BaseBBoxCoder):
    """Phase-shifting angle coder with a 360-degree period.

    mmrotate's ``PSCCoder`` encodes 2*theta (and 4*theta), i.e. an angle modulo 180 degrees.
    This encodes ``cos(f * theta + 2 pi k / num_step)`` for every frequency ``f`` in ``freqs``
    and phase step ``k``. The code is continuous in theta over the full circle, so there is no
    seam at +-180 degrees (or anywhere else) for the angle loss.

    Decoding starts from the lowest frequency, which must be 1 so that the heading is unique,
    and refines it with each higher frequency: the higher frequency's phase gives the angle
    modulo 360/f degrees, and the lower-frequency estimate picks the branch. With
    ``freqs=(1, 2)``, the 1*theta code only has to tell forward from backward, while the
    2*theta code gives the box axis, which the image shows more clearly.

    Args:
        freqs: Frequencies, ascending, starting with 1.
        num_step: Number of phase steps per frequency, >= 3.
        detach: Decode from detached codes, so that the decoded angle passes no gradient back,
            e.g. from a rotated IoU box loss; the angle is then learned from the angle loss only.
    """

    def __init__(self, freqs=(1, 2), num_step=3, detach=False):
        super().__init__()
        self.detach = detach
        assert freqs[0] == 1 and list(freqs) == sorted(freqs), freqs
        assert num_step >= 3
        self.freqs = tuple(freqs)
        self.num_step = num_step
        self.encode_size = len(self.freqs) * num_step
        steps = 2 * math.pi * torch.arange(num_step) / num_step
        self.coef_cos = torch.cos(steps)
        self.coef_sin = torch.sin(steps)

    def encode(self, angle_targets):
        steps = 2 * math.pi * torch.arange(self.num_step).to(angle_targets) / self.num_step
        return torch.cat(
            [torch.cos(f * angle_targets + steps) for f in self.freqs], dim=-1
        )

    def decode(self, angle_preds, keepdim=False):
        if self.detach:
            angle_preds = angle_preds.detach()
        coef_cos = self.coef_cos.to(angle_preds)
        coef_sin = self.coef_sin.to(angle_preds)
        angle = None
        for i, f in enumerate(self.freqs):
            code = angle_preds[..., i * self.num_step : (i + 1) * self.num_step]
            # sum_k cos(phi + s_k) cos(s_k) = N/2 cos(phi), sum_k cos(phi + s_k) sin(s_k) = -N/2 sin(phi).
            # Summed term by term: onnxruntime's ReduceSum returns an empty input unreduced, which
            # broke the exported model on images without any candidate.
            sin_sum = sum(code[..., k] * coef_sin[k] for k in range(self.num_step))
            cos_sum = sum(code[..., k] * coef_cos[k] for k in range(self.num_step))
            phase = torch.atan2(-sin_sum, cos_sum)
            if angle is None:
                angle = phase
            else:
                branch = torch.round((f * angle - phase) / (2 * math.pi))
                angle = (phase + 2 * math.pi * branch) / f
        angle = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
        return angle.unsqueeze(-1) if keepdim else angle


@TRANSFORMS.register_module()
class ConvertMask2RBoxDirected(BaseTransform):
    """Direction-preserving replacement for ``ConvertMask2BoxType(box_type="rbox")``.

    ``ConvertMask2BoxType`` goes through ``cv2.minAreaRect``, which drops the heading. This
    reads the ordered 4-point polygons instead. It also rewrites the ``instances`` boxes, which
    the metric reads as ground truth.
    """

    def transform(self, results):
        assert "gt_masks" in results, "needs LoadAnnotations(with_mask=True, poly2mask=False)"
        boxes = [poly4_to_rbox_directed(m[0]) for m in results.pop("gt_masks").masks]
        results["gt_bboxes"] = RotatedBoxes(
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        )
        if "instances" in results:
            results["instances"] = [
                {
                    **{k: v for k, v in inst.items() if k != "mask"},
                    "bbox": poly4_to_rbox_directed(inst["mask"][0]),
                }
                for inst in results["instances"]
            ]
        return results


def _average_precision(tp, scores, n_gt):
    """All-point interpolated AP from per-detection TP flags."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, dtype=np.float64)[order]
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def score_predictions(
    results,
    class_names,
    iou_thr=0.5,
    score_thr=0.3,
    dir_thr=30.0,
    boundary=10.0,
):
    """Scores rotated detections for placement and heading.

    Args:
        results: One ``(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)`` per
            image, boxes as (N, 5) [cx, cy, w, h, heading_rad].
        class_names: Class names, indexed by label.
        iou_thr: Rotated IoU for a match. Rotated IoU ignores the heading.
        score_thr: Detections below it are left out of the heading statistics.
        dir_thr: Heading error in degrees below which a match counts for the directed AP.
        boundary: GT headings within this many degrees of a multiple of 90 are "boundary".

    Returns:
        Dict with ``AP50`` (direction-blind), ``dAP50`` (matches also need a heading error
        below ``dir_thr``), and heading statistics of the matches: ``dir_acc`` (error < 90),
        ``mae``, ``p95`` and ``flip`` (error > 90), overall and for the boundary (``bnd_*``),
        the +-180 seam (``seam_*``, GT within ``boundary`` of 180) and the interior
        (``int_*``) headings; per class with a ``<class>/`` prefix as well.
    """
    from mmcv.ops import box_iou_rotated

    per_class = {c: dict(tp=[], dtp=[], scores=[], n_gt=0, err=[], gt_deg=[]) for c in range(len(class_names))}
    for pb, ps, pl, gb, gl in results:
        pb, ps, pl = (np.asarray(x) for x in (pb, ps, pl))
        gb, gl = np.asarray(gb).reshape(-1, 5), np.asarray(gl).reshape(-1)
        for c, acc in per_class.items():
            g = gb[gl == c]
            sel = pl == c
            p, s = pb[sel].reshape(-1, 5), ps[sel]
            order = np.argsort(-s, kind="stable")
            p, s = p[order], s[order]
            acc["n_gt"] += len(g)
            if len(p) == 0:
                continue
            ious = (
                box_iou_rotated(torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(g, dtype=torch.float32)).numpy()
                if len(g)
                else np.zeros((len(p), 0))
            )
            used = np.zeros(len(g), bool)
            for i in range(len(p)):
                j = int(np.argmax(ious[i])) if len(g) else -1
                if j < 0 or ious[i, j] < iou_thr or used[j]:
                    acc["tp"].append(0)
                    acc["dtp"].append(0)
                    acc["scores"].append(s[i])
                    continue
                used[j] = True
                err = abs(float(wrap_deg(math.degrees(p[i, 4] - g[j, 4]))))
                acc["tp"].append(1)
                acc["dtp"].append(int(err < dir_thr))
                acc["scores"].append(s[i])
                if s[i] >= score_thr:
                    acc["err"].append(err)
                    acc["gt_deg"].append(math.degrees(g[j, 4]))

    def heading_stats(err, gt_deg, prefix):
        err, gt_deg = np.asarray(err), np.asarray(gt_deg)
        to_90 = np.abs(wrap_deg(gt_deg - 90 * np.round(gt_deg / 90)))
        to_180 = np.abs(wrap_deg(gt_deg - 180))
        out = {}
        for name, mask in (
            ("", np.ones(len(err), bool)),
            ("bnd_", to_90 < boundary),
            ("seam_", to_180 < boundary),
            ("int_", to_90 >= boundary),
        ):
            e = err[mask]
            out[f"{prefix}{name}n"] = int(len(e))
            if len(e):
                out[f"{prefix}{name}mae"] = round(float(e.mean()), 2)
                out[f"{prefix}{name}p95"] = round(float(np.percentile(e, 95)), 2)
                out[f"{prefix}{name}flip"] = round(float((e > 90).mean()), 4)
        if len(err):
            out[f"{prefix}dir_acc"] = round(float((err < 90).mean()), 4)
        return out

    metrics, aps, daps, all_err, all_gt = {}, [], [], [], []
    for c, acc in per_class.items():
        if acc["n_gt"] == 0:
            continue
        ap = _average_precision(acc["tp"], acc["scores"], acc["n_gt"])
        dap = _average_precision(acc["dtp"], acc["scores"], acc["n_gt"])
        aps.append(ap)
        daps.append(dap)
        name = class_names[c]
        metrics[f"{name}/AP50"] = round(ap, 4)
        metrics[f"{name}/dAP50"] = round(dap, 4)
        if len(per_class) > 1:
            metrics.update(heading_stats(acc["err"], acc["gt_deg"], f"{name}/"))
        all_err += acc["err"]
        all_gt += acc["gt_deg"]
    metrics["AP50"] = round(float(np.mean(aps)), 4)
    metrics["dAP50"] = round(float(np.mean(daps)), 4)
    metrics.update(heading_stats(all_err, all_gt, ""))
    return metrics


@METRICS.register_module()
class DirectedRotatedMetric(BaseMetric):
    """Validation metric of mmengine runs: ``score_predictions`` on the model's predictions."""

    default_prefix = "dir"

    def __init__(self, class_names, **kwargs):
        self.class_names = list(class_names)
        self.score_kwargs = {
            k: kwargs.pop(k)
            for k in ("iou_thr", "score_thr", "dir_thr", "boundary")
            if k in kwargs
        }
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        for ds in data_samples:
            pred = ds["pred_instances"]
            gt = ds["instances"]
            self.results.append(
                (
                    pred["bboxes"].cpu().float().numpy(),
                    pred["scores"].cpu().numpy(),
                    pred["labels"].cpu().numpy(),
                    np.array([i["bbox"] for i in gt], np.float32).reshape(-1, 5),
                    np.array([i["bbox_label"] for i in gt], np.int64),
                )
            )

    def compute_metrics(self, results):
        return score_predictions(results, self.class_names, **self.score_kwargs)


def coco_headings(ann_file):
    """Headings (degrees) and aspect ratios (long / short edge) of a COCO split's polygons."""
    coco = json.load(open(ann_file))
    boxes = np.array([poly4_to_rbox_directed(a["segmentation"][0]) for a in coco["annotations"]])
    aspect = np.maximum(boxes[:, 2], boxes[:, 3]) / np.maximum(np.minimum(boxes[:, 2], boxes[:, 3]), 1e-6)
    return np.degrees(boxes[:, 4]), aspect


# ---------------------------------------------------------------------------------------------
# Export and ONNX evaluation
# ---------------------------------------------------------------------------------------------


class RotatedDetectionExport(nn.Module):
    """Export wrapper for Rotated RTMDet, as in the production notebook.

    Angle-range agnostic: the angle coder decodes the heading and ``DistanceAnglePointCoder``
    wraps it with the head's angle version (r360 here). The AABB-NMS suppresses a box's 180
    degree twin as well (same AABB), keeping the higher-scoring heading.

    Output: boxes (N, 5) = (cx, cy, w, h, angle_rad), labels (N,) int64, scores (N,) float32.
    """

    def __init__(self, base_model, conf_thre=0.05, nms_iou_thr=0.5, max_per_img=100, max_candidates=200):
        super().__init__()
        self.backbone = base_model.backbone
        self.neck = base_model.neck
        self.bbox_head = base_model.bbox_head
        self.conf_thre = conf_thre
        self.nms_iou_thr = nms_iou_thr
        self.max_per_img = max_per_img
        self.max_candidates = max_candidates
        self.num_classes = self.bbox_head.cls_out_channels
        if getattr(self.bbox_head, "share_conv", False):
            # separate copies per level, so the export can fold each level's BatchNorm
            for convs in (self.bbox_head.cls_convs, self.bbox_head.reg_convs):
                for level in range(1, len(convs)):
                    for layer in convs[level]:
                        layer.conv = copy.deepcopy(layer.conv)
        self.eval()

    @staticmethod
    def _rbox_to_aabb(rboxes):
        cx, cy, w, h, a = rboxes.unbind(-1)
        cos, sin = a.cos().abs(), a.sin().abs()
        dx = w / 2 * cos + h / 2 * sin
        dy = w / 2 * sin + h / 2 * cos
        return torch.stack([cx - dx, cy - dy, cx + dx, cy + dy], dim=-1)

    @staticmethod
    def _flatten(tensors, last_dim):
        return torch.cat([t.permute(0, 2, 3, 1).reshape(-1, last_dim) for t in tensors])

    def forward(self, x):
        cls_scores, bbox_preds, angle_preds = self.bbox_head(self.neck(self.backbone(x)))
        mlvl_priors = self.bbox_head.prior_generator.grid_priors(
            [cs.shape[-2:] for cs in cls_scores], dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )
        scores, labels = self._flatten(cls_scores, self.num_classes).sigmoid().max(dim=1)
        keep = scores >= self.conf_thre
        if self.max_candidates is not None and self.max_candidates < scores.numel():
            keep = keep & (scores >= scores.topk(self.max_candidates).values[-1])
        scores, labels = scores[keep], labels[keep]

        priors = torch.cat(mlvl_priors)[keep]
        distances = self._flatten(bbox_preds, 4)[keep]
        coder = self.bbox_head.angle_coder
        codes = self._flatten(angle_preds, coder.encode_size)[keep]
        # degenerate predictions can regress distances of 1e16 px; no real box reaches farther
        # than the input's diagonal from its prior (inside the image): drop them, non-finite too
        valid = (distances < float(x.shape[-2] ** 2 + x.shape[-1] ** 2) ** 0.5).all(dim=1)
        scores, labels, priors, distances, codes = (t[valid] for t in (scores, labels, priors, distances, codes))
        angles = coder.decode(codes, keepdim=True)
        boxes = self.bbox_head.bbox_coder.decode(priors, torch.cat([distances, angles], dim=-1))

        aabbs = self._rbox_to_aabb(boxes)
        class_offset = labels.to(aabbs).unsqueeze(1) * (aabbs.max() + 1)
        keep = nms(aabbs + class_offset, scores, self.nms_iou_thr)[: self.max_per_img]
        return boxes[keep], labels[keep].to(torch.int64), scores[keep]


def preprocess(image, target_size, model_cfg, pad_value=(114, 114, 114)):
    """Resize keeping the aspect ratio, pad bottom/right, normalise; as in the production notebook.

    Returns the (3, H, W) float32 input and the resize scale.
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)  # as mmdet's FixShapeResize
    img = cv2.resize(image, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, target_size[0] - new_h, 0, target_size[1] - new_w, cv2.BORDER_CONSTANT, value=pad_value)
    if model_cfg["input_color_space"] == "RGB":
        img = img[:, :, ::-1]
    img = img.astype(np.float32)
    if model_cfg["unit_scaling"]:
        img = img / 255
    img = (img - np.array(model_cfg["mean"], np.float32)) / np.array(model_cfg["std"], np.float32)
    return np.ascontiguousarray(np.moveaxis(img, -1, 0)), scale


def load_coco_gt(root, split):
    """Per image: (path, gt_boxes (N, 5) with heading, gt_labels (N,)) of a COCO split."""
    coco = json.load(open(Path(root) / split / "annotations" / "instances_default.json"))
    cat_index = {c["id"]: k for k, c in enumerate(coco["categories"])}
    by_image = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    out = []
    for im in coco["images"]:
        anns = by_image[im["id"]]
        out.append(
            (
                Path(root) / split / "images" / "default" / im["file_name"],
                np.array([poly4_to_rbox_directed(a["segmentation"][0]) for a in anns], np.float32).reshape(-1, 5),
                np.array([cat_index[a["category_id"]] for a in anns], np.int64),
            )
        )
    return out


def evaluate_predict_fn(predict, gt, class_names, **score_kwargs):
    """Scores ``predict(image_bgr) -> (boxes, labels, scores)`` (original image coordinates)
    over ``gt`` from ``load_coco_gt``."""
    results = []
    for path, gb, gl in gt:
        boxes, labels, scores = predict(cv2.imread(str(path)))
        results.append((boxes, scores, labels, gb, gl))
    return score_predictions(results, class_names, **score_kwargs)


def onnx_predict_fn(model_path, target_size, model_cfg, optimization="all"):
    """``predict`` for ``evaluate_predict_fn`` running an exported ONNX model on the CPU.

    ``optimization="basic"`` keeps ONNX Runtime from fusing the quantize / dequantize pairs of an
    INT8 model into its int8 kernels, so the quantized layers run as float convolutions on the int8
    values. On x86 CPUs without VNNI those kernels can saturate, a loss the model itself doesn't have.
    """
    import onnxruntime as ort

    levels = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = levels[optimization]
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(image):
        x, scale = preprocess(image, target_size, model_cfg)
        boxes, labels, scores = session.run(None, {input_name: x[None]})
        boxes = boxes.copy()
        boxes[:, :4] /= scale  # the heading is unchanged by resize and pad
        return boxes, labels, scores

    return predict


def torch_predict_fn(wrapper, target_size, model_cfg, device="cuda"):
    """``predict`` for ``evaluate_predict_fn`` running the PyTorch export wrapper."""

    @torch.no_grad()
    def predict(image):
        x, scale = preprocess(image, target_size, model_cfg)
        boxes, labels, scores = wrapper(torch.from_numpy(x[None]).to(device))
        boxes = boxes.cpu().numpy().copy()
        boxes[:, :4] /= scale
        return boxes, labels.cpu().numpy(), scores.cpu().numpy()

    return predict


def draw_directed_boxes(image, boxes, labels=None, scores=None, class_names=None, colour=(0, 90, 255), score_thr=0.3):
    """Draws rotated boxes with an arrow from the centre along the heading (the box angle).

    The arrow length does not depend on w, so it stays visible on thin boxes: half the longer
    side, at least 12 px. The dot marks its tail."""
    img = image.copy()
    for k, (cx, cy, w, h, a) in enumerate(np.asarray(boxes).reshape(-1, 5)):
        if scores is not None and scores[k] < score_thr:
            continue
        cv2.polylines(img, [np.round(oriented_corners(cx, cy, w, h, a)).astype(np.int32)], True, colour, 2)
        length = max(max(w, h) / 2, 12)
        tail = (int(round(cx)), int(round(cy)))
        tip = (int(round(cx + length * math.cos(a))), int(round(cy + length * math.sin(a))))
        cv2.arrowedLine(img, tail, tip, colour, 2, tipLength=0.3)
        cv2.circle(img, tail, 3, colour, -1)
        if labels is not None and class_names is not None:
            text = class_names[int(labels[k])] + (f" {scores[k]:.2f}" if scores is not None else "")
            cv2.putText(img, text, (int(cx), int(cy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return img
