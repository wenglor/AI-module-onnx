import json
import pathlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from datumaro.components.annotation import Bbox


def remap_coco_ids(coco_path, id_map):
    p = pathlib.Path(coco_path)
    data = json.loads(p.read_text())
    # remap categories
    cat_by_old = {c["id"]: c for c in data["categories"]}
    for old, new in id_map.items():
        if old in cat_by_old:
            cat_by_old[old]["id"] = new
    # drop any categories not in id_map (optional)
    data["categories"] = [c for c in data["categories"] if c["id"] in id_map.values()]
    # remap annotations
    for ann in data["annotations"]:
        if ann["category_id"] in id_map:
            ann["category_id"] = id_map[ann["category_id"]]
    p.write_text(json.dumps(data))
    print(f"Remapped {coco_path}")


def xywh_to_xyxy(bbox: Bbox):
    """
    Convert Datumaro (x, y, w, h) to xyxy
    """
    x1 = bbox.x
    y1 = bbox.y
    x2 = bbox.x + bbox.w
    y2 = bbox.y + bbox.h
    return x1, y1, x2, y2


def visualize_detection_results(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.0,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize object detection results on an image.

    Args:
        image: Input image in BGR format (H, W, 3)
        boxes: Bounding boxes in xyxy format, shape (N, 4)
        labels: Class labels, shape (N,)
        scores: Confidence scores, shape (N,)
        class_names: Optional list of class names for labels
        score_threshold: Minimum score to display
        box_color: Color for bounding boxes (B, G, R)
        text_color: Color for text labels (B, G, R)
        thickness: Line thickness for boxes
        save_path: Optional path to save the visualization

    Returns:
        Annotated image in BGR format
    """
    # Make a copy to avoid modifying the original
    vis_image = image.copy()

    # Filter by score threshold
    mask = scores > score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, thickness)

        # Prepare label text
        if class_names is not None and 0 <= label < len(class_names):
            label_text = f"{class_names[label]}: {score:.2f}"
        else:
            label_text = f"Class {label}: {score:.2f}"

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            box_color,
            -1,
        )

        # Draw label text
        cv2.putText(
            vis_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, vis_image)

    return vis_image


def visualize_bbox(
    image_path: str,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    model_input_size: Tuple[int, int] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    save_path: Optional[str] = None,
    display: bool = False,
) -> np.ndarray:
    """
    Convenience function to load an image and visualize bounding boxes. Additionally this function can apply YOLOX postprocess
    to the bounding boxes using the model input size parameter.

    Args:
        image_path: Path to the input image
        boxes: Bounding boxes (N, 4) in xyxy format
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        model_input_size: tuple (height, width). Used for YOLOX postprocessing.
            Use `None` if you are visualizing the raw image directly.
        class_names: Optional list of class names
        score_threshold: Minimum score to display
        save_path: Optional path to save the visualization
        display: Whether to display the image using cv2.imshow

    Returns:
        Annotated image in BGR format
    """
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    if model_input_size:
        input_h, input_w = model_input_size
    else:
        input_h, input_w = orig_h, orig_w

    # Calculate preprocessing transforms (matches preprocess_img_yolox)
    # Resize keeping aspect ratio to fit within (input_h, input_w)
    scale = min(input_h / orig_h, input_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    # Bottom/right padding (matches preprocess_img_yolox with pad bottom/right)
    pad_top = 0
    pad_left = 0

    # Convert bboxes from input tensor coordinates to original image coordinates
    boxes_original = boxes.copy()
    boxes_original[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale  # x coordinates
    boxes_original[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale  # y coordinates

    # Visualize on original image with converted coordinates
    vis_image = visualize_detection_results(
        image=image,
        boxes=boxes_original,
        labels=labels,
        scores=scores,
        class_names=class_names,
        score_threshold=score_threshold,
        save_path=save_path,
    )

    # Display if requested
    if display:
        cv2.imshow("Detection Results", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis_image
