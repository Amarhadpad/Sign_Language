import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def load_labels(labels_path: Path) -> list[str]:
    labels: list[str] = []
    with labels_path.open(encoding="utf-8") as label_file:
        for expected_index, line in enumerate(label_file):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit():
                raise ValueError(f"Invalid label entry on line {expected_index + 1}: {line.rstrip()!r}")
            if int(parts[0]) != expected_index or not parts[1]:
                raise ValueError(f"Expected label index {expected_index} on line {expected_index + 1}.")
            labels.append(parts[1])

    if not labels:
        raise ValueError(f"No labels found in {labels_path}.")
    return labels


def prepare_hand_image(
    frame: np.ndarray,
    bbox: Sequence[int],
    offset: int = 20,
    image_size: int = 300,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    x, y, width, height = (int(value) for value in bbox)
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(frame_width, x + width + offset)
    y2 = min(frame_height, y + height + offset)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop_height, crop_width = crop.shape[:2]
    if crop_height == 0 or crop_width == 0:
        return None

    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    if crop_height / crop_width > 1:
        resized_width = max(1, min(image_size, math.ceil(image_size * crop_width / crop_height)))
        resized = cv2.resize(crop, (resized_width, image_size))
        gap = (image_size - resized_width) // 2
        canvas[:, gap:gap + resized_width] = resized
    else:
        resized_height = max(1, min(image_size, math.ceil(image_size * crop_height / crop_width)))
        resized = cv2.resize(crop, (image_size, resized_height))
        gap = (image_size - resized_height) // 2
        canvas[gap:gap + resized_height, :] = resized

    return (x1, y1, x2, y2), crop, canvas
