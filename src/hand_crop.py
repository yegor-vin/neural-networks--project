
import json
import os
from multiprocessing import Pool
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "hand_landmarker.task"


class HandDetector:
    def __init__(self, model_path=DEFAULT_MODEL, min_conf=0.1, upscale_to=256, pad_frac=1 / 3):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.upscale_to = upscale_to
        self.pad_frac = pad_frac

    def detect_box(self, img):
        img = img.convert("RGB")
        w0, h0 = img.size
        scale = max(1.0, self.upscale_to / max(w0, h0))
        if scale > 1:
            img = img.resize((round(w0 * scale), round(h0 * scale)), Image.BILINEAR)
        pad = round(max(img.size) * self.pad_frac)
        canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad))
        canvas.paste(img, (pad, pad))

        result = self.landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(canvas))
        )
        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        xs = np.array([p.x for p in lm]) * canvas.width
        ys = np.array([p.y for p in lm]) * canvas.height
        xs = (xs - pad) / scale
        ys = (ys - pad) / scale
        x0, x1 = np.clip([xs.min(), xs.max()], 0, w0)
        y0, y1 = np.clip([ys.min(), ys.max()], 0, h0)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return [float(x0), float(y0), float(x1), float(y1)]

    def close(self):
        self.landmarker.close()


def expand_box(box, img_size, margin=0.25, square=True):
    w_img, h_img = img_size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    if square:
        w = h = max(w, h)
    x0, x1 = max(0, cx - w / 2), min(w_img, cx + w / 2)
    y0, y1 = max(0, cy - h / 2), min(h_img, cy + h / 2)
    return round(x0), round(y0), round(x1), round(y1)


def crop_hand(img, box, margin=0.25, square=True):
    if box is None:
        return img
    return img.crop(expand_box(box, img.size, margin, square))



_worker_detector = None


def _init_worker(model_path):
    global _worker_detector
    _worker_detector = HandDetector(model_path)


def _detect_file(path):
    with Image.open(path) as img:
        return path, _worker_detector.detect_box(img)


def compute_boxes(paths, cache_path, model_path=DEFAULT_MODEL, num_workers=None, save_every=2000):
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    keys = [str(p) for p in paths]
    todo = sorted(set(k for k in keys if k not in cache))
    if todo:
        num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
        with Pool(num_workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
            it = pool.imap_unordered(_detect_file, todo, chunksize=32)
            for n, (k, box) in enumerate(tqdm(it, total=len(todo), desc="Detecting hands"), 1):
                cache[k] = box
                if n % save_every == 0:
                    cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))

    boxes = {k: cache[k] for k in keys}
    found = sum(b is not None for b in boxes.values())
    print(f"Hands found in {found}/{len(boxes)} images ({found / max(1, len(boxes)):.1%})")
    return boxes


class HandCropDataset(Dataset):
    def __init__(self, ds, paths, boxes, margin=0.25, square=True, drop_missing=False):
        self.ds = ds
        self.margin = margin
        self.square = square
        self.classes = getattr(ds, "classes", None)
        all_boxes = [boxes[str(p)] for p in paths]
        targets = list(getattr(ds, "targets", [None] * len(ds)))
        idx = range(len(ds))
        if drop_missing:
            idx = [i for i in idx if all_boxes[i] is not None]
        self.indices = list(idx)
        self.boxes = [all_boxes[i] for i in self.indices]
        self.targets = [targets[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.ds[self.indices[i]]
        return crop_hand(img, self.boxes[i], self.margin, self.square), y
