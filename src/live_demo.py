"""Live ASL letter recognition from the webcam.

Loads the best model saved by the training cell in project.ipynb (resnet18_best.pt),
finds the hand in each camera frame with MediaPipe, crops it exactly like during
training and shows the predicted letter on screen.

Run from the project root or from src/:
    python src/live_demo.py
    python src/live_demo.py --img-size 160          # if the model was trained at 160 px
    python src/live_demo.py --weights other.pt --camera 1

Keys: q / Esc = quit, m = toggle mirrored preview.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from hand_crop import HandDetector, crop_hand, expand_box

SRC_DIR = Path(__file__).resolve().parent

DEFAULT_CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_meta(weights_path):
    """Read classes / img_size the notebook saved next to the weights (resnet18_best.json)."""
    meta_path = Path(weights_path).with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    print(f"{meta_path.name} not found, using default classes")
    return {}


def load_model(weights_path, device, num_classes):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


class LetterClassifier:
    """Hand detection + crop + ResNet prediction for a single RGB frame."""

    def __init__(self, weights_path, img_size=None, margin=0.25, device=None):
        meta = load_meta(weights_path)
        self.classes = meta.get("classes", DEFAULT_CLASSES)
        img_size = img_size or meta.get("img_size", 200)
        self.img_size = img_size
        self.device = device or pick_device()
        self.model = load_model(weights_path, self.device, len(self.classes))
        self.detector = HandDetector()
        self.margin = margin
        # must match eval_tf from build_transforms() in the notebook
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @torch.no_grad()
    def __call__(self, rgb):
        """rgb: HxWx3 uint8 array. Returns (probs or None, hand box or None, crop as PIL or None)."""
        img = Image.fromarray(rgb)
        box = self.detector.detect_box(img)
        if box is None:
            return None, None, None
        crop = crop_hand(img, box, self.margin)
        x = self.transform(crop).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(1)[0].cpu().numpy()
        return probs, expand_box(box, img.size, self.margin), crop

    def close(self):
        self.detector.close()


def draw_text(frame, text, org, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def run(args):
    weights = Path(args.weights)
    if not weights.is_absolute() and not weights.exists():
        weights = SRC_DIR / weights
    if not weights.exists():
        raise SystemExit(f"Model weights not found: {weights}\nTrain the model in project.ipynb first.")

    clf = LetterClassifier(weights, img_size=args.img_size, margin=args.margin)
    classes = clf.classes
    print(f"Loaded {weights} on {clf.device}, input size {clf.img_size}px, classes {''.join(classes)}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        clf.close()
        raise SystemExit(f"Cannot open camera {args.camera}. On macOS allow camera access for your "
                         "terminal/IDE in System Settings > Privacy & Security > Camera.")

    history = deque(maxlen=args.smooth)   # recent probability vectors, averaged for a stable answer
    mirror, fps, last = True, 0.0, time.time()
    window = "ASL letter recognition (q to quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera returned no frame, stopping.")
                break

            # the model gets the real (unmirrored) image; only the preview is mirrored
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            probs, box, crop = clf(rgb)
            if probs is None:
                history.clear()
            else:
                history.append(probs)

            view = cv2.flip(frame, 1) if mirror else frame.copy()
            h, w = view.shape[:2]

            if box is not None:
                x0, y0, x1, y1 = box
                if mirror:
                    x0, x1 = w - x1, w - x0
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if history:
                avg = np.mean(history, axis=0)
                top = np.argsort(avg)[::-1][:3]
                best, conf = classes[top[0]], float(avg[top[0]])
                if conf >= args.threshold:
                    draw_text(view, best, (20, 90), scale=3.0, color=(0, 255, 0), thickness=5)
                else:
                    draw_text(view, "?", (20, 90), scale=3.0, color=(0, 200, 255), thickness=5)
                draw_text(view, f"{conf:.0%}", (20, 130), scale=0.9)
                for k, i in enumerate(top):
                    draw_text(view, f"{classes[i]}: {avg[i]:.2f}", (20, 170 + 30 * k), scale=0.7)
            else:
                draw_text(view, "No hand detected", (20, 60), scale=1.0, color=(0, 0, 255))

            # small preview of what the model actually sees
            if crop is not None and args.show_crop:
                side = min(160, h // 3)
                thumb = cv2.cvtColor(np.asarray(crop.resize((side, side))), cv2.COLOR_RGB2BGR)
                if mirror:
                    thumb = cv2.flip(thumb, 1)
                view[10:10 + side, w - side - 10:w - 10] = thumb
                cv2.rectangle(view, (w - side - 10, 10), (w - 10, 10 + side), (255, 255, 255), 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            draw_text(view, f"{fps:.0f} FPS", (20, h - 20), scale=0.6)

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
    finally:
        cap.release()
        cv2.destroyAllWindows()
        clf.close()


def parse_args():
    p = argparse.ArgumentParser(description="Classify ASL letters from the webcam.")
    p.add_argument("--weights", default="resnet18_best.pt", help="state_dict saved by the notebook")
    p.add_argument("--img-size", type=int, default=None,
                   help="override; by default read from <weights>.json, else 200")
    p.add_argument("--margin", type=float, default=0.25, help="hand crop margin, as crop_margin in training")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--smooth", type=int, default=5, help="number of frames to average predictions over")
    p.add_argument("--threshold", type=float, default=0.5, help="below this confidence show '?'")
    p.add_argument("--no-crop-preview", dest="show_crop", action="store_false",
                   help="hide the model-input preview in the corner")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
