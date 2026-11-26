"""Live ASL letter recognition from the webcam.

Loads the best model saved by the training cell in project.ipynb (resnet18_best.pt),
finds the hand in each camera frame with MediaPipe, crops it exactly like during
training and shows the predicted letter on screen.

Run from the project root or from src/:
    python src/live_demo.py
    python src/live_demo.py --img-size 160          # if the model was trained at 160 px
    python src/live_demo.py --weights other.pt --camera 1

Keys: q / Esc = quit, m = toggle mirrored preview.

Evaluation mode (final test on live data, e.g. several people in front of the camera):
    python src/live_demo.py --eval --person alice --shots 3
    python src/live_demo.py --summarize eval_results     # combined accuracy over all saved sessions

  The target letter is shown on screen. The person shows that letter and you press Space to
  record ONE trial (the smoothed prediction at that moment). With --shots N the target moves
  to the next letter after N trials; any letter key jumps to that letter.
  Keys in eval mode: A-Z = set target, Space = record, Backspace = undo last trial,
  Tab = toggle mirror, Esc = finish (letters are targets, so q/m do not quit/mirror here).
  Each session is saved as a CSV in --out-dir; a summary (overall accuracy with a 95 % Wilson
  interval, per-letter accuracy, most common confusions) is printed at the end.
"""

import argparse
import csv
import json
import math
import time
from collections import Counter, deque
from datetime import datetime
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
    def __init__(self, weights_path, img_size=None, margin=0.25, device=None):
        meta = load_meta(weights_path)
        self.classes = meta.get("classes", DEFAULT_CLASSES)
        img_size = img_size or meta.get("img_size", 160)
        self.img_size = img_size
        self.device = device or pick_device()
        self.model = load_model(weights_path, self.device, len(self.classes))
        self.detector = HandDetector()
        self.margin = margin
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @torch.no_grad()
    def __call__(self, rgb):
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


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


CSV_FIELDS = ["time", "person", "target", "pred", "conf", "correct"]


def summarize_trials(trials, classes=None, title="Evaluation summary"):
    n = len(trials)
    print(f"\n=== {title} ===")
    if n == 0:
        print("No trials recorded.")
        return {"n": 0}
    k = sum(t["correct"] for t in trials)
    lo, hi = wilson_interval(k, n)
    print(f"Overall accuracy: {k}/{n} = {k / n:.1%}   (95% Wilson CI {lo:.1%} - {hi:.1%})")

    people = sorted({t["person"] for t in trials})
    if len(people) > 1:
        print("\nPer person:")
        for person in people:
            ts = [t for t in trials if t["person"] == person]
            kp = sum(t["correct"] for t in ts)
            print(f"  {person:<15} {kp:>4}/{len(ts):<4} {kp / len(ts):.1%}")

    letters = classes or sorted({t["target"] for t in trials})
    per_letter = {}
    print("\nPer letter:")
    for c in letters:
        ts = [t for t in trials if t["target"] == c]
        if not ts:
            continue
        kc = sum(t["correct"] for t in ts)
        per_letter[c] = kc / len(ts)
        print(f"  {c}: {kc:>3}/{len(ts):<3} {kc / len(ts):.0%}")

    confusions = Counter((t["target"], t["pred"]) for t in trials if not t["correct"])
    if confusions:
        print("\nMost common confusions (true -> predicted):")
        for (true, pred), cnt in confusions.most_common(8):
            print(f"  {true} -> {pred}: {cnt}")
    return {"n": n, "correct": k, "accuracy": k / n, "ci95": [lo, hi], "per_letter": per_letter}


def load_trials(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    trials = []
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                row["correct"] = row["correct"] in ("1", "True", "true")
                row["conf"] = float(row["conf"])
                trials.append(row)
    print(f"Loaded {len(trials)} trials from {len(files)} file(s)")
    return trials


class EvalSession:
    def __init__(self, classes, person, shots, out_dir):
        self.classes = classes
        self.person = person
        self.shots = shots
        self.target = 0
        self.trials = []
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = out_dir / f"{person}_{stamp}.csv"

    @property
    def target_letter(self):
        return self.classes[self.target]

    def shots_for_target(self):
        return sum(t["target"] == self.target_letter for t in self.trials)

    def set_target(self, letter):
        if letter in self.classes:
            self.target = self.classes.index(letter)

    def record(self, pred, conf):
        trial = {"time": datetime.now().isoformat(timespec="seconds"), "person": self.person,
                 "target": self.target_letter, "pred": pred, "conf": round(conf, 4),
                 "correct": pred == self.target_letter}
        self.trials.append(trial)
        self.save()
        mark = "OK " if trial["correct"] else "ERR"
        print(f"[{mark}] target {trial['target']}  predicted {pred} ({conf:.0%})  "
              f"running acc {self.accuracy():.1%} over {len(self.trials)}")
        if self.shots and self.shots_for_target() >= self.shots:
            if self.target == len(self.classes) - 1:
                print("All letters done - press Esc to finish or keep recording.")
            else:
                self.target += 1

    def undo(self):
        if self.trials:
            t = self.trials.pop()
            self.set_target(t["target"])
            self.save()
            print(f"Removed last trial (target {t['target']}, predicted {t['pred']})")

    def accuracy(self):
        return sum(t["correct"] for t in self.trials) / len(self.trials) if self.trials else 0.0

    def save(self):
        with open(self.path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({**t, "correct": int(t["correct"])} for t in self.trials)

    def draw(self, view):
        h, w = view.shape[:2]
        x = w - 330
        draw_text(view, f"Show: {self.target_letter}", (x, h - 110), scale=1.3, color=(255, 255, 0), thickness=3)
        shots = f"  shot {self.shots_for_target() + 1}/{self.shots}" if self.shots else ""
        draw_text(view, f"person {self.person}{shots}", (x, h - 75), scale=0.6)
        n = len(self.trials)
        k = sum(t["correct"] for t in self.trials)
        draw_text(view, f"acc {k}/{n} = {self.accuracy():.1%}" if n else "Space = record trial",
                  (x, h - 45), scale=0.7)
        if self.trials:
            last = self.trials[-1]
            color = (0, 255, 0) if last["correct"] else (0, 0, 255)
            draw_text(view, f"last: {last['target']} -> {last['pred']}", (x, h - 15), scale=0.6, color=color)


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

    session = EvalSession(classes, args.person, args.shots, args.out_dir) if args.eval else None
    if session:
        print(f"Eval mode: saving trials to {session.path}")

    history = deque(maxlen=args.smooth)
    mirror, fps, last = True, 0.0, time.time()
    window = "ASL letter recognition (Esc to quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera returned no frame, stopping.")
                break

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
            if session:
                session.draw(view)

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if session:
                if key == 32:
                    if history:
                        avg = np.mean(history, axis=0)
                        i = int(np.argmax(avg))
                        session.record(classes[i], float(avg[i]))
                    else:
                        print("No hand detected - trial not recorded")
                elif key in (8, 127):
                    session.undo()
                elif key == 9:
                    mirror = not mirror
                elif key < 128 and chr(key).isalpha():
                    session.set_target(chr(key).upper())
            else:
                if key == ord("q"):
                    break
                if key == ord("m"):
                    mirror = not mirror
    finally:
        cap.release()
        cv2.destroyAllWindows()
        clf.close()
        if session:
            summary = summarize_trials(session.trials, classes, title=f"Session {session.person}")
            if session.trials:
                session.path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
                print(f"\nTrials saved to {session.path}")


def parse_args():
    p = argparse.ArgumentParser(description="Classify ASL letters from the webcam.")
    p.add_argument("--weights", default="resnet18_best.pt", help="state_dict saved by the notebook")
    p.add_argument("--img-size", type=int, default=None,
                   help="override; by default read from <weights>.json, else 160")
    p.add_argument("--margin", type=float, default=0.25, help="hand crop margin, as crop_margin in training")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--smooth", type=int, default=5, help="number of frames to average predictions over")
    p.add_argument("--threshold", type=float, default=0.5, help="below this confidence show '?'")
    p.add_argument("--no-crop-preview", dest="show_crop", action="store_false",
                   help="hide the model-input preview in the corner")
    p.add_argument("--eval", action="store_true", help="evaluation mode: record labelled trials, report accuracy")
    p.add_argument("--person", default="anon", help="eval mode: id of the person signing (used in file name)")
    p.add_argument("--shots", type=int, default=3,
                   help="eval mode: trials per letter before auto-advancing (0 = never advance)")
    p.add_argument("--out-dir", default=str(SRC_DIR.parent / "eval_results"), help="eval mode: where CSVs go")
    p.add_argument("--summarize", nargs="+", metavar="PATH",
                   help="no camera: print combined accuracy from saved trial CSVs / directories")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.summarize:
        summarize_trials(load_trials(args.summarize), title="Combined live evaluation")
    else:
        run(args)
