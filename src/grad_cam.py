
"""
    target_layer = find_last_conv(model)
    show_gradcam(model, target_layer, test_loader, classes)
    show_class_cams(model, target_layer, valid_loader, classes)
    energy = hand_energy(model, target_layer, valid_loader_base, boxes)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hand_crop import expand_box


def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("model has no nn.Conv2d layer; pass target_layer explicitly")
    return last


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._handles = [target_layer.register_forward_hook(self._save_acts)]

    def _save_acts(self, module, inp, out):
        self._acts = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_grads)

    def _save_grads(self, grad):
        self._grads = grad.detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, images, targets=None):
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.detach().to(device).clone().requires_grad_(True)

        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(images)
            if targets is None:
                targets = logits.argmax(1)
            elif isinstance(targets, int):
                targets = torch.full((len(images),), targets, device=device)
            else:
                targets = torch.as_tensor(targets, device=device)
            score = logits.gather(1, targets.view(-1, 1)).sum()
            score.backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * self._acts).sum(1, keepdim=True))    
        cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cams = cams.squeeze(1)
        flat = cams.flatten(1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cams = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cams)

        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)
        return cams.cpu(), logits.detach().cpu()



def denormalize(img, mean=(0.5,) * 3, std=(0.5,) * 3):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam, alpha=0.45, cmap="jet"):
    heat = plt.get_cmap(cmap)(np.asarray(cam))[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def _take(loader, n, only=None, model=None, device=None, seed=None, batch=64):
    ds = loader.dataset
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    order = torch.randperm(len(ds), generator=g).tolist()
    xs, ys, got = [], [], 0
    for start in range(0, len(order), batch):
        items = [ds[i] for i in order[start:start + batch]]
        x = torch.stack([it[0] for it in items])
        y = torch.tensor([it[1] for it in items])
        if only is not None:
            with torch.no_grad():
                pred = model(x.to(device)).argmax(1).cpu()
            keep = (pred != y) if only == "wrong" else (pred == y)
            x, y = x[keep], y[keep]
        xs.append(x)
        ys.append(y)
        got += len(y)
        if got >= n or only is None:
            break
    return torch.cat(xs)[:n], torch.cat(ys)[:n]


def show_gradcam(model, target_layer, loader, classes, n=8, only=None, mean=(0.5,) * 3,
                 std=(0.5,) * 3, title=None, seed=None):
    device = next(model.parameters()).device
    model.eval()
    x, y = _take(loader, n, only=only, model=model, device=device, seed=seed)
    if len(x) == 0:
        print("no samples to show")
        return
    with GradCAM(model, target_layer) as gc:
        cams, logits = gc(x)
    probs = logits.softmax(1)
    pred = probs.argmax(1)

    n = len(x)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5), squeeze=False)
    for j in range(n):
        img = denormalize(x[j], mean, std)
        ok = pred[j] == y[j]
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"true {classes[y[j]]}", fontsize=9)
        axes[1, j].imshow(overlay(img, cams[j]))
        axes[1, j].set_title(f"pred {classes[pred[j]]} ({probs[j, pred[j]]:.0%})",
                             fontsize=9, color="black" if ok else "red")
    for ax in axes.flat:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_compare_classes(model, target_layer, image, classes, class_names, mean=(0.5,) * 3,
                         std=(0.5,) * 3):
    idxs = [classes.index(c) for c in class_names]
    batch = image.unsqueeze(0).repeat(len(idxs), 1, 1, 1)
    with GradCAM(model, target_layer) as gc:
        cams, logits = gc(batch, idxs)
    probs = logits[0].softmax(0)
    img = denormalize(image, mean, std)
    fig, axes = plt.subplots(1, len(idxs) + 1, figsize=(2.5 * (len(idxs) + 1), 2.8))
    axes[0].imshow(img)
    axes[0].set_title("input")
    for ax, c, i, cam in zip(axes[1:], class_names, idxs, cams):
        ax.imshow(overlay(img, cam))
        ax.set_title(f"{c} ({probs[i]:.0%})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def class_mean_cams(model, target_layer, loader, num_classes, max_batches=None):
    sums, counts = None, torch.zeros(num_classes)
    with GradCAM(model, target_layer) as gc:
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            cams, _ = gc(x, y)
            if sums is None:
                sums = torch.zeros(num_classes, *cams.shape[-2:])
            sums.index_add_(0, y, cams)
            counts += torch.bincount(y, minlength=num_classes).float()
    return sums / counts.clamp_min(1).view(-1, 1, 1), counts


def show_class_cams(model, target_layer, loader, classes, max_batches=None, cols=8, title=None):
    means, counts = class_mean_cams(model, target_layer, loader, len(classes), max_batches)
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, c, m, k in zip(axes.flat, classes, means, counts):
        ax.imshow(m, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"{c} (n={int(k)})", fontsize=9)
    fig.suptitle(title or "Mean Grad-CAM per class")
    plt.tight_layout()
    plt.show()
    return means


def _unwrap(ds):

    from torch.utils.data import Subset
    from torchvision import datasets as tvd

    from hand_crop import HandCropDataset

    margin = None
    index = None

    def compose(idx):
        nonlocal index
        index = list(idx) if index is None else [idx[i] for i in index]

    cur = ds
    while True:
        if isinstance(cur, HandCropDataset):
            margin = cur.margin
            compose(cur.indices)
            cur = cur.ds
        elif isinstance(cur, Subset) or hasattr(cur, "indices") and hasattr(cur, "ds"):
            compose(cur.indices)
            cur = cur.dataset if isinstance(cur, Subset) else cur.ds
        elif hasattr(cur, "ds"):
            cur = cur.ds
        elif isinstance(cur, tvd.ImageFolder):
            paths = [p for p, _ in cur.samples]
            break
        elif hasattr(cur, "files"):
            paths = list(cur.files)
            break
        else:
            raise TypeError(f"don't know how to get paths from {type(cur).__name__}")
    if index is not None:
        paths = [paths[i] for i in index]
    return [str(p) for p in paths], margin


def _box_mask(box, img_size, out_hw, margin=None):
    w, h = img_size
    x0, y0, x1, y1 = box
    if margin is not None:
        cx0, cy0, cx1, cy1 = expand_box(box, img_size, margin)
        x0, x1, y0, y1 = x0 - cx0, x1 - cx0, y0 - cy0, y1 - cy0
        w, h = cx1 - cx0, cy1 - cy0
    H, W = out_hw
    sx, sy = W / w, H / h
    mask = torch.zeros(H, W)
    mask[max(0, int(y0 * sy)):int(np.ceil(y1 * sy)), max(0, int(x0 * sx)):int(np.ceil(x1 * sx))] = 1
    return mask


def hand_energy(model, target_layer, loader, boxes, max_batches=None):
    paths, margin = _unwrap(loader.dataset)
    out = {"energy": [], "area": [], "correct": [], "label": []}
    pos = 0
    with GradCAM(model, target_layer) as gc:
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            cams, logits = gc(x)
            pred = logits.argmax(1)
            for j in range(len(x)):
                p = paths[pos + j]
                box = boxes.get(p)
                if box is None:
                    continue
                with Image.open(p) as im:
                    size = im.size
                mask = _box_mask(box, size, cams.shape[-2:], margin)
                cam = cams[j]
                out["energy"].append(float((cam * mask).sum() / cam.sum().clamp_min(1e-8)))
                out["area"].append(float(mask.mean()))
                out["correct"].append(bool(pred[j] == y[j]))
                out["label"].append(int(y[j]))
            pos += len(x)
    return {k: np.array(v) for k, v in out.items()}


def summarize_energy(results):
    print(f"{'model/data':28s} {'n':>6s} {'energy':>8s} {'box area':>9s} {'ratio':>6s}"
          f" {'E|right':>8s} {'E|wrong':>8s}")
    for name, r in results.items():
        if len(r["energy"]) == 0:
            print(f"{name:28s} no images with a detected hand")
            continue
        e, a, c = r["energy"], r["area"], r["correct"]
        right = e[c].mean() if c.any() else float("nan")
        wrong = e[~c].mean() if (~c).any() else float("nan")
        print(f"{name:28s} {len(e):6d} {e.mean():8.2f} {a.mean():9.2f} {e.mean() / a.mean():6.2f}"
              f" {right:8.2f} {wrong:8.2f}")
