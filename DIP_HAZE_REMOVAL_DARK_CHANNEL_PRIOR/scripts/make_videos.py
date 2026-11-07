"""Side-by-side demo: hazy | V0 | V1 | V3 | GT  (mp4) and a transmission-map figure.

Usage: python scripts/make_videos.py [clip]   (default J004, REVIDE Test)
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcpvid import dehaze_video
from eval.revide_runner import load_clip

clip = sys.argv[1] if len(sys.argv) > 1 else "J004"
W = 480
hz, gt = load_clip("Test", clip, W)
runs = {"V0": dehaze_video(hz, "V0", return_stages=True),
        "V1": dehaze_video(hz, "V1", 0.9, return_stages=True),
        "V3": dehaze_video(hz, "V3", 0.9, return_stages=True)}
os.makedirs("figures", exist_ok=True)


def label(img, text):
    u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.putText(u8, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


h = hz[0].shape[0]
vw = cv2.VideoWriter(f"figures/demo_{clip}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (W * 5, h))
for k in range(len(hz)):
    cols = [label(hz[k], "hazy")] + [label(runs[v]["J"][k], v) for v in ["V0", "V1", "V3"]] + [label(gt[k], "GT")]
    vw.write(np.hstack(cols))
vw.release()

k = len(hz) // 2
fig, ax = plt.subplots(2, 4, figsize=(14, 5.5))
for a, im, t in zip(ax[0], [hz[k], runs["V0"]["J"][k], runs["V3"]["J"][k], gt[k]], ["hazy", "V0", "V3", "GT"]):
    a.imshow(np.clip(im, 0, 1)); a.set_title(t)
for a, v in zip(ax[1], ["V0", "V1", "V3"]):
    m = a.imshow(runs[v]["t"][k], vmin=0, vmax=1, cmap="viridis"); a.set_title(f"t ({v})")
ax[1][3].axis("off")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.colorbar(m, ax=ax[1][3], fraction=0.5)
fig.tight_layout(); fig.savefig(f"figures/transmission_{clip}.png", dpi=120)
print("wrote figures/demo_%s.mp4 and transmission_%s.png" % (clip, clip))
