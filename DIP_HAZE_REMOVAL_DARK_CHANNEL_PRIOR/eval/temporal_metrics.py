"""Temporal-consistency metrics, measured against flow computed on the *ground-truth* clip
so the motion estimate is independent of dehazing artefacts.

flicker_index   mean |J_t - warp(J_{t-1})| over non-occluded pixels, divided by the same
                quantity for the GT clip. 1.0 = same temporal change as the real clean video;
                > 1 = extra (spurious) temporal variation, i.e. flicker.
roi_variance    REVIDE has a moving camera, so no pixel is static and the proposal's "static ROI"
                cannot be used as stated. Instead: split the frame into a GRID x GRID block grid,
                take each block's mean colour error e_t = mean(J_t) - mean(GT_t) (GT cancels the
                legitimate scene/camera change), and report the temporal variance of e_t averaged
                over blocks and channels. Low = the error is steady; high = it fluctuates (flicker).
"""
import numpy as np

from dcpvid.flow import flow, warp
from dcpvid.occlusion import valid_mask


GRID = 4


def _block_means(frames):
    stack = np.stack(frames)                                        # T,H,W,3
    T, H, W, _ = stack.shape
    hb, wb = H // GRID, W // GRID
    return stack[:, :hb * GRID, :wb * GRID].reshape(T, GRID, hb, GRID, wb, 3).mean(axis=(2, 4))


class ClipReference:
    """Caches GT flows/masks for one clip so every variant is scored against the same ones."""

    def __init__(self, gt_frames):
        self.gt = gt_frames
        self.f_cp, self.ok = [None], [None]
        for k in range(1, len(gt_frames)):
            fcp = flow(gt_frames[k], gt_frames[k - 1])
            fpc = flow(gt_frames[k - 1], gt_frames[k])
            self.f_cp.append(fcp)
            self.ok.append(valid_mask(fcp, fpc))
        self.gt_flicker = self._raw_flicker(gt_frames)
        self.gt_blocks = _block_means(gt_frames)

    def _raw_flicker(self, frames):
        errs = []
        for k in range(1, len(frames)):
            d = np.abs(frames[k] - warp(frames[k - 1], self.f_cp[k])).mean(-1)
            errs.append(d[self.ok[k]].mean())
        return float(np.mean(errs))

    def flicker_index(self, frames):
        return self._raw_flicker(frames) / self.gt_flicker

    def roi_variance(self, frames):
        e = _block_means(frames) - self.gt_blocks                   # T,G,G,3
        return float(e.var(axis=0).mean())
