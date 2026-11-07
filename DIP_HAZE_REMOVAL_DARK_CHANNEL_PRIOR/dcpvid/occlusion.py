"""Forward-backward flow consistency check (Sundaram et al., ECCV 2010)."""
import numpy as np

from .flow import warp


def valid_mask(f_cur_to_prev, f_prev_to_cur, a1=0.01, a2=0.5):
    """True where the cur->prev flow is reliable (not occluded, not inconsistent).

    A pixel x in cur maps to x+f_cp(x) in prev; the prev->cur flow sampled there
    should return to x, i.e. f_cp(x) + f_pc(x+f_cp(x)) ~ 0.
    """
    f_pc_at = warp(f_prev_to_cur, f_cur_to_prev)
    diff = f_cur_to_prev + f_pc_at
    lhs = (diff ** 2).sum(-1)
    rhs = a1 * ((f_cur_to_prev ** 2).sum(-1) + (f_pc_at ** 2).sum(-1)) + a2
    return lhs < rhs
