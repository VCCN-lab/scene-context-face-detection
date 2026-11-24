#!/usr/bin/env python3
"""
Does CNN predictivity change with layer depth?

For each network, task and condition, the noise-corrected predictivity of the
seven read-out layers (Supplementary Figure 5) is correlated with layer index
(Spearman, n = 7 layers). These are the layer-depth correlations reported in
Supplementary Note 6. The per-layer values are the ones plotted in the figure,
from encoding.layer_predictivity.

With seven points the smallest attainable p-values are coarse, the CIs are
wide (Fisher z with the Bonett-Wright SE for Spearman's rho), and the test asks
only whether predictivity rises or falls monotonically with depth.

Run: python3 run_layer_depth.py      (about a minute)
"""

import numpy as np
from scipy import stats

from facescenes import config as cfg
from facescenes import encoding as E
from facescenes import stats as S

NETWORKS = (("Untrained", "cnn_untrained.npz"),
            ("SceneFaceDet", "cnn_scenefacedet.npz"))


def main():
    layers = np.arange(1, E.N_CNN_LAYERS + 1)
    for task in cfg.TASKS:
        print(cfg.TASK_LABEL[task])
        for net, npz in NETWORKS:
            r = E.layer_predictivity(task, npz)
            for cond in cfg.CONDITIONS:
                rho = stats.spearmanr(layers, r[cond])
                vals = " ".join(f"{v:5.3f}" for v in r[cond])
                p = (f"= {rho.pvalue:.3f}" if rho.pvalue >= 0.001
                     else f"< .001 ({rho.pvalue:.1e})")
                lo, hi = S.spearman_ci(rho.statistic, len(layers))
                print(f"   {net:13s} {cfg.COND_LABEL[cond]:11s} "
                      f"rho = {rho.statistic:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], p {p}   "
                      f"layers 1-7: {vals}")
        print()


if __name__ == "__main__":
    main()
