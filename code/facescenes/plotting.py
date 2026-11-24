"""
Shared figure style.

Conventions the journal requires, applied here rather than repeated per figure:
axes start at zero, y-ticks step by 0.1 (0.05 where the range is small) and
always include zero, and the top and right spines are removed. Where an axis
break is unavoidable it is drawn explicitly with break_marker.

Every predictivity axis in the release says "Noise-corrected", because it plots
predictivity divided by the split-half reliability of the latencies it is scored
against, so the ceiling is 1 rather than the reliability. The sole exception is
Supplementary Figure 4, which repeats the single-feature analysis uncorrected and
therefore uses RAW_PREDICTIVITY_LABEL. The measure
itself is a signed squared Spearman correlation between predicted and observed
held-out latency, which is not the coefficient of determination of a fitted model,
so calling it explained variance would be misleading. Regression coefficients are
NOT noise-corrected -- the correction rescales a score, not a slope -- so no beta
axis carries the phrase.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg

FONT = 28
PREDICTIVITY_LABEL = "Noise-corrected predictivity ($r^2$)"
UNIQUE_LABEL = "Noise-corrected unique predictivity ($\\Delta r^2$)"
# The one axis that is NOT corrected: Supplementary Figure 4 repeats the
# single-feature analysis on the raw scale, to show what the correction does.
RAW_PREDICTIVITY_LABEL = "Predictivity ($r^2$)"

# The journal does not allow panel titles in figure files; the captions say
# which task each panel shows. Set to True for titled panels when exploring.
SHOW_TITLES = False

SOURCE_DATA = cfg.ROOT / "derivatives" / "source_data"


def use_style(dpi=300):
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"
    # Embed TrueType fonts in the PDFs, so text stays editable in Illustrator.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def title(target, text, **kw):
    """Task title on an axes (or a figure's suptitle), drawn only if SHOW_TITLES."""
    if not SHOW_TITLES:
        return
    if hasattr(target, "suptitle"):
        target.suptitle(text, **kw)
    else:
        target.set_title(text, **kw)


def clean(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def zero_anchored_ticks(lo, hi, step=None):
    """Ticks that always include zero and step by a round amount."""
    if step is None:
        step = 0.1 if (max(hi, 0) - min(lo, 0)) > 0.3 else 0.05
    lo_t = np.floor(min(lo, 0) / step) * step
    hi_t = np.ceil(max(hi, 0) / step) * step
    ticks = np.arange(lo_t, hi_t + step / 2, step)
    return ticks[(ticks >= lo_t - 1e-9) & (ticks <= hi_t + 1e-9)], (lo_t, hi_t)


def break_marker(ax, stub=None, y=0.03, d=0.018):
    """Two short diagonals on the y-spine, marking a truncated axis.

    With `stub` -- the short axes drawn below the break, which carries the zero
    tick -- the two diagonals straddle the gap instead: one at the foot of the
    main axes and one above the stub. That is what a broken axis looks like when
    zero is still shown, as it is in Figure 1b.
    """
    kw = dict(color="k", clip_on=False, lw=1.5)
    if stub is None:
        for off in (0, 2.2 * d):
            ax.plot((-d, +d), (y + off - d, y + off + d),
                    transform=ax.transAxes, **kw)
        return
    ax.plot((-d, +d), (-d / 2, +d / 2), transform=ax.transAxes, **kw)
    # In stub coordinates: 1.35 sits inside the gap, 0.65 just below it. The
    # stub is a twentieth of the height of the main axes, so these fractions are
    # short marks on the page.
    for pos in (0.65, 1.35):
        stub.plot((-d, +d), (pos - 0.25, pos + 0.25),
                  transform=stub.transAxes, **kw)


def significance(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def one_sided_errorbar(values, sd):
    """Error bar drawn away from zero only, so bars stay readable.

    Not matplotlib's `lolims`, which declares the value a lower limit and draws
    a caret -- a different claim.
    """
    values = np.asarray(values)
    sd = np.asarray(sd)
    return [np.where(values >= 0, 0, sd), np.where(values >= 0, sd, 0)]


def condition_colours():
    return {"preview": cfg.C_PREVIEW, "no_preview": cfg.C_NO_PREVIEW}


def save(fig, name, data=None):
    """Write the figure as an RGB PNG on white and as a vector PDF, and its
    source data as CSV.

    The PNG is flattened to RGB because some submission systems render a
    transparent background as black. `data` is a DataFrame holding the numbers
    plotted in this panel; it goes to derivatives/source_data/<name>.csv, so the
    source data are produced by the same code as the figure and cannot drift.
    """
    from PIL import Image
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", transparent=False)
    Image.open(path).convert("RGB").save(path, dpi=(fig.dpi, fig.dpi))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    if data is not None:
        SOURCE_DATA.mkdir(parents=True, exist_ok=True)
        data.to_csv(SOURCE_DATA / (path.stem + ".csv"), index=False,
                    float_format="%.6g")
    return path
