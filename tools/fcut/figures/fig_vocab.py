"""
Figure 1 for segmentation.md: the vocabulary, on one curve.

Names every term the document uses: the initial plateau, the fundamental
that ends it, a cliff, a shelf, the drop, the collapse, and which side is
stiff and which is flexible.

`Chlorobenzene__LPYE__60-70__2` is used because it is a genuine staircase
and its initial plateau ends at the fundamental, which is the claim the
text makes about every curve.

Label convention: every label is a bold term with its explanatory
comment set underneath in smaller type, and every block is left-aligned
on the same edge, so a term is never confused with its comment. The
"initial plateau" block sits above the half-drop line and the "half the
total drop" block below it, so neither crosses the other's arrow.
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt

CACHE = ('/home/esteban/Simulation/DFT/separation_part2/runs/'
         'DW_prod_2026-08-23/r2_cache')
OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'Chlorobenzene__LPYE__60-70__2'
N_USED = 400
COLLAPSE_LEVEL = 0.5

TERM_SIZE = 10.5
NOTE_SIZE = 8.5

GREEN, ORANGE, PURPLE, BLUE, RED = (
    '#2e6b4a', '#8a6a1f', '#7a3fa0', '#1f4e79', '#b3402f')
GREY = '#7a7a7a'


def main():
    f = glob.glob(os.path.join(CACHE, f'{STEM}__r2__*.npz'))[0]
    d = np.load(f)
    fr, r2 = d['fcut_range'], d['r2_val']
    fund = 1.0 / N_USED
    lo, hi = float(np.nanmin(r2)), float(np.nanmax(r2))
    half = lo + COLLAPSE_LEVEL * (hi - lo)

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.plot(fr, r2, lw=1.7, color=BLUE, zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(fr[0], fr[-1])
    ax.set_ylim(-0.08, 1.20)
    ax.set_xlabel('cutoff frequency  $f_{cut}$', fontsize=10)
    ax.set_ylabel('$r^2$   (squared lag-1 autocorrelation)', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=.2, zorder=0)

    pending = []

    def label(x, y, term, note=None, color='k', arrow=None, side='right',
              gap=3.0, pad=7.0):
        """Bold term at (x, y), comment underneath, both left-aligned.

        Arrows are not placed from guessed coordinates. The text is drawn
        first, its rendered bounding box is measured, and the arrow leaves
        from the named edge of that box, so it can neither float away from
        its label nor start underneath it.
        """
        t = ax.text(x, y, term, fontsize=TERM_SIZE, fontweight='bold',
                    color=color, ha='left', va='bottom', zorder=7)
        n = None
        if note:
            n = ax.annotate(note, xy=(x, y), xycoords='data',
                            xytext=(0, -4), textcoords='offset points',
                            fontsize=NOTE_SIZE, color=color, ha='left',
                            va='top', zorder=7)
        if arrow:
            pending.append((t, n, arrow, color, side, gap, pad))

    def draw_arrows():
        """Anchor each arrow to the measured edge of its text block."""
        fig.canvas.draw()
        inv = ax.transData.inverted()
        for t, n, target, color, side, gap, pad in pending:
            boxes = [t.get_window_extent()]
            if n is not None:
                boxes.append(n.get_window_extent())
            x0 = min(b.x0 for b in boxes)
            x1 = max(b.x1 for b in boxes)
            y0 = min(b.y0 for b in boxes)
            y1 = max(b.y1 for b in boxes)
            if side == 'up':
                # Rise vertically to meet the curve head-on; a shallow
                # approach runs tangent to it near the tip. An x of None
                # means "straight above the bold term", used where the
                # curve is flat and any x on it will do.
                tb = t.get_window_extent()
                if target[0] is None:
                    px = 0.5 * (tb.x0 + tb.x1)
                    tx = inv.transform((px, 0))[0]
                else:
                    px = ax.transData.transform(target)[0]
                    tx = target[0]
                ty = target[1]
                if ty is None:
                    # Stop just under the curve at whatever x the arrow
                    # rises from, rather than at a y fixed by hand for a
                    # different x, which is how the tip ended up on it.
                    ty = float(np.interp(np.log10(tx), np.log10(fr), r2))
                target = (tx, ty)
                py = y1 + pad
            elif side == 'top':
                px, py = 0.5 * (t.get_window_extent().x0
                                + t.get_window_extent().x1), y1 + pad
            elif side == 'bottom':
                px, py = 0.5 * (x0 + x1), y0 - pad
            elif side == 'left':
                px, py = x0 - pad, 0.5 * (y0 + y1)
            else:
                px, py = x1 + pad, 0.5 * (y0 + y1)
            start = inv.transform((px, py))
            ax.annotate('', xy=target, xytext=tuple(start), zorder=6,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, shrinkA=0,
                                        shrinkB=gap))

    # which way is stiff, on its own line clear of everything
    ax.annotate('', xy=(1.2e-5, 1.16), xytext=(2.0e-4, 1.16),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
    ax.text(2.3e-4, 1.16, 'stiffer baseline', fontsize=NOTE_SIZE,
            color=GREY, va='center', ha='left')
    ax.annotate('', xy=(4.2e-1, 1.16), xytext=(4.5e-2, 1.16),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))
    ax.text(4.2e-2, 1.16, 'more flexible baseline', fontsize=NOTE_SIZE,
            color=GREY, va='center', ha='right')

    # bands
    ax.axvspan(fr[0], fund, color='#4c9f70', alpha=.16, lw=0, zorder=1)
    ax.axvspan(4.0e-3, 2.4e-2, color='#d9a441', alpha=.26, lw=0, zorder=1)

    # the half-drop line: its label goes BELOW it, the plateau label ABOVE
    ax.axhline(half, color=GREY, lw=1.0, ls=':', zorder=2)

    label(7.5e-5, 0.79, 'initial plateau',
          'every cutoff here returns\nthe same rigid baseline',
          color=GREEN, arrow=(None, 0.993), side='up')

    label(1.25e-5, half - 0.085, 'half the total drop',
          'a plateau below this line counts as "past the collapse",\n'
          'where a cutoff eats analyte peak area', color=GREY)

    # the fundamental, along its own line
    ax.axvline(fund, color=RED, lw=1.4, ls='--', zorder=3)
    ax.text(fund * 0.87, 0.02, 'the fundamental,  $1/n$',
            fontsize=TERM_SIZE - 1, fontweight='bold', color=RED,
            rotation=90, ha='right', va='bottom', zorder=7)

    # Dropped straight down onto the cliff: any approach from the left
    # runs along the plateau and crosses the curve repeatedly.
    label(3.30e-3, 1.050, 'cliff', color=PURPLE,
          arrow=(3.62e-3, 0.947), side='bottom')

    label(3.1e-2, 1.045, 'shelf',
          'flat against its neighbours,\nnot against the whole curve',
          color=ORANGE, arrow=(1.05e-2, 0.795), side='left')

    label(3.4e-2, 0.40, 'the drop', color=BLUE,
          arrow=(None, None), side='up', gap=6.0)

    i_min = int(np.nanargmin(r2))
    ax.plot([fr[i_min]], [r2[i_min]], 'o', ms=6.5, color=RED, zorder=8)
    # Set in three short lines so the block spans the minimum without
    # running off the axis, letting the arrow rise straight to it.
    label(5.0e-3, 0.15, 'the collapse',
          'the baseline has absorbed everything it can;\n'
          'past it the tail climbs back, as here, or stays near zero',
          color=RED, arrow=(fr[i_min], r2[i_min]), side='right',
          gap=10.0, pad=1.5)

    fig.tight_layout()
    draw_arrows()
    png = os.path.join(OUT, 'fig_vocab.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    print(png)


if __name__ == '__main__':
    main()
