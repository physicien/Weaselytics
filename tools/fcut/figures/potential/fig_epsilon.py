"""
Which way the log-transform offset cuts as the signal span changes.

The transform is ``log10(s - min(s) + epsilon)``, Navarro-Huerta et al.
Eq. (8), introduced in their Section 3.3.2 to compress the dynamic range
of a chromatogram whose peaks differ greatly in height.

Its local gain is ``1 / ((u + epsilon) ln 10)`` with ``u = s - min(s)``,
so the gain at the bottom of a record divided by the gain at its top is
exactly ``(S + epsilon) / epsilon`` for a record of span ``S``. That
ratio is the dynamic-range compression the transform applies, and it is
a pure function of ``S / epsilon``: an identity, with no fit and no data
behind it.

It falls as the span falls, so a fixed ``epsilon = 1`` treats a small
record more gently than a large one. Our chromatograms span 0.6 to 65.5,
well under the 60 to 2100 of the chromatograms Navarro-Huerta et al.
print, and they therefore sit at the gentle end of the same law.

Their figure scales are read off the printed axes and are approximate,
used here only as orders of magnitude. Nothing in the paper tests
``epsilon``: Section 3.3.2 states the value and the paper applies it
unchanged to every chromatogram it shows, including the 60 A.U. one in
Fig. 8.

Not part of ``render_all.sh``: this documents a staging constant rather
than one of the fcut selection figures.
"""

import glob  # noqa: I001
import os
import sys

import matplotlib
import numpy as np
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 200
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, '/home/esteban/Simulation/DFT/separation_part2/'
                   'Weaselytics')
from weaselytics.parsers import ParsedData  # noqa: E402

DATA = '/home/esteban/Simulation/DFT/separation_part2/data'
OUT = os.path.dirname(os.path.abspath(__file__))
EPS = 1.0
CNH, COUR = '#c2521f', '#1f4e79'

# Full y-axis scale of every chromatogram the paper prints, read off the
# printed axes. Approximate, and used only as orders of magnitude.
THEIRS = [(60.0, 'Fig. 8'), (400.0, 'Fig. 9'), (500.0, 'Fig. 6a'),
          (1700.0, 'Fig. 6b'), (2100.0, 'Fig. 6c')]


def amplitudes(root: str) -> np.ndarray:
    """Span of every parsable chromatogram under `root`."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, '**', '*.txt'),
                              recursive=True)):
        try:
            _, y = ParsedData(f).data
        except Exception:
            continue
        y = np.asarray(y, dtype=float)
        if y.size >= 10:
            out.append(float(y.max() - y.min()))
    return np.array(out)


def main() -> None:
    a = amplitudes(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3))

    ax = axes[0]
    span = np.geomspace(0.3, 4e3, 400)
    ax.plot(span, (span + EPS) / EPS, lw=2.4, color='0.3',
            label=r'$(S+\epsilon)\,/\,\epsilon$, exact')
    ax.fill_betweenx([1, 1e4], np.percentile(a, 5), np.percentile(a, 95),
                     color=COUR, alpha=.18, lw=0,
                     label=f'our {len(a)}, p5 to p95')
    med = float(np.median(a))
    ax.axvline(med, color=COUR, lw=2.0)
    ax.plot([med], [(med + EPS) / EPS], 'o', ms=9, color=COUR, zorder=5)
    ax.annotate(f'our median\nspan {med:.1f}, {(med + EPS) / EPS:.0f}x',
                xy=(med, (med + EPS) / EPS), xytext=(0.8, 300),
                fontsize=9, color=COUR,
                arrowprops=dict(arrowstyle='-', color=COUR, lw=1.2))
    for v, lab in THEIRS:
        ax.plot([v], [(v + EPS) / EPS], 's', ms=7, color=CNH, zorder=5)
        ax.text(v * 1.12, (v + EPS) / EPS * 0.72, lab, fontsize=8.4,
                color=CNH)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.3, 4e3)
    ax.set_ylim(1, 1e4)
    ax.set_xlabel('record span $S$ = max $-$ min')
    ax.set_ylabel('dynamic-range compression applied')
    ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Compression falls with span: our records get the '
                 'gentlest treatment', fontsize=10.5, loc='left')

    ax = axes[1]
    for s, col, lab in ((2100.0, '#8c3410', 'their Fig. 6c, span 2100'),
                        (60.0, CNH, 'their Fig. 8, span 60'),
                        (med, COUR, f'our median, span {med:.1f}')):
        u = np.linspace(0, s, 4000)
        ax.plot(u / s, np.log10(u + EPS) / np.log10(s + EPS),
                lw=2.3, color=col, label=lab)
    ax.plot([0, 1], [0, 1], color='0.4', lw=1.3, ls='--',
            label='no compression')
    ax.set_xlabel('position in the raw record, $(s-\\min s)\\,/\\,S$')
    ax.set_ylabel('position after the transform')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('Our curve is the closest to linear of the three',
                 fontsize=10.5, loc='left')

    fig.suptitle('Navarro-Huerta Eq. (8), '
                 '$\\log_{10}(s-\\min s+\\epsilon)$ at $\\epsilon=1$: a '
                 'fixed offset is gentler on a smaller record, not '
                 'harsher', fontsize=11.5, x=0.005, ha='left')
    fig.tight_layout()
    p = os.path.join(OUT, 'epsilon_condition.png')
    fig.savefig(p, bbox_inches='tight')
    print(p)


if __name__ == '__main__':
    main()
