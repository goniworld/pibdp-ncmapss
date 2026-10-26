"""Framework schematic of PI-BDP (fig0_framework.pdf/png)."""
import sys, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1] if len(sys.argv) > 1 else '.'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'savefig.dpi': 300})
INK, INK2, LINE = '#0b0b0b', '#52514e', '#8a8984'
BLUE, BLUE_BG = '#2a78d6', '#e7f1fb'
ORNG, ORNG_BG = '#eb6834', '#fbeee8'
GRN, GRN_BG = '#1baf7a', '#e5f4ec'

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.set_xlim(0, 100); ax.set_ylim(-4, 41); ax.axis('off')

def box(x, y, w, h, title, body, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2',
                                lw=0.9, ec=ec, fc=fc))
    ax.text(x + w / 2, y + h - 1.4, title, ha='center', va='top', fontsize=6.6, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h - 4.9, body, ha='center', va='top', fontsize=5.9, color=INK2, linespacing=1.3)
    return (x, y, w, h)

def arrow(p, q, color=LINE, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=0.9, color=color,
                                 linestyle=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))

# --- band labels ---
ax.text(0.5, 40.5, 'Offline: development fleet (run-to-failure and censored engines)', fontsize=7,
        color=ORNG, fontweight='bold', va='top')
ax.text(0.5, -1.2, 'Online: one monitored engine, updated after every flight', fontsize=7,
        color=BLUE, fontweight='bold', va='top')

# --- offline row ---
W, H, Y1 = 22.5, 12, 25
o1 = box(1.0, Y1, W, H, 'Healthy baseline', 'ridge regression f$_j$(w)\nper sensor, first n$_h$ cycles\n→ healthy SD σ$_j$', ORNG, ORNG_BG)
o2 = box(26.0, Y1, W, H, 'Fault signatures', 'end-of-life deltas of failed\nengines → signature matrix H\nsensor screening (c$_{min}$)', ORNG, ORNG_BG)
o3 = box(51.0, Y1, W, H, 'Fleet prior', 'per-engine exponential fits\n→ N(μ$_θ$, Σ$_θ$), shrinkage λ\nnoise σ$_ε$, threshold D', ORNG, ORNG_BG)
o4 = box(76.0, Y1, W, H, 'Lifetime & calibration', 'censored Weibull life model\nconformal constants c$_α$\nfrom LOUO-CV residuals', ORNG, ORNG_BG)

# --- online row ---
W2, Y2 = 15.4, 1.5
n1 = box(0.5, Y2, W2, 15, 'Flight data', 'w = (alt, Mach,\nTRA, T2)\nx$_s$ (14 sensors)\n0.1 Hz', BLUE, BLUE_BG)
n2 = box(17.3, Y2, W2, 15, 'Sensor deltas', 'cycle median of\ny$_j$ − f$_j$(w)\nscaled by σ$_j$\n→ Δz(t)', BLUE, BLUE_BG)
n3 = box(34.1, Y2, W2, 15, 'Health index', 'GLS damage\nestimate x̂(t)\nfrom Δz, H, R\nHI = 4 Σ x̂$_m$', BLUE, BLUE_BG)
n4 = box(50.9, Y2, W2, 15, 'Bayesian model', 'HI = a + b e$^{ct}$\nfleet prior +\nLaplace posterior', BLUE, BLUE_BG)
n5 = box(67.7, Y2, W2, 15, 'RUL distribution', 'sample θ, D → RUL\nWeibull re-weight\nmedian, quantiles\nconformal scaling', BLUE, BLUE_BG)
n6 = box(84.5, Y2, 15.0, 15, 'Decision', 'order maintenance\nwhen calibrated\n90% lower bound\n≤ L + Δt (Sec. 6.2)', GRN, GRN_BG)

# online flow
for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6)]:
    arrow((a[0] + a[2] + 0.3, a[1] + a[3] / 2), (b[0] - 0.3, b[1] + b[3] / 2), color=BLUE)

# offline → online
def down(o, n, dx_o=0.0, dx_n=0.0):
    arrow((o[0] + o[2] / 2 + dx_o, o[1] - 0.3), (n[0] + n[2] / 2 + dx_n, n[1] + n[3] + 0.3), color=ORNG, ls=(0, (3, 2)))
down(o1, n2)
down(o2, n3)
down(o3, n4)
down(o4, n5)

fig.savefig(f'{out}/fig0_framework.png', bbox_inches='tight')
fig.savefig(f'{out}/fig0_framework.pdf', bbox_inches='tight')
print('saved')
