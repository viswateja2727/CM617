"""
Heat and green cover across Bengaluru's 500 m grid.

Input : data/bengaluru_grid_500m_v2.csv  (from gap_analysis_grid.ipynb)
Output: figures/*.png, supporting_green_vs_heat/results.txt, data/grid_with_scores.csv

Design rules (to avoid circular results):
- Green cover and built-up share compete for the same land, so built-up share is
  never added to a score that is then compared with green cover. It is only used
  to compare like with like (squares of similar built-up share).
- Population is not used: both population datasets (GHS-POP 2025, WorldPop 2020)
  show implausibly low population across south and east Bengaluru.
- Neighbouring squares are alike, so p-values from 2,800 squares are overstated.
  The main check is repeated on 2 km blocks, and results are reported as effect
  sizes in degrees C.

Part A (test): among squares with similar built-up share, are greener squares cooler?
Part B (description only, no statistics): where are the hot squares, and are they
built-up (where people likely live) or open land?
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # AURA/ (holds data/ and figures/)
DATA = os.path.join(ROOT, 'data', 'bengaluru_grid_500m_v2.csv')
FIG = os.path.join(ROOT, 'figures')
if len(sys.argv) > 1:
    DATA = sys.argv[1]
os.makedirs(FIG, exist_ok=True)

out = []
def report(line=''):
    print(line)
    out.append(line)

def ols(df, y, xs):
    """Plain least squares. Returns coefficients and 95% half-widths, keyed by name."""
    X = np.column_stack([np.ones(len(df))] + [df[x].values for x in xs])
    b, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    resid = df[y].values - X @ b
    s2 = resid @ resid / (len(df) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((df[y] - df[y].mean()) ** 2).sum()
    names = ['const'] + xs
    return dict(zip(names, b)), dict(zip(names, 1.96 * se)), r2

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA)
n_all = len(df)
df = df.dropna(subset=['lst', 'green', 'built'])
df = df[df.water < 0.5]          # lakes cool differently from vegetation - leave them out
report(f'Grid squares: {n_all}, used after removing water/missing: {len(df)}')

df['green_pct'] = df.green * 100
df['built_pct'] = df.built * 100
report(f'Correlation between green cover and built-up share: {df.green_pct.corr(df.built_pct):+.2f} '
       '(they compete for land, so built-up share is only used as a control)')

# ---------- 2. Part A: green vs heat, comparing like with like ----------
df['built_group'] = pd.qcut(df.built_pct, 4, labels=['Least built-up', 'Low-mid', 'Mid-high', 'Most built-up'])
rows = []
for bg, x in df.groupby('built_group', observed=True):
    x = x.copy()
    x['green_q'] = pd.qcut(x.green_pct.rank(method='first'), 4,
                           labels=['Least green', 'Q2', 'Q3', 'Most green'])
    t = x.groupby('green_q', observed=True).agg(lst=('lst', 'mean'), green=('green_pct', 'mean'),
                                                built=('built_pct', 'mean'))
    rows.append(dict(built_group=bg, squares=len(x),
                     built_range=f'{x.built_pct.min():.0f}-{x.built_pct.max():.0f}%',
                     **{f'LST {q}': t.loc[q, 'lst'] for q in t.index},
                     **{f'green {q}': t.loc[q, 'green'] for q in t.index},
                     **{f'built {q}': t.loc[q, 'built'] for q in t.index}))
within = pd.DataFrame(rows).set_index('built_group')
within['LST difference (least - most green)'] = within['LST Least green'] - within['LST Most green']
report('\nA. Within each built-up group, average LST (°C) from least-green to most-green quarter:')
report(within[['squares', 'built_range', 'LST Least green', 'LST Q2', 'LST Q3', 'LST Most green',
               'LST difference (least - most green)']].round(1).to_string())
report('\n   Average green cover (%) in those quarters:')
report(within[['green Least green', 'green Q2', 'green Q3', 'green Most green']].round(1).to_string())
report('\n   Average built-up (%) in those quarters (should be similar across a row):')
report(within[['built Least green', 'built Q2', 'built Q3', 'built Most green']].round(1).to_string())

# Regression: LST ~ green + built-up, plus robustness checks
b, ci, r2 = ols(df, 'lst', ['green_pct', 'built_pct'])
report('\nRegression LST ~ green cover + built-up (all squares):')
report(f'   +10% green cover: {10 * b["green_pct"]:+.2f} °C (±{10 * ci["green_pct"]:.2f}),  '
       f'+10% built-up: {10 * b["built_pct"]:+.2f} °C (±{10 * ci["built_pct"]:.2f}),  R² = {r2:.2f}')
checks = [('All squares (main)', 10 * b['green_pct'], 10 * ci['green_pct'], len(df))]

df['block'] = (((df.lon - df.lon.min()) // 0.018).astype(int).astype(str) + '_'
               + ((df.lat - df.lat.min()) // 0.018).astype(int).astype(str))      # ~2 km blocks
blocks = df.groupby('block').agg(lst=('lst', 'mean'), green_pct=('green_pct', 'mean'),
                                 built_pct=('built_pct', 'mean'), n=('lst', 'size'))
blocks = blocks[blocks.n >= 8]
bb, bci, _ = ols(blocks, 'lst', ['green_pct', 'built_pct'])
checks.append(('2 km blocks (neighbours averaged)', 10 * bb['green_pct'], 10 * bci['green_pct'], len(blocks)))

dry = df[df.water < 0.05]
bd, dci, _ = ols(dry, 'lst', ['green_pct', 'built_pct'])
checks.append(('Squares with < 5% water', 10 * bd['green_pct'], 10 * dci['green_pct'], len(dry)))

for bg, x in df.groupby('built_group', observed=True):
    bx, xci, _ = ols(x, 'lst', ['green_pct', 'built_pct'])
    checks.append((f'{bg} group only', 10 * bx['green_pct'], 10 * xci['green_pct'], len(x)))

checks = pd.DataFrame(checks, columns=['version', '°C per +10% green', '± (95%)', 'n'])
report('\nRobustness: effect of +10% green cover on LST, holding built-up fixed:')
report(checks.round(2).to_string(index=False))

bn, nci, _ = ols(df, 'lst', ['ndvi', 'built_pct'])
report(f'\nSame model with mean NDVI instead of green cover (no 0.4 threshold): '
       f'+0.1 NDVI = {bn["ndvi"] / 10:+.2f} °C (±{nci["ndvi"] / 10:.2f})')

# ---------- 3. Part B: where are the hot squares? (description only) ----------
hot = df.lst >= df.lst.quantile(0.75)
built_up = df.built_pct >= df.built_pct.median()
df['category'] = np.select([hot & built_up, hot & ~built_up], ['Hot, built-up', 'Hot, open land'], 'Not in hottest quarter')
c = df.groupby('category').agg(squares=('lst', 'size'), mean_lst=('lst', 'mean'),
                               mean_built_pct=('built_pct', 'mean'), mean_green_pct=('green_pct', 'mean'))
report('\nB. Hottest quarter of squares (LST >= '
       f'{df.lst.quantile(0.75):.1f} °C), split at the median built-up share ({df.built_pct.median():.0f}%):')
report(c.round(1).to_string())

# ---------- 4. Figures ----------
# Fig 1: LST by green quarter, one line per built-up group (ordinal -> one hue, light to dark)
q_cols = ['LST Least green', 'LST Q2', 'LST Q3', 'LST Most green']
shades = ['#9ecae1', '#6baed6', '#2171b5', '#08306b']
fig, ax = plt.subplots(figsize=(7, 4.5))
for (bg, r), col in zip(within.iterrows(), shades):
    ax.plot(range(4), r[q_cols].values, '-o', color=col, lw=2, ms=6, label=f'{bg} ({r.built_range} built)')
    ax.annotate(f'−{r["LST difference (least - most green)"]:.1f} °C', (3, r['LST Most green']),
                xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color='#333333')
ax.set_xticks(range(4), ['Least green\nquarter', '2nd', '3rd', 'Most green\nquarter'])
ax.set_xlim(-0.2, 3.6)
ax.set_ylabel('Average LST (°C)')
ax.set_title('Greener squares are cooler at every level of built-up share')
ax.grid(axis='y', color='#e5e5e5'); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, title='Built-up group', title_fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'heat_by_green_within_builtup.png'), dpi=200); plt.close()

# Fig 2: map of hot squares, built-up vs open land
def square_size(ax, n_deg=0.0045):
    """Marker size so each point is roughly one 500 m square on the map."""
    ax.figure.canvas.draw()
    x0, x1 = ax.get_xlim()
    width_pts = ax.get_window_extent().width * 72 / ax.figure.dpi
    return (width_pts * n_deg / (x1 - x0)) ** 2

colors = {'Not in hottest quarter': '#dddddd', 'Hot, open land': '#f4a261', 'Hot, built-up': '#c0392b'}
fig, ax = plt.subplots(figsize=(7, 7.6))
ax.set_aspect('equal')
ax.set_xlim(df.lon.min() - 0.01, df.lon.max() + 0.01); ax.set_ylim(df.lat.min() - 0.01, df.lat.max() + 0.01)
s = square_size(ax)
for name, col in colors.items():
    d = df[df.category == name]
    ax.scatter(d.lon, d.lat, s=s, marker='s', c=col, linewidths=0, label=f'{name} ({len(d)})')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, markerscale=2, frameon=False)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title('Hottest quarter of the city: built-up areas vs open land')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'hot_squares_map.png'), dpi=200); plt.close()

df.to_csv(os.path.join(ROOT, 'data', 'grid_with_scores.csv'), index=False)
with open(os.path.join(HERE, 'results.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\nSaved figures to figures/ and numbers to supporting_green_vs_heat/results.txt')
