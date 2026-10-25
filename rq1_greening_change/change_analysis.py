"""
Did places in Bengaluru that greened between 2000-04 and 2022-26 cool?

Input : data/bengaluru_regreening_change.csv  (from regreening_change.ipynb)
Output: rq1_greening_change/change_results.txt, figures/change_*.png

Each point is compared with itself over time, so fixed differences between places (building form, elevation,
soil) drop out. Temperature change is measured two ways:
  - city-relative : LST minus the city median of the same period (removes sensor and weather differences)
  - local         : LST minus the mean LST of the surrounding 1 km (like the park study's control ring)
Greenness change uses NDVI minus the city median of each period, for the same reason.
Built-up change (GHSL 2000 -> 2025) is held fixed, because new building lowers NDVI and raises LST at once.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'bengaluru_regreening_change.csv')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

out = []
def report(line=''):
    print(line)
    out.append(line)

def ols(df, y, xs):
    """Plain least squares. Returns coefficients and 95% half-widths, keyed by name, and R²."""
    X = np.column_stack([np.ones(len(df))] + [df[x].values for x in xs])
    b, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    resid = df[y].values - X @ b
    se = np.sqrt(np.diag(resid @ resid / (len(df) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((df[y] - df[y].mean()) ** 2).sum()
    names = ['const'] + xs
    return dict(zip(names, b)), dict(zip(names, 1.96 * se)), r2

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA)
n_all = len(df)
df = df.dropna(subset=['ndvi_1', 'ndvi_2', 'lst_1', 'lst_2', 'nbhd_1', 'nbhd_2', 'built_2000', 'built_2025'])
df = df[(df.mndwi_1 < 0) & (df.mndwi_2 < 0)]          # water in either period
report(f'Points: {n_all} sampled, {len(df)} used (land in both periods, no missing values)')

# ---------- 2. Changes ----------
for k in ['1', '2']:
    df[f'ndvi_anom_{k}'] = df[f'ndvi_{k}'] - df[f'ndvi_{k}'].median()
    df[f'lst_anom_{k}'] = df[f'lst_{k}'] - df[f'lst_{k}'].median()
    df[f'lst_local_{k}'] = df[f'lst_{k}'] - df[f'nbhd_{k}']
df['d_ndvi'] = df.ndvi_anom_2 - df.ndvi_anom_1
df['d_lst_city'] = df.lst_anom_2 - df.lst_anom_1
df['d_lst_local'] = df.lst_local_2 - df.lst_local_1
df['d_built'] = df.built_2025 - df.built_2000
df['d_built_pct'] = 100 * df.d_built
report(f'City median LST: {df.lst_1.median():.1f} °C (2000-04) -> {df.lst_2.median():.1f} °C (2022-26); '
       f'city median NDVI: {df.ndvi_1.median():.3f} -> {df.ndvi_2.median():.3f}')
report(f'Built-up share, median point: {100 * df.built_2000.median():.0f}% -> {100 * df.built_2025.median():.0f}%')

# ---------- 3. Groups ----------
df['change'] = np.select([df.d_ndvi >= 0.10, df.d_ndvi <= -0.10, df.d_ndvi.abs() < 0.03],
                         ['Greened (NDVI +0.10 or more)', 'Lost green (NDVI -0.10 or more)', 'Stable (within ±0.03)'],
                         'Other')
df['building'] = np.where(df.d_built < 0.05, 'Little new building (<5 pts)', 'New building (5+ pts)')
groups = (df[df.change != 'Other']
          .groupby(['building', 'change'])
          .agg(points=('d_ndvi', 'size'), d_ndvi=('d_ndvi', 'mean'), d_built_pct=('d_built_pct', 'mean'),
               d_lst_city=('d_lst_city', 'mean'), d_lst_local=('d_lst_local', 'mean')))
report('\nAverage change by group (temperature changes in °C, relative to city median and to own 1 km neighbourhood):')
report(groups.round(2).to_string())

calm = df[df.building.str.startswith('Little')]
g, s = calm[calm.change.str.startswith('Greened')], calm[calm.change.str.startswith('Stable')]
report(f'\nWith little new building: greened points changed by {g.d_lst_local.mean() - s.d_lst_local.mean():+.2f} °C '
       f'relative to stable points (local measure), {g.d_lst_city.mean() - s.d_lst_city.mean():+.2f} °C (city measure). '
       f'n = {len(g)} greened, {len(s)} stable.')

# ---------- 4. Regression and checks ----------
rows = []
for label, y in [('city-relative', 'd_lst_city'), ('local (1 km)', 'd_lst_local')]:
    b, ci, r2 = ols(df, y, ['d_ndvi', 'd_built'])
    rows.append(dict(version=f'All points, {label}', per_01_ndvi=b['d_ndvi'] / 10, ci=ci['d_ndvi'] / 10,
                     per_10pct_built=b['d_built'] / 10, r2=r2, n=len(df)))

b, ci, r2 = ols(calm, 'd_lst_local', ['d_ndvi', 'd_built'])
rows.append(dict(version='Little new building only, local', per_01_ndvi=b['d_ndvi'] / 10, ci=ci['d_ndvi'] / 10,
                 per_10pct_built=b['d_built'] / 10, r2=r2, n=len(calm)))

b, ci, r2 = ols(df, 'd_lst_local', ['d_ndvi', 'd_built', 'ndvi_anom_1'])
rows.append(dict(version='Plus starting NDVI, local', per_01_ndvi=b['d_ndvi'] / 10, ci=ci['d_ndvi'] / 10,
                 per_10pct_built=b['d_built'] / 10, r2=r2, n=len(df)))

# ~2 km blocks, to allow for neighbouring points being alike. Uses the city-relative measure: averaged over a
# 2 km block, "LST minus own 1 km neighbourhood" cancels out by construction, so the local measure is not usable here.
df['block'] = (((df.lon - df.lon.min()) // 0.018).astype(int).astype(str) + '_'
               + ((df.lat - df.lat.min()) // 0.018).astype(int).astype(str))
blocks = df.groupby('block').agg(d_ndvi=('d_ndvi', 'mean'), d_built=('d_built', 'mean'),
                                 d_lst_city=('d_lst_city', 'mean'), n=('d_ndvi', 'size'))
blocks = blocks[blocks.n >= 10]
b, ci, r2 = ols(blocks, 'd_lst_city', ['d_ndvi', 'd_built'])
rows.append(dict(version='2 km blocks, city-relative', per_01_ndvi=b['d_ndvi'] / 10, ci=ci['d_ndvi'] / 10,
                 per_10pct_built=b['d_built'] / 10, r2=r2, n=len(blocks)))
checks = pd.DataFrame(rows)
report('\nRegression of temperature change on greenness change and built-up change:')
report('(per_01_ndvi = °C per +0.1 NDVI; per_10pct_built = °C per +10 percentage points built-up)')
report(checks.round(3).to_string(index=False))

# Greened places started out different (more open land) from stable ones (more old built-up core).
# Compare greened vs stable only among points that started out alike.
def matched(col, edges, label):
    sub = calm.copy()
    sub['band'] = pd.cut(sub[col], edges)
    rows_ = []
    for band, x in sub.groupby('band', observed=True):
        gg, ss = x[x.change.str.startswith('Greened')], x[x.change.str.startswith('Stable')]
        if len(gg) >= 30 and len(ss) >= 30:
            rows_.append(dict(band=str(band), greened=len(gg), stable=len(ss),
                              diff_local=gg.d_lst_local.mean() - ss.d_lst_local.mean(),
                              diff_city=gg.d_lst_city.mean() - ss.d_lst_city.mean()))
    report(f'\nGreened minus stable (°C), little new building, within bands of {label}:')
    report(pd.DataFrame(rows_).round(2).to_string(index=False))

matched('ndvi_1', [-1, 0.15, 0.2, 0.25, 0.3, 0.4, 1], 'starting NDVI (2000-04)')
matched('built_2000', [-0.01, 0.05, 0.15, 0.3, 0.5, 1.01], 'starting built-up share (2000)')

# Rank-based check: robust to sensor differences in NDVI scale between Landsat 5/7 and 8/9
rk = lambda s: s.rank(pct=True)
rho, p = spearmanr(rk(df.ndvi_2) - rk(df.ndvi_1), rk(df.lst_2 - df.nbhd_2) - rk(df.lst_1 - df.nbhd_1))
report(f'\nRank check (change in NDVI rank vs change in local LST rank): Spearman {rho:+.2f} (p = {p:.2g})')
report('Reference points: same-period comparison across places (RQ1, 500 m squares) -1.57 °C per +0.1 NDVI; '
       'Freedom Park before/after -1.32 °C for NDVI +0.138 (about -0.96 °C per +0.1).')

# ---------- 5. Figures ----------
bins = np.arange(-0.30, 0.31, 0.05)
fig, ax = plt.subplots(figsize=(7, 4.3))
for (name, sub), col in zip(df.groupby('building'), ['#2b5f8a', '#d08c1f']):
    cut = pd.cut(sub.d_ndvi, bins)
    m = sub.groupby(cut, observed=True).d_lst_local.agg(['mean', 'sem', 'count'])
    m = m[m['count'] >= 30]
    x = [i.mid for i in m.index]
    ax.errorbar(x, m['mean'], yerr=1.96 * m['sem'], fmt='-o', ms=5, capsize=3, color=col, label=name)
ax.axhline(0, color='#999999', lw=0.8); ax.axvline(0, color='#999999', lw=0.8)
ax.set_xlabel('Change in greenness, 2000–04 → 2022–26 (NDVI, relative to city)')
ax.set_ylabel('Change in LST relative to\nown 1 km neighbourhood (°C)')
ax.set_title('Places that greened cooled; places that lost green warmed')
for s_ in ['top', 'right']:
    ax.spines[s_].set_visible(False)
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'change_lst_vs_ndvi.png'), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 6.8))
other = df[~df.change.isin(['Greened (NDVI +0.10 or more)', 'Lost green (NDVI -0.10 or more)'])]
ax.scatter(other.lon, other.lat, s=1, c='#dddddd', linewidths=0)
for name, col in [('Lost green (NDVI -0.10 or more)', '#b23a26'), ('Greened (NDVI +0.10 or more)', '#1a9850')]:
    d = df[df.change == name]
    ax.scatter(d.lon, d.lat, s=3, c=col, linewidths=0, label=f'{name} ({len(d):,})')
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=8, markerscale=4, frameon=False)
ax.set_title('Where Bengaluru greened and lost green, 2000–04 → 2022–26')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'change_map.png'), dpi=200); plt.close()

with open(os.path.join(HERE, 'change_results.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\nSaved figures to figures/ and numbers to rq1_greening_change/change_results.txt')
