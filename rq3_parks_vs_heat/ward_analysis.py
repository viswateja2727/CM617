"""
Are Bengaluru's parks in the wards with the highest heat vulnerability?

Inputs (data/):
  bengaluru_ward_heat.csv          ward average LST, land only (from ward_heat.ipynb)
  bbmp_wards_198.geojson           198 BBMP ward outlines (2015 map, 2009 delimitation)
  raw/census2011_wardwise.csv      ward population, Census 2011 (OpenCity)
  raw/bbmp_parks_map_2023.kml      887 BBMP parks as points (OpenCity)
Outputs: figures/ward_*.png, rq3_parks_vs_heat/ward_results.txt, data/ward_scores.csv

Method:
  vulnerability = 1/2 heat rank + 1/2 population-density rank (each 0-1; ranks, because
                  density is very skewed). No built-up share or green cover in the score.
  parks         = BBMP parks per km2 of ward.
  Four groups by splitting both at the median; chi-square test of independence;
  Spearman correlation for direction. Checks: heat only, density only, and parks per
  10,000 people instead of per km2.
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import PolyCollection
from scipy.stats import chi2_contingency, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # AURA/ (holds data/ and figures/)
DATA = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figures')
HEAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'bengaluru_ward_heat.csv')
os.makedirs(FIG, exist_ok=True)

out = []
def report(line=''):
    print(line)
    out.append(line)

# ---------- 1. Wards: outlines and area ----------
gj = json.load(open(os.path.join(DATA, 'bbmp_wards_198.geojson'), encoding='utf-8'))
LAT0 = np.radians(12.97)

def rings_of(geom):
    polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
    return [np.array(p[0]) for p in polys]          # outer rings only

def area_km2(ring):
    x = ring[:, 0] * 111.32 * np.cos(LAT0)
    y = ring[:, 1] * 110.57
    return abs(0.5 * np.sum(x * np.roll(y, 1) - np.roll(x, 1) * y))

wards = pd.DataFrame([dict(ward_num=f['properties']['ward_num'], ward_name=f['properties']['ward_name'],
                           rings=rings_of(f['geometry'])) for f in gj['features']])
wards['area_km2'] = wards.rings.apply(lambda rs: sum(area_km2(r) for r in rs))

# ---------- 2. Population (Census 2011) ----------
census = pd.read_csv(os.path.join(DATA, 'raw', 'census2011_wardwise.csv'))
census = census.rename(columns={'Ward Num': 'ward_num', 'Population': 'population'})
wards = wards.merge(census[['ward_num', 'population']], on='ward_num', how='left')
wards['density'] = wards.population / wards.area_km2

# ---------- 3. Parks (BBMP 2023) -> ward ----------
kml = open(os.path.join(DATA, 'raw', 'bbmp_parks_map_2023.kml'), encoding='utf-8', errors='ignore').read()
pts = np.array([[float(a), float(b)] for a, b in
                re.findall(r'<Point>\s*<coordinates>\s*(\d+\.\d+),(\d+\.\d+)', kml)])
pts = np.unique(pts, axis=0)                          # drop exact duplicates
park_ward = np.zeros(len(pts), int)
for _, w in wards.iterrows():
    for r in w.rings:
        park_ward[(park_ward == 0) & Path(r).contains_points(pts)] = w.ward_num
counts = pd.Series(park_ward[park_ward > 0]).value_counts()
wards['parks'] = wards.ward_num.map(counts).fillna(0).astype(int)
wards['parks_per_km2'] = wards.parks / wards.area_km2
wards['parks_per_10k'] = wards.parks / wards.population * 1e4
report(f'Parks: {len(pts)} unique locations, {int((park_ward > 0).sum())} inside a ward; '
       f'{int((wards.parks > 0).sum())} of 198 wards have at least one park')

# ---------- 4. Heat ----------
heat = pd.read_csv(HEAT)[['ward_num', 'lst', 'green', 'water']]
wards = wards.merge(heat, on='ward_num', how='left')
report(f'Wards with heat values: {wards.lst.notna().sum()} of 198')
wards = wards.dropna(subset=['lst', 'population'])

# ---------- 5. Vulnerability and four groups ----------
rank = lambda s: s.rank(pct=True)
wards['heat_rank'] = rank(wards.lst)
wards['density_rank'] = rank(wards.density)
wards['vulnerability'] = 0.5 * wards.heat_rank + 0.5 * wards.density_rank

GROUPS = ['Priority gap', 'Well-targeted', 'Possibly inefficient', 'Low priority']

def classify(vuln, parks):
    hv, hp = vuln > vuln.median(), parks > parks.median()
    return np.select([hv & ~hp, hv & hp, ~hv & hp], GROUPS[:3], GROUPS[3])

def test(vuln, parks, label):
    g = pd.Series(classify(vuln, parks), index=vuln.index)
    table = pd.crosstab(vuln > vuln.median(), parks > parks.median())
    chi2, p, _, _ = chi2_contingency(table)
    rho, rp = spearmanr(vuln, parks)
    counts = g.value_counts().reindex(GROUPS).fillna(0).astype(int)
    return dict(version=label, **counts.to_dict(), chi2=chi2, chi2_p=p, spearman=rho, spearman_p=rp), g

main, wards['group'] = test(wards.vulnerability, wards.parks_per_km2, 'Main: heat + density, parks per km2')

report(f'\nMedian split points: vulnerability {wards.vulnerability.median():.2f}, '
       f'parks per km2 {wards.parks_per_km2.median():.2f}')
summary = wards.groupby('group').agg(wards=('ward_num', 'size'), people_2011=('population', 'sum'),
                                     area_km2=('area_km2', 'sum'), parks=('parks', 'sum'),
                                     mean_lst=('lst', 'mean'))
summary['people_per_km2'] = summary.people_2011 / summary.area_km2
summary['parks_per_km2'] = summary.parks / summary.area_km2
summary['people_per_park'] = summary.people_2011 / summary.parks.replace(0, np.nan)
summary['share_of_people_pct'] = 100 * summary.people_2011 / summary.people_2011.sum()
summary['share_of_parks_pct'] = 100 * summary.parks / summary.parks.sum()
summary = summary.reindex(GROUPS)
report('\nFour groups (main version):')
report(summary.round(1).to_string())

checks = [main]
checks.append(test(wards.heat_rank, wards.parks_per_km2, 'Heat only, parks per km2')[0])
checks.append(test(wards.density_rank, wards.parks_per_km2, 'Density only, parks per km2')[0])
checks.append(test(wards.vulnerability, wards.parks_per_10k, 'Heat + density, parks per 10k people')[0])
checks = pd.DataFrame(checks)
report('\nChi-square and Spearman, main version and checks:')
report(checks.round(3).to_string(index=False))

# Ratio-free checks. Wards were drawn to hold similar populations, so density is almost
# 1 / area, and both parks-per-km2 and parks-per-person inherit ward size. These checks
# use raw park counts and compare wards of similar size instead.
report(f'\nWard area vs population density: Spearman {spearmanr(wards.area_km2, wards.density)[0]:+.2f} '
       '(density is mostly ward size, so ratio-based checks above are unreliable)')
wards['size_group'] = pd.qcut(wards.area_km2, 4, labels=['Smallest', 'Small-mid', 'Mid-large', 'Largest'])
rows = []
for sg, x in wards.groupby('size_group', observed=True):
    rp, pp = spearmanr(x.population, x.parks)
    rh, ph = spearmanr(x.lst, x.parks)
    rows.append(dict(size_group=sg, wards=len(x), area_km2=f'{x.area_km2.min():.1f}-{x.area_km2.max():.1f}',
                     parks_vs_population=rp, p_pop=pp, parks_vs_heat=rh, p_heat=ph))
report('\nWithin wards of similar size, Spearman of park COUNT with population and with heat:')
report(pd.DataFrame(rows).round(3).to_string(index=False))

X = np.column_stack([np.ones(len(wards)), np.log(wards.population), np.log(wards.area_km2), wards.lst])
y = np.log(wards.parks + 1)
b, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ b
se = np.sqrt(np.diag(resid @ resid / (len(y) - 4) * np.linalg.inv(X.T @ X)))
report('\nRegression log(parks + 1) ~ log(population) + log(area) + heat:')
for name, bb, ss in zip(['log population', 'log area', 'heat (per °C)'], b[1:], se[1:]):
    report(f'   {name:<15} {bb:+.3f} ± {1.96 * ss:.3f}')
report(f'   => each +1 °C of ward heat goes with about {100 * (1 - np.exp(b[3])):.0f}% fewer parks, '
       'for wards of equal population and area')
report(f'\nParks vs satellite green cover: Spearman {spearmanr(wards.parks_per_km2, wards.green)[0]:+.2f}')
report(f'If every park were 1 ha, parks would cover a median {(wards.parks * 0.01 / wards.area_km2 * 100).median():.1f}% '
       'of a ward, too little to explain ward-level heat differences')

zero = wards[wards.parks == 0]
report(f'\nWards with no park in the 2023 list: {len(zero)} ({zero.population.sum():,} people in 2011, '
       f'{100 * zero.population.sum() / wards.population.sum():.0f}%); '
       f'{int((zero.vulnerability > wards.vulnerability.median()).sum())} are in the more vulnerable half. '
       f'Median density {zero.density.median():,.0f} vs {wards[wards.parks > 0].density.median():,.0f}; '
       f'median LST {zero.lst.median():.2f} vs {wards[wards.parks > 0].lst.median():.2f} °C')

gap = (wards[wards.group == 'Priority gap']
       .sort_values('vulnerability', ascending=False)
       [['ward_num', 'ward_name', 'population', 'density', 'lst', 'parks', 'parks_per_km2', 'vulnerability']])
report(f'\nPriority-gap wards ({len(gap)}), most vulnerable first:')
report(gap.round(2).to_string(index=False))

# ---------- 6. Figures ----------
colors = {'Priority gap': '#c0392b', 'Well-targeted': '#1a9850',
          'Possibly inefficient': '#e9a23b', 'Low priority': '#4575b4'}

fig, ax = plt.subplots(figsize=(7, 7.8))
for name, col in colors.items():
    sub = wards[wards.group == name]
    polys = [r for rs in sub.rings for r in rs]
    ax.add_collection(PolyCollection(polys, facecolors=col, edgecolors='white', linewidths=0.4,
                                     label=f'{name} ({len(sub)} wards)'))
ax.autoscale(); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9, frameon=False)
ax.set_title('Heat vulnerability vs BBMP parks, by ward')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'ward_groups_map.png'), dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
for name, col in colors.items():
    sub = wards[wards.group == name]
    ax.scatter(sub.vulnerability, sub.parks_per_km2 + 0.05, s=np.clip(sub.population / 1500, 8, 120),
               c=col, alpha=0.8, edgecolors='white', linewidths=0.5, label=name)
ax.axvline(wards.vulnerability.median(), color='#888888', lw=1, ls='--')
ax.axhline(wards.parks_per_km2.median() + 0.05, color='#888888', lw=1, ls='--')
ax.set_yscale('log')
ax.set_xlabel('Heat vulnerability (½ heat rank + ½ density rank)')
ax.set_ylabel('BBMP parks per km² (+0.05, log scale)')
ax.set_title(f'Do parks follow vulnerability?  Spearman ρ = {main["spearman"]:+.2f}')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'ward_vulnerability_vs_parks.png'), dpi=200); plt.close()

# Simple view of the key result: wards hold similar populations (~40,000), so parks per ward
# is a fair comparison across temperature quarters.
wards['heat_q'] = pd.qcut(wards.lst, 4, labels=['Coolest\nquarter', '2nd', '3rd', 'Hottest\nquarter'])
hq = wards.groupby('heat_q', observed=True).agg(mean_parks=('parks', 'mean'), lst_lo=('lst', 'min'),
                                                lst_hi=('lst', 'max'), zero=('parks', lambda s: (s == 0).sum()),
                                                wards=('parks', 'size'), median_pop=('population', 'median'))
report('\nParks per ward by ward-temperature quarter:')
report(hq.round(2).to_string())
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(range(4), hq.mean_parks, color=['#4575b4', '#91bfdb', '#fc8d59', '#c0392b'], width=0.62)
for i, r in enumerate(hq.itertuples()):
    ax.text(i, r.mean_parks + 0.15, f'{r.mean_parks:.1f} parks\n{r.zero} of {r.wards} wards have none',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(range(4), [f'{q}\n{lo:.1f}–{hi:.1f} °C' for q, lo, hi in zip(hq.index, hq.lst_lo, hq.lst_hi)])
ax.set_ylabel('Average BBMP parks per ward')
ax.set_ylim(0, hq.mean_parks.max() * 1.35)
ax.set_title('Hotter wards have fewer parks')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'ward_parks_by_heat.png'), dpi=200); plt.close()

wards.drop(columns=['rings', 'heat_q']).to_csv(os.path.join(DATA, 'ward_scores.csv'), index=False)
with open(os.path.join(HERE, 'ward_results.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\nSaved figures to figures/ and numbers to rq3_parks_vs_heat/ward_results.txt')
