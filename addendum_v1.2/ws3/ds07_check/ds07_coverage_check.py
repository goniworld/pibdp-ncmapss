"""Descriptive check of DS07 calibrated 90% coverage (Supplementary Note S6). Usage: python ds07_coverage_check.py test_PI-BDP_V0_DS07.csv.gz"""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); c, eps = 1.8424733212808164, 0.5
m, lo, hi, y = d.q500, d.q050, d.q950, d.true
def cover(k):
    U = m + k * np.maximum(hi - m, eps); L = np.maximum(m - k * np.maximum(m - lo, eps), 0)
    return (y >= L) & (y <= U), U
cov, U = cover(c)
print('coverage', round(cov.mean(), 4), '| misses above', int((y > U).sum()), 'below', int((y < np.maximum(m - c * np.maximum(m - lo, eps), 0)).sum()))
print(d.assign(cov=cov, bias=m - y).groupby('unit').agg(n=('cov', 'size'), coverage=('cov', 'mean'), mean_median_minus_true=('bias', 'mean')).round(3))
print('distance of misses above upper bound: median', round(float((y - U)[y > U].median()), 2))
for k in [1.38, 1.6, c, 2.0, 2.46, 3, 4, 5]: print('constant', round(k, 2), 'coverage', round(cover(k)[0].mean(), 3))
need = np.where(y > m, (y - m) / np.maximum(hi - m, eps), (m - y) / np.maximum(m - lo, eps)); print('constant for 90%:', round(float(np.quantile(need, 0.9)), 2))
