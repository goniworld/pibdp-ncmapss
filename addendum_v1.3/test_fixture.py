"""Code check for addendum v1.3 on synthetic data only (no real N-CMAPSS data or predictions are used).

  python test_fixture.py <empty_dir>

1) builds fake N-CMAPSS-like engines and a mini Phase 3 with the frozen library (fake V0/GB predictions),
2) compares the notebook's ER rule with an independent loop implementation on 600 random inputs,
3) runs every code cell of the v1.3 notebook on the fake data (NCM_TEST=1, NCM_ROOT=<empty_dir>).
"""
import os, sys, json, pickle, hashlib, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
ROOT = os.path.abspath(sys.argv[1]); LIB = f'{REPO}/freeze_v1/pibdp_ncm.py'
NB = f'{HERE}/NCMAPSS_v13_정비규칙_계산비용.ipynb'
for d in ['phase0', 'phase1', 'phase2', 'phase3', 'freeze_v1']: os.makedirs(f'{ROOT}/{d}', exist_ok=True)
shutil.copy(LIB, f'{ROOT}/freeze_v1/pibdp_ncm.py'); sys.path.insert(0, f'{ROOT}/freeze_v1'); import pibdp_ncm as P
SUBS = ['DS01-005', 'DS04', 'DS07']
cfg = dict(baseline='poly3', n_h=10, shrink=0.5, k={s: 1 for s in SUBS})
json.dump(cfg, open(f'{ROOT}/phase1/config.json', 'w'))
json.dump(dict(c={'50': 1.5511199087332235, '80': 1.584242838316265, '90': 1.8424733212808164, '95': 2.0857639458801254}),
          open(f'{ROOT}/phase1/calibration.json', 'w'))
shutil.copy(f'{REPO}/freeze_v1/phase2__ml_selected.json', f'{ROOT}/phase2/ml_selected.json')
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
rng = np.random.default_rng(7); NS, NW, NT, ROWS = 14, 4, 10, 12
Wm = rng.normal(size=(NW, NS)); Sens = np.zeros((NT, NS)); Sens[0] = rng.normal(0, 1, NS); Sens[1] = rng.normal(0, 1, NS)
def engine(u, eol, cens):
    last = 100 if cens else eol
    A, W, X, T, Y = [], [], [], [], []
    for c in range(1, last + 1):
        w = rng.normal(size=(ROWS, NW)) * 0.5
        dmg = np.exp(3.0 * c / eol) - 1; th = np.zeros(NT); th[0] = -0.004 * dmg; th[1] = -0.003 * dmg
        x = w @ Wm + 0.2 * (w ** 2) @ Wm + 3.0 * (th @ Sens) + rng.normal(0, 0.01, (ROWS, NS))
        A.append(np.column_stack([np.full(ROWS, u), np.full(ROWS, c), np.full(ROWS, 1), np.zeros(ROWS)]))
        W.append(w); X.append(x); T.append(np.tile(th, (ROWS, 1))); Y.append(np.full(ROWS, eol - c))
    return [np.vstack(A), np.vstack(W), np.vstack(X), np.vstack(T), np.concatenate(Y)]
def save(path, units):
    parts = [engine(*u) for u in units]
    A, W, X, T, Y = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    np.savez(path, A=A, A_var=np.array(['unit', 'cycle', 'Fc', 'hs']), W=W, W_var=np.array(['alt', 'Mach', 'TRA', 'T2']),
             X_s=X, X_s_var=np.array([f's{j}' for j in range(NS)]), T=T, T_var=np.array([f't{j}' for j in range(NT)]), Y=Y[:, None])
rec, gb_test, gb_dev = {}, [], []
SEL = json.load(open(f'{ROOT}/phase2/ml_selected.json'))
from sklearn.ensemble import HistGradientBoostingRegressor
for s in SUBS:
    dev = [(u, int(rng.integers(55, 95)), False) for u in range(1, 8)] + [(8, 130, True)]
    test = [(u, int(rng.integers(55, 95)), False) for u in range(9, 13)]
    save(f'{ROOT}/phase0/dev_{s}.npz', dev); save(f'{ROOT}/phase0/sealed_test_{s}.npz', test)
    rec[f'sealed_test_{s}.npz'] = sha(f'{ROOT}/phase0/sealed_test_{s}.npz')
    Dd = P.load_dev(f'{ROOT}/phase0/dev_{s}.npz'); Dt = P.load_dev(f'{ROOT}/phase0/sealed_test_{s}.npz')
    dev_units = list(Dd['units']); test_units = [u for u in Dt['units'] if not Dt['censored'][u]]
    m = P.PIBDP(baseline=cfg['baseline'], n_h=cfg['n_h'], k=1, shrink=cfg['shrink']).fit(Dd, dev_units)
    Zdev = m.Z.copy(); m.use_data(Dt)
    pickle.dump(dict(Zdev=Zdev, Zte=m.Z.copy(), eol_dev=Dd['eol'], cens_dev=Dd['censored'], eol_te=Dt['eol'],
                     dev_units=[int(u) for u in dev_units], test_units=[int(u) for u in test_units]), open(f'{ROOT}/phase3/deltas_{s}.pkl', 'wb'))
    rows = [r for x in test_units for r in m.predict_unit(x, n=4000)]
    P.score_rows(rows, Dt['eol']).assign(subset=s, model='PI-BDP_V0').to_csv(f'{ROOT}/phase3/test_PI-BDP_V0_{s}.csv.gz', index=False)
    tr = [u for u in dev_units if not Dd['censored'][u]]
    Xtr, ytr, itr = P.fold_xy(Zdev, tr, Dd['eol'], 10); Xte, yte, ite = P.fold_xy(m.Z, test_units, Dt['eol'], 10)
    for sd in range(2):
        g = HistGradientBoostingRegressor(random_state=sd, early_stopping=False, **SEL['GB']['params']).fit(Xtr, ytr)
        gb_test.append(ite.assign(subset=s, seed=sd, pred=g.predict(Xte), true=yte))
        gb_dev.append(itr.assign(subset=s, seed=sd, pred=g.predict(Xtr) + rng.normal(0, 6, len(ytr)), true=ytr))
    print(s, 'test cycles', len(rows), 'dev rows', len(ytr))
json.dump(rec, open(f'{ROOT}/phase0/sha256_phase0.json', 'w'))
pd.concat(gb_test).assign(model='GB').to_csv(f'{ROOT}/phase3/test_GB.csv.gz', index=False)
pd.concat(gb_dev).assign(model='GB').to_csv(f'{ROOT}/phase2/preds_GB.csv.gz', index=False)
print('fixture ok')


# ---- 2) ER rule: notebook cell vs independent loop implementation
import math
nb = json.load(open(NB)); cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
exec([c for c in cells if c.startswith('# === M1 규칙')][0])

P_ = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
def F_ref(qv, r):
    pts = []
    for x, p in zip(qv, P_):
        if pts and x <= pts[-1][0] + 1e-9: pts[-1] = (pts[-1][0], max(pts[-1][1], p))
        else: pts.append((x, p))
    d_lo = (P_[1] - P_[0]) / max(qv[1] - qv[0], 1e-6); d_hi = (P_[-1] - P_[-2]) / max(qv[-1] - qv[-2], 1e-6)
    x0 = pts[0][0] - pts[0][1] / d_lo
    if x0 >= 0: pts = [(x0, 0.0)] + pts
    elif pts[0][0] > 0: pts = [(0.0, pts[0][1] - d_lo * pts[0][0])] + pts
    pts = pts + [(pts[-1][0] + (1 - pts[-1][1]) / d_hi, 1.0)]
    if r < pts[0][0]: return 0.0
    if r >= pts[-1][0]: return 1.0
    for (xa, pa), (xb, pb) in zip(pts[:-1], pts[1:]):
        if xa <= r <= xb: return pa + (pb - pa) * (r - xa) / (xb - xa) if xb > xa else pb
    return 1.0
def order_ref(qv, t, rho, L=5, Dp=2.0):
    up = F_upper = None
    # upper end of support
    xs, _ = knots(qv); R = max(L, math.ceil(xs[-1]))
    best, best_r = None, None
    for r in range(L, R + 1):
        used = t + sum(1 - F_ref(qv, s) for s in range(0, r))
        Fr = F_ref(qv, r); g = Dp * ((1 - Fr) + rho * Fr) / used
        if best is None or g < best - 1e-12: best, best_r = g, r
    return best_r == L
rng = np.random.default_rng(3); n_bad = 0; n = 0; orders = 0
for i in range(600):
    med = rng.uniform(0, 80); sd = rng.uniform(0.5, 30)
    z = np.array([-1.96, -1.645, -1.2816, -0.6745, 0, 0.6745, 1.2816, 1.645, 1.96])
    qv = np.maximum(med + sd * z * rng.uniform(0.6, 1.4), 0); qv = np.maximum.accumulate(qv)
    if rng.random() < 0.2: qv[:3] = 0; qv = np.maximum.accumulate(qv)
    t = int(rng.integers(1, 90)); rho = [2, 5, 10, 20, 50][i % 5]
    a = order_now(qv, t, rho); b = order_ref(qv, t, rho); n += 1; orders += a
    if a != b: n_bad += 1; print('MISMATCH', qv.round(2), t, rho, a, b)
print(f'{n} cases, {n_bad} mismatches, order-now share {orders / n:.2f}')

assert n_bad == 0
# ---- 3) run all notebook code cells on the fake data
import subprocess, tempfile
script = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False); script.write('\n\n'.join(cells)); script.close()
r = subprocess.run([sys.executable, '-I', script.name], env=dict(os.environ, NCM_TEST='1', NCM_ROOT=ROOT), capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-3000:]); assert r.returncode == 0
print('notebook ran on synthetic data: OK')
