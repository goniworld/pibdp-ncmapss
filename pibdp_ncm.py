"""PI-BDP for N-CMAPSS (cycle level).

Extends the C-MAPSS PI-BDP code (gpa.py, physics_model.py) to continuous flight conditions:
  1) healthy-baseline regression y_j = f_j(W) fitted on the first n_h cycles of training units,
     per-cycle median residual / sigma_j  ->  Delta z_j(t)              (generalises eq. 1)
  2) GPA-structured GLS severities with an empirical fault-signature matrix   (eqs. 2-3)
  3) Bayesian exponential degradation (empirical Bayes, Gauss-Newton Laplace)  (eqs. 4-6)
  4) Weibull importance re-weighting, Weibull fitted with right censoring.
Units whose last cycle equals CAP_CYCLES (=100) are treated as right-censored (the simulation
stops at 100 cycles): used for the healthy baseline and the fleet prior, excluded from the
failure threshold and fault signatures, censored in the Weibull fit, not scored in CV.
Health parameters T are never used for fitting; only for validation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CAP_CYCLES = 100
TS = 100.0          # time scaling (same as C-MAPSS code)
HI_EOL = 4.0
QUANTS = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])


# ----------------------------------------------------------------------------- data
def load_dev(path):
    d = np.load(path, allow_pickle=False)
    A = d['A']; an = list(d['A_var'])
    meta = pd.DataFrame({'unit': A[:, an.index('unit')].astype(int),
                         'cycle': A[:, an.index('cycle')].astype(int),
                         'Fc': A[:, an.index('Fc')].astype(int)})
    out = dict(meta=meta, W=d['W'].astype(np.float64), X=d['X_s'].astype(np.float64),
               T=d['T'].astype(np.float64), Y=d['Y'].ravel().astype(np.float64),
               X_names=list(d['X_s_var']), T_names=list(d['T_var']))
    d.close()
    eol = meta.groupby('unit').cycle.max()
    out['eol'] = eol
    out['censored'] = (eol >= CAP_CYCLES)
    out['units'] = eol.index.values
    # per-cycle true health parameters (validation only)
    Tc = pd.DataFrame(out['T']).groupby([meta.unit.values, meta.cycle.values]).mean()
    Tc.index.names = ['unit', 'cycle']
    out['Tc'] = Tc
    return out


def cycle_median(R, meta):
    df = pd.DataFrame(R)
    g = df.groupby([meta.unit.values, meta.cycle.values]).median()
    g.index.names = ['unit', 'cycle']
    return g


# ----------------------------------------------------------------------------- 1) baseline
class Baseline:
    """Healthy-engine regression of each measured sensor on operating conditions W."""
    def __init__(self, kind='poly2', n_h=10):
        self.kind, self.n_h = kind, n_h

    def _healthy(self, meta, units):
        return (meta.unit.isin(units) & (meta.cycle <= self.n_h)).values

    def fit(self, D, units):
        m = self._healthy(D['meta'], units)
        W, X = D['W'][m], D['X'][m]
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12
        Xn = (X - self.xm) / self.xs
        if self.kind == 'hgb':
            self.models = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
                           .fit(W, Xn[:, j]) for j in range(X.shape[1])]
        else:
            deg = {'poly2': 2, 'poly3': 3}[self.kind]
            self.model = make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=1e-3)).fit(W, Xn)
        Rc = cycle_median(self.resid(W, X), D['meta'][m])
        self.sig = Rc.std(0).values + 1e-12
        return self

    def resid(self, W, X):
        Xn = (X - self.xm) / self.xs
        P = np.column_stack([mdl.predict(W) for mdl in self.models]) if self.kind == 'hgb' else self.model.predict(W)
        return Xn - P

    def deltas(self, D):
        """Delta z for every (unit, cycle) of the data set: cycle-median residual / sigma."""
        Rc = cycle_median(self.resid(D['W'], D['X']), D['meta'])
        return Rc / self.sig


# ----------------------------------------------------------------------------- 2) GPA-GLS HI
class GPAGLS:
    def __init__(self, k=1, n_h=10, n_e=5, c_min=5.0, min_keep=4):
        self.k, self.n_h, self.n_e, self.c_min, self.min_keep = k, n_h, n_e, c_min, min_keep

    def fit(self, Z, units, eol, censored):
        u = Z.index.get_level_values('unit').values; c = Z.index.get_level_values('cycle').values
        tr = np.isin(u, units)
        h = tr & (c <= self.n_h)
        Zv = Z.values
        self.R = np.cov(Zv[h].T) + 1e-6 * np.eye(Zv.shape[1])
        fail_units = [x for x in units if not censored[x]]
        S = np.array([Zv[(u == x) & (c > eol[x] - self.n_e)].mean(0) for x in fail_units])
        if self.k == 1 or len(S) < 2 * self.k:
            H = S.mean(0, keepdims=True).T; lab = np.zeros(len(S), int); self.k_eff = 1
        else:
            Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
            lab = KMeans(self.k, n_init=20, random_state=0).fit_predict(Sn)
            H = np.stack([S[lab == i].mean(0) for i in range(self.k)], 1); self.k_eff = self.k
        cons = np.min([np.abs(S[lab == i].mean(0)) / (S[lab == i].std(0) + 1e-9) for i in range(self.k_eff)], 0)
        keep = cons >= self.c_min
        if keep.sum() < max(self.min_keep, self.k_eff + 1):
            keep = np.zeros_like(keep); keep[np.argsort(-cons)[:max(self.min_keep, self.k_eff + 1)]] = True
        self.keep, self.cons, self.labels, self.fail_units = keep, cons, lab, fail_units
        Hk = H[keep]; Rki = np.linalg.inv(self.R[np.ix_(keep, keep)])
        self.H = Hk
        self.G = np.linalg.solve(Hk.T @ Rki @ Hk + 1e-9 * np.eye(Hk.shape[1]), Hk.T @ Rki)
        return self

    def severities(self, Zv):
        return Zv[:, self.keep] @ self.G.T

    def hi(self, Zv):
        return HI_EOL * self.severities(Zv).sum(1)


# ----------------------------------------------------------------------------- 3) Bayesian exponential model
def model(th, t):
    a, lb, lc = th
    return a + np.exp(lb) * np.exp(np.exp(lc) * t)


class BayesExp:
    def __init__(self, shrink=0.0):
        self.shrink = shrink

    def fit(self, trajs, failed):
        """trajs: list of (t, y); failed: bool list (threshold D only from failed units)."""
        self.fit_units(trajs, failed)
        return self.aggregate(np.arange(len(trajs)), inplace=True)

    def fit_units(self, trajs, failed):
        """Per-unit nonlinear least-squares fits (done once; bootstrap re-aggregates them)."""
        P, R, D = [], [], []
        for (t, y), f in zip(trajs, failed):
            ts = t / TS
            r = least_squares(lambda th: model(th, ts) - y, x0=[0.0, np.log(0.3), np.log(1.0)],
                              bounds=([-5, -12, -4], [5, 3, 4]))
            P.append(r.x); R.append(r.fun); D.append(model(r.x, ts[-1]) if f else np.nan)
        self.P_u, self.R_u, self.D_u, self.failed_u = np.array(P), R, np.array(D), np.asarray(failed, bool)
        return self

    def aggregate(self, idx, inplace=False):
        """Fleet prior, noise and threshold from the units idx (indices may repeat: bootstrap)."""
        o = self if inplace else BayesExp(self.shrink)
        P = self.P_u[idx]
        o.mu = np.median(P, 0)
        S = np.cov(P.T)
        S = (1 - self.shrink) * S + self.shrink * np.diag(np.diag(S))
        o.S = S + 1e-6 * np.eye(3)
        o.Linv = np.linalg.cholesky(np.linalg.inv(o.S)).T
        o.sig = np.std(np.concatenate([self.R_u[i] for i in idx]))
        D = self.D_u[idx]; D = D[~np.isnan(D)]
        o.D_mu, o.D_sd = np.mean(D), np.std(D) + 1e-9
        return o

    def rul_samples(self, t, y, n, rng):
        ts = t / TS
        fun = lambda th: np.concatenate([(model(th, ts) - y) / self.sig, self.Linv @ (th - self.mu)])
        r = least_squares(fun, x0=self.mu)
        cov = np.linalg.pinv(r.jac.T @ r.jac)
        Th = rng.multivariate_normal(r.x, cov, n, method='svd')
        D = rng.normal(self.D_mu, self.D_sd, n)
        a, b, c = Th[:, 0], np.exp(Th[:, 1]), np.exp(Th[:, 2])
        arg = (D - a) / b
        tc = np.where(arg > 1, np.log(np.maximum(arg, 1)) / c, 0.0) * TS
        return np.clip(tc - t[-1], 0, 1000)


class WeibullCens:
    """2-parameter Weibull, MLE with right censoring."""
    def fit(self, L, cens):
        L = np.asarray(L, float); cens = np.asarray(cens, bool)

        def nll(p):
            k, lam = np.exp(p); z = L / lam
            lf = np.log(k / lam) + (k - 1) * np.log(z) - z ** k
            return -(lf[~cens].sum() - (z[cens] ** k).sum())
        r = minimize(nll, x0=[np.log(3.0), np.log(L.mean())], method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 4000})
        self.k, self.lam = np.exp(r.x)
        return self

    def logpdf(self, x):
        x = np.maximum(x, 1e-9); k, lam = self.k, self.lam
        return np.log(k / lam) + (k - 1) * np.log(x / lam) - (x / lam) ** k


def wquantile(x, w, q):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w); c /= c[-1]
    return np.interp(q, c, x)


def wcrps(x, w, y):
    """CRPS of a weighted sample: E|X-y| - 0.5 E|X-X'|."""
    i = np.argsort(x); x, w = x[i], w[i] / w.sum()
    C = np.cumsum(w) - w
    e1 = np.sum(w * np.abs(x - y))
    e2 = 2 * np.sum(w * x * (2 * C + w - 1))
    return e1 - 0.5 * e2


# ----------------------------------------------------------------------------- full pipeline
class DHI:
    """Ablation HI: ridge regression of 4 x life ratio (t / EOL) on Delta z (failed training units)."""
    k_eff = 0

    def fit(self, Z, units, eol, censored):
        u = Z.index.get_level_values('unit').values; c = Z.index.get_level_values('cycle').values
        m = np.isin(u, [x for x in units if not censored[x]])
        y = HI_EOL * c[m] / eol.reindex(u[m]).values
        self.reg = Ridge(alpha=1.0).fit(Z.values[m], y)
        return self

    def hi(self, Zv):
        return self.reg.predict(Zv)


class PIBDP:
    def __init__(self, baseline='poly2', n_h=10, k=1, shrink=0.0, fuse=True, n_e=5, c_min=5.0, boot=0, boot_seed=12345,
                 hi='gpa'):
        self.cfg = dict(baseline=baseline, n_h=n_h, k=k, shrink=shrink, fuse=fuse, n_e=n_e, c_min=c_min,
                        boot=boot, boot_seed=boot_seed, hi=hi)

    def fit(self, D, units, Z=None, base=None):
        c = self.cfg
        self.base = base or Baseline(c['baseline'], c['n_h']).fit(D, units)
        self.Z = Z if Z is not None else self.base.deltas(D)
        eol, cens = D['eol'], D['censored']
        if c['hi'] == 'dhi':
            self.gpa = DHI().fit(self.Z, units, eol, cens)
        else:
            self.gpa = GPAGLS(c['k'], c['n_h'], c['n_e'], c['c_min']).fit(self.Z, units, eol, cens)
        trajs, failed = [], []
        for x in units:
            Zu = self.Z.loc[x]
            trajs.append((Zu.index.values.astype(float), self.gpa.hi(Zu.values))); failed.append(not cens[x])
        self.deg = BayesExp(c['shrink']).fit(trajs, failed)
        L, C = eol.loc[units].values, cens.loc[units].values
        self.wb = WeibullCens().fit(L, C)
        # bootstrap fleet: propagate uncertainty of the fleet prior, noise, threshold and Weibull
        self.fleets = []
        if c['boot']:
            rng = np.random.default_rng(c['boot_seed']); fl = np.asarray(failed, bool); nu = len(units)
            while len(self.fleets) < c['boot']:
                idx = rng.integers(0, nu, nu)
                if fl[idx].sum() < 2 or len(np.unique(idx)) < 3: continue
                self.fleets.append((self.deg.aggregate(idx), WeibullCens().fit(L[idx], C[idx])))
        return self

    def use_data(self, Dnew):
        """Switch to new (e.g. test) data: Delta z from the already-fitted baseline. Nothing is refitted."""
        self.Z = self.base.deltas(Dnew)
        return self

    def hi_unit(self, x):
        Zu = self.Z.loc[x]
        return Zu.index.values.astype(float), self.gpa.hi(Zu.values)

    def predict_unit(self, x, n=2000, seed=0, t_from=1):
        """Predict at every cycle t >= t_from using HI(1..t). Returns DataFrame per cycle."""
        t_all, y_all = self.hi_unit(x)
        rows = []
        for j in range(len(t_all)):
            if t_all[j] < t_from: continue
            rng = np.random.default_rng(seed + j)
            fleets = self.fleets or [(self.deg, self.wb)]
            nb = max(1, n // len(fleets)); rs, ws = [], []
            for deg, wb in fleets:
                r = deg.rul_samples(t_all[:j + 1], y_all[:j + 1], nb, rng)
                w = np.ones_like(r)
                if self.cfg['fuse']:
                    lw = wb.logpdf(t_all[j] + r); w = np.exp(lw - lw.max())
                rs.append(r); ws.append(w / w.sum())   # each fleet gets equal mass
            r, w = np.concatenate(rs), np.concatenate(ws)
            q = wquantile(r, w, QUANTS)
            rows.append(dict(unit=x, cycle=int(t_all[j]), HI=y_all[j],
                             **{f'q{int(round(p * 1000)):03d}': v for p, v in zip(QUANTS, q)},
                             _r=r, _w=w))
        return rows


def score_rows(rows, eol):
    """Point/probabilistic metrics for prediction rows of non-censored units."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows])
    df['true'] = df.unit.map(eol) - df.cycle
    df['crps'] = [wcrps(r['_r'], r['_w'], t) for r, t in zip(rows, df['true'].values)]
    e = df.q500 - df.true
    df['err'] = e
    df['nasa'] = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)
    for lo, hi, name in [(250, 750, 'cov50'), (100, 900, 'cov80'), (50, 950, 'cov90'), (25, 975, 'cov95')]:
        df[name] = ((df.true >= df[f'q{lo:03d}']) & (df.true <= df[f'q{hi:03d}'])).astype(float)
    df['width90'] = df.q950 - df.q050
    return df


def summarize(df):
    return dict(n=len(df), rmse=float(np.sqrt(np.mean(df.err ** 2))), nasa_mean=float(df.nasa.mean()),
                phm21=float(0.5 * np.sqrt(np.mean(df.err ** 2)) + 0.5 * df.nasa.mean()),
                crps=float(df.crps.mean()), cov50=float(df.cov50.mean()), cov80=float(df.cov80.mean()),
                cov90=float(df.cov90.mean()), cov95=float(df.cov95.mean()), width90=float(df.width90.mean()))


def damage_true(Tc_u, tol=1e-4):
    """Normalised true damage d(t) from health parameters that change for this unit (validation only)."""
    T = Tc_u.values
    dT = T[-1] - T[0]
    P = np.abs(dT) > tol
    if not P.any(): return None
    return np.mean(np.abs(T[:, P] - T[0, P]) / np.abs(dT[P]), axis=1)


def hi_validation(model_, D, x):
    t, hi = model_.hi_unit(x)
    d = damage_true(D['Tc'].loc[x])
    out = dict(unit=x, censored=bool(D['censored'][x]))
    if d is not None and len(d) == len(hi):
        out['spearman_hi_damage'] = float(spearmanr(hi, d).correlation)
    out['hi_eol'] = float(hi[-1]); out['hi_start'] = float(np.mean(hi[:5]))
    if model_.gpa.k_eff >= 1:
        sev = model_.gpa.severities(model_.Z.loc[x].values)
        Tc = D['Tc'].loc[x].values
        dT = Tc - Tc[0]
        for m in range(sev.shape[1]):
            for j, name in enumerate(D['T_names']):
                if np.abs(dT[:, j]).max() > 1e-4:
                    out[f'rho_sev{m + 1}_{name}'] = float(spearmanr(sev[:, m], -dT[:, j]).correlation)
    return out


# ----------------------------------------------------------------------------- Phase 1 runners (dev only)
def stage_a(D, kinds=('poly2', 'poly3', 'hgb'), n_h=10, n_eval=5):
    """Leave-one-unit-out: fit baseline on other units' healthy cycles, score the held-out
    unit's first n_eval cycles (cycle-median residuals in sensor-SD units)."""
    rows = []
    for kind in kinds:
        for x in D['units']:
            tr = [u for u in D['units'] if u != x]
            b = Baseline(kind, n_h).fit(D, tr)
            m = ((D['meta'].unit == x) & (D['meta'].cycle <= n_eval)).values
            Rc = cycle_median(b.resid(D['W'][m], D['X'][m]), D['meta'][m])
            rows.append(dict(kind=kind, unit=int(x), mse=float(np.mean(Rc.values ** 2)),
                             mse_centered=float(np.mean((Rc.values - Rc.values.mean(0)) ** 2))))
    return pd.DataFrame(rows)


def stage_b(D, baseline, grid, n=2000, log=print):
    """Leave-one-unit-out CV of PI-BDP over the design grid. Censored units are trained on but not scored."""
    unit_rows, val_rows, pred_frames = [], [], []
    for x in D['units']:
        tr = [u for u in D['units'] if u != x]
        cache = {}
        for g in grid:
            n_h = g['n_h']
            if n_h not in cache:
                b = Baseline(baseline, n_h).fit(D, tr); cache[n_h] = (b, b.deltas(D))
            b, Z = cache[n_h]
            m = PIBDP(baseline=baseline, **g).fit(D, tr, Z=Z, base=b)
            v = hi_validation(m, D, x); v.update(g); val_rows.append(v)
            if D['censored'][x]: continue
            df = score_rows(m.predict_unit(x, n=n), D['eol'])
            s = summarize(df); s.update(unit=int(x), k_eff=m.gpa.k_eff, **g); unit_rows.append(s)
            df = df.assign(**{k: v for k, v in g.items()})
            pred_frames.append(df)
        log(f'  unit {x} done')
    U = pd.DataFrame(unit_rows); V = pd.DataFrame(val_rows)
    P = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return U, V, P


def _louo_fold(D, x, cfg, n):
    tr = [u for u in D['units'] if u != x]
    m = PIBDP(**cfg).fit(D, tr)
    return score_rows(m.predict_unit(x, n=n), D['eol'])


def stage_single(D, cfg, n=2000, n_jobs=-1):
    """LOUO-CV of one configuration, parallel over held-out (non-censored) units."""
    from joblib import Parallel, delayed
    units = [x for x in D['units'] if not D['censored'][x]]
    dfs = Parallel(n_jobs=n_jobs)(delayed(_louo_fold)(D, x, cfg, n) for x in units)
    return pd.concat(dfs, ignore_index=True)


def pooled(P, keys):
    out = []
    for key, g in P.groupby(keys):
        s = summarize(g); s.update(dict(zip(keys, key if isinstance(key, tuple) else (key,)))); out.append(s)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Phase 2 helpers (comparison models)
def build_folds(D, baseline, n_h):
    """For every non-censored held-out unit x: Delta z of all units from a baseline fitted on the other units.
    Returns {x: DataFrame(index=(unit, cycle), columns=sensors)}. Same preprocessing as PI-BDP."""
    out = {}
    for x in D['units']:
        if D['censored'][x]: continue
        tr = [u for u in D['units'] if u != x]
        out[int(x)] = Baseline(baseline, n_h).fit(D, tr).deltas(D).astype(np.float32)
    return out


def window_features(Zu, t, W=10):
    """Per-cycle features from the history up to each cycle: last-5 mean, last-W mean, last-W slope, cycle/100."""
    n, p = Zu.shape; F = np.empty((n, 3 * p + 1), np.float32)
    cs = np.vstack([np.zeros((1, p)), np.cumsum(Zu, 0)])
    for j in range(n):
        a5, aW = max(0, j - 4), max(0, j - W + 1)
        F[j, :p] = (cs[j + 1] - cs[a5]) / (j + 1 - a5)
        F[j, p:2 * p] = (cs[j + 1] - cs[aW]) / (j + 1 - aW)
        seg = Zu[aW:j + 1]
        if len(seg) >= 3:
            tt = np.arange(len(seg)) - (len(seg) - 1) / 2
            F[j, 2 * p:3 * p] = (tt @ (seg - seg.mean(0))) / (tt @ tt)
        else:
            F[j, 2 * p:3 * p] = 0
        F[j, -1] = t[j] / 100.0
    return F


def fold_xy(Z, units, eol, W=10):
    """Stack features and RUL labels (uncapped) for the given units."""
    X, y, idx = [], [], []
    for u in units:
        Zu = Z.loc[u]; t = Zu.index.values.astype(float)
        X.append(window_features(Zu.values.astype(np.float64), t, W)); y.append(eol[u] - t)
        idx.append(pd.DataFrame({'unit': u, 'cycle': t.astype(int)}))
    return np.vstack(X), np.concatenate(y), pd.concat(idx, ignore_index=True)


def point_scores(pred, true):
    e = pred - true
    nasa = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(n=len(e), rmse=rmse, nasa_mean=float(nasa.mean()), phm21=0.5 * rmse + 0.5 * float(nasa.mean()))


# ----------------------------------------------------------------------------- Phase 3 helpers (test evaluation)
CAL_LEVELS = {50: ('q250', 'q750'), 80: ('q100', 'q900'), 90: ('q050', 'q950'), 95: ('q025', 'q975')}
CAL_EPS = 0.5


def interval_metrics(df, c=None):
    """Coverage / width / 90% interval score, raw (c=None) or conformal-scaled with c = {level: factor}."""
    y, m = df.true.values, df.q500.values; out = {}
    for a, (lo, hi) in CAL_LEVELS.items():
        L, U = df[lo].values, df[hi].values
        if c is not None:
            ca = c[a] if a in c else c[str(a)]
            L = np.maximum(m - ca * np.maximum(m - L, CAL_EPS), 0); U = m + ca * np.maximum(U - m, CAL_EPS)
        out[f'cov{a}'] = float(((y >= L) & (y <= U)).mean()); out[f'width{a}'] = float(np.mean(U - L))
        if a == 90:
            out['iscore90'] = float(np.mean((U - L) + 20 * (L - y) * (y < L) + 20 * (y - U) * (y > U)))
    return out


def clip_physical(pred, cycle, cap=CAP_CYCLES):
    """Secondary post-processing: RUL can never exceed cap - t in N-CMAPSS (simulation stops at 100 cycles)."""
    return np.clip(pred, 0, cap - np.asarray(cycle))


def alpha_lambda(df, alpha=0.2, lambdas=(0.25, 0.5, 0.75)):
    """Fraction of units whose prediction at t = lambda * EOL is within +-alpha * true RUL."""
    out = {}
    for lam in lambdas:
        ok = []
        for _, g in df.groupby(['subset', 'unit']):
            eol = int(g.cycle.max() + g.true[g.cycle == g.cycle.max()].iloc[0])
            t = max(1, int(round(lam * eol))); r = g[g.cycle == t]
            if len(r): ok.append(abs(r.pred.iloc[0] - r.true.iloc[0]) <= alpha * r.true.iloc[0])
        out[f'alpha_lambda_{lam}'] = float(np.mean(ok)) if ok else np.nan
    return out


def prognostic_horizon(df, alpha=0.1):
    """Mean PH / EOL: PH = EOL - first cycle after which all predictions stay within true +- alpha * EOL."""
    ph = []
    for _, g in df.groupby(['subset', 'unit']):
        g = g.sort_values('cycle'); eol = int(g.cycle.iloc[-1] + g.true.iloc[-1])
        inside = (np.abs(g.pred.values - g.true.values) <= alpha * eol)
        bad = np.flatnonzero(~inside)
        start = g.cycle.values[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(g) else (g.cycle.values[0] if not len(bad) else None)
        ph.append(0.0 if start is None else (eol - start) / eol)
    return {'ph_frac_mean': float(np.mean(ph))}


def life_bins(df, pred='pred'):
    f = df.cycle / (df.cycle + df.true)
    b = pd.cut(f, [0, .25, .5, .75, 1.0001], labels=['0-25%', '25-50%', '50-75%', '75-100%'], include_lowest=True)
    return df.assign(bin=b).groupby('bin', observed=True).apply(lambda g: np.sqrt(np.mean((g[pred] - g.true) ** 2)))


def hier_boot_rmse_diff(A, Bm, n_boot=100000, seed=0):
    """Paired hierarchical bootstrap (subset -> unit) of pooled RMSE(A) - RMSE(B).
    A, B: DataFrames with subset, unit, cycle, pred, true on identical rows."""
    k = ['subset', 'unit', 'cycle']
    M = A[k + ['pred', 'true']].merge(Bm[k + ['pred']], on=k, suffixes=('_a', '_b'))
    assert len(M) == len(A) == len(Bm), 'rows differ'
    M['ea'] = (M.pred_a - M.true) ** 2; M['eb'] = (M.pred_b - M.true) ** 2
    U = M.groupby(['subset', 'unit']).agg(sa=('ea', 'sum'), sb=('eb', 'sum'), n=('ea', 'size')).reset_index()
    subsets = sorted(U.subset.unique()); rng = np.random.default_rng(seed)
    per = [U[U.subset == s][['sa', 'sb', 'n']].values for s in subsets]
    SA = np.zeros(n_boot); SB = np.zeros(n_boot); N = np.zeros(n_boot)
    pick = rng.integers(0, len(subsets), (n_boot, len(subsets)))
    for j in range(len(subsets)):
        for si, arr in enumerate(per):
            rows = np.flatnonzero(pick[:, j] == si)
            if not len(rows): continue
            idx = rng.integers(0, len(arr), (len(rows), len(arr)))
            tot = arr[idx].sum(1)
            SA[rows] += tot[:, 0]; SB[rows] += tot[:, 1]; N[rows] += tot[:, 2]
    d = np.sqrt(SA / N) - np.sqrt(SB / N)
    obs = float(np.sqrt(U.sa.sum() / U.n.sum()) - np.sqrt(U.sb.sum() / U.n.sum()))
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(diff=obs, ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), p_boot=p)


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj
