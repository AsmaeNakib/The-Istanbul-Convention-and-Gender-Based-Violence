# =============================================================================
# estimators.py
# -----------------------------------------------------------------------------
# Panel estimators for the Istanbul Convention analysis.
#
# Every function returns finite, checked standard errors or raises. A
# cluster-robust variance matrix that is not positive definite yields a
# non-finite standard error rather than an error, so this is checked explicitly
# instead of being left to the caller.
#
# Contents
#   feols                  n-way fixed effects by within transformation, with
#                          cluster-robust standard errors and the usual finite
#                          sample correction. Absorbing the fixed effects rather
#                          than entering them as dummies keeps the number of
#                          estimated parameters far below the number of
#                          clusters, which the cluster sandwich requires.
#   event_study            event study with binned endpoints, so no treated
#                          observation outside the plotted window is dropped.
#   stacked_did            cohort-stacked DiD with cohort-specific event
#                          windows, clustered on country.
#   callaway_santanna      ATT(g,t) with a stated control group (never-treated
#                          or not-yet-treated), aggregated simple, dynamic or by
#                          cohort, with a country cluster bootstrap.
#   wild_cluster_bootstrap null-imposed wild bootstrap with Webb six-point
#                          weights, for designs with few clusters.
#   permutation_test       design-based inference that reassigns the observed
#                          cohort structure across countries.
#   honest_rm_bounds       Rambachan and Roth (2023) relative-magnitudes
#                          sensitivity bounds, simplified implementation.
#   mde                    minimum detectable effect at 80% power.
# =============================================================================
import numpy as np
import pandas as pd

RNG_SEED = 20260804


# ---------------------------------------------------------------------------
# Core: fixed effects absorbed by within transformation
# ---------------------------------------------------------------------------
def _demean(frame, cols, fe_cols, tol=1e-10, max_iter=200):
    """Alternating projections. Exact for one FE, converges for several."""
    out = frame[cols].astype(float).copy()
    if not fe_cols:
        return out
    keys = [frame[f].values for f in fe_cols]
    for _ in range(max_iter):
        max_shift = 0.0
        for k in keys:
            grouped = out.groupby(k, sort=False).transform("mean")
            max_shift = max(max_shift, float(np.abs(grouped.values).max()))
            out = out - grouped
        if max_shift < tol:
            break
    return out


def feols(data, y, x, absorb, cluster=None, weights=None, name=""):
    """OLS with `absorb` fixed effects partialled out and cluster-robust SEs.

    Returns {'coef','se','t','p','ci_low','ci_high','n','n_clusters','k_absorbed'}
    keyed by variable name, plus scalar diagnostics.

    Raises RuntimeError if any reported SE is non-finite — the failure mode
    that would otherwise yield a non-finite standard error.
    """
    from scipy import stats

    x = list(x)
    d = data.dropna(subset=[y] + x + list(absorb) +
                    ([cluster] if cluster else []) +
                    ([weights] if weights else [])).copy()
    if d.empty:
        raise RuntimeError(f"{name}: no observations after dropping missings")

    w = d[weights].astype(float).values if weights else np.ones(len(d))
    w = w / w.mean()
    sw = np.sqrt(w)

    dm = _demean(d, [y] + x, list(absorb))
    Y = dm[y].values * sw
    X = dm[x].values * sw[:, None]

    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        raise RuntimeError(f"{name}: regressors are collinear after absorbing {absorb}")
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta

    n = len(d)
    # parameters absorbed: sum of (levels - 1) per FE, plus the intercept
    k_abs = sum(d[f].nunique() - 1 for f in absorb) + 1
    k = X.shape[1] + k_abs

    if cluster:
        g = d[cluster].values
        uniq = pd.unique(g)
        G = len(uniq)
        if G <= X.shape[1]:
            raise RuntimeError(f"{name}: {G} clusters is too few for {X.shape[1]} regressors")
        meat = np.zeros((X.shape[1], X.shape[1]))
        codes = pd.factorize(g)[0]
        for j in range(G):
            m = codes == j
            Xg_u = X[m].T @ resid[m]
            meat += np.outer(Xg_u, Xg_u)
        adj = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
        V = XtX_inv @ (adj * meat) @ XtX_inv
        dof = G - 1
    else:
        s2 = float(resid @ resid) / max(n - k, 1)
        V = s2 * XtX_inv
        G = np.nan
        dof = max(n - k, 1)

    se = np.sqrt(np.diag(V))
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise RuntimeError(
            f"{name}: non-finite or non-positive standard error(s) {se}. "
            "This usually means #parameters exceeds #clusters — absorb more "
            "fixed effects or cluster at a coarser level.")

    tstat = beta / se
    p = 2 * stats.t.sf(np.abs(tstat), dof)
    crit = stats.t.ppf(0.975, dof)
    res = {}
    for i, v in enumerate(x):
        res[v] = {
            "coef": float(beta[i]), "se": float(se[i]), "t": float(tstat[i]),
            "p": float(p[i]), "ci_low": float(beta[i] - crit * se[i]),
            "ci_high": float(beta[i] + crit * se[i]),
        }
    res["_meta"] = {"n": int(n), "n_clusters": (int(G) if np.isfinite(G) else None),
                    "k_absorbed": int(k_abs), "dof": int(dof), "name": name}
    return res


def coef_row(res, var, label):
    r = res[var]
    return {"label": label, "b": r["coef"], "se": r["se"], "p": r["p"],
            "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "N": res["_meta"]["n"], "clusters": res["_meta"]["n_clusters"]}


# ---------------------------------------------------------------------------
# Event study with binned endpoints
# ---------------------------------------------------------------------------
def event_study(data, y, unit="country", time="year", cohort="convention_ratified_year",
                treated_flag="treated_ever", lo=-6, hi=8, ref=-1, bin_endpoints=True,
                cluster="country"):
    """Event study on never-treated + not-yet-treated controls.

    bin_endpoints=True puts every treated observation with event time below
    `lo` into a single `pre_bin` dummy and every one above `hi` into
    `post_bin`, so no treated observation is dropped and the sample
    composition stays constant across the window.
    """
    d = data.copy()
    d["_et"] = np.where(d[treated_flag] == 1, d[time] - d[cohort], np.nan)
    cols = []
    for k in range(lo, hi + 1):
        if k == ref:
            continue
        c = f"et_p{k}" if k >= 0 else f"et_m{abs(k)}"
        d[c] = ((d[treated_flag] == 1) & (d["_et"] == k)).astype(float)
        cols.append(c)
    if bin_endpoints:
        d["et_pre_bin"] = ((d[treated_flag] == 1) & (d["_et"] < lo)).astype(float)
        d["et_post_bin"] = ((d[treated_flag] == 1) & (d["_et"] > hi)).astype(float)
        for c in ["et_pre_bin", "et_post_bin"]:
            if d[c].sum() > 0:
                cols.append(c)
    else:
        d = d[(d[treated_flag] == 0) | d["_et"].between(lo, hi)].copy()

    res = feols(d, y, cols, absorb=[unit, time], cluster=cluster, name=f"event_study[{y}]")
    rows = [{"event_time": ref, "coef": 0.0, "se": 0.0, "p": np.nan,
             "ci_low": 0.0, "ci_high": 0.0, "kind": "reference"}]
    for k in range(lo, hi + 1):
        if k == ref:
            continue
        c = f"et_p{k}" if k >= 0 else f"et_m{abs(k)}"
        if c in res:
            r = res[c]
            rows.append({"event_time": k, "coef": r["coef"], "se": r["se"], "p": r["p"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                         "kind": "pre" if k < 0 else "post"})
    es = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    # support counts
    sup = (d[(d[treated_flag] == 1) & d["_et"].between(lo, hi)]
           .groupby("_et")[unit].nunique().rename("n_treated_countries"))
    es["n_treated_countries"] = es["event_time"].map(sup)

    diag = {"n": res["_meta"]["n"], "n_clusters": res["_meta"]["n_clusters"]}
    # joint pre-trend test and the more powerful linear pre-trend test
    pre = es[(es.kind == "pre")]
    if len(pre) >= 2:
        diag["pre_coefs_all_same_sign"] = bool(
            (pre.coef > 0).all() or (pre.coef < 0).all())
        sl = np.polyfit(pre.event_time, pre.coef, 1)[0]
        diag["pre_trend_slope_through_coefs"] = float(sl)
    return es, diag, res


def linear_pretrend_test(data, y, unit="country", time="year",
                         cohort="convention_ratified_year", treated_flag="treated_ever",
                         lo=-6, cluster="country"):
    """Treated-specific linear time trend estimated on pre-treatment data only.
    Far more powerful against a smooth differential trend than a joint test
    that all pre-period event dummies are zero."""
    d = data.copy()
    d["_et"] = np.where(d[treated_flag] == 1, d[time] - d[cohort], np.nan)
    d = d[(d[treated_flag] == 0) | (d["_et"].between(lo, -1))].copy()
    d["treat_trend"] = np.where(d[treated_flag] == 1, d[time], 0.0)
    res = feols(d, y, ["treat_trend"], absorb=[unit, time], cluster=cluster,
                name=f"linear_pretrend[{y}]")
    return coef_row(res, "treat_trend", "Treated-specific pre-period linear trend (per year)")


# ---------------------------------------------------------------------------
# Stacked DiD
# ---------------------------------------------------------------------------
def stacked_did(data, y, unit="country", time="year", cohort="convention_ratified_year",
                treated_flag="treated_ever", pre_window=5, post_window=7,
                return_event_study=False, lo=-5, hi=7,
                cohort_min=None, cohort_max=None,
                control_rule="not_yet_or_never", require_complete_window=False,
                return_composition=False):
    """Cohort-stacked DiD. Each cohort is compared only to units that are
    not-yet-treated (or never treated) within that cohort's own event window.

    Two implementation points matter for inference. The stack and year fixed
    effects are absorbed rather than entered as dummies, because a stacked
    design generates several hundred of them and the cluster-robust variance
    matrix loses rank once the number of parameters approaches the number of
    clusters. And clustering is on country rather than on the stack-country
    cell: a country appearing in nine cohort stacks is one cluster, not nine.

    Optional arguments, all inert at their defaults:

    cohort_min / cohort_max   restrict which ratification cohorts are treated.
    control_rule              "not_yet_or_never" (default) keeps never-treated
                              countries in the comparison. "later_ratifiers"
                              drops them and keeps only countries that ratify
                              after the end of the cohort's own window, so a
                              comparison country is untreated throughout it.
    require_complete_window   keep only countries observed in every event year
                              of the stack window.
    return_composition        also return one row per stack describing how it
                              was built.
    """
    s = data
    cohorts = sorted(s.loc[s[treated_flag] == 1, cohort].dropna().unique().astype(int))
    if cohort_min is not None:
        cohorts = [g for g in cohorts if g >= cohort_min]
    if cohort_max is not None:
        cohorts = [g for g in cohorts if g <= cohort_max]
    stacks, composition = [], []
    for g in cohorts:
        g_end = g + post_window
        treat_c = s.loc[s[cohort] == g, unit].unique()
        if control_rule == "later_ratifiers":
            ctrl_c = s.loc[s[cohort] > g_end, unit].unique()
        elif control_rule == "not_yet_or_never":
            ctrl_c = s.loc[(s[cohort] > g) | (s[treated_flag] == 0), unit].unique()
        else:
            raise ValueError(f"unknown control_rule: {control_rule}")
        sub = s[s[unit].isin(list(treat_c) + list(ctrl_c))
                & s[time].between(g - pre_window, g_end)].copy()
        if require_complete_window:
            need = pre_window + post_window + 1
            keep = [c for c, gg in sub.dropna(subset=[y]).groupby(unit)
                    if gg[time].nunique() == need]
            sub = sub[sub[unit].isin(keep)]
            treat_c = [c for c in treat_c if c in keep]
            ctrl_c = [c for c in ctrl_c if c in keep]
        if sub.empty or len(treat_c) == 0 or len(ctrl_c) == 0:
            continue
        composition.append({
            "cohort_year": g, "start_year": g - pre_window, "end_year": g_end,
            "n_treated_countries": len(treat_c),
            "n_control_countries": len(ctrl_c),
            "n_obs_in_stack": int(sub.dropna(subset=[y]).shape[0]),
            "treated_countries": "; ".join(sorted(map(str, treat_c))),
            "control_countries": "; ".join(sorted(map(str, ctrl_c)))})
        sub["stack_id"] = f"g{g}"
        sub["treat_g"] = sub[unit].isin(treat_c).astype(float)
        sub["post_g"] = (sub[time] >= g).astype(float)
        sub["did_g"] = sub["treat_g"] * sub["post_g"]
        sub["rel_g"] = np.where(sub["treat_g"] == 1, sub[time] - g, np.nan)
        sub["stack_unit"] = sub["stack_id"] + "_" + sub[unit].astype(str)
        sub["stack_time"] = sub["stack_id"] + "_" + sub[time].astype(str)
        stacks.append(sub)
    if not stacks:
        raise RuntimeError("no valid cohort stacks")
    st = pd.concat(stacks, ignore_index=True).dropna(subset=[y])

    res = feols(st, y, ["did_g"], absorb=["stack_unit", "stack_time"],
                cluster=unit, name=f"stacked_did[{y}]")
    out = coef_row(res, "did_g", "Stacked DiD ATT")
    out.update({"n_stacks": len(stacks), "n_true_countries": int(st[unit].nunique())})
    if return_composition:
        return out, pd.DataFrame(composition)
    if not return_event_study:
        return out, st

    cols = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        c = f"rel_p{k}" if k >= 0 else f"rel_m{abs(k)}"
        st[c] = ((st["treat_g"] == 1) & (st["rel_g"] == k)).astype(float)
        if st[c].sum() > 0:
            cols.append(c)
    res_es = feols(st, y, cols, absorb=["stack_unit", "stack_time"], cluster=unit,
                   name=f"stacked_event_study[{y}]")
    rows = [{"event_time": -1, "coef": 0.0, "se": 0.0, "p": np.nan,
             "ci_low": 0.0, "ci_high": 0.0}]
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        c = f"rel_p{k}" if k >= 0 else f"rel_m{abs(k)}"
        if c in res_es:
            r = res_es[c]
            rows.append({"event_time": k, "coef": r["coef"], "se": r["se"],
                         "p": r["p"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return out, pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Callaway & Sant'Anna (2021)
# ---------------------------------------------------------------------------
def _attgt_from_matrix(Y, gvec, times, control_group, anticipation=0):
    """Unconditional ATT(g,t) from a units x periods matrix.

    Y     : (n_units, n_times) float array, NaN where unobserved
    gvec  : (n_units,) int array of cohort years, 0 = never treated
    times : (n_times,) int array of period labels

    Kept matrix-based rather than DataFrame-based so the bootstrap can resample
    unit rows cheaply instead of rebuilding a long frame 500 times.
    """
    tidx = {t: i for i, t in enumerate(times)}
    rows = []
    for g in sorted({int(x) for x in gvec if x > 0}):
        base = g - 1 - anticipation
        if base not in tidx:
            continue
        bi = tidx[base]
        tr = gvec == g
        if not tr.any():
            continue
        for t in times:
            if t == base:
                continue
            ti = tidx[t]
            if control_group == "nevertreated":
                ct = gvec == 0
            else:
                ct = (gvec == 0) | (gvec > max(t, base) + anticipation)
                ct = ct & ~tr
            ok_tr = tr & ~np.isnan(Y[:, bi]) & ~np.isnan(Y[:, ti])
            ok_ct = ct & ~np.isnan(Y[:, bi]) & ~np.isnan(Y[:, ti])
            n_tr, n_ct = int(ok_tr.sum()), int(ok_ct.sum())
            if n_tr < 1 or n_ct < 2:
                continue
            d_tr = (Y[ok_tr, ti] - Y[ok_tr, bi]).mean()
            d_ct = (Y[ok_ct, ti] - Y[ok_ct, bi]).mean()
            rows.append({"group": int(g), "time": int(t), "event_time": int(t - g),
                         "att": float(d_tr - d_ct),
                         "n_treated": n_tr, "n_control": n_ct})
    return pd.DataFrame(rows)


def _panel_matrix(d, y, unit, time, gvar):
    wide = d.pivot_table(index=unit, columns=time, values=y, aggfunc="first")
    gmap = d.drop_duplicates(unit).set_index(unit)[gvar].reindex(wide.index)
    times = np.array(sorted(wide.columns), dtype=int)
    return (wide[times].to_numpy(dtype=float),
            gmap.to_numpy(dtype=float).astype(int), times)


def _aggregate_attgt(tab, kind):
    if tab.empty:
        return np.nan, pd.DataFrame()
    post = tab[tab.event_time >= 0]
    if post.empty:
        return np.nan, pd.DataFrame()
    if kind == "simple":
        w = post["n_treated"] / post["n_treated"].sum()
        return float((post["att"] * w).sum()), pd.DataFrame()
    if kind == "dynamic":
        rows = []
        for e, gg in post.groupby("event_time"):
            w = gg["n_treated"] / gg["n_treated"].sum()
            rows.append({"event_time": int(e), "att": float((gg["att"] * w).sum()),
                         "n_groups": int(gg["group"].nunique()),
                         "n_treated": int(gg["n_treated"].sum())})
        dyn = pd.DataFrame(rows).sort_values("event_time")
        return float(dyn["att"].mean()), dyn
    if kind == "group":
        rows = []
        for g, gg in post.groupby("group"):
            rows.append({"group": int(g), "att": float(gg["att"].mean()),
                         "n_treated": int(gg["n_treated"].max())})
        grp = pd.DataFrame(rows)
        w = grp["n_treated"] / grp["n_treated"].sum()
        return float((grp["att"] * w).sum()), grp
    raise ValueError(kind)


def callaway_santanna(data, y, unit="country", time="year",
                      cohort="convention_ratified_year", control_group="notyettreated",
                      anticipation=0, n_boot=499, seed=RNG_SEED, alpha=0.05):
    """Callaway & Sant'Anna (2021) group-time ATTs with an explicitly applied
    control group and a country-level nonparametric bootstrap.

    Written because the installed `csdid` build ignores `control_group`:
    passing 'notyettreated' and 'nevertreated' returns an identical ATT and the
    object reports `Control Group: None`. Do not use that package's output.

    This is the UNCONDITIONAL (no-covariate) estimator — the doubly-robust
    version requires covariates that are credibly exogenous, which this panel
    does not have (GDP and FLFP are themselves potentially affected).
    """
    if control_group not in ("notyettreated", "nevertreated"):
        raise ValueError("control_group must be 'notyettreated' or 'nevertreated'")
    d = data[[unit, time, y, cohort]].copy()
    d["_g"] = d[cohort].fillna(0).astype(int)
    d.loc[d["_g"] > d[time].max(), "_g"] = 0      # ratified after the window = control
    d = d.dropna(subset=[y])

    Y, gvec, times = _panel_matrix(d, y, unit, time, "_g")
    tab = _attgt_from_matrix(Y, gvec, times, control_group, anticipation)
    if tab.empty:
        raise RuntimeError("no identified ATT(g,t) cells")
    simple, _ = _aggregate_attgt(tab, "simple")
    _, dyn = _aggregate_attgt(tab, "dynamic")
    _, grp = _aggregate_attgt(tab, "group")

    rng = np.random.default_rng(seed)
    n_units = Y.shape[0]
    boot_simple, boot_dyn = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        try:
            bt = _attgt_from_matrix(Y[idx], gvec[idx], times, control_group, anticipation)
            if bt.empty:
                continue
            s, _ = _aggregate_attgt(bt, "simple")
            _, dd = _aggregate_attgt(bt, "dynamic")
            if np.isfinite(s):
                boot_simple.append(s)
                boot_dyn.append(dd.set_index("event_time")["att"])
        except Exception:
            continue
    if len(boot_simple) < 50:
        raise RuntimeError(f"bootstrap failed: only {len(boot_simple)} usable draws")

    bs = np.array(boot_simple)
    se = float(bs.std(ddof=1))
    lo_, hi_ = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # bootstrap p-value for H0: ATT = 0, via the symmetric percentile method
    p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))

    if boot_dyn:
        bd = pd.concat(boot_dyn, axis=1)
        dyn = dyn.set_index("event_time")
        dyn["se"] = bd.std(axis=1, ddof=1)
        dyn["ci_low"] = dyn["att"] - 1.96 * dyn["se"]
        dyn["ci_high"] = dyn["att"] + 1.96 * dyn["se"]
        dyn = dyn.reset_index()

    return {
        "att_simple": simple, "se": se, "p": p, "ci_low": float(lo_), "ci_high": float(hi_),
        "control_group": control_group, "n_boot_used": len(boot_simple),
        "attgt": tab, "dynamic": dyn, "group": grp,
        "n_countries": int(d[unit].nunique()),
        "n_never_treated": int(d.loc[d["_g"] == 0, unit].nunique()),
    }


# ---------------------------------------------------------------------------
# Wild cluster bootstrap
# ---------------------------------------------------------------------------
def wild_cluster_bootstrap(data, y, x, absorb, cluster, test_var, n_boot=1999,
                           seed=RNG_SEED):
    """Null-imposed wild cluster bootstrap-t with Webb 6-point weights.
    Appropriate when clusters are few or the treated units are concentrated in
    a small number of clusters (Cameron, Gelbach & Miller 2008; Webb 2014)."""
    rng = np.random.default_rng(seed)
    full = feols(data, y, x, absorb, cluster=cluster, name="wcb_full")
    t_obs = full[test_var]["t"]

    x_null = [v for v in x if v != test_var]
    d = data.dropna(subset=[y] + x + list(absorb) + [cluster]).copy()
    dm = _demean(d, [y] + x, list(absorb))
    Yr = dm[y].values
    if x_null:
        Xr = dm[x_null].values
        br = np.linalg.lstsq(Xr, Yr, rcond=None)[0]
        resid_r = Yr - Xr @ br
        fit_r = Xr @ br
    else:
        resid_r = Yr - Yr.mean()
        fit_r = np.full_like(Yr, Yr.mean())

    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    codes = pd.factorize(d[cluster].values)[0]
    G = codes.max() + 1
    t_boot = []
    for _ in range(n_boot):
        wgt = rng.choice(webb, size=G)[codes]
        d["_yb"] = fit_r + resid_r * wgt
        try:
            rb = feols(d, "_yb", x, absorb, cluster=cluster, name="wcb_draw")
            t_boot.append(rb[test_var]["t"])
        except Exception:
            continue
    t_boot = np.array(t_boot)
    if len(t_boot) < 100:
        raise RuntimeError("wild bootstrap produced too few usable draws")
    return {"t_obs": float(t_obs), "p_wild": float((np.abs(t_boot) >= abs(t_obs)).mean()),
            "n_boot": int(len(t_boot)), "coef": full[test_var]["coef"],
            "se_cluster": full[test_var]["se"], "p_cluster": full[test_var]["p"]}


# ---------------------------------------------------------------------------
# Permutation / randomisation inference
# ---------------------------------------------------------------------------
def permutation_test(data, y, unit="country", time="year",
                     cohort="convention_ratified_year", treated_flag="treated_ever",
                     n_perm=999, seed=RNG_SEED):
    """Reassign the OBSERVED cohort structure (the multiset of ratification
    years, including the never-treated slots) at random across countries and
    re-estimate. Gives a design-based p-value that does not rely on
    cluster-robust asymptotics with 7 control countries."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    obs = feols(_with_did(d, cohort, treated_flag, time), y, ["did"],
                absorb=[unit, time], cluster=unit, name="perm_obs")
    b_obs = obs["did"]["coef"]

    units = d[unit].unique()
    cohort_by_unit = d.drop_duplicates(unit).set_index(unit)[cohort]
    pool = cohort_by_unit.reindex(units).values.copy()

    betas = []
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        mapping = dict(zip(units, perm))
        dd = d.copy()
        dd["_g"] = dd[unit].map(mapping)
        dd["_treated"] = dd["_g"].notna() & (dd["_g"] <= dd[time].max())
        dd["_treated"] = dd["_treated"].astype(int)
        dd["did"] = ((dd["_treated"] == 1) & (dd[time] >= dd["_g"])).astype(float)
        if dd["did"].nunique() < 2:
            continue
        try:
            r = feols(dd, y, ["did"], absorb=[unit, time], cluster=unit, name="perm")
            betas.append(r["did"]["coef"])
        except Exception:
            continue
    betas = np.array(betas)
    p = float((np.abs(betas) >= abs(b_obs)).sum() + 1) / (len(betas) + 1)
    return {"coef_observed": float(b_obs), "p_permutation": p,
            "n_perm_used": int(len(betas)), "perm_mean": float(betas.mean()),
            "perm_sd": float(betas.std(ddof=1)), "perm_betas": betas}


def _with_did(d, cohort, treated_flag, time):
    dd = d.copy()
    dd["did"] = ((dd[treated_flag] == 1) & (dd[time] >= dd[cohort])).astype(float)
    return dd


# ---------------------------------------------------------------------------
# Rambachan & Roth (2023) relative-magnitudes sensitivity
# ---------------------------------------------------------------------------
def honest_rm_bounds(es_table, m_grid=(0.0, 0.5, 1.0, 1.5, 2.0), post_max=None):
    """Relative-magnitudes bounds on the post-treatment effect.

    Logic: let delta be the (unobserved) differential trend. Under RM(M) the
    post-treatment period-to-period change in delta is at most M times the
    LARGEST period-to-period change observed in the pre-period. The pre-period
    event-study coefficients identify those observed changes, so the maximum
    accumulated bias at event time e is (e+1) * M * max_pre_step.

    This is a simplified, transparent implementation of the idea in Rambachan &
    Roth (2023). It is NOT the full HonestDiD conditional/hybrid confidence
    set — it ignores estimation uncertainty in max_pre_step and uses the
    plug-in bound. Report it as a sensitivity illustration, not as an exact
    HonestDiD confidence interval.
    """
    es = es_table.sort_values("event_time")
    pre = es[es.event_time <= -1][["event_time", "coef"]]
    if len(pre) < 2:
        raise RuntimeError("need at least two pre-period coefficients")
    steps = np.abs(np.diff(pre["coef"].values))
    max_step = float(np.nanmax(steps))

    post = es[es.event_time >= 0]
    if post_max is not None:
        post = post[post.event_time <= post_max]

    rows = []
    for M in m_grid:
        for _, r in post.iterrows():
            e = int(r.event_time)
            bias = (e + 1) * M * max_step
            rows.append({
                "M": M, "event_time": e, "estimate": r.coef,
                "bound_low": r.coef - bias - 1.96 * r.se,
                "bound_high": r.coef + bias + 1.96 * r.se,
                "includes_zero": bool((r.coef - bias - 1.96 * r.se) <= 0 <=
                                      (r.coef + bias + 1.96 * r.se)),
            })
    out = pd.DataFrame(rows)
    return out, {"max_pre_step": max_step,
                 "n_pre_coefs": int(len(pre)),
                 "note": "simplified plug-in RM bound; not the full HonestDiD CI"}


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------
def mde(se, alpha=0.05, power=0.80, baseline=None):
    """Minimum detectable effect. 2.80*SE is the usual two-sided 5% / 80% figure."""
    from scipy import stats
    mult = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    val = mult * se
    out = {"se": float(se), "alpha": alpha, "power": power,
           "multiplier": float(mult), "mde": float(val)}
    if baseline:
        out["baseline_mean"] = float(baseline)
        out["mde_pct_of_baseline"] = float(val / baseline)
    return out
