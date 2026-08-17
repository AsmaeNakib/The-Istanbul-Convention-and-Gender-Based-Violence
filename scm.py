# =============================================================================
# scm.py
# -----------------------------------------------------------------------------
# Synthetic control for single treated units, with the feasibility and fit
# diagnostics needed to decide whether a fitted synthetic may be interpreted.
#
# Three properties of this implementation matter for the results:
#
#   * The synthetic series is formed only from donors carrying non-trivial
#     weight. numpy evaluates nan * 0 as nan, so a zero-weight donor with a
#     missing year would otherwise delete that year from the synthetic.
#   * Optimiser convergence is checked, and a failure raises.
#   * Fit is judged on pre-period R-squared and on RMSPE relative to the treated
#     unit's own pre-period standard deviation, and is reported either way. A
#     synthetic that explains none of the pre-period variation scores an RMSPE
#     ratio near 1.0, so a threshold above that would accept anything.
#
# Donors are restricted to countries that never ratify inside the observation
# window. A country that ratifies during the treated unit's post-treatment
# period is itself treated, and cannot represent the untreated counterfactual.
# =============================================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize

WEIGHT_TOL = 1e-6


def _solve_weights(Y_pre, D_pre, seed_starts=3):
    """Convex weights minimising pre-period SSR. Multiple starts because SLSQP
    on a simplex can stop at a face depending on the initial point."""
    J = D_pre.shape[1]
    best = None
    starts = [np.ones(J) / J]
    rng = np.random.default_rng(20260804)
    for _ in range(seed_starts - 1):
        v = rng.random(J)
        starts.append(v / v.sum())
    for w0 in starts:
        r = minimize(lambda w: float(np.sum((Y_pre - D_pre @ w) ** 2)), w0,
                     method="SLSQP", bounds=[(0.0, 1.0)] * J,
                     constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                     options={"ftol": 1e-12, "maxiter": 1000})
        if best is None or (r.success and r.fun < best.fun) or (not best.success and r.success):
            best = r
    if not best.success:
        raise RuntimeError(f"SCM optimiser did not converge: {best.message}")
    return best


def eligible_donors(df, treated_country, treat_year, outcome, pre_years,
                    cohort_col="convention_ratified_year",
                    treated_flag="treated_ever", micro_col="microstate_country"):
    """Countries admissible as donors for `treated_country`.

    A donor must never ratify inside the observation window, since a country
    that ratifies during the treated unit's post-treatment period is itself
    treated and cannot represent the untreated counterfactual. Microstates are
    excluded because their rates are dominated by single events.
    """
    end_year = int(df["year"].max())
    base = (df[micro_col] != 1) & (df["country"] != treated_country)
    mask = base & ((df[treated_flag] == 0) | (df[cohort_col] > end_year))
    cand = sorted(df.loc[mask, "country"].unique())

    pivot = df.pivot_table(index="year", columns="country", values=outcome, aggfunc="first")
    clean, dropped = [], {}
    for c in cand:
        if c not in pivot.columns:
            dropped[c] = "no data"
            continue
        nmiss = int(pivot.loc[pre_years, c].isna().sum())
        if nmiss:
            dropped[c] = f"{nmiss} missing pre-period year(s)"
        else:
            clean.append(c)
    return clean, dropped, pivot


def run_scm(df, treated_country, treat_year, outcome, all_years,
            interpolate_gaps=0,
            min_pre_years=8, verbose=True):
    """Fit one synthetic control and return results plus honest fit diagnostics."""
    pre_years = [y for y in all_years if y < treat_year]
    post_years = [y for y in all_years if y >= treat_year]
    if len(pre_years) < min_pre_years:
        raise RuntimeError(
            f"{treated_country}/{outcome}: only {len(pre_years)} pre-treatment years "
            f"(minimum {min_pre_years}). Synthetic control is not appropriate.")

    donors, dropped, pivot = eligible_donors(
        df, treated_country, treat_year, outcome, pre_years)
    if treated_country not in pivot.columns:
        raise RuntimeError(f"{treated_country} has no {outcome} data")
    if len(donors) < 2:
        raise RuntimeError(
            f"{treated_country}/{outcome}: only {len(donors)} eligible donor(s) "
            "under the never-treated donor rule. A synthetic control cannot be "
            "constructed.")

    y_pre_raw = pivot.loc[pre_years, treated_country].copy()
    n_missing = int(y_pre_raw.isna().sum())
    if n_missing:
        if n_missing <= interpolate_gaps:
            y_pre_raw = y_pre_raw.interpolate(limit_direction="both")
        else:
            raise RuntimeError(
                f"{treated_country}/{outcome}: {n_missing} missing pre-period year(s) "
                f"(interpolate_gaps={interpolate_gaps}).")
    Y_pre = y_pre_raw.values.astype(float)
    D_pre = pivot.loc[pre_years, donors].values.astype(float)

    res = _solve_weights(Y_pre, D_pre)              # S2: raises if not converged
    w = res.x

    keep = w > WEIGHT_TOL                            # S1: drop the numerical zeros
    kept_donors = list(np.array(donors)[keep])
    w_keep = w[keep]
    w_keep = w_keep / w_keep.sum()
    synth = pivot.loc[all_years, kept_donors].values.astype(float) @ w_keep
    actual = pivot.loc[all_years, treated_country].values.astype(float).copy()
    if n_missing:
        actual[[all_years.index(y) for y in pre_years]] = Y_pre
    gap = actual - synth

    pre_mask = np.array([y < treat_year for y in all_years])
    rmspe_pre = float(np.sqrt(np.nanmean(gap[pre_mask] ** 2)))
    rmspe_post = float(np.sqrt(np.nanmean(gap[~pre_mask] ** 2))) if (~pre_mask).any() else np.nan
    avg_post_gap = float(np.nanmean(gap[~pre_mask])) if (~pre_mask).any() else np.nan

    # S3: honest fit metrics
    pre_sd = float(np.nanstd(Y_pre))
    ss_res = float(np.nansum(gap[pre_mask] ** 2))
    ss_tot = float(np.nansum((Y_pre - np.nanmean(Y_pre)) ** 2))
    r2_pre = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmspe_ratio_sd = rmspe_pre / pre_sd if pre_sd > 0 else np.nan
    fit_ok = bool(np.isfinite(r2_pre) and r2_pre >= 0.70 and rmspe_ratio_sd <= 0.55)

    # how much of the weight sits on units treated during the post-period
    cohort = df.drop_duplicates("country").set_index("country")["convention_ratified_year"]
    end_year = int(df["year"].max())
    contaminated = sum(
        wt for c, wt in zip(kept_donors, w_keep)
        if pd.notna(cohort.get(c, np.nan)) and cohort.get(c) <= end_year)

    weights_df = (pd.DataFrame({"donor": kept_donors, "weight": w_keep})
                  .assign(ratified=lambda t: t.donor.map(cohort),
                          treated_in_window=lambda t: t.ratified.notna() & (t.ratified <= end_year))
                  .sort_values("weight", ascending=False).reset_index(drop=True))
    results = pd.DataFrame({"year": all_years, "actual": actual,
                            "synthetic": synth, "gap": gap})

    diag = {
        "treated_country": treated_country, "outcome": outcome, "treat_year": treat_year,
        "n_eligible_donors": len(donors),
        "n_positive_weight_donors": int(keep.sum()),
        "n_pre_years": len(pre_years), "n_post_years": len(post_years),
        "rmspe_pre": rmspe_pre, "rmspe_post": rmspe_post,
        "rmspe_ratio_post_pre": rmspe_post / rmspe_pre if rmspe_pre > 0 else np.nan,
        "avg_post_gap": avg_post_gap,
        "treated_pre_sd": pre_sd, "rmspe_over_pre_sd": rmspe_ratio_sd,
        "pre_period_r2": r2_pre,
        "share_weight_on_later_ratifiers": float(contaminated),
        "fit_acceptable": fit_ok,
        "dropped_donors": dropped,
        "optimiser_success": bool(res.success), "optimiser_ssr": float(res.fun),
        "n_synthetic_missing_post": int(np.isnan(synth[~pre_mask]).sum()),
    }
    if verbose:
        print(f"\n  {treated_country} / {outcome} (treatment year {treat_year})")
        print(f"    eligible donors={len(donors)}  positive weight={int(keep.sum())}")
        print(f"    pre-RMSPE={rmspe_pre:.4f}   treated pre-SD={pre_sd:.4f}   "
              f"ratio={rmspe_ratio_sd:.3f}")
        print(f"    pre-period R^2={r2_pre:.4f}   -> fit {'ACCEPTABLE' if fit_ok else 'NOT ACCEPTABLE'}")
        print(f"    post/pre RMSPE ratio={diag['rmspe_ratio_post_pre']:.4f}   "
              f"avg post gap={avg_post_gap:+.4f}")
        if contaminated > 0.01:
            print(f"    WARNING: {contaminated:.1%} of donor weight is on countries that "
                  f"ratify inside the observation window")
        if not fit_ok:
            print(f"    -> Do not interpret the post-treatment gap causally. "
                  f"The donor pool cannot reproduce this unit before treatment.")
    return results, weights_df, diag


def leave_one_donor_out(df, treated_country, treat_year, outcome, all_years,
                        weights_df,
                        interpolate_gaps=0):
    """Refit dropping each positively weighted donor in turn."""
    pre_years = [y for y in all_years if y < treat_year]
    donors, _, pivot = eligible_donors(df, treated_country, treat_year, outcome,
                                       pre_years)
    y_pre = pivot.loc[pre_years, treated_country].astype(float)
    if interpolate_gaps:
        y_pre = y_pre.interpolate(limit_direction="both")
    Y_pre = y_pre.values
    pre_mask = np.array([y < treat_year for y in all_years])
    rows = []
    for drop in weights_df["donor"]:
        dd = [c for c in donors if c != drop]
        if len(dd) < 2:
            continue
        try:
            r = _solve_weights(Y_pre, pivot.loc[pre_years, dd].values.astype(float))
        except RuntimeError:
            continue
        keep = r.x > WEIGHT_TOL
        kd = list(np.array(dd)[keep]); wk = r.x[keep] / r.x[keep].sum()
        sy = pivot.loc[all_years, kd].values.astype(float) @ wk
        g = pivot.loc[all_years, treated_country].values.astype(float) - sy
        rows.append({"dropped_donor": drop,
                     "rmspe_pre": float(np.sqrt(np.nanmean(g[pre_mask] ** 2))),
                     "avg_post_gap": float(np.nanmean(g[~pre_mask]))})
    return pd.DataFrame(rows)


def placebo_in_space(df, treated_country, treat_year, outcome, all_years,
                     actual_ratio=None):
    """Fit the same model treating each donor as if it were treated.
    Returns the placebo RMSPE ratios and, if `actual_ratio` is supplied, the
    rank-based p-value."""
    pre_years = [y for y in all_years if y < treat_year]
    donors, _, pivot = eligible_donors(df, treated_country, treat_year, outcome,
                                       pre_years)
    pre_mask = np.array([y < treat_year for y in all_years])
    rows = []
    for placebo in donors:
        others = [c for c in donors if c != placebo]
        if len(others) < 2:
            continue
        Yp = pivot.loc[pre_years, placebo].values.astype(float)
        if np.isnan(Yp).any():
            continue
        try:
            r = _solve_weights(Yp, pivot.loc[pre_years, others].values.astype(float))
        except RuntimeError:
            continue
        keep = r.x > WEIGHT_TOL
        kd = list(np.array(others)[keep]); wk = r.x[keep] / r.x[keep].sum()
        sy = pivot.loc[all_years, kd].values.astype(float) @ wk
        g = pivot.loc[all_years, placebo].values.astype(float) - sy
        pre_g = g[pre_mask][~np.isnan(g[pre_mask])]
        post_g = g[~pre_mask][~np.isnan(g[~pre_mask])]
        if len(pre_g) < 3 or len(post_g) < 1:
            continue
        rp = float(np.sqrt(np.mean(pre_g ** 2)))
        if rp < 1e-8:
            continue
        rows.append({"country": placebo, "rmspe_pre": rp,
                     "rmspe_post": float(np.sqrt(np.mean(post_g ** 2))),
                     "ratio": float(np.sqrt(np.mean(post_g ** 2)) / rp),
                     "gaps": g.tolist()})
    pl = pd.DataFrame(rows)
    p = np.nan
    if actual_ratio is not None and len(pl):
        p = float((pl["ratio"] >= actual_ratio).sum() + 1) / (len(pl) + 1)
    return pl, p


def feasibility_report(df, candidates, outcome, all_years):
    """Before fitting anything, ask whether SCM is even possible: how many
    uncontaminated donors exist, and do their outcome levels bracket the
    treated unit? If the treated unit lies outside the donor convex hull on
    levels, no convex combination can track it."""
    rows = []
    for country, treat_year in candidates:
        pre_years = [y for y in all_years if y < treat_year]
        try:
            donors, dropped, pivot = eligible_donors(
                df, country, treat_year, outcome, pre_years)
        except Exception as e:
            rows.append({"country": country, "treat_year": treat_year,
                         "n_donors": 0, "note": f"error: {e}"})
            continue
        if country not in pivot.columns or not donors:
            rows.append({"country": country, "treat_year": treat_year,
                         "n_donors": len(donors), "note": "no usable data"})
            continue
        tmean = float(np.nanmean(pivot.loc[pre_years, country]))
        dmeans = pivot.loc[pre_years, donors].mean()
        rows.append({
            "country": country, "treat_year": treat_year, "n_pre_years": len(pre_years),
            "n_donors": len(donors), "n_donors_dropped_missing": len(dropped),
            "treated_pre_mean": tmean,
            "donors_below_treated": int((dmeans < tmean).sum()),
            "donors_above_treated": int((dmeans > tmean).sum()),
            "donor_mean_min": float(dmeans.min()), "donor_mean_max": float(dmeans.max()),
            "inside_convex_hull_on_levels": bool((dmeans < tmean).any() and (dmeans > tmean).any()),
        })
    return pd.DataFrame(rows)
