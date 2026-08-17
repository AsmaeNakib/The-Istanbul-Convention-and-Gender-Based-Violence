# =============================================================================
# analysis_helpers.py
# -----------------------------------------------------------------------------
# Shared setup for the Istanbul Convention analysis: locating the canonical
# dataset, building per-outcome estimation samples, constructing named control
# groups, and exporting result tables.
#
# Imported by every notebook, so no notebook depends on another's kernel state.
#
# Two design choices are worth knowing before reading further:
#
#   * There is exactly one analysis dataset, data/gbv_panel_analysis.csv.
#     resolve_data_path() raises rather than choosing between candidates, so an
#     estimate can always be attributed to a specific file.
#
#   * The outcome registry (OUTCOMES, below) records which estimators each
#     outcome can support. Data coverage and measurement properties differ
#     sharply across outcomes, so not every method is appropriate for every one.
#     Notebook 02 runs only what the registry permits.
# =============================================================================
import os
import platform
import warnings
import importlib.metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("default")
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# ── Colours ──────────────────────────────────────────────────────────────────
C_TREATED = "#1F77B4"
C_CONTROL = "#7F7F7F"
C_POLICY = "#D62728"
C_ACCENT = "#FF7F0E"
C_MID = "#2CA02C"
C_LATE = "#9467BD"
C_WBL_DV = "#E377C2"
C_WBL_FEM = "#8C564B"
C_NEG = "#17BECF"
CANONICAL_DATASET = "gbv_panel_analysis.csv"

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 20260804
N_BOOT = 499
N_PERM = 999


# ── Inference validation ─────────────────────────────────────────────────────
def validate_inference(rows, name, raise_on_failure=True):
    """Check a list of result dicts before they are exported or plotted.

    A cluster-robust variance matrix can fail to be positive definite when the
    number of parameters approaches the number of clusters, which yields a
    non-finite standard error rather than an error. Checking here means such a
    result cannot reach a table or a figure unnoticed.
    """
    problems = []
    for r in rows:
        lbl = r.get("label", r.get("outcome", "?"))
        for key in ("b", "se", "p"):
            v = r.get(key, None)
            if v is None or not np.isfinite(float(v)):
                problems.append(f"{lbl}: non-finite {key}")
        se = r.get("se", np.nan)
        if np.isfinite(se) and se <= 0:
            problems.append(f"{lbl}: non-positive SE ({se})")
    if problems:
        msg = f"{name} failed inference validation: " + "; ".join(problems)
        if raise_on_failure:
            raise RuntimeError(msg)
        print("[NON-REPORTABLE]", msg)
        return False
    return True


def export_table(rows, path, name, raise_on_failure=True):
    """Write a result table. Every table in outputs/ goes through this function,
    so validation applies uniformly."""
    df = pd.DataFrame(rows)
    if {"b", "se", "p"}.issubset(df.columns):
        validate_inference(rows, name, raise_on_failure=raise_on_failure)
    df.to_csv(path, index=False)
    return df


# ── Paths ────────────────────────────────────────────────────────────────────
def resolve_data_path(project_dir, filename=CANONICAL_DATASET):
    """Locate the analysis panel, raising if the choice is ambiguous.

    Every result must be attributable to a specific file, so this refuses to
    choose when several candidates exist rather than picking one. Set the
    GBV_DATA_PATH environment variable to override.
    """
    project_dir = Path(project_dir)
    env_path = os.environ.get("GBV_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GBV_DATA_PATH is set to {p}, which does not exist.")
        return p

    candidates = [project_dir / filename,
                  project_dir / "data" / filename,
                  project_dir.parent / filename,
                  project_dir.parent / "data" / filename]
    existing, seen = [], set()
    for p in candidates:
        p = p.resolve()
        if p.exists() and p not in seen:
            existing.append(p)
            seen.add(p)
    if not existing:
        raise FileNotFoundError(
            f"Could not find {filename}. Run `python src/build_analysis_panel.py` "
            f"first, or set GBV_DATA_PATH.")
    if len(existing) > 1:
        listing = "\n".join(f"  - {p}" for p in existing)
        raise RuntimeError(
            f"Ambiguous dataset: {len(existing)} files named {filename} were found:\n"
            f"{listing}\nRefusing to guess. Set GBV_DATA_PATH to the intended file.")
    return existing[0]


def make_output_dirs(project_dir):
    """Three output directories, by role rather than by topic.

    tables       result tables quoted in the thesis or its appendix
    figures      every figure the notebooks produce
    diagnostics  design diagnostics and reproducibility records
    """
    project_dir = Path(project_dir)
    out = {k: project_dir / "outputs" / k
           for k in ["tables", "figures", "diagnostics"]}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def save_fig(path, fn):
    plt.tight_layout()
    plt.savefig(Path(path) / fn, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fn}")
    plt.close()


def fig_note(text="Source: UNODC UN-CTS; World Bank WDI/WBL; Council of Europe GREVIO; WAVE."):
    plt.figtext(0.01, -0.03, text, ha="left", fontsize=8.5, color="#555")


def cohort_label(y):
    if pd.isna(y):
        return "Never ratified"
    y = int(y)
    if y <= 2014:
        return "Early (2012-2014)"
    if y <= 2017:
        return "Middle (2015-2017)"
    return "Late (2018+)"


# ── Outcome registry ─────────────────────────────────────────────────────────
# One entry per outcome. `role` countries the inferential job the outcome does in
# the thesis; `methods` countries which estimators its data can support, and
# nothing in the pipeline runs a method not listed here. The restrictions come
# from coverage and measurement properties, which differ sharply across
# outcomes: a series that is short, sparse, near-absorbing, or produced by the
# reporting process the treatment itself changes cannot sustain the same
# estimators as female homicide. Each `note` gives the reason.
OUTCOMES = [
    {"key": "fhr", "file_slug": "female_homicide", "label": "Female homicide", "source_col":
     "outcome_primary_female_intentional_homicide_rate_per_100k",
     "sample_flag": "sample_primary_female_homicide_2000_2023", "is_binary": False,
     "is_reporting": False, "color": C_TREATED, "role": "primary",
     "methods": ["twfe", "event_study", "stacked", "cs", "permutation", "honest"],
     "note": ("Primary outcome. The best-measured series in the panel: lethal "
              "violence is the least sensitive to changes in reporting.")},

    {"key": "male_hom", "file_slug": "male_homicide", "label": "Male homicide (comparison)",
     "source_col": "male_intentional_homicide_rate_per_100k",
     "sample_flag": None, "is_binary": False, "is_reporting": False, "color": C_NEG,
     "role": "negative_control",
     "methods": ["twfe", "event_study", "stacked", "cs"],
     "note": ("Male homicide comparison. The Convention targets violence against women. "
              "A comparable estimate on male homicide indicates a general "
              "homicide/recording shock rather than a gender-specific effect.")},

    {"key": "log_ratio", "file_slug": "homicide_sex_ratio", "label": "log(female/male homicide)",
     "source_col": "log_female_male_homicide_ratio",
     "sample_flag": None, "is_binary": False, "is_reporting": False, "color": C_MID,
     "role": "triple_difference",
     "methods": ["twfe", "event_study", "stacked", "cs"],
     "note": ("Triple difference. Any country-year shock common to male and "
              "female homicide (recording changes, conflict, data revisions) "
              "differences out.")},

    {"key": "fem_ipf", "file_slug": "female_ipf_homicide", "label": "Female intimate-partner/family homicide",
     "source_col": "female_ipf_homicide_rate_per_100k",
     "sample_flag": None, "is_binary": False, "is_reporting": False, "color": C_POLICY,
     "role": "targeted",
     "methods": ["twfe", "event_study", "descriptive"],
     "note": ("The outcome the Convention actually targets. Only available from "
              "2005 and for ~32 countries, so pre-treatment history is short for "
              "early ratifiers — treat event-study leads with care.")},

    {"key": "svr", "file_slug": "sexual_violence", "label": "Recorded sexual violence", "source_col":
     "outcome_secondary_sexual_violence_rate_per_100k",
     "sample_flag": "sample_secondary_sexual_violence_2005_2023", "is_binary": False,
     "is_reporting": True, "color": C_ACCENT, "role": "reporting_sensitive",
     "methods": ["event_study", "descriptive"],
     "note": ("Recorded crime, not incidence. 71x cross-country spread; the Convention itself "
              "mandates better reporting (Arts 18/21/55). Level DiD is not "
              "interpretable; analyse in logs and read as reporting behaviour.")},

    {"key": "rape", "file_slug": "rape", "label": "Rape", "source_col":
     "outcome_secondary_rape_rate_per_100k",
     "sample_flag": "sample_secondary_rape_2005_2023", "is_binary": False,
     "is_reporting": True, "color": C_LATE, "role": "reporting_sensitive",
     "methods": ["descriptive"],
     "note": ("Recorded crime, and the least reliable series in the panel: Italy "
              "reports no observations, the cross-country spread is 91x, seven "
              "countries show measurement breaks, and the placebo timing test "
              "detects an effect before treatment. Descriptive use only.")},

    {"key": "wbl_dv_legislation", "file_slug": "dv_legislation", "label": "DV legislation",
     "source_col": "wbl_dv_legislation",
     "sample_flag": "sample_wbl_dv_legislation", "is_binary": True,
     "is_reporting": False, "color": C_WBL_DV, "role": "legal",
     "methods": ["descriptive"],
     "note": ("Only 2 of 38 in-sample adoptions occur after the adopter's own "
              "ratification, and the event study fails its pre-trend test. "
              "Descriptive adoption timeline only — no DiD.")},

    {"key": "wbl_femicide_law", "file_slug": "femicide_law", "label": "Femicide law",
     "source_col": "wbl_femicide_law",
     "sample_flag": "sample_wbl_femicide_law", "is_binary": True,
     "is_reporting": False, "color": C_WBL_FEM, "role": "legal",
     "methods": ["descriptive"],
     "note": ("Seven countries ever adopt; three adopted before ratifying, and "
              "three of the four that adopted afterwards did so in 2023. No "
              "control country ever adopts, so there is no counterfactual. "
              "Descriptive event history only — no DiD.")},
]
OUTCOME_BY_KEY = {o["key"]: o for o in OUTCOMES}


def supports(key, method):
    return method in OUTCOME_BY_KEY[key]["methods"]



# ── Control-group construction ───────────────────────────────────────────────
def control_group(df, kind="never_treated", outcome="fhr"):
    """Return the countries serving as controls under a named rule.

    The rules are named rather than implicit so that every reported estimate
    countries which control group produced it.

    never_treated   the six non-parties plus Latvia, which ratified in 2024,
                    outside the estimation window. This is the conventional
                    choice, and it is not a comparable group: all six
                    non-parties are post-socialist countries, their mean GDP is
                    about a third of the treated group's, and Lithuania alone
                    moves the main estimate by 55%.
    no_lithuania    never-treated minus Lithuania, whose pre-2013 mean female
                    homicide rate (4.72) is five times the treated median.
    cee_only        Czechia, Hungary and Slovakia, the never-treated countries
                    closest to the treated group in level and trend.
    """
    ctrl = sorted(df.loc[df["treated_ever"].eq(0), "country"].unique())
    if kind == "never_treated":
        return ctrl
    if kind == "no_lithuania":
        return [c for c in ctrl if c != "Lithuania"]
    if kind == "cee_only":
        return [c for c in ctrl if c in {"Czechia", "Hungary", "Slovakia"}]
    raise ValueError(kind)


def comparability_restricted(sample, how="post_socialist"):
    """Restrict the estimation sample so treated and control units are drawn
    from the same population, rather than restricting only the control group.

    All six non-parties are post-socialist countries while the treated group is
    dominated by Western Europe, so the conventional comparison mixes a
    treatment contrast with a regional one. Comparing post-socialist ratifiers
    against post-socialist non-ratifiers removes that imbalance. The cost is a
    smaller sample and a different estimated change: the ATT among post-socialist
    ratifiers, not among all ratifiers.
    """
    if how == "post_socialist":
        if "post_socialist" not in sample.columns:
            raise KeyError("post_socialist column missing — rebuild the analysis panel")
        return sample[sample["post_socialist"].eq(1)].copy()
    if how == "same_subregion":
        subs = set(sample.loc[sample["treated_ever"].eq(0), "unodc_subregion"].dropna())
        return sample[sample["unodc_subregion"].isin(subs)].copy()
    raise ValueError(how)


def pretreatment_balance(sample, outcome="fhr", cutoff=2013,
                         covars=("gdp_per_capita_constant_2015_usd",
                                 "female_labor_force_participation_pct")):
    """Per-country pre-treatment profile used for balance tables and matching."""
    pre = sample[sample["year"] < cutoff]
    rows = []
    for c, g in pre.groupby("country"):
        g = g.dropna(subset=[outcome])
        if len(g) < 4:
            continue
        row = {"country": c,
               "treated": int(g["treated_ever"].iloc[0]),
               "pre_mean": float(g[outcome].mean()),
               "pre_sd": float(g[outcome].std()),
               "pre_slope": float(np.polyfit(g["year"], g[outcome], 1)[0]),
               "n_pre": int(len(g))}
        for v in covars:
            if v in g.columns:
                row[v] = float(g[v].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# Estimation window per outcome. Module-level so that a notebook building a
# variant sample uses the same window as the corresponding entry in SAMPLES;
# otherwise the variant would differ in its year range as well as in whatever
# it was meant to vary.
SAMPLE_WINDOWS = {
    "fhr": (2000, 2023), "male_hom": (2000, 2023), "log_ratio": (2000, 2023),
    "fem_ipf": (2005, 2023), "svr": (2005, 2023), "rape": (2005, 2023),
    "wbl_dv_legislation": (1990, 2023), "wbl_femicide_law": (1990, 2023),
}


def build_sample(data, outcome_col, sample_flag=None, excl_micro=True,
                 excl_turkey_post_denunciation=True,
                 year_min=None, year_max=None):
    """Build the estimation sample for one outcome.

    excl_micro drops Andorra, Liechtenstein, Monaco and San Marino, whose
    populations are small enough that a single homicide moves the rate by
    several points.

    excl_turkey_post_denunciation drops Turkey's 2021-2023 rows. Turkey itself
    stays in the sample; only the years after its withdrawal took legal effect
    are removed, since in those years it is not a party and the rows are not
    treated in any meaningful sense.

    Pass year_min and year_max from SAMPLE_WINDOWS when building a variant of an
    existing sample, so the two span the same years.
    """
    work = data.copy()
    if sample_flag and sample_flag in work.columns:
        work = work.loc[work[sample_flag].eq(1)]
    if year_min is not None:
        work = work.loc[work["year"] >= year_min]
    if year_max is not None:
        work = work.loc[work["year"] <= year_max]
    if excl_micro and "microstate_country" in work:
        work = work.loc[work["microstate_country"].ne(1)]
    if (excl_turkey_post_denunciation
            and "turkey_post_denunciation_2021_onward" in work):
        work = work.loc[work["turkey_post_denunciation_2021_onward"].ne(1)]
    work = work.dropna(subset=[outcome_col]).copy()
    work["did_interaction"] = work["treated_ever"] * work["post_ratification"]
    return work.sort_values(["country", "year"]).reset_index(drop=True)


def build_turkey_panel(df, outcome_col, withdrawal_year=2021, sample_start=2012):
    """Panel comparing Turkey with ratifiers that remained party to the
    Convention, used for the descriptive withdrawal case study."""
    base = build_sample(df, outcome_col, excl_micro=True,
                        excl_turkey_post_denunciation=False)
    still_active = base[(base["treated_ever"] == 1) & (base["country"] != "Turkey") &
                        (base["year"].between(sample_start, 2023))]
    turkey = base[(base["country"] == "Turkey") & (base["year"].between(sample_start, 2023))]
    panel = pd.concat([turkey, still_active], ignore_index=True)
    panel["is_turkey"] = (panel["country"] == "Turkey").astype(int)
    panel["post_withdrawal"] = (panel["year"] >= withdrawal_year).astype(int)
    panel["did_withdrawal"] = panel["is_turkey"] * panel["post_withdrawal"]
    return panel


# ── Load and prepare ─────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "year", "convention_signed_year", "convention_ratified_year",
    "convention_entry_into_force_year", "convention_withdrew_effective_year",
    "ratified_by_2023", "treatment_main_ratification", "post_entry_into_force",
    "event_time_main_ratification_years",
    "outcome_primary_female_intentional_homicide_rate_per_100k",
    "outcome_secondary_sexual_violence_rate_per_100k",
    "outcome_secondary_rape_rate_per_100k",
    "male_intentional_homicide_rate_per_100k",
    "total_intentional_homicide_rate_per_100k",
    "female_ipf_homicide_rate_per_100k", "male_ipf_homicide_rate_per_100k",
    "female_ip_homicide_rate_per_100k", "female_share_of_homicide_victims",
    "log_female_male_homicide_ratio", "post_socialist",
    "gdp_per_capita_constant_2015_usd", "female_labor_force_participation_pct",
    "government_effectiveness_wgi", "rule_of_law_wgi",
    "microstate_country", "population_total",
    "grevio_baseline_report_year_verified",
    "wave_shelter_beds_per_10k_total_population_2020",
    "wave_shelter_meets_ic_minimum_standard_2020",
    "turkey_post_denunciation_2021_onward",
    "wbl_dv_legislation", "wbl_femicide_law",
    "sample_primary_female_homicide_2000_2023",
    "sample_secondary_sexual_violence_2005_2023",
    "sample_secondary_rape_2005_2023",
    "sample_wbl_dv_legislation",
    "sample_wbl_femicide_law",
]


def load_and_prepare_data(data_path, out_dir, verbose=True):
    """Load the canonical analysis panel, validate it, and build every
    per-outcome sample. Raises on any structural problem rather than repairing
    it silently."""
    if verbose:
        print("=" * 72)
        print("Loading and validating the analysis panel")
        print("=" * 72)
        print(f"Reading: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan}), errors="coerce")

    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df = df[df["year"].between(1990, 2023)].copy()

    if df.duplicated(["country", "year"]).any():
        raise ValueError("Duplicate country-year rows in the analysis panel")

    # Outcome aliases. fhr is FEMALE INTENTIONAL HOMICIDE, not femicide.
    for o in OUTCOMES:
        if o["source_col"] in df.columns:
            df[o["key"]] = pd.to_numeric(df[o["source_col"]], errors="coerce")
        else:
            df[o["key"]] = np.nan
            if verbose:
                print(f"  WARNING: source column {o['source_col']} not found for {o['key']}")
    df["log1p_fhr"] = np.log1p(df["fhr"])
    df["log1p_svr"] = np.log1p(df["svr"])
    df["log1p_rape"] = np.log1p(df["rape"])
    # log_svr is the transformation the reported recorded sexual-violence
    # regression uses. Defined here so the main model, its event study, its
    # pre-trend test and its robustness checks all read the same column.
    df["log_svr"] = np.log(df["svr"].where(df["svr"] > 0))

    # Treatment. Latvia ratified 2024 -> untreated inside the window.
    df["treated_ever"] = pd.to_numeric(df["ratified_by_2023"], errors="coerce").fillna(0).astype(int)
    df.loc[df["country"].str.casefold().eq("latvia") &
           df["convention_ratified_year"].gt(df["year"].max()), "treated_ever"] = 0
    df["post_ratification"] = pd.to_numeric(
        df["treatment_main_ratification"], errors="coerce").fillna(0).astype(int)
    df["did_interaction"] = df["treated_ever"] * df["post_ratification"]
    df["did_eif"] = df["treated_ever"] * pd.to_numeric(
        df["post_entry_into_force"], errors="coerce").fillna(0).astype(int)
    # Within-year timing variants. Ratification dates are days, not years, and
    # several fall in the last quarter (Austria 14 Nov 2013, Bosnia 7 Nov 2013,
    # Serbia 21 Nov 2013), so coding the calendar year of ratification as fully
    # treated attenuates the first-year effect. These switch treatment on in the
    # first FULL calendar year instead. Robustness only; the main definition is
    # did_interaction. Built by src/build_analysis_panel.py from the verified
    # treaty-date record.
    for src_col, new_col in [
            ("treatment_first_full_year_ratification", "did_first_full_year"),
            ("treatment_first_full_year_eif", "did_first_full_year_eif")]:
        if src_col in df.columns:
            df[new_col] = df["treated_ever"] * pd.to_numeric(
                df[src_col], errors="coerce").fillna(0).astype(int)
    # Non-absorbing variant: Turkey returns to untreated from its 2021
    # withdrawal. Not valid for stacked DiD or Callaway-Sant'Anna, both of
    # which assume absorbing treatment.
    if "treatment_ratification_nonabsorbing" in df.columns:
        df["did_nonabsorbing"] = df["treated_ever"] * pd.to_numeric(
            df["treatment_ratification_nonabsorbing"],
            errors="coerce").fillna(0).astype(int)
    df["years_active"] = np.where(df["post_ratification"].eq(1),
                                  (df["year"] - df["convention_ratified_year"]).clip(lower=0), 0)
    df["cohort"] = df["convention_ratified_year"].apply(cohort_label)

    for col in ["treated_ever", "post_ratification", "did_interaction",
                "microstate_country", "wbl_dv_legislation", "wbl_femicide_law"]:
        vals = set(df[col].dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            raise ValueError(f"{col} must be binary. Found {sorted(vals)}")
    bad = df[(df["post_ratification"] == 1) & df["convention_ratified_year"].notna() &
             (df["year"] < df["convention_ratified_year"])]
    if len(bad):
        raise ValueError(f"{len(bad)} rows coded post-ratification before ratification")

    country_info = (df[["country", "iso3", "convention_signed_year",
                        "convention_ratified_year", "convention_entry_into_force_year",
                        "treated_ever", "microstate_country", "post_socialist",
                        "unodc_subregion"]]
                    .drop_duplicates("country").copy())
    country_info["cohort"] = country_info["convention_ratified_year"].apply(cohort_label)

    SAMPLES = {}
    for o in OUTCOMES:
        y0, y1 = SAMPLE_WINDOWS[o["key"]]
        SAMPLES[o["key"]] = build_sample(df, o["key"], sample_flag=None,
                                         year_min=y0, year_max=y1)

    inv = []
    for o in OUTCOMES:
        s = SAMPLES[o["key"]]
        inv.append({"outcome": o["key"], "label": o["label"], "role": o["role"],
                    "N": len(s), "countries": s["country"].nunique(),
                    "treated_countries": int(s.loc[s.treated_ever.eq(1), "country"].nunique()),
                    "control_countries": int(s.loc[s.treated_ever.eq(0), "country"].nunique()),
                    "year_min": int(s["year"].min()) if len(s) else None,
                    "year_max": int(s["year"].max()) if len(s) else None,
                    "methods_allowed": "|".join(o["methods"])})
    inventory = pd.DataFrame(inv)
    inventory.to_csv(Path(out_dir["diagnostics"]) / "analysis_sample_inventory.csv", index=False)

    pd.DataFrame({"column": df.columns,
                  "dtype": [str(df[c].dtype) for c in df.columns],
                  "missing_n": [int(df[c].isna().sum()) for c in df.columns],
                  "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns]}
                 ).to_csv(Path(out_dir["diagnostics"]) / "data_schema_missingness.csv", index=False)

    if verbose:
        print(f"\nLoaded {len(df):,} rows, {df['country'].nunique()} countries, "
              f"{df.year.min()}-{df.year.max()}")
        print(inventory[["label", "N", "countries", "treated_countries",
                         "control_countries", "methods_allowed"]].to_string(index=False))
    return {"df": df, "SAMPLES": SAMPLES, "country_info": country_info,
            "inventory": inventory,
            "ALL_CTRL": control_group(SAMPLES["fhr"], "never_treated")}


def software_manifest(out_dir, data_path):
    pkgs = ["pandas", "numpy", "scipy", "statsmodels", "matplotlib", "seaborn",
            "nbformat", "csdid", "pyfixest", "linearmodels"]
    rows = []
    for p in pkgs:
        try:
            rows.append({"package": p, "version": importlib_metadata.version(p)})
        except Exception:
            rows.append({"package": p, "version": "NOT INSTALLED"})
    pd.DataFrame(rows).to_csv(Path(out_dir["diagnostics"]) / "software_manifest.csv", index=False)
    import sys
    (Path(out_dir["diagnostics"]) / "runtime.txt").write_text(
        f"Python: {sys.version}\nPlatform: {platform.platform()}\nData: {data_path}\n"
        f"Seed: {SEED}\n", encoding="utf-8")


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, (m - rank + 1) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj
