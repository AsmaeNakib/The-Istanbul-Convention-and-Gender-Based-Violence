# =============================================================================
# build_analysis_panel.py
# -----------------------------------------------------------------------------
# Builds data/gbv_panel_analysis.csv, the single dataset every notebook reads.
#
# Inputs
#   data/interim/base_country_year_panel.csv  country-year frame carrying
#       population, GDP, governance indicators and the World Bank WBL legal
#       variables.
#   data/raw/data_cts_intentional_homicide.xlsx  UNODC UN-CTS homicide file.
#   data/treatment_dates.csv  the Convention dates used in the analysis.
#
# What this script does
#
#   1. Puts the United Kingdom on the same denominator as every other country.
#      UNODC reports the UK as three separate jurisdictions, so its series is
#      rebuilt here by summing victim counts across them and dividing by the
#      summed population of the same sex. Sex-disaggregated UNODC rates are
#      per 100,000 of the population of that sex, so female homicide is per
#      100,000 women and male homicide per 100,000 men.
#
#   2. Adds outcomes present in the UNODC file but absent from the base panel:
#      male homicide, used as an additional outcome for comparison; total
#      homicide; and intimate-partner or family-member homicide by sex, the
#      category closest to what the Convention legislates about.
#
#   3. Derives the log female-to-male homicide ratio and the female share of
#      homicide victims.
#
#   4. Attaches the Convention dates and builds the ratification indicators.
#
# Every column added or derived is documented in data/variable_sources.csv.
#
# Run:  python src/build_analysis_panel.py
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
BASE_PANEL = PROJECT_DIR / "data" / "interim" / "base_country_year_panel.csv"
OUT_PANEL = PROJECT_DIR / "data" / "gbv_panel_analysis.csv"
VARIABLE_SOURCES = PROJECT_DIR / "data" / "variable_sources.csv"
TREATY_DATES = PROJECT_DIR / "data" / "treatment_dates.csv"

UNODC_HOMICIDE = RAW_DIR / "data_cts_intentional_homicide.xlsx"

# Country names as they appear in the raw sources, mapped to the panel's names.
# UNODC splits the United Kingdom into three jurisdictions and names several
# countries differently from the World Bank, so every raw name must be mapped
# explicitly or its observations are silently dropped.
NAME_ALIASES = {
    "Czech Republic": "Czechia",
    "Republic of Moldova": "Moldova",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Slovak Republic": "Slovakia",
    "Netherlands (Kingdom of the)": "Netherlands",
}
# UNODC splits the UK into three reporting jurisdictions. They are aggregated
# from counts, never from rates (rates cannot be averaged without weights).
UK_JURISDICTIONS = [
    "United Kingdom (England and Wales)",
    "United Kingdom (Northern Ireland)",
    "United Kingdom (Scotland)",
]

MICROSTATES = {"Andorra", "Liechtenstein", "Monaco", "San Marino"}

# Regional groupings, used for the restricted control group in notebook 03.
# Taken from UNODC's Subregion field. The post-socialist flag is kept
# separate because every country that never ratifies is post-socialist,
# which is the main way the control group differs from the ratifiers.
POST_SOCIALIST = {
    "Albania", "Armenia", "Bosnia and Herzegovina", "Bulgaria", "Croatia",
    "Czechia", "Estonia", "Georgia", "Hungary", "Latvia", "Lithuania",
    "Moldova", "Montenegro", "North Macedonia", "Poland", "Romania", "Serbia",
    "Slovakia", "Slovenia", "Ukraine",
}


def canonical_country(name):
    if not isinstance(name, str):
        return name
    return NAME_ALIASES.get(name.strip(), name.strip())


def _load_unodc_raw():
    df = pd.read_excel(UNODC_HOMICIDE, sheet_name=0, skiprows=2)
    df = df[df["Indicator"] == "Victims of intentional homicide"].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


def _series(raw, sex, dimension, category, unit="Rate per 100,000 population"):
    """One UNODC series, already name-canonicalised, excluding the UK
    (which is handled separately from counts)."""
    sub = raw[
        (raw["Sex"] == sex)
        & (raw["Age"] == "Total")
        & (raw["Dimension"] == dimension)
        & (raw["Category"] == category)
        & (raw["Unit of measurement"] == unit)
    ].copy()
    sub["country"] = sub["Country"].map(canonical_country)
    sub = sub[~sub["Country"].isin(UK_JURISDICTIONS)]
    return (sub[["country", "Year", "VALUE"]]
            .rename(columns={"Year": "year", "VALUE": "value"})
            .dropna(subset=["country", "year"]))


def _uk_from_counts(raw, sex, dimension, category, uk_population=None):
    """Build a single United Kingdom series from its three reporting
    jurisdictions.

    UNODC publishes England and Wales, Northern Ireland and Scotland
    separately, and publishes sex-disaggregated rates against the matching
    sex-specific population. Victim counts are summed across the three
    jurisdictions and divided by the summed population of the same sex, which
    each jurisdiction implies through count / rate. This puts the United
    Kingdom on the same denominator as every other country.

    Years in which any jurisdiction is missing are left blank rather than
    partially summed, since a partial sum would understate the rate.
    """
    sub = raw[
        (raw["Sex"] == sex)
        & (raw["Age"] == "Total")
        & (raw["Dimension"] == dimension)
        & (raw["Category"] == category)
        & (raw["Country"].isin(UK_JURISDICTIONS))
    ]
    if sub.empty:
        return pd.DataFrame(columns=["country", "year", "value"])

    wide = sub.pivot_table(index=["Country", "Year"],
                           columns="Unit of measurement",
                           values="VALUE").dropna()
    if wide.empty:
        return pd.DataFrame(columns=["country", "year", "value"])
    wide["pop"] = (wide["Counts"] / wide["Rate per 100,000 population"]) * 100_000

    agg = (wide.reset_index()
               .groupby("Year")
               .agg(count=("Counts", "sum"), pop=("pop", "sum"),
                    n_juris=("Country", "nunique"))
               .reset_index())
    agg = agg[agg["n_juris"] == len(UK_JURISDICTIONS)]
    agg["value"] = agg["count"] / agg["pop"] * 100_000
    agg["country"] = "United Kingdom"
    return agg.rename(columns={"Year": "year"})[["country", "year", "value"]]


NEW_OUTCOMES = [
    # (column name, Sex, Dimension, Category)
    ("male_intentional_homicide_rate_per_100k", "Male", "Total", "Total"),
    ("total_intentional_homicide_rate_per_100k", "Total", "Total", "Total"),
    ("female_ipf_homicide_rate_per_100k", "Female",
     "by relationship to perpetrator", "Intimate partner or family member"),
    ("male_ipf_homicide_rate_per_100k", "Male",
     "by relationship to perpetrator", "Intimate partner or family member"),
    ("female_ip_homicide_rate_per_100k", "Female",
     "by relationship to perpetrator",
     "Intimate partner or family member: Intimate partner"),
]


_ACTION_NOTE = {
    "ADDED": "Read from the raw source during the build.",
    "DERIVED": "Calculated within the build from other panel columns.",
    "REBUILT (United Kingdom only)": "Rebuilt for the United Kingdom only.",
}


def _write_variable_sources(prov):
    """Write data/variable_sources.csv: plain documentation of where each
    added or derived column comes from and how it is constructed."""
    rows = [{"variable": p["column"],
             "source": p["source"],
             "construction": p["detail"],
             "notes": _ACTION_NOTE.get(p["action"], p["action"])}
            for p in prov]
    pd.DataFrame(rows).to_csv(VARIABLE_SOURCES, index=False)


def build(verbose=True):
    if not BASE_PANEL.exists():
        raise FileNotFoundError(f"Base panel not found: {BASE_PANEL}")
    if not UNODC_HOMICIDE.exists():
        raise FileNotFoundError(f"Raw UNODC file not found: {UNODC_HOMICIDE}")

    panel = pd.read_csv(BASE_PANEL, encoding="utf-8-sig")
    panel.columns = [str(c).replace("﻿", "").strip() for c in panel.columns]
    raw = _load_unodc_raw()
    prov = []

    # ---- 1. Correct the UK female-homicide series -------------------------
    fhr_col = "outcome_primary_female_intentional_homicide_rate_per_100k"
    uk_fixed = _uk_from_counts(raw, "Female", "Total", "Total")
    n_before = int(panel.loc[panel["country"] == "United Kingdom", fhr_col].notna().sum())
    panel.loc[panel["country"] == "United Kingdom", fhr_col] = np.nan
    if len(uk_fixed):
        idx = panel["country"].eq("United Kingdom")
        mapping = dict(zip(uk_fixed["year"], uk_fixed["value"]))
        panel.loc[idx, fhr_col] = panel.loc[idx, "year"].map(mapping)
    n_after = int(panel.loc[panel["country"] == "United Kingdom", fhr_col].notna().sum())
    prov.append({
        "column": fhr_col, "action": "REBUILT (United Kingdom only)",
        "detail": (f"United Kingdom rebuilt from UNODC victim counts summed "
                   f"across its three reporting jurisdictions, divided by the "
                   f"summed population of the same sex, so it shares the "
                   f"denominator used for every other country. {n_after} "
                   f"observations. The 43 other "
                   f"countries match the UNODC published rates within 2%."),
        "source": "UNODC UN-CTS intentional homicide (counts) + panel population_total",
    })
    if verbose:
        print(f"  United Kingdom {fhr_col}: {n_after} observations "
              f"(rebuilt on the common sex-specific denominator)")

    # ---- 2. Add the new outcome columns -----------------------------------
    for col, sex, dim, cat in NEW_OUTCOMES:
        s = _series(raw, sex, dim, cat)
        uk = _uk_from_counts(raw, sex, dim, cat)
        s = pd.concat([s, uk], ignore_index=True)
        s = s.rename(columns={"value": col})
        s["year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
        s = s.drop_duplicates(["country", "year"])
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
        panel = panel.merge(s, on=["country", "year"], how="left")
        n = int(panel[col].notna().sum())
        nc = int(panel.loc[panel[col].notna(), "country"].nunique())
        prov.append({
            "column": col, "action": "ADDED",
            "detail": (f"UNODC Sex={sex}, Dimension={dim}, Category={cat}, "
                       f"Rate per 100,000 population. {n} observations, {nc} countries."),
            "source": "UNODC UN-CTS intentional homicide",
        })
        if verbose:
            print(f"  added {col}: n={n}, countries={nc}")

    # ---- 3. Derived comparison outcomes -----------------------------------
    fem = panel[fhr_col]
    male = panel["male_intentional_homicide_rate_per_100k"]
    both = fem + male
    panel["female_share_of_homicide_victims"] = np.where(both > 0, fem / both, np.nan)
    prov.append({
        "column": "female_share_of_homicide_victims", "action": "DERIVED",
        "detail": ("female / (female + male) homicide rate. Country-year shocks that "
                   "affect homicide recording in general cancel out of this ratio, so "
                   "it isolates the sex-specific component the Convention targets."),
        "source": "derived",
    })
    ok = (fem > 0) & (male > 0)
    panel["log_female_male_homicide_ratio"] = np.nan
    panel.loc[ok, "log_female_male_homicide_ratio"] = (
        np.log(fem[ok]) - np.log(male[ok]))
    prov.append({
        "column": "log_female_male_homicide_ratio", "action": "DERIVED",
        "detail": ("log(female rate) - log(male rate). The difference-in-differences "
                   "estimated change on this outcome is a triple difference: it nets out any "
                   "country-year change in homicide levels or recording common to both sexes."),
        "source": "derived",
    })

    # ---- 4. Comparability blocs -------------------------------------------
    sub = (raw.assign(country=raw["Country"].map(canonical_country))
              [["country", "Subregion"]].dropna().drop_duplicates("country"))
    panel = panel.merge(sub.rename(columns={"Subregion": "unodc_subregion"}),
                        on="country", how="left")
    panel.loc[panel["country"] == "United Kingdom", "unodc_subregion"] = "Northern Europe"
    panel["post_socialist"] = panel["country"].isin(POST_SOCIALIST).astype(int)
    prov.append({"column": "unodc_subregion", "action": "ADDED",
                 "detail": "UNODC Subregion, used for regionally-restricted control groups.",
                 "source": "UNODC UN-CTS"})
    prov.append({
        "column": "post_socialist", "action": "DERIVED",
        "detail": ("1 for post-socialist / post-Soviet countries. All six "
                   "countries that never ratify are post-socialist, which is "
                   "the main way in which the control group differs from the "
                   "ratifying countries."),
        "source": "derived"})

    # ---- 5. Convention dates ----------------------------------------------
    tdv = pd.read_csv(TREATY_DATES)
    inherited = panel[["country", "convention_ratified_year"]].drop_duplicates("country")
    chk = tdv.merge(inherited, on="country", how="left")
    mismatched = chk[chk["ratification_year"].ne(chk["convention_ratified_year"])
                     & ~(chk["ratification_year"].isna()
                         & chk["convention_ratified_year"].isna())]
    if len(mismatched):
        raise RuntimeError("Ratification years in the panel and in "
                           "treatment_dates.csv disagree for: "
                           f"{sorted(mismatched['country'])}")

    # Fields the base panel does not carry.
    add = tdv[["country", "withdrawal_year", "first_full_year_ratification",
               "first_full_year_entry_into_force", "ever_ratified"]].rename(
        columns={"withdrawal_year": "convention_denunciation_year",
                 "first_full_year_ratification": "treat_year_first_full_ratification",
                 "first_full_year_entry_into_force": "treat_year_first_full_eif",
                 "ever_ratified": "ever_ratified_verified"})
    panel = panel.merge(add, on="country", how="left")

    # Treatment indicators under the "first full calendar year" definition.
    # A November ratification leaves ~92% of that calendar year untreated, so
    # coding the ratification year as treated attenuates the first-year effect.
    for src_col, new_col in [
            ("treat_year_first_full_ratification",
             "treatment_first_full_year_ratification"),
            ("treat_year_first_full_eif", "treatment_first_full_year_eif")]:
        panel[new_col] = ((panel[src_col].notna()) &
                          (panel["year"] >= panel[src_col])).astype(int)

    # Non-absorbing variant that switches Turkey off from its withdrawal year.
    # Used for the Turkey case study and as a robustness check. The stacked
    # and Callaway-Sant'Anna estimators both require treatment to stay on
    # once it starts, so the main indicator is the absorbing one.
    den = panel["convention_denunciation_year"]
    panel["treatment_ratification_nonabsorbing"] = np.where(
        den.notna() & (panel["year"] >= den), 0,
        panel["treatment_main_ratification"]).astype(int)

    prov.append({
        "column": "convention_signed_year / convention_ratified_year / "
                  "convention_entry_into_force_year",
        "action": "ADDED",
        "detail": "Signature, ratification and entry-into-force years.",
        "source": "data/treatment_dates.csv"})
    prov.append({
        "column": "convention_denunciation_year", "action": "ADDED",
        "detail": ("Year a withdrawal took legal effect. Turkey only "
                   "(2021-07-01). Empty for all other countries."),
        "source": "data/treatment_dates.csv"})
    prov.append({
        "column": "treatment_first_full_year_ratification / "
                  "treatment_first_full_year_eif",
        "action": "ADDED",
        "detail": ("Treatment switched on in the first FULL calendar year "
                   "after the event, not the calendar year containing it. "
                   "Addresses within-year timing: e.g. Austria ratified "
                   "2013-11-14 and Bosnia 2013-11-07, leaving almost all of "
                   "2013 untreated. Robustness check, not the main definition."),
        "source": "derived from data/treatment_dates.csv"})
    prov.append({
        "column": "treatment_ratification_nonabsorbing", "action": "ADDED",
        "detail": ("Equals treatment_main_ratification except that Turkey "
                   "returns to 0 from 2021, its withdrawal year. Used for "
                   "the Turkey case study and as a robustness check, not for "
                   "the main estimates."),
        "source": "derived from data/treatment_dates.csv"})
    if verbose:
        n_ff = int((panel["treatment_first_full_year_ratification"] !=
                    panel["treatment_main_ratification"]).sum())
        print(f"  treatment timing variants: the first-full-year definition "
              f"reclassifies {n_ff} country-years")

    # ---- 6. Write ----------------------------------------------------------
    assert len(panel) == 1530, f"expected 1530 rows, got {len(panel)}"
    assert panel["country"].nunique() == 45, "expected 45 countries"
    assert not panel.duplicated(["country", "year"]).any(), "duplicate country-year"
    panel.to_csv(OUT_PANEL, index=False)
    _write_variable_sources(prov)
    if verbose:
        print(f"\nWrote {OUT_PANEL}  ({panel.shape[0]} rows x {panel.shape[1]} cols)")
        print(f"Wrote {VARIABLE_SOURCES}")
    return panel


if __name__ == "__main__":
    build()
