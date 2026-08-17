# International Policy and Gender-Based Violence: The Istanbul Convention in Europe

**Author** Asmae Nakib · **Degree** Master Thesis · **School** School of
Business, Social & Decision Sciences, Constructor University ·

---

## Project overview

The study asks whether ratifying the Council of Europe Convention on
Preventing and Combating Violence against Women and Domestic Violence (the
Istanbul Convention) is followed by a measurable change in gender-based
violence (GBV) outcomes in Europe.

The unit of analysis is the country-year. The panel covers 45 European
countries; the main estimation sample runs from 2000 to 2023. Treatment is the
calendar year in which a country ratified the Convention, dated from the
Council of Europe Treaty Office record; entry into force and first-full-year
codings are carried as alternative timings. Turkey is the panel's only
withdrawal, effective 1 July 2021; its 2021–2023 rows are dropped from the main
samples and restored as a robustness check.

The primary outcome is the female intentional homicide rate. Male homicide
serves as an additional outcome for comparison, and the log female/male ratio
as a within-country comparison. Recorded sexual violence, domestic violence legislation, GREVIO
monitoring and shelter capacity are used to describe implementation.

The main result is that no method finds a statistically significant decline in
female homicide after ratification, and that the design's minimum detectable
effect is large relative to the effects the literature would predict. The
manuscript reports this as a limit on what the data can establish, not as
evidence of no effect.

**Manuscript files.** `thesis text final.md` is the source of the thesis text.
`thesis text final.tex` is generated from it by `build_thesis.py`, so the two
always carry identical substantive content. Edit the Markdown, then run
`python build_thesis.py`.

---

## Data

The analysis runs on three files, all included here: `data/gbv_panel_analysis.csv`
(the panel every notebook reads), `data/treatment_dates.csv` and
`variable_sources.csv`.

### Files included in this repository

| File | Read by | Used for |
| --- | --- | --- |
| `gbv_panel_analysis.csv` | all four notebooks | every estimate in the thesis |
| `treatment_dates.csv` | `build_analysis_panel.py`, `make_thesis_figures.py` | signature, ratification, entry-into-force and withdrawal dates |
| `variable_sources.csv` | notebook `01` | source and construction of every built variable |
| `base_country_year_panel.csv` | `build_analysis_panel.py` | merged country-year frame the panel build starts from |
| `data_cts_intentional_homicide.xlsx` | `build_analysis_panel.py` | female and male homicide rates, intimate-partner/family homicide, and the United Kingdom rebuild |
| `WBL2024 Safety data_Website.xlsx` | consulted directly | the domestic-violence legal criterion described in Section 5.7.1 |

`data_cts_intentional_homicide.xlsx` is the only raw file any script opens.

### Sources already merged into the interim panel

The remaining sources were harmonised into
`data/interim/base_country_year_panel.csv` and are not re-hosted here. Download
them from the publishers if you want to rebuild that file from scratch.

| Source | Used for | Where to obtain it |
| --- | --- | --- |
| UNODC, Violent and Sexual Crime (UN-CTS) | recorded sexual violence and rape | <https://dataunodc.un.org/> |
| UNODC & UN Women femicide estimates | descriptive context | <https://dataunodc.un.org/> |
| World Bank, Women, Business and the Law | domestic-violence legislation and gender-related-killing indicators | <https://wbl.worldbank.org/> |
| World Bank, World Development Indicators | population, GDP per capita, female labour force participation | <https://data.worldbank.org/> |
| Worldwide Governance Indicators | government effectiveness, rule of law | <https://www.worldbank.org/en/publication/worldwide-governance-indicators> |
| WAVE Network country report | shelter capacity | <https://wave-network.org/> |
| GREVIO baseline evaluation reports | first baseline evaluation years | <https://www.coe.int/en/web/istanbul-convention/grevio> |

Treaty dates come from the Council of Europe registry for CETS No. 210 and are
stored in `data/treatment_dates.csv`. `data/variable_sources.csv` documents, for
every variable added or derived during the build, its source, how it was
constructed, and any note attached to it.

Sex-disaggregated UNODC homicide rates are per 100,000 of the population of that
sex: female homicide is per 100,000 women and male homicide per 100,000 men.
The United Kingdom, which UNODC reports as three separate jurisdictions, is
rebuilt on the same basis in `src/build_analysis_panel.py`.

---

## Methods

- **Two-way fixed effects DiD** — country and year fixed effects, standard
  errors clustered by country. Baseline specification.
- **Stacked DiD** — one clean event window per ratification cohort, so an
  already-ratified country never serves as a comparison.
- **Callaway and Sant'Anna group-time ATT** — reported against both
  not-yet-treated and never-treated control groups.
- **Event studies** — relative-time coefficients with binned endpoints and
  contributing-country counts reported alongside.
- **Male homicide comparison** — male homicide as an additional outcome, and
  the log female/male ratio, which removes changes affecting both sexes.
- **Parallel-trends sensitivity** — Rambachan and Roth (2023) bounds.
- **Inference** — cluster-robust, wild cluster bootstrap (Webb weights, 999
  replications) and a permutation test (999 reassignments); all three are
  reported rather than reconciled. Holm correction within the confirmatory
  family.
- **Power** — minimum detectable effect at 80% power, expressed as a share of
  the treated mean.
- **Placebo timing** — treatment shifted four years earlier, screened at
  p = 0.10.
- **Synthetic control** — used only as a feasibility check on the control
  countries (pre-period RMSPE against the outcome's own standard deviation),
  not as an effect estimate.

A table of permitted methods
(`outputs/diagnostics/outcome_method_permissions.csv`) records which estimators
each outcome supports, and is written before any estimate is produced.

---

## Main findings

- Two-way fixed effects gives +0.302 deaths per 100,000 women (SE 0.200,
  p = 0.140, N = 858). Stacked DiD gives +0.108 (SE 0.114, p = 0.349) and
  Callaway–Sant'Anna +0.117 against not-yet-treated countries (SE 0.173,
  p = 0.477). None is statistically significant.
- The minimum detectable effect for female homicide at 80% power is 0.56 deaths
  per 100,000 women, about 56% of the treated mean. The design cannot rule out
  the smaller changes the literature would predict.
- Male homicide moves in the same direction (+0.756, p = 0.168) and the log
  female/male ratio is flat (−0.054, p = 0.511), so the female series does not
  move distinctively.
- The result is not stable against the composition of the control group:
  dropping Lithuania alone moves the estimate by roughly half.
- A later-ratifier comparison that removes the permanent non-ratifiers
  altogether, and compares the 2013–2019 cohorts only with countries ratifying
  after each cohort's window, gives +0.007 (p = 0.970). The magnitude depends on
  the counterfactual; the qualitative reading does not.
- The three inference procedures disagree (p = 0.140, 0.170, 0.041); the
  disagreement is reported as a feature of a design with a small, structurally
  distinct control group.
- Implementation indicators diverge sharply from treaty status, which is the
  substantive point: a binary ratification indicator carries limited
  information about what a country has actually put in place.

---
