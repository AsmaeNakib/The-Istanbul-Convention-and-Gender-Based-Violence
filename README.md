# International Policy and Gender-Based Violence: The Istanbul Convention in Europe

Asmae Nakib

Constructor University

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
serves as a negative control, and the log female/male ratio as a within-country
comparison. Recorded sexual violence, domestic violence legislation, GREVIO
monitoring and shelter capacity are used to describe implementation.

The headline result is that no estimator detects an additional decline in
female homicide after ratification, and that the design's minimum detectable
effect is large relative to the effects the literature would predict. The
manuscript reports this as a limit on what the data can establish, not as
evidence of no effect.

**Manuscript files.** `thesis_draft_v8.md` is the source of the thesis text.
`thesis_draft_v8.tex` is generated from it by `build_tex.py`, so the two always
carry identical substantive content. Edit the Markdown, then run
`python build_tex.py`.

---

## Data

All raw files are in `data/raw/` exactly as obtained. Nothing in the analysis
reads from outside that folder.

| Source | File(s) | Used for |
| --- | --- | --- |
| UNODC, Victims of Intentional Homicide (UN-CTS) | `data_cts_intentional_homicide.xlsx` | female and male homicide rates; intimate-partner/family homicide |
| UNODC, Violent and Sexual Crime (UN-CTS) | `data_cts_violent_and_sexual_crime*.xlsx` | recorded sexual violence and rape |
| UNODC & UN Women femicide estimates | `Femicide_Data.xlsx`, `Raw Data from Underlying Data Sources (1996-2024).xlsx` | descriptive context |
| World Bank, Women, Business and the Law | `WBL_Historical_Panel_Data.xlsx`, `WBL2024 Safety data_Website.xlsx`, `WBL2024 methodology handbook` | domestic violence legislation, femicide law indicators |
| World Bank, World Development Indicators | `API_NY_GDP_PCAP_CD.csv`, `API_SL_TLF_CACT_FE_ZS.csv` + metadata | GDP per capita, female labour force participation |
| WAVE Network country report | `WAVE-Country-Report-2025.pdf` | shelter capacity |
| GREVIO | `PREMS 050325 ... Rapport Grevio ...pdf` | monitoring and baseline evaluation years |
| Council of Europe Treaty Office | transcribed in `src/make_treatment_dates.py` | signature, ratification, entry-into-force and withdrawal dates |

Treaty dates were transcribed by hand from the Council of Europe registry and
are stored in `data/treatment_dates_verified.csv` with a per-country note.
`data/analysis_panel_provenance.csv` records, for every column in the analysis
panel, where it came from and how it was derived.

---

## Guide to the CSV files

| File | Type | Description | Used for |
| --- | --- | --- | --- |
| `data/raw/API_NY_GDP_PCAP_CD.csv` | source | World Bank GDP per capita, current US$ | covariate, balance table |
| `data/raw/API_SL_TLF_CACT_FE_ZS.csv` | source | World Bank female labour force participation | covariate, balance table |
| `data/raw/Metadata_*.csv` | source | World Bank country and indicator metadata | country-name harmonisation |
| `data/interim/base_country_year_panel.csv` | cleaned | merged country-year frame before derived columns are added | intermediate build step |
| `data/gbv_panel_analysis.csv` | processed | the analysis panel: outcomes, treatment indicators, covariates | every estimate in the thesis |
| `data/treatment_dates_verified.csv` | source | signature, ratification, entry-into-force and withdrawal dates with per-country notes | treatment coding, all timing variants |
| `data/analysis_panel_provenance.csv` | reference | column-level build log for the analysis panel | reproducibility audit |

---

## Methods

- **Two-way fixed effects DiD** — country and year fixed effects, standard
  errors clustered by country. Baseline specification.
- **Stacked DiD** — one clean event window per ratification cohort, so an
  already-ratified country never serves as a comparison.
- **Callaway and Sant'Anna group-time ATT** — reported against both
  not-yet-treated and never-treated comparison groups.
- **Event studies** — relative-time coefficients with binned endpoints and
  contributing-country counts reported alongside.
- **Falsification** — male homicide as a negative control and the log
  female/male ratio as a triple difference.
- **Parallel-trends sensitivity** — Rambachan and Roth (2023) bounds.
- **Inference** — cluster-robust, wild cluster bootstrap (Webb weights, 999
  replications) and a permutation test (999 reassignments); all three are
  reported rather than reconciled. Holm correction within the confirmatory
  family.
- **Power** — minimum detectable effect at 80% power, expressed as a share of
  the treated mean.
- **Placebo timing** — treatment shifted four years earlier, screened at
  p = 0.10.
- **Synthetic control** — used only as a donor-pool feasibility diagnostic
  (pre-period RMSPE against the outcome's own standard deviation), not as an
  effect estimate.

An outcome-by-method permissions table
(`outputs/diagnostics/outcome_method_permissions.csv`) fixes which estimators
each outcome may support, and is written before any estimate is produced.

---

## Reproducibility

```bash
pip install -r requirements.txt      # Python 3.13.3

python src/make_treatment_dates.py   # data/treatment_dates_verified.csv
python src/build_analysis_panel.py   # data/gbv_panel_analysis.csv

# then, in order:
jupyter lab notebooks/01_data_design_diagnostics.ipynb
jupyter lab notebooks/02_main_analysis.ipynb
jupyter lab notebooks/03_case_studies_mechanisms.ipynb
jupyter lab notebooks/04_robustness_appendix.ipynb

python viz/make_thesis_figures.py    # regenerates every file in Figures/
python build_tex.py                  # regenerates thesis_draft_v8.tex
```

The notebooks must run in order: `01` writes the sample inventory and the
outcome-method permissions that `02`–`04` read. Every estimation step is seeded
(`20260804`, recorded in `outputs/diagnostics/runtime.txt`), so the bootstrap
and permutation p-values reproduce exactly.

All four notebooks were re-run end to end against this repository state and the
reported numbers reproduce. Package versions actually used are recorded in
`outputs/diagnostics/software_manifest.csv`; `csdid` 0.2.9 supplies the
Callaway–Sant'Anna estimator, and `pyfixest` and `linearmodels` are not
required.
