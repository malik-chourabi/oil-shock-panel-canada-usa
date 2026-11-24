# Oil Price Shocks & Macroeconomic Effects: Canada & USA (2023–2026)

Panel-data econometrics project for **Advanced Econometrics** (Prof. Naceur Khraief, academic year 2025–2026).
We estimate how the **2026 Gulf Crisis oil price shock** (Brent: ~$70.9 → ~$103.1 per barrel, +45.5% in March 2026) affected
**inflation (CPI)** and **industrial production (IPI)** in Canada and the United States, using Pooled OLS and Fixed Effects
with Arellano HC3 robust standard errors, plus ARIMA(1,1,1) forecasts to complete the panel through June 2026.

**Authors:** Khadija Slaoui · Jihen Cherif · Emna Jaouadi · Chayma Lessis · Malik Chourabi

## Key findings

| Result | Finding |
|---|---|
| Post-crisis effect on CPI | CPI about **4.1% higher** in the crisis period (FE, β = 0.041, p < 0.001); the most robust result across specifications |
| Oil shock → CPI | Small negative coefficient (β ≈ −0.0003), consistent with net-exporter income effects offsetting cost-push inflation |
| Oil shock → IPI | No significant contemporaneous effect |
| Pooled OLS vs panel | Breusch-Pagan LM rejects pooling (χ² = 278.6, p < 2.2e-16); FE R² 0.647 vs POLS 0.253 |
| Serial correlation | Wooldridge test rejects no-autocorrelation (p = 0.0012), so robust SEs are used throughout |

**Known limitations (see report, Section 12):** with only N = 2 countries, Random Effects and the Hausman test cannot be estimated,
cluster-robust inference is not valid, and both countries are net exporters, so the importer-vs-exporter comparison is
replaced by a Canada-vs-USA comparison. April–June 2026 values for CPI, IPI, NEER and the policy rate are ARIMA forecasts, not observed data.

## Repository structure

```
.
├── data/
│   └── panel_dataset.csv            # Balanced panel: 2 countries × 42 months = 84 rows, 17 columns
├── R/
│   └── oil_shock_panel_analysis.R   # Full analysis: EDA, POLS/FE, diagnostics, asymmetry, heterogeneity, plots
├── python/
│   └── replicate_main_results.py    # Quick cross-check of descriptive stats + POLS/FE coefficients
├── figures/                         # Figures 1–5 used in the report
├── report/
│   ├── Final_Project_Report.docx    # Full written report
│   └── Tables_and_Graphs.docx       # Tables, graphs and R output interpretations
├── outputs/                         # Generated tables/logs/plots land here
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data

`data/panel_dataset.csv`: monthly, January 2023 – June 2026, Canada and USA.

| Variable | Description |
|---|---|
| `country`, `date` | Panel identifiers (YYYY-MM-DD) |
| `cpi`, `ipi` | Consumer price index; industrial production index (2015 = 100) |
| `neer` | Nominal effective exchange rate |
| `policy_rate` | Central bank policy rate (%) |
| `brent` | Brent crude (USD/barrel), global |
| `vix` | CBOE VIX, global |
| `ln_cpi`, `ln_ipi`, `ln_neer` | Natural logs |
| `oil_pct_change` | Monthly % change in Brent |
| `oil_shock_pos`, `oil_shock_neg` | Positive / negative parts of the oil change (asymmetry analysis) |
| `post_crisis` | 1 from March 2026 onward |
| `oil_balance`, `country_group` | Classification (all rows: exporter / advanced_exporter) |

Sources cited in the report: IMF IFS, OECD, Statistics Canada, BIS, U.S. EIA, FRED. April–June 2026 values for several series are ARIMA forecasts (see Report, Section 4).

## Reproducing the results

Run everything **from the repository root** so relative paths resolve.

**R (main analysis)**
```r
install.packages(c("tidyverse", "plm", "lmtest", "sandwich", "stargazer", "car", "forecast"))
```
```bash
Rscript R/oil_shock_panel_analysis.R
```
Writes the log, regression tables (`Table0`–`Table5`) and plots (`Plot1`–`Plot5`) to `outputs/`.

**Python (cross-check)**
```bash
pip install -r requirements.txt
python python/replicate_main_results.py
```
Coefficients match the report; standard errors differ slightly because statsmodels HC3 is not plm's Arellano HC3.

## License

Code is released under the MIT License (see `LICENSE`). The report and data are academic coursework; please cite the authors if you reuse them.
