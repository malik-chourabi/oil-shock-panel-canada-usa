"""Python cross-check of the main results (Pooled OLS and Fixed Effects).

The primary analysis is in R/oil_shock_panel_analysis.R. This script gives
reviewers without R a quick way to reproduce the headline numbers.

Notes
- Coefficients should match the report. Standard errors are statsmodels HC3
  (country-dummy / LSDV form), not plm's Arellano HC3, so SEs and p-values
  differ somewhat. R-squared for FE here is the LSDV R-squared, not the
  within R-squared reported by plm.
- Following the report, the IPI model excludes January 2023 (N = 82), because
  the report treats the first monthly oil change as unavailable.

Usage (from the repository root):
    python python/replicate_main_results.py
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "panel_dataset.csv", parse_dates=["date"])
print(f"Panel: {df.shape[0]} rows, {df.country.nunique()} countries, "
      f"{df.date.min():%b %Y} to {df.date.max():%b %Y}")

# ---- Descriptive statistics -------------------------------------------------
cols = ["cpi", "ln_cpi", "ipi", "ln_ipi", "brent", "oil_pct_change",
        "ln_neer", "policy_rate", "vix"]
desc = df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc.to_csv(OUT / "py_descriptive_stats.csv")
print("\nDescriptive statistics (pooled):\n", desc.round(4))

# ---- Models -----------------------------------------------------------------
X = "oil_pct_change + ln_neer + policy_rate + vix + post_crisis"
rows = []
for dep in ["ln_cpi", "ln_ipi"]:
    sample = df[df.date > "2023-01-01"] if dep == "ln_ipi" else df
    # Pooled OLS and Fixed Effects (country dummies == within estimator)
    for name, f in [("Pooled OLS", f"{dep} ~ {X}"),
                    ("Fixed Effects", f"{dep} ~ {X} + C(country)")]:
        m = smf.ols(f, data=sample).fit(cov_type="HC3")
        print(f"\n{dep} | {name} | N={int(m.nobs)} | R2={m.rsquared:.3f}")
        for v in X.split(" + "):
            rows.append(dict(dependent=dep, model=name, variable=v,
                             coef=m.params[v], robust_se=m.bse[v],
                             p_value=m.pvalues[v], n=int(m.nobs)))
            print(f"  {v:15s} {m.params[v]: .6f} (se {m.bse[v]:.6f}) p={m.pvalues[v]:.3f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "py_model_results.csv", index=False)
print(f"\nSaved results to {OUT}/")
