
# Lab 20: Time Series Diagnostics & Advanced Decomposition

**Course:** ECON 5200 — Causal Machine Learning & Applied Analytics  
**Author:** Umang Rayamajhi  
**Tools:** Python · statsmodels · FRED API · ruptures · NumPy · pandas · Matplotlib

---

## Overview

This lab takes a diagnosis-first approach to time series analysis. Rather than applying methods to clean data, each part presents **deliberately flawed code or reasoning** — the objective is to identify the bug, explain why it produces wrong output, fix it, and extend the corrected analysis. The lab covers STL decomposition, stationarity testing, multi-seasonal decomposition, block bootstrap uncertainty quantification, and structural break detection.
              
---

## Part-by-Part Summary

### Part 1 — Diagnose & Fix: Broken STL Decomposition

**Bug:** STL (an additive model) was applied directly to FRED retail sales (`RSXFSN`), which has multiplicative seasonality — the seasonal amplitude grows proportionally with the level of the series.

**Why it fails:** Additive STL assumes the seasonal component is a constant absolute offset. When the true structure is multiplicative (Y = Trend × Seasonal × Error), applying additive STL causes the growing seasonal signal to leak into the residual, producing a distorted decomposition where seasonal amplitude increases over time.

**Fix:** Log-transform the series before applying STL. This converts the multiplicative structure to additive (log Y = log T + log S + log E), making STL appropriate.

**Verification:** The ratio of latest-to-earliest seasonal amplitude drops from >3x to within the 0.7–1.3 range after the fix.

```python
log_retail = np.log(retail)
stl_fixed = STL(log_retail, period=12, robust=True).fit()
```

---

### Part 2 — Diagnose & Fix: Misspecified ADF Test

**Bug:** The Augmented Dickey-Fuller test was run with `regression='n'` (no constant, no trend) on Real GDP (`GDPC1`), which has both a non-zero mean and a clear upward deterministic trend.

**Why it fails:** Omitting the constant and trend from the ADF regression misspecifies the model. The test statistic is inflated, and the test can falsely reject the unit root null — incorrectly concluding that GDP is stationary when it is not.

**Fix:** Use `regression='ct'` (constant + linear trend) to correctly account for GDP's deterministic components. Complement with a KPSS test and apply the 2×2 decision table.

**Result:** ADF fails to reject the unit root (p > 0.05); KPSS rejects stationarity (p < 0.05) → GDP is **I(1), non-stationary**.

```python
adf_stat, adf_p, *_ = adfuller(gdp, regression='ct')
kpss_stat, kpss_p, *_ = kpss(gdp, regression='ct', nlags='auto')
```

| ADF rejects H₀ | KPSS rejects H₀ | Verdict |
|:-:|:-:|:-:|
| No | Yes | **Non-stationary** |
| Yes | No | Stationary |
| Yes | Yes | Contradictory |
| No | No | Inconclusive |

---

### Part 3 — Extend: MSTL for Multiple Seasonal Periods

Standard STL handles only one seasonal period. Real-world time series — like hourly electricity demand — often have multiple overlapping cycles. `MSTL` (Multiple STL) decomposes all of them simultaneously.

**Simulation:** 6 months of hourly electricity demand with:
- A slow linear trend (growing demand)
- A **daily cycle** (period = 24h) — demand peaks at noon
- A **weekly cycle** (period = 168h) — demand lower on weekends
- Gaussian noise (σ = 15 MW)

**Result:** MSTL cleanly separates both seasonal components. Residual standard deviation recovers the true noise level (~15 MW), confirming accurate decomposition.

```python
mstl = MSTL(demand_series, periods=[24, 168])
mstl_result = mstl.fit()
```

To add an annual cycle: `MSTL(series, periods=[24, 168, 8760])`

---

### Part 4 — Extend: Block Bootstrap for Trend Uncertainty

STL trend extraction produces a single curve — but how much should you trust it? The **moving block bootstrap** quantifies trend uncertainty by resampling overlapping blocks of residuals (preserving their autocorrelation), reconstructing synthetic series, re-running STL on each, and computing pointwise confidence bands.

**Why block, not i.i.d.?** Macro residuals are serially correlated. Shuffling observations independently (i.i.d. bootstrap) destroys that structure and underestimates uncertainty. Sampling contiguous blocks preserves within-block autocorrelation.

**Configuration:** `block_size=8` (2 years of quarterly data), `n_bootstrap=200`, 90% pointwise confidence bands.

**Key finding:** The confidence band widens around the 2008–09 recession and the 2020 COVID shock, where residual volatility is highest.

---

### Part 5 — Extend: Structural Break Detection + Per-Regime Stationarity

Combines **PELT** (Pruned Exact Linear Time) structural break detection with ADF and KPSS tests run independently on each detected regime.

**PELT** minimizes a penalized cost function over the series to find changepoints in mean and/or variance. The penalty parameter controls the bias-variance tradeoff: higher penalty → fewer breaks.

**Finding:** GDP growth segments are identified near major macroeconomic turning points (early 1980s, 2008 financial crisis, 2020 pandemic). Most segments are stationary in growth rates, confirming that GDP is I(1) — its *levels* are non-stationary, but *first differences* are stationary.

```python
algo = rpt.Pelt(model='rbf').fit(signal)
breakpoints = algo.predict(pen=10)
```

---

### Part 6 — Production Module: `decompose.py`

A reusable, portfolio-grade Python module with full docstrings, type hints, and error handling.

| Function | Description |
|---|---|
| `run_stl(series, period, log_transform, robust)` | STL decomposition with optional log-transform for multiplicative data |
| `test_stationarity(series, alpha)` | ADF + KPSS with 2×2 decision table verdict |
| `detect_breaks(series, pen)` | PELT structural break detection, returns break dates |

```python
from decompose import run_stl, test_stationarity, detect_breaks

result  = run_stl(retail, period=12, log_transform=True)
verdict = test_stationarity(gdp)          # → {'verdict': 'non-stationary', ...}
breaks  = detect_breaks(gdp_growth, pen=10)
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/your-username/econ-lab-20-time-series.git
cd econ-lab-20-time-series
```

### 2. Install dependencies
```bash
pip install fredapi statsmodels ruptures numpy pandas matplotlib
```

### 3. Add your FRED API key

Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html), then replace the placeholder in the notebook:

```python
FRED_API_KEY = 'your_key_here'
```

### 4. Run in Colab

Upload `lab-ch20-diagnostic.ipynb` to [Google Colab](https://colab.research.google.com) and run all cells top to bottom.

---

## Key Takeaways

- **Always check additive vs. multiplicative structure** before applying STL. A growing seasonal amplitude is a clear signal to log-transform first.
- **ADF test specification matters.** Using `regression='n'` on a trended series is a common and consequential mistake. Always match the regression deterministics to the data's actual structure.
- **Combine ADF and KPSS** — each has a different null, so together they give a clearer picture than either alone.
- **Block bootstrap > i.i.d. bootstrap** for autocorrelated time series. Residuals from macro data carry serial dependence that must be preserved during resampling.
- **Stationarity conclusions can change across regimes.** PELT + per-regime testing reveals that GDP growth behaves differently in different economic eras.

---

## Data Sources

All series retrieved from the [Federal Reserve Bank of St. Louis (FRED)](https://fred.stlouisfed.org/):

| Series | Description | Frequency |
|---|---|---|
| `RSXFSN` | Retail & Food Services Sales (Not Seasonally Adjusted) | Monthly |
| `GDPC1` | Real Gross Domestic Product | Quarterly |

---

