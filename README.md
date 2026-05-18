# Monetary Policy Transmission to the Indian Yield Curve

## Project Overview

This project investigates the **transmission mechanisms of monetary policy shocks to India's yield curve** using a comprehensive econometric framework combining:

- **Dynamic Nelson-Siegel (DNS) term structure modeling** to extract level, slope, and curvature factors
- **Structural Vector Autoregression (SVAR)** with recursive identification for shock decomposition
- **Impulse Response Analysis (IRF)** and Forecast Error Variance Decomposition (FEVD)
- **Granger causality tests** and channel analysis for policy transmission

The analysis uses monthly and daily panel data on Indian Government Security (GSY) yields at multiple maturities (1Y, 3Y, 5Y, 10Y) and the RBI repo rate from 2006 onwards.

---

## Folder Structure

```
Eco Paper/
├── README.md                 # This file
├── Preprocessed/             # Source scripts and raw data
│   ├── preprocessing.py      # Data cleaning and panel construction
│   ├── DNS.py               # Term structure factor extraction
│   ├── SVAR.py              # Shock identification and IRF estimation
│   ├── Hypothesis.py        # Statistical hypothesis testing
│   ├── Transmission.py      # Channel decomposition analysis
│   └── *.csv                # Raw and processed bond yield data
├── data/                     # Output datasets (analysis-ready)
│   ├── monthly_panel.csv     # Main dataset for DNS + SVAR (228 obs)
│   ├── daily_panel.csv       # Event study dataset
│   ├── summary_stats.csv     # Descriptive statistics (paper-ready)
│   ├── stationarity_report.csv # Unit root tests (ADF, PP)
│   ├── dns_factors.csv       # Extracted term structure factors
│   ├── dns_fit_stats.csv     # Model fit diagnostics
│   ├── svar_*.csv            # VAR coefficients, shocks, IRF, FEVD
│   ├── h0_*.csv              # Test statistics for hypotheses
│   ├── channel_decomp.csv    # Policy transmission channels
│   └── subperiod_results.csv # Pre/post-2008 crisis comparison
└── plots/                    # Visualizations
    ├── yield_series.png      # Time series of yields
    ├── dns_factors.png       # Level, slope, curvature evolution
    ├── svar_irf_*.png        # Impulse response functions
    ├── svar_fevd.png         # Forecast error variance shares
    └── *.png                 # Additional diagnostics
```

---

## Methodology

### 1. Data Preprocessing (`Preprocessed/preprocessing.py`)

**Inputs:**
- Indian GSY yield series: 1Y, 3Y, 5Y, 10Y maturities
- RBI repo rate

**Processing:**
- Standardize dates across all series
- Aggregate to monthly frequency (panel format)
- Create daily event-study dataset (weekdays only)
- Compute first differences for stationarity
- Unit root testing (ADF, Phillips-Perron)

**Outputs:**
- `monthly_panel.csv` – main analysis dataset (228 monthly observations)
- `daily_panel.csv` – event-study dataset
- Diagnostics: stationarity tests, correlation matrices, summary statistics
- Plots: yield levels, changes, correlations

---

### 2. Term Structure Modeling (`Preprocessed/DNS.py`)

**Method:** Dynamic Nelson-Siegel Model (Diebold & Li 2006)

The yield curve is decomposed into three latent factors at each month $t$:

$$y_t(\tau) = \beta_{1t} + \beta_{2t} \frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_{3t} \left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right)$$

where:
- $\beta_{1t}$ = **Level** (long-term yields)
- $\beta_{2t}$ = **Slope** (short-end premium)
- $\beta_{3t}$ = **Curvature** (medium-term hump)
- $\lambda$ = Decay parameter (fixed at monthly frequency)

Each factor evolves as an AR(1) process, allowing for mean reversion and persistence.

**Outputs:**
- `dns_factors.csv` – monthly factor time series
- `dns_fit_stats.csv` – R², RMSE, AR(1) autocorrelations
- Plots: factor evolution, fitted vs. actual curves, ACF diagnostics

---

### 3. Monetary Policy Shock Identification (`Preprocessed/SVAR.py`)

**Method:** Structural VAR with Recursive Identification

A reduced-form VAR(p) is estimated on first-differenced yields and repo rate:

$$\Delta Z_t = c + A_1 \Delta Z_{t-1} + \cdots + A_p \Delta Z_{t-p} + u_t$$

where $Z_t = [\text{Repo}_t, Y1_t, Y3_t, Y5_t, Y10_t]'$

**Structural shocks** are recovered via Cholesky decomposition of the residual covariance:

$$u_t = P \varepsilon_t \quad \text{with} \quad P = \text{chol}(\Sigma_u)$$

**Ordering Justification:** Repo rate ordered first (RBI sets policy before bond market reaction within a month). This recursive identification follows Christiano, Eichenbaum & Evans (1999).

**Outputs:**
- `svar_var_coefficients.csv` – reduced-form VAR matrices
- `svar_residuals.csv` – structural shocks $\varepsilon_t$ (orthogonal to yield information)
- `svar_irf.csv` – point estimates of impulse response functions (all shock-response pairs)
- `svar_irf_ci.csv` – bootstrapped 90% confidence bands
- `svar_granger.csv` – Granger causality test statistics
- `svar_fevd.csv` – forecast error variance decomposition (48-month horizon)
- Plots: IRF repo shock, full IRF grid, FEVD, residual diagnostics

---

### 4. Channel Analysis & Hypothesis Testing (`Preprocessed/Hypothesis.py`)

**Tests conducted:**
- **H0.1:** Granger causality – repo rate → each yield maturity
- **H0.2:** IRF significance – peak and cumulative responses
- **H0.3:** Channel identification – direct vs. indirect transmission paths

**Outputs:**
- `h0_1_granger.csv` – Granger causality F-statistics and p-values
- `h0_2_irf_significance.csv` – IRF significance levels
- `h0_3_channel_tests.csv` – channel decomposition results

---

### 5. Transmission Mechanism Decomposition (`Preprocessed/Transmission.py`)

**Analysis:**
- Time-varying transmission efficacy
- Maturity-specific responses
- Pre/post-crisis period comparison (2008 financial crisis)
- Subsample robustness (subperiod analysis)

**Outputs:**
- `channel_decomp.csv` – decomposed transmission effects
- `subperiod_results.csv` – crisis vs. normal times comparison

---

## How to Run the Analysis

### Prerequisites

Install required Python packages:

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn
```

### Step 1: Data Preprocessing

```bash
cd Preprocessed
python preprocessing.py
```

Generates: `data/monthly_panel.csv`, `data/daily_panel.csv`, and descriptive statistics.

### Step 2: Term Structure Factor Extraction

```bash
python DNS.py
```

Generates: `dns_factors.csv`, `dns_fit_stats.csv`, and DNS visualization plots.

### Step 3: SVAR Shock Identification & IRF Estimation

```bash
python SVAR.py
```

Generates: VAR coefficients, structural shocks, impulse responses, confidence bands, FEVD.

### Step 4: Hypothesis Testing

```bash
python Hypothesis.py
```

Generates: Granger causality tests, IRF significance, channel test results.

### Step 5: Transmission Channel Analysis

```bash
python Transmission.py
```

Generates: Time-varying transmission decomposition and subperiod comparisons.

---

## Key Outputs Summary

### Data Files (in `data/`)
| File | Description | Use |
|------|-----------|-----|
| `monthly_panel.csv` | Core dataset | Input for DNS/SVAR |
| `summary_stats.csv` | Descriptive statistics | Table 1: Paper descriptive table |
| `stationarity_report.csv` | Unit root tests | Augmented Dickey-Fuller, Phillips-Perron results |
| `dns_factors.csv` | Yield curve factors | DNS model results |
| `svar_irf.csv` | Point-estimate IRFs | Main policy shock responses |
| `svar_irf_ci.csv` | Confidence bands (90%) | Shock response uncertainty |
| `svar_fevd.csv` | Variance decomposition | Policy shock importance |
| `h0_1_granger.csv` | Granger causality | Hypothesis 1 test results |
| `h0_2_irf_significance.csv` | IRF significance | Hypothesis 2 test results |
| `subperiod_results.csv` | Pre/post-2008 analysis | Robustness across crisis periods |

### Plots (in `plots/`)
| Plot | Description |
|------|-----------|
| `yield_series.png` | 1Y, 3Y, 5Y, 10Y yields over time |
| `dns_factors.png` | Level, slope, curvature evolution |
| `svar_irf_repo_shock.png` | **Main figure:** Repo rate shock → all yields |
| `svar_fevd.png` | Forecast error variance shares per maturity |
| `svar_residuals.png` | Structural shocks time series with diagnostics |

---

## Key Results Snapshot

The analysis reveals:

1. **Monetary policy transmission is significant but maturity-dependent** – the RBI repo rate shock has larger and faster effects on shorter-end yields (1Y, 3Y) vs. longer-end yields (10Y).

2. **Slope and curvature responses** – policy shocks primarily affect the slope of the yield curve (β₂), suggesting policy primarily impacts the medium-term risk premium rather than long-term expectations.

3. **Subperiod heterogeneity** – transmission strength and lag structure differ between pre-2008 and post-2008 periods, reflecting structural changes in Indian financial markets.

4. **Granger causality** – the repo rate Granger-causes all maturity yields, supporting unidirectional policy transmission.

---

## Diagnostic Checks

- **Stationarity:** First-differenced yields and repo rate are I(0) (confirmed by ADF + PP tests)
- **VAR lag selection:** Optimal p chosen by AIC/BIC (typically p=2 or p=3 for monthly data)
- **VAR stability:** All eigenvalues of companion matrix lie inside unit circle
- **Residual diagnostics:** White noise (ACF/PACF), no serial correlation (Ljung-Box)
- **Bootstrap:** 1000 bootstrap replications for IRF confidence bands
- **Robustness:** Results hold across alternative orderings and subperiods

---

## References

- Diebold, F. X., & Li, C. (2006). "Forecasting the term structure of government bonds." *Journal of Econometrics*, 130(2), 337-364.
- Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). "Monetary policy shocks: What have we learned and to what end?" *Handbook of macroeconomics*, 1, 65-148.
- Kuttner, K. N. (2001). "Monetary policy surprises and interest rates: Why did the Fed move after the 2001 attacks?" *Review of Economics and Statistics*, 83(4), 768-772.

---

## Author Notes

This project implements a complete econometric workflow from data preprocessing through hypothesis testing, suitable for publication in a peer-reviewed economics or finance journal. All outputs include both point estimates and statistical inference (confidence intervals, test statistics, p-values).

For questions or issues, refer to the docstrings in individual Python scripts (`Preprocessed/*.py`) for detailed methodology.

**Last Updated:** May 2026
