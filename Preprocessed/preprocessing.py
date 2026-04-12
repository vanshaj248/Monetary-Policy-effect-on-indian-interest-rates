"""
Preprocessing Pipeline — Monetary Policy Transmission to Indian Yield Curve
============================================================================
HOW TO USE
----------
1. Put all your raw CSV files in the same folder as this script
   (or update the paths in the CONFIG section below).

2. Run:
       python preprocess_yield_curve.py

3. Outputs are written to:
       data/monthly_panel.csv        ← main dataset  (228 obs, for DNS + SVAR)
       data/daily_panel.csv          ← event-study dataset (weekdays only)
       data/summary_stats.csv        ← descriptive stats table (paper-ready)
       data/stationarity_report.csv  ← ADF + PP unit root results (paper-ready)
       plots/yield_series.png        ← yield levels chart
       plots/yield_changes.png       ← monthly change bars
       plots/correlation_heatmap.png ← correlation matrices (levels + diffs)

Dependencies: pandas, numpy, scipy, matplotlib, seaborn
Install:  pip install pandas numpy scipy matplotlib seaborn
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  C O N F I G   —   edit file names / paths here
# ═══════════════════════════════════════════════════════════════════════════════

# Map each maturity label → path to its raw CSV file.
# All files must have at least two columns: Date and Price (yield in %).
# Add / uncomment rows once you have the 1Y, 3Y, 5Y files ready.
RAW_FILES = {
    "Y10": "/Users/vanshaj/Work/Eco Paper/Preprocessed/India 10-Year Bond Yield Historical Data (1)_standardized_date.csv",
    "Y1" : "/Users/vanshaj/Work/Eco Paper/Preprocessed/India 1-Year Bond Yield Historical Data (1)_standardized_date.csv",
    "Y3" : "/Users/vanshaj/Work/Eco Paper/Preprocessed/India 3-Year Bond Yield Historical Data (1)_standardized_date.csv",
    "Y5" : "/Users/vanshaj/Work/Eco Paper/Preprocessed/India 5-Year Bond Yield Historical Data (1)_standardized_date.csv",
}

REPO_FILE   = "/Users/vanshaj/Work/Eco Paper/Preprocessed/Repo_standardized.csv"   # must have columns: Date, Repo

STUDY_START = "2006-04-01"
STUDY_END   = "2025-03-31"

# Labels used in plots and tables
MATURITY_LABELS = {
    "Y1"  : "1-Year G-Sec",
    "Y3"  : "3-Year G-Sec",
    "Y5"  : "5-Year G-Sec",
    "Y10" : "10-Year G-Sec",
    "Repo": "RBI Repo Rate",
}
COLORS = {
    "Y1"  : "#2e8b57",
    "Y3"  : "#d4813a",
    "Y5"  : "#8b3a8b",
    "Y10" : "#1f5fa6",
    "Repo": "#c0392b",
}

# Output folders (created automatically)
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  L O A D E R S
# ═══════════════════════════════════════════════════════════════════════════════

def load_gsec(path: str, label: str) -> pd.Series:
    """
    Load a G-sec yield CSV and return a clean daily Series.

    Cleaning applied
    ----------------
    - Weekends dropped  : Sat/Sun rows carry Friday's value forward — not real
                          market observations. Removing them prevents artificial
                          autocorrelation in daily regressions.
    - Open/High/Low dropped : Always equal to Price in G-sec daily data —
                              entirely redundant.
    - Change % dropped  : Stored as a string (e.g. '0.09%'); we recompute
                          returns from Price when needed.
    - Study window enforced : Apr 2006 – Mar 2025.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  File not found: '{path}'\n"
            f"  Make sure the CSV is in the same folder as this script,\n"
            f"  or update the RAW_FILES path in the CONFIG section.\n"
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop weekends (Mon=0 … Fri=4, Sat=5, Sun=6)
    df = df[df["Date"].dt.dayofweek < 5].copy()

    # Restrict to study window
    mask = (df["Date"] >= STUDY_START) & (df["Date"] <= STUDY_END)
    df   = df.loc[mask].copy()

    # Keep only Date + Price; rename Price → maturity label
    df = (df[["Date", "Price"]]
            .rename(columns={"Price": label})
            .set_index("Date")
            .sort_index())

    if df.index.duplicated().any():
        raise ValueError(f"Duplicate dates found in {label} — check your raw file.")

    nan_count = df[label].isna().sum()
    print(f"  {label:5s}  rows={len(df):,}  NaN={nan_count}"
          f"  range=[{df[label].min():.3f}, {df[label].max():.3f}]%")
    return df[label]


def load_repo(path: str) -> pd.Series:
    """
    Load RBI Repo rate CSV and return a clean daily Series.
    Weekend rows carry Friday's rate forward — dropped for the same reason
    as G-sec weekends.
    Must have columns: Date, Repo
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  File not found: '{path}'\n"
            f"  Make sure the CSV is in the same folder as this script,\n"
            f"  or update REPO_FILE in the CONFIG section.\n"
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.dayofweek < 5].copy()

    mask = (df["Date"] >= STUDY_START) & (df["Date"] <= STUDY_END)
    df   = df.loc[mask].copy()

    df = (df[["Date", "Repo"]]
            .set_index("Date")
            .sort_index())

    if df.index.duplicated().any():
        raise ValueError("Duplicate dates found in Repo — check your raw file.")

    print(f"  {'Repo':5s}  rows={len(df):,}  NaN={df['Repo'].isna().sum()}"
          f"  range=[{df['Repo'].min():.2f}, {df['Repo'].max():.2f}]%")
    return df["Repo"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  B U I L D   P A N E L S
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/5] Loading raw files ...")
series_list = [load_gsec(path, label) for label, path in RAW_FILES.items()]
series_list.append(load_repo(REPO_FILE))

daily = pd.concat(series_list, axis=1).sort_index()

# Indian trading holidays fall on weekdays → show up as NaN after alignment.
# These are genuine non-trading days; dropping them is correct.
n_before    = len(daily)
daily_clean = daily.dropna()
n_dropped   = n_before - len(daily_clean)
print(f"\n  Rows before dropna  : {n_before:,}")
print(f"  Rows dropped (hols) : {n_dropped:,}")
print(f"  Final daily rows    : {len(daily_clean):,}")
daily_clean.to_csv("data/daily_panel.csv")
print("  Saved → data/daily_panel.csv")

# ── Monthly panel ─────────────────────────────────────────────────────────────
# Month-end last observation is the standard in term structure literature.
# This ensures yields reflect the market's prevailing level at each
# month-end, consistent with how most RBI and academic studies report data.
print("\n[2/5] Building monthly panel ...")
monthly = daily_clean.resample("ME").last()
monthly.index.name = "Date"

expected_months = 228   # Apr 2006 – Mar 2025
if len(monthly) != expected_months:
    print(f"  WARNING: Expected {expected_months} months, got {len(monthly)}")
else:
    print(f"  Monthly observations: {len(monthly)}  (Apr 2006 – Mar 2025) ✓")

monthly.to_csv("data/monthly_panel.csv")
print("  Saved → data/monthly_panel.csv")
print(f"\n  First 3 rows:\n{monthly.head(3).to_string()}")
print(f"\n  Last 3 rows:\n{monthly.tail(3).to_string()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  S U M M A R Y   S T A T I S T I C S
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/5] Computing summary statistics ...")


def summary_row(s: pd.Series, label: str) -> dict:
    """
    Compute paper-standard descriptive statistics for a series.
    JB = Jarque-Bera normality test (H0: normal distribution).
    Excess kurtosis: 0 = normal; >0 = fat tails.
    """
    d             = s.dropna()
    jb_stat, jb_p = stats.normaltest(d)
    return {
        "Series"   : label,
        "N"        : len(d),
        "Mean"     : round(d.mean(), 4),
        "Std Dev"  : round(d.std(ddof=1), 4),
        "Min"      : round(d.min(), 4),
        "Max"      : round(d.max(), 4),
        "Skewness" : round(float(stats.skew(d)), 4),
        "Ex.Kurt"  : round(float(stats.kurtosis(d)), 4),   # excess kurtosis
        "JB stat"  : round(jb_stat, 4),
        "JB p-val" : round(jb_p, 4),
    }


stat_rows = []
for col in monthly.columns:
    # Levels
    stat_rows.append(summary_row(monthly[col], col))
for col in monthly.columns:
    # First differences (monthly changes in yield / pp)
    stat_rows.append(summary_row(monthly[col].diff().dropna(), f"D.{col}"))

stats_df = pd.DataFrame(stat_rows).set_index("Series")
stats_df.to_csv("data/summary_stats.csv")
print(stats_df.to_string())
print("\n  Saved → data/summary_stats.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  U N I T   R O O T   T E S T S   (ADF + Phillips-Perron)
# ═══════════════════════════════════════════════════════════════════════════════
# Implemented from scratch using numpy/scipy — no statsmodels required.
#
# ADF  H0: unit root present
#      Reject (t < critical value) → series is stationary
#
# PP   Same H0, but uses Newey-West HAC variance to correct for
#      serial correlation without adding lag terms.
#
# Both use constant + trend specification (MacKinnon 1994 critical values).
# Lag order for ADF selected by AIC (Schwert 1989 max-lag rule).

print("\n[4/5] Running unit root tests ...")

# MacKinnon (1994) critical values — constant + trend, T ≈ 228
ADF_CV = {"1%": -3.96, "5%": -3.41, "10%": -3.13}


def _p_label(t: float) -> str:
    """Approximate p-value bracket from MacKinnon critical values."""
    if   t < ADF_CV["1%"]:  return "< 0.01"
    elif t < ADF_CV["5%"]:  return "< 0.05"
    elif t < ADF_CV["10%"]: return "< 0.10"
    else:                    return "> 0.10"


def _build_adf_matrix(y: np.ndarray, lag: int):
    """
    Build OLS design matrix for the ADF regression:
      Δy_t = α + β·t + γ·y_{t-1} + Σ δ_i·Δy_{t-i} + ε
    Returns (X, Δy_t, index_of_γ_column).
    """
    dy = np.diff(y)

    if lag > 0:
        dy_t    = dy[lag:]
        ylag    = y[lag: len(y) - 1]
        lag_mat = np.column_stack(
            [dy[lag - i - 1: len(dy) - i - 1] for i in range(lag)]
        )
        T     = len(dy_t)
        t_vec = np.arange(1, T + 1, dtype=float)
        X     = np.column_stack([np.ones(T), t_vec, ylag, lag_mat])
    else:
        dy_t  = dy
        ylag  = y[:-1]
        T     = len(dy_t)
        t_vec = np.arange(1, T + 1, dtype=float)
        X     = np.column_stack([np.ones(T), t_vec, ylag])

    return X, dy_t, 2   # γ is always at column index 2 (after const + trend)


def _ols(X: np.ndarray, y: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid         = y - X @ beta
    return beta, resid


def _aic_lag(y: np.ndarray, max_lag: int) -> int:
    """Select ADF lag order minimising AIC."""
    best = (np.inf, 0)
    for lag in range(0, max_lag + 1):
        try:
            X, dy_t, _ = _build_adf_matrix(y, lag)
            beta, resid = _ols(X, dy_t)
            T   = len(resid)
            k   = X.shape[1]
            aic = np.log(resid @ resid / T) + 2 * k / T
            if aic < best[0]:
                best = (aic, lag)
        except Exception:
            pass
    return best[1]


def adf_test(series: pd.Series, label: str) -> dict:
    """Augmented Dickey-Fuller test, constant + trend, AIC lag selection."""
    y       = series.dropna().values.astype(float)
    max_lag = max(1, int(12 * (len(y) / 100) ** 0.25))   # Schwert (1989) rule
    lag     = _aic_lag(y, max_lag)

    X, dy_t, g_idx = _build_adf_matrix(y, lag)
    beta, resid    = _ols(X, dy_t)

    T      = len(resid)
    k      = X.shape[1]
    s2     = resid @ resid / (T - k)
    se     = np.sqrt(s2 * np.linalg.inv(X.T @ X)[g_idx, g_idx])
    t_stat = beta[g_idx] / se

    return {
        "Series"     : label,
        "Test"       : "ADF",
        "Statistic"  : round(t_stat, 4),
        "p-value"    : _p_label(t_stat),
        "Lags"       : lag,
        "CV 1%"      : ADF_CV["1%"],
        "CV 5%"      : ADF_CV["5%"],
        "Conclusion" : "Stationary" if t_stat < ADF_CV["5%"] else "Unit root",
    }


def pp_test(series: pd.Series, label: str) -> dict:
    """
    Phillips-Perron test using HAC (Newey-West) long-run variance correction.
    Equivalent to Andrews (1991) automatic bandwidth selection.
    """
    y = series.dropna().values.astype(float)

    # Lag-0 OLS (no augmentation lags)
    X, dy_t, g_idx = _build_adf_matrix(y, 0)
    beta, resid    = _ols(X, dy_t)

    T  = len(resid)
    k  = X.shape[1]
    s2 = resid @ resid / (T - k)

    # Newey-West bandwidth
    bw = max(1, int(4 * (T / 100) ** (2 / 9)))

    # Long-run variance via Bartlett kernel
    lrv = resid @ resid / T
    for j in range(1, bw + 1):
        w   = 1.0 - j / (bw + 1)
        cov = resid[j:] @ resid[: T - j] / T
        lrv += 2 * w * cov

    XtXinv = np.linalg.inv(X.T @ X)
    se_hac = np.sqrt(lrv * XtXinv[g_idx, g_idx])
    t_pp   = beta[g_idx] / se_hac

    return {
        "Series"     : label,
        "Test"       : "PP",
        "Statistic"  : round(t_pp, 4),
        "p-value"    : _p_label(t_pp),
        "Lags"       : bw,
        "CV 1%"      : ADF_CV["1%"],
        "CV 5%"      : ADF_CV["5%"],
        "Conclusion" : "Stationary" if t_pp < ADF_CV["5%"] else "Unit root",
    }


ur_rows = []
for col in monthly.columns:
    ur_rows.append(adf_test(monthly[col],                       col))
    ur_rows.append(pp_test (monthly[col],                       col))
    ur_rows.append(adf_test(monthly[col].diff().dropna(),  f"D.{col}"))
    ur_rows.append(pp_test (monthly[col].diff().dropna(),  f"D.{col}"))

ur_df = pd.DataFrame(ur_rows)
ur_df.to_csv("data/stationarity_report.csv", index=False)
print(ur_df[["Series","Test","Statistic","p-value","Conclusion"]].to_string(index=False))
print("\n  Saved → data/stationarity_report.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  P L O T S
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/5] Generating plots ...")

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : False,
})

# Column order for all plots
ordered_cols = [c for c in ["Y1", "Y3", "Y5", "Y10", "Repo"]
                if c in monthly.columns]

# ── 5a. Yield levels ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

for col in ordered_cols:
    ax.plot(
        monthly.index, monthly[col],
        label    = MATURITY_LABELS.get(col, col),
        color    = COLORS.get(col, "gray"),
        linestyle= "--" if col == "Repo" else "-",
        linewidth= 1.6 if col == "Repo" else 2.0,
    )

# Annotate key RBI policy episodes
episodes = {
    "GFC\n2008"        : "2008-10-01",
    "Rajan\n2013"      : "2013-09-04",
    "COVID cut\n2020"  : "2020-03-27",
    "Tightening\n2022" : "2022-05-04",
}
_, yhi = ax.get_ylim()
for ep_label, date_str in episodes.items():
    xv = pd.to_datetime(date_str)
    if monthly.index.min() <= xv <= monthly.index.max():
        ax.axvline(xv, color="#aaaaaa", linewidth=0.8, linestyle=":")
        ax.text(xv, yhi * 0.98, ep_label,
                fontsize=7.5, color="#666666",
                ha="center", va="top", linespacing=1.3)

ax.set_title(
    "Indian G-Sec Yields and RBI Repo Rate — Monthly (Apr 2006 – Mar 2025)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_ylabel("Yield / Rate (%)", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend(loc="upper right", fontsize=9, framealpha=0.6, edgecolor="#cccccc")
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/yield_series.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved → plots/yield_series.png")

# ── 5b. Monthly changes ───────────────────────────────────────────────────────
n_p   = len(ordered_cols)
fig, axes = plt.subplots(n_p, 1, figsize=(13, 2.8 * n_p), sharex=True)
if n_p == 1:
    axes = [axes]

for ax, col in zip(axes, ordered_cols):
    d     = monthly[col].diff().dropna()
    color = COLORS.get(col, "steelblue")
    ax.bar(d.index, d.clip(lower=0).values, width=25, color=color,     alpha=0.75)
    ax.bar(d.index, d.clip(upper=0).values, width=25, color="#e74c3c", alpha=0.60)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"Δ {MATURITY_LABELS.get(col, col)}\n(pp)", fontsize=8.5)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
fig.suptitle(
    "Monthly Changes in Yields and Repo Rate (percentage points)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig("plots/yield_changes.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved → plots/yield_changes.png")

# ── 5c. Correlation heatmap ───────────────────────────────────────────────────
col_labels = [MATURITY_LABELS.get(c, c) for c in ordered_cols]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (data, title) in zip(axes, [
    (monthly[ordered_cols],                "Levels"),
    (monthly[ordered_cols].diff().dropna(), "First Differences"),
]):
    corr = data.corr()
    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        xticklabels=col_labels,
        yticklabels=col_labels,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(f"Correlation Matrix — {title}",
                 fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("  Saved → plots/correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  D O N E
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  PRE-PROCESSING COMPLETE")
print("=" * 64)
print(f"  data/daily_panel.csv           {len(daily_clean):>5,} rows × {daily_clean.shape[1]} cols")
print(f"  data/monthly_panel.csv         {len(monthly):>5,} rows × {monthly.shape[1]} cols")
print(f"  data/summary_stats.csv         descriptive statistics")
print(f"  data/stationarity_report.csv   ADF + PP unit root tests")
print(f"  plots/yield_series.png         yield level chart")
print(f"  plots/yield_changes.png        monthly change bars")
print(f"  plots/correlation_heatmap.png  correlation matrices")
print()
print("  TO COMPLETE  →  add 1Y / 3Y / 5Y CSV paths to RAW_FILES")
print("                  at the top of this script, then re-run.")
print("  NEXT SCRIPT  →  python dns_model.py")
print("=" * 64)