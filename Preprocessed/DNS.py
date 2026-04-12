"""
Section 2 — Term Structure Modelling
Dynamic Nelson-Siegel (DNS) Model
======================================
Paper: Monetary Policy Transmission to the Indian Yield Curve

Theory
------
The Nelson-Siegel model fits the yield curve at each point in time as:

    y(τ) = β1                           ← Level    (affects all maturities equally)
          + β2 * [(1 - e^{-λτ}) / λτ]  ← Slope    (decays with maturity)
          + β3 * [(1 - e^{-λτ}) / λτ  - e^{-λτ}]  ← Curvature (hump shape)

where τ = maturity in months, λ = decay parameter (controls where hump peaks).

The "Dynamic" version (Diebold & Li 2006) treats β1, β2, β3 as time-varying
factors estimated by OLS at each month t, then models their dynamics as AR(1)
processes. This is the standard approach in the empirical term structure
literature and directly feeds into the SVAR in Section 3.

Inputs
------
  data/monthly_panel.csv   (from preprocess_yield_curve.py)

Outputs
-------
  data/dns_factors.csv           → β1(t), β2(t), β3(t), fitted yields, residuals
  data/dns_fit_stats.csv         → R², RMSE per maturity + factor AR(1) stats
  data/dns_lambda.txt            → chosen λ value with justification
  plots/dns_factors.png          → time series of three factors
  plots/dns_fit_curves.png       → fitted vs actual yield curves (4 snapshots)
  plots/dns_residuals.png        → fitting residuals per maturity
  plots/dns_factor_acf.png       → ACF of factors (confirms persistence)

Dependencies: pandas, numpy, scipy, matplotlib
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr

os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── 0. CONFIGURATION ─────────────────────────────────────────────────────────

# Maturities in months matching your column names.
# Add Y1, Y3, Y5 columns to monthly_panel.csv to unlock full DNS estimation.
# With only Y10 we demonstrate the framework — full model needs ≥3 maturities.
MATURITY_MAP = {
    "Y1" :  12,   # 1-year  = 12 months
    "Y3" :  36,   # 3-year  = 36 months
    "Y5" :  60,   # 5-year  = 60 months
    "Y10": 120,   # 10-year = 120 months
}

COLORS = {
    "β1 (Level)"    : "#1f5fa6",
    "β2 (Slope)"    : "#c0392b",
    "β3 (Curvature)": "#2e8b57",
    "Repo"          : "#8b3a8b",
}

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 62)
print("  DNS TERM STRUCTURE MODEL")
print("=" * 62)

monthly = pd.read_csv(
    "data/monthly_panel.csv", index_col="Date", parse_dates=True
)

# Identify which yield columns are present
yield_cols = [c for c in ["Y1","Y3","Y5","Y10"] if c in monthly.columns]
maturities = np.array([MATURITY_MAP[c] for c in yield_cols])
Y = monthly[yield_cols].values          # shape: (T, n_maturities)
T, n_mat = Y.shape

print(f"\n  Yield columns  : {yield_cols}")
print(f"  Maturities (mo): {maturities.tolist()}")
print(f"  Time periods   : {T}  ({monthly.index[0].date()} – {monthly.index[-1].date()})")
print(f"  Missing values : {np.isnan(Y).sum()}")

if n_mat < 3:
    print("""
  NOTE: DNS requires at least 3 maturities to identify all three factors
  (Level, Slope, Curvature). Currently only {n} maturity columns found.

  To enable full DNS estimation:
    1. Ensure 1Y, 3Y, 5Y raw CSVs are pre-processed (same format as 10Y).
    2. Add their paths to RAW_FILES in preprocess_yield_curve.py and re-run it.
    3. Re-run this script — it will automatically pick up all four columns.

  Running single-maturity demonstration with Y10 only for now.
  All plots and factor logic are identical once all maturities are loaded.
""".format(n=n_mat))


# ── 2. NELSON-SIEGEL FACTOR LOADINGS ─────────────────────────────────────────

def ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute Nelson-Siegel factor loading matrix for given maturities and λ.

    Returns matrix of shape (len(tau), 3):
      col 0: loading on β1 (Level)    = 1
      col 1: loading on β2 (Slope)    = (1 - e^{-λτ}) / (λτ)
      col 2: loading on β3 (Curvature)= (1 - e^{-λτ}) / (λτ) - e^{-λτ}
    """
    lt  = lam * tau
    # Numerically stable for very small lt
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(lt < 1e-8, 1.0, (1 - np.exp(-lt)) / lt)
    L1 = np.ones(len(tau))
    L2 = decay
    L3 = decay - np.exp(-lt)
    return np.column_stack([L1, L2, L3])


# ── 3. OPTIMAL λ SELECTION ───────────────────────────────────────────────────
# Diebold & Li (2006) fix λ at 0.0609 (peaks at ~30 months for US data).
# For India, we grid-search to minimise total in-sample RMSE.
# This is standard practice for emerging market applications.

print("\n[1/5] Selecting optimal λ (decay parameter) ...")

def total_rmse(lam: float, Y: np.ndarray, tau: np.ndarray) -> float:
    """Total RMSE across all months and maturities for a given λ."""
    if lam <= 0:
        return 1e10
    L    = ns_loadings(tau, lam)
    errs = []
    for t in range(len(Y)):
        y_t = Y[t]
        mask = ~np.isnan(y_t)
        if mask.sum() < 2:
            continue
        # OLS: β = (L'L)^{-1} L'y
        L_m   = L[mask]
        y_m   = y_t[mask]
        try:
            beta  = np.linalg.lstsq(L_m, y_m, rcond=None)[0]
            resid = y_m - L_m @ beta
            errs.append(resid @ resid / mask.sum())
        except Exception:
            pass
    return np.mean(errs) if errs else 1e10


# Grid search over λ ∈ [0.01, 0.30], then refine with scalar minimisation
lam_grid  = np.linspace(0.01, 0.30, 200)
rmse_grid = [total_rmse(l, Y, maturities) for l in lam_grid]
lam_init  = lam_grid[np.argmin(rmse_grid)]

result    = minimize_scalar(
    total_rmse, bounds=(0.005, 0.50),
    method="bounded", args=(Y, maturities),
    options={"xatol": 1e-6}
)
lam_opt   = result.x
rmse_opt  = result.fun

# Peak of curvature loading = 1/λ months
peak_months = 1.0 / lam_opt

print(f"  Diebold-Li fixed λ (US)    : 0.0609  (curvature peaks at ~30 months)")
print(f"  Optimal λ for Indian data  : {lam_opt:.4f}  (curvature peaks at {peak_months:.1f} months)")
print(f"  In-sample RMSE at opt λ    : {rmse_opt:.6f}")

# Save λ
with open("data/dns_lambda.txt", "w") as f:
    f.write(f"optimal_lambda={lam_opt:.6f}\n")
    f.write(f"curvature_peak_months={peak_months:.2f}\n")
    f.write(f"total_rmse={rmse_opt:.8f}\n")
    f.write(f"diebold_li_us_lambda=0.0609\n")
print("  Saved  data/dns_lambda.txt")


# ── 4. OLS FACTOR EXTRACTION ─────────────────────────────────────────────────
# At each time t, run OLS of y_t on NS loading matrix → β1(t), β2(t), β3(t)

print("\n[2/5] Extracting DNS factors via cross-sectional OLS ...")

L = ns_loadings(maturities, lam_opt)   # shape: (n_mat, 3)

betas  = np.full((T, 3), np.nan)       # [β1, β2, β3] for each month
fitted = np.full((T, n_mat), np.nan)   # fitted yields
resids = np.full((T, n_mat), np.nan)   # residuals

for t in range(T):
    y_t  = Y[t]
    mask = ~np.isnan(y_t)
    if mask.sum() < 2:
        continue
    L_m = L[mask]
    y_m = y_t[mask]
    try:
        beta_t          = np.linalg.lstsq(L_m, y_m, rcond=None)[0]
        betas[t]        = beta_t
        fitted[t, mask] = L_m @ beta_t
        resids[t, mask] = y_m - fitted[t, mask]
    except Exception:
        pass

# Build factors DataFrame
factors = pd.DataFrame(
    betas,
    index=monthly.index,
    columns=["beta1_level", "beta2_slope", "beta3_curvature"]
)

# Attach fitted yields and residuals
for i, col in enumerate(yield_cols):
    factors[f"fitted_{col}"]  = fitted[:, i]
    factors[f"residual_{col}"] = resids[:, i]

# Attach actual yields and Repo for convenience
for col in yield_cols:
    factors[col] = monthly[col].values
if "Repo" in monthly.columns:
    factors["Repo"] = monthly["Repo"].values

factors.index.name = "Date"
factors.to_csv("data/dns_factors.csv")
print(f"  Saved  data/dns_factors.csv  ({len(factors)} rows × {factors.shape[1]} cols)")
print(f"\n  Factor preview (first 3 months):")
print(factors[["beta1_level","beta2_slope","beta3_curvature"]].head(3).round(4).to_string())


# ── 5. FIT DIAGNOSTICS ───────────────────────────────────────────────────────

print("\n[3/5] Computing fit statistics ...")


def ar1_stats(series: np.ndarray) -> dict:
    """OLS AR(1): x_t = c + ρ x_{t-1} + ε. Returns ρ, t-stat, R²."""
    s    = series[~np.isnan(series)]
    x    = s[:-1]
    y    = s[1:]
    X    = np.column_stack([np.ones(len(x)), x])
    beta, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    T_    = len(y)
    k     = 2
    s2    = resid @ resid / (T_ - k)
    se    = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    rho   = beta[1]
    t_rho = rho / se
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"rho": round(rho, 4), "t_stat": round(t_rho, 4), "AR1_R2": round(r2, 4)}


fit_rows = []

# Per-maturity R² and RMSE
for i, col in enumerate(yield_cols):
    r   = resids[:, i]
    y_c = Y[:, i]
    mask = ~np.isnan(r)
    r_m  = r[mask];  y_m = y_c[mask]
    rmse = np.sqrt((r_m ** 2).mean())
    mae  = np.abs(r_m).mean()
    ss_res = (r_m ** 2).sum()
    ss_tot = ((y_m - y_m.mean()) ** 2).sum()
    r2   = 1 - ss_res / ss_tot
    fit_rows.append({
        "Maturity": col, "R2": round(r2, 6),
        "RMSE (bp)": round(rmse * 100, 4),
        "MAE (bp)": round(mae * 100, 4),
        "N": int(mask.sum()),
    })

# Factor AR(1) dynamics
factor_ar_rows = []
for col, name in zip(
    ["beta1_level","beta2_slope","beta3_curvature"],
    ["Level (β1)", "Slope (β2)", "Curvature (β3)"]
):
    ar = ar1_stats(factors[col].values)
    factor_ar_rows.append({"Factor": name, **ar})

fit_df = pd.DataFrame(fit_rows)
ar_df  = pd.DataFrame(factor_ar_rows)

fit_df.to_csv("data/dns_fit_stats.csv", index=False)
ar_df.to_csv("data/dns_factor_ar1.csv", index=False)

print("\n  Fit statistics per maturity:")
print(fit_df.to_string(index=False))
print("\n  Factor AR(1) dynamics:")
print(ar_df.to_string(index=False))
print("  Saved  data/dns_fit_stats.csv")
print("  Saved  data/dns_factor_ar1.csv")


# ── 6. ECONOMIC INTERPRETATION CHECK ─────────────────────────────────────────
# In well-identified DNS: β1 ≈ long yield, β1+β2 ≈ short yield, β2 = -(slope)

print("\n[4/5] Economic interpretation check ...")

b1 = factors["beta1_level"].dropna()
b2 = factors["beta2_slope"].dropna()
b3 = factors["beta3_curvature"].dropna()

# β1 should correlate highly with long-end yield (10Y)
if "Y10" in monthly.columns:
    r_b1_y10, _ = pearsonr(b1.values, monthly["Y10"].dropna().values[:len(b1)])
    print(f"  Corr(β1, 10Y yield)  : {r_b1_y10:.4f}  [expect ≈ +1.0]")

# β2 negative → downward-sloping curve (β2 ≈ short minus long)
if "Y1" in monthly.columns and "Y10" in monthly.columns:
    spread = monthly["Y1"] - monthly["Y10"]
    r_b2_sp, _ = pearsonr(b2.values, spread.dropna().values[:len(b2)])
    print(f"  Corr(β2, 1Y−10Y)     : {r_b2_sp:.4f}  [expect ≈ +1.0]")
else:
    print(f"  β2 mean: {b2.mean():.4f}  [negative = normal upward-sloping curve]")

print(f"\n  β1 (Level)     mean={b1.mean():.3f}  std={b1.std():.3f}")
print(f"  β2 (Slope)     mean={b2.mean():.3f}  std={b2.std():.3f}")
print(f"  β3 (Curvature) mean={b3.mean():.3f}  std={b3.std():.3f}")


# ── 7. PLOTS ─────────────────────────────────────────────────────────────────

print("\n[5/5] Generating plots ...")

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : False,
})

# ── 7a. Factor time series ────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

# Panel 1: Repo rate (policy variable — for visual reference)
if "Repo" in factors.columns:
    axes[0].plot(factors.index, factors["Repo"],
                 color=COLORS["Repo"], linewidth=1.6, linestyle="--")
    axes[0].set_ylabel("Repo Rate (%)", fontsize=9)
    axes[0].set_title("RBI Repo Rate (Policy Reference)", fontsize=9,
                       color="#555555")
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.5)

# Panel 2: β1 Level
axes[1].plot(factors.index, factors["beta1_level"],
             color=COLORS["β1 (Level)"], linewidth=1.8)
axes[1].set_ylabel("β₁ — Level (%)", fontsize=9)
axes[1].set_title("β₁: Level factor  (long-run yield component)", fontsize=9,
                   color="#555555")
axes[1].grid(axis="y", alpha=0.2, linewidth=0.5)

# Panel 3: β2 Slope
axes[2].plot(factors.index, factors["beta2_slope"],
             color=COLORS["β2 (Slope)"], linewidth=1.8)
axes[2].axhline(0, color="black", linewidth=0.6, linestyle=":")
axes[2].set_ylabel("β₂ — Slope (%)", fontsize=9)
axes[2].set_title("β₂: Slope factor  (monetary policy stance proxy)", fontsize=9,
                   color="#555555")
axes[2].grid(axis="y", alpha=0.2, linewidth=0.5)

# Panel 4: β3 Curvature
axes[3].plot(factors.index, factors["beta3_curvature"],
             color=COLORS["β3 (Curvature)"], linewidth=1.8)
axes[3].axhline(0, color="black", linewidth=0.6, linestyle=":")
axes[3].set_ylabel("β₃ — Curvature (%)", fontsize=9)
axes[3].set_title("β₃: Curvature factor  (medium-term hump)", fontsize=9,
                   color="#555555")
axes[3].grid(axis="y", alpha=0.2, linewidth=0.5)

# Key RBI policy episodes
episodes = {
    "GFC 2008"      : "2008-10-01",
    "Rajan 2013"    : "2013-09-04",
    "COVID 2020"    : "2020-03-27",
    "Tightening\n2022": "2022-05-04",
}
for ax in axes:
    for ep, ds in episodes.items():
        xv = pd.to_datetime(ds)
        if factors.index.min() <= xv <= factors.index.max():
            ax.axvline(xv, color="#bbbbbb", linewidth=0.7, linestyle=":")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

fig.suptitle(
    f"Dynamic Nelson-Siegel Factors — Indian G-Sec Yield Curve\n"
    f"λ = {lam_opt:.4f}  (curvature peaks at {peak_months:.1f} months)",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("plots/dns_factors.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/dns_factors.png")


# ── 7b. Fitted vs Actual yield curves (snapshot dates) ───────────────────────
# Choose 4 economically interesting dates
snap_dates_str = ["2008-09-30", "2013-08-31", "2020-05-31", "2022-09-30"]
snap_dates = [pd.to_datetime(d) for d in snap_dates_str]
snap_labels = ["Sep 2008\n(GFC peak)", "Aug 2013\n(Rajan shock)",
               "May 2020\n(COVID cut)", "Sep 2022\n(Tightening)"]

# Dense maturity grid for smooth fitted curve
tau_dense = np.array([1, 3, 6, 12, 24, 36, 48, 60, 84, 120], dtype=float)
L_dense   = ns_loadings(tau_dense, lam_opt)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flatten()

for ax, dt, label in zip(axes_flat, snap_dates, snap_labels):
    # Find nearest month-end in index
    idx = factors.index.searchsorted(dt)
    idx = min(idx, len(factors) - 1)
    t_actual = factors.index[idx]

    b = factors.loc[t_actual, ["beta1_level","beta2_slope","beta3_curvature"]].values
    fitted_dense = L_dense @ b

    ax.plot(tau_dense, fitted_dense, "-",
            color="#1f5fa6", linewidth=2.2, label="DNS fitted curve", zorder=3)

    # Plot actual data points
    y_actual = Y[idx]
    ax.scatter(maturities, y_actual,
               color="#c0392b", s=60, zorder=4,
               label="Observed yields", edgecolors="white", linewidths=0.8)

    ax.set_title(f"{label}\n"
                 f"β₁={b[0]:.2f}  β₂={b[1]:.2f}  β₃={b[2]:.2f}",
                 fontsize=9)
    ax.set_xlabel("Maturity (months)", fontsize=8)
    ax.set_ylabel("Yield (%)", fontsize=8)
    ax.set_xticks([12, 36, 60, 120])
    ax.set_xticklabels(["1Y", "3Y", "5Y", "10Y"], fontsize=8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=7, loc="lower right")

fig.suptitle("DNS Fitted Yield Curves vs Observed — Selected Dates",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/dns_fit_curves.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/dns_fit_curves.png")


# ── 7c. Residuals per maturity ────────────────────────────────────────────────
n_panels = len(yield_cols)
fig, axes = plt.subplots(n_panels, 1,
                          figsize=(13, 2.8 * n_panels), sharex=True)
if n_panels == 1:
    axes = [axes]

for ax, col in zip(axes, yield_cols):
    r = factors[f"residual_{col}"].values * 100   # convert to basis points
    rmse_bp = np.sqrt(np.nanmean(r**2))
    ax.bar(factors.index, np.where(r >= 0, r, 0),
           width=25, color="#1f5fa6", alpha=0.7)
    ax.bar(factors.index, np.where(r < 0, r, 0),
           width=25, color="#c0392b", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"{col} residual\n(basis pts)", fontsize=8.5)
    ax.text(0.02, 0.92, f"RMSE = {rmse_bp:.2f} bp",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
fig.suptitle("DNS Fitting Residuals per Maturity (basis points)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/dns_residuals.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/dns_residuals.png")


# ── 7d. Loading curves (visualise what each factor does) ─────────────────────
tau_plot = np.linspace(1, 120, 300)
L_plot   = ns_loadings(tau_plot, lam_opt)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tau_plot, L_plot[:, 0], color=COLORS["β1 (Level)"],
        linewidth=2.2, label="β₁ loading: Level (constant = 1)")
ax.plot(tau_plot, L_plot[:, 1], color=COLORS["β2 (Slope)"],
        linewidth=2.2, label="β₂ loading: Slope")
ax.plot(tau_plot, L_plot[:, 2], color=COLORS["β3 (Curvature)"],
        linewidth=2.2, label="β₃ loading: Curvature")

ax.axvline(peak_months, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(peak_months + 2, 0.05, f"Curvature peak\n({peak_months:.0f} mo)",
        fontsize=8, color="#666666")

ax.set_xlabel("Maturity (months)", fontsize=10)
ax.set_ylabel("Factor Loading", fontsize=10)
ax.set_xticks([12, 36, 60, 120])
ax.set_xticklabels(["1Y", "3Y", "5Y", "10Y"])
ax.legend(fontsize=9, loc="center right")
ax.grid(axis="y", alpha=0.2, linewidth=0.5)
ax.set_title(f"Nelson-Siegel Factor Loadings  (λ = {lam_opt:.4f})",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("plots/dns_loadings.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/dns_loadings.png")


# ── 7e. λ grid search plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(lam_grid, rmse_grid, color="#1f5fa6", linewidth=1.8)
ax.axvline(lam_opt, color="#c0392b", linewidth=1.2, linestyle="--",
           label=f"Optimal λ = {lam_opt:.4f}")
ax.axvline(0.0609, color="#aaaaaa", linewidth=1.0, linestyle=":",
           label="Diebold-Li (US) λ = 0.0609")
ax.set_xlabel("λ (decay parameter)", fontsize=10)
ax.set_ylabel("Total RMSE (%)", fontsize=10)
ax.set_title("λ Selection: Grid Search over Total In-Sample RMSE",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2, linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/dns_lambda_search.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/dns_lambda_search.png")


# ── 8. SUMMARY FOR PAPER ─────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  DNS MODEL COMPLETE — PAPER SUMMARY")
print("=" * 62)
print(f"""
  Model      : Dynamic Nelson-Siegel (Diebold & Li 2006)
  Maturities : {yield_cols}  →  {maturities.tolist()} months
  Sample     : Apr 2006 – Mar 2025  (T = {T})
  λ (optimal): {lam_opt:.4f}  (curvature peaks at {peak_months:.1f} months)

  Factor AR(1) persistence:
{ar_df.to_string(index=False)}

  Fit quality:
{fit_df.to_string(index=False)}

  Economic interpretation:
    β1 (Level)     → long-run yield level; tracks inflation expectations
    β2 (Slope)     → term spread proxy; responds to RBI policy stance
    β3 (Curvature) → medium-term hump; driven by business cycle uncertainty

  These three factors feed directly into the SVAR in Section 3.
  Run: python3 svar_model.py
""")