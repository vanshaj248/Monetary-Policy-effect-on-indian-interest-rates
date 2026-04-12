"""
Section 4 — Transmission Analysis
===================================
Paper: Monetary Policy Transmission to the Indian Yield Curve

This section has four analytical components:

  4A. Term Premium Decomposition
      Decompose each yield into:
        Yield(τ,t) = Expected Short Rate(τ,t) + Term Premium(τ,t)
      Using VAR-implied expectations of the future repo rate path
      (Adrian, Crump & Moench 2013 methodology applied to Indian data).

  4B. Maturity Attenuation Analysis
      Test whether RBI shock transmission attenuates at longer maturities.
      Uses IRF peak responses and cumulative impulse responses by maturity.
      Directly addresses: "do long-end yields respond less than short-end?"

  4C. Channel Decomposition
      For each maturity, decompose yield response to policy shock into:
        - Expectations channel: change in expected short rate path
        - Term premium channel: residual (risk/uncertainty adjustment)
      This is the core test for H0 Part 3.

  4D. Sub-period Structural Stability
      Re-estimate SVAR across three RBI policy regimes:
        Period 1: Apr 2006 – Sep 2013  (Pre-IT, Subbarao)
        Period 2: Oct 2013 – Mar 2020  (IT framework, Rajan/Patel/Das)
        Period 3: Apr 2020 – Mar 2025  (COVID + tightening cycle)
      Tests whether transmission mechanisms changed across regimes.

Inputs
------
  data/monthly_panel.csv       → yield levels + repo
  data/svar_var_coefficients.csv → VAR(p) coefficient matrix
  data/svar_irf_ci.csv         → IRF point estimates + bootstrap CI
  data/svar_fevd.csv           → FEVD

Outputs
-------
  data/tp_decomposition.csv    → expectations + term premium series per maturity
  data/tp_variance_shares.csv  → variance share of each component per maturity
  data/channel_decomp.csv      → expectations vs TP channel shares in IRF
  data/maturity_attenuation.csv→ peak/cumulative IRF by maturity
  data/subperiod_results.csv   → IRF peak responses per sub-period
  plots/tp_series.png          → term premium time series (all maturities)
  plots/tp_channel_bar.png     → expectations vs TP channel decomposition
  plots/maturity_attenuation.png → IRF response profile across maturities
  plots/subperiod_irf.png      → IRF comparison across three regimes
  plots/tp_vs_repo.png         → term premium vs repo rate (scatter + time)
  plots/channel_heatmap.png    → channel share heatmap (maturity × horizon)

Dependencies: pandas, numpy, scipy, matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy import stats

os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────

YIELD_COLS  = ["Y1", "Y3", "Y5", "Y10"]
MAT_MONTHS  = {"Y1": 12, "Y3": 36, "Y5": 60, "Y10": 120}
MAT_LABELS  = {"Y1": "1Y", "Y3": "3Y", "Y5": "5Y", "Y10": "10Y"}
VAR_ORDER   = ["Repo", "Y1", "Y3", "Y5", "Y10"]

# Sub-period definitions (RBI policy regime breaks)
SUBPERIODS  = {
    "Pre-IT\n(2006–2013)"   : ("2006-04-01", "2013-09-30"),
    "IT Framework\n(2013–2020)": ("2013-10-01", "2020-03-31"),
    "COVID+Tightening\n(2020–2025)": ("2020-04-01", "2025-03-31"),
}

N_BOOT    = 300   # bootstrap replications for sub-period IRFs
HORIZON   = 24    # IRF horizon in months
CI_LEVEL  = 0.90

COLORS = {
    "Y1"  : "#2e8b57",
    "Y3"  : "#d4813a",
    "Y5"  : "#1f5fa6",
    "Y10" : "#c0392b",
    "Repo": "#8b3a8b",
    "Exp" : "#1f5fa6",
    "TP"  : "#c0392b",
}
SP_COLORS = ["#1f5fa6", "#2e8b57", "#c0392b"]

print("=" * 62)
print("  SECTION 4 — TRANSMISSION ANALYSIS")
print("=" * 62)

# ── 1. LOAD ALL INPUTS ────────────────────────────────────────────────────────

mp    = pd.read_csv("data/monthly_panel.csv",
                    index_col=0, parse_dates=True)
B_df  = pd.read_csv("data/svar_var_coefficients.csv", index_col=0)
ic_df = pd.read_csv("data/svar_lag_selection.csv", index_col=0)
p_sel = int(ic_df["BIC"].idxmin())

irf_ci = pd.read_csv("data/svar_irf_ci.csv")
fevd   = pd.read_csv("data/svar_fevd.csv")

avail = [v for v in VAR_ORDER if v in mp.columns]
K     = len(avail)

print(f"\n  Variables : {avail}")
print(f"  VAR lag   : p = {p_sel}")
print(f"  Sample    : {mp.index[0].date()} – {mp.index[-1].date()}  (T={len(mp)})")

# ── Rebuild companion matrix from saved coefficients ──────────────────────────

B_arr = B_df.values   # shape (1 + K*p, K)

A_comp = np.zeros((K * p_sel, K * p_sel))
for i in range(p_sel):
    A_comp[:K, i*K:(i+1)*K] = B_arr[1 + i*K : 1 + (i+1)*K, :].T

# Pre-compute A^h powers for all needed horizons
max_mat  = max(MAT_MONTHS.values())
Ah_cache = [np.eye(K)]
for h in range(1, max_mat + 1):
    Ah_cache.append(Ah_cache[-1] @ A_comp)

e_repo = np.zeros(K)
e_repo[0] = 1.0       # selection vector for Repo (first variable)

print("\n  Companion matrix eigenvalues (all < 1 → VAR stable):")
eigs = np.abs(np.linalg.eigvals(A_comp))
print(f"    max|eigenvalue| = {eigs.max():.4f}  "
      f"({'STABLE' if eigs.max() < 1 else 'UNSTABLE — check VAR'})")


# ══════════════════════════════════════════════════════════════════════════════
# 4A. TERM PREMIUM DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/4] 4A — Term Premium Decomposition ...")

dZ     = mp[avail].diff().dropna()
T_d    = len(dZ)
dates  = dZ.index

exp_rates = pd.DataFrame(index=dates, columns=YIELD_COLS, dtype=float)
term_prem = pd.DataFrame(index=dates, columns=YIELD_COLS, dtype=float)

for t_idx in range(T_d):
    z_t        = dZ.values[t_idx]
    repo_level = mp["Repo"].iloc[t_idx + 1]

    for col in YIELD_COLS:
        if col not in avail:
            continue
        tau   = MAT_MONTHS[col]
        y_t   = mp[col].iloc[t_idx + 1]

        # E[Δrepo_{t+h}] = e_repo' A^h z_t  for h = 1..tau
        exp_changes = np.array([e_repo @ Ah_cache[h] @ z_t
                                for h in range(1, tau + 1)])
        # Expected repo level path
        exp_levels  = repo_level + np.cumsum(exp_changes)
        # Expectations component = average expected short rate over maturity
        exp_rate    = float(exp_levels.mean())
        # Term premium = actual yield minus expectations component
        tp_val      = float(y_t) - exp_rate

        exp_rates.at[dates[t_idx], col] = exp_rate
        term_prem.at[dates[t_idx], col] = tp_val

# Save decomposition
tp_out = pd.DataFrame(index=dates)
for col in YIELD_COLS:
    tp_out[f"yield_{col}"]   = mp[col].iloc[1:1 + T_d].values
    tp_out[f"exp_rate_{col}"] = exp_rates[col].values
    tp_out[f"term_prem_{col}"] = term_prem[col].values
tp_out.index.name = "Date"
tp_out.to_csv("data/tp_decomposition.csv")
print("  Saved  data/tp_decomposition.csv")

# Variance share decomposition: Var(yield) ≈ Var(exp) + Var(TP) + 2·Cov
var_rows = []
for col in YIELD_COLS:
    y    = mp[col].iloc[1:1 + T_d].values.astype(float)
    er   = exp_rates[col].values.astype(float)
    tp   = term_prem[col].values.astype(float)
    mask = ~(np.isnan(y) | np.isnan(er) | np.isnan(tp))
    y, er, tp = y[mask], er[mask], tp[mask]

    var_y      = float(np.var(y, ddof=1))
    var_er     = float(np.var(er, ddof=1))
    var_tp     = float(np.var(tp, ddof=1))
    cov_er_tp  = float(np.cov(er, tp)[0, 1])
    corr_er_y, _ = stats.pearsonr(er, y)
    corr_tp_y, _ = stats.pearsonr(tp, y)

    var_rows.append({
        "Maturity"       : col,
        "Var(Yield)"     : round(var_y, 5),
        "Var(Exp)"       : round(var_er, 5),
        "Var(TP)"        : round(var_tp, 5),
        "2*Cov(Exp,TP)"  : round(2 * cov_er_tp, 5),
        "Exp share (%)"  : round(var_er / var_y * 100, 2),
        "TP share (%)"   : round(var_tp / var_y * 100, 2),
        "Corr(Exp,Yield)": round(corr_er_y, 4),
        "Corr(TP,Yield)" : round(corr_tp_y, 4),
    })

var_df = pd.DataFrame(var_rows).set_index("Maturity")
var_df.to_csv("data/tp_variance_shares.csv")

print("\n  Variance decomposition (Expectations vs Term Premium):")
print(var_df[["Var(Exp)","Var(TP)","Exp share (%)","TP share (%)","Corr(TP,Yield)"]].to_string())
print("  Saved  data/tp_variance_shares.csv")

# Mean term premium by maturity
print("\n  Mean term premium by maturity:")
for col in YIELD_COLS:
    tp_vals = term_prem[col].dropna()
    print(f"    {col}: mean={tp_vals.mean():.3f}%  "
          f"std={tp_vals.std():.3f}%  "
          f"min={tp_vals.min():.3f}%  max={tp_vals.max():.3f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 4B. MATURITY ATTENUATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2/4] 4B — Maturity Attenuation Analysis ...")

avail_yields = [c for c in YIELD_COLS if c in avail]
horizons     = np.arange(HORIZON + 1)

atten_rows = []
for col in avail_yields:
    sub = irf_ci[(irf_ci["shock"] == "Repo") & (irf_ci["response"] == col)]
    sub = sub.sort_values("horizon").reset_index(drop=True)

    irf_vals = sub["irf"].values
    lo_vals  = sub["ci_lo"].values
    hi_vals  = sub["ci_hi"].values

    # Peak response (absolute value)
    peak_idx    = np.argmax(np.abs(irf_vals))
    peak_h      = int(sub.at[peak_idx, "horizon"])
    peak_v      = float(irf_vals[peak_idx])

    # Cumulative IRF (h=0..24) — total area under response curve
    cum_irf     = float(irf_vals.sum())

    # Significant horizons: CI does not contain zero
    sig_hs      = [int(sub.at[i, "horizon"])
                   for i in range(len(sub))
                   if not (lo_vals[i] <= 0 <= hi_vals[i])]
    first_sig   = min(sig_hs) if sig_hs else None
    last_sig    = max(sig_hs) if sig_hs else None
    n_sig       = len(sig_hs)

    # h=1 response (one month after shock)
    h1_irf = float(sub[sub["horizon"] == 1]["irf"].values[0])
    h1_lo  = float(sub[sub["horizon"] == 1]["ci_lo"].values[0])
    h1_hi  = float(sub[sub["horizon"] == 1]["ci_hi"].values[0])

    atten_rows.append({
        "Maturity"          : col,
        "Maturity (months)" : MAT_MONTHS[col],
        "Peak response"     : round(peak_v, 5),
        "Peak at horizon"   : peak_h,
        "h=1 response"      : round(h1_irf, 5),
        "h=1 CI lo"         : round(h1_lo, 5),
        "h=1 CI hi"         : round(h1_hi, 5),
        "h=1 significant"   : "Yes" if not (h1_lo <= 0 <= h1_hi) else "No",
        "Cumulative IRF"    : round(cum_irf, 5),
        "N sig horizons"    : n_sig,
        "First sig horizon" : first_sig,
        "Last sig horizon"  : last_sig,
    })

atten_df = pd.DataFrame(atten_rows).set_index("Maturity")
atten_df.to_csv("data/maturity_attenuation.csv")

print("\n  IRF response to Repo shock by maturity:")
print(atten_df[["Peak response","Peak at horizon","h=1 response",
                 "h=1 significant","Cumulative IRF","N sig horizons"]].to_string())
print("  Saved  data/maturity_attenuation.csv")

# Attenuation slope: regress cumulative IRF on log(maturity)
log_mat  = np.log([MAT_MONTHS[c] for c in avail_yields])
cum_irfs = atten_df.loc[avail_yields, "Cumulative IRF"].values.astype(float)
_atten_reg  = stats.linregress(log_mat, cum_irfs)
slope       = float(_atten_reg.slope)
intercept   = float(_atten_reg.intercept)
r_atten     = float(_atten_reg.rvalue)
p_att       = float(_atten_reg.pvalue)
print(f"\n  Attenuation regression: CumIRF = {intercept:.4f} + {slope:.4f}·ln(τ)")
print(f"    R² = {r_atten**2:.4f}  p = {p_att:.4f}  "
      f"({'significant attenuation' if p_att < 0.05 else 'no significant attenuation'})")


# ══════════════════════════════════════════════════════════════════════════════
# 4C. CHANNEL DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3/4] 4C — Expectations vs Term Premium Channel Decomposition ...")

# For each maturity and horizon, decompose the IRF response into:
# (1) Expectations channel: change in avg expected short rate from shock
# (2) Term premium channel: residual (change in TP = IRF - expectations channel)
#
# Method: apply a 1-unit policy shock (ε_repo = 1) to the VAR state
# and trace out the implied expected short rate path change.
# The difference between total IRF and expectations channel = TP channel.

# Load Cholesky matrix to translate a structural shock into VAR state space
# P (Cholesky lower triangular) maps structural shocks to reduced-form residuals
# We need the first column of P (repo shock impact)
# P[0,0] = impact of repo shock on repo (the structural shock size)
# We can reconstruct from: P = chol(Sigma_u), and Sigma_u from residuals

res_df   = pd.read_csv("data/svar_residuals.csv", index_col=0, parse_dates=True)
U        = res_df[[f"shock_{v}" for v in avail]].values

# Reconstruct Sigma_u from dZ
dZ_full  = mp[avail].diff().dropna()
B_arr2   = B_df.values
X_full   = np.ones((len(dZ_full) - p_sel, 1 + K * p_sel))
for i in range(p_sel):
    X_full[:, 1 + i*K:1 + (i+1)*K] = dZ_full.values[p_sel-1-i: len(dZ_full)-1-i]
Y_dep    = dZ_full.values[p_sel:]
U_rf     = Y_dep - X_full @ B_arr2          # reduced-form residuals
T_eff    = len(U_rf)
Sigma_u  = (U_rf.T @ U_rf) / (T_eff - 1 - K * p_sel)
P_chol   = np.linalg.cholesky(Sigma_u)

# Impact vector: 1-unit structural repo shock → initial VAR state
impact_0 = P_chol[:, 0]   # first column of P = impact of ε_repo on all vars

channel_rows = []
for col in avail_yields:
    tau = MAT_MONTHS[col]
    for h_resp in range(HORIZON + 1):
        # Total IRF at horizon h_resp (from saved data)
        sub_row  = irf_ci[(irf_ci["shock"] == "Repo") &
                           (irf_ci["response"] == col) &
                           (irf_ci["horizon"] == h_resp)]
        total_irf = float(sub_row["irf"].values[0]) if len(sub_row) else np.nan

        # Expectations channel at horizon h_resp:
        # After shock at t=0, at horizon h, the VAR state is A^h @ impact_0
        # Expected short rate path change = avg of E[Δrepo] from h to h+tau
        z_shock = impact_0.copy()   # initial shock at h=0

        # State at horizon h_resp
        z_h = Ah_cache[h_resp] @ z_shock

        # Change in expected repo path from this new state z_h
        # E[Δrepo_{h+k}] = e_repo' A^k z_h  for k = 1..tau
        exp_changes_h = np.array([
            float(e_repo @ Ah_cache[k] @ z_h)
            for k in range(1, tau + 1)
        ])
        # Expectations channel = change in average expected short rate
        exp_channel = float(exp_changes_h.mean())
        # Term premium channel = total response - expectations channel
        tp_channel  = total_irf - exp_channel

        # Shares (as % of absolute total response)
        abs_total = abs(total_irf) if abs(total_irf) > 1e-8 else 1e-8
        channel_rows.append({
            "Maturity"        : col,
            "Horizon"         : h_resp,
            "Total IRF"       : round(total_irf,   6),
            "Exp channel"     : round(exp_channel,  6),
            "TP channel"      : round(tp_channel,   6),
            "Exp share (%)"   : round(exp_channel / abs_total * 100, 2),
            "TP share (%)"    : round(tp_channel  / abs_total * 100, 2),
        })

chan_df = pd.DataFrame(channel_rows)
chan_df.to_csv("data/channel_decomp.csv", index=False)

print("\n  Channel decomposition at h=0,1,3,6 by maturity:")
print(f"  {'Mat':5s}  {'h':3s}  {'Total':8s}  {'Exp ch':8s}  {'TP ch':8s}  "
      f"{'Exp%':7s}  {'TP%':7s}")
print(f"  {'-'*55}")
for col in avail_yields:
    for h in [0, 1, 3, 6]:
        row = chan_df[(chan_df["Maturity"] == col) & (chan_df["Horizon"] == h)]
        if len(row):
            r = row.iloc[0]
            print(f"  {col:5s}  {h:3d}  {r['Total IRF']:8.4f}  "
                  f"{r['Exp channel']:8.4f}  {r['TP channel']:8.4f}  "
                  f"{r['Exp share (%)']:6.1f}%  {r['TP share (%)']:6.1f}%")
    print()
print("  Saved  data/channel_decomp.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 4D. SUB-PERIOD STRUCTURAL STABILITY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4/4] 4D — Sub-period Structural Stability Analysis ...")


def var_ols(Y, p):
    T, K = Y.shape
    X    = np.ones((T - p, 1 + K * p))
    for i in range(p):
        X[:, 1 + i*K : 1 + (i+1)*K] = Y[p - 1 - i : T - 1 - i]
    Y_dep  = Y[p:]
    B      = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    U      = Y_dep - X @ B
    Sigma  = (U.T @ U) / (len(U) - 1 - K * p)
    return B, U, Sigma, X


def companion_matrix(B, p, K):
    A = np.zeros((K * p, K * p))
    for i in range(p):
        A[:K, i*K:(i+1)*K] = B[1 + i*K:1 + (i+1)*K, :].T
    if p > 1:
        A[K:, :-K] = np.eye(K * (p - 1))
    return A


def compute_irf(B, P, p, K, horizon=24):
    A    = companion_matrix(B, p, K)
    J    = np.zeros((K, K * p)); J[:, :K] = np.eye(K)
    irfs = np.zeros((horizon + 1, K, K))
    Phi  = np.eye(K * p)
    for h in range(horizon + 1):
        irfs[h] = J @ Phi @ J.T @ P
        Phi = Phi @ A
    return irfs


sp_results = {}
sp_rows    = []

dZ_full = mp[avail].diff().dropna()

for sp_name, (start, end) in SUBPERIODS.items():
    sp_data = dZ_full[(dZ_full.index >= start) & (dZ_full.index <= end)]
    Y_sp    = sp_data.values
    T_sp    = len(Y_sp)

    if T_sp < (K * p_sel + 10):
        print(f"  {sp_name}: too few obs ({T_sp}) — skipped")
        continue

    try:
        B_sp, U_sp, S_sp, _ = var_ols(Y_sp, p_sel)
        eigs = np.linalg.eigvalsh(S_sp)
        if eigs.min() <= 0:
            print(f"  {sp_name}: Sigma not PD — skipped")
            continue
        P_sp   = np.linalg.cholesky(S_sp)
        irfs_sp = compute_irf(B_sp, P_sp, p_sel, K, HORIZON)
    except np.linalg.LinAlgError:
        print(f"  {sp_name}: LinAlgError — skipped")
        continue

    # Bootstrap CI for sub-period
    np.random.seed(2024)
    boot_sp = np.zeros((N_BOOT, HORIZON + 1, K, K))
    T_eff_sp = T_sp - p_sel
    for b in range(N_BOOT):
        idx   = np.random.choice(T_eff_sp, T_eff_sp, replace=True)
        U_b   = U_sp[idx]
        Y_b   = np.zeros((T_eff_sp + p_sel, K))
        Y_b[:p_sel] = Y_sp[:p_sel]
        for t in range(T_eff_sp):
            Xrow = np.ones(1 + K * p_sel)
            for i in range(p_sel):
                Xrow[1+i*K:1+(i+1)*K] = Y_b[p_sel+t-1-i]
            Y_b[p_sel+t] = Xrow @ B_sp + U_b[t]
        try:
            Bb, _, Sb, _ = var_ols(Y_b, p_sel)
            if np.linalg.eigvalsh(Sb).min() > 0:
                Pb = np.linalg.cholesky(Sb)
                boot_sp[b] = compute_irf(Bb, Pb, p_sel, K, HORIZON)
            else:
                boot_sp[b] = irfs_sp
        except Exception:
            boot_sp[b] = irfs_sp

    ci_lo_sp = np.percentile(boot_sp,  5, axis=0)
    ci_hi_sp = np.percentile(boot_sp, 95, axis=0)

    sp_results[sp_name] = {
        "irfs"   : irfs_sp,
        "ci_lo"  : ci_lo_sp,
        "ci_hi"  : ci_hi_sp,
        "T"      : T_sp,
        "start"  : start,
        "end"    : end,
    }

    repo_j = avail.index("Repo")
    for col in avail_yields:
        i       = avail.index(col)
        irf_v   = irfs_sp[:, i, repo_j]
        peak_v  = float(irf_v[np.argmax(np.abs(irf_v))])
        cum_v   = float(irf_v.sum())
        h1_v    = float(irf_v[1]) if HORIZON >= 1 else np.nan
        sp_rows.append({
            "Period"     : sp_name.replace("\n", " "),
            "Maturity"   : col,
            "T (months)" : T_sp,
            "Peak IRF"   : round(peak_v, 5),
            "h=1 IRF"    : round(h1_v,   5),
            "Cum IRF"    : round(cum_v,   5),
        })

    print(f"  {sp_name.replace(chr(10),' ')} "
          f"(T={T_sp}): estimated ✓  (boot CI done)")

sp_df = pd.DataFrame(sp_rows)
sp_df.to_csv("data/subperiod_results.csv", index=False)
print("\n  Sub-period peak IRF comparison:")
print(sp_df.pivot(index="Maturity", columns="Period", values="Cum IRF").round(4).to_string())
print("  Saved  data/subperiod_results.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Plots] Generating all Section 4 figures ...")

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : False,
})


# ── Plot 1: Term premium time series (all maturities) ───────────────────────
fig, axes = plt.subplots(len(avail_yields), 1,
                          figsize=(13, 2.8 * len(avail_yields)), sharex=True)
if len(avail_yields) == 1:
    axes = [axes]

for ax, col in zip(axes, avail_yields):
    tp_vals  = term_prem[col].values
    exp_vals = exp_rates[col].values
    y_vals   = mp[col].iloc[1:1 + T_d].values

    ax.fill_between(dates, 0, tp_vals,
                    where=(tp_vals >= 0), color=COLORS["TP"],  alpha=0.35)
    ax.fill_between(dates, 0, tp_vals,
                    where=(tp_vals < 0),  color="#1f5fa6", alpha=0.35)
    ax.plot(dates, tp_vals, color=COLORS["TP"], linewidth=1.5,
            label="Term premium")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"{MAT_LABELS[col]} TP (%)", fontsize=9)
    ax.text(0.01, 0.92, f"Mean = {np.nanmean(tp_vals):.2f}%",
            transform=ax.transAxes, fontsize=8, color="#333333")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

# Mark RBI regime breaks
for ax in axes:
    for ds, lbl in [("2013-10-01", "IT\n2013"), ("2020-04-01", "COVID\n2020")]:
        xv = pd.to_datetime(ds)
        ax.axvline(xv, color="#aaaaaa", linewidth=0.8, linestyle=":")

fig.suptitle("Term Premium Decomposition — VAR-Implied Expectations Channel\n"
             "(Term Premium = Yield − Expected Short Rate Path)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/tp_series.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/tp_series.png")


# ── Plot 2: Expectations vs TP channel bar chart ─────────────────────────────
fig, axes = plt.subplots(1, len(avail_yields),
                          figsize=(3.8 * len(avail_yields), 5), sharey=False)
if len(avail_yields) == 1:
    axes = [axes]

key_horizons = [0, 1, 3, 6, 12]
x_pos        = np.arange(len(key_horizons))
bar_w        = 0.35

for ax, col in zip(axes, avail_yields):
    sub = chan_df[chan_df["Maturity"] == col].set_index("Horizon")

    exp_vals = [sub.loc[h, "Exp channel"] if h in sub.index else 0
                for h in key_horizons]
    tp_vals  = [sub.loc[h, "TP channel"]  if h in sub.index else 0
                for h in key_horizons]

    bars_exp = ax.bar(x_pos - bar_w/2, exp_vals, bar_w,
                      color=COLORS["Exp"], alpha=0.80,
                      label="Expectations channel")
    bars_tp  = ax.bar(x_pos + bar_w/2, tp_vals,  bar_w,
                      color=COLORS["TP"],  alpha=0.80,
                      label="Term premium channel")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"h={h}" for h in key_horizons], fontsize=8)
    ax.set_title(f"{MAT_LABELS[col]} Yield", fontsize=10, fontweight="bold")
    ax.set_ylabel("Response (pp)" if ax == axes[0] else "", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

fig.suptitle("Channel Decomposition: Expectations vs Term Premium\n"
             "Response of Yield to 1-unit RBI Repo Rate Shock",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/tp_channel_bar.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/tp_channel_bar.png")


# ── Plot 3: Maturity attenuation profile ────────────────────────────────────
horizons_plot = np.arange(HORIZON + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: IRF by maturity (overlaid)
ax = axes[0]
for col in avail_yields:
    sub   = irf_ci[(irf_ci["shock"] == "Repo") & (irf_ci["response"] == col)]
    sub   = sub.sort_values("horizon")
    color = COLORS.get(col, "gray")
    ax.plot(sub["horizon"], sub["irf"],
            color=color, linewidth=2.0, label=f"{MAT_LABELS[col]} G-Sec")
    ax.fill_between(sub["horizon"], sub["ci_lo"], sub["ci_hi"],
                    color=color, alpha=0.10)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Months after shock", fontsize=10)
ax.set_ylabel("Response (pp)", fontsize=10)
ax.set_title("IRF by Maturity — Repo Shock", fontsize=11, fontweight="bold")
ax.set_xticks([0, 6, 12, 18, 24])
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2, linewidth=0.5)

# Right: Cumulative IRF vs log(maturity) — attenuation slope
ax2 = axes[1]
log_mat_vals = np.array([np.log(MAT_MONTHS[c]) for c in avail_yields])
cum_irf_vals = atten_df.loc[avail_yields, "Cumulative IRF"].values.astype(float)

for col, xv, yv in zip(avail_yields, log_mat_vals, cum_irf_vals):
    ax2.scatter(xv, yv, color=COLORS.get(col, "gray"), s=80,
                zorder=4, edgecolors="white", linewidths=0.8)
    ax2.text(xv + 0.04, yv, MAT_LABELS[col], fontsize=9,
             color=COLORS.get(col, "gray"))

# Regression line
x_line = np.linspace(log_mat_vals.min() - 0.3, log_mat_vals.max() + 0.3, 100)
y_line = intercept + slope * x_line
ax2.plot(x_line, y_line, "--", color="#555555", linewidth=1.2, alpha=0.7)
ax2.text(0.05, 0.92, f"slope = {slope:.4f}  R00b2 = {r_atten**2:.3f}  p = {p_att:.3f}",
         transform=ax2.transAxes, fontsize=8.5, color="#333333")

ax2.set_xlabel("ln(Maturity in months)", fontsize=10)
ax2.set_ylabel("Cumulative IRF (h=0..24)", fontsize=10)
ax2.set_xticks([np.log(MAT_MONTHS[c]) for c in avail_yields])
ax2.set_xticklabels([f"ln({MAT_LABELS[c]})" for c in avail_yields], fontsize=8)
ax2.set_title("Attenuation: Cumulative Response vs Maturity",
              fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.2, linewidth=0.5)

fig.suptitle("Maturity Attenuation of Monetary Policy Transmission",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/maturity_attenuation.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/maturity_attenuation.png")


# ── Plot 4: Sub-period IRF comparison ────────────────────────────────────────
if sp_results:
    n_sp    = len(sp_results)
    sp_names = list(sp_results.keys())
    repo_j   = avail.index("Repo")

    fig, axes = plt.subplots(len(avail_yields), n_sp,
                              figsize=(4.5 * n_sp, 3.5 * len(avail_yields)),
                              sharex=True)

    if len(avail_yields) == 1:
        axes = axes.reshape(1, -1)
    if n_sp == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, col in enumerate(avail_yields):
        i = avail.index(col)
        for sp_idx, sp_name in enumerate(sp_names):
            ax     = axes[col_idx, sp_idx]
            res    = sp_results[sp_name]
            irfs_v = res["irfs"][:, i, repo_j]
            lo_v   = res["ci_lo"][:, i, repo_j]
            hi_v   = res["ci_hi"][:, i, repo_j]
            color  = SP_COLORS[sp_idx % len(SP_COLORS)]
            h_arr  = np.arange(len(irfs_v))

            ax.fill_between(h_arr, lo_v, hi_v, color=color, alpha=0.15)
            ax.plot(h_arr, lo_v, color=color, linewidth=0.5,
                    linestyle="--", alpha=0.5)
            ax.plot(h_arr, hi_v, color=color, linewidth=0.5,
                    linestyle="--", alpha=0.5)
            ax.plot(h_arr, irfs_v, color=color, linewidth=2.0)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)

            if col_idx == 0:
                ax.set_title(sp_name + f"\n(T={res['T']})",
                             fontsize=9, fontweight="bold")
            if sp_idx == 0:
                ax.set_ylabel(f"→ {MAT_LABELS[col]}", fontsize=9)
            if col_idx == len(avail_yields) - 1:
                ax.set_xlabel("Months", fontsize=8)

    fig.suptitle("Sub-period Stability: IRF Repo Shock → Yields\n"
                 "90% Bootstrap CI per Regime",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/subperiod_irf.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  plots/subperiod_irf.png")


# ── Plot 5: Term premium vs Repo rate (scatter + joint time axis) ────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Scatter TP(10Y) vs Repo
ax = axes[0]
tp10 = term_prem["Y10"].values if "Y10" in avail_yields else term_prem[avail_yields[-1]].values
repo_aligned = mp["Repo"].iloc[1:1 + T_d].values
col_scatter  = "Y10" if "Y10" in avail_yields else avail_yields[-1]

scatter_c = np.linspace(0, 1, len(tp10))
sc = ax.scatter(repo_aligned, tp10, c=scatter_c, cmap="RdYlGn_r",
                s=18, alpha=0.7, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Time (early→late)", shrink=0.8)

slope_s, intercept_s, r_s, p_s, _ = stats.linregress(repo_aligned, tp10)
x_r = np.linspace(repo_aligned.min(), repo_aligned.max(), 100)
ax.plot(x_r, intercept_s + slope_s * x_r, "--",
        color="#c0392b", linewidth=1.5, alpha=0.8)
ax.text(0.05, 0.93,
        f"slope={slope_s:.3f}  R²={r_s**2:.3f}  p={p_s:.3f}",
        transform=ax.transAxes, fontsize=8.5)
ax.set_xlabel("Repo Rate (%)", fontsize=10)
ax.set_ylabel(f"{MAT_LABELS[col_scatter]} Term Premium (%)", fontsize=10)
ax.set_title("Term Premium vs Policy Rate\n(colour = time)", fontsize=11,
             fontweight="bold")
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.grid(alpha=0.15)

# Right: Overlaid time series (TP 10Y + Repo)
ax2 = axes[1]
ax2_twin = ax2.twinx()
ax2.plot(dates, tp10, color=COLORS["TP"], linewidth=1.8,
         label=f"{MAT_LABELS[col_scatter]} Term Premium")
ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax2_twin.plot(dates, repo_aligned, color=COLORS["Repo"],
              linewidth=1.5, linestyle="--", alpha=0.8, label="Repo Rate")
ax2.set_ylabel(f"{MAT_LABELS[col_scatter]} Term Premium (%)", fontsize=9,
               color=COLORS["TP"])
ax2_twin.set_ylabel("Repo Rate (%)", fontsize=9, color=COLORS["Repo"])
ax2.set_title("Term Premium and Repo Rate Over Time", fontsize=11,
              fontweight="bold")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(3))

lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc="upper right")
ax2.grid(axis="y", alpha=0.15)

for ds in ["2013-10-01","2020-04-01"]:
    xv = pd.to_datetime(ds)
    ax2.axvline(xv, color="#aaaaaa", linewidth=0.8, linestyle=":")

fig.suptitle("Term Premium Relationship with Monetary Policy",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/tp_vs_repo.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/tp_vs_repo.png")


# ── Plot 6: Channel share heatmap (maturity × horizon) ───────────────────────
key_hs  = [0, 1, 2, 3, 6, 9, 12, 18, 24]
exp_mat = np.zeros((len(avail_yields), len(key_hs)))
tp_mat  = np.zeros((len(avail_yields), len(key_hs)))

for i, col in enumerate(avail_yields):
    for j, h in enumerate(key_hs):
        row = chan_df[(chan_df["Maturity"] == col) & (chan_df["Horizon"] == h)]
        if len(row):
            exp_mat[i, j] = row.iloc[0]["Exp share (%)"]
            tp_mat[i, j]  = row.iloc[0]["TP share (%)"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
y_labels = [MAT_LABELS[c] for c in avail_yields]
x_labels = [f"h={h}" for h in key_hs]

for ax, data, title, cmap in [
    (axes[0], exp_mat, "Expectations Channel Share (%)", "Blues"),
    (axes[1], tp_mat,  "Term Premium Channel Share (%)", "Reds"),
]:
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-100, vmax=200)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xticks(range(len(key_hs)))
    ax.set_yticks(range(len(avail_yields)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Horizon", fontsize=9)
    ax.set_ylabel("Maturity", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(len(avail_yields)):
        for j in range(len(key_hs)):
            ax.text(j, i, f"{data[i,j]:.0f}%",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(data[i,j]) > 120 else "#333333")

fig.suptitle("Channel Decomposition: Share of IRF Explained by Each Channel\n"
             "(Maturity × Horizon)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/channel_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/channel_heatmap.png")


# ── 9. PAPER SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 4 COMPLETE — RESULTS FOR PAPER")
print("=" * 62)
print(f"""
  4A — Term Premium Decomposition
       data/tp_decomposition.csv   → full time series
       data/tp_variance_shares.csv → % variance explained by each channel
       plots/tp_series.png         → TP time series by maturity
       plots/tp_vs_repo.png        → TP vs repo rate relationship

  4B — Maturity Attenuation
       data/maturity_attenuation.csv → peak, cumul, sig horizons by maturity
       plots/maturity_attenuation.png → IRF overlay + attenuation slope

  4C — Channel Decomposition
       data/channel_decomp.csv     → exp vs TP share by maturity + horizon
       plots/tp_channel_bar.png    → bar chart of channel shares
       plots/channel_heatmap.png   → maturity × horizon heatmap

  4D — Sub-period Stability
       data/subperiod_results.csv  → peak/cumul IRF per regime
       plots/subperiod_irf.png     → IRF grid across three regimes

  Key numbers for null hypothesis tests:
    H0 Part 2 (IRF significance):
      → See maturity_attenuation.csv: 'h=1 significant' column
    H0 Part 3 (expectations vs term premium):
      → See tp_variance_shares.csv: 'Exp share %' and 'TP share %'
      → See channel_decomp.csv: channel breakdown at key horizons

  Next: python3 hypothesis_tests.py  (Section 5 — Formal Tests)
""")