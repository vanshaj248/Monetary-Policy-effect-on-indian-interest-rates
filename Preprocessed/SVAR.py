"""
Section 3 — Monetary Policy Shock Identification
Structural VAR (SVAR) with Cholesky Decomposition
==================================================
Paper: Monetary Policy Transmission to the Indian Yield Curve

Theory
------
We estimate a reduced-form VAR(p) on first-differenced yields and the repo rate:

    ΔZ_t = c + A1·ΔZ_{t-1} + ... + Ap·ΔZ_{t-p} + u_t

where Z_t = [Repo_t, Y1_t, Y3_t, Y5_t, Y10_t]' and u_t ~ (0, Σ_u).

Structural shocks ε_t are recovered via Cholesky decomposition of Σ_u:
    u_t = P · ε_t,   P = chol(Σ_u)  (lower triangular)

Ordering: Repo first (RBI sets policy before bond markets react within
a month). This is the standard recursive identification used in the
monetary policy literature (Christiano, Eichenbaum & Evans 1999).

This gives us the structural monetary policy shock as the first element
of ε_t — the component of repo rate changes orthogonal to all yield
information within the period.

Outputs
-------
  data/svar_var_coefficients.csv   → reduced-form VAR coefficient matrices
  data/svar_residuals.csv          → structural residuals ε_t (all K shocks)
  data/svar_irf.csv                → point-estimate IRFs (all response/shock pairs)
  data/svar_irf_ci.csv             → bootstrapped 90% confidence bands
  data/svar_granger.csv            → Granger causality test results (H0 test 1)
  data/svar_fevd.csv               → forecast error variance decomposition
  data/svar_lag_selection.csv      → AIC/BIC table for lag order choice
  plots/svar_irf_repo_shock.png    → main IRF figure: repo shock → all yields
  plots/svar_irf_grid.png          → full IRF grid (K×K panel)
  plots/svar_fevd.png              → FEVD stacked bars per maturity
  plots/svar_residuals.png         → structural shock time series
  plots/svar_granger.png           → Granger causality summary table

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

N_BOOT    = 500    # bootstrap replications for IRF confidence bands
HORIZON   = 24     # IRF horizon in months
MAX_LAG   = 12     # maximum lag order to search
CI_LEVEL  = 0.90   # confidence interval level (90% is standard in SVAR papers)
ALPHA_CI  = (1 - CI_LEVEL) / 2

# Variable ordering: REPO MUST BE FIRST (Cholesky identification assumption)
VAR_ORDER = ["Repo", "Y1", "Y3", "Y5", "Y10"]

VAR_LABELS = {
    "Repo": "Repo Rate",
    "Y1" : "1Y G-Sec",
    "Y3" : "3Y G-Sec",
    "Y5" : "5Y G-Sec",
    "Y10": "10Y G-Sec",
}
COLORS = {
    "Repo": "#8b3a8b",
    "Y1"  : "#2e8b57",
    "Y3"  : "#d4813a",
    "Y5"  : "#1f5fa6",
    "Y10" : "#c0392b",
}

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

print("=" * 62)
print("  SECTION 3 — SVAR MONETARY POLICY SHOCK IDENTIFICATION")
print("=" * 62)

factors = pd.read_csv(
    "data/dns_factors.csv", index_col="Date", parse_dates=True
)

# Use actual yields (not DNS factors) — all are available after preprocessing
avail   = [v for v in VAR_ORDER if v in factors.columns]
missing = [v for v in VAR_ORDER if v not in factors.columns]

if missing:
    print(f"\n  WARNING: Missing columns: {missing}")
    print(f"  Running with available: {avail}")
    print(f"  Add 1Y/3Y/5Y data to monthly_panel.csv to enable full model.\n")

core = factors[avail].copy()
K    = len(avail)

# ── 2. FIRST-DIFFERENCE (yields are I(1) from Section 2 unit root tests) ────
dZ   = core.diff().dropna()
T, _ = dZ.shape
dates_diff = dZ.index

print(f"\n  Variables    : {avail}")
print(f"  Observations : {T}  (first-differenced, {dates_diff[0].date()} – {dates_diff[-1].date()})")
print(f"  VAR ordering : Repo first (Cholesky recursive identification)")

Y = dZ.values   # shape (T, K)


# ══════════════════════════════════════════════════════════════════════════════
# CORE VAR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_regressor(Y: np.ndarray, p: int) -> tuple:
    """
    Build OLS regressor matrix for VAR(p).
    Returns (X, Y_dep) where Y_dep = Y[p:] and X = [1, Y_{t-1}, ..., Y_{t-p}].
    """
    T, K = Y.shape
    X     = np.ones((T - p, 1 + K * p))
    for i in range(p):
        X[:, 1 + i*K : 1 + (i+1)*K] = Y[p - 1 - i : T - 1 - i]
    return X, Y[p:]


def var_ols(Y: np.ndarray, p: int) -> tuple:
    """
    Estimate VAR(p) by OLS.
    Returns (B, U, Sigma_u, X)
      B       : coefficient matrix (1 + K*p, K)  [intercept + lags]
      U       : residuals (T-p, K)
      Sigma_u : residual covariance (K, K)
      X       : regressor matrix
    """
    X, Y_dep = build_regressor(Y, p)
    T_eff    = len(Y_dep)
    n_params = K * (1 + K * p)
    B        = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    U        = Y_dep - X @ B
    Sigma_u  = (U.T @ U) / (T_eff - 1 - K * p)
    return B, U, Sigma_u, X


def companion_matrix(B: np.ndarray, p: int, K: int) -> np.ndarray:
    """Build VAR companion matrix from coefficient matrix B."""
    A = np.zeros((K * p, K * p))
    for i in range(p):
        A[:K, i*K : (i+1)*K] = B[1 + i*K : 1 + (i+1)*K, :].T
    if p > 1:
        A[K:, : -K] = np.eye(K * (p - 1))
    return A


def compute_irf(B: np.ndarray, P: np.ndarray, p: int, K: int,
                horizon: int = 24) -> np.ndarray:
    """
    Compute orthogonalised IRFs (Cholesky) for all K structural shocks.
    Returns array of shape (horizon+1, K, K):
      irf[h, i, j] = response of variable i to shock j at horizon h.
    """
    A    = companion_matrix(B, p, K)
    J    = np.zeros((K, K * p))
    J[:, :K] = np.eye(K)           # selection matrix (pick first K rows)
    irfs = np.zeros((horizon + 1, K, K))
    Phi  = np.eye(K * p)
    for h in range(horizon + 1):
        irfs[h] = J @ Phi @ J.T @ P
        Phi = Phi @ A
    return irfs


def compute_fevd(irfs: np.ndarray) -> np.ndarray:
    """
    Forecast Error Variance Decomposition from orthogonalised IRFs.
    Returns array (horizon+1, K, K):
      fevd[h, i, j] = fraction of variance of variable i at horizon h
                      attributable to shock j (rows sum to 1).
    """
    H = irfs.shape[0]
    cum_sq = np.cumsum(irfs**2, axis=0)        # cumulative squared IRFs
    total  = cum_sq.sum(axis=2, keepdims=True)  # total variance per var
    with np.errstate(divide="ignore", invalid="ignore"):
        fevd = np.where(total > 0, cum_sq / total, 0.0)
    return fevd


# ── 3. LAG SELECTION ─────────────────────────────────────────────────────────

print("\n[1/6] Lag order selection (AIC / BIC) ...")

ic_rows = []
for p in range(1, MAX_LAG + 1):
    try:
        B, U, Sigma_u, X = var_ols(Y, p)
        T_eff    = T - p
        n_params = K * (1 + K * p)
        sign, logdet = np.linalg.slogdet(Sigma_u)
        if sign <= 0:
            continue
        aic = logdet + 2 * n_params / T_eff
        bic = logdet + np.log(T_eff) * n_params / T_eff
        ic_rows.append({"Lag": p, "AIC": round(aic, 4), "BIC": round(bic, 4)})
    except Exception:
        pass

ic_df  = pd.DataFrame(ic_rows).set_index("Lag")
p_aic  = int(ic_df["AIC"].idxmin())
p_bic  = int(ic_df["BIC"].idxmin())
p_sel  = p_bic   # use BIC (more parsimonious; preferred in small samples)

ic_df.to_csv("data/svar_lag_selection.csv")
print(ic_df.to_string())
print(f"\n  AIC optimal lag: {p_aic}")
print(f"  BIC optimal lag: {p_bic}  ← selected (parsimonious, T={T})")
print("  Saved  data/svar_lag_selection.csv")


# ── 4. ESTIMATE VAR AND STRUCTURAL DECOMPOSITION ─────────────────────────────

print(f"\n[2/6] Estimating VAR({p_sel}) and Cholesky decomposition ...")

B, U, Sigma_u, X_reg = var_ols(Y, p_sel)
T_eff = T - p_sel

# Cholesky: Sigma_u = P P'   (lower triangular P)
P  = np.linalg.cholesky(Sigma_u)

# Structural shocks: ε_t = P^{-1} u_t
P_inv         = np.linalg.inv(P)
struct_shocks = U @ P_inv.T   # shape (T_eff, K)

# Check: structural shocks should be uncorrelated (near-identity covariance)
check_cov = np.corrcoef(struct_shocks.T)
print(f"\n  VAR({p_sel}) residual std per variable:")
for i, v in enumerate(avail):
    print(f"    {v:5s}  σ = {U[:,i].std():.5f}")
print(f"\n  Max off-diagonal correlation in structural shocks: "
      f"{np.abs(check_cov - np.eye(K)).max():.4f}  [should be ≈ 0]")

# Save VAR coefficients
coef_rows = []
row_labels = ["const"] + [f"{v}_lag{l+1}" for l in range(p_sel) for v in avail]
for j, label in enumerate(row_labels):
    row = {"param": label}
    for i, v in enumerate(avail):
        row[f"eq_{v}"] = round(B[j, i], 6)
    coef_rows.append(row)
pd.DataFrame(coef_rows).set_index("param").to_csv("data/svar_var_coefficients.csv")
print("  Saved  data/svar_var_coefficients.csv")

# Save structural shocks
shock_dates  = dates_diff[p_sel:]
shocks_df    = pd.DataFrame(
    struct_shocks,
    index=shock_dates,
    columns=[f"shock_{v}" for v in avail]
)
shocks_df.index.name = "Date"
shocks_df.to_csv("data/svar_residuals.csv")
print("  Saved  data/svar_residuals.csv")


# ── 5. IMPULSE RESPONSE FUNCTIONS ────────────────────────────────────────────

print(f"\n[3/6] Computing IRFs (horizon = {HORIZON} months) ...")

irfs  = compute_irf(B, P, p_sel, K, HORIZON)   # shape (H+1, K, K)

# Save point-estimate IRFs
irf_rows = []
for h in range(HORIZON + 1):
    for i, resp in enumerate(avail):
        for j, shock in enumerate(avail):
            irf_rows.append({
                "horizon"  : h,
                "response" : resp,
                "shock"    : shock,
                "irf"      : round(irfs[h, i, j], 6),
            })
irf_df = pd.DataFrame(irf_rows)
irf_df.to_csv("data/svar_irf.csv", index=False)
print("  Saved  data/svar_irf.csv")


# ── 6. BOOTSTRAP CONFIDENCE BANDS ────────────────────────────────────────────

print(f"\n[4/6] Bootstrapping {N_BOOT} replications for {int(CI_LEVEL*100)}% CI bands ...")

np.random.seed(2024)
boot_irfs = np.zeros((N_BOOT, HORIZON + 1, K, K))

for b in range(N_BOOT):
    # Residual bootstrap: resample VAR residuals with replacement
    idx  = np.random.choice(T_eff, T_eff, replace=True)
    U_b  = U[idx]

    # Generate bootstrap sample by iterating VAR forward from initial obs
    Y_b       = np.zeros((T_eff + p_sel, K))
    Y_b[:p_sel] = Y[:p_sel]
    for t in range(T_eff):
        Xrow = np.ones(1 + K * p_sel)
        for i in range(p_sel):
            Xrow[1 + i*K : 1 + (i+1)*K] = Y_b[p_sel + t - 1 - i]
        Y_b[p_sel + t] = Xrow @ B + U_b[t]

    try:
        Bb, Ub, Sb, _ = var_ols(Y_b, p_sel)
        # Check Sb is positive definite before Cholesky
        eigs = np.linalg.eigvalsh(Sb)
        if eigs.min() <= 0:
            boot_irfs[b] = irfs
            continue
        Pb               = np.linalg.cholesky(Sb)
        boot_irfs[b]     = compute_irf(Bb, Pb, p_sel, K, HORIZON)
    except np.linalg.LinAlgError:
        boot_irfs[b] = irfs   # fallback to point estimate

ci_lo = np.percentile(boot_irfs, ALPHA_CI * 100,       axis=0)
ci_hi = np.percentile(boot_irfs, (1 - ALPHA_CI) * 100, axis=0)

# Save CI
ci_rows = []
for h in range(HORIZON + 1):
    for i, resp in enumerate(avail):
        for j, shock in enumerate(avail):
            ci_rows.append({
                "horizon"  : h,
                "response" : resp,
                "shock"    : shock,
                "irf"      : round(irfs[h, i, j], 6),
                "ci_lo"    : round(ci_lo[h, i, j], 6),
                "ci_hi"    : round(ci_hi[h, i, j], 6),
            })
ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv("data/svar_irf_ci.csv", index=False)
print("  Saved  data/svar_irf_ci.csv")


# ── 7. FORECAST ERROR VARIANCE DECOMPOSITION ──────────────────────────────────

print("\n[5/6] Computing FEVD ...")

fevd  = compute_fevd(irfs)   # shape (H+1, K, K)

fevd_rows = []
for h in range(HORIZON + 1):
    for i, resp in enumerate(avail):
        for j, shock in enumerate(avail):
            fevd_rows.append({
                "horizon" : h,
                "response": resp,
                "shock"   : shock,
                "fevd"    : round(fevd[h, i, j], 6),
            })
fevd_df = pd.DataFrame(fevd_rows)
fevd_df.to_csv("data/svar_fevd.csv", index=False)

# Print FEVD at h=1, 6, 12, 24 for Repo shock → each yield
print(f"\n  FEVD: share of variance explained by Repo shock at key horizons")
print(f"  {'Maturity':10s}  h=1      h=6      h=12     h=24")
repo_j = avail.index("Repo")
for i, v in enumerate(avail):
    vals = [round(fevd[h, i, repo_j]*100, 2) for h in [1, 6, 12, 24]]
    print(f"  {v:10s}  {vals[0]:6.2f}%  {vals[1]:6.2f}%  {vals[2]:6.2f}%  {vals[3]:6.2f}%")
print("  Saved  data/svar_fevd.csv")


# ── 8. GRANGER CAUSALITY (H0 TEST 1) ─────────────────────────────────────────
# H0: Repo does NOT Granger-cause variable X
# Restricted model: omit all lags of Repo from equation for X
# Test: F-statistic on joint significance of excluded lags

print("\n[5b/6] Granger causality tests (H0: Repo does not Granger-cause yield) ...")


def granger_test(Y: np.ndarray, cause_idx: int, effect_idx: int,
                 p: int) -> dict:
    """
    F-test for Granger causality: does 'cause' variable Granger-cause 'effect'?
    Restricted model omits all p lags of cause variable from effect equation.
    """
    T, K = Y.shape
    X, Y_dep = build_regressor(Y, p)
    y_eq      = Y_dep[:, effect_idx]    # dependent variable

    # Unrestricted: full X
    Bu   = np.linalg.lstsq(X, y_eq, rcond=None)[0]
    RSS_u = ((y_eq - X @ Bu) ** 2).sum()

    # Restricted: remove columns corresponding to lags of cause variable
    # Cause lags are at columns: 1 + cause_idx, 1+K+cause_idx, 1+2K+cause_idx ...
    cause_cols = [1 + l * K + cause_idx for l in range(p)]
    mask_r     = np.ones(X.shape[1], dtype=bool)
    mask_r[cause_cols] = False
    X_r   = X[:, mask_r]
    Br    = np.linalg.lstsq(X_r, y_eq, rcond=None)[0]
    RSS_r = ((y_eq - X_r @ Br) ** 2).sum()

    T_eff = len(y_eq)
    q     = p                          # number of restrictions
    dof_u = T_eff - X.shape[1]        # unrestricted df
    F     = ((RSS_r - RSS_u) / q) / (RSS_u / dof_u)
    p_val = 1 - stats.f.cdf(F, q, dof_u)

    return {
        "Cause"    : avail[cause_idx],
        "Effect"   : avail[effect_idx],
        "F-stat"   : round(F, 4),
        "p-value"  : round(p_val, 4),
        "Lags"     : p,
        "Conclusion": "Granger-causes" if p_val < 0.05 else "Does NOT cause",
    }


repo_idx   = avail.index("Repo")
yield_vars = [v for v in avail if v != "Repo"]
gc_rows    = []

for v in avail:
    if v == "Repo":
        continue
    eff_idx = avail.index(v)
    # Repo → yield
    gc_rows.append(granger_test(Y, repo_idx, eff_idx, p_sel))
    # yield → Repo (reverse causality check)
    gc_rows.append(granger_test(Y, eff_idx, repo_idx, p_sel))

gc_df = pd.DataFrame(gc_rows)
gc_df.to_csv("data/svar_granger.csv", index=False)

print(f"\n  {'Cause':10s} → {'Effect':10s}  F-stat   p-value   Decision")
print(f"  {'-'*58}")
for _, row in gc_df.iterrows():
    sig = "***" if row["p-value"] < 0.01 else ("**" if row["p-value"] < 0.05
          else ("*" if row["p-value"] < 0.10 else ""))
    print(f"  {row['Cause']:10s} → {row['Effect']:10s}  "
          f"{row['F-stat']:7.4f}  {row['p-value']:7.4f}  {row['Conclusion']} {sig}")
print("\n  Significance: *** p<0.01  ** p<0.05  * p<0.10")
print("  Saved  data/svar_granger.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6/6] Generating plots ...")

horizons  = np.arange(HORIZON + 1)
repo_j    = avail.index("Repo")

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : False,
})


# ── Plot 1: Repo shock → all yield responses (MAIN PAPER FIGURE) ─────────────
# This is the central result of Section 3

yield_resp = [v for v in avail if v != "Repo"]
n_resp     = len(yield_resp)

fig, axes = plt.subplots(1, n_resp, figsize=(4.5 * n_resp, 5), sharey=False)
if n_resp == 1:
    axes = [axes]

for ax, v in zip(axes, yield_resp):
    i       = avail.index(v)
    irf_pts = irfs[:, i, repo_j]
    lo      = ci_lo[:, i, repo_j]
    hi      = ci_hi[:, i, repo_j]
    color   = COLORS.get(v, "#333333")

    # Confidence band
    ax.fill_between(horizons, lo, hi, color=color, alpha=0.15,
                    label=f"{int(CI_LEVEL*100)}% CI")
    # CI borders
    ax.plot(horizons, lo, color=color, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.plot(horizons, hi, color=color, linewidth=0.6, linestyle="--", alpha=0.5)
    # Point estimate
    ax.plot(horizons, irf_pts, color=color, linewidth=2.2,
            label="IRF point estimate")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-")

    # Mark first horizon where CI crosses zero (fades out)
    sig_end = HORIZON
    for h in range(1, HORIZON + 1):
        if lo[h] <= 0 <= hi[h]:
            sig_end = h
            break
    if sig_end < HORIZON:
        ax.axvline(sig_end, color="#aaaaaa", linewidth=0.8, linestyle=":",
                   label=f"CI crosses 0 at h={sig_end}")

    ax.set_title(f"Repo shock → {VAR_LABELS.get(v, v)}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Months after shock", fontsize=9)
    ax.set_ylabel("Response (pp)", fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

fig.suptitle(
    f"IRF: Response of G-Sec Yields to a 1-unit RBI Repo Rate Shock\n"
    f"SVAR({p_sel}), Cholesky ID, {int(CI_LEVEL*100)}% Bootstrap CI "
    f"({N_BOOT} replications)",
    fontsize=12, fontweight="bold", y=1.03
)
plt.tight_layout()
plt.savefig("plots/svar_irf_repo_shock.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/svar_irf_repo_shock.png  ← MAIN FIGURE")


# ── Plot 2: Full K×K IRF grid ─────────────────────────────────────────────────
fig, axes = plt.subplots(K, K, figsize=(3.5 * K, 3.0 * K), sharex=True)

for i, resp in enumerate(avail):
    for j, shock in enumerate(avail):
        ax       = axes[i, j]
        irf_pts  = irfs[:, i, j]
        lo_ij    = ci_lo[:, i, j]
        hi_ij    = ci_hi[:, i, j]
        color    = COLORS.get(shock, "#555555")

        ax.fill_between(horizons, lo_ij, hi_ij, color=color, alpha=0.12)
        ax.plot(horizons, lo_ij, color=color, linewidth=0.5, linestyle="--", alpha=0.4)
        ax.plot(horizons, hi_ij, color=color, linewidth=0.5, linestyle="--", alpha=0.4)
        ax.plot(horizons, irf_pts, color=color, linewidth=1.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks([0, 12, 24])
        ax.tick_params(labelsize=7)

        if i == 0:
            ax.set_title(f"Shock: {VAR_LABELS.get(shock, shock)}",
                         fontsize=8, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"{VAR_LABELS.get(resp, resp)}", fontsize=8)
        ax.grid(axis="y", alpha=0.15, linewidth=0.4)

fig.suptitle(f"Full IRF Matrix — SVAR({p_sel}), {int(CI_LEVEL*100)}% Bootstrap CI",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/svar_irf_grid.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/svar_irf_grid.png")


# ── Plot 3: FEVD stacked bars for each variable ───────────────────────────────
fevd_horizons = [1, 3, 6, 12, 24]
shock_colors  = [COLORS.get(v, "#555555") for v in avail]

fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 5), sharey=True)
if K == 1:
    axes = [axes]

for ax, i, resp in zip(axes, range(K), avail):
    fevd_data = np.array([[fevd[h, i, j] * 100 for j in range(K)]
                           for h in fevd_horizons])
    bottoms   = np.zeros(len(fevd_horizons))
    x_pos     = np.arange(len(fevd_horizons))

    for j, shock in enumerate(avail):
        ax.bar(x_pos, fevd_data[:, j], bottom=bottoms,
               color=shock_colors[j], alpha=0.85,
               label=VAR_LABELS.get(shock, shock), width=0.6)
        bottoms += fevd_data[:, j]

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"h={h}" for h in fevd_horizons], fontsize=8)
    ax.set_title(f"{VAR_LABELS.get(resp, resp)}", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    if i == 0:
        ax.set_ylabel("Variance share (%)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

fig.suptitle("Forecast Error Variance Decomposition by Shock",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/svar_fevd.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/svar_fevd.png")


# ── Plot 4: Structural shocks time series ─────────────────────────────────────
fig, axes = plt.subplots(K, 1, figsize=(13, 2.5 * K), sharex=True)
if K == 1:
    axes = [axes]

for ax, v in zip(axes, avail):
    s     = shocks_df[f"shock_{v}"].values
    color = COLORS.get(v, "#333333")
    ax.bar(shock_dates, np.where(s >= 0, s, 0), width=25,
           color=color, alpha=0.7)
    ax.bar(shock_dates, np.where(s <  0, s, 0), width=25,
           color="#e74c3c", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"ε_{v}\n(structural)", fontsize=8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
fig.suptitle("Structural Monetary Policy Shocks — SVAR Residuals",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/svar_residuals.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/svar_residuals.png")


# ── Plot 5: Granger causality visual summary ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 0.6 * len(gc_df) + 2))
ax.axis("off")

col_labels = ["Cause", "→", "Effect", "F-stat", "p-value", "Decision"]
table_data = []
for _, row in gc_df.iterrows():
    stars = ("***" if row["p-value"] < 0.01 else
             ("**"  if row["p-value"] < 0.05 else
              ("*"   if row["p-value"] < 0.10 else "")))
    table_data.append([
        row["Cause"], "→", row["Effect"],
        f"{row['F-stat']:.4f}",
        f"{row['p-value']:.4f}{stars}",
        row["Conclusion"],
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)

# Colour header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1f5fa6")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Colour significant rows
for i, (_, row) in enumerate(gc_df.iterrows(), start=1):
    color = "#e8f5e9" if row["p-value"] < 0.05 else "#fff3e0"
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(color)

ax.set_title(f"Granger Causality Tests — VAR({p_sel})\n"
             f"H₀: Cause does not Granger-cause Effect\n"
             f"*** p<0.01  ** p<0.05  * p<0.10",
             fontsize=10, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("plots/svar_granger.png", dpi=180, bbox_inches="tight")
plt.close()
print("  plots/svar_granger.png")


# ── 9. PAPER SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SECTION 3 COMPLETE — RESULTS FOR PAPER")
print("=" * 62)

print(f"""
  Model      : SVAR({p_sel}), Cholesky recursive identification
  Variables  : {avail}
  Ordering   : Repo first (policy exogenous within month)
  Sample     : {shock_dates[0].date()} – {shock_dates[-1].date()}
  Obs used   : {T_eff} (after {p_sel} lags)
  Bootstrap  : {N_BOOT} replications, {int(CI_LEVEL*100)}% confidence bands

  Key results for null hypothesis testing:
    H0 Part 1 (Granger): Repo does not affect yields
      → See data/svar_granger.csv and plots/svar_granger.png

    H0 Part 2 (IRF): Yield responses not significantly different from 0
      → See plots/svar_irf_repo_shock.png  ← PRIMARY FIGURE
      → CI bands that exclude 0 = reject H0

    H0 Part 3 (FEVD): Long-end yields not driven by expectations/term premia
      → See data/svar_fevd.csv and plots/svar_fevd.png
      → Repo share of Y10 variance decomposition is the key number

  All results saved to data/ and plots/.
  Next: python3 term_premium.py  (Section 4 — Expectations vs Term Premium)
""")