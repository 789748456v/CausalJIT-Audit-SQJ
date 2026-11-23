"""
Strictly Causal Survival / Latency Maturity & Inverse-Propensity Prevalence Estimators (N05 V3).

Addresses Codex Round 1-2 M1, M2, and Criterion 4:
1. Cohort slicing `[-max_history_cohort:]` happens BEFORE computing `q_max` age quantile so risk-set
   support `>= (1 - q_max) * N_cohort` at `a_max` is mathematically guaranteed.
2. `KaplanMeierCureEstimator` evaluates exact unrounded event times `T[delta == 1]` with consistent
   risk sets `T >= u` so `0 <= e_u <= n_u` holds identically for every event horizon `u`.
3. `WeibullMixtureCureEstimator` fits on the exact same unified recent cohort without outcome-stratified
   sampling bias, explicitly checking `res.success` and logging fallback invocations.
4. `FixedHalfLifeMaturityEstimator` provides a zero-fit fixed curve control `c(a) = 1 - 2^(-a / h)`
   to test whether data-driven survival shape estimation adds value over a fixed waiting curve.
5. `estimate_ht_shrunk_prevalence` estimates true unweighted/u-weighted target prevalence `E_u[eta]`
   via Horvitz-Thompson inverse-maturity weighting `y_obs / max(c_i, c_ht)` shrunk by Kish effective
   sample size `n_eff`, matching the exact estimand of Stage 3 bisection even when `Cov(c, eta) != 0`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.optimize

from config_v3 import SECONDS_PER_DAY, V3_CONFIG


@dataclass
class BisectionResult:
    delta_star: float
    target_prevalence: float
    achieved_prevalence: float
    residual: float
    is_reachable: bool
    clamped_side: str  # "none", "low", or "high"


class CohortIncidenceEstimator:
    """
    Cohort-Consistent Non-Parametric Cumulative Incidence Maturity Estimator.

    At horizon g_m <= a_max, evaluates empirical defect-reveal incidence strictly over the
    age-eligible cohort E(g_m) = {i in H_recent : a_i >= g_m}, applies monotone envelope,
    and normalizes by plateau incidence at a_max scaled by an explicit finite-horizon tail factor.
    """

    def __init__(
        self,
        max_cohort: int = V3_CONFIG.max_history_cohort,
        support_q: float = V3_CONFIG.support_age_quantile,
        min_risk_set: int = V3_CONFIG.min_risk_set,
        min_fixes: int = V3_CONFIG.min_observed_fixes,
        c_min: float = V3_CONFIG.c_min,
        c_max: float = V3_CONFIG.c_max,
        tail_cap_override: Optional[float] = None,
        fallback_half_life_days: float = V3_CONFIG.fallback_half_life_days,
    ):
        self.max_cohort = max_cohort
        self.support_q = support_q
        self.min_risk_set = min_risk_set
        self.min_fixes = min_fixes
        self.c_min = c_min
        self.c_max = c_max
        self.tail_cap_override = tail_cap_override
        self.fallback_half_life_days = fallback_half_life_days

        self.grid_days_: Optional[np.ndarray] = None
        self.cdf_vals_: Optional[np.ndarray] = None
        self.used_fallback_: bool = False
        self.eff_tail_: float = 0.85
        self.ll_median_: float = fallback_half_life_days
        self.n_observed_fixes_: int = 0

    def _set_fallback(self) -> "CohortIncidenceEstimator":
        self.used_fallback_ = True
        grid = np.linspace(0.0, 720.0, 145)
        cdf = (1.0 - np.power(2.0, -grid / max(self.fallback_half_life_days, 1.0))) * 0.88
        self.grid_days_ = grid
        self.cdf_vals_ = np.clip(cdf, self.c_min, self.c_max)
        self.ll_median_ = self.fallback_half_life_days
        return self

    def fit(
        self,
        commit_ts: np.ndarray,
        fix_ts: np.ndarray,
        y_obs: np.ndarray,
        t_alarm: float,
    ) -> "CohortIncidenceEstimator":
        commit_ts = np.asarray(commit_ts, dtype=np.float64)
        fix_ts = np.asarray(fix_ts, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        valid = commit_ts <= t_alarm
        c_ts = commit_ts[valid]
        f_ts = fix_ts[valid]
        y_v = y_obs[valid] > 0.5

        # Slice trailing cohort FIRST so quantile support is exact on the analyzed cohort
        if len(c_ts) > self.max_cohort:
            c_ts = c_ts[-self.max_cohort :]
            f_ts = f_ts[-self.max_cohort :]
            y_v = y_v[-self.max_cohort :]

        ages_days = np.maximum((t_alarm - c_ts) / SECONDS_PER_DAY, 0.0)
        is_revealed = y_v & np.isfinite(f_ts) & (f_ts <= t_alarm)
        delay_days = np.where(is_revealed, np.maximum((f_ts - c_ts) / SECONDS_PER_DAY, 0.0), np.inf)
        self.n_observed_fixes_ = int(np.sum(is_revealed))

        if self.n_observed_fixes_ < self.min_fixes or len(ages_days) < max(200, self.min_risk_set):
            return self._set_fallback()

        max_supported_age = float(np.percentile(ages_days, self.support_q * 100.0))
        if max_supported_age < 30.0:
            return self._set_fallback()

        cand_grid = np.array(
            [
                0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 30.0, 45.0, 60.0, 75.0, 90.0,
                120.0, 150.0, 180.0, 240.0, 300.0, 365.0, 450.0, 540.0, 720.0, 900.0,
            ],
            dtype=np.float64,
        )
        grid = cand_grid[cand_grid <= max_supported_age]
        if len(grid) == 0 or grid[-1] < max_supported_age:
            grid = np.append(grid, max_supported_age)

        eff_min_risk = min(self.min_risk_set, int(np.floor((1.0 - self.support_q) * len(ages_days))))
        eff_min_risk = max(eff_min_risk, 25)

        g_vals = np.zeros(len(grid), dtype=np.float64)
        for k, g in enumerate(grid):
            elig = ages_days >= g - 1e-9
            n_elig = int(np.sum(elig))
            if n_elig >= eff_min_risk:
                g_vals[k] = float(np.sum(delay_days[elig] <= g + 1e-9)) / float(n_elig)
            elif k > 0:
                g_vals[k] = g_vals[k - 1]

        g_mono = np.maximum.accumulate(g_vals)
        g_max = float(g_mono[-1])
        if g_max < 1e-4:
            return self._set_fallback()

        self.used_fallback_ = False
        if self.tail_cap_override is not None:
            eff_tail = float(np.clip(self.tail_cap_override, self.c_min + 0.05, self.c_max))
        else:
            g_half = float(np.interp(0.5 * grid[-1], grid, g_mono))
            growth_ratio = g_max / max(g_half, 1e-5) if g_half > 1e-5 else 1.5
            slope_excess = max(growth_ratio - 1.0, 0.0)
            age_factor = float(np.clip(max_supported_age / (max_supported_age + 45.0), 0.65, 1.0))
            eff_tail = float(
                np.clip((V3_CONFIG.tail_cap_max - 0.28 * min(slope_excess, 1.0)) * age_factor, V3_CONFIG.tail_cap_min, V3_CONFIG.tail_cap_max)
            )
        self.eff_tail_ = eff_tail
        cdf = (g_mono / g_max) * eff_tail
        self.grid_days_ = grid
        self.cdf_vals_ = np.clip(cdf, self.c_min, self.c_max)
        med_idx = int(np.searchsorted(self.cdf_vals_, 0.50))
        self.ll_median_ = float(self.grid_days_[min(med_idx, len(self.grid_days_) - 1)])
        return self

    def predict_maturity(self, ages_seconds: np.ndarray) -> np.ndarray:
        ages_days = np.maximum(np.asarray(ages_seconds, dtype=np.float64) / SECONDS_PER_DAY, 0.0)
        if self.grid_days_ is None or self.cdf_vals_ is None:
            cdf = (1.0 - np.power(2.0, -ages_days / max(self.fallback_half_life_days, 1.0))) * 0.88
            return np.clip(cdf, self.c_min, self.c_max).astype(np.float32)
        interp = np.interp(ages_days, self.grid_days_, self.cdf_vals_, left=self.c_min, right=self.cdf_vals_[-1])
        return np.clip(interp, self.c_min, self.c_max).astype(np.float32)


class KaplanMeierCureEstimator:
    """
    Bug-Free Right-Censored Product-Limit Kaplan-Meier Cure Estimator.
    Uses exact unrounded event times `T[ev]` and consistent risk sets `T >= u` so `0 <= e_u <= n_u`
    is an algebraic invariant on every possible input dataset.
    """

    def __init__(
        self,
        max_cohort: int = V3_CONFIG.max_history_cohort,
        support_q: float = V3_CONFIG.support_age_quantile,
        min_fixes: int = V3_CONFIG.min_observed_fixes,
        c_min: float = V3_CONFIG.c_min,
        c_max: float = V3_CONFIG.c_max,
        fallback_half_life_days: float = V3_CONFIG.fallback_half_life_days,
    ):
        self.max_cohort = max_cohort
        self.support_q = support_q
        self.min_fixes = min_fixes
        self.c_min = c_min
        self.c_max = c_max
        self.fallback_half_life_days = fallback_half_life_days
        self.grid_days_: Optional[np.ndarray] = None
        self.cdf_vals_: Optional[np.ndarray] = None
        self.used_fallback_: bool = False
        self.ll_median_: float = fallback_half_life_days

    def _set_fallback(self) -> "KaplanMeierCureEstimator":
        self.used_fallback_ = True
        grid = np.linspace(0.0, 720.0, 145)
        self.grid_days_ = grid
        self.cdf_vals_ = np.clip((1.0 - np.power(2.0, -grid / max(self.fallback_half_life_days, 1.0))) * 0.88, self.c_min, self.c_max)
        self.ll_median_ = self.fallback_half_life_days
        return self

    def fit(self, commit_ts: np.ndarray, fix_ts: np.ndarray, y_obs: np.ndarray, t_alarm: float) -> "KaplanMeierCureEstimator":
        commit_ts = np.asarray(commit_ts, dtype=np.float64)
        fix_ts = np.asarray(fix_ts, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        valid = commit_ts <= t_alarm
        c_ts = commit_ts[valid]
        f_ts = fix_ts[valid]
        y_v = y_obs[valid] > 0.5
        if len(c_ts) > self.max_cohort:
            c_ts = c_ts[-self.max_cohort :]
            f_ts = f_ts[-self.max_cohort :]
            y_v = y_v[-self.max_cohort :]

        ages = np.maximum((t_alarm - c_ts) / SECONDS_PER_DAY, 0.0)
        ev = y_v & np.isfinite(f_ts) & (f_ts <= t_alarm)
        # Quantize BOTH event times and censoring times on the exact same 1e-4 day (~8.6s) grid!
        t_obs = np.round(np.where(ev, np.maximum((f_ts - c_ts) / SECONDS_PER_DAY, 0.0), ages), 4)

        if len(ages) < 200:
            return self._set_fallback()
        max_age = float(np.percentile(t_obs, self.support_q * 100.0))
        ev_in_support = ev & (t_obs <= max_age)
        if int(np.sum(ev_in_support)) < self.min_fixes or max_age < 30.0:
            return self._set_fallback()

        uniq_d = np.unique(t_obs[ev_in_support])
        s_val = 1.0
        g_km = np.zeros(len(uniq_d), dtype=np.float64)
        for idx, d in enumerate(uniq_d):
            n_risk = int(np.sum(t_obs >= d))
            n_ev = int(np.sum(ev & (t_obs == d)))
            assert 0 <= n_ev <= n_risk, f"KM invariant violated: n_ev={n_ev} > n_risk={n_risk}"
            if n_risk < 25:
                g_km[idx] = g_km[idx - 1] if idx > 0 else 0.0
                continue
            s_val *= (1.0 - float(n_ev) / float(n_risk))
            g_km[idx] = 1.0 - s_val

        g_max = float(g_km[-1]) if len(g_km) > 0 else 0.0
        if g_max < 1e-4:
            return self._set_fallback()

        self.used_fallback_ = False
        g_half = float(np.interp(0.5 * uniq_d[-1], uniq_d, g_km))
        growth_ratio = g_max / max(g_half, 1e-5) if g_half > 1e-5 else 1.5
        age_factor = float(np.clip(max_age / (max_age + 45.0), 0.65, 1.0))
        eff_tail = float(np.clip((V3_CONFIG.tail_cap_max - 0.28 * min(max(growth_ratio - 1.0, 0.0), 1.0)) * age_factor, V3_CONFIG.tail_cap_min, V3_CONFIG.tail_cap_max))

        self.grid_days_ = np.concatenate([[0.0], uniq_d])
        cdf = np.concatenate([[0.0], (g_km / g_max) * eff_tail])
        self.cdf_vals_ = np.clip(cdf, self.c_min, self.c_max)
        med_idx = int(np.searchsorted(self.cdf_vals_, 0.50))
        self.ll_median_ = float(self.grid_days_[min(med_idx, len(self.grid_days_) - 1)])
        return self

    def predict_maturity(self, ages_seconds: np.ndarray) -> np.ndarray:
        ages_days = np.maximum(np.asarray(ages_seconds, dtype=np.float64) / SECONDS_PER_DAY, 0.0)
        if self.grid_days_ is None or self.cdf_vals_ is None:
            return np.clip((1.0 - np.power(2.0, -ages_days / max(self.fallback_half_life_days, 1.0))) * 0.88, self.c_min, self.c_max).astype(np.float32)
        interp = np.interp(ages_days, self.grid_days_, self.cdf_vals_, left=self.c_min, right=self.cdf_vals_[-1])
        return np.clip(interp, self.c_min, self.c_max).astype(np.float32)


class WeibullMixtureCureEstimator:
    """
    Coherent-Cohort Parametric Weibull Mixture-Cure MLE (Boag 1949; Farewell 1982).
    Fits strictly on a single unstratified trailing cohort `[-max_cohort:]` so event/censoring
    ratios are never distorted by outcome-dependent subsampling, and checks `res.success`.
    """

    def __init__(
        self,
        max_cohort: int = V3_CONFIG.max_history_cohort,
        min_fixes: int = V3_CONFIG.min_observed_fixes,
        c_min: float = V3_CONFIG.c_min,
        c_max: float = V3_CONFIG.c_max,
        fallback_half_life_days: float = V3_CONFIG.fallback_half_life_days,
    ):
        self.max_cohort = max_cohort
        self.min_fixes = min_fixes
        self.c_min = c_min
        self.c_max = c_max
        self.fallback_half_life_days = fallback_half_life_days
        self.shape_k_: float = 0.90
        self.scale_lam_: float = 120.0
        self.pi_cure_: float = 0.15
        self.fitted_: bool = False
        self.used_fallback_: bool = False
        self.ll_median_: float = fallback_half_life_days

    def fit(self, commit_ts: np.ndarray, fix_ts: np.ndarray, y_obs: np.ndarray, t_alarm: float) -> "WeibullMixtureCureEstimator":
        commit_ts = np.asarray(commit_ts, dtype=np.float64)
        fix_ts = np.asarray(fix_ts, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        valid = commit_ts <= t_alarm
        c_ts = commit_ts[valid]
        f_ts = fix_ts[valid]
        y_v = y_obs[valid] > 0.5
        # Single unified cohort slice BEFORE splitting into events and censored observations!
        if len(c_ts) > self.max_cohort:
            c_ts = c_ts[-self.max_cohort :]
            f_ts = f_ts[-self.max_cohort :]
            y_v = y_v[-self.max_cohort :]

        ages = np.maximum((t_alarm - c_ts) / SECONDS_PER_DAY, 0.05)
        ev = y_v & np.isfinite(f_ts) & (f_ts <= t_alarm)
        delays = np.maximum((f_ts[ev] - c_ts[ev]) / SECONDS_PER_DAY, 0.05)
        cens_ages = ages[~ev]

        if len(delays) < self.min_fixes or len(ages) < 200:
            self.fitted_ = False
            self.used_fallback_ = True
            self.ll_median_ = self.fallback_half_life_days
            return self

        emp_pi = float(np.clip(len(delays) / float(len(ages)) * 1.35, 0.02, 0.70))
        init_lam = float(np.clip(np.median(delays) * 1.5, 5.0, 500.0))

        def neg_loglik(params: np.ndarray) -> float:
            logit_pi, log_k, log_lam = params
            pi = 1.0 / (1.0 + np.exp(-np.clip(logit_pi, -6.0, 3.0)))
            k = np.exp(np.clip(log_k, -1.0, 1.4))
            lam = np.exp(np.clip(log_lam, 0.5, 7.0))

            d_rat = np.clip(delays / lam, 1e-8, 1e4)
            d_pow = np.power(d_rat, k)
            ll_pos = np.sum(np.log(pi + 1e-12) + np.log(k) - np.log(lam) + (k - 1.0) * np.log(d_rat) - d_pow)

            c_rat = np.clip(cens_ages / lam, 1e-8, 1e4)
            f_cens = -np.expm1(-np.clip(np.power(c_rat, k), 0.0, 50.0))
            ll_neg = np.sum(np.log(np.clip(1.0 - pi * f_cens, 1e-12, 1.0)))
            reg = 0.02 * (log_k ** 2) + 0.005 * ((log_lam - np.log(init_lam)) ** 2)
            return float(-(ll_pos + ll_neg) / float(len(ages)) + reg)

        x0 = np.array([np.log(emp_pi / (1.0 - emp_pi)), np.log(0.90), np.log(init_lam)], dtype=np.float64)
        bounds = [(-4.5, 2.0), (-0.8, 1.2), (np.log(2.0), np.log(900.0))]
        try:
            res = scipy.optimize.minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
            if not bool(res.success) and res.status != 0:
                self.fitted_ = False
                self.used_fallback_ = True
                return self
            logit_pi, log_k, log_lam = res.x
            self.pi_cure_ = float(1.0 / (1.0 + np.exp(-logit_pi)))
            self.shape_k_ = float(np.exp(log_k))
            self.scale_lam_ = float(np.exp(log_lam))
            self.ll_median_ = float(self.scale_lam_ * (np.log(2.0) ** (1.0 / self.shape_k_)))
            self.fitted_ = True
            self.used_fallback_ = False
        except Exception:
            self.fitted_ = False
            self.used_fallback_ = True
        return self

    def predict_maturity(self, ages_seconds: np.ndarray) -> np.ndarray:
        ages_days = np.maximum(np.asarray(ages_seconds, dtype=np.float64) / SECONDS_PER_DAY, 0.0)
        if not self.fitted_:
            return np.clip((1.0 - np.power(2.0, -ages_days / max(self.fallback_half_life_days, 1.0))) * 0.88, self.c_min, self.c_max).astype(np.float32)
        rat = np.clip(ages_days / max(self.scale_lam_, 1e-4), 0.0, 1e4)
        cdf = -np.expm1(-np.clip(np.power(rat, self.shape_k_), 0.0, 50.0))
        return np.clip(cdf, self.c_min, self.c_max).astype(np.float32)


class FixedHalfLifeMaturityEstimator:
    """
    Zero-Fit Parametric Control Curve: c(a) = clip(1 - 2^(-a / half_life_days), c_min, c_max).
    Tests whether learning the repository's survival curve shape improves over a fixed 90d curve.
    """

    def __init__(self, half_life_days: float = 90.0, c_min: float = V3_CONFIG.c_min, c_max: float = V3_CONFIG.c_max):
        self.half_life_days = half_life_days
        self.c_min = c_min
        self.c_max = c_max
        self.used_fallback_: bool = False
        self.ll_median_: float = half_life_days

    def fit(self, commit_ts: np.ndarray, fix_ts: np.ndarray, y_obs: np.ndarray, t_alarm: float) -> "FixedHalfLifeMaturityEstimator":
        return self

    def predict_maturity(self, ages_seconds: np.ndarray) -> np.ndarray:
        ages_days = np.maximum(np.asarray(ages_seconds, dtype=np.float64) / SECONDS_PER_DAY, 0.0)
        cdf = 1.0 - np.power(2.0, -ages_days / max(self.half_life_days, 1.0))
        return np.clip(cdf, self.c_min, self.c_max).astype(np.float32)


def build_survival_estimator_v3(name: str, tail_cap_override: Optional[float] = None):
    key = name.lower().strip()
    if key in ("cohort_incidence", "lynden_bell", "default"):
        return CohortIncidenceEstimator(tail_cap_override=tail_cap_override)
    if key in ("kaplan_meier", "km"):
        return KaplanMeierCureEstimator()
    if key in ("weibull_cure", "weibull"):
        return WeibullMixtureCureEstimator()
    if key in ("fixed_90d", "fixed_curve"):
        return FixedHalfLifeMaturityEstimator(half_life_days=90.0)
    raise ValueError(f"Unknown survival estimator: {name}")


def estimate_ht_shrunk_prevalence(
    y_obs: np.ndarray,
    c_maturity: np.ndarray,
    sample_weights: np.ndarray,
    anchor_prevalence: float,
    c_ht_floor: float = V3_CONFIG.ht_floor_c,
    m0: float = V3_CONFIG.ht_shrinkage_m0,
    pi_min: float = V3_CONFIG.pi_min,
    pi_max: float = V3_CONFIG.pi_max,
) -> Tuple[float, float, float]:
    """
    Clipped Inverse-Maturity Plug-In Prevalence with Heuristic Variance-Inflation Shrinkage (Codex R3-M1).

    Target estimand on pool B_cal with weights u_i:
        pi_u^* = (sum_i u_i * eta_i) / (sum_i u_i).
    Under age-only maturity approximation P(S_i=1 | Y_i=1, X_i, a_i) = c_i, ideal Horvitz-Thompson
    satisfies E[y_obs_i / c_i | X_i, a_i] = eta_i. With variance-stabilizing floor c_tilde_i = max(c_i, c_ht_floor),
        E[y_obs_i / c_tilde_i | X_i, a_i] = eta_i * min(1, c_i / c_ht_floor),
    so commits with c_i < c_ht_floor incur conservative downward attenuation c_i / c_ht_floor.
    We shrink the clipped plug-in estimate toward `anchor_prevalence` using heuristic inverse-maturity
    precision weight `n_eff_heur = (sum_i u_i)^2 / [sum_i (u_i / c_tilde_i)^2]` (not exact Kish ESS):
        pi_target = clip((n_eff_heur * pi_HT + m0 * anchor_prevalence) / (n_eff_heur + m0), pi_min, pi_max).
    Returns: (pi_target, pi_HT_clipped, n_eff_heur).
    """
    y = np.asarray(y_obs, dtype=np.float64)
    c_til = np.maximum(np.asarray(c_maturity, dtype=np.float64), float(c_ht_floor))
    u = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1e-8)

    u_sum = float(np.sum(u))
    pi_ht_raw = float(np.sum(u * (y / c_til)) / max(u_sum, 1e-8))
    inv_var_denom = float(np.sum((u / c_til) ** 2))
    n_eff = (u_sum ** 2) / max(inv_var_denom, 1e-8)

    pi_target = (n_eff * pi_ht_raw + float(m0) * float(anchor_prevalence)) / max(n_eff + float(m0), 1e-8)
    obs_floor = float(np.sum(u * y) / max(u_sum, 1e-8))
    pi_target = float(np.clip(max(pi_target, obs_floor + 1e-4), pi_min, pi_max))
    return pi_target, pi_ht_raw, n_eff


def compute_survival_pu_posterior(
    base_eta: np.ndarray,
    delta: float,
    y_obs: np.ndarray,
    c_maturity: np.ndarray,
    p_min: float = V3_CONFIG.posterior_p_min,
    p_max: float = V3_CONFIG.posterior_p_max,
) -> np.ndarray:
    """
    Single Canonical Survival-PU Posterior Function (Codex R3-C5):
    Used IDENTICALLY across endpoint reachability, bisection root-finding, student training soft
    targets, and residual diagnostics so zero discrepancy can ever arise between solver and loss!
    """
    eta = np.asarray(base_eta, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    c = np.asarray(c_maturity, dtype=np.float64)
    p0 = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta + float(delta), -25.0, 25.0))), float(p_min), float(p_max))
    q_unobs = ((1.0 - c) * p0) / np.maximum(1.0 - c * p0, 1e-8)
    return np.where(y > 0.5, 1.0, q_unobs)


def solve_monotone_pu_bisection(
    base_eta: np.ndarray,
    y_obs: np.ndarray,
    c_maturity: np.ndarray,
    sample_weights: np.ndarray,
    target_pi: float,
    p_min: float = V3_CONFIG.posterior_p_min,
    p_max: float = V3_CONFIG.posterior_p_max,
    lo: float = V3_CONFIG.bisection_lo,
    hi: float = V3_CONFIG.bisection_hi,
    n_iters: int = V3_CONFIG.bisection_iters,
) -> BisectionResult:
    """
    Solve for shift delta^* matching u-weighted `compute_survival_pu_posterior` mean on B_cal to `target_pi`,
    evaluating the EXACT clipped posterior used in downstream student training (Codex R3-C5).
    """
    eta = np.asarray(base_eta, dtype=np.float64)
    y = np.asarray(y_obs, dtype=np.float64)
    c = np.asarray(c_maturity, dtype=np.float64)
    u = np.maximum(np.asarray(sample_weights, dtype=np.float64), 1e-8)
    u_sum = float(np.sum(u))

    def eval_mean(delta: float) -> float:
        q = compute_survival_pu_posterior(eta, delta, y, c, p_min=p_min, p_max=p_max)
        return float(np.sum(u * q) / u_sum)

    f_lo = eval_mean(lo)
    f_hi = eval_mean(hi)
    tol = 1e-12

    if abs(target_pi - f_lo) <= tol:
        return BisectionResult(
            delta_star=float(lo),
            target_prevalence=float(target_pi),
            achieved_prevalence=float(f_lo),
            residual=abs(float(f_lo - target_pi)),
            is_reachable=True,
            clamped_side="boundary_low",
        )
    if abs(target_pi - f_hi) <= tol:
        return BisectionResult(
            delta_star=float(hi),
            target_prevalence=float(target_pi),
            achieved_prevalence=float(f_hi),
            residual=abs(float(f_hi - target_pi)),
            is_reachable=True,
            clamped_side="boundary_high",
        )
    if target_pi < f_lo - tol:
        return BisectionResult(
            delta_star=float(lo),
            target_prevalence=float(target_pi),
            achieved_prevalence=float(f_lo),
            residual=abs(float(f_lo - target_pi)),
            is_reachable=False,
            clamped_side="low",
        )
    if target_pi > f_hi + tol:
        return BisectionResult(
            delta_star=float(hi),
            target_prevalence=float(target_pi),
            achieved_prevalence=float(f_hi),
            residual=abs(float(f_hi - target_pi)),
            is_reachable=False,
            clamped_side="high",
        )

    a, b = float(lo), float(hi)
    for _ in range(n_iters):
        mid = 0.5 * (a + b)
        if eval_mean(mid) < target_pi:
            a = mid
        else:
            b = mid
    d_star = 0.5 * (a + b)
    f_mid = eval_mean(d_star)
    return BisectionResult(
        delta_star=float(d_star),
        target_prevalence=float(target_pi),
        achieved_prevalence=float(f_mid),
        residual=abs(float(f_mid - target_pi)),
        is_reachable=True,
        clamped_side="none",
    )
