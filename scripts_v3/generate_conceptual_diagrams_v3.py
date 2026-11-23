"""
Generate Publication-Grade Conceptual Diagrams for N05 V3 (Software Quality Journal):
  1. `figures/fig0a_causal_timeline_v3.pdf` & `.png`: Causal Supervision Timeline & Poisoning-Staleness Dilemma
  2. `figures/fig0b_survival_pu_architecture_v3.pdf` & `.png`: Clean 3-Stage V3 Survival-PU Architecture

Zero internal note codes (`N46`, `N60`, `N33`), crisp typography, generous margins, zero text overlap.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _box(
    ax, x, y, w, h, title, body,
    fc="#F8FAFC", ec="#334155", lw=1.3,
    title_color="#0F172A", body_color="#1E293B",
    title_fs=9.4, body_fs=8.1, pad=0.010, linespacing=1.36
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.016",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + w / 2.0, y + h - 0.20 * h, title,
            ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color, zorder=3
        )
        ax.text(
            x + w / 2.0, y + 0.39 * h, body,
            ha="center", va="center", fontsize=body_fs, color=body_color, linespacing=linespacing, zorder=3
        )
    else:
        ax.text(
            x + w / 2.0, y + h / 2.0, title,
            ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color, zorder=3
        )


def _arrow(ax, x0, y0, x1, y1, color="#334155", lw=1.5, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=4
    )
    ax.add_patch(arr)


def build_fig0a_causal_timeline(fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top title banner
    ax.text(
        0.50, 0.968,
        "Causal Supervision Audit at Drift Alarm $t_{\\mathrm{alarm}}$: Offline Hindsight vs. Live CI Deployment",
        ha="center", va="center", fontsize=12.0, fontweight="bold", color="#0F172A"
    )

    # Panel A: Offline Hindsight Oracle (B1b_4W_Oracle)
    p1 = FancyBboxPatch(
        (0.02, 0.57), 0.96, 0.36,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.3, zorder=1
    )
    ax.add_patch(p1)
    ax.text(
        0.035, 0.902,
        "(a) Offline Retrospective Benchmark ($B_{1b}^*$ Window Oracle — Non-Deployable Hindsight Reference)",
        ha="left", va="center", fontsize=10.3, fontweight="bold", color="#92400E"
    )

    # Timeline axis in Panel A
    y_tl_a = 0.708
    ax.plot([0.055, 0.915], [y_tl_a, y_tl_a], color="#475569", lw=1.8, zorder=2)
    _arrow(ax, 0.90, y_tl_a, 0.935, y_tl_a, color="#475569", lw=1.8)
    ax.text(0.942, y_tl_a, "Time $t$", ha="left", va="center", fontsize=9.2, color="#334155")

    # Shaded regions on Panel A timeline
    ax.add_patch(Rectangle((0.065, y_tl_a - 0.036), 0.295, 0.072, facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.1, zorder=2))
    ax.text(0.2125, y_tl_a, "Aged History ($a_i \\geq 90\\mathrm{d}$)",
            ha="center", va="center", fontsize=9.1, color="#1E3A8A", fontweight="bold")

    ax.add_patch(Rectangle((0.375, y_tl_a - 0.036), 0.235, 0.072, facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.2, zorder=2))
    ax.text(0.4925, y_tl_a, "Drifted Window $W_{\\mathrm{drift}}$",
            ha="center", va="center", fontsize=9.2, color="#991B1B", fontweight="bold")

    # Alarm vertical marker
    ax.plot([0.61, 0.61], [0.642, 0.835], color="#DC2626", lw=1.9, linestyle="--", zorder=3)
    ax.text(
        0.61, 0.854, "Alarm $t_{\\mathrm{alarm}}$",
        ha="center", va="center", fontsize=9.2, fontweight="bold", color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFFBEB", edgecolor="#FCA5A5", lw=0.9)
    )

    # Future bug-fix events crossing backward
    for cx, fx, lbl, rad_v in [
        (0.435, 0.725, "$t_1^{\\mathrm{fix}}$ (+110d)", 0.24),
        (0.545, 0.845, "$t_2^{\\mathrm{fix}}$ (+175d)", 0.26),
    ]:
        ax.scatter([cx], [y_tl_a + 0.036], s=46, color="#DC2626", zorder=5)
        ax.scatter([fx], [y_tl_a], s=50, color="#B45309", marker="D", zorder=5)
        ax.text(fx, y_tl_a - 0.050, lbl, ha="center", va="center", fontsize=8.8, color="#92400E")
        _arrow(ax, fx, y_tl_a + 0.016, cx + 0.005, y_tl_a + 0.044, color="#D97706", lw=1.5, rad=rad_v)

    ax.text(
        0.822, 0.858,
        "Future fixes ($t_{\\mathrm{fix}} > t_{\\mathrm{alarm}}$)\nretroactively leak $y_i^* = 1$ into $W_{\\mathrm{drift}}$",
        ha="center", va="center", fontsize=8.8, color="#92400E",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FEF3C7", edgecolor="#F59E0B", lw=0.9)
    )
    ax.text(
        0.055, 0.602,
        "Takeaway: Chronological commit splitting alone does NOT prevent future-fix leakage whenever $t_i \\leq t_{\\mathrm{alarm}} < t_i^{\\mathrm{fix}}$.",
        ha="left", va="center", fontsize=9.0, style="italic", color="#78350F"
    )

    # Panel B: Live Causal CI Deployment (t <= t_alarm)
    p2 = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.515,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.3, zorder=1
    )
    ax.add_patch(p2)
    ax.text(
        0.035, 0.504,
        "(b) Live Causal CI Deployment ($t \\leq t_{\\mathrm{alarm}}$ Strictly Enforced — Causal Wall Blocks All $t > t_{\\mathrm{alarm}}$)",
        ha="left", va="center", fontsize=10.3, fontweight="bold", color="#0F172A"
    )

    # Timeline in Panel B with strict causal wall
    y_tl_b = 0.402
    ax.plot([0.055, 0.61], [y_tl_b, y_tl_b], color="#0F172A", lw=1.8, zorder=2)
    ax.add_patch(Rectangle((0.61, y_tl_b - 0.048), 0.34, 0.096, facecolor="#E2E8F0", edgecolor="#64748B", lw=1.0, hatch="///", zorder=2))
    ax.plot([0.61, 0.61], [y_tl_b - 0.060, y_tl_b + 0.060], color="#0F172A", lw=2.5, zorder=4)
    ax.text(
        0.78, y_tl_b,
        "Unobservable Future ($t > t_{\\mathrm{alarm}}$)\n68.8%–94.6% of 1W defects\nstill pending in multi-year cohorts",
        ha="center", va="center", fontsize=8.4, color="#1E293B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#F8FAFC", edgecolor="none", alpha=0.92)
    )

    ax.add_patch(Rectangle((0.065, y_tl_b - 0.034), 0.285, 0.068, facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.1, zorder=3))
    ax.text(0.2075, y_tl_b, "Matured History ($c(a_i) \\geq 0.45$)", ha="center", va="center", fontsize=9.0, color="#1E3A8A", fontweight="bold")

    ax.add_patch(Rectangle((0.365, y_tl_b - 0.034), 0.235, 0.068, facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.1, zorder=3))
    ax.text(0.4825, y_tl_b, "Censored $W_{\\mathrm{drift}}$ ($y_{\\mathrm{obs}}=0$)", ha="center", va="center", fontsize=9.0, color="#991B1B", fontweight="bold")

    # Three strategic choices below timeline in Panel B
    _box(
        ax, 0.04, 0.048, 0.285, 0.255,
        "1. Delayed Window (Wait-90)",
        "Drops unfixed commits with $a_i < 90\\mathrm{d}$.\nReduces $<90\\mathrm{d}$ false negatives,\nbut discards recent $W_{\\mathrm{drift}}$\n(Staleness Under Shift).",
        fc="#EFF6FF", ec="#2563EB", title_color="#1D4ED8", title_fs=9.8, body_fs=8.9, linespacing=1.35
    )
    _box(
        ax, 0.352, 0.048, 0.285, 0.255,
        "2. Naive Real-Time ($B_2$-1W/4W)",
        "Trains directly on $y_{\\mathrm{obs}}(t_{\\mathrm{alarm}})$.\nUnfixed defects act as $y=0$,\ncompressing posteriors downward\n(Right-Censoring Bias).",
        fc="#FEF2F2", ec="#DC2626", title_color="#B91C1C", title_fs=9.8, body_fs=8.9, linespacing=1.35
    )
    _box(
        ax, 0.665, 0.048, 0.295, 0.255,
        "3. Maturity-Weighted (Survival-PU)",
        "Estimates maturity $c(a_i)$ causally;\nbuilds prior-anchored soft targets $q_i$\nto adapt on $W_{\\mathrm{drift}}$ without\ndropping recent commits.",
        fc="#ECFDF5", ec="#059669", title_color="#047857", title_fs=9.8, body_fs=8.9, linespacing=1.35
    )

    _arrow(ax, 0.2075, y_tl_b - 0.038, 0.182, 0.316, color="#2563EB", lw=1.5)
    _arrow(ax, 0.4825, y_tl_b - 0.038, 0.495, 0.316, color="#DC2626", lw=1.5)
    _arrow(ax, 0.54, y_tl_b - 0.038, 0.81, 0.316, color="#059669", lw=1.5)

    os.makedirs(fig_dir, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(fig_dir, "fig0a_causal_timeline_v3.pdf"))
    fig.savefig(os.path.join(fig_dir, "fig0a_causal_timeline_v3.png"), dpi=300)
    plt.close(fig)


def build_fig0b_architecture(fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.1))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.50, 0.958,
        "Survival-PU: Three-Stage Input–Output Pipeline Triggered at Drift Alarm $t_{\\mathrm{alarm}}$ (Formulas in Section 4)",
        ha="center", va="center", fontsize=11.6, fontweight="bold", color="#0F172A"
    )

    # Column 1: Stream Input & Block DriftLens
    _box(
        ax, 0.02, 0.55, 0.20, 0.33,
        "Commit Stream & Host",
        "Inputs: commit metrics $x_{\\mathrm{tab}}$\n(+ diff/msg on Bench. A).\nOutputs: logits $z(x)$ and\n24-D PCA embeddings.",
        fc="#F1F5F9", ec="#475569", title_color="#0F172A", title_fs=10.0, body_fs=8.9, linespacing=1.38
    )
    _box(
        ax, 0.02, 0.13, 0.20, 0.34,
        "Block DriftLens Monitor",
        "Input: window $W_{\\mathrm{drift}}$ emb.\nCircular block bootstrap\n($p_{95}$, cooldown 1) + rolling\n3W reference refresh.\nOutput: alarm at $t_{\\mathrm{alarm}}$.",
        fc="#FEF3C7", ec="#D97706", title_color="#92400E", title_fs=10.0, body_fs=8.8, linespacing=1.34
    )
    _arrow(ax, 0.12, 0.538, 0.12, 0.482, color="#475569", lw=1.6)

    # Column 2: Stage 1
    _box(
        ax, 0.26, 0.55, 0.215, 0.33,
        "Stage 1: Maturity $c(a_i)$",
        "Input: causal history\n($t_i \\leq t_{\\mathrm{alarm}}$, revealed fixes).\nCohort cumulative incidence\n(Eqs. 4–5).\nOutput: commit maturity $c_i$.",
        fc="#EFF6FF", ec="#2563EB", title_color="#1D4ED8", title_fs=10.0, body_fs=8.8, linespacing=1.34
    )
    _box(
        ax, 0.26, 0.13, 0.215, 0.34,
        "Stage 1 Control Curves",
        "Swappable controls (RQ3/4a):\n• Kaplan–Meier Cure\n• Weibull Mixture-Cure\n• Zero-fit FixedCurve-PU\n  (90-day half-life prior)",
        fc="#F8FAFC", ec="#64748B", title_color="#334155", title_fs=10.0, body_fs=8.9, linespacing=1.36
    )
    _arrow(ax, 0.3675, 0.482, 0.3675, 0.538, color="#64748B", lw=1.4)

    # Column 3: Stage 2
    _box(
        ax, 0.515, 0.55, 0.22, 0.33,
        "Stage 2: Reliable Teacher",
        "Input: revealed positives $\\cup$\nmature commits ($c_i \\geq 0.45$).\nCovariate + recency weights.\nOutput: de-weighted base\nlog-odds $\\eta_i^{(0)}$ on pool $\\mathcal{B}$.",
        fc="#F5F3FF", ec="#7C3AED", title_color="#5B21B6", title_fs=10.0, body_fs=8.8, linespacing=1.34
    )
    _box(
        ax, 0.515, 0.13, 0.22, 0.34,
        "Stage 2: Prior $\\hat{\\pi}_{\\mathrm{target}}$",
        "Input: calibration pool $\\mathcal{B}_{\\mathrm{cal}}$,\n$y_{i,\\mathrm{obs}}$, and $c_i$.\nClipped IP prevalence shrunk\ntoward mature anchor (Eq. 6).\nOutput: target prior $\\hat{\\pi}_{\\mathrm{target}}$.",
        fc="#FAF5FF", ec="#7C3AED", title_color="#5B21B6", title_fs=10.0, body_fs=8.8, linespacing=1.34
    )

    # Column 4: Stage 3 & Scoring
    _box(
        ax, 0.775, 0.48, 0.205, 0.40,
        "Stage 3: PU Distillation",
        "Inputs: $\\eta_i^{(0)}$, $c_i$, $\\hat{\\pi}_{\\mathrm{target}}$.\nMonotone bisection $\\delta^*$ sets\nsoft targets $q_i(\\delta^*)$ (Eq. 7).\nFits student host on anchor\n+ $W_{\\mathrm{drift}}$ (weighted BCE).",
        fc="#ECFDF5", ec="#059669", title_color="#047857", title_fs=10.0, body_fs=8.8, linespacing=1.35
    )
    _box(
        ax, 0.775, 0.13, 0.205, 0.27,
        "Causal Stream Scoring",
        "Scores next window $[e, e+W)$.\nCausal Platt $\\hat{p}_{\\mathrm{cal}}$ & threshold\n$\\tau_{\\mathrm{th}}$ fit strictly on past\naged commits ($a_i \\geq 90\\mathrm{d}$).",
        fc="#F0FDF4", ec="#16A34A", title_color="#15803D", title_fs=10.0, body_fs=8.8, linespacing=1.34
    )

    # Clean non-overlapping arrows between columns
    _arrow(ax, 0.232, 0.715, 0.248, 0.715, color="#334155", lw=1.6)
    _arrow(ax, 0.232, 0.30, 0.248, 0.64, color="#D97706", lw=1.5, rad=-0.08)
    _arrow(ax, 0.487, 0.715, 0.503, 0.715, color="#2563EB", lw=1.6)
    _arrow(ax, 0.487, 0.63, 0.503, 0.31, color="#2563EB", lw=1.5, rad=0.08)
    _arrow(ax, 0.747, 0.715, 0.763, 0.715, color="#7C3AED", lw=1.6)
    _arrow(ax, 0.747, 0.30, 0.763, 0.60, color="#7C3AED", lw=1.5, rad=-0.08)
    _arrow(ax, 0.8775, 0.468, 0.8775, 0.412, color="#059669", lw=1.6)

    ax.text(
        0.50, 0.045,
        "Default hyperparameters are specified in Table 2 (frozen prior to confirmatory unlock); step-matched Teacher-Only and Legacy Gate are isolated as controls.",
        ha="center", va="center", fontsize=8.8, style="italic", color="#475569"
    )

    os.makedirs(fig_dir, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(fig_dir, "fig0b_survival_pu_architecture_v3.pdf"))
    fig.savefig(os.path.join(fig_dir, "fig0b_survival_pu_architecture_v3.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fdir = os.path.join(base, "figures")
    build_fig0a_causal_timeline(fdir)
    build_fig0b_architecture(fdir)
    print(f"Generated conceptual diagrams in {fdir}")
