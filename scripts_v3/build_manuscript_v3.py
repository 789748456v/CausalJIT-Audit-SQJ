r"""
Programmatic Manuscript, Online Resource 1, & Replication Documentation Assembler
(`scripts_v3/build_manuscript_v3.py`).

Reads:
  - 11 cell-registered Main empirical tables from `build_artifacts/v3_tables_generated.tex`
  - 16 cell-registered Online Resource 1 empirical tables from `build_artifacts/v3_online_resource_tables.tex`

Writes:
  1. `manuscript/main.tex` (Camera-ready Springer *Software Quality Journal* LaTeX manuscript,
     calibrated to SQJ v3 readability, pacing, 7.5pt table typography, and compact <= 40p structure)
  2. `manuscript/online_resource.tex` (Standalone Springer Online Resource 1 companion PDF containing
     Tables S1--S18: conceptual/architecture tables, full-column mirrors, and secondary audits)
  3. `manuscript/N05_SQJ_Manuscript.md` (Synchronized Markdown manuscript summary)
  4. `README.md` (Complete academic replication & verification guide)
"""
from __future__ import annotations

import os
import re
from typing import Dict

BASE_DIR = os.environ.get(
    "CAUSALJIT_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
BUILD_DIR = os.path.join(BASE_DIR, "build_artifacts")


def load_generated_tables(tables_tex_path: str) -> Dict[str, str]:
    with open(tables_tex_path, "r", encoding="utf-8") as f:
        txt = f.read()
    blocks: Dict[str, str] = {}
    for m in re.finditer(r"\\begin\{table\}\[[^\]]+\].*?\\end\{table\}", txt, re.DOTALL):
        blk = m.group(0)
        lm = re.search(r"\\label\{(tab:[^\}]+)\}", blk)
        if not lm:
            raise ValueError(f"Missing label in generated table:\n{blk[:200]}")
        blocks[lm.group(1)] = blk
    return blocks


def build_main_tex(tmap: Dict[str, str]) -> str:
    tex = r"""% Springer Nature LaTeX Template for Software Quality Journal (SQJ)
% Compilation: pdflatex main && bibtex main && pdflatex main && pdflatex main
\RequirePackage{fix-cm}
\documentclass[pdflatex,sn-basic]{sn-jnl}

% ============================================================================
% Standard Springer Nature Archival Packages
% ============================================================================
\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathtools}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%
\usepackage{tabularx}%
\usepackage{array}%
\usepackage{enumitem}%
\usepackage{xurl}%
\usepackage{microtype}%

\graphicspath{{figures/}{../figures/}}

\setlength{\emergencystretch}{2.0em}
\renewcommand{\arraystretch}{1.06}
\raggedbottom
\hypersetup{bookmarksdepth=2}

\renewcommand{\topfraction}{0.90}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.82}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{3}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{plain}
\newtheorem{proposition}{Proposition}

\newcommand{\rqbox}[2]{%
  \vspace{0.30em}
  \begin{center}
  \setlength{\fboxsep}{6.0pt}%
  \fcolorbox{black!35}{gray!6}{%
    \begin{minipage}{0.945\textwidth}
      \footnotesize\textbf{#1:} #2
    \end{minipage}%
  }
  \end{center}
  \vspace{0.18em}
}

\begin{document}

% ============================================================================
% Title, Authors, & Affiliations (Springer Nature Software Quality Journal)
% ============================================================================
\title[Detecting Drift Is Not Safe Retraining]{Detecting Drift Is Not Safe Retraining: Causal Supervision Boundaries for Adaptive Just-in-Time Defect Prediction}

\author[1]{\fnm{Zhijie} \sur{Li}}\email{lizhijie@mails.neu.edu.cn}
\author[1]{\fnm{Shaoyang} \sur{Xu}}\email{xusy3@mails.neu.edu.cn}
\author[1]{\fnm{Ketong} \sur{Liu}}\email{liuketong@mails.neu.edu.cn}
\author[1]{\fnm{Shibo} \sur{Meng}}\email{mengsb1@mails.neu.edu.cn}
\author[2]{\fnm{Feiran} \sur{Gao}}\email{3120260997@bit.edu.cn}
\author*[1]{\fnm{Yuli} \sur{Zhao}}\email{zhaoyl@swc.neu.edu.cn}

\affil*[1]{\orgdiv{Software College}, \orgname{Northeastern University}, \orgaddress{\street{No.~195 Chuangxin Road, Hunnan District}, \city{Shenyang}, \postcode{110169}, \country{China}}}
\affil[2]{\orgdiv{School of Computer Science and Technology}, \orgname{Beijing Institute of Technology}, \orgaddress{\street{No.~5 South Zhongguancun Street, Haidian District}, \city{Beijing}, \postcode{100081}, \country{China}}}

% ============================================================================
% Unstructured Single-Paragraph Abstract (~215 Words — Springer SQJ Standard)
% ============================================================================
\abstract{Unsupervised drift monitors such as DriftLens flag representation shifts in Just-in-Time Defect Prediction without defect labels, motivating immediate window retraining. Label-free monitoring determines only when to retrain, however, not how to supervise weight updates. Under multi-month SZZ verification latency, most recently introduced defects remain unfixed at alarm time and appear clean. We audit this causal supervision gap and map operating boundaries across linear, streaming, and Positive-Unlabeled learners. Across 34 natural repository streams totaling 347,122 commits, eight controlled shifts, and a paired Code-LLM case study, we evaluate circular block-bootstrap DriftLens alongside three-stage Survival-PU, which combines cohort-incidence maturity estimation, clipped prior shrinkage, and soft-target distillation. Across scheduled windows in multi-year ecosystems, 68.8\% to 94.6\% of recent defects remain right-censored. Leaking future fixes into matched window-oracle updates significantly inflates Average Precision over causal retraining across multi-year donor clusters on both neural and tree hosts. On the shorter confirmatory cohort, however, neural hindsight optimism fails to replicate. Once strict causal visibility holds, no single learner dominates every regime. Frozen linear models lead deployable multimodal streams, simpler causal baselines and zero-fit FixedCurve-PU outperform Survival-PU on gradual whole-stream ranking, and streaming Hoeffding trees led by ODaSC excel on the confirmatory cohort. By contrast, Survival-PU leads full post-shift ranking at 0.5764 Average Precision, avoids zero-latency misspecification, and protects thresholded classification under severe latency where standard post-hoc calibration degrades.}

\keywords{Just-in-Time Defect Prediction, Distribution Shift, Verification Latency, Right-Censoring, Positive-Unlabeled Learning, Survival Analysis}

\maketitle

% ============================================================================
% 1. Introduction
% ============================================================================
\section{Introduction}
\label{sec:intro}

Just-in-Time Software Defect Prediction (JIT-SDP) helps quality engineering teams prioritize code review at commit time, when design context is still fresh \citep{kamei2013large,hoang2019deepjit,pornprasit2021jitline,marrium2026flame}. Static commit classifiers still degrade over time. Refactoring, framework migrations, subsystem turnover, and shifting review norms steadily erode fixed weights \citep{mcintosh2018are,gama2014survey,lu2018learning}. Unlike streaming tasks with instant ground truth, JIT-SDP suffers heavy-tailed \textit{verification latency} \citep{cabral2019class,tabassam2020handling,song2022odasc,song2023humla}. A bug-inducing commit authored at timestamp $t_i$ receives its defect label $y_i^* = 1$ via SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework,rosa2021evaluating,fan2021impact,bludau2026quantifying} only after a fault surfaces and a developer merges a fix at $t_i^{\text{fix}}$. In practice, the fix delay $\Delta_i = (t_i^{\text{fix}} - t_i)/86400$ has a median of three to six months.

To adapt without waiting months for confirmed fixes, recent quality studies adopt \textit{unsupervised embedding-space drift monitors}. \citet{pote2026adaptive} paired bimodal encoders, Attention-CNN and CodeBERT, with DriftLens \citep{greco2021driftlens,greco2023driftlens}. DriftLens tracks the multivariate Gaussian Fr\'echet distance \citep{dowson1982frechet} between pre-stream reference embeddings and incoming commit windows $W_{\text{drift}}$. Because it needs no defect labels, it flags shifts in milliseconds and triggers lightweight retraining on the latest window, escalating to four-window replay after consecutive alarms. Ordering commits chronologically made retrospective evaluation appear aligned with live continuous integration \citep{pote2026adaptive}.

Moving drift-triggered retraining from archival datasets into live continuous integration exposes a fundamental causal supervision gap:

\begin{quote}
\small\textit{Unsupervised drift monitoring removes label dependence only when \underline{deciding when to retrain}, not when \underline{supervising how to update model weights}. At the exact wall-clock timestamp $t_{\text{alarm}}$ when DriftLens flags a shifted window $W_{\text{drift}}$, most genuinely bug-inducing commits inside $W_{\text{drift}}$ are only days or weeks old, so $t_i^{\text{fix}} > t_{\text{alarm}}$. Under strictly causal repository visibility, their observable labels are right-censored into apparent clean negatives, yielding $y_{i,\text{obs}}(t_{\text{alarm}}) = 0$.}
\end{quote}

Our study tests when drift-triggered retraining avoids censoring-induced posterior collapse and recall loss. Three linked questions frame the investigation. First, we measure \textit{what label-visibility timing changes} once future SZZ fixes are blocked alongside commit-time feature hygiene and archive follow-up audits. Second, we test \textit{what three-stage maturity correction genuinely adds} over frozen linear models \citep{zeng2021deep}, zero-fit maturity priors, alarm-synchronized post-hoc Causal-4W controls, and verification-latency stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}. Third, we evaluate \textit{what these differences mean operationally} under equal-cap commit and lines-of-code review budgets.

Our matched audit reveals clear, regime-specific operating boundaries across linear, streaming, and Positive-Unlabeled learners. Across $K=25$ multi-year donor clusters, a Pote-inspired offline window-oracle counterfactual significantly inflates whole-stream Average Precision over matched causal four-window retraining. Circular block-bootstrap calibration also cuts validation-pool null alarm rates by $4.8\times$. Once strict causal visibility holds, simpler learners win several prominent regimes. Frozen tabular logistic regression leads deployable Benchmark~A. On gradual $K=25$ whole-stream ranking, Causal-4W, Wait-90, and zero-fit FixedCurve-PU significantly outperform Survival-PU. Within-window score standardization reverses that mean ranking gap against Causal-4W without reaching statistical significance. On the short confirmatory ApacheJIT cohort, neural hindsight optimism fails to replicate, and adapted stream learners led by ODaSC prevail.

Survival-PU shows targeted, repeatable strengths in three specific regimes. It leads full post-shift ranking on controlled shifts and improves LightGBM soft-target distillation over a step-matched teacher. It also protects classification at both latency extremes. At zero fix delay, it avoids FixedCurve-PU's hardcoded misspecification. Under heavy latency, it beats Causal-4W, two post-hoc controls, and FixedCurve-PU on threshold-tuned $F_{1,\text{th}}$. At the same time, evaluating raw against calibrated scores exposes a concrete hazard: fitting standard 90-day Platt scaling to a heavily censored calibration pool degrades probability quality across methods.

\subsection{Research Questions}
\label{sec:rqs}

We operationalize these goals through five sequential research questions:

\begin{enumerate}[leftmargin=2.2em,label=\textbf{RQ\arabic*:},itemsep=0.02em,topsep=0.25em]
  \item \textbf{Causal Audit of Right-Censoring and Hindsight Optimism.} What fraction of recent defects remain right-censored at scheduled boundaries and fired alarms, and how strongly does offline window-oracle retraining diverge from matched causal retraining?
  \item \textbf{Cross-Paradigm, Granularity, and Confirmatory Operating Boundaries.} How does Survival-PU compare against within-host counterfactuals and linear or stream baselines across four multimodal hosts, $K=25$ natural donor clusters, four controlled shifts, and five confirmatory repositories?
  \item \textbf{Maturity Sensitivity, Mechanism Audit, and Ablations.} How sensitive is Survival-PU to alternative maturity estimators, how accurately do its stages estimate maturity and unrevealed soft targets versus positive writeback, and which components carry weight?
  \item \textbf{Stress Tests Across Latency, Monitoring, Noise, and Prompting.} How do adaptive learners and post-hoc controls behave across latency scaling $\kappa \in [0.0, 3.0]$, a $2 \times 2$ DriftLens factorial, semi-synthetic SZZ noise \citep{bludau2026quantifying}, window sweeps, and a paired Code-LLM case study?
  \item \textbf{Runtime Cost and Equal-Cap Review Budgets.} What cached versus encoding-inclusive overhead does each paradigm incur, and what are the paired defect trade-offs under fixed thresholding versus equal-cap whole-stream and per-window review budgets?
\end{enumerate}

\subsection{Summary of Contributions}
\label{sec:contributions}

Ordered by evidentiary strength, our four contributions are as follows:

\begin{enumerate}[leftmargin=1.7em,itemsep=0.02em,topsep=0.25em]
  \item \textbf{Formal and Empirical Causal Supervision Audit.} Building on drift-triggered JIT-SDP \citep{pote2026adaptive}, delayed feedback \citep{chapelle2014modeling}, and SZZ audits \citep{song2023waiting,bludau2026quantifying}, we formalize future-fix leakage at $t_{\text{alarm}} < t_i^{\text{fix}}$, separate threshold crossing from rank inversion and scale drift, quantify $K=25$ hindsight inflation, and audit archive follow-up across all 34 natural streams.
  \item \textbf{Cross-Paradigm Operating Boundaries and Data Audit.} Across 37 development streams and five confirmatory repositories, we map where linear models, FixedCurve-PU, Causal-4W, BORB, and ODaSC lead, uncover an upstream Hadoop clean-commit split, isolate scale effects via multi-seed standardization, and quantify matched-budget yields.
  \item \textbf{Maturity-Calibrated PU Testbed and Calibration Controls.} We instrument gate-free Survival-PU to test what maturity modeling adds over simpler baselines, documenting identifiability assumptions, bisection unreachability, unrevealed soft-target accuracy, raw versus 90-day Platt calibration against post-hoc controls, and block-bootstrap trade-offs.
  \item \textbf{Open Replication Suite and Automated Verifiers.} We release a checksum-verified benchmark covering 94 stream bundles, 41 prefix-invariant branches, 50 prediction archives, lossless prompting logs, Online Resource~1, and automated cell-, parameter-, and prose-level verifiers.
\end{enumerate}

% ============================================================================
% 2. Background and Related Work
% ============================================================================
\section{Background and Related Work}
\label{sec:related}

We situate our audit across three quality engineering threads: commit-level defect prediction with drift monitors, verification-latency stream learners, and delayed-feedback PU learning (structured comparison in Online Resource~1, Table~S-1).

\subsection{JIT Defect Prediction, Linear Baselines, and Drift Monitoring}
\label{sec:rel_jitsdp}

Classical JIT-SDP represents commits through change metrics capturing size, diffusion, history, and developer experience \citep{kamei2013large}. Later studies added review signals \citep{mcintosh2018are}, AST and pull-request features \citep{bludau2022feature,bludau2022prszz}, graph fault localization \citep{zheng2026hgfe}, and line-level models \citep{pornprasit2021jitline,marrium2026flame}. Deep and tree hosts such as LightGBM \citep{ke2017lightgbm}, DeepJIT \citep{hoang2019deepjit}, and CodeBERT \citep{feng2020codebert} further expanded commit modeling. Simple linear models remain competitive: \citet{zeng2021deep} showed that single-feature LApredict often rivals deep predictors, and \citet{tantithamthavorn2017mysterious} emphasized leak-free validation. Because Gerrit review fields do not exist at initial push, our primary multimodal benchmark retains only the 13 authoring metrics and isolates 28-metric triage in Online Resource~1 (Table~S-4).

To adapt without defect labels, \citet{pote2026adaptive} monitored neural embeddings with DriftLens \citep{greco2021driftlens,greco2023driftlens}. DriftLens tracks Gaussian Fr\'echet distance \citep{dowson1982frechet} to trigger one-window retraining and four-window escalation. \citet{shah2026detecting} showed that representation shifts do not always induce accuracy drops. Because we add causal linear-residual skips and commit-time filtering, we treat window oracle $B_{1b}^*$ strictly as a Pote-inspired counterfactual isolating future-fix visibility.

\subsection{Verification Latency, Stream Learners, and SZZ Bias}
\label{sec:rel_szz}

SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework} introduces both permanent mislabeling \citep{rosa2021evaluating,fan2021impact,keshavarz2022apachejit,bludau2026quantifying} and multi-month verification latency \citep{cabral2019class,tabassam2020handling,song2023humla}. \citet{song2023waiting} audited how fixed waiting windows inject label noise into retrospective evaluation. Our work complements theirs by isolating future-fix leakage inside alarm-time weight updates.

Three latency-aware stream learners anchor our Track-2 comparisons. \citet{cabral2019class} proposed ORB over OzaBagging \citep{oza2005online} of Online Hoeffding Trees \citep{domingos2000mining} with a 90-day clean-maturity wait. \citet{cabral2023borb} followed with BORB, refreshing learners every 200 commits over matured pools. \citet{song2022odasc} proposed ODaSC, pairing a 15-day provisional-negative horizon with DenStream micro-cluster weights.

\subsection{Delayed Feedback and Positive-Unlabeled Maturity Modeling}
\label{sec:rel_pu}

\citet{chapelle2014modeling} formalized delayed feedback in display advertising by jointly modeling positive probability $p(x)$ and exponential delay $\lambda(x)$. Learning from positives and unlabeled mixtures builds on propensity weighting \citep{elkan2008learning,bekker2020learning}, non-negative PU risk \citep{kiryo2017positive}, density ratios \citep{sugiyama2008direct}, and kernel tests \citep{gretton2012kernel}. We adapt maturity estimation specifically to drift-triggered, heterogeneous-host JIT-SDP. Our testbed compares cohort incidence \citep{lyndenbell1971method}, Kaplan--Meier cure \citep{kaplan1958nonparametric}, and global Weibull cure \citep{boag1949cure,farewell1982cure} against zero-fit FixedCurve-PU and post-hoc Causal-4W controls.

% ============================================================================
% 3. Theoretical Analysis
% ============================================================================
\section{Theoretical Analysis of Right-Censored Retraining and Block Calibration}
\label{sec:theory}

We formalize two mechanisms governing live drift-triggered JIT-SDP: how right-censoring reshapes the Bayes-optimal predictor under uniform versus heterogeneous maturity, and how positive lag autocovariance inflates contiguous window-mean variance over independent draws. Complete algebraic proofs appear in Appendix~\ref{app:repro}.

\subsection{Posterior Compression and Rank Inversion Under Right-Censoring}
\label{sec:theory_poison}

Consider a chronologically ordered commit stream $\mathcal{S} = \{(x_i, t_i, y_i^*, t_i^{\text{fix}})\}_{i=1}^N$ with non-decreasing timestamps $t_1 \le \dots \le t_N$ in seconds. Here $x_i \in \mathcal{X}$ holds commit-authoring features, $y_i^* \in \{0, 1\}$ is the archive-horizon defect label, and $t_i^{\text{fix}} \ge t_i$ is the earliest SZZ fix timestamp, set to $+\infty$ when $y_i^* = 0$ or unlinked. The fix delay in days is $\Delta_i = (t_i^{\text{fix}} - t_i)/86400$.

Upstream datasets omit separate collection-completeness timestamps, so we track observation reach via the observable archive-horizon proxy $T_{\text{end}} = \max(\max_i t_i, \max_{i:\, t_i^{\text{fix}} < \infty} t_i^{\text{fix}})$ over finite fixes. Alarm-time right-censoring occurs when $t_{\text{alarm}} < t_i^{\text{fix}} \le T_{\text{end}}$, distinct from end-of-archive censoring where $t_i^{\text{fix}} > T_{\text{end}}$. Section~\ref{sec:benchmarks} and Online Resource~1 (Tables~S-8--S-9) audit remaining follow-up $T_{\text{end}} - t_i$ across all 34 natural streams. Figure~\ref{fig:causal_timeline} contrasts offline evaluation against strictly causal deployment.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.72\textwidth]{fig0a_causal_timeline_v3.pdf}
  \caption{Causal supervision audit at drift alarm $t_{\text{alarm}}$: (a) offline window-oracle evaluation ($B_{1b}^*$) leaks future SZZ fixes ($t_{\text{fix}} > t_{\text{alarm}}$) into $W_{\text{drift}}$ despite chronological commit ordering; (b) live causal deployment ($t \le t_{\text{alarm}}$) faces a trade-off between delayed Wait-90 and naive real-time Causal-1W/4W, addressed via maturity-weighted Bayesian PU soft targets.}
  \label{fig:causal_timeline}
\end{figure}

\begin{definition}[Causal Label Visibility and Structural Maturity Factorization]
\label{def:observability}
At alarm timestamp $t_{\text{alarm}} = t_e$ on window $W_{\text{drift}} = \{e - W + 1, \dots, e\}$, causal deployment forbids accessing events after $t_{\text{alarm}}$. For any commit $i \le e$ of age $a_i = (t_{\text{alarm}} - t_i)/86400 \ge 0$ days, the observable label is
\begin{equation}
  y_{i,\text{obs}}(t_{\text{alarm}}) = \mathbb{I}\!\left(y_i^* = 1 \;\wedge\; t_i^{\text{fix}} \le t_{\text{alarm}}\right) = y_i^* \cdot \mathbb{I}\!\left(\Delta_i \le a_i\right).
  \label{eq:y_obs}
\end{equation}
Assume locally on an adaptation window that (i) latent defect risk is age-stationary given features, $\Pr(y^* = 1 \mid x_i, a_i) = \eta^*(x_i)$, and (ii) reveal delay among true defects is conditionally independent of $x_i$ given age, $\Pr(\Delta \le a_i \mid y^* = 1, x_i, a_i) = c(a_i) \in [0, 1]$. Then $\Pr(y_{\text{obs}} = 1 \mid x_i, a_i) = c(a_i)\,\eta^*(x_i)$.
\end{definition}

\begin{proposition}[Bayes-Optimal Censored Posterior Under Uniform vs.\ Heterogeneous Maturity]
\label{prop:bias}
Under Assumptions~(i)--(ii) of Definition~\ref{def:observability}, suppose predictor $f_\theta(x)$ minimizes class-weighted binary cross-entropy $\ell(p, y_{\text{obs}}) = -\omega_+ y_{\text{obs}} \log p - (1 - y_{\text{obs}})\log(1 - p)$ with positive weight $\omega_+ > 0$ over a window with conditional effective maturity $\bar{c}(x) = \mathbb{E}[c(a) \mid x] > 0$. For any $x$ with $\eta^*(x) \in (0, 1)$, the population minimizer is
\begin{equation}
  \hat{p}_{B_2}(x) = \frac{\omega_+ \, \bar{c}(x) \, \eta^*(x)}{1 + \big(\omega_+ - 1\big)\,\bar{c}(x)\,\eta^*(x)}.
  \label{eq:b2_bayes}
\end{equation}
Equation~\eqref{eq:b2_bayes} implies two structural regimes:
\begin{enumerate}[leftmargin=1.8em,itemsep=0.03em,topsep=0.2em]
  \item \textbf{Uniform-Maturity Order Preservation and Threshold Crossing:} When $\bar{c}(x) \equiv \bar{c}_W \in (0, 1)$ is constant across features, $\partial \hat{p}_{B_2}/\partial \eta^* = \omega_+ \bar{c}_W / [1 + (\omega_+ - 1)\bar{c}_W \eta^*]^2 > 0$, preserving population AP and ROC-AUC. Relative to the uncensored weighted minimizer $\hat{p}_{B_1^*}(x)$, right-censoring compresses $\hat{p}_{B_2}(x) < \hat{p}_{B_1^*}(x)$ and forces $\hat{p}_{B_2}(x) < \tau_{\text{th}}$ if and only if
  \begin{equation}
    \bar{c}_W < \frac{\tau_{\text{th}}}{\eta^*(x)\,\big[\omega_+(1 - \tau_{\text{th}}) + \tau_{\text{th}}\big]}.
    \label{eq:exact_crossing}
  \end{equation}
  \item \textbf{Heterogeneous-Maturity Rank Inversion:} Suppose subpopulations $x_{\text{old}}$ and $x_{\text{new}}$ have maturities $\bar{c}(x_{\text{old}}) = c_{\text{hi}} > c_{\text{lo}} = \bar{c}(x_{\text{new}}) > 0$ with $\eta^*(x_{\text{new}}) > \eta^*(x_{\text{old}})$. Censored training inverts their order, giving $\hat{p}_{B_2}(x_{\text{new}}) < \hat{p}_{B_2}(x_{\text{old}})$, if and only if
  \begin{equation*}
    \frac{\eta^*(x_{\text{new}})}{1 - c_{\text{lo}}\eta^*(x_{\text{new}})} < \frac{c_{\text{hi}}}{c_{\text{lo}}} \cdot \frac{\eta^*(x_{\text{old}})}{1 - c_{\text{hi}}\eta^*(x_{\text{old}})}.
  \end{equation*}
\end{enumerate}
\end{proposition}

\textit{Proof in Appendix~\ref{app:repro}.} Right-censoring does not always worsen raw Brier error when $\omega_+ > 1$ inflates uncensored posteriors. At $\eta^* = 0.5$ and $\omega_+ = 4$, for instance, the uncensored weighted minimizer yields $0.80$, and censoring at $\bar{c}_W = 0.40$ pulls $\hat{p}_{B_2}$ back to $0.50$. Moreover, Claim~1's monotone ranking holds only within a single window's model. Concatenating predictions across post-alarm windows introduces inter-window score-scale shifts as estimated priors $\hat{\pi}_{\text{target}}(t)$ or class weights change. Proposition~\ref{prop:bias} therefore explains how Causal-4W retains gradual whole-stream AP, loses thresholded recall under heavy lag, and loses ranking accuracy when shifted features arrive young.

\subsection{Window-Mean Variance Under Serial Autocorrelation}
\label{sec:theory_alarm_storm}

Every alarm exposes the model to right-censored supervision, tying monitor calibration directly to update safety. Commits arrive in serially correlated sprints and refactorings, breaking the i.i.d.\ assumption behind uniform bootstrap calibration.

\begin{proposition}[Window-Mean Covariance Under Positive Lag Autocovariance]
\label{prop:autocorr}
Let $\{z_t\}_{t \in \mathbb{Z}} \subset \mathbb{R}^d$ be a weakly stationary embedding sequence with mean $\mu_0$, marginal covariance $\Gamma_0$, and lag-$k$ autocovariance $\Gamma_k = \operatorname{Cov}(z_t, z_{t+k})$ satisfying $\Delta_\Gamma \triangleq \sum_{k=1}^{W-1}\frac{W-k}{W}(\Gamma_k + \Gamma_k^\top) \succ 0$. Comparing the contiguous window mean $\bar{z}_{\text{contig}} = \frac{1}{W}\sum_{j=1}^W z_{t_0+j}$ against the mean $\bar{z}_{\text{ind}}$ of $W$ independent draws yields
\begin{equation*}
  \operatorname{Cov}(\bar{z}_{\text{contig}}) = \operatorname{Cov}(\bar{z}_{\text{ind}}) + \frac{1}{W}\Delta_\Gamma \succ \operatorname{Cov}(\bar{z}_{\text{ind}}), \qquad
  \mathbb{E}\|\bar{z}_{\text{contig}} - \mu_0\|_2^2 > \mathbb{E}\|\bar{z}_{\text{ind}} - \mu_0\|_2^2.
\end{equation*}
\end{proposition}

\textit{Proof in Appendix~\ref{app:repro}.} Uniform resampling breaks sprint autocorrelation across positive lags \citep{kunsch1989jackknife,politis1992circular}. Proposition~\ref{prop:autocorr} characterizes the mean-shift component rather than the full Fr\'echet statistic, and motivates calibrating DriftLens with circular contiguous blocks of length $b = \max(\min(\lfloor W/6 \rfloor, 80), 25)$.

To separate resampling from post-alarm updating cleanly, RQ4b runs a $2 \times 2$ factorial at fixed PCA dimension, 95th percentile, and unit cooldown. It crosses IID versus Circular Block resampling against Static versus Rolling Ref+Th updating, alongside Pote et al.'s static 98th-percentile reference with zero cooldown. Under Rolling Ref+Th, each alarm refreshes reference statistics on trailing $3W$ commits and blends thresholds via $0.70\,\tau_{\text{new}} + 0.30\,\tau_{\text{old}}$.

% ============================================================================
% 4. Methodology: Three-Stage Survival-PU Adaptation Protocol
% ============================================================================
\section{Methodology: Three-Stage Survival-PU Adaptation Protocol}
\label{sec:method}

Proposition~\ref{prop:bias} points to three coupled goals for causal adaptation at alarm timestamp $t_{\text{alarm}}$ using only $t_i \le t_{\text{alarm}}$: estimate commit-age maturity $c(a_i)$ without right-truncation bias, estimate target prevalence $\hat{\pi}_{\text{target}}$ with a reliable teacher, and distil soft targets across $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$. Figure~\ref{fig:architecture} outlines our gate-free three-stage Survival-PU pipeline. All 33 locked protocol parameters appear in Appendix~Table~\ref{tab:hyperparams}, with exact host constructors in Online Resource~1 (Table~S-2). Legacy three-state gating is excluded from the core method and tested only as an ablation.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.72\textwidth]{fig0b_survival_pu_architecture_v3.pdf}
  \caption{Three-stage input--output pipeline of Survival-PU at each DriftLens alarm $t_{\text{alarm}}$: (Stage~1) risk-eligible cohort cumulative incidence $c_i$ on trailing history $\mathcal{H}_{\text{rec}}$; (Stage~2) reliable teacher $T_\phi$ and clipped inverse-maturity shrunk prevalence $\hat{\pi}_{\text{target}}$ ($\tilde{c}_i = \max(c_i, 0.10)$) on $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$; and (Stage~3) monotone bisection producing maturity- and prior-adjusted soft targets $q_i(\delta^*)$ across $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$.}
  \label{fig:architecture}
\end{figure}

\subsection{Stage 1: Risk-Eligible Cohort Incidence and Survival Controls}
\label{sec:stage1}

A naive empirical CDF over observed fixes underestimates latency because slow bugs are absent from recent commits. Stage~1 slices trailing $N_{\text{hist}}$ causal commits $\mathcal{H}_{\text{rec}} \subset \{i : t_i \le t_{\text{alarm}}\}$. For each commit, it records age $a_i = \max((t_{\text{alarm}} - t_i)/86400, 0)$ and observed fix delay $d_i = \max((t_i^{\text{fix}} - t_i)/86400, 0)$ when $y_{i,\text{obs}}(t_{\text{alarm}}) = 1$. Unrevealed commits and quarantined retroactive links receive $d_i = +\infty$.

We evaluate knots $g_0 = 0 < g_1 < \dots < g_M = a_{\max}$ up to support quantile $a_{\max} = Q_{q_{\max}}(\{a_i\})$, enforcing risk-set floor $n_{\text{eff}}^{\text{risk}} = \max(\min(n_{\min}^{\text{risk}}, \lfloor (1 - q_{\max})|\mathcal{H}_{\text{rec}}| \rfloor), 25)$. Restricting the denominator to commits aged at least $g_m$, $\mathcal{E}(g_m) = \{i \in \mathcal{H}_{\text{rec}} : a_i \ge g_m\}$, carrying forward $\hat{G}(g_{m-1})$ below the risk floor, and taking a running maximum yields
\begin{equation}
  \hat{G}(g_m) = \frac{1}{|\mathcal{E}(g_m)|}\sum_{i \in \mathcal{E}(g_m)} \mathbb{I}\!\left(y_{i,\text{obs}} = 1 \;\wedge\; d_i \le g_m\right), \qquad \tilde{G}(g_m) = \max_{0 \le k \le m}\hat{G}(g_k).
  \label{eq:cohort_inc}
\end{equation}
Because $\tilde{G}(a_{\max})$ reflects finite observation up to $a_{\max}$, Stage~1 rescales by bounded tail-completeness factor $\gamma_{\text{tail}} \in [\gamma_{\min}, \gamma_{\max}]$ using half-horizon incidence $\tilde{G}_{\text{half}}$ and tail growth ratio $r_g$ (default $1.5$ when $\tilde{G}_{\text{half}} \le 10^{-5}$):
\begin{equation}
  \begin{aligned}
    r_g &= \frac{\tilde{G}(a_{\max})}{\tilde{G}_{\text{half}}}, \qquad \phi_a = \operatorname{clip}\!\left(\frac{a_{\max}}{a_{\max} + 45},\; 0.65,\; 1.0\right), \\
    \gamma_{\text{tail}} &= \operatorname{clip}\!\Big(\big[\gamma_{\max} - 0.28\operatorname{clip}(r_g - 1, 0, 1)\big]\,\phi_a,\; \gamma_{\min},\; \gamma_{\max}\Big),
  \end{aligned}
  \label{eq:tail_cap}
\end{equation}
and clips knot maturities before piecewise-linear interpolation:
\begin{equation}
  \begin{aligned}
    c_{\text{knot}}(g_m) &= \operatorname{clip}\!\Big(\gamma_{\text{tail}}\,\frac{\tilde{G}(g_m)}{\tilde{G}(a_{\max})},\; c_{\min},\; c_{\max}\Big), \\
    c_i &= \operatorname{clip}\!\big(\operatorname{interp}(a_i;\, \{g_m\}, \{c_{\text{knot}}(g_m)\}),\; c_{\min},\; c_{\max}\big),
  \end{aligned}
  \label{eq:ci_interp}
\end{equation}
bounded by $c_{\min}$ on the left and $c_{\text{knot}}(g_M)$ on the right. Whenever support is sparse (too few observed fixes, insufficient history, short support horizon, or near-zero incidence), Stage~1 applies a deterministic fallback $c_{\text{fb}}(g) = \operatorname{clip}(0.88(1 - 2^{-g/90}), c_{\min}, c_{\max})$ on uniform 5-day knots up to 720 days.

RQ2 and RQ3 compare three Stage~1 controls under the same interface: quantized Kaplan--Meier Cure ($10^{-4}$-day grid preserving $0 \le e_u \le n_u$), unstratified Weibull Mixture-Cure MLE \citep{boag1949cure,farewell1982cure}, and zero-fit FixedCurve-PU setting $c_{\text{fix}}(a_i) = \operatorname{clip}(1 - 2^{-a_i/90}, c_{\min}, c_{\max})$. Canceling $\pi$ in Eq.~\eqref{eq:ci_interp} assumes that trailing-cohort prevalence $\pi$ and delay law $F_\Delta$ are locally stationary inside $\mathcal{H}_{\text{rec}}$ and that $\gamma_{\text{tail}}$ approximates completeness at $a_{\max}$. Abrupt prevalence shifts can confound prevalence with reveal speed, which we audit across 62 alarms in Section~\ref{sec:rq3}.

\subsection{Stage 2: Clipped Inverse-Maturity Prevalence and Reliable Teacher}
\label{sec:stage2}

At each alarm, $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ combines trailing anchors $\mathcal{A}_{\text{anc}}$ with the flagged window $W_{\text{drift}}$. An $L_2$-logistic domain classifier separates $\mathcal{A}_{\text{anc}}$ from $W_{\text{drift}}$ to produce bounded density ratios $w_{\text{imp}}(x_i) \in [w_{\min}, w_{\max}]$ \citep{sugiyama2008direct}. On $\mathcal{A}_{\text{anc}}$, multiplying $w_{\text{imp}}(x_i)$ by linear recency in $[0.70, 1.15]$ gives base weights $w_i$; commits in $W_{\text{drift}}$ receive $w_i = \gamma_{\text{drift}}$. Multiplying by maturity yields observation-independent weights $u_i = w_i[u_0 + (1 - u_0)c_i]$.

Dividing $y_{i,\text{obs}}$ by tiny $c_i \approx 0.05$ inflates variance. Stage~2 therefore uses the trailing subset $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$ of size $N_{\text{cal}} = \min(|\mathcal{B}|, \max(3|W_{\text{drift}}|, 900))$ with floor $\tilde{c}_i = \max(c_i, c_{\text{ht}})$ at $c_{\text{ht}} = 0.10$, blending clipped inverse-maturity estimation with shrinkage toward mature anchor prior $\pi_{\text{anc}}$:
\begin{equation}
  \begin{aligned}
    \hat{\pi}_{\text{IP}} &= \frac{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\,(y_{i,\text{obs}} / \tilde{c}_i)}{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i}, \qquad n_{\text{eff,heur}} = \frac{\big(\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\big)^2}{\sum_{i \in \mathcal{B}_{\text{cal}}} (u_i / \tilde{c}_i)^2}, \\
    \hat{\pi}_{\text{target}} &= \operatorname{clip}\!\left(\max\!\left(\frac{n_{\text{eff,heur}}\,\hat{\pi}_{\text{IP}} + m_0\,\pi_{\text{anc}}}{n_{\text{eff,heur}} + m_0},\; \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4}\right),\; \pi_{\min},\; \pi_{\max}\right).
  \end{aligned}
  \label{eq:ht_prev}
\end{equation}
Here $\pi_{\text{anc}}$ averages inverse-maturity labels over $\{j : c_j \ge 0.55\}$ (falling back to pre-stream prevalence below 80 commits), and $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ is the weighted observed positive rate. Whenever $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} > 0.70$, the outer cap $\pi_{\max} = 0.70$ clamps $\hat{\pi}_{\text{target}}$ below the observed positive rate and makes Stage~3's bisection target unreachable.

In parallel, Stage~2 fits teacher $T_\phi$ on trailing $N_{\text{rel}}$ commits of the reliable cohort $\mathcal{R} = \{i : y_{i,\text{obs}} = 1 \vee c_i \ge \tau_{\text{rel}}\}$ whenever $|\mathcal{R}| \ge 100$:
\begin{equation}
  \begin{aligned}
    w_{\text{rel},i} &= w_{\text{imp}}(x_i) \cdot r_{\text{conf},i} \cdot \text{recency}_i, \qquad
    r_{\text{conf},i} = \begin{cases} 0.55 + 0.45\,c_i, & c_i < \tau_{\text{rel}}, \\ 1, & c_i \ge \tau_{\text{rel}}, \end{cases} \\
    \omega_{+,\text{teach}} &= \tfrac{1}{2}\omega_+ + \tfrac{1}{2}\operatorname{clip}\!\left(\frac{1 - \hat{\pi}_{\text{target}}}{\max(\hat{\pi}_{\text{target}}, 0.03)},\; 1.5,\; \max(4.5, \omega_+)\right).
  \end{aligned}
  \label{eq:teacher_weights}
\end{equation}
Neural teachers start from an $85\%/15\%$ blend of the current checkpoint and initial anchor for $E_{\text{teach}}$ epochs; LightGBM fits a fresh $R_{\text{alm}}$-tree booster. Both emit class-weight-adjusted log-odds $\eta_i^{(0)} = \tau_{\text{sharp}}(z_{\text{teach}}(x_i) - \log \omega_{+,\text{teach}})$. Because residual censoring remains among aged commits, Stage~3 treats $q_i(\delta^*)$ as maturity- and prior-adjusted soft targets rather than exact posteriors.

\subsection{Stage 3: Soft Targets, Linear-Residual Skip, and Callable GBDT}
\label{sec:stage3}

Stage~3 sets $q_i(\delta) = 1$ when $y_{i,\text{obs}} = 1$ and assigns every unrevealed commit ($y_{i,\text{obs}} = 0$) the target
\begin{equation}
  q_i(\delta) = \frac{\tilde{p}_i(\delta)\,(1 - c_i)}{1 - \tilde{p}_i(\delta)\,c_i}, \qquad \tilde{p}_i(\delta) = \operatorname{clip}\!\big(\sigma(\eta_i^{(0)} + \delta),\; p_{\min},\; p_{\max}\big).
  \label{eq:bayes_pu}
\end{equation}
This drives $q_i \to 0$ on aged unfixed commits and preserves teacher risk on young commits. To calibrate mean target prevalence on $\mathcal{B}_{\text{cal}}$, Stage~3 runs $K_{\text{bis}}$-step monotone bisection over $\delta \in [-8, 8]$ targeting $\sum_{i \in \mathcal{B}_{\text{cal}}} u_i q_i(\delta) / \sum_{i \in \mathcal{B}_{\text{cal}}} u_i = \hat{\pi}_{\text{target}}$, clamping at the boundary when unreachable. Auditing all runs confirms zero unreachable solves on Benchmark~A and RQ4a--e and $0.62\%$ unreachable alarms across Benchmark~B's natural streams, rising to $28.6\%$ (MLP) and $30.0\%$ (LightGBM) on Confirmatory HDFS* due to its elevated observed prevalence.

Student training minimizes $\mathcal{L}_{\text{SPU}}(z) = \sum_{i \in \mathcal{B}} u_i [\omega_{+,\text{teach}} q_i(\delta^*) \log(1 + e^{-z_i}) + (1 - q_i(\delta^*))\log(1 + e^{z_i})]$. Neural students train for $E_{\text{stud}}$ epochs from $T_\phi$; LightGBM fits a fresh $R_{\text{alm}}$-tree booster matching Teacher-Matched's $2R_{\text{alm}}$ budget.

Two host mechanisms ensure fair execution across model families. Multimodal neural hosts add a frozen causal tabular linear-residual skip $z(x) = w_{\text{LR}}^\top x_{\text{tab}} + b_{\text{LR}} + f_\theta(x)$ \citep{zeng2021deep}, chosen on pre-stream validation over a trainable skip ($0.0586$ vs.\ $0.0575$ AP). Meanwhile, LightGBM uses exact analytical continuous soft-target derivatives $(g_i, h_i)$ with consistent initial margin $z_{\text{init}}$ \citep{friedman2001greedy,ke2017lightgbm} (Appendix~\ref{app:repro}, Proposition~\ref{prop:lgbm}).

% ============================================================================
% 5. Experimental Design & Evaluation Protocol
% ============================================================================
\section{Experimental Design and Evaluation Protocol}
\label{sec:setup}

We follow a strict causal sequence: audited repository cohorts first, two-track learners second, and endpoints plus statistical protocol third. Development analyses on Benchmarks~A and~B had codified comparison families prior to confirmatory unlock; Phase~C tests five untouched ApacheJIT repositories with one disclosed post-unlock split-sizing rule for small repositories.

\subsection{Audited Stream Cohorts, Data Funnels, and Archive Follow-Up}
\label{sec:benchmarks}

Our corpus spans 37 development streams and five confirmatory repositories, totaling 347,122 commits across 34 natural streams plus a 6,400-call Code-LLM study.

\paragraph{Benchmark A: Multimodal Qt and OpenStack.}
Cross-linking DeepJIT \citep{hoang2019deepjit}, \citet{mcintosh2018are}, and Git histories yields 25,704 Qt and 13,304 OpenStack commits. Verified high-confidence SZZ links cover 96.49\% of Qt and 99.32\% of OpenStack defects; heuristic and unlinked defects are quarantined to $+\infty$. Primary streams retain only the 13 authoring metrics and isolate 28-metric pre-merge triage in Online Resource~1 (Table~S-4). Because exports provide only \texttt{author\_date}, Benchmark~A models merged-commit replay across five seeds of Attention-CNN \citep{pote2026adaptive}, CodeBERT \citep{feng2020codebert}, DeepJIT-CNN \citep{hoang2019deepjit}, and LightGBM \citep{ke2017lightgbm}.

\paragraph{Benchmark B: Natural Donor Clusters and Controlled Shifts.}
Benchmark~B integrates 14 CommitGuru streams \citep{cabral2019class}, eight ApacheJIT streams \citep{keshavarz2022apachejit}, and five PR-SZZ streams \citep{bludau2026quantifying}, totaling 300,637 commits across $K=25$ independent donor clusters ($K=24$ Benchmark-A-disjoint check). Four controlled concatenated shifts (Nova$\to$Zulip, Ignite$\to$Cassandra, Airflow$\to$Pulsar, and OpenStack Nova$\to$Neutron) stay outside $K=25$ inference, imposing a one-hour splice across which $23.7\%$ to $60.0\%$ of revealed pre-onset fixes arrive after onset (Online Resource~1, Table~S-6).

\paragraph{Phase C: Confirmatory ApacheJIT Cohort and Monorepo Disclosure.}
All modeling and statistical code was frozen before downloading \texttt{apachejit\_total.csv} \citep{keshavarz2022apachejit}. Applying Benchmark~B's chronological cutoff yielded 7,477 commits across five repositories with at least one year of follow-up. Auditing the upstream file (Online Resource~1, Table~S-7) showed that 11,964 clean Hadoop commits with zero buggy rows reside under \texttt{apache/hadoop}, inflating evaluation prevalence in HDFS* ($89.2\%$) and MapReduce* ($78.8\%$).

Spark, Zeppelin, and ZooKeeper undergo no monorepo split and keep $44.6\%$ to $48.5\%$ cut prevalence, so we foreground these three standalone repositories alongside the frozen four-umbrella record. Because four confirmatory repositories had under two thousand commits, we added one post-unlock split-sizing branch for $N < 4{,}000$ without modifying any development bundle or parameter:
\begin{equation*}
  \begin{aligned}
    n_{\text{init}} &= \operatorname{clip}\!\big(\operatorname{round}(0.25N/25)\cdot 25,\, 150,\, 1500\big), \\
    n_{\text{val}} &= \operatorname{clip}\!\big(\operatorname{round}(0.08N/25)\cdot 25,\, 50,\, 500\big).
  \end{aligned}
\end{equation*}

\paragraph{Observable Archive-Horizon Follow-Up Audit.}
Online Resource~1 (Table~S-8) audits remaining follow-up $T_{\text{end}} - t_i$ across all 34 natural streams. Every CommitGuru and Phase~C commit, plus over $99.3\%$ of Benchmark~B ApacheJIT commits, retains ten months to over a year of follow-up, and four of five PR-SZZ streams retain nearly a year.

Natural Qt extends to $T_{\text{end}}$ ($15.5\%$ below 180 days and $41.0\%$ below 365 days), and OpenStack leaves $30.4\%$ below 365 days. Re-evaluating Benchmark~A at restricted follow-up cutoffs (Online Resource~1, Table~S-9) preserves Full-LR's AP lead and Survival-PU's Track-1 $F_1$ gain across every slice, with minor pairwise AP swaps on one-year subsets.

\subsection{Two-Track Learners and Baseline Controls}
\label{sec:baselines}

We evaluate 29 methods (26 deployable plus three hindsight references across 41 branches) in a two-track design controlling compute budget, sample pool, and trigger timing (full constructor specs in Online Resource~1, Table~S-2):
\begin{itemize}[leftmargin=1.6em,itemsep=0.02em,topsep=0.15em]
  \item \textbf{Track 1 Within-Host Counterfactuals.} Holding host architecture and DriftLens triggers fixed, Track~1 compares Frozen and Pote-inspired hindsight oracles against single- and four-window causal replay, delayed wait baselines (Wait-30/60/90), Pure-nnPU \citep{kiryo2017positive}, ODaSC-Host, exact and step-matched Teacher-Only, zero-fit FixedCurve-PU, legacy-gated Survival-PU, and gate-free Survival-PU. RQ4a adds alarm-synchronized PostHoc Logit and PostHoc Ratio controls.
  \item \textbf{Track 2 Linear and Adapted Stream Baselines.} Track~2 pairs unsupervised LOC-Only with frozen and adaptive LApredict \citep{zeng2021deep}, Churn-LR, and Full-LR. It adds adapted ElasticNet BORB \citep{cabral2023borb} (native 200-commit and alarm-triggered BORB-Alarm) plus event-ordered River Hoeffding ensembles for ORB \citep{cabral2019class} and ODaSC \citep{song2022odasc} consuming $t_{\text{event}} < t_{\text{commit}}$ events; BORB results apply specifically to this ElasticNet instance.
\end{itemize}

\subsection{Endpoints, Causal Calibration, and Statistical Protocol}
\label{sec:metrics_protocol}

\paragraph{Endpoints and Three-Branch Causal Calibration.}
Whole-stream Average Precision (AP) is our primary endpoint, contrasted explicitly with per-window AP in Section~\ref{sec:rq2}. Secondary endpoints cover ROC-AUC, fixed-threshold $F_1$, causally tuned $F_{1,\text{th}}$, Recall@20\%Effort, raw and post-Platt Brier score, ECE, and calibration slope.

At each evaluation window $w = [s_w, e_w)$ with cutoff $t_w = t_{s_w - 1}$, our prefix-invariant calibrator selects up to $1{,}200$ prior evaluation commits aged at least 90 days whenever at least 80 qualify, falling back to the initialization slice or a prior-odds shift at threshold $0.50$. On qualifying pools $\mathcal{M}_w$, it fits 1D $L_2$-Platt scaling \citep{platt1999probabilistic} and searches 60 thresholds on raw scores $\hat{p}_j = \sigma(z_j)$:
\begin{equation}
  \begin{aligned}
    \hat{p}_{i,\text{cal}} &= \sigma(a_w z_i + b_w), \qquad \text{Brier}_{\text{cal}} = \frac{1}{N_{\text{eval}}}\sum_{i=s_0}^{N-1}\big(\hat{p}_{i,\text{cal}} - y_i^*\big)^2, \\
    \tau_{\text{th}}(w) &= \operatorname*{arg\,max}_{\tau \in \mathcal{G}_w} F_1\!\Big(\big\{\mathbb{I}(\hat{p}_j \ge \tau),\, y_{j,\text{obs}}(t_w)\big\}_{j \in \mathcal{M}_w}\Big).
  \end{aligned}
  \label{eq:causal_cal}
\end{equation}
All 41 execution branches passed deterministic prefix-invariance checks with zero perturbation error. Across empirical tables, bold marks the best deployable value per row or column excluding hindsight oracles; in Table~\ref{tab:bench_b_holm}, bold marks $p_{\text{Holm}} < 0.05$.

\paragraph{Cluster Aggregation and Multiplicity Control.}
Stream scores average across random seeds (ten for MLP, five for LightGBM and multimodal hosts), and donor-cluster scores average equally within each cluster. Formal $\alpha = 0.05$ inference covers $K=25$ natural donor clusters and the Benchmark-A-disjoint $K=24$ check via exact two-sided paired Wilcoxon signed-rank tests (Pratt tie convention), Holm--Bonferroni adjustment \citep{holm1979simple}, unadjusted cluster bootstrap 95\% CIs, and Cohen's $d_z$.

Comparisons divide into three families: Family~1 tests Window Oracle, Frozen, and Survival-PU against Causal-4W; Family~2 tests Survival-PU against Frozen, Wait-90, Pure-nnPU, Full-LR, BORB, ORB, and ODaSC; Family~3 tests Survival-PU against FixedCurve-PU, Teacher-Matched, Teacher-Exact, and legacy gating. Non-significant contrasts report non-establishment verbatim without claiming equivalence. Diagnostic $p$-values outside Table~\ref{tab:bench_b_holm} are exploratory unadjusted Wilcoxon tests, and all deltas use unrounded means.

% ============================================================================
% 6. Empirical Results
% ============================================================================
\section{Empirical Results}
\label{sec:results}

We report empirical findings in sequential RQ order, progressing from the causal supervision audit (RQ1) to operating boundaries (RQ2), maturity sensitivity and mechanism audits (RQ3), stress tests and calibration (RQ4), and runtime plus review-budget yield (RQ5).

% ----------------------------------------------------------------------------
% 6.1 RQ1
% ----------------------------------------------------------------------------
\subsection{RQ1: Causal Audit of Right-Censoring and Hindsight Optimism}
\label{sec:rq1}

\paragraph{Severe Right-Censoring Across Multi-Year Cohorts.}
Table~\ref{tab:audit_censoring} and Figure~\ref{fig:rq1_overview}(a) track right-censoring at both scheduled commit boundaries and actual fired alarms. Across multi-year ecosystems, median SZZ fix delays span three to nine months. Most recent defects therefore remain unrevealed whenever a drift alarm triggers an update. Across scheduled single-window boundaries, 94.6\% of window defects in natural Qt remain unfixed, corresponding to a pooled rate of $95.3\%$ across $981$ window-defects.

Every other multi-year cohort exhibits similarly heavy censoring, ranging from $68.8\%$ in CommitGuru to $87.5\%$ in OpenStack, with fired alarms tracking scheduled rates closely. Even expanding replay to four windows leaves $45.1\%$ to $80.5\%$ of defects unrevealed across multi-year cohorts. By contrast, the shorter confirmatory ApacheJIT cohort sees lighter censoring ($26.4\%$ overall and $18.0\%$ across the three standalone repositories), illustrating how archive cadence governs alarm-time visibility.

[[TAB_AUDIT_CENSORING]]

\paragraph{Matched Hindsight Optimism Versus Causal Replay.}
To quantify how strongly future-fix leakage inflates accuracy, we compare offline window-oracle retraining ($B_{1b}^*$) against matched causal four-window replay ($B_2\text{-4W}$) across $K=25$ natural donor clusters (Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}). On MLP, $B_{1b}^*$ reaches $0.5241$ AP versus $0.5004$ for $B_2\text{-4W}$, a statistically significant inflation of $\Delta\text{AP} = +0.0237$ ($p_{\text{Holm}} = 6.63 \times 10^{-5}$).

On LightGBM, hindsight optimism more than doubles. Here $B_{1b}^*$ reaches $0.5409$ AP versus $0.4933$ for $B_2\text{-4W}$, gaining $\Delta\text{AP} = +0.0476$ ($p_{\text{Holm}} = 3.70 \times 10^{-5}$). Restricting causal replay to a single window depresses accuracy further on both hosts (Table~\ref{tab:bench_b_clusters}).

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig1_multimodal_audit_v3.pdf}
  \caption{(a) Window-averaged fraction of unrevealed true defects across scheduled $400$-commit ($1W$) and $1{,}600$-commit ($4W$) boundaries (RQ1). (b) Six-stream mean AP across all four Benchmark~A host families with frozen linear skips relative to frozen tabular Full-LR ($0.3653$), bridging directly into RQ2.}
  \label{fig:rq1_overview}
\end{figure}

\paragraph{Why Censoring Strikes Differently Across Metrics and Regimes.}
Hindsight inflation does not mean naive causal replay always fails. Consistent with Proposition~\ref{prop:bias}, older commits inside a four-window buffer retain enough partial maturity for Causal-4W to beat frozen $B_0$ on full-stream MLP AP across gradual natural clusters, at $0.5004$ versus $0.4566$ ($p_{\text{Holm}} = 0.0041$). It also improves over freezing on the controlled abrupt shifts, reaching $0.5578$ versus $0.5064$ post-onset AP.

Still, Causal-4W trails matched oracle updates in both regimes. Censoring strikes hardest along the two axes predicted by Proposition~\ref{prop:bias}. Fixed-threshold $F_1$ collapses on multimodal Benchmark~A, where Causal-4W and Causal-1W fall far behind Survival-PU (Table~\ref{tab:bench_a_main}). Meanwhile, single-window causal replay falls below freezing on the controlled abrupt shifts.

\rqbox{Summary for RQ1}{Across scheduled single-window boundaries in multi-year ecosystems, 68.8\%--94.6\% of recent defects remain right-censored, with fired alarms tracking scheduled rates closely. Leaking future fixes significantly inflates whole-stream AP across $K=25$ clusters on both MLP and LightGBM. Four-window causal replay still beats freezing on gradual AP, yet it trails matched oracles and suffers severe fixed-threshold $F_1$ loss on multimodal hosts.}

% ----------------------------------------------------------------------------
% 6.2 RQ2
% ----------------------------------------------------------------------------
\subsection{RQ2: Multi-Host, Cross-Project, and Confirmatory Operating Boundaries}
\label{sec:rq2}

\paragraph{Benchmark A: Linear Predictors Lead Deployable Multimodal Hosts.}
Table~\ref{tab:bench_a_main} and Figure~\ref{fig:rq1_overview}(b) expose a clear operating boundary across the six Benchmark~A streams (complete 10-column per-host breakdown in Online Resource~1, Table~S-3). Frozen logistic regression, Full-LR Frozen, leads every deployable single-host configuration at $0.3653$ AP, closely followed by single-feature LApredict Frozen. Among the four multimodal and tree hosts, the strongest deployable individual model is frozen DeepJIT-CNN at $B_0 = 0.3531$ AP. Only non-deployable full-prefix hindsight DeepJIT-CNN ($B_{1a}^* = 0.3768$) exceeds Full-LR Frozen. Simple Churn-LR Frozen also beats every window-adapted four-host mean, and unsupervised LOC-Only reaches $0.6520$ R@20\%Effort, ahead of Survival-PU's $0.6156$.

[[TAB_BENCH_A_MAIN]]

\paragraph{Architectural Skip Gains and Sensitivity Robustness.}
Adding our causal tabular linear-residual skip narrows the neural deficit substantially (Figure~\ref{fig:rq2_skip}(b)), lifting six-stream frozen AP by roughly 8 to 17 points across Attention-CNN, CodeBERT, and DeepJIT-CNN. Even with this anchor, fine-tuning high-capacity encoders on small drift windows still trails frozen linear models in AP. Within identical Track-1 hosts, base-rate drift and censoring erode fixed-threshold $F_1$ on Frozen and Causal-4W, whereas Survival-PU recovers the highest Track-1 deployable $F_1$ and $F_{1,\text{th}}$ at lower mean AP (Table~\ref{tab:bench_a_main}).

Provenance and feature-scope sensitivity checks (Online Resource~1, Table~S-4) confirm that Causal-4W leads deployable AP across Qt variants and Frozen leads across OpenStack. Archive follow-up checks at 180-day and 365-day cutoffs (Online Resource~1, Table~S-9) likewise preserve Full-LR's AP lead and Survival-PU's Track-1 $F_1$ gain.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig2_trajectories_and_skip_v3.pdf}
  \caption{(a) Prequential 3-window rolling AP trajectory on natural Qt (CodeBERT with frozen linear skip). (b) Effect of adding the causal tabular linear-residual skip (no skip vs.\ dev-selected frozen linear skip) on natural Qt and OpenStack.}
  \label{fig:rq2_skip}
\end{figure}

\paragraph{Benchmark B: Gradual Natural Donor Clusters.}
Table~\ref{tab:bench_b_clusters} and Figure~\ref{fig:rq2_bench_b}(a) report powered cross-project inference across $K=25$ natural donor clusters alongside the Benchmark-A-disjoint sensitivity check (full 8-column mirror with ROC-AUC in Online Resource~1, Table~S-5). On gradual whole-stream MLP AP, Full-LR Adapt-4W leads in mean at $0.5108$, followed by Causal-4W at $0.5004$, Wait-90, and zero-fit FixedCurve-PU.

Secondary endpoints split cleanly across paradigms (Table~\ref{tab:bench_b_clusters}). ElasticNet BORB and Hoeffding ODaSC achieve the highest causally tuned $F_{1,\text{th}}$ overall, and Survival-PU leads Track-1 hosts. On post-Platt Brier loss, FixedCurve-PU and Survival-PU tie at four-decimal display, ahead of Full-LR Adapt-4W and Causal-4W.

[[TAB_BENCH_B_CLUSTERS]]

\paragraph{Holm--Bonferroni Contrasts and Codified Component Controls.}
Table~\ref{tab:bench_b_holm} summarizes all three pre-declared hypothesis families on $K=25$. Survival-PU scores $0.4866$ AP, significantly outperforming Pure-nnPU at $0.4461$ ($p_{\text{Holm}} = 0.0020$) and Hoeffding ORB ($p_{\text{Holm}} = 0.0279$). No significant Holm-adjusted gap emerges against Frozen, ODaSC, BORB, or Full-LR Adapt-4W.

However, \textbf{Survival-PU significantly trails Causal-4W by $\Delta\text{AP} = -0.0138$ ($p_{\text{Holm}} = 0.0056$) and Wait-90 ($p_{\text{Holm}} = 0.0069$)} on primary whole-stream AP.

Family~3 tests whether adaptive survival fitting justifies its complexity. On $K=25$ primary MLP AP, \textbf{zero-fit FixedCurve-PU significantly exceeds cohort-estimated Survival-PU at $0.4942$ versus $0.4866$ ($p_{\text{Holm}} = 0.0295$)}. Quoting our codified decision rule verbatim: ``\textit{An incremental benefit over FixedCurve\_PU was not established on this cohort.}'' The same direction holds on LightGBM at $0.4680$ versus $0.4619$ ($p_{\text{Holm}} = 0.0367$) and on the disjoint check.

Against step-matched Teacher-Only on MLP, Survival-PU likewise shows no established gain ($p_{\text{Holm}} = 0.9157$). Quoting our codified rule verbatim: ``\textit{An incremental benefit over Teacher\_Only\_Matched was not established on this cohort.}'' On LightGBM, by contrast, Stage~3 distillation significantly lifts AP over Teacher-Matched by $+3.79$ points ($p_{\text{Holm}} = 9.87 \times 10^{-4}$), with Causal-4W still leading deployable LightGBM AP overall. Legacy gating adds no established benefit.

[[TAB_BENCH_B_HOLM]]

\paragraph{Evaluation Granularity and Cross-Window Score Comparability.}
To test whether Survival-PU's whole-stream deficit reflects weaker within-window ranking or cross-window score-scale drift (Proposition~\ref{prop:bias}), Table~\ref{tab:window_ap_scale} evaluates five-seed predictions across all operational 400-commit windows on $K=25$, with zero windows excluded. Moving from whole-stream AP to unweighted per-window AP reverses the sign against Causal-4W into a modest edge without reaching statistical significance: Survival-PU reaches $0.5246$ versus $0.5192$ for Causal-4W (exploratory $p = 0.0569$), close to FixedCurve-PU, Wait-90, and Full-LR Adapt-4W.

Applying strictly monotone within-window z-score or percentile-rank standardization to the \textit{exact same predictions} before computing whole-stream AP yields the matching reversal. Relative to Causal-4W, standardization flips Survival-PU's deficit into modest leads under Z-Norm and Rank-Norm, reaching $0.5027$ versus $0.4989$ under Rank-Norm ($p = 0.0877$). Gaps against FixedCurve-PU and Wait-90 also shrink near zero under Rank-Norm. Inter-window score scaling therefore explains a major share of the whole-stream gap against Causal-4W, without pushing Survival-PU past Wait-90, FixedCurve-PU, or Full-LR.

[[TAB_WINDOW_AP_SCALE]]

\paragraph{Controlled Concatenated Distribution Shifts.}
Under sharp cross-project or cross-subsystem regime changes, method rankings reverse with horizon length (Table~\ref{tab:bench_b_clusters}, Figure~\ref{fig:rq2_bench_b}(b), and Online Resource~1, Table~S-6). Over the full post-onset horizon, \textbf{Survival-PU achieves the highest deployable mean AP at $0.5764$}, ahead of ODaSC-Host at $0.5727$, balanced Causal-4W, and Wait-30. Single-window causal replay, age-blind Pure-nnPU, and ORB fall below freezing.

Over the immediate first 1,200 post-onset commits, however, ElasticNet BORB-Alarm at $0.5931$ and BORB-Native at $0.5734$ adapt fastest, surpassing Survival-PU at $0.5591$. Online Resource~1 (Table~S-6) also documents stream heterogeneity: OpenStack Nova$\to$Neutron yields lower post-onset AP across all adaptive methods than the three cross-project splices.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig3_benchmark_b_clusters_v3.pdf}
  \caption{(a) Mean AP ($\pm 1$ SEM) across $K=25$ independent natural donor clusters in Benchmark~B. (b) Post-shift AP across the four separated controlled shift streams on both full post-onset and first 1,200 post-onset commit horizons.}
  \label{fig:rq2_bench_b}
\end{figure}

\paragraph{Phase C: Confirmatory Non-Replication on MLP.}
Table~\ref{tab:confirmatory} reports out-of-sample replication on the five untouched ApacheJIT repositories. Because upstream Hadoop monorepo splitting inflates base rates in HDFS* and MapReduce* (Online Resource~1, Table~S-7), the three standalone repositories (Spark, Zeppelin, ZooKeeper) provide the cleaner external check. On those standalone repositories, \textbf{MLP hindsight optimism did not replicate}: Window Oracle scores $0.5951$ AP, trailing Causal-4W at $0.5985$ and Frozen at $0.6565$. Hoeffding ODaSC leads clearly at $0.7857$, followed by ElasticNet BORB and Hoeffding ORB, all far above every window-retrained MLP variant.

[[TAB_CONFIRMATORY]]

\paragraph{Frozen Four-Umbrella Record and Horizon Covariation.}
The frozen four-umbrella MLP AP record confirms the identical direction across all four umbrella ecosystems (Table~\ref{tab:confirmatory} and Figure~\ref{fig:rq2_confirmatory}). Window Oracle reaches $0.6801$ AP, trailing Causal-4W at $0.6858$ (exact two-sided $p = 0.125$) and Frozen at $0.7244$. ODaSC again leads the four-umbrella cohort at $0.8289$, ahead of BORB and ORB.

On LightGBM four-umbrella AP, Window Oracle edges past Causal-4W without statistical significance, and Frozen leads both across the standalone repositories (Table~\ref{tab:confirmatory}). Figure~\ref{fig:rq2_confirmatory}(b) plots evaluation horizon against the matched hindsight gap across all natural Benchmark~B and confirmatory streams. Because short horizon length, higher prevalence, lighter censoring, alarm count, and project identity covary on Phase~C, the pattern favoring ODaSC remains an exploratory hypothesis rather than a validated selection rule.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig4_confirmatory_boundary_v3.pdf}
  \caption{(a) Out-of-sample AP across the five confirmatory ApacheJIT repositories (asterisks mark Hadoop monorepo subprojects). (b) Exploratory association plotting evaluation horizon $N_{\text{eval}}$ against the matched oracle-minus-causal hindsight gap on MLP across 27 natural Benchmark~B streams ($22/27 > 0$) and five confirmatory repositories ($0/5 > 0$).}
  \label{fig:rq2_confirmatory}
\end{figure}

\rqbox{Summary for RQ2}{Frozen Full-LR leads every deployable host on Benchmark~A, and Survival-PU recovers Track-1 $F_1$. On gradual $K=25$ whole-stream MLP AP, Full-LR Adapt-4W leads in mean. Causal-4W, Wait-90, and zero-fit FixedCurve-PU significantly outperform Survival-PU, with within-window standardization reversing the mean Causal-4W gap without reaching significance. On controlled shifts, Survival-PU leads full post-onset AP and BORB-Alarm leads early recovery. On Phase~C, MLP hindsight optimism fails to replicate and ODaSC leads both the standalone repositories and the four-umbrella cohort.}

% ----------------------------------------------------------------------------
% 6.3 RQ3
% ----------------------------------------------------------------------------
\subsection{RQ3: Maturity Sensitivity, Mechanism Audit, and Component Ablations}
\label{sec:rq3}

\paragraph{Low Sensitivity Across Stage~1 Maturity Estimators.}
Table~\ref{tab:rq3_ablations} dissects Stage~1 estimator sensitivity and internal ablations across the ten-stream representative subset (full 7-column table with ROC-AUC in Online Resource~1, Table~S-10). Because seven natural streams have exact $p_{\min} = 0.015625 > 0.0125$, these contrasts are reported descriptively.

Substituting alternative Stage~1 estimators shifts aggregate accuracy only modestly. Default cohort incidence, quantized Kaplan--Meier cure, global Weibull cure, and zero-fit FixedCurve-PU span just $1.09$ AP points and $0.45$ $F_{1,\text{th}}$ points, with zero survival fallbacks and zero unreachable bisections. On this subset, Causal-4W leads descriptive AP and trails Survival-PU on $F_{1,\text{th}}$ and $\text{Brier}_{\text{cal}}$ (Table~\ref{tab:rq3_ablations}).

[[TAB_RQ3_ABLATIONS]]

\paragraph{Unrevealed-Commit Mechanism Audit Against FixedCurve-PU and Writeback.}
Because Eq.~\eqref{eq:bayes_pu} clamps revealed positives to $q_i = 1$, all-pool error conflates label writeback with genuine maturity estimation on unrevealed commits. Online Resource~1 (Table~S-11) therefore audits all 62 seed-42 DriftLens alarms across the ten streams ($59{,}600$ unique commits), comparing Survival-PU against FixedCurve-PU. Reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is evaluated over archive positives per stream, counting unlinked fixes as unrevealed. Two patterns emerge:
\begin{itemize}[leftmargin=1.6em,itemsep=0.05em]
  \item \textit{Non-Monotonic Error Propagation Across Stages.} Adaptive cohort incidence estimates empirical maturity more accurately than the fixed 90-day curve, cutting alarm-level MAE from $0.1204$ to $0.0938$ and stream-level MAE from $0.1310$ to $0.0873$. Better Stage~1 fit does not propagate monotonically downstream: FixedCurve-PU attains slightly lower Stage~2 prevalence MAE ($0.0565$ vs.\ $0.0620$), with both improving over unshrunk $\hat{\pi}_{\text{IP}}$.
  \item \textit{Positive Writeback Versus Unrevealed Soft Targets.} Across the full pool, positive writeback lowers de-weighted teacher Brier from $0.1525$ to $0.1134$, driving most of the all-pool gain. On strictly unrevealed commits ($y_{\text{obs}} = 0$, zero writeback), soft targets still beat de-weighted teachers across every age stratum, lowering Brier from $0.1303$ to $0.1236$ for Survival-PU. FixedCurve-PU reaches slightly lower overall unrevealed Brier at $0.1200$, leading young and mid-age commits, and Survival-PU leads aged commits.
\end{itemize}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig5_ablations_and_controls_v3.pdf}
  \caption{(a) Ten-stream mean AP across three causal maturity estimators, zero-fit FixedCurve-PU, and single-factor internal ablations. (b) Paired AP and Recall@20\%Effort across gate-free Survival-PU, step-matched teacher controls, and legacy three-state gating.}
  \label{fig:rq3_abl}
\end{figure}

\paragraph{Component Ablations and Simplification Candidates.}
Figure~\ref{fig:rq3_abl} and Table~\ref{tab:rq3_ablations} identify which internal components carry weight. Freezing the teacher to the initial checkpoint causes the steepest AP drop (over three points) and triggers $4.2$ unreachable bisections per stream, confounding teacher staleness with solver clamping. Removing monotone bisection or anchor shrinkage degrades both AP and calibration. Conversely, omitting Stage~2 covariate density ratios slightly lifts descriptive AP to $0.5558$ along with $F_{1,\text{th}}$ and effort recall at lower cost, pointing to density-ratio omission as a simplification candidate for future validation. Survival-PU also beats step-matched Teacher-Only here, and legacy gating alters AP negligibly.

\rqbox{Summary for RQ3}{All four Stage-1 estimators span just $1.09$ AP points with zero fallbacks. Across 62 alarms, cohort incidence estimates Stage-1 maturity more accurately than FixedCurve-PU, yet FixedCurve-PU attains slightly lower Stage-2 prior MAE and strictly unrevealed Brier, showing non-monotonic cross-stage error propagation. Teacher adaptation, bisection, and shrinkage carry the main weight; density-ratio omission emerges as a simplification candidate.}

% ----------------------------------------------------------------------------
% 6.4 RQ4
% ----------------------------------------------------------------------------
\subsection{RQ4: Stress Tests Across Latency, Monitoring, Noise, and Prompting}
\label{sec:rq4}

We run five targeted stress tests: future-only latency scaling with post-hoc controls and raw versus Platt calibration (RQ4a), an unconfounded $2 \times 2$ DriftLens factorial (RQ4b), semi-synthetic PR-SZZ noise (RQ4c), matched-horizon window sweeps (RQ4d), and a paired Code-LLM case study (RQ4e).

\paragraph{RQ4a: Latency Scaling and Ranking-Versus-Classification Split.}
Table~\ref{tab:rq4a_kappa} and Figure~\ref{fig:rq4_lat_mon}(a) scale finite post-$t_0$ fix delays from zero to heavy lag while freezing pre-$t_0$ supervision (full mirror with Wait-90 in Online Resource~1, Table~S-12). Frozen $B_0$ stays invariant at $0.4872$ AP, and post-$t_0$ oracle visibility satisfies exact Causal-4W $=$ Window Oracle $= 0.5980$ AP. Across the AP trajectory, Causal-4W stays slightly ahead of Survival-PU across all latency levels, and age-blind Pure-nnPU collapses in effort recall.

On classification under heavy lag ($\kappa = 3.0\times$), however, the ranking-versus-classification split sharpens dramatically. Causal-4W drops to $0.3825$ fixed-threshold $F_1$ and $0.4465$ tuned $F_{1,\text{th}}$. By contrast, \textbf{Survival-PU sustains $F_1 = 0.5259$ and causally tuned $F_{1,\text{th}} = 0.5022$}, also outperforming Wait-90 (Online Resource~1, Table~S-12).

[[TAB_RQ4A_KAPPA]]

\paragraph{Stage-1 Misspecification Protection Against FixedCurve-PU.}
Comparing adaptive cohort incidence against FixedCurve-PU across identical latency regimes (Online Resource~1, Table~S-13) shows when adaptive Stage~1 pays off. Under immediate fixes ($\kappa = 0.00\times$), FixedCurve-PU's 90-day prior treats clean young commits as censored. It therefore trails Survival-PU across all four metrics, losing $2.16$ points in AP ($0.5544$ vs.\ $0.5760$) and four points in $F_1$, with Causal-4W leading overall.

At the opposite extreme of heavy latency, the fixed 90-day curve overestimates maturity. Adaptive Survival-PU consequently beats FixedCurve-PU on fixed $F_1$, tuned $F_{1,\text{th}}$, and post-Platt Brier loss (Online Resource~1, Table~S-13). On pure whole-stream AP across nonzero lag, however, FixedCurve-PU consistently outranks Survival-PU, reaching top deployable heavy-latency AP at $0.5547$ versus Causal-4W's $0.5466$ and Survival-PU's $0.5438$.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig6_latency_and_drift_factorial_v3.pdf}
  \caption{(a) Post-$t_0$ verification-latency scaling ($\kappa \in [0.0, 3.0]$ with frozen pre-$t_0$ supervision, keeping Frozen $B_0$ invariant). (b) Unconfounded $2 \times 2$ DriftLens bootstrap and refresh factorial ($p_{95}$, cooldown 1) plus Pote-inspired reference ($p_{98}$, cooldown 0) and RBF-MMD.}
  \label{fig:rq4_lat_mon}
\end{figure}

\paragraph{Raw Versus 90-Day Platt Calibration and Post-Hoc Controls.}
Auditing the ninety-day calibration pool $\mathcal{M}_w$ confirms heavy residual censoring whenever median fix lag exceeds three months: $55.3\%$ in Benchmark~A, $43.8\%$ in Benchmark~B, and $65.9\%$ at heavy lag. Table~\ref{tab:rq4a_posthoc_b2} evaluates all five methods under raw and shared ninety-day Platt probabilities using identical threshold search (complete 10-column mirror in Online Resource~1, Table~S-14).

Both post-hoc controls leave Causal-4W's weights untouched and update strictly at fired alarms. PostHoc Logit solves a scalar logit shift $\delta_k$ via 28-step bisection on trailing Causal-4W logits to match $\hat{\pi}_{\text{target}}$. PostHoc Ratio removes training class weight via $r_{\text{dewt}}(x) = \sigma(z_{B_2}(x) - \log \omega_{\text{base}})$, divides by trailing-$4W$ mean maturity $\bar{c}_{4W}$, and clips to $[10^{-4}, 1 - 10^{-4}]$. Four concrete findings follow:
\begin{enumerate}[leftmargin=1.6em,itemsep=0.04em]
  \item \textit{Scale Shifts and Upper Clipping at Zero Lag.} PostHoc Logit loses whole-stream AP via alarm-to-alarm shifts without harming WinAP. PostHoc Ratio achieves the lowest raw Brier error, and its slight WinAP drop comes solely from upper-bound clipping ($7.9\%$ of commits across $71.8\%$ of windows); removing that cap restores Causal-4W's exact WinAP.
  \item \textit{Post-Hoc Controls Do Not Match PU Classification.} At heavy latency, PostHoc Logit at $0.3593$ $F_1$ and PostHoc Ratio at $0.3216$ $F_1$ trail Survival-PU's $0.5259$ sharply, with narrow WinAP spread. Because PostHoc Ratio removes training class weights and Survival-PU retains weighted scores, fixed-threshold $F_1$ compares asymmetric scales. On scale-fair tuned $F_{1,\text{th}}$, Survival-PU still leads PostHoc Ratio by $4.79$ points and Causal-4W by $5.57$ points (Table~\ref{tab:rq4a_posthoc_b2}).
  \item \textit{Mechanism Behind Raw Brier Differences.} Across all three latency levels, Survival-PU has higher raw Brier and ECE than Causal-4W, and FixedCurve-PU achieves the best raw calibration at heavy lag. This closes Proposition~\ref{prop:bias}'s remark: Stage~3 overlays $\omega_{+,\text{teach}} > 1$ atop prevalence-calibrated targets $q_i(\delta^*)$, pushing raw probabilities upward. Causal-4W's censoring compression and FixedCurve-PU's negative target bias pull raw scores downward, partially offsetting $\omega_+ > 1$.
  \item \textit{Hazard of 90-Day Platt Scaling Under Heavy Lag.} At heavy latency, raw probabilities beat 90-day Platt scaling on ECE for all five methods and on Brier for four of five, leaving PostHoc Logit nearly flat (Table~\ref{tab:rq4a_posthoc_b2}). Fitting standard Platt scaling to a $65.9\%$-censored pool therefore degrades calibration. Under that shared calibrator, Survival-PU retains the lowest final post-Platt Brier and ECE, and its $F_{1,\text{th}}$ lead holds on both raw and Platted scores.
\end{enumerate}

[[TAB_RQ4A_POSTHOC_B2]]

\paragraph{RQ4b: Unconfounded $2 \times 2$ DriftLens Calibration Factorial.}
Table~\ref{tab:rq4b_monitors} and Figure~\ref{fig:rq4_lat_mon}(b) fix percentile and cooldown at $p_{95}$ and unit cooldown while estimating validation-pool resampled-null alarm rates via block-bootstrap windows. Replacing IID with Circular Block cuts validation-pool null alarm rates by $4.8\times$ under matched refresh rules, dropping from $20.2\%$ to $4.2\%$ under static references and from $16.4\%$ to $3.4\%$ under rolling references.

Across 20 controlled-shift runs, enabling Rolling Ref+Th eliminates missed transitions completely, dropping from $10.0\%$ under Block + Static and $5.0\%$ under IID + Static to $0.0\%$. Conditional detection delays average $310$ commits for IID + Rolling and $350$ commits for Block + Rolling, versus $1{,}050$ commits for periodic retraining.

Fewer alarms do not automatically lift AP: default Block + Rolling fires on $32.1\%$ of windows versus $43.5\%$ for IID + Rolling and trails IID + Rolling slightly in AP (Table~\ref{tab:rq4b_monitors}). Even so, block-calibrated DriftLens stays operationally preferable to periodic schedules because it catches abrupt shifts $3\times$ faster with an order-of-magnitude lower null rate.

[[TAB_RQ4B_MONITORS]]

\paragraph{RQ4c--d: Semi-Synthetic PR-SZZ Noise and Window-Size Sweeps.}
Figure~\ref{fig:rq4_szz_win}(a) and Online Resource~1 (Table~S-15) inject independent conditional flip noise into five PR-SZZ streams using project-averaged B-SZZ error rates \citep{bludau2026quantifying}: clean commits flip to buggy at $\alpha_{\text{FP}} = 8.42\%$, and buggy commits permanently lose fixes at $\beta_{\text{FN}} = 47.15\%$.

Training on noisy labels and evaluating on unperturbed PR-SZZ produces a sharp family split. Batch window and PU learners cluster within $0.0034$ AP, led descriptively by Survival-PU at $0.3802$, Wait-90, and Causal-4W. Stream learners degrade far more steeply, with BORB falling from $0.4224$ to $0.3187$ and ORB dropping further. Generalizing this gap requires manually verified SZZ corpora because independent flips omit feature-dependent noise correlation.

Figure~\ref{fig:rq4_szz_win}(b) sweeps window size from narrow to wide windows on the fixed 1,200-commit post-onset horizon across three representative controlled shifts. Survival-PU leads deployable methods at narrow and medium windows. At the widest window ($W = 800$), Wait-90 at $0.6078$ and Causal-4W at $0.5989$ surpass Survival-PU at $0.5974$ once wide windows span older matured history. On full ten-stream AP across all window sizes, Causal-4W remains ahead of Survival-PU, and sweeping shrinkage $m_0$ or tail cap $\gamma_{\text{tail}}$ confirms low sensitivity.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig7_szz_window_sensitivity_v3.pdf}
  \caption{(a) $2 \times 2$ frozen-prediction semi-synthetic SZZ perturbation across five PR-SZZ streams ($\alpha_{\text{FP}}=8.42\%, \beta_{\text{FN}}=47.15\%$). (b) Matched-horizon fixed 1,200-commit post-onset AP across $W \in \{200, 400, 600, 800\}$ on the three representative controlled shifts.}
  \label{fig:rq4_szz_win}
\end{figure}

\paragraph{RQ4e: Paired Code-LLM Prompting Case Study.}
To test whether parameter-free in-context learning sidesteps weight-update censoring, we audited deepseek-flash across 320-commit Qt and OpenStack slices ($6{,}400$ logical evaluations across $6{,}819$ HTTP attempts; full protocol and Table~S-16 in Online Resource~1). Comparing strictly paired six-shot prompts sharing identical exemplars and differing solely in observable $y_{\text{obs}}(t)$ versus eventual $y^*$ shows that exemplar censoring depresses OpenStack $F_1$ by $10.70$ points ($0.4946$ to $0.3876$) and barely shifts Qt $F_1$.

By contrast, OpenStack AP rises slightly under censored six-shot at $0.4277$ versus $0.4177$ paired oracle, echoing Proposition~\ref{prop:bias}'s probability-shift effect. Because this case study evaluates one lightweight model on public historical commits where zero alarms fired, it shows only that exemplar visibility alters classification on this model; domain-trained JIT predictors lead every prompting arm in AP (Figure~\ref{fig:rq4f_rt}(a)).

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig8_llm_and_runtime_forensics_v3.pdf}
  \caption{(a) Paired prompting case study (deepseek-flash on 320-commit Qt and OpenStack slices) vs.\ domain-trained JIT predictors (initial checkpoints for CodeBERT and LightGBM, 0 alarms on slice). (b) Standalone wall-clock runtime on pre-extracted tabular features across 27 natural Benchmark~B streams.}
  \label{fig:rq4f_rt}
\end{figure}

\rqbox{Summary for RQ4}{At $\kappa=3.0\times$, FixedCurve-PU leads deployable AP and raw calibration, and Survival-PU beats Causal-4W, both post-hoc controls, and FixedCurve-PU on scale-fair $F_{1,\text{th}}$ while also beating FixedCurve-PU across all metrics at $\kappa=0\times$. Fitting 90-day Platt scaling to a heavily censored pool worsens calibration relative to raw scores. Circular block bootstrap cuts null alarms $4.8\times$ and detects shifts $3\times$ faster than periodic schedules. Batch window and PU learners resist independent SZZ noise better than BORB and ORB, and exemplar censoring lowers deepseek-flash OpenStack $F_1$ substantially.}

% ----------------------------------------------------------------------------
% 6.5 RQ5
% ----------------------------------------------------------------------------
\subsection{RQ5: Runtime Telemetry and Equal-Cap Review-Budget Error Analysis}
\label{sec:rq5}

\paragraph{Cached Execution Versus Feature-Encoding Runtime.}
Figure~\ref{fig:rq4f_rt}(b) and Online Resource~1 (Table~S-17) report phase-decomposed runtimes on pre-extracted tabular features across Benchmark~B's natural streams, charging each alarm-triggered method its standalone anchor and monitoring chain. Standalone Survival-PU finishes in $14.43\text{ s}$ per stream (roughly $1.66\text{ s}$ per alarm), matching Teacher-Matched. Online Hoeffding ORB, ElasticNet BORB, and Frozen run fastest under four seconds per stream, whereas ODaSC requires nearly $70\text{ s}$ due to per-event tree updates and DenStream maintenance.

On multimodal Qt, cached Survival-PU runs in $18.19\text{ s}$ on CPU LightGBM and $7.26\text{ s}$ on RTX~4080 GPU CodeBERT. Adding offline GPU CodeBERT batch feature encoding ($37.32\text{ s}$ on Qt and $19.32\text{ s}$ on OpenStack) brings total Qt batch processing to $44.58\text{ s}$, distinct from unbatched webhook latency.

[[TAB_RQ5_PAIRED_BUDGET]]

\paragraph{Paired Whole-Stream Versus Per-Window Equal-Cap Budgets.}
Directional rescue anecdotes cannot establish net review yield, and whole-stream retrospective budgeting answers a different question from sprint-level capacity. Table~\ref{tab:rq5_paired_budget} compares Survival-PU minus Causal-4W across unmatched fixed thresholding, equal-cap whole-stream budgets, and equal-cap 400-commit per-window budgets, paired with five-seed windowed 20\%-LOC recall $\Delta\text{WinR}_{20}$. Under 20\%-LOC budgets, commits rank by risk density $\hat{p}/\text{LOC}$ (ties favoring smaller LOC) and fill greedily without splitting commits, yielding closely matched LOC consumption across all eight hosts. Two operational lessons stand out:
\begin{itemize}[leftmargin=1.6em,itemsep=0.05em]
  \item \textit{Unmatched Fixed-Threshold Volume.} Preventing posterior collapse rescues hundreds of defects on Qt and dozens on OpenStack, at the cost of flagging $+1{,}838$ to $+3{,}211$ extra clean commits on Qt and $+130$ to $+357$ on OpenStack.
  \item \textit{Whole-Stream Versus Per-Window Equal Caps.} Echoing Section~\ref{sec:rq2}, retrospective whole-stream budgeting penalizes Survival-PU on five of seven alarmed hosts, most steeply on Qt DeepJIT-CNN and OpenStack LightGBM. Switching to operational 400-commit per-window budgets reverses or removes nearly every steep whole-stream loss. Qt LightGBM stays strongly positive across commit and LOC caps, both DeepJIT-CNN hosts turn positive on five-seed $\Delta\text{WinR}_{20}$, and Qt Attention-CNN plus OpenStack LightGBM turn near-even. Qt CodeBERT remains negative across both budget granularities. Pre-declared high-confidence SZZ rescue illustrations and FN/FP churn skew appear in Online Resource~1 (Table~S-18).
\end{itemize}

\rqbox{Summary for RQ5}{On cached features, ORB and BORB finish under four seconds per stream, Survival-PU takes $14.43\text{ s}$ ($44.58\text{ s}$ with Qt CodeBERT batch encoding), and ODaSC takes nearly $70\text{ s}$. Under equal caps, retrospective whole-stream budgeting penalizes Survival-PU on five of seven alarmed hosts; operational per-window 400-commit budgeting reverses most steep losses on LightGBM and DeepJIT-CNN while remaining negative on Qt CodeBERT.}

% ============================================================================
% 7. Discussion, Observed Operating Boundaries, and Limitations
% ============================================================================
\section{Discussion, Observed Operating Boundaries, and Limitations}
\label{sec:discussion}

We synthesize our findings around the three framing questions of this study without repeating Section~\ref{sec:results} verbatim.

\subsection{What Label-Visibility Timing Changes}
\label{sec:disc_timing}

Our audit separates two questions often conflated in drift-triggered JIT-SDP \citep{pote2026adaptive}: whether small-window retraining outperforms freezing, and whether future-fix leakage inflates retrained weights. On long multi-year tabular streams, leaking future fixes inflates whole-stream AP significantly on both MLP and LightGBM. In two other regimes, however, even Window Oracle fails to beat Frozen or Causal-4W on MLP. One is low-prevalence multimodal Benchmark~A, where frozen Full-LR leads in line with \citet{zeng2021deep}. The other is the short confirmatory ApacheJIT cohort, where horizon length, higher standalone prevalence, lighter censoring, and project identity shift together. Chronological benchmarks should therefore disclose end-of-archive follow-up horizons (Online Resource~1, Tables~S-8--S-9) alongside upstream repository-key splits such as the clean Hadoop monorepo partition (Table~S-7).

\subsection{What Three-Stage Maturity Correction Adds Over Simpler Baselines}
\label{sec:disc_increment}

Once strict causal visibility holds, our two-track controls define exactly where three-stage Survival-PU adds value and where simpler baselines suffice. First, on gradual $K=25$ streams, Full-LR Adapt-4W, Causal-4W, Wait-90, and FixedCurve-PU outperform Survival-PU on whole-stream AP. Yet Table~\ref{tab:window_ap_scale} shows that within-window standardization reverses the mean deficit against Causal-4W, keeping per-window AP close across Wait-90, FixedCurve-PU, and Full-LR.

Second, on strictly unrevealed commits (Online Resource~1, Table~S-11), adaptive cohort incidence estimates maturity more accurately than FixedCurve-PU and improves soft-target Brier over de-weighted teachers. FixedCurve-PU still achieves slightly lower target-prior MAE and unrevealed Brier, showing non-monotonic error propagation across coupled stages.

Third, under heavy latency (Table~\ref{tab:rq4a_posthoc_b2}), neither post-hoc Causal-4W control matches Survival-PU's threshold-tuned $F_{1,\text{th}}$ recovery, and adaptive Stage~1 simultaneously prevents FixedCurve-PU's zero-latency misspecification. Stage~3's class weight $\omega_{+,\text{teach}} > 1$ explains why Survival-PU trails Causal-4W and FixedCurve-PU on raw Brier, closing Proposition~\ref{prop:bias}'s remark. Because raw scores outperform 90-day Platt calibration under heavy pool censoring, our calibration audit warns against blind 90-day Platt scaling and highlights maturity-aware post-hoc calibration as a key open problem.

\subsection{Operational Budgets, Observed Boundaries, and Limitations}
\label{sec:disc_implications}

Translating ranking differences into code-review impact depends directly on budget granularity (Table~\ref{tab:rq5_paired_budget}). Preventing fixed-threshold recall collapse rescues hundreds of defects only by flagging thousands of clean commits, and retrospective whole-stream budgeting penalizes Survival-PU on five of seven alarmed hosts. Under realistic 400-commit sprint budgets with matched LOC caps, steep losses disappear or flip positive on LightGBM and DeepJIT-CNN, remaining negative on Qt CodeBERT.

Because practitioners cannot observe future latency multipliers or shift horizons beforehand, Table~\ref{tab:decision_matrix} separates directly observed empirical patterns (Part~I) from exploratory hypotheses (Part~II), which are not validated deployment rules. For benchmark designers, three concrete practices follow. Enforce $t_i^{\text{fix}} \le t_{\text{alarm}}$ alongside archive follow-up disclosure. Include linear predictors, zero-fit maturity priors, post-hoc controls, and adapted stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}. Finally, report whole-stream AP alongside per-window AP, scale-normalized AP, and operational per-window equal-cap review yields.

\begin{table}[t]
\centering
\caption{Synthesis separating (Part~I) directly observed empirical patterns across evaluated cohorts from (Part~II) exploratory hypotheses (not validated deployment selection rules).}
\label{tab:decision_matrix}
\fontsize{7.5}{9.0}\selectfont
\renewcommand{\arraystretch}{1.05}
\setlength{\tabcolsep}{2.2pt}
\begin{tabularx}{\textwidth}{L{2.55cm}Y Y}
\toprule
\textbf{Cohort / Factor} & \textbf{Observed Empirical Evidence (Part~I)} & \textbf{Scope Boundary / Hypothesis (Part~II)} \\
\midrule
\multicolumn{3}{l}{\textit{Part I: Directly Observed Empirical Patterns Within Evaluated Benchmarks}} \\
\midrule
Multimodal streams (Bench~A, 13 metrics) & Full-LR ($0.3653$ AP) and LApredict ($0.3601$) lead all deployable hosts (top single host: DeepJIT-CNN Frozen $0.3531$). & Within Track~1, SPU recovers $F_1$ ($0.2907$); per-window equal-cap yield is host-dependent (Table~\ref{tab:rq5_paired_budget}). \\
Gradual multi-year clusters ($K=25$) & Full-LR ($0.5108$), Causal-4W ($0.5004$), Wait-90 ($0.4982$), and FixedCurve-PU ($0.4942 > 0.4866$ SPU) lead whole AP; BORB leads $F_{1\mathrm{t}}$ ($0.5144$). & 5-seed within-window Rank-Norm ($0.5027$ vs.\ $0.4989$) and Z-Norm ($0.4990$ vs.\ $0.4968$) reverse the mean Causal-4W gap; Win-AP gives $0.5246$ vs.\ $0.5192$ ($p=0.0569$). \\
Controlled shifts (4 streams) & SPU leads full post-onset AP ($0.5764$ vs.\ ODaSC-Host $0.5727$); BORB-Alarm leads first 1,200c ($0.5931$ vs.\ SPU $0.5591$). & 1-hour splices ($23.7\%$--$60.0\%$ cross-boundary fixes) differ from natural drift; at $W=800$, Wait-90 ($0.6078$) tops SPU ($0.5974$). \\
Latency extremes ($\kappa=0\times, 3\times$) & At $0\times$, Causal-4W ($0.5974$) $>$ SPU ($0.5760$) $>$ Fix ($0.5544$). At $3\times$, SPU leads $F_{1\mathrm{t}}$ ($0.5022$) and $F_1$ ($0.5259$); Fix leads AP ($0.5547$) and raw Br./ECE. & Raw scores beat 90d Platt under $65.9\%$ censoring; feature-dependent post-hoc calibration remains untested, and density-ratio omission is an unvalidated simplification candidate not tested under extreme $\kappa$. \\
Semi-synthetic SZZ noise (5 streams) & Under $\alpha_{\text{FP}}=8.42\%, \beta_{\text{FN}}=47.15\%$, batch learners ($0.3768$--$0.3802$) exceed BORB ($0.3187$) and ORB ($0.2782$). & Scoped to independent synthetic flips over PR-SZZ; feature- or age-dependent real SZZ noise remains untested. \\
\midrule
\multicolumn{3}{l}{\textit{Part II: Exploratory Hypotheses (Not Validated Selection Rules)}} \\
\midrule
Confirmatory repos ($N < 3{,}000$) & Standalone 3: ODaSC $0.7857$; Oracle $0.5951 <$ Causal-4W $0.5985 <$ Frozen $0.6565$. Umbrella-4: ODaSC $0.8289$; Oracle $0.6801 <$ Causal-4W $0.6858$. & Unvalidated hypothesis: Hadoop splits omit 11,964 clean rows; horizon, prevalence, censoring, and project identity remain confounded. \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Scope Limitations.}
Six limitations bound our conclusions. First, Benchmark~A orders merged Git revisions by \texttt{author\_date} rather than patch-set-1 uploads, and $T_{\text{end}}$ is an observable proxy rather than a certified collection cutoff. Second, Stage~1 assumes local cohort stationarity, Stage~2 uses heuristic shrinkage, and the upper prior clamp ($\pi_{\max}=0.70$) causes unreachable bisection when observed prevalence exceeds $0.70$. Third, 90-day Platt calibration retains heavy censoring under multi-month lag and degrades probability estimates at high latency.

Fourth, RQ4b's null alarm rate resamples validation pools rather than independent stationary streams. Fifth, controlled shifts impose a one-hour splice across which substantial pre-onset fixes arrive after onset. Sixth, RQ4e examines one lightweight model on 640 public commits predating LLM pretraining cutoffs.

% ============================================================================
% 8. Threats to Validity
% ============================================================================
\section{Threats to Validity}
\label{sec:threats}

\paragraph{Construct Validity.}
SZZ back-tracing suffers from tangled commits, refactorings, and unlinked fixes \citep{dacosta2017framework,rosa2021evaluating,fan2021impact}, and archival snapshots incur end-of-archive censoring when $t_i^{\text{fix}} > T_{\text{end}}$. We mitigate these risks by quarantining heuristic and retroactive links to $+\infty$ and auditing follow-up across all 34 natural streams alongside Benchmark~A restricted-horizon cutoffs (Online Resource~1, Tables~S-8--S-9). We also disclose the clean Hadoop monorepo split (Table~S-7), pair whole-stream AP with per-window and scale-normalized AP, report LOC-Only overlap, and verify paired equal-cap budget net TP yields (Table~\ref{tab:rq5_paired_budget}).

\paragraph{Internal Validity.}
All 41 execution branches passed deterministic prefix-invariance tests with zero perturbation error. Prefix invariance certifies leak-free pipeline code rather than commit-time provenance of upstream Git exports. We separated development analyses on $K=25$ from the pre-unlock confirmatory cohort, disclosed the post-unlock sizing amendment for small repositories, and treated Window Oracle strictly as a Pote-inspired counterfactual rather than a verbatim repository reproduction. Automated verifiers check all $839$ Main cells and $1{,}054$ Online Resource~1 cells ($1{,}893$ verified cells across $27$ empirical tables), all $33$ locked parameters, exact-path prose claims, and relational overclaim guards (Appendix~\ref{app:repro}).

\paragraph{External and Conclusion Validity.}
Our adapted Track-2 baselines (ElasticNet BORB, Hoeffding ORB, and Hoeffding ODaSC) represent specific base learners rather than entire algorithmic families, and independent synthetic SZZ flips omit feature-dependent real SZZ noise. Formal $\alpha=0.05$ Holm--Bonferroni inference applies strictly to $K=25$ and $K=24$ natural donor clusters, distinct from exploratory $p$-values in Table~\ref{tab:window_ap_scale}. Failed MLP directional replication on Phase~C is reported directly without post hoc retuning.

% ============================================================================
% 9. Conclusion
% ============================================================================
\section{Conclusion}
\label{sec:conclusion}

Unsupervised drift monitors remove label dependence when deciding \textit{when} to retrain, not when \textit{supervising} model updates under multi-month SZZ verification latency. Across multi-year ecosystems, 68.8\%--94.6\% of scheduled-window defects remain unrevealed at update time. Leaking future fixes inflates whole-stream AP significantly across multi-year donor clusters on both MLP ($+0.0237$) and LightGBM ($+0.0476$). Circular block-bootstrap calibration cuts validation-pool null alarms $4.8\times$ and catches abrupt shifts $3\times$ faster than periodic schedules.

No single adaptive paradigm dominates every regime. Frozen linear models lead deployable Benchmark~A. On gradual $K=25$ whole-stream AP, Causal-4W, Wait-90, and zero-fit FixedCurve-PU significantly outperform Survival-PU. Within-window standardization reverses that mean Causal-4W gap without reaching statistical significance. ElasticNet BORB leads $K=25$ $F_{1,\text{th}}$ and early post-shift recovery, and Hoeffding ODaSC leads the short confirmatory cohort where MLP hindsight optimism fails to replicate.

Within these operating boundaries, Survival-PU delivers targeted gains on full post-shift AP ($0.5764$), LightGBM soft-target distillation, and classification protection at both latency extremes. It avoids FixedCurve-PU's zero-latency misspecification and beats Causal-4W, two post-hoc controls, and FixedCurve-PU on scale-fair $F_{1,\text{th}}$ under heavy latency. Finally, our calibration audit warns that standard 90-day Platt scaling degrades probability estimates under heavy pool censoring. Developing maturity-aware post-hoc calibration and prospectively validating observable model-selection rules remain central priorities for adaptive JIT-SDP.

\backmatter

% ============================================================================
% Declarations (Springer Nature / Software Quality Journal Standard)
% ============================================================================
\section*{Declarations}
\begin{itemize}[leftmargin=1.5em,itemsep=0.10em]
  \item \textbf{Funding:} This research received no external funding.
  \item \textbf{Competing interests:} The authors declare no competing financial or non-financial interests relevant to this article.
  \item \textbf{Ethics approval and consent to participate:} Not applicable; this study analyzes public open-source repository histories without human-subject experimentation.
  \item \textbf{Consent for publication:} Not applicable.
  \item \textbf{Data availability:} The complete replication package---including immutable \texttt{src\_v3/} modules, \texttt{scripts\_v3/} companion audits and verifiers, pinned \texttt{requirements\_\allowbreak lock\_\allowbreak v3.txt}, SHA-256 manifests for all 94 stream bundles, 50 commit-level \texttt{.npz} prediction archives, lossless LLM exchange logs, and Online Resource~1 (\texttt{Online\_Resource\_1.pdf})---is publicly available at \url{https://github.com/789748456v/CausalJIT-Audit-SQJ} with a full offline archival bundle prepared for permanent Zenodo deposit upon acceptance.
  \item \textbf{Author contributions:} Z.L.: Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Writing -- Original Draft. S.X.: Software, Validation, Data Curation, Writing -- Review \& Editing. K.L.: Validation, Investigation, Visualization, Writing -- Review \& Editing. S.M.: Formal Analysis, Investigation, Writing -- Review \& Editing. F.G.: Validation, Visualization, Writing -- Review \& Editing. Y.Z.: Supervision, Conceptualization, Methodology, Project Administration, Writing -- Review \& Editing. All authors read and approved the final manuscript.
\end{itemize}

% ============================================================================
% Compact Appendix A (1 Table at 7.5pt + Proofs + Traceability)
% ============================================================================
\begin{appendices}
\renewcommand{\theHtable}{\thesection.\arabic{table}}

\section{Locked Parameters, Algebraic Proofs, and Code Traceability}
\label{app:repro}

Table~\ref{tab:hyperparams} lists all 33 locked protocol parameters frozen in \texttt{src\_v3/\allowbreak config\_\allowbreak v3.py} before unlocking the confirmatory cohort. Complete host architecture tables, executed competitor specifications, full-column mirrors of Main tables, and secondary data-funnel, follow-up, mechanism, shift, prompting, runtime, and forensic audits appear in companion \textbf{Online Resource~1} (Tables~S-1--S-18).

\begin{table}[t]
\centering
\caption{Locked protocol parameters of Survival-PU and DriftLens frozen in \texttt{V3\_CONFIG} prior to unlocking the confirmatory repositories.}
\label{tab:hyperparams}
\fontsize{7.5}{9.0}\selectfont
\renewcommand{\arraystretch}{1.03}
\setlength{\tabcolsep}{1.2pt}
\begin{tabularx}{\textwidth}{lllL{3.70cm}Y}
\toprule
\textbf{Mod.} & \textbf{Sym.} & \textbf{Default} & \textbf{Config Field} & \textbf{Role} \\
\midrule
Stg 1 & $N_{\text{h}}, q_{\max}$ & $6{,}000, 0.85$ & \texttt{max\_\allowbreak history\_\allowbreak cohort}, \texttt{support\_\allowbreak age\_\allowbreak quantile} & Slice, $a_{\max}$ \\
Stg 1 & $n_{\min}^{\text{r}}, n_{\min}^{\text{f}}$ & $120, 15$ & \texttt{min\_\allowbreak risk\_\allowbreak set}, \texttt{min\_\allowbreak observed\_\allowbreak fixes} & Fallback floors \\
Stg 1 & $[c_{\min}, c_{\max}]$ & $[0.05, 0.96]$ & \texttt{c\_min}, \texttt{c\_max} & Maturity cap \\
Stg 1 & $[\gamma_{\min}, \gamma_{\max}]$ & $[0.60, 0.94]$ & \texttt{tail\_\allowbreak cap\_\allowbreak min}, \texttt{tail\_\allowbreak cap\_\allowbreak max} & Tail bounds \\
Stg 2 & $\tau_{\text{r}}, N_{\text{r}}$ & $0.45, 2{,}200$ & \texttt{reliable\_\allowbreak maturity\_\allowbreak thresh}, \texttt{max\_\allowbreak reliable\_\allowbreak commits} & Reliable set \\
Stg 2 & $[w_{\min}, w_{\max}]$ & $[0.50, 2.50]$ & \texttt{imp\_\allowbreak weight\_\allowbreak min}, \texttt{imp\_\allowbreak weight\_\allowbreak max} & Ratio cap \\
Stg 2 & $c_{\text{ht}}, m_0, \pi$ & $0.10, 50, [0.02, 0.70]$ & \texttt{ht\_\allowbreak floor\_\allowbreak c}, \texttt{ht\_\allowbreak shrinkage\_\allowbreak m0}, \texttt{pi\_min}, \texttt{pi\_max} & Prior clamp \\
Stg 2 & $\tau_{\text{s}}, \beta_{\text{L}}$ & $1.15, 0.25$ & \texttt{teacher\_\allowbreak sharpness}, \texttt{llm\_\allowbreak blend\_\allowbreak weight} & Scale, blend \\
Stg 3 & $N_{\text{a}}, \gamma, u_0$ & $2{,}000, 1.40, 0.45$ & \texttt{max\_\allowbreak anchor\_\allowbreak commits}, \texttt{drift\_\allowbreak emphasis\_\allowbreak gamma}, \texttt{maturity\_\allowbreak weight\_\allowbreak floor} & Pool, $u_i$ \\
Stg 3 & $p_{\text{r}}, K_{\text{b}}$ & $[0.003, 0.997], 28$ & \texttt{posterior\_\allowbreak p\_\allowbreak min}, \texttt{posterior\_\allowbreak p\_\allowbreak max}, \texttt{bisection\_\allowbreak iters} & Bisection \\
Budg. & $R_{\text{i}}, R_{\text{a}}$ & $90, 75$ & \texttt{lgb\_\allowbreak init\_\allowbreak rounds}, \texttt{lgb\_\allowbreak alarm\_\allowbreak rounds} & GBDT trees \\
Budg. & $E_{\text{t}}, E_{\text{s}}, \eta$ & $4, 4, 4.5\text{e-}4$ & \texttt{neural\_\allowbreak teacher\_\allowbreak epochs}, \texttt{neural\_\allowbreak student\_\allowbreak epochs}, \texttt{neural\_\allowbreak adapt\_\allowbreak lr} & Epochs, lr \\
Mon. & $W, d, p, \text{cd}$ & $400, 24, 95.0, 1$ & \texttt{window\_\allowbreak size\_\allowbreak default}, \texttt{drift\_\allowbreak pca\_\allowbreak dim}, \texttt{drift\_\allowbreak block\_\allowbreak pct}, \texttt{drift\_\allowbreak cooldown\_\allowbreak windows} & DriftLens \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Proof of Proposition~\ref{prop:bias}.}
Under $y_{\text{obs}} \mid x \sim \operatorname{Bernoulli}(\bar{c}(x)\eta^*(x))$, setting $\partial \mathbb{E}[\ell \mid x]/\partial p = 0$ yields $\omega_+ \bar{c}(x)\eta^*(x)/p = (1 - \bar{c}(x)\eta^*(x))/(1 - p)$, which rearranges directly to Eq.~\eqref{eq:b2_bayes}. Differentiating Eq.~\eqref{eq:b2_bayes} with respect to $\eta^*$ for constant $\bar{c}_W$ confirms Claim~1's monotonicity, and solving $\hat{p}_{B_2}(x) < \tau_{\text{th}}$ gives Eq.~\eqref{eq:exact_crossing}. Comparing the monotone odds $\hat{p}_{B_2}/(1 - \hat{p}_{B_2})$ at $x_{\text{new}}$ and $x_{\text{old}}$ establishes Claim~2. \qed

\paragraph{Proof of Proposition~\ref{prop:autocorr}.}
Expanding the contiguous sample-mean covariance gives $\operatorname{Cov}(\bar{z}_{\text{contig}}) = \frac{1}{W}\Gamma_0 + \frac{1}{W^2}\sum_{k=1}^{W-1}(W-k)(\Gamma_k + \Gamma_k^\top) = \operatorname{Cov}(\bar{z}_{\text{ind}}) + \frac{1}{W}\Delta_\Gamma$, since $\operatorname{Cov}(\bar{z}_{\text{ind}}) = \frac{1}{W}\Gamma_0$. Positive definiteness of $\Delta_\Gamma$ directly implies $\operatorname{tr}(\Delta_\Gamma)/W > 0$. \qed

\begin{proposition}[Analytical Margin Derivatives for Soft-Target GBDT]
\label{prop:lgbm}
For $\mathcal{L}_{\text{SPU},i}(z_i) = u_i[\omega_+ q_i \log(1 + e^{-z_i}) + (1 - q_i)\log(1 + e^{z_i})]$ with $q_i \in [0, 1]$, $u_i > 0$, and margin $z_i = z_{\text{init}} + \sum_{m=1}^M T_m(x_i)$, the exact derivatives are $g_i = u_i[(1 + (\omega_+ - 1)q_i)\sigma(z_i) - \omega_+ q_i]$ and $h_i = u_i(1 + (\omega_+ - 1)q_i)\sigma(z_i)(1 - \sigma(z_i)) > 0$. Passing $z_{\text{init}}$ as the training initial margin and adding $z_{\text{init}}$ to raw tree sums at inference preserves exact split-gain margins without dataset duplication.
\end{proposition}

\begin{proof}
Differentiating $\mathcal{L}_{\text{SPU},i}(z_i)$ via $\frac{d}{dz}\log(1+e^{-z}) = \sigma(z)-1$ and $\frac{d}{dz}\log(1+e^z) = \sigma(z)$ yields $g_i$ and $h_i = \partial g_i/\partial z_i$. Adding $z_{\text{init}}$ at inference restores $z_i$ (matching native hard-label LightGBM within $1.19 \times 10^{-7}$ max float32 margin error in our pre-unlock gate).
\end{proof}

\paragraph{Code-to-Manuscript Traceability.}
\begingroup\raggedright
Risk-eligible cohort incidence (\texttt{Cohort\allowbreak Incidence\allowbreak Estimator} and fallback \texttt{\_set\_fallback()}), clipped prior shrinkage (\texttt{estimate\_\allowbreak ht\_\allowbreak shrunk\_\allowbreak prevalence}), and Bayesian soft targets (\texttt{compute\_\allowbreak survival\_\allowbreak pu\_\allowbreak posterior}) live in \texttt{src\_v3/\allowbreak survival\_\allowbreak estimators\_\allowbreak v3.py}. Host architectures (\texttt{train\_\allowbreak supervised\_\allowbreak v3}, \texttt{LightGBM\allowbreak HostV3}) live in \texttt{src\_v3/\allowbreak models\_\allowbreak and\_\allowbreak baselines\_\allowbreak v3.py}; Track-2 competitors and three-branch causal calibration (\texttt{calibrate\_\allowbreak causal\_\allowbreak stream\_\allowbreak probs\_\allowbreak strict}) live in \texttt{src\_v3/\allowbreak stream\_\allowbreak competitors\_\allowbreak v3.py}; stream sizing (\texttt{compute\_\allowbreak split\_\allowbreak sizes}), codified rules (\texttt{PRE\_\allowbreak REGISTERED\_\allowbreak INTERPRETATION\_\allowbreak BOUNDARIES}), and the ten-stream subset (\texttt{REPRESENTATIVE\_\allowbreak 10\_\allowbreak STREAMS}) live in \texttt{config\_\allowbreak v3.py}, \texttt{data\_\allowbreak audit\_\allowbreak v3.py}, and \texttt{run\_\allowbreak v3\_\allowbreak experiments.py}. Historical JSON names \texttt{rq4d\_\allowbreak szz\_\allowbreak perturbation\_\allowbreak v3.json}, \texttt{rq4e\_\allowbreak sensitivity\_\allowbreak sweeps\_\allowbreak v3.json}, and \texttt{rq4f\_\allowbreak llm\_\allowbreak audit\_\allowbreak v3.json} map to RQ4c, RQ4d, and RQ4e. All 15 provenance-stamped JSONs and 839 Main cells verify via \texttt{src\_v3/\allowbreak verify\_\allowbreak manuscript\_\allowbreak v3.py}; all 1,054 Online Resource~1 cells verify via \texttt{scripts\_v3/\allowbreak verify\_\allowbreak online\_\allowbreak resource\_\allowbreak v3.py}; all 33 config fields and exact-path prose claims verify via \texttt{scripts\_v3/\allowbreak verify\_\allowbreak prose\_\allowbreak claims\_\allowbreak v3.py}; and the sealed pre-unlock gate (\texttt{766e3fd6...}) plus live Checks~2--14 verify via \texttt{scripts\_v3/\allowbreak run\_\allowbreak post\_\allowbreak campaign\_\allowbreak mechanism\_\allowbreak checks\_\allowbreak v3.py}.
\endgroup

\end{appendices}

\bibliography{references}

\end{document}
"""
    replacements = {
        "[[TAB_AUDIT_CENSORING]]": tmap["tab:audit_censoring"],
        "[[TAB_BENCH_A_MAIN]]": tmap["tab:bench_a_main"],
        "[[TAB_BENCH_B_CLUSTERS]]": tmap["tab:bench_b_clusters"],
        "[[TAB_BENCH_B_HOLM]]": tmap["tab:bench_b_holm"],
        "[[TAB_WINDOW_AP_SCALE]]": tmap["tab:window_ap_scale"],
        "[[TAB_CONFIRMATORY]]": tmap["tab:confirmatory"],
        "[[TAB_RQ3_ABLATIONS]]": tmap["tab:rq3_ablations"],
        "[[TAB_RQ4A_KAPPA]]": tmap["tab:rq4a_kappa"],
        "[[TAB_RQ4A_POSTHOC_B2]]": tmap["tab:rq4a_posthoc_b2"],
        "[[TAB_RQ4B_MONITORS]]": tmap["tab:rq4b_monitors"],
        "[[TAB_RQ5_PAIRED_BUDGET]]": tmap["tab:rq5_paired_budget"],
    }
    for k, v in replacements.items():
        assert k in tex, f"Placeholder {k} missing in main.tex template!"
        tex = tex.replace(k, v)
    return tex


def build_online_resource_tex(supp_tmap: Dict[str, str]) -> str:
    tex = r"""% Springer Software Quality Journal — Online Resource 1 (Supplementary Material)
% Compilation: pdflatex online_resource && bibtex online_resource && pdflatex online_resource && pdflatex online_resource
\RequirePackage{fix-cm}
\documentclass[10pt,a4paper]{article}

\usepackage[margin=1.8cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xurl}
\usepackage{microtype}
\usepackage[authoryear,round]{natbib}
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black]{hyperref}

\setlength{\emergencystretch}{2.0em}
\renewcommand{\arraystretch}{1.08}
\raggedbottom
\renewcommand{\thetable}{S-\arabic{table}}
\renewcommand{\thesection}{S-\arabic{section}}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}

\title{\Large\textbf{Online Resource 1: Supplementary Tables, Full-Column Mirrors, and Companion Audit Details}\\[0.5em]
\large Accompanying ``Detecting Drift Is Not Safe Retraining: Causal Supervision Boundaries for Adaptive Just-in-Time Defect Prediction''\\[0.3em]
\normalsize\textit{Software Quality Journal} (Springer Nature)}

\author{Zhijie Li$^1$, Shaoyang Xu$^1$, Ketong Liu$^1$, Shibo Meng$^1$, Feiran Gao$^2$, and Yuli Zhao$^{1,*}$\\[0.3em]
\small $^1$Software College, Northeastern University, Shenyang 110169, China\\
\small $^2$School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China\\
\small $^*$Corresponding author: \texttt{zhaoyl@swc.neu.edu.cn}}
\date{}

\begin{document}
\maketitle

\noindent This companion document provides all 18 supplementary tables (Tables~S-1--S-18) referenced in the main manuscript. Every empirical cell across Tables~S-3--S-18 (1,054 cells across 16 supplementary tables) is verified programmatically against the 15 provenance-stamped JSON artifacts in \texttt{results\_v3/} via \texttt{scripts\_v3/\allowbreak verify\_\allowbreak online\_\allowbreak resource\_\allowbreak v3.py}.

% ============================================================================
% S1. Conceptual Positioning & Executed Specifications
% ============================================================================
\section{Conceptual Positioning and Executed Host/Competitor Specifications}
\label{sec:or_specs}

Table~\ref{tab:positioning} contrasts prior JIT-SDP, drift monitoring, verification-latency stream learning, PU learning, and SZZ auditing paradigms along update triggers, alarm-time supervision, and observed operating boundaries. Table~\ref{tab:backbone_specs} details the exact neural and gradient-boosted host constructors executed in \texttt{src\_v3/models\_and\_baselines\_v3.py}.

\begin{table}[!htbp]
\centering
\caption{Conceptual positioning relative to static JIT-SDP, unsupervised drift retraining, passive shift monitoring, verification-latency stream learners, canonical PU learning, and SZZ evaluation-bias auditing.}
\label{tab:positioning}
\fontsize{7.5}{9.0}\selectfont
\renewcommand{\arraystretch}{1.05}
\setlength{\tabcolsep}{2.4pt}
\begin{tabularx}{\textwidth}{L{3.1cm}L{2.1cm}Y L{2.8cm}L{3.6cm}}
\toprule
\textbf{Paradigm / Study} & \textbf{Update Trigger} & \textbf{Supervision at $t_{\text{alarm}}$ / Step $t$} & \textbf{Unmatured $W_{\text{drift}}$} & \textbf{Empirical Regime / Boundary} \\
\midrule
Static JIT-SDP \citep{kamei2013large,hoang2019deepjit,feng2020codebert,zeng2021deep} (Frozen, Linear) & None (frozen) & Causal pre-stream prefix ($t \le t_0$) & Not used & Full-LR leads deployable Benchmark~A; stale after shift \\
Pote-inspired oracle \citep{pote2026adaptive} (Window Oracle) & IID DriftLens & Hindsight $y^*$ ($t_{\text{fix}} > t_{\text{alarm}}$) & Non-causal oracle & Non-deployable counterfactual reference \\
Pote-inspired causal \citep{pote2026adaptive} (Causal-1W/4W) & DriftLens & Observable $y_{\text{obs}}(t_{\text{alarm}})$ & Treated as $y=0$ & Competitive gradual AP ($4W$); $F_1$ drops under $3\times$ lag \\
\citet{shah2026detecting} & Shift metrics & Passive monitoring only & Not applicable & Evaluates alert correlation without weight updates \\
Delayed Window (Wait-90) & Block DriftLens & Revealed fixes $\vee$ age $a_i \ge 90\text{d}$ & $<90\text{d}$ unfixed dropped & Reduces $<90\text{d}$ FN noise; lags $W=200$ shifts \\
\citet{cabral2019class} (ORB) & Per-event OHT & $90\text{d}$ clean + fast fixes + OBF Bagging & Excluded until $90\text{d}$ or fix & Lightweight CPU Hoeffding ensemble; sensitive to SZZ noise \\
\citet{cabral2023borb} (BORB) & $200$-c / alarm & $90\text{d}$ clean + fixes + ElasticNet SGD & Excluded until $90\text{d}$ or fix & Best $K=25$ $F_{1,\text{th}}$ \& first-1,200 post-shift AP \\
\citet{song2022odasc} (ODaSC) & Per-event OHT & $15\text{d}$ provisional $0$ + DenStream $\alpha_i$ & Provisional $0$ after $15\text{d}$ & Highest confirmatory AP; higher DenStream CPU cost \\
\citet{chapelle2014modeling} (Delayed FB) & Online updates & Joint $p(x)$ + exp.\ $\lambda(x)$ delay MLE & Soft delayed-event likelihood & Display-ad delayed feedback; inspired global Weibull cure \\
\citet{kiryo2017positive} (Pure-nnPU) & Block DriftLens & Disjoint $P$ vs.\ marginal $U$ under flat $\hat{\pi}$ & Age-blind unlabeled & Trails on $K=25$ AP \& $\kappa=3\times$ effort recall \\
\citet{song2023waiting,bludau2026quantifying} & Offline audit & Eval waiting-time noise / PR-SZZ vs.\ B-SZZ & Static/eval focus & Audits retrospective eval waiting noise \& SZZ FP/FN rates \\
\midrule
\textbf{Survival-PU} & \textbf{Block DriftLens} & \textbf{Cohort $c(a_i)$ + IP $\hat{\pi}_{\text{target}}$ + PU targets} & \textbf{Age-weighted $q_i(\delta^*)$} & \textbf{Best full post-shift AP \& $3\times$-lag $F_1$/$F_{1,\text{th}}$; trails FixedCurve-PU on gradual AP} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[!htbp]
\centering
\caption{Executed architectural, DriftLens monitoring-space, and optimization specifications across all five host predictors (\texttt{src\_v3/models\_and\_baselines\_v3.py}).}
\label{tab:backbone_specs}
\fontsize{7.5}{9.0}\selectfont
\renewcommand{\arraystretch}{1.05}
\setlength{\tabcolsep}{2.2pt}
\begin{tabularx}{\textwidth}{L{2.3cm}L{2.7cm}Y L{2.8cm}L{3.7cm}}
\toprule
\textbf{Host Model} & \textbf{Causal Input Features} & \textbf{Architecture \& Linear-Residual Skip} & \textbf{DriftLens Space ($24$-D PCA)} & \textbf{Executed Init \& Per-Alarm Budgets} \\
\midrule
Attention-CNN \citep{pote2026adaptive} & $64$ msg + $128$ diff tokens + $13$ authoring metrics & $64$-D dual emb.; $k\in\{1,2,3\}$ 1D-CNN ($48$ filters $\to 144$-D); $6$-head self/cross-attn; $192\to 64$-D ($\text{drop}=0.30$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-ReLU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=7{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ (Oracle, Causal, Wait, ODaSC-H), $6$ ep.\ (nnPU), $4{+}4$ ep.\ (SPU/Fixed), or step-matched cap \\
CodeBERT \citep{feng2020codebert} & $768$-D msg + $768$-D diff (codebert-base) + $13$ metrics & Dual $128$-D LayerNorm-GELU adapters + $4$-head bimodal self-attn + $|a_m-a_c| \oplus (a_m\odot a_c) \to 160 \to 64$-D ($\text{drop}=0.25$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=8{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ (Oracle, Causal, Wait, ODaSC-H), $6$ ep.\ (nnPU), $4{+}4$ ep.\ (SPU/Fixed), or step-matched cap \\
DeepJIT-CNN \citep{hoang2019deepjit} & $64$ msg + $128$ diff tokens + $13$ metrics & $64$-D dual emb.; parallel $k\in\{1,2,3\}$ 1D-CNNs ($64$ filters) + $48$-D metric MLP $\to 64$-D ($\text{drop}=0.25$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=7{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ (Oracle, Causal, Wait, ODaSC-H), $6$ ep.\ (nnPU), $4{+}4$ ep.\ (SPU/Fixed), or step-matched cap \\
Tabular MLP (Bench B/C) & Standardized commit-time / PR-SZZ metrics ($D$-dim) & LayerNorm-GELU ($D\to 128$) + residual block ($128\to 128$) $\to 64$-D head ($\text{drop}=0.20$) & $64$-D penultimate LayerNorm-GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=8{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ (Oracle, Causal, Wait, ODaSC-H), $6$ ep.\ (nnPU), $4{+}4$ ep.\ (SPU/Fixed), or step-matched cap \\
LightGBM \citep{ke2017lightgbm} & $112$-D msg+diff SVD + $13$ metrics (or $D$ tab.) & Callable-objective GBDT, $\text{num\_leaves}=15$, $\text{lr}=0.05$, subsample $0.85$, colsample $0.85$, $\ell_1=0.1, \ell_2=1.0$ & $[x_{i,1:\min(D,48)}, z_i]$ with exact initial margin $z_i$ & Init: $90$ rounds; Alarm: fresh $75$-tree fit (Oracle, Causal, Wait, ODaSC-H, Teach-Exact), fresh $75{+}75$ fits (SPU/Fixed), fresh $150$-tree fit (Teach-Matched); 2 threads \\
\bottomrule
\end{tabularx}
\end{table}

\paragraph{Executed Competitor Specifications (\texttt{src\_v3/stream\_competitors\_v3.py}).}
All neural models optimize with AdamW (weight decay $10^{-4}$, batch size $128$, gradient-norm clip $1.0$, deterministic CuDNN). Token vocabularies ($8{,}000$ tokens), TF-IDF/SVD projections ($48$-D msg + $64$-D diff), and $z$-score scalers fit strictly on $[0, n_{\text{init}})$; initial host weights and frozen tabular skips train on $[0, n_{\text{init}}+n_{\text{val}})$ using $y_{\text{obs}}(t_{\text{start}})$.
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textbf{ORB (Hoeffding)} \citep{cabral2019class}: Ensemble of 10 \texttt{river.tree.HoeffdingTreeClassifier} learners ($\text{grace\_period}=200, \delta=10^{-7}, \tau=0.05$, \texttt{leaf\_prediction="nba"}) trained via OzaBagging \citep{oza2005online} on strict $t_{\text{event}} < t_{\text{commit}}$ events ($\text{waiting\_days}=90.0$ clean maturity plus revealed fixes), with duplicate clean-count noise filtering ($\text{count}>5$ skipped), EMA decay $0.99$, window $100$, $th=0.40, l_0=10, l_1=12, m=1.5$.
  \item \textbf{BORB (ElasticNet) \& BORB-Alarm} \citep{cabral2023borb}: Single \texttt{SGDClassifier} (\texttt{loss="log\_loss"}, \texttt{penalty="elasticnet"}, $\alpha=0.02$, \texttt{l1\_ratio}$=0.15$, \texttt{learning\_rate="optimal"}) refreshed every 200 commits (BORB-Native) or strictly post-window upon DriftLens alarms (BORB-Alarm), running 20 inner skewed-resampling \texttt{partial\_fit} steps ($\text{max\_sample\_size}=2{,}000, th=0.40, l_0=10, l_1=12, m=1.5$) over revealed positives and unfixed commits aged $\ge 90.0$ days.
  \item \textbf{ODaSC (Hoeffding)} \citep{song2022odasc}: Ensemble of 10 \texttt{river} Hoeffding trees ($\text{grace\_period}=200, \delta=10^{-7}, \tau=0.05$) consuming $t_{\text{event}} < t_{\text{commit}}$ events with $\text{waiting\_days}=15.0$ provisional negatives and \texttt{SongDenStreamConfidenceTracker} one-sided micro-cluster confidences ($\delta_{\text{thresh}}=0.85$, radius initialized from 15th pairwise distance percentile on $[0, \min(s_0, 250))$).
  \item \textbf{Pure-nnPU} \citep{kiryo2017positive}: Disjoint historical confirmed positives $P$ and $4W$ unlabeled marginal pool $U$ trained with non-negative PU risk ($\beta=0.0, \gamma=1.0$; 6 neural epochs or unit pseudo-Hessian GBDT steps) using clipped inverse-maturity shrunk prior $\hat{\pi}_{\text{target}}$.
\end{itemize}

\clearpage
% ============================================================================
% S2. Complete Multi-Host Benchmark A, Benchmark B, & Controlled-Shift Mirrors
% ============================================================================
\section{Complete Benchmark A, Benchmark B, and Controlled-Shift Mirrors}
\label{sec:or_rq2}

Tables~\ref{tab:or_bench_a_full}--\ref{tab:shift_4streams} archive the full-column mirrors of Main Tables~2--3, the Benchmark~A provenance and feature-scope sensitivity check, and the per-stream breakdown across all four Benchmark~B controlled concatenated shifts.

[[TAB_OR_BENCH_A_FULL]]

[[TAB_BENCH_A_SENS]]

[[TAB_OR_BENCH_B_FULL]]

[[TAB_SHIFT_4STREAMS]]

\clearpage
% ============================================================================
% S3. Confirmatory Data Funnel & 34-Stream Archive Follow-Up Audits
% ============================================================================
\section{Confirmatory Data-Construction Funnel and Archive Follow-Up Audits}
\label{sec:or_funnel}

Tables~\ref{tab:confirmatory_funnel}--\ref{tab:followup_sensitivity} document the upstream \texttt{apachejit\_total.csv} filtering funnel (reconciling the 11,964 clean \texttt{apache/hadoop} monorepo commits), observable archive-horizon follow-up across all 34 natural streams, and Benchmark~A sensitivity at $\ge 180\text{d}$ and $\ge 365\text{d}$ cutoffs.

[[TAB_CONFIRMATORY_FUNNEL]]

[[TAB_FOLLOWUP_AUDIT]]

[[TAB_FOLLOWUP_SENSITIVITY]]

\clearpage
% ============================================================================
% S4. RQ3 Full Ablations & 62-Alarm Unrevealed Mechanism Audit
% ============================================================================
\section{RQ3 Complete Ablation Table and 62-Alarm Unrevealed Mechanism Audit}
\label{sec:or_rq3}

Table~\ref{tab:or_rq3_full} gives the complete 7-column RQ3 ablation mirror including ROC-AUC. Table~\ref{tab:mechanism_audit} audits Stage~1--3 intermediate accuracy across all 62 seed-42 DriftLens alarms on the 10 representative streams ($74{,}400$ pool evaluations across $59{,}600$ unique commits), separating strictly unrevealed commits ($y_{\text{obs}}=0$) from the positive-writeback control while comparing Survival-PU against FixedCurve-PU. Reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is computed per stream over archive positives (counting unlinked defects as unrevealed at finite ages).

[[TAB_OR_RQ3_FULL]]

[[TAB_MECHANISM_AUDIT]]

\clearpage
% ============================================================================
% S5. RQ4 Extended Stress Tests
% ============================================================================
\section{RQ4 Extended Latency, Calibration, SZZ Noise, and Code-LLM Tables}
\label{sec:or_rq4}

Tables~\ref{tab:or_rq4a_kappa_full}--\ref{tab:rq4f_llm} archive the full 11-column RQ4a latency scaling table including Wait-90, the matched FixedCurve-PU Stage-1 misspecification control, the complete 10-column Raw versus 90-day Platt calibration table including raw ECE and calibration slope, the RQ4c $2\times 2$ semi-synthetic PR-SZZ perturbation table, and the RQ4e Code-LLM prompting study.

[[TAB_OR_RQ4A_KAPPA_FULL]]

[[TAB_RQ4A_FIXEDCURVE]]

[[TAB_OR_RQ4A_POSTHOC_FULL]]

[[TAB_RQ4D_SZZ]]

[[TAB_RQ4F_LLM]]

\paragraph{RQ4e Paired Prompting Protocol Details.}
The 6,400 logical evaluations in Table~\ref{tab:rq4f_llm} queried \texttt{deepseek-flash} at \texttt{temperature=0.0} with strict JSON schema output across the first 320 evaluation commits of natural Qt and OpenStack. Across $6{,}819$ HTTP attempts, $419$ transient transport errors were retried and only two fell back to $0.50$, yielding a $99.97\%$ valid-response rate, $100\%$ JSON compliance across $6{,}398$ responses, and $99.95\%$ non-empty diff coverage. Zero-shot prompts supplied the commit message, truncated code diff, and 13 commit-authoring metrics. Paired 6-shot arms sampled identical historical $1W$ exemplars across three random seeds differing solely in $y_{\text{obs}}(t)$ versus $y^*$. The descriptive causal-mature arm drew aged $\ge 90\text{d}$ exemplars, reaching $0.5072$ $F_1$ on OpenStack and $0.2984$ on Qt.

\clearpage
% ============================================================================
% S6. RQ5 Wall-Clock Runtime & Forensic Illustrations
% ============================================================================
\section{RQ5 Phase-Decomposed Runtime Telemetry and Forensic Illustrations}
\label{sec:or_rq5}

Table~\ref{tab:rq5_runtime} details standalone and incremental wall-clock runtime across 27 natural Benchmark~B streams, and Table~\ref{tab:rq5_forensics} archives the pre-declared high-confidence SZZ rescue illustrations on natural Qt.

[[TAB_RQ5_RUNTIME]]

[[TAB_RQ5_FORENSICS]]

\paragraph{RQ5 Qualitative Commit Illustrations and Churn Skew.}
Applying our pre-declared high-confidence forensic filter (lag $\ge 14\text{d}$, Frozen $\ge 0.45$, Causal-4W $\le 0.35$, Survival-PU $\ge 0.50$) on natural Qt CodeBERT seed 42 yields 22 rescue illustrations (Table~\ref{tab:rq5_forensics}), where Alarm~\#1 at commit $9{,}900$ experienced $88.19\%$ $4W$ censoring; zero alarms fired on OpenStack CodeBERT seed 42. Across full evaluation sets, high-confidence SZZ false-negative candidates skew smaller than all defects: $436$ Qt FNs have median $32$ versus $69$ LOC across all $981$ defects, and $477$ OpenStack FNs have median $74$ versus $89$ LOC. Conversely, high-score false positives skew larger than clean commits, with medians of $65$ versus $9$ LOC on Qt ($1{,}620$ FPs) and $146$ versus $19$ LOC on OpenStack ($18$ FPs), spanning $2$-LOC faults (\texttt{qtmultimedia:4bb75559}) to $570{,}767$-LOC imports (\texttt{qtwebkit:d441d6f3}).

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
    supp_replacements = {
        "[[TAB_OR_BENCH_A_FULL]]": supp_tmap["tab:or_bench_a_full"],
        "[[TAB_BENCH_A_SENS]]": supp_tmap["tab:bench_a_sens"],
        "[[TAB_OR_BENCH_B_FULL]]": supp_tmap["tab:or_bench_b_full"],
        "[[TAB_SHIFT_4STREAMS]]": supp_tmap["tab:shift_4streams"],
        "[[TAB_CONFIRMATORY_FUNNEL]]": supp_tmap["tab:confirmatory_funnel"],
        "[[TAB_FOLLOWUP_AUDIT]]": supp_tmap["tab:followup_audit"],
        "[[TAB_FOLLOWUP_SENSITIVITY]]": supp_tmap["tab:followup_sensitivity"],
        "[[TAB_OR_RQ3_FULL]]": supp_tmap["tab:or_rq3_full"],
        "[[TAB_MECHANISM_AUDIT]]": supp_tmap["tab:mechanism_audit"],
        "[[TAB_OR_RQ4A_KAPPA_FULL]]": supp_tmap["tab:or_rq4a_kappa_full"],
        "[[TAB_RQ4A_FIXEDCURVE]]": supp_tmap["tab:rq4a_fixedcurve"],
        "[[TAB_OR_RQ4A_POSTHOC_FULL]]": supp_tmap["tab:or_rq4a_posthoc_full"],
        "[[TAB_RQ4D_SZZ]]": supp_tmap["tab:rq4d_szz"],
        "[[TAB_RQ4F_LLM]]": supp_tmap["tab:rq4f_llm"],
        "[[TAB_RQ5_RUNTIME]]": supp_tmap["tab:rq5_runtime"],
        "[[TAB_RQ5_FORENSICS]]": supp_tmap["tab:rq5_forensics"],
    }
    for k, v in supp_replacements.items():
        assert k in tex, f"Placeholder {k} missing in online_resource.tex template!"
        tex = tex.replace(k, v)
    return tex


def build_markdown_manuscript() -> str:
    return """# Detecting Drift Is Not Safe Retraining: Causal Supervision Boundaries for Adaptive Just-in-Time Defect Prediction

> **Synchronized Manuscript Summary (`manuscript/main.tex` / `manuscript/main.pdf` + companion `manuscript/Online_Resource_1.pdf`, verified across 839 Main empirical table cells + 1,054 Online Resource 1 empirical table cells = 1,893 cells across 27 tables and 15 JSON outputs by `src_v3/verify_manuscript_v3.py`, `scripts_v3/verify_online_resource_v3.py`, and `scripts_v3/verify_prose_claims_v3.py`).**

## Abstract

Unsupervised drift monitors such as DriftLens flag representation shifts in Just-in-Time Defect Prediction without defect labels, motivating immediate window retraining. Label-free monitoring determines only when to retrain, however, not how to supervise weight updates. Under multi-month SZZ verification latency, most recently introduced defects remain unfixed at alarm time and appear clean. We audit this causal supervision gap and map operating boundaries across linear, streaming, and Positive-Unlabeled learners. Across 34 natural repository streams totaling 347,122 commits, eight controlled shifts, and a paired Code-LLM case study, we evaluate circular block-bootstrap DriftLens alongside three-stage Survival-PU, which combines cohort-incidence maturity estimation, clipped prior shrinkage, and soft-target distillation. Across scheduled windows in multi-year ecosystems, 68.8% to 94.6% of recent defects remain right-censored. Leaking future fixes into matched window-oracle updates significantly inflates Average Precision over causal retraining across multi-year donor clusters on both neural and tree hosts. On the shorter confirmatory cohort, however, neural hindsight optimism fails to replicate. Once strict causal visibility holds, no single learner dominates every regime. Frozen linear models lead deployable multimodal streams, simpler causal baselines and zero-fit FixedCurve-PU outperform Survival-PU on gradual whole-stream ranking, and streaming Hoeffding trees led by ODaSC excel on the confirmatory cohort. By contrast, Survival-PU leads full post-shift ranking at 0.5764 Average Precision, avoids zero-latency misspecification, and protects thresholded classification under severe latency where standard post-hoc calibration degrades.

---

## Core Empirical & Statistical Findings

1. **RQ1 — Causal Supervision Audit:**
   - Across scheduled 400-commit ($1W$) boundaries in multi-year cohorts, 68.8%–94.6% of window defects are right-censored (`Qt`: 94.6% scheduled, 95.3% pooled across 981 window-defects, 95.7% at LightGBM alarms; `OpenStack`: 87.5% / 84.4%; `Cabral-14`: 68.8% / 68.2%; `ApacheJIT-8`: 70.4% / 65.8%; `PR-SZZ-5`: 78.8% / 78.9%).
   - Across $K=25$ independent donor clusters, offline window-oracle supervision (`B1b_4W_Oracle`) significantly inflates AP over matched causal `B2_4W` on both MLP (`0.5241` vs. `0.5004`, $+0.0237$, 20/0/5, $p_{\\text{Holm}}=6.63\\times 10^{-5}$) and LightGBM (`0.5409` vs. `0.4933`, $+0.0476$, 23/0/2, $p_{\\text{Holm}}=3.70\\times 10^{-5}$).

2. **RQ2 — Operating Boundaries, Evaluation Granularity, & Confirmatory Non-Replication:**
   - **Benchmark A:** Frozen `Full-LR` (`0.3653` AP) and `LApredict` (`0.3601` AP) lead all deployable single-host configurations (top deployable single host: `DeepJIT-CNN Frozen = 0.3531`), while `Survival-PU` recovers the top Track-1 $F_1$ (`0.2907`) and $F_{1,\\text{th}}$ (`0.3043`).
   - **Benchmark B ($K=25$):** `Full-LR Adapt-4W` (`0.5108`, $p_{\\text{Holm}}=0.0639$) leads in mean. `Causal-4W` (`0.5004`, $p_{\\text{Holm}}=0.0056$), `Wait-90` (`0.4982`, $p_{\\text{Holm}}=0.0069$), and zero-fit `FixedCurve-PU` (`0.4942`, $p_{\\text{Holm}}=0.0295$) significantly exceed `Survival-PU` (`0.4866`) on whole-stream AP; within-window standardization on 5-seed predictions reverses the mean `Causal-4W` gap without reaching significance (`Rank-Norm: 0.5027 vs. 0.4989`, $p=0.0877$).
   - **Controlled Shifts:** `Survival-PU` leads full post-onset AP (`0.5764`), whereas `BORB-Alarm` (`0.5931`) leads the first 1,200 post-onset commits.
   - **Confirmatory Phase C:** Disclosing upstream `apachejit_total.csv`'s 11,964-commit clean `apache/hadoop` split, MLP hindsight optimism does not replicate on the three standalone repositories (`0.5951 vs. 0.5985`; `ODaSC = 0.7857`) or the four-umbrella cohort (`0.6801 vs. 0.6858`; `ODaSC = 0.8289`).

3. **RQ3–RQ5 — Mechanism Audit, Latency Extremes, Block Calibration, & Budget Granularity:**
   - All four Stage-1 maturity estimators span `1.09` AP points with zero fallbacks; across 62 alarms, cohort incidence improves Stage-1 maturity MAE over `FixedCurve-PU` (`0.0938 vs. 0.1204`) yet `FixedCurve-PU` achieves slightly lower Stage-2 prior MAE (`0.0565 vs. 0.0620`) and unrevealed Brier (`0.1200 vs. 0.1236`).
   - At $\\kappa=3.0\\times$, `FixedCurve-PU` leads AP (`0.5547`) and raw calibration (`0.1693 / 0.0973`), while `Survival-PU` beats `Causal-4W`, both post-hoc controls, and `FixedCurve-PU` on scale-fair $F_{1,\\text{th}}$ (`0.5022` raw; `0.4898` Platt) and beats `FixedCurve-PU` across all metrics at $\\kappa=0\\times$. Standard 90-day Platt calibration degrades probability quality under `65.9%` pool censoring.
   - Circular block-bootstrap DriftLens cuts null alarms `4.8x` and catches shifts `3x` faster than periodic `3W` retraining. Operational per-window 400-commit budgets reverse most steep whole-stream net-TP losses on Benchmark A while remaining negative on Qt CodeBERT.
"""


def build_n05_readme() -> str:
    return """# Detecting Drift Is Not Safe Retraining: Causal Supervision Boundaries for Adaptive Just-in-Time Defect Prediction

Official replication package, companion **Online Resource 1**, and automated verification suite for the manuscript submitted to *Software Quality Journal* (Springer Nature).

**Authors:** Zhijie Li¹ · Shaoyang Xu¹ · Ketong Liu¹ · Shibo Meng¹ · Feiran Gao² · Yuli Zhao¹\\*  
¹ Software College, Northeastern University, Shenyang 110169, China  
² School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China  
\\* Corresponding author: `zhaoyl@swc.neu.edu.cn`

---

## Study Overview

Unsupervised drift monitors such as DriftLens detect embedding-space shifts in Just-in-Time Software Defect Prediction (JIT-SDP) without defect labels, motivating immediate window retraining. Yet label-free monitoring removes label dependence only when deciding *when* to retrain—not when *supervising* weight updates. Under multi-month SZZ verification latency, most genuinely bug-inducing commits inside newly flagged windows remain unfixed at alarm time ($t_i^{\\mathrm{fix}} > t_{\\mathrm{alarm}}$) and appear clean under strictly causal deployment.

This replication package accompanies our causal supervision audit and cross-paradigm operating-boundary study spanning **37 development streams** and **5 confirmatory ApacheJIT repositories** (**34 natural repository streams** and **8 controlled-shift streams**; **347,122 natural commits**), **29 evaluated methods** (26 deployable + 3 hindsight references across 41 prefix-invariant execution branches), and a **6,400-call paired Code-LLM prompting case study** (`deepseek-flash`).

### Key Headline Findings

- **Severe Alarm-Time Right-Censoring & Hindsight Optimism (RQ1):** Across scheduled 400-commit ($1W$) boundaries in multi-year ecosystems, `68.8%–94.6%` of window defects remain right-censored (`65.8%–95.7%` at fired alarms; `45.1%–80.5%` even at $4W$), whereas the shorter confirmatory cohort exhibits lighter censoring (`26.4%` across 5 repositories; `18.0%` on the three standalone repositories). Leaking future SZZ fixes into matched window-oracle updates (`Window Oracle` vs. `Causal-4W`) significantly inflates whole-stream Average Precision (AP) across $K=25$ independent donor clusters on both MLP (`+0.0237`, `20/0/5`, $p_{\\mathrm{Holm}}=6.63\\times 10^{-5}$) and LightGBM (`+0.0476`, `23/0/2`, $p_{\\mathrm{Holm}}=3.70\\times 10^{-5}$).
- **Cross-Paradigm Operating Boundaries, Granularity, & Confirmatory Non-Replication (RQ2):** No single adaptive paradigm dominates across regimes. Frozen 13-metric logistic regression (`Full-LR Frozen = 0.3653` AP) leads every deployable multimodal/tree host on Benchmark A. On gradual $K=25$ whole-stream AP, `Full-LR Adapt-4W` (`0.5108`) leads in mean. `Causal-4W` (`0.5004`, $p_{\\mathrm{Holm}}=0.0056$), `Wait-90` (`0.4982`, $p_{\\mathrm{Holm}}=0.0069$), and zero-fit `FixedCurve-PU` (`0.4942`, $p_{\\mathrm{Holm}}=0.0295$) significantly exceed `Survival-PU` (`0.4866`)—though within-window standardization on identical 5-seed predictions reverses the mean `Causal-4W` gap without reaching significance (`Rank-Norm: 0.5027 vs. 0.4989`, unadjusted $p=0.0877$). On the short confirmatory cohort, disclosing upstream `apachejit_total.csv`'s `11,964`-commit clean `apache/hadoop` split, MLP hindsight optimism does not replicate on the three standalone repositories (`0.5951 vs. 0.5985`; `ODaSC = 0.7857`) or the four-umbrella cohort (`0.6801 vs. 0.6858`; `ODaSC = 0.8289`). Conversely, `Survival-PU` leads full post-shift AP (`0.5764`) and LightGBM Stage-3 distillation (`+0.0379` over `Teacher-Matched`, $p_{\\mathrm{Holm}}=9.87\\times 10^{-4}$).
- **Unrevealed-Sample Audit & Non-Monotonic Cross-Stage Error Propagation (RQ3):** Across 62 fired DriftLens alarms (`74,400` pool evaluations / `59,600` unique commits), adaptive cohort incidence improves Stage-1 maturity MAE over `FixedCurve-PU` (`0.0938` vs. `0.1204`) and reduces strictly unrevealed ($y_{\\mathrm{obs}}=0$) soft-target Brier over de-weighted teachers (`0.1303 -> 0.1236`), yet `FixedCurve-PU` attains slightly lower Stage-2 prior MAE (`0.0565` vs. `0.0620`) and unrevealed Brier (`0.1200`).
- **Extreme Latency, Post-Hoc Controls, Block Calibration, & Budget Granularity (RQ4–RQ5):** Under $\\kappa=3.0\\times$ latency, `FixedCurve-PU` leads AP (`0.5547`) and raw Brier/ECE (`0.1693 / 0.0973`), while `Survival-PU` beats `Causal-4W`, alarm-synchronized `PostHoc Logit`, Proposition-1-inspired `PostHoc Ratio`, and `FixedCurve-PU` on scale-fair threshold-tuned $F_{1,\\mathrm{th}}$ (`0.5022` raw; `0.4898` Platt) and avoids `FixedCurve-PU`'s $\\kappa=0\\times$ misspecification (`0.5760 vs. 0.5544` AP). Standard 90-day Platt calibration degrades probability quality under `65.9%` calibration-pool censoring. Circular block-bootstrap DriftLens cuts validation-pool null alarms `4.8x` and detects shifts `3x` faster than periodic `3W` retraining (`350` vs. `1,050` commits; `0/20` missed). Operational per-window 400-commit budgets reverse most steep whole-stream net-TP losses on Benchmark A while remaining negative on Qt CodeBERT.

---

## Manuscript & Online Resource 1 Crosswalk

Every empirical table cell across the **Main Manuscript** (`839` cells across `11` empirical tables) and **Online Resource 1** (`1,054` cells across `16` supplementary/full-column tables; **`1,893` total machine-verified cells**), plus all `33` locked `V3_CONFIG` fields and all key prose comparisons (`34` exact-path claims + `60` relational guards), is verified automatically against the archived JSON artifacts in this repository.

| Document & Element | Backing Artifact File(s) in This Repository | Generator / Verifier Script |
| :--- | :--- | :--- |
| **Main Table 1** (RQ1 Censoring Audit) | `results_v3/data_provenance_audit_v3.json`, `manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 2** (RQ2 Benchmark A Summary) | `results_v3/benchmark_a_multihost_v3.json`, `manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 3** (RQ2 Benchmark B $K=25$ / Shift) | `results_v3/benchmark_b_main_v3.json`, `manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 4** (RQ2 Holm–Bonferroni Inference) | `results_v3/benchmark_b_main_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 5** (RQ2 5-Seed Win-AP & Scale Removal) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_f_window_vs_whole_ap`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Main Table 6** (RQ2 Confirmatory ApacheJIT) | `results_v3/confirmatory_cohort_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 7** (RQ3 Estimators & Ablations) | `results_v3/rq3_ablations_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 8** (RQ4a Future-Only Latency Scaling) | `results_v3/rq4a_latency_intervention_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 9** (RQ4a Raw vs. 90d Platt & Post-Hoc) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_d_calibration_and_posthoc`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Main Table 10** (RQ4b $2\\times 2$ DriftLens Factorial) | `results_v3/rq4b_factorial_monitors_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Main Table 11** (RQ5 Equal-Cap Review Budgets) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_g_paired_net_gain`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Main Table 12** (Synthesis: Patterns vs. Hypotheses) | Synthesized from Main Tables 1–11 & Online Resource 1 | `scripts_v3/build_manuscript_v3.py` |
| **Main Table A1** (33 Locked Protocol Parameters) | `src_v3/config_v3.py` (`V3_CONFIG`) | `scripts_v3/verify_prose_claims_v3.py` |
| **Online Resource 1 (Tables S-1–S-2)** | Conceptual Positioning & Executed Host/Competitor Specs | `scripts_v3/build_manuscript_v3.py` |
| **Online Resource 1 (Tables S-3–S-6)** | Full 10-col Bench A, Provenance Sens., Full 8-col Bench B, 4-Shift Breakdown | `scripts_v3/verify_online_resource_v3.py` |
| **Online Resource 1 (Tables S-7–S-9)** | ApacheJIT Funnel, 34-Stream Follow-Up Audit, Bench A $\\ge 180/365$d Sens. | `scripts_v3/verify_online_resource_v3.py` |
| **Online Resource 1 (Tables S-10–S-11)** | Full 7-col RQ3 Ablations & 62-Alarm Unrevealed Mechanism Audit | `scripts_v3/verify_online_resource_v3.py` |
| **Online Resource 1 (Tables S-12–S-16)** | Full 11-col RQ4a, FixedCurve Control, Full 10-col Post-Hoc, SZZ Noise, Code-LLM | `scripts_v3/verify_online_resource_v3.py` |
| **Online Resource 1 (Tables S-17–S-18)** | RQ5 Wall-Clock Runtime Telemetry & Qt High-Confidence SZZ Forensics | `scripts_v3/verify_online_resource_v3.py` |
| **Main Fig. 1–10** (Conceptual & Empirical Figures) | `figures/fig{0a,0b,1..8}_*_v3.{pdf,png}` | `scripts_v3/generate_conceptual_diagrams_v3.py`, `analyze_and_plot_v3.py` |

> **Artifact Naming Conventions:** To preserve exact SHA-256 provenance locks across all immutable `src_v3/` modules and archived `results_v3/*_v3.json` files, internal identifiers retain their frozen execution names: `v3` marks the locked V3 campaign; `n60_*` denotes the five PR-SZZ streams from Bludau & Pretschner (2026); historical JSON filenames `rq4d_szz_perturbation_v3.json`, `rq4e_sensitivity_sweeps_v3.json`, and `rq4f_llm_audit_v3.json` map to manuscript subsections **RQ4c**, **RQ4d**, and **RQ4e**.

---

## Repository Structure

```text
CausalJIT-Audit-SQJ/
├── README.md
├── LICENSE                                         # MIT License (code/scripts)
├── requirements_lock_v3.txt                        # Exact pinned Python environment (Python 3.13 / CUDA 12.6)
├── src_v3/                                         # [Immutable, SHA-256 provenance-locked core implementation]
│   ├── config_v3.py                                # Frozen V3_CONFIG, donor-cluster maps (K=25/24), codified boundaries
│   ├── data_audit_v3.py                            # Leak-free stream builder, feature filtering, confirmatory unlocker
│   ├── build_codebert_cache_v3.py                  # Offline microsoft/codebert-base feature cache builder
│   ├── survival_estimators_v3.py                   # CohortIncidence, 1e-4d KM cure, Weibull cure, FixedCurve-PU, bisection
│   ├── models_and_baselines_v3.py                  # Att-CNN, CodeBERT, DeepJIT-CNN, MLP, LightGBMHostV3, 2x2 DriftLens
│   ├── stream_competitors_v3.py                    # Prefix-invariant ORB, BORB, ODaSC, Full-LR, 3-branch causal calibration
│   ├── metrics_and_stats_v3.py                     # Horizon evaluator, exact/Pratt paired Wilcoxon, Holm-Bonferroni
│   ├── run_v3_experiments.py                       # Master Phase A / B / mechanisms / confirmatory runner
│   ├── run_v3_llm_and_forensics.py                 # Audited Code-LLM prompting (RQ4e) & forensics (RQ5) runner
│   ├── verify_mechanisms_and_protocol_v3.py        # Historical pre-unlock 14-check gate (archived SHA-256 766e3fd6...)
│   └── verify_manuscript_v3.py                     # [Tier 1] SHA-256 provenance, .npz, BibTeX & 839 Main cell verifier
├── scripts_v3/                                     # [Portable companion audits, plotting, assembly & verification]
│   ├── verify_online_resource_v3.py                # [Tier 1] Verifies all 1,054 Online Resource 1 cells against results_v3
│   ├── run_post_campaign_mechanism_checks_v3.py    # [Tier 1] Verifies pre-unlock lock + runs live Checks 2-14 (41 branches)
│   ├── verify_prose_claims_v3.py                   # [Tier 1] Verifies V3_CONFIG, 34 exact-path claims & 60 relational guards
│   ├── generate_conceptual_diagrams_v3.py          # [Tier 1] Regenerates conceptual vector/PNG diagrams (Fig. 1-2)
│   ├── analyze_and_plot_v3.py                      # [Tier 1] Regenerates Fig. 3-10, 27 tables & both cell registries
│   ├── build_manuscript_v3.py                      # [Tier 1] Assembles main.tex, online_resource.tex, Markdown & README
│   ├── run_rq4a_fixedcurve_control_v3.py           # [Tier 2] Matched RQ4a zero-fit FixedCurve-PU vs. Survival-PU control
│   ├── run_reviewer_mechanism_audits_v3.py         # [Tier 2] Audits A-G (funnel, follow-up, Stage 1-3, PostHoc, WinAP, budgets)
│   ├── run_parallel_campaign_v3.py                 # [Tier 3] Multi-worker campaign orchestrator over src_v3 runners
│   └── package_sqj_submission_v3.py                # Release bundler & clean-room LaTeX compilation verifier
├── cache/
│   ├── benchmark_b_v3/                             # 27 natural + 4 controlled-shift tabular .npz bundles + SHA-256 manifest
│   ├── confirmatory_v3/                            # 5 confirmatory ApacheJIT .npz bundles + SHA-256 manifest
│   ├── multimodal_v3_pure13/                       # 6 Benchmark A tabular *_tab.npz bundles + manifest (tokens/CB in Zenodo)
│   ├── multimodal_v3_sens/                         # Benchmark A provenance/feature sensitivity manifest
│   └── codebert_v3/                                # CodeBERT cache SHA-256 manifest (768-D embeddings in Zenodo)
├── results_v3/
│   ├── *_v3.json                                   # 15 provenance-stamped JSON outputs & Main cell registry
│   ├── online_resource_cell_registry.json          # 1,054-cell Online Resource 1 cell registry
│   ├── prerun_gate_archive/                        # Frozen pre-unlock 14-check gate report (SHA-256 766e3fd6...)
│   ├── commit_predictions/                         # 50 commit-level seed-0 .npz archives (385,620 commits)
│   ├── per_cell_a/ & per_stream_b/                 # Per-cell Benchmark A & per-stream Benchmark B archived summaries
├── logs/
│   ├── llm_raw_exchange_v3.jsonl.gz                # Lossless archive of all 6,400 Code-LLM evaluations (6,819 HTTP calls)
│   └── llm_api_audit_v3.jsonl                      # Per-call latency, token, JSON-compliance & retry audit ledger
├── figures/                                        # Vector PDF & 300-DPI PNG figures (Fig. 1-10)
└── manuscript/                                     # Camera-ready Main PDF/TeX, Online_Resource_1.pdf/TeX, BibTeX & Markdown
```

---

## Reproduction & Verification Guide (Four Tiers)

All post-processing and verification scripts automatically locate the repository root relative to `__file__` (or honor `CAUSALJIT_BASE_DIR`) and contain zero hardcoded host paths.

### Tier 1 — Immediate Offline Verification & Full Figure/Table Regeneration (Zero External Downloads)

Requires only Python 3.10+ (tested on Python `3.13.0`) and the packages in `requirements_lock_v3.txt`. Runs completely offline on CPU directly from a fresh GitHub clone:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\\Scripts\\activate
pip install -r requirements_lock_v3.txt

# 1. Verify all 15 JSON provenance hashes, 50 .npz archives (385,620 commits), BibTeX & 839 Main table cells
python src_v3/verify_manuscript_v3.py

# 2. Verify all 1,054 Online Resource 1 table cells across 16 supplementary tables
python scripts_v3/verify_online_resource_v3.py

# 3. Verify all 33 V3_CONFIG table fields, 34 exact-path prose claims, 60 relational guards & adversarial self-tests
python scripts_v3/verify_prose_claims_v3.py

# 4. Verify archived pre-unlock gate SHA-256 (766e3fd6...) and run live mechanism Checks 2-14 (41 prefix-invariant branches)
python scripts_v3/run_post_campaign_mechanism_checks_v3.py

# 5. Regenerate all 10 figures (Fig. 1-10), 27 empirical LaTeX tables, manuscript/main.tex, and online_resource.tex
python scripts_v3/generate_conceptual_diagrams_v3.py
python scripts_v3/analyze_and_plot_v3.py
python scripts_v3/build_manuscript_v3.py
```

Compiling `manuscript/main.tex` and `manuscript/online_resource.tex` via `pdflatex -> bibtex -> pdflatex -> pdflatex` reproduces the camera-ready **39-page Springer SQJ Main PDF** (`Section 6 Empirical Results` starting at page 13) and **11-page companion Online Resource 1 PDF** (`Online_Resource_1.pdf`), both with `0` Errors, `0` Warnings, and `0` Overfull hboxes.

### Tier 2 — Tabular Benchmark & Companion Audit Re-Execution

Because all 31 Benchmark B `.npz` bundles, all 5 confirmatory `.npz` bundles, all 6 Benchmark A `*_tab.npz` bundles, and all 50 seed-0 commit prediction `.npz` archives are included directly in this repository, you can re-execute the tabular campaigns and companion audits offline without extra downloads:

```bash
python src_v3/run_v3_experiments.py --phase phase_b
python src_v3/run_v3_experiments.py --phase mechanisms
python src_v3/run_v3_experiments.py --phase confirmatory
python scripts_v3/run_rq4a_fixedcurve_control_v3.py
python scripts_v3/run_reviewer_mechanism_audits_v3.py
```

> **Note on Audit A (Upstream ApacheJIT Funnel):** Re-running Audit A's raw CSV row counts requires the upstream `apachejit_total.csv` (`data_sync/confirmatory_apachejit_raw/`, included in the Tier 3 Zenodo archive).

### Tier 3 — Full Multimodal Re-Execution & Raw-to-Bundle Reconstruction (Full Zenodo Archive)

Re-executing Benchmark A neural hosts (`Attention-CNN`, `CodeBERT`, `DeepJIT-CNN`), rebuilding `.npz` bundles from raw corpora (`data_sync/`), or re-extracting `768`-D `microsoft/codebert-base` embeddings requires the complete `1.55 GB` offline Zenodo replication archive (available via the corresponding author `zhaoyl@swc.neu.edu.cn` and scheduled for permanent Zenodo DOI deposit upon acceptance):

```bash
python src_v3/build_codebert_cache_v3.py
python src_v3/data_audit_v3.py --rebuild-all --unlock-confirmatory
python scripts_v3/run_parallel_campaign_v3.py --workers 8
python src_v3/run_v3_llm_and_forensics.py --mode rq5
```

### Tier 4 — Optional Live Code-LLM Prompting Audit (RQ4e)

Offline inspection and verification of all `6,400` logical `deepseek-flash` evaluations (`6,819` HTTP attempts) in `logs/llm_raw_exchange_v3.jsonl.gz` requires **no** API key or network access. Re-issuing live API calls requires placing a DeepSeek key at `../../.llm_api/.deepseek.key` relative to the repository root and running `python src_v3/run_v3_llm_and_forensics.py --mode rq4f`.

---

## Citation

```bibtex
@article{li2026causaljit,
  title   = {Detecting Drift Is Not Safe Retraining: Causal Supervision Boundaries for Adaptive Just-in-Time Defect Prediction},
  author  = {Li, Zhijie and Xu, Shaoyang and Liu, Ketong and Meng, Shibo and Gao, Feiran and Zhao, Yuli},
  journal = {Software Quality Journal},
  year    = {2026},
  note    = {Under review. Replication package: \\url{https://github.com/789748456v/CausalJIT-Audit-SQJ}}
}
```

---

## License

- **Code & Scripts (`src_v3/`, `scripts_v3/`):** [MIT License](LICENSE).
- **Structured Results, Manifests, Tables, & Figures (`results_v3/`, `figures/`, `manuscript/`):** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), respecting upstream open-source project licenses.
"""


def main() -> None:
    tables_tex_path = os.path.join(BUILD_DIR, "v3_tables_generated.tex")
    tmap = load_generated_tables(tables_tex_path)
    assert len(tmap) == 11, f"Expected 11 generated Main tables, got {len(tmap)}: {list(tmap.keys())}"

    supp_tables_tex_path = os.path.join(BUILD_DIR, "v3_online_resource_tables.tex")
    supp_tmap = load_generated_tables(supp_tables_tex_path)
    assert len(supp_tmap) == 16, f"Expected 16 generated Online Resource tables, got {len(supp_tmap)}: {list(supp_tmap.keys())}"

    ms_dir = os.path.join(BASE_DIR, "manuscript")
    os.makedirs(ms_dir, exist_ok=True)

    tex_txt = build_main_tex(tmap)
    with open(os.path.join(ms_dir, "main.tex"), "w", encoding="utf-8") as f:
        f.write(tex_txt)

    or_txt = build_online_resource_tex(supp_tmap)
    with open(os.path.join(ms_dir, "online_resource.tex"), "w", encoding="utf-8") as f:
        f.write(or_txt)

    md_txt = build_markdown_manuscript()
    with open(os.path.join(ms_dir, "N05_SQJ_Manuscript.md"), "w", encoding="utf-8") as f:
        f.write(md_txt)

    rm_txt = build_n05_readme()
    with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(rm_txt)

    print(
        f"Assembled manuscript/main.tex ({len(tmap)} empirical tables), "
        f"manuscript/online_resource.tex ({len(supp_tmap)} empirical tables), "
        f"manuscript/N05_SQJ_Manuscript.md, and README.md successfully."
    )


if __name__ == "__main__":
    main()
