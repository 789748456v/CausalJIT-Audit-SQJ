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
\PassOptionsToPackage{hyphens}{url}
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
\usepackage{microtype}%
\def\UrlBigBreaks{\do\/}%
\def\UrlBreaks{\do\/\do-\do.}%
\def\UrlNoBreaks{\do\:}%

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
\abstract{Unsupervised drift monitors such as DriftLens flag representation shifts in Just-in-Time Defect Prediction without defect labels, motivating immediate window retraining. Label-free monitoring decides only when to retrain, however, not how to supervise weight updates. Under multi-month SZZ verification latency, most recently introduced defects remain unfixed at alarm time and appear clean. We audit this causal supervision gap across 34 natural repository streams totaling 347,122 commits, spanning 29 multi-year streams and five shorter confirmatory projects, alongside eight controlled shifts and a Code-LLM case study. Across linear, streaming, and Positive-Unlabeled learners, we evaluate circular block-bootstrap DriftLens alongside Survival-PU, a controlled maturity-aware three-stage testbed trained on soft targets. In multi-year ecosystems, 68.8\% to 94.6\% of defects inside recent retraining windows remain right-censored at alarm time. Leaking future fixes into matched window-oracle updates significantly inflates Average Precision over causal retraining across multi-year neural and tree models, and block-bootstrap calibration cuts DriftLens false alarms nearly fivefold. On the shorter confirmatory projects, however, that neural hindsight optimism fails to replicate. Once strict causal visibility holds, no single paradigm dominates every regime. Frozen linear models lead deployable multimodal streams, simpler causal baselines and a fixed maturity curve outperform Survival-PU on gradual whole-stream ranking, and streaming Hoeffding trees led by ODaSC excel on the confirmatory projects. By contrast, Survival-PU leads post-shift ranking, avoids hardcoded-lag errors when fixes arrive immediately, and protects thresholded classification under severe latency where standard post-hoc calibration degrades.}

\keywords{Just-in-Time Defect Prediction, Distribution Shift, Verification Latency, Right-Censoring, Positive-Unlabeled Learning, Survival Analysis}

\maketitle

% ============================================================================
% 1. Introduction
% ============================================================================
\section{Introduction}
\label{sec:intro}

Just-in-Time Software Defect Prediction (JIT-SDP) helps engineering teams prioritize code review at commit time, when design intent is still fresh \citep{kamei2013large,hoang2019deepjit,pornprasit2021jitline,marrium2026flame}. Static commit classifiers steadily lose accuracy over time. Refactoring, framework migrations, subsystem turnover, and changing review habits gradually erode fixed weights \citep{mcintosh2018are,gama2014survey,lu2018learning}. Adapting those weights is unusually hard because JIT-SDP suffers heavy-tailed \textit{verification latency} \citep{cabral2019class,tabassam2020handling,song2022odasc,song2023humla}. A bug-inducing commit enters the repository at timestamp $t_i$, and SZZ back-tracing assigns its defect label only after a fault surfaces and a developer merges a fix at $t_i^{\text{fix}}$ \citep{sliwerski2005when,dacosta2017framework,rosa2021evaluating,fan2021impact,bludau2026quantifying}. In mature projects such as Qt and OpenStack, a faulty change typically sits in production for five to six months before a merged fix lets SZZ mark it defective.

To adapt without waiting months for confirmed fixes, recent quality studies turn to \textit{unsupervised embedding-space drift monitors}. \citet{pote2026adaptive} paired deep commit encoders, Attention-CNN and CodeBERT, with DriftLens \citep{greco2021driftlens,greco2023driftlens}. DriftLens measures the Fr\'echet distance \citep{dowson1982frechet} between pre-stream reference embeddings and incoming commit windows $W_{\text{drift}}$. Because it needs no defect labels, it detects shifts in milliseconds and triggers retraining on the latest window, expanding to four-window replay after consecutive alarms. Since these retrospective evaluations also replayed commits in chronological order, they appeared faithful to live continuous integration \citep{pote2026adaptive}.

Moving drift-triggered retraining from archival datasets into live continuous integration exposes a deeper causal supervision gap:

\begin{quote}
\small\textit{Unsupervised drift monitoring removes label dependence only for \underline{deciding when to retrain}, not for \underline{supervising how to update model weights}. At the timestamp $t_{\text{alarm}}$ when DriftLens flags a shifted window $W_{\text{drift}}$, most genuinely bug-inducing commits inside $W_{\text{drift}}$ are only days or weeks old, so their fix arrives later ($t_i^{\text{fix}} > t_{\text{alarm}}$). Under strict causal visibility, those unrevealed defects appear clean at alarm time.}
\end{quote}

This supervision gap raises a practical question: when does drift-triggered retraining actually help rather than corrupt a live predictor? To investigate it, we pair circular block-bootstrap DriftLens calibration with Survival-PU, a controlled three-stage Positive-Unlabeled testbed that estimates commit maturity from past fix delays and retrains on soft targets rather than treating unfixed commits as clean. Three concrete inquiries guide our study. First, we measure \textit{how blocking future SZZ fixes changes apparent retraining gains} under realistic commit-time visibility. Second, we test \textit{what maturity-aware adaptation genuinely adds} over frozen linear models \citep{zeng2021deep}, simpler causal baselines, fixed maturity curves, and latency-aware stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}. Third, we evaluate \textit{what these accuracy differences mean in daily code review} under fixed commit-count and lines-of-code budgets.

Our audit uncovers sharp, regime-specific operating boundaries rather than a single universal winner. Across 25 multi-year open-source project clusters, oracle retraining that leaks future fixes significantly inflates whole-stream Average Precision over causal four-window updates. Once strict causal visibility holds, simpler learners win several major regimes. Frozen logistic regression outperforms every deployable neural and tree model on our multimodal Qt and OpenStack benchmark. On gradual multi-year streams, simpler causal baselines and a zero-fit fixed maturity curve outrank Survival-PU on whole-stream ranking. On five shorter confirmatory ApacheJIT repositories, neural hindsight optimism fails to replicate altogether, and online Hoeffding-tree learners led by ODaSC prevail.

At the monitoring layer, replacing uniform with circular block-bootstrap calibration delivers a broad operational gain, cutting DriftLens false alarms nearly fivefold while preserving rapid shift detection. At the adaptation layer, Survival-PU earns repeatable gains in three targeted regimes: leading full post-shift ranking after abrupt shifts, avoiding hardcoded-lag errors when fixes arrive immediately, and protecting thresholded classification under severe verification lag where simpler causal baselines falter. Finally, post-hoc calibration offers no easy shortcut: fitting standard 90-day Platt scaling to a heavily censored calibration pool degrades probability quality across methods.

\subsection{Research Questions}
\label{sec:rqs}

We structure our empirical investigation around five sequential research questions:

\begin{enumerate}[leftmargin=2.2em,label=\textbf{RQ\arabic*:},itemsep=0.02em,topsep=0.20em]
  \item \textbf{Right-Censoring and Hindsight Optimism.} What share of recent defects stay right-censored at scheduled windows and fired alarms, and how far does window-oracle retraining diverge from causal retraining?
  \item \textbf{Cross-Paradigm and Confirmatory Boundaries.} How does Survival-PU compare against causal baselines, frozen linear models, and online stream learners across multimodal, multi-year, shifted, and short-horizon cohorts?
  \item \textbf{Maturity Sensitivity and Ablations.} How sensitive is Survival-PU to maturity estimators, how accurately do its stages estimate maturity and soft targets, and which components drive its gains?
  \item \textbf{Latency, Monitor, Noise, and Prompting Stress Tests.} How robust are adaptive learners and drift monitors to verification lag, detector calibration, SZZ label noise \citep{bludau2026quantifying}, window size, and LLM prompting?
  \item \textbf{Runtime Cost and Review-Budget Utility.} What runtime overhead does each paradigm incur with and without feature encoding, and how do yields shift across fixed thresholds and fixed review budgets?
\end{enumerate}

\subsection{Summary of Contributions}
\label{sec:contributions}

This paper makes four main contributions:

\begin{enumerate}[leftmargin=1.7em,itemsep=0.02em,topsep=0.20em]
  \item \textbf{Formal and Empirical Causal Audit.} Building on drift-triggered JIT-SDP \citep{pote2026adaptive}, delayed feedback \citep{chapelle2014modeling}, and SZZ audits \citep{song2023waiting,bludau2026quantifying}, we formalize how right-censoring compresses posteriors and inverts rankings, quantify multi-year hindsight inflation, and audit observation horizons across 34 natural streams.
  \item \textbf{Operating Boundaries and Upstream Data Audit.} Across multi-year, shifted, and confirmatory cohorts, we map where frozen linear models, causal baselines, stream learners, and Survival-PU win, uncover an upstream Hadoop splitting artifact in ApacheJIT, separate cross-window score drift from ranking quality, and quantify review-budget yields.
  \item \textbf{Block-Calibrated Monitoring and Maturity-Aware PU Testbed.} We introduce circular block-bootstrap DriftLens to curb autocorrelation false alarms and audit Survival-PU as a controlled testbed, mapping identifiability limits, soft-target accuracy, and raw versus 90-day Platt calibration against post-hoc controls.
  \item \textbf{Open Replication Package and Verifiers.} We release a checksum-verified benchmark covering all streams, traces, prediction archives, Code-LLM logs, Online Resource~1, and automated verification scripts.
\end{enumerate}

% ============================================================================
% 2. Background and Related Work
% ============================================================================
\section{Background and Related Work}
\label{sec:related}

Our study connects commit-level defect prediction with drift monitors, verification-latency stream learning, and delayed-feedback positive-unlabeled modeling. Supplementary Tables~S-1 through~S-18 appear in companion Online Resource~1; Table~S-1 contrasts related paradigms by trigger, supervision, and operating boundaries.

\subsection{JIT Defect Prediction, Linear Baselines, and Drift Monitoring}
\label{sec:rel_jitsdp}

Classical JIT-SDP describes commits through change metrics capturing size, diffusion, history, and developer experience \citep{kamei2013large}, later extended with code-review signals \citep{mcintosh2018are}, AST or pull-request metadata \citep{bludau2022feature,bludau2022prszz}, graph fault localization \citep{zheng2026hgfe}, and line-level models \citep{pornprasit2021jitline,marrium2026flame}. Tree and neural architectures such as LightGBM \citep{ke2017lightgbm}, DeepJIT \citep{hoang2019deepjit}, and CodeBERT \citep{feng2020codebert} further broadened commit modeling. Even so, simple linear models remain formidable: \citet{zeng2021deep} showed that single-feature LApredict frequently rivals deep predictors, and \citet{tantithamthavorn2017mysterious} demonstrated the need for leak-free evaluation. Following that principle, our primary multimodal benchmark uses commit-time authoring metrics and tests post-push review fields separately in Table~S-4.

To handle concept drift without defect labels, \citet{pote2026adaptive} paired neural embeddings with DriftLens \citep{greco2021driftlens,greco2023driftlens}, monitoring Gaussian Fr\'echet distance \citep{dowson1982frechet} to trigger single-window retraining and four-window escalation after consecutive alarms. \citet{shah2026detecting} recently showed that embedding shifts do not reliably signal accuracy drops on their own. Neither line examines what happens after an alarm fires when defect labels remain months away, a gap our study isolates by contrasting offline Window Oracle retraining against strict causal deployment.

\subsection{Verification Latency, Stream Learners, and SZZ Bias}
\label{sec:rel_szz}

SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework} introduces permanent mislabeling \citep{rosa2021evaluating,fan2021impact,keshavarz2022apachejit,bludau2026quantifying} alongside multi-month verification latency \citep{cabral2019class,tabassam2020handling,song2023humla}. Although \citet{song2023waiting} audited how waiting windows distort offline evaluation labels, our audit shifts focus to live alarm-time retraining, where future fixes remain physically unobservable.

Three latency-aware stream learners anchor our continuous-learning baselines. ORB \citep{cabral2019class} pairs OzaBagging \citep{oza2005online} of Online Hoeffding Trees \citep{domingos2000mining} with a 90-day clean-commit wait; BORB \citep{cabral2023borb} refreshes ElasticNet models every 200 commits on matured pools; and ODaSC \citep{song2022odasc} combines a 15-day provisional-negative horizon with DenStream micro-cluster weights. All three rely on fixed waits or periodic tabular refreshes rather than immediate drift-triggered retraining across neural and tree hosts.

\subsection{Delayed Feedback and Positive-Unlabeled Maturity Modeling}
\label{sec:rel_pu}

\citet{chapelle2014modeling} formalized delayed feedback in online advertising by jointly modeling positive probability and conversion delay. In parallel, positive-unlabeled learning builds on propensity weighting \citep{elkan2008learning,bekker2020learning}, non-negative PU risk \citep{kiryo2017positive}, and density-ratio adjustment \citep{sugiyama2008direct}. Combining these ideas to estimate age-dependent commit maturity, we compare cohort incidence \citep{lyndenbell1971method}, Kaplan--Meier cure \citep{kaplan1958nonparametric}, and Weibull mixture cure \citep{boag1949cure,farewell1982cure} against fixed maturity curves and causal baselines. None of these studies separates how alarm-time right-censoring reshapes the Bayes-optimal predictor from how sprint autocorrelation destabilizes drift thresholds; we derive both mechanisms next.

% ============================================================================
% 3. Theoretical Analysis
% ============================================================================
\section{Theoretical Analysis of Right-Censored Retraining and Block Calibration}
\label{sec:theory}

Two statistical mechanisms govern live drift-triggered JIT-SDP. First, right-censoring reshapes the Bayes-optimal predictor differently depending on whether commit maturity is uniform or feature-dependent. Second, positive serial autocorrelation across commits inflates the variance of contiguous window means relative to independent draws. Complete algebraic proofs appear in Appendix~\ref{app:repro}.

\subsection{Posterior Compression and Rank Inversion Under Right-Censoring}
\label{sec:theory_poison}

Consider a chronological commit stream $\mathcal{S} = \{(x_i, t_i, y_i^*, t_i^{\text{fix}})\}_{i=1}^N$ with non-decreasing timestamps $t_1 \le \dots \le t_N$ in seconds. Each record pairs commit-authoring features $x_i \in \mathcal{X}$ with its archive-horizon defect label $y_i^* \in \{0, 1\}$ and earliest SZZ fix timestamp $t_i^{\text{fix}} \ge t_i$, set to $+\infty$ for clean or unlinked commits. The resulting fix delay in days is $\Delta_i = (t_i^{\text{fix}} - t_i)/86400$.

Two censoring horizons govern label observability at an alarm $t_{\text{alarm}}$. Alarm-time right-censoring hides defects fixed between $t_{\text{alarm}}$ and the archive horizon $T_{\text{end}}$ ($t_{\text{alarm}} < t_i^{\text{fix}} \le T_{\text{end}}$); end-of-archive censoring hides defects fixed after $T_{\text{end}}$. Because upstream corpora omit separate snapshot timestamps, we proxy $T_{\text{end}} = \max(\max_i t_i, \max_{i:\, t_i^{\text{fix}} < \infty} t_i^{\text{fix}})$ and audit remaining follow-up across all 34 streams in Section~\ref{sec:benchmarks} and Tables~S-8--S-9. Figure~\ref{fig:causal_timeline} contrasts offline Window Oracle retraining against strict causal deployment.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.68\textwidth]{fig0a_causal_timeline_v3.pdf}
  \caption{Causal supervision at drift alarm $t_{\text{alarm}}$. Panel~(a) shows Window Oracle leaking future SZZ fixes into $W_{\text{drift}}$; Panel~(b) contrasts delayed and naive causal retraining against maturity-weighted PU soft targets.}
  \label{fig:causal_timeline}
\end{figure}

\begin{definition}[Causal Label Visibility and Structural Maturity Factorization]
\label{def:observability}
At alarm timestamp $t_{\text{alarm}} = t_e$ closing a $W$-commit window $W_{\text{drift}} = \{e - W + 1, \dots, e\}$, causal deployment forbids inspecting events after $t_{\text{alarm}}$. For any commit $i \le e$ of age $a_i = (t_{\text{alarm}} - t_i)/86400 \ge 0$ days, the observable label is
\begin{equation}
  y_{i,\text{obs}}(t_{\text{alarm}}) = \mathbb{I}\!\left(y_i^* = 1 \;\wedge\; t_i^{\text{fix}} \le t_{\text{alarm}}\right) = y_i^* \cdot \mathbb{I}\!\left(\Delta_i \le a_i\right).
  \label{eq:y_obs}
\end{equation}
Locally on an adaptation window, assume latent defect risk is age-stationary given features, $\Pr(y_i^* = 1 \mid x_i, a_i) = \eta^*(x_i)$. Assume also that among true defects, reveal probability by age $a_i$ depends only on age, $\Pr(\Delta_i \le a_i \mid y_i^* = 1, x_i, a_i) = c(a_i) \in [0, 1]$, giving $\Pr(y_{i,\text{obs}} = 1 \mid x_i, a_i) = c(a_i)\,\eta^*(x_i)$.
\end{definition}

\begin{proposition}[Bayes-Optimal Censored Posterior Under Uniform vs.\ Heterogeneous Maturity]
\label{prop:bias}
Under Assumptions~(i)--(ii) of Definition~\ref{def:observability}, suppose predictor $f_\theta(x)$ minimizes class-weighted binary cross-entropy $\ell(p, y_{\text{obs}}) = -\omega_+ y_{\text{obs}} \log p - (1 - y_{\text{obs}})\log(1 - p)$ with positive weight $\omega_+ > 0$ over a window whose feature-age distribution induces conditional effective maturity $\bar{c}(x) = \mathbb{E}[c(a) \mid x] > 0$. For any $x$ with $\eta^*(x) \in (0, 1)$, the population minimizer is
\begin{equation}
  \hat{p}_{\text{obs}}(x) = \frac{\omega_+ \, \bar{c}(x) \, \eta^*(x)}{1 + \big(\omega_+ - 1\big)\,\bar{c}(x)\,\eta^*(x)}.
  \label{eq:b2_bayes}
\end{equation}
Equation~\eqref{eq:b2_bayes} implies two structural regimes:
\begin{enumerate}[leftmargin=1.8em,itemsep=0.03em,topsep=0.2em]
  \item \textbf{Uniform-Maturity Order Preservation and Threshold Crossing:} When $\bar{c}(x) \equiv \bar{c}_W \in (0, 1)$ is constant across features, $\partial \hat{p}_{\text{obs}}/\partial \eta^* = \omega_+ \bar{c}_W / [1 + (\omega_+ - 1)\bar{c}_W \eta^*]^2 > 0$, preserving population AP and ROC-AUC. Relative to the uncensored weighted minimizer $\hat{p}_{\text{ora}}(x)$ ($\bar{c} \equiv 1$), right-censoring compresses $\hat{p}_{\text{obs}}(x) < \hat{p}_{\text{ora}}(x)$ and forces $\hat{p}_{\text{obs}}(x) < \tau_{\text{th}}$ if and only if
  \begin{equation}
    \bar{c}_W < \frac{\tau_{\text{th}}}{\eta^*(x)\,\big[\omega_+(1 - \tau_{\text{th}}) + \tau_{\text{th}}\big]}.
    \label{eq:exact_crossing}
  \end{equation}
  \item \textbf{Heterogeneous-Maturity Rank Inversion:} Suppose subpopulations $x_{\text{old}}$ and $x_{\text{new}}$ have maturities $\bar{c}(x_{\text{old}}) = c_{\text{hi}} > c_{\text{lo}} = \bar{c}(x_{\text{new}}) > 0$ with $\eta^*(x_{\text{new}}) > \eta^*(x_{\text{old}})$. Censored training inverts their order, giving $\hat{p}_{\text{obs}}(x_{\text{new}}) < \hat{p}_{\text{obs}}(x_{\text{old}})$, if and only if
  \begin{equation*}
    \frac{\eta^*(x_{\text{new}})}{1 - c_{\text{lo}}\eta^*(x_{\text{new}})} < \frac{c_{\text{hi}}}{c_{\text{lo}}} \cdot \frac{\eta^*(x_{\text{old}})}{1 - c_{\text{hi}}\eta^*(x_{\text{old}})}.
  \end{equation*}
\end{enumerate}
\end{proposition}

Proposition~\ref{prop:bias} resolves two seemingly contradictory patterns that recur throughout our experiments. First, right-censoring does not automatically increase raw Brier error when class weighting already pushes uncensored posteriors upward. With $\omega_+ > 1$, downward censoring compression can partially cancel upward class-weight inflation and leave raw probabilities accidentally better centered than uncensored weighted scores.

Second, order preservation holds only inside a single window's model under uniform maturity. Once consecutive alarms shift estimated priors or class weights, concatenating raw scores across windows introduces inter-window scale drift even when within-window ranking stays intact. When a sudden concept shift brings new defect patterns in young commits, maturity becomes heterogeneous across features and censored training actively inverts their risk order. Proposition~\ref{prop:bias} thus predicts why causal multi-window retraining preserves ranking under gradual drift, loses thresholded recall under heavy lag, and falters when shifted features arrive in young commits.

\subsection{Window-Mean Variance Under Serial Autocorrelation}
\label{sec:theory_alarm_storm}

Every drift alarm exposes the predictor to right-censored labels, tying monitor calibration directly to retraining safety. Commits arrive in bursty sprints and coordinated refactorings rather than independent draws, which violates the i.i.d.\ assumption behind standard uniform bootstrap thresholds.

\begin{proposition}[Window-Mean Covariance Under Positive Lag Autocovariance]
\label{prop:autocorr}
Let $\{z_t\}_{t \in \mathbb{Z}} \subset \mathbb{R}^d$ be a weakly stationary embedding sequence with mean $\mu_0$, marginal covariance $\Gamma_0$, and lag-$k$ autocovariance $\Gamma_k = \operatorname{Cov}(z_t, z_{t+k})$ satisfying $\Delta_\Gamma \triangleq \sum_{k=1}^{W-1}\frac{W-k}{W}(\Gamma_k + \Gamma_k^\top) \succ 0$. Comparing the contiguous window mean $\bar{z}_{\text{contig}} = \frac{1}{W}\sum_{j=1}^W z_{t_0+j}$ against the mean $\bar{z}_{\text{ind}}$ of $W$ independent draws yields
\begin{equation*}
  \operatorname{Cov}(\bar{z}_{\text{contig}}) = \operatorname{Cov}(\bar{z}_{\text{ind}}) + \frac{1}{W}\Delta_\Gamma \succ \operatorname{Cov}(\bar{z}_{\text{ind}}), \qquad
  \mathbb{E}\|\bar{z}_{\text{contig}} - \mu_0\|_2^2 > \mathbb{E}\|\bar{z}_{\text{ind}} - \mu_0\|_2^2.
\end{equation*}
\end{proposition}

Uniform resampling destroys positive lag correlation across development sprints and therefore underestimates the true null spread of contiguous commit windows \citep{kunsch1989jackknife,politis1992circular}. Proposition~\ref{prop:autocorr} isolates the mean-shift term of the Fr\'echet distance rather than its covariance term, and the same serial dependence motivates replacing uniform draws with circular block resampling. Combined with updating the reference window after confirmed alarms, block calibration limits false-alarm storms and preserves sensitivity to genuine shifts, a design we isolate in Section~\ref{sec:rq4}.

% ============================================================================
% 4. Block-Calibrated Monitoring and the Survival-PU Testbed
% ============================================================================
\section{Block-Calibrated Monitoring and the Survival-PU Testbed}
\label{sec:method}

Propositions~\ref{prop:bias} and~\ref{prop:autocorr} define two linked engineering requirements for live drift adaptation: stabilize the drift trigger against sprint autocorrelation, and test how far explicit right-censoring correction can push causal accuracy without peeking past $t_{\text{alarm}}$. At the monitoring layer, we preserve within-sprint correlation during DriftLens calibration by resampling circular contiguous commit blocks rather than i.i.d.\ draws. After each alarm, DriftLens refreshes its reference window on trailing commits and exponentially blends the updated threshold with its prior value (Table~\ref{tab:hyperparams}), curbing repeated triggers across the same regime shift.

At the adaptation layer, rather than assuming explicit maturity modeling dominates every setting, we build Survival-PU as a controlled censoring-aware testbed to measure where three-stage soft-target distillation genuinely beats simpler causal baselines, fixed maturity curves, and post-hoc controls. Using only causal history $t_i \le t_{\text{alarm}}$, it estimates commit-age maturity $c(a_i)$ without right-truncation bias, infers target prevalence $\hat{\pi}_{\text{target}}$ alongside a reliable teacher, and distills calibrated soft targets across the combined anchor and drift pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ (Figure~\ref{fig:architecture}). All 33 locked protocol hyperparameters appear in Table~\ref{tab:hyperparams}, exact host architectures appear in Table~S-2, and a three-state confidence-gated variant is evaluated solely as an ablation.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\textwidth]{fig0b_survival_pu_architecture_v3.pdf}
  \caption{Block-calibrated DriftLens monitor and three-stage Survival-PU testbed triggered at drift alarm $t_{\text{alarm}}$. Stage~1 estimates commit maturity on trailing history, Stage~2 fits a reliable teacher and target prevalence, and Stage~3 distills soft targets across $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$.}
  \label{fig:architecture}
\end{figure}

\subsection{Stage 1: Risk-Eligible Cohort Incidence and Survival Controls}
\label{sec:stage1}

To estimate how quickly defects surface without undercounting slow fixes, Stage~1 constructs a survival-style maturity curve over recent commits. Rather than taking a naive empirical CDF over already fixed bugs, it asks what share of commits old enough to reach each age milestone have surfaced a defect by that age.

Using only the trailing $N_{\text{hist}}$ causal commits $\mathcal{H}_{\text{rec}} \subset \{i : t_i \le t_{\text{alarm}}\}$, Stage~1 records commit age $a_i \ge 0$ in days at $t_{\text{alarm}}$ and fix delay $d_i \ge 0$ for revealed defects (setting $d_i = +\infty$ otherwise). Along age knots $g_0 = 0 < g_1 < \dots < g_M = a_{\max}$, it computes cumulative incidence over age-eligible commits $\mathcal{E}(g_m) = \{i \in \mathcal{H}_{\text{rec}} : a_i \ge g_m\}$, carrying forward $\hat{G}(g_{m-1})$ below the support floor (Table~\ref{tab:hyperparams}) and enforcing monotonicity:
\begin{equation}
  \hat{G}(g_m) = \frac{1}{|\mathcal{E}(g_m)|}\sum_{i \in \mathcal{E}(g_m)} \mathbb{I}\!\left(y_{i,\text{obs}} = 1 \;\wedge\; d_i \le g_m\right), \qquad \tilde{G}(g_m) = \max_{0 \le k \le m}\hat{G}(g_k),
  \label{eq:cohort_inc}
\end{equation}
so $\tilde{G}(g_m) \approx \pi\, c(g_m)$. Because defects fixing after $a_{\max}$ remain unobserved, Stage~1 damps tail completeness using second-half growth $r_g = \tilde{G}(a_{\max}) / \tilde{G}_{\text{half}}$ and horizon factor $\phi_a$:
\begin{equation}
  \begin{aligned}
    \phi_a &= \mathrm{clip}\!\Big(\frac{a_{\max}}{a_{\max} + a_0},\; \phi_{\min},\; 1\Big), \\
    \gamma_{\text{tail}} &= \mathrm{clip}\!\Big(\big[\gamma_{\max} - \lambda_g\,\mathrm{clip}(r_g - 1, 0, 1)\big]\phi_a,\; \gamma_{\min},\; \gamma_{\max}\Big),
  \end{aligned}
  \label{eq:tail_cap}
\end{equation}
where $(a_0, \phi_{\min}, \lambda_g)$ are pre-locked damping constants documented in Appendix~\ref{app:repro}. Normalizing $\tilde{G}(g_m)$ by $\tilde{G}(a_{\max}) / \gamma_{\text{tail}}$ cancels $\pi$ and yields clipped knot maturities:
\begin{equation}
  c_{\text{knot}}(g_m) = \mathrm{clip}\!\Big(\gamma_{\text{tail}}\,\frac{\tilde{G}(g_m)}{\tilde{G}(a_{\max})},\; c_{\min},\; c_{\max}\Big),
  \label{eq:ci_interp}
\end{equation}
from which each commit's maturity $c_i$ is interpolated piecewise-linearly and clipped to $[c_{\min}, c_{\max}]$. Under sparse trailing support, Stage~1 reverts deterministically to the smooth 90-day half-life fallback curve in Table~\ref{tab:hyperparams}.

This normalization assumes locally stationary prevalence and fix latency across $\mathcal{H}_{\text{rec}}$; abrupt shifts can temporarily confound prevalence with reveal speed, as audited in Section~\ref{sec:rq3}. To isolate whether learned curves outperform simpler priors, we test three drop-in Stage~1 controls: quantized Kaplan--Meier Cure, unstratified Weibull Mixture-Cure MLE \citep{boag1949cure,farewell1982cure}, and parameter-free FixedCurve-PU using a fixed 90-day half-life curve clipped to $[c_{\min}, c_{\max}]$.

\subsection{Stage 2: Clipped Inverse-Maturity Prevalence and Reliable Teacher}
\label{sec:stage2}

Before distilling soft targets, Stage~2 answers two linked questions: what latent defect rate the drifted window is hiding, and which past commits are trustworthy enough to train a teacher. Intuitively, it reweights historical anchors toward the incoming window, inflates observed fixes by commit maturity without letting tiny maturities explode variance, and trains a teacher solely on confirmed defects plus sufficiently aged clean commits.

Formally, on the combined anchor and drift pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$, an $L_2$-logistic domain classifier estimates bounded density-ratio weights $w_{\text{imp}}(x_i) \in [w_{\min}, w_{\max}]$ \citep{sugiyama2008direct}. Combining $w_{\text{imp}}(x_i)$ with anchor recency and drift emphasis $\gamma_{\text{drift}}$ yields base weights $w_i$, and label-independent weights $u_i = w_i[u_0 + (1 - u_0)c_i]$ downweight young commits without skewing class balance. On a trailing calibration slice $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$, Stage~2 floors maturities at $\tilde{c}_i = \max(c_i, c_{\text{ht}})$ with $c_{\text{ht}} = 0.10$ and shrinks the inverse-maturity prevalence toward a mature anchor prior $\pi_{\text{anc}}$ (Table~\ref{tab:hyperparams}):
\begin{equation}
  \begin{aligned}
    \hat{\pi}_{\text{IP}} &= \frac{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\,(y_{i,\text{obs}} / \tilde{c}_i)}{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i}, \qquad n_{\text{eff,heur}} = \frac{\big(\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\big)^2}{\sum_{i \in \mathcal{B}_{\text{cal}}} (u_i / \tilde{c}_i)^2}, \\
    \hat{\pi}_{\text{target}} &= \operatorname{clip}\!\left(\max\!\left(\frac{n_{\text{eff,heur}}\,\hat{\pi}_{\text{IP}} + m_0\,\pi_{\text{anc}}}{n_{\text{eff,heur}} + m_0},\; \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4}\right),\; \pi_{\min},\; \pi_{\max}\right).
  \end{aligned}
  \label{eq:ht_prev}
\end{equation}

Stage~2 then fits teacher $T_\phi$ on the reliable cohort $\mathcal{R} = \{i : y_{i,\text{obs}} = 1 \vee c_i \ge \tau_{\text{rel}}\}$ with sample weights $w_{\text{rel},i} = w_{\text{imp}}(x_i) \cdot r_{\text{conf},i} \cdot \text{recency}_i$, where $r_{\text{conf},i} = \rho_0 + (1 - \rho_0)c_i$ for $c_i < \tau_{\text{rel}}$ and $1$ otherwise, and blended positive class weight:
\begin{equation}
  \omega_{+,\text{teach}} = \tfrac{1}{2}\omega_+ + \tfrac{1}{2}\operatorname{clip}\!\left(\frac{1 - \hat{\pi}_{\text{target}}}{\max(\hat{\pi}_{\text{target}}, \epsilon_\pi)},\; \omega_{\min},\; \max(\omega_{\max}, \omega_+)\right),
  \label{eq:teacher_weights}
\end{equation}
with pre-locked scaling bounds $(\rho_0, \epsilon_\pi, \omega_{\min}, \omega_{\max})$ listed in Appendix~\ref{app:repro}. Neural teachers warm-start from a blend of the current checkpoint and pre-stream anchor for $E_{\text{teach}}$ epochs; LightGBM fits a fresh $R_{\text{alm}}$-tree booster. Both emit temperature-scaled, class-weight-corrected log-odds $\eta_i^{(0)} = \tau_{\text{sharp}}(z_{\text{teach}}(x_i) - \log \omega_{+,\text{teach}})$.

\subsection{Stage 3: Soft-Target Distillation and Host-Fair Integration}
\label{sec:stage3}

Stage~3 converts the teacher's risk ranking and Stage~1's maturity estimates into soft labels across $\mathcal{B}$, letting the student adapt on drifted commits without mistaking young unfixed changes for clean examples. Already revealed defects stay pinned at $q_i(\delta) = 1$, while each unrevealed commit ($y_{i,\text{obs}} = 0$) receives a maturity-adjusted soft target shifted by one global logit offset $\delta$:
\begin{equation}
  q_i(\delta) = \frac{\tilde{p}_i(\delta)\,(1 - c_i)}{1 - \tilde{p}_i(\delta)\,c_i}, \qquad \tilde{p}_i(\delta) = \operatorname{clip}\!\big(\sigma(\eta_i^{(0)} + \delta),\; p_{\min},\; p_{\max}\big).
  \label{eq:bayes_pu}
\end{equation}
Intuitively, this Bayes-inspired update pushes $q_i$ toward zero on aged unfixed commits and preserves teacher risk on young commits. To align mean target prevalence on $\mathcal{B}_{\text{cal}}$ with $\hat{\pi}_{\text{target}}$, Stage~3 runs $K_{\text{bis}}$-step monotone bisection over $\delta \in [-8, 8]$, clamping cleanly when observed prevalence exceeds $\pi_{\max}$.

The student minimizes weighted soft-target cross-entropy $\mathcal{L}_{\text{SPU}}(z) = \sum_{i \in \mathcal{B}} u_i [\omega_{+,\text{teach}}\, q_i(\delta^*) \log(1 + e^{-z_i}) + (1 - q_i(\delta^*))\log(1 + e^{z_i})]$. Neural students fine-tune from $T_\phi$ for $E_{\text{stud}}$ epochs; LightGBM fits a fresh $R_{\text{alm}}$-tree booster so teacher and student together match the $2R_{\text{alm}}$ tree budget of single-stage tree baselines.

Two host mechanisms ensure fair execution across model families. To prevent multimodal neural models from underfitting tabular metrics on small drift windows, they retain a frozen pre-stream linear residual $z(x) = w_{\text{LR}}^\top x_{\text{tab}} + b_{\text{LR}} + f_\theta(x)$ \citep{zeng2021deep}, favored over a trainable skip during pre-stream validation. For LightGBM, exact analytical gradients and Hessians $(g_i, h_i)$ with matching initial margin $z_{\text{init}}$ optimize $\mathcal{L}_{\text{SPU}}$ directly (Proposition~\ref{prop:lgbm}, Appendix~\ref{app:repro}) \citep{friedman2001greedy,ke2017lightgbm}.

% ============================================================================
% 5. Experimental Design & Evaluation Protocol
% ============================================================================
\section{Experimental Design and Evaluation Protocol}
\label{sec:setup}

We organize our evaluation into three parts: audited commit cohorts, controlled learner tracks, and causal endpoints with formal inference. All Benchmark~A and~B hypotheses and comparison families were locked before downloading the confirmatory ApacheJIT cohort, with one small-sample split rule added for its smallest repositories.

\subsection{Audited Stream Cohorts, Data Funnels, and Archive Follow-Up}
\label{sec:benchmarks}

Our benchmark totals 347,122 commits across 34 natural repository streams: two multimodal Benchmark~A streams, 27 multi-ecosystem Benchmark~B streams, and five held-out Phase~C repositories, plus eight controlled shifts and a 6,400-call Code-LLM case study.

\paragraph{Benchmark A: Multimodal Qt and OpenStack.}
Benchmark~A evaluates Attention-CNN \citep{pote2026adaptive}, CodeBERT \citep{feng2020codebert}, DeepJIT-CNN \citep{hoang2019deepjit}, and LightGBM \citep{ke2017lightgbm} on 25,704 Qt and 13,304 OpenStack commits cross-linked from DeepJIT, \citet{mcintosh2018are}, and Git histories, alongside four controlled cross-subsystem shifts. High-confidence SZZ links cover 96.5\% of Qt and 99.3\% of OpenStack defects; heuristic and unlinked defects receive infinite discovery times. Primary streams model merged-commit replay on the 13 commit-time authoring metrics because exports expose only authoring timestamps, with 28-metric pre-merge triage evaluated in Table~S-4.

\paragraph{Benchmark B: Natural Donor Clusters and Controlled Shifts.}
Benchmark~B evaluates a tabular multilayer perceptron (MLP) and LightGBM across 300,637 commits from 14 CommitGuru \citep{cabral2019class}, eight ApacheJIT \citep{keshavarz2022apachejit}, and five PR-SZZ streams \citep{bludau2026quantifying}. Merging shared Camel and OpenStack repositories yields 25 independent donor clusters alongside a 24-cluster Benchmark-A-disjoint subset. To stress-test abrupt shifts outside cluster inference, Benchmark~B adds four concatenated pairs (Nova-to-Zulip, Ignite-to-Cassandra, Airflow-to-Pulsar, and OpenStack Nova-to-Neutron), whose one-hour splices push 23.7\% to 60.0\% of pre-onset fixes past onset (Table~S-6). Deep ablations and stress tests in RQ3, RQ4a--b, and RQ4d use a fixed ten-stream subset pairing seven natural streams with the first three shifts over five seeds.

\paragraph{Phase C: Confirmatory ApacheJIT Cohort and Monorepo Disclosure.}
All code stayed frozen before downloading the upstream ApacheJIT archive \citep{keshavarz2022apachejit}. Applying Benchmark~B's chronological cutoff produced 7,477 commits across five held-out repositories with at least one year of follow-up on MLP and LightGBM. Upstream auditing revealed that 11,964 clean Hadoop commits reside under the umbrella Hadoop key rather than subproject keys, inflating prevalence in HDFS* and MapReduce* to 89.2\% and 78.8\% in Table~S-7. Spark, Zeppelin, and ZooKeeper suffer no split and retain 44.6\% to 48.5\% prevalence, so we foreground those three standalone repositories alongside the four-project umbrella record merging HDFS* and MapReduce* under Hadoop. For confirmatory repositories under 4,000 commits, one small-sample rule scales initialization and validation slices to 25\% and 8\% of stream length rounded to multiples of 25, bounded at 150--1,500 and 50--500 commits.

\paragraph{Observable Archive-Horizon Follow-Up Audit.}
Nearly every Benchmark~B and Phase~C commit retains ten to twelve months of follow-up, whereas Benchmark~A runs to $T_{\text{end}}$ (Table~S-8). All CommitGuru and Phase~C commits, over 99.3\% of Benchmark~B ApacheJIT commits, and four of five PR-SZZ streams retain nearly one year; natural Qt leaves 15.5\% under half a year and 41.0\% under one year, OpenStack leaves 30.4\% under one year, and Table~S-9 confirms stability at both horizons.

\subsection{Two-Track Learners and Baseline Controls}
\label{sec:baselines}

We evaluate 29 methods across 41 prefix-invariant configurations in two tracks (constructors in Table~S-2). \textbf{Track~1} holds host architecture and DriftLens triggers fixed to isolate alarm-time supervision across hindsight references, causal replay, waits, PU learners, teachers, and post-hoc shifts. \textbf{Track~2} benchmarks external paradigms: unsupervised size ranking, linear models \citep{zeng2021deep}, ElasticNet BORB \citep{cabral2023borb}, and online River Hoeffding ensembles for ORB \citep{cabral2019class} and ODaSC \citep{song2022odasc} consuming strict causal events $t_{\text{event}} < t_{\text{commit}}$. The matrix below indexes core methods in Section~\ref{sec:results}.

\par\smallskip\noindent
\begin{minipage}{\textwidth}
\fontsize{7.5}{8.7}\selectfont
\setlength{\tabcolsep}{1.4pt}
\renewcommand{\arraystretch}{0.99}
\begin{tabularx}{\textwidth}{L{2.88cm}C{0.42cm}L{1.28cm}Y}
\toprule
\textbf{Method / Family} & \textbf{Tr.} & \textbf{Trigger} & \textbf{Alarm-Time Supervision \& Core Role} \\
\midrule
Frozen Host & T1 & None & Pre-stream prefix only; tests whether updating helps. \\
Window / Full Oracle & T1 & DriftLens & Hindsight $y^*$ ($4W$/full prefix); measures oracle bias. \\
Causal-1W / 4W & T1 & DriftLens & Observable $y_{\text{obs}}(t_{\text{alarm}})$ on $1W$ or escalated $4W$. \\
Wait-30 / 60 / 90 & T1 & DriftLens & Drops unfixed commits younger than 30/60/90 days. \\
nnPU / ODaSC-Host & T1 & DriftLens & Age-blind nnPU; host-ported 15d provisional-0 weights. \\
FixedCurve-PU & T1 & DriftLens & Survival-PU with Stage~1 fixed to a 90d half-life curve. \\
Teacher-Ex. / Match & T1 & DriftLens & Stage~2 reliable teacher alone ($1\times$ or $2\times$ budget). \\
Survival-PU (+Gate) & T1 & DriftLens & Full 3-stage maturity PU distillation (optional gate). \\
PostHoc Logit/Ratio & T1 & DriftLens & Frozen Causal-4W with logit shift or ratio rescaling. \\
LOC / LApred. / LR & T2 & None/DL & Unsupervised LOC; frozen or $4W$ linear baselines. \\
BORB-Nat. / Alarm & T2 & 200c/DL & ElasticNet SGD on fixes + $\ge 90$d clean commits. \\
ORB / ODaSC & T2 & Per-event & Hoeffding ensembles (90d wait or 15d DenStream). \\
\bottomrule
\addlinespace[2pt]
\multicolumn{4}{@{}p{\textwidth}@{}}{\scriptsize Key: T1/T2 denote Track~1/2, DL denotes DriftLens, and LR denotes logistic regression.} \\
\end{tabularx}
\end{minipage}
\par\smallskip

At the monitor layer, RQ4b crosses IID vs.\ Circular Block resampling against Static vs.\ Rolling refresh at fixed PCA dimension, 95th-percentile threshold, and unit cooldown, compared with Pote et al.'s static 98th-percentile baseline, closed-loop refresh, and RBF-MMD and PCA-KS.

\subsection{Endpoints, Causal Calibration, and Statistical Protocol}
\label{sec:metrics_protocol}

\paragraph{Endpoints and Three-Branch Causal Calibration.}
Whole-stream Average Precision (AP) serves as our primary ranking endpoint, contrasted with per-window AP in Section~\ref{sec:rq2}, alongside ROC-AUC, fixed $F_1$, tuned $F_{1,\text{th}}$, Recall@20\%Effort, raw and post-Platt Brier score, expected calibration error (ECE), and slope. Every branch passed deterministic prefix-invariance checks with zero perturbation error.

At each evaluation window $w = [s_w, e_w)$ with cutoff $t_w = t_{s_w - 1}$, our calibrator selects its pool $\mathcal{M}_w$ via three deterministic branches. First, it draws up to 1,200 prior commits aged $\ge 90$ days whenever $\ge 80$ qualify with both observed classes. Otherwise, it falls back to the causal initialization slice when that slice contains both classes. Finally, if neither holds, it applies a deterministic prior-odds shift at threshold 0.50 without Platt fitting. Under branches one or two, it fits 1D $L_2$-Platt scaling \citep{platt1999probabilistic}, selects $\tau_{\text{th}}(w)$ over a sixty-point grid on $\hat{p}_j = \sigma(z_j)$, and evaluates post-Platt Brier and ECE against archive labels $y_i^*$:
{\small\begin{equation}
  \hat{p}_{i,\text{cal}} = \sigma(a_w z_i + b_w), \quad
  \tau_{\text{th}}(w) = \textstyle\arg\max_{\tau \in \mathcal{G}_w} F_1\!\big(\{\mathbb{I}(\hat{p}_j \ge \tau),\, y_{j,\text{obs}}(t_w)\}_{j \in \mathcal{M}_w}\big).
  \label{eq:causal_cal}
\end{equation}}

\paragraph{Cluster Aggregation and Multiplicity Control.}
Metrics average across ten seeds on Benchmark~B/C MLP and five on LightGBM and multimodal hosts, weighting every project cluster equally. Formal $\alpha = 0.05$ inference on all 25 clusters and the 24-cluster Benchmark-A-disjoint subset uses exact two-sided paired Wilcoxon tests under Pratt ties with Holm--Bonferroni adjustment \citep{holm1979simple}, bootstrap 95\% intervals, and Cohen's $d_z$. Family~1 contrasts Causal-4W against Window Oracle, Frozen Host, and Survival-PU; Family~2 tests Survival-PU against cross-track baselines; Family~3 ablates Survival-PU against FixedCurve-PU, Teacher-Matched, Teacher-Exact, and gating. Non-significant contrasts stay inconclusive, checks outside Table~\ref{tab:bench_b_holm} are unadjusted, differences use unrounded means, and bold marks the best non-oracle deployable value (or $p_{\text{Holm}} < 0.05$ in Table~\ref{tab:bench_b_holm}).

% ============================================================================
% 6. Empirical Results
% ============================================================================
\section{Empirical Results}
\label{sec:results}

We present our empirical results across the five research questions, moving from the causal right-censoring audit (RQ1) and cross-paradigm operating boundaries (RQ2) to internal mechanism ablations (RQ3), stress tests and calibration hazards (RQ4), and runtime overhead and code-review yields (RQ5).

% ----------------------------------------------------------------------------
% 6.1 RQ1
% ----------------------------------------------------------------------------
\subsection{RQ1: Causal Audit of Right-Censoring and Hindsight Optimism}
\label{sec:rq1}

\paragraph{Severe Right-Censoring Across Multi-Year Cohorts.}
Most newly introduced defects remain unfixed whenever a drift alarm triggers retraining in multi-year ecosystems (Table~\ref{tab:audit_censoring}, Figure~\ref{fig:rq1_overview}a). Median SZZ fix delays span three to nine months across these cohorts. As a result, 68.8\% to 94.6\% of recent defects stay unrevealed at scheduled single-window boundaries, peaking in natural Qt, and fired DriftLens alarms track those rates closely.

Widening replay to four windows offers only partial relief, still leaving 45.1\% to 80.5\% of recent defects hidden across multi-year projects. By contrast, the shorter confirmatory ApacheJIT cohort sees far lighter single-window censoring, at 26.4\% overall and 18.0\% across its three standalone repositories. Project cadence and archive horizon therefore directly govern alarm-time label visibility.

[[TAB_AUDIT_CENSORING]]

\paragraph{Matched Hindsight Optimism Versus Causal Replay.}
Leaking future SZZ fixes into drift-triggered updates significantly inflates whole-stream ranking accuracy across the 25 natural donor clusters (Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}). On MLP, Window Oracle exceeds matched Causal-4W by 2.4 AP points, winning 20 of 25 clusters after Holm--Bonferroni correction. On LightGBM, hindsight inflation doubles to 4.8 AP points, winning 23 of 25 clusters. Restricting causal replay to a single window depresses accuracy further on both hosts because almost every defect in that window remains unfixed at alarm time.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig1_multimodal_audit_v3.pdf}
  \caption{Alarm-time right-censoring and cross-host baseline performance. Panel~(a) reports unrevealed defect shares across scheduled single- and four-window boundaries; Panel~(b) previews six-stream mean AP across Benchmark~A hosts relative to frozen Full-LR.}
  \label{fig:rq1_overview}
\end{figure}

\paragraph{Why Censoring Strikes Differently Across Metrics and Regimes.}
Hindsight inflation does not mean naive causal retraining fails uniformly. Consistent with Proposition~\ref{prop:bias}, older commits inside a four-window buffer retain enough partial maturity for Causal-4W to outperform a frozen MLP by 4.4 AP points under gradual natural drift (Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}). Across the four controlled abrupt shifts, it similarly beats freezing by roughly 5 AP points.

This partial recovery still hides two structural vulnerabilities predicted by Proposition~\ref{prop:bias}. Because unrevealed defects suppress predicted risk toward zero, fixed-threshold recall and $F_1$ collapse on multimodal hosts whenever base rates drift or latency deepens. Worse, shrinking the buffer to a single window removes those matured older commits altogether, pushing Causal-1W below a frozen model after abrupt shifts. These failure modes set up RQ2, where we map how far censoring-aware adaptation expands reliable operating boundaries across hosts and drift regimes, previewed in Figure~\ref{fig:rq1_overview}b.

\rqbox{Summary for RQ1}{Across scheduled single-window boundaries in multi-year ecosystems, 68.8\% to 94.6\% of recent defects remain right-censored, with fired alarms tracking scheduled rates closely. Leaking future fixes significantly inflates whole-stream AP across 25 natural clusters by 2.4 points on MLP and 4.8 points on LightGBM. Four-window causal replay still beats freezing in AP under gradual drift, but it trails matched oracles and suffers severe fixed-threshold $F_1$ loss on multimodal hosts.}

% ----------------------------------------------------------------------------
% 6.2 RQ2
% ----------------------------------------------------------------------------
\subsection{RQ2: Multi-Host, Cross-Project, and Confirmatory Operating Boundaries}
\label{sec:rq2}

\paragraph{Benchmark A: Linear Predictors Lead Deployable Multimodal Hosts.}
Simple frozen linear models outperform every deployable neural and tree host on multimodal Benchmark~A, as shown in Table~\ref{tab:bench_a_main} and Figure~\ref{fig:rq1_overview}b, with full per-host breakdowns in Table~S-3. Frozen full-feature logistic regression (Full-LR Frozen) edges past single-feature LApredict and beats the strongest deployable neural host, frozen DeepJIT-CNN, by more than a full AP point; only non-deployable Full Hindsight DeepJIT-CNN scores higher. Even minimal Churn-LR Frozen beats the four-host average of every window-adapted strategy, and unsupervised LOC-Only outscores Survival-PU by over three points in effort recall.

[[TAB_BENCH_A_MAIN]]

\paragraph{Architectural Skip Gains and Sensitivity Robustness.}
Adding our frozen linear skip narrows the neural deficit substantially, lifting frozen AP across the six multimodal streams by roughly 8 to 17 points for Attention-CNN, CodeBERT, and DeepJIT-CNN in Figure~\ref{fig:rq2_skip}b, alongside the rolling CodeBERT trajectory on Qt in Figure~\ref{fig:rq2_skip}a. Even with that linear anchor, fine-tuning high-capacity encoders on small drift windows still trails frozen linear models in AP. Within identical Track-1 hosts, base-rate drift and right-censoring erode fixed-threshold $F_1$ sharply on Frozen Host and Causal-4W; by contrast, Survival-PU recovers the highest Track-1 deployable $F_1$ and $F_{1,\text{th}}$ at lower mean AP (Table~\ref{tab:bench_a_main}).

These Benchmark~A boundaries persist under alternate feature sets and longer maturity windows. Across provenance and 28-metric pre-merge variants, Track-1 deployable AP splits by project, with Causal-4W leading on Qt and Frozen Host leading on OpenStack in Table~S-4. Restricting evaluation to commits with 180 or 365 days of archive follow-up likewise preserves Full-LR's AP lead and Survival-PU's Track-1 $F_1$ gain across every slice, with only minor pairwise AP swaps at one year in Table~S-9.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig2_trajectories_and_skip_v3.pdf}
  \caption{Panel~(a) traces prequential three-window rolling AP on natural Qt with CodeBERT; Panel~(b) compares multimodal hosts with and without the frozen linear skip.}
  \label{fig:rq2_skip}
\end{figure}

\paragraph{Benchmark B: Gradual Natural Donor Clusters.}
On gradual multi-year streams, simpler causal baselines match or outperform three-stage survival correction in whole-stream ranking across Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm} and Figure~\ref{fig:rq2_bench_b}a, with complete metrics in Table~S-5. Across the 25 donor clusters, Full-LR Adapt-4W achieves the highest mean MLP AP, with Causal-4W, Wait-90, and zero-fit FixedCurve-PU all finishing ahead of Survival-PU.

Holm--Bonferroni contrasts reveal a sharp two-sided boundary on whole-stream AP (Table~\ref{tab:bench_b_holm}). Survival-PU significantly beats age-blind Pure-nnPU by 4.1 AP points and Hoeffding ORB by 4.0 points, showing no adjusted gap against Frozen Host, ODaSC, BORB, or Full-LR Adapt-4W. However, \textbf{Survival-PU significantly trails Causal-4W by 1.4 AP points and Wait-90 by 1.2 points}.

[[TAB_BENCH_B_CLUSTERS]]

\paragraph{Component Controls and Secondary Metrics.}
Component controls show that adaptive Stage~1 estimation adds no whole-stream AP advantage over a fixed 90-day curve on gradual streams (Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}). On primary MLP AP, \textbf{zero-fit FixedCurve-PU significantly exceeds cohort-estimated Survival-PU by 0.8 points}, with the same direction holding on LightGBM and the 24-cluster disjoint check. Stage~3 distillation is host-dependent. On MLP, it yields no significant AP gain over Teacher-Matched. On LightGBM, it lifts AP over Teacher-Matched by 3.8 points. Even on LightGBM, Causal-4W still leads deployable AP overall, and confidence gating changes AP negligibly.

Thresholded detection and probability calibration diverge sharply from whole-stream AP (Table~\ref{tab:bench_b_clusters}). Online stream learners, specifically ElasticNet BORB and Hoeffding ODaSC, deliver the strongest causally tuned $F_{1,\text{th}}$ overall. Within Track~1, however, survival-aware weighting pays off: Survival-PU leads all Track-1 hosts in $F_{1,\text{th}}$ and ties FixedCurve-PU for the lowest post-Platt Brier loss, ahead of Full-LR Adapt-4W and Causal-4W; raw versus post-Platt calibration divergence is dissected in RQ4a.

[[TAB_BENCH_B_HOLM]]

\paragraph{Evaluation Granularity and Cross-Window Score Comparability.}
Inter-window score-scale drift accounts for a major share of Survival-PU's whole-stream ranking deficit against Causal-4W (Table~\ref{tab:window_ap_scale}). Scored within individual 400-commit operational windows rather than pooled across years, the sign against Causal-4W flips: per-window AP favors Survival-PU by half a point, closing most of the gap to FixedCurve-PU, Wait-90, and Full-LR Adapt-4W.

Applying monotone within-window z-score or percentile-rank standardization to the \textit{exact same predictions} before computing whole-stream AP reproduces that reversal. Standardization turns Survival-PU's deficit against Causal-4W into a small exploratory lead and shrinks gaps to FixedCurve-PU and Wait-90 close to zero. This pattern directly echoes Proposition~\ref{prop:bias}. Because Survival-PU recalibrates its target prior at every alarm, its score scale drifts smoothly across windows even when within-sprint commit ranking stays competitive, so pooling raw multi-year scores penalizes it. Removing that scale shift still does not lift Survival-PU above Wait-90, FixedCurve-PU, or Full-LR Adapt-4W.

[[TAB_WINDOW_AP_SCALE]]

\paragraph{Controlled Concatenated Distribution Shifts.}
Under abrupt cross-project or cross-subsystem regime changes, method rankings invert sharply with recovery horizon (Table~\ref{tab:bench_b_clusters}, Figure~\ref{fig:rq2_bench_b}b, Table~S-6). Over the full post-onset horizon, \textbf{Survival-PU leads every deployable learner}, edging past ODaSC-Host and beating Causal-4W by 1.9 AP points. By contrast, single-window causal replay, age-blind Pure-nnPU, and ORB drop below Frozen Host because newly shifted defects arrive young and unrevealed.

Over the immediate first 1,200 post-onset commits, however, lightweight linear adaptation responds faster than three-stage neural retraining. ElasticNet BORB-Alarm adapts fastest, outscoring Survival-PU by 3.4 AP points, with native BORB 1.4 points ahead. Recovery difficulty also varies by domain: OpenStack Nova-to-Neutron yields lower post-onset AP across all adaptive methods than the three cross-project splices (Table~S-6).

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig3_benchmark_b_clusters_v3.pdf}
  \caption{Panel~(a) compares mean AP with standard errors across 25 natural Benchmark~B clusters; Panel~(b) reports full and first-1,200-commit post-shift AP across the four controlled shifts.}
  \label{fig:rq2_bench_b}
\end{figure}

\paragraph{Phase C: Confirmatory Non-Replication on MLP.}
On the shorter held-out ApacheJIT cohort, neural hindsight optimism fails to replicate and online Hoeffding-tree ensembles dominate (Table~\ref{tab:confirmatory}, Figure~\ref{fig:rq2_confirmatory}). Because upstream Hadoop monorepo splitting inflates base rates in HDFS* and MapReduce* (Table~S-7), the three standalone repositories (Spark, Zeppelin, and ZooKeeper) provide the cleaner external test. Across those standalone repositories, \textbf{Window Oracle trails Causal-4W slightly} and lags Frozen Host by roughly six AP points. Hoeffding ODaSC leads every window-retrained MLP variant by more than ten points, followed by ElasticNet BORB and Hoeffding ORB.

[[TAB_CONFIRMATORY]]

\paragraph{Four-Umbrella Aggregation and Horizon Covariation.}
Aggregating Hadoop subprojects into the four-umbrella view confirms the identical ordering across all four project ecosystems (Table~\ref{tab:confirmatory}, Figure~\ref{fig:rq2_confirmatory}a). Window Oracle trails Causal-4W by half a point and lags Frozen Host by more than four points, and ODaSC again leads the cohort ahead of BORB and ORB. On LightGBM four-umbrella AP, Window Oracle edges past Causal-4W without reaching statistical significance, and Frozen Host leads both across the standalone repositories.

Evaluation horizon cleanly separates the two cohorts (Figure~\ref{fig:rq2_confirmatory}b): every shorter confirmatory repository falls at or below zero hindsight inflation, in stark contrast to 22 of 27 multi-year Benchmark~B streams showing positive oracle inflation. Because shorter horizons, higher baseline prevalence, lighter censoring, and project identity shift together on Phase~C, isolating which single driver explains both the vanishing oracle inflation and ODaSC's dominance remains an open prospective question.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig4_confirmatory_boundary_v3.pdf}
  \caption{Panel~(a) reports out-of-sample AP across the five confirmatory ApacheJIT repositories, with asterisks marking Hadoop subprojects. Panel~(b) plots evaluation horizon against MLP hindsight inflation across Benchmark~B and confirmatory streams.}
  \label{fig:rq2_confirmatory}
\end{figure}

\rqbox{Summary for RQ2}{On multimodal Benchmark~A, frozen Full-LR leads every deployable host in AP, and Survival-PU recovers Track-1 $F_1$. On gradual multi-year Benchmark~B streams, Full-LR Adapt-4W posts the top mean MLP AP, and Causal-4W, Wait-90, and FixedCurve-PU all significantly outrank Survival-PU. Within-window score standardization reverses that Causal-4W gap in exploratory checks without lifting Survival-PU past the other leaders. Under abrupt concatenated shifts, Survival-PU wins over the full post-onset horizon, and BORB-Alarm adapts fastest in early recovery. Finally, on held-out Phase~C repositories, MLP hindsight optimism fails to replicate and online ODaSC leads both standalone and four-umbrella evaluations.}

% ----------------------------------------------------------------------------
% 6.3 RQ3
% ----------------------------------------------------------------------------
\subsection{RQ3: Maturity Sensitivity, Mechanism Audit, and Component Ablations}
\label{sec:rq3}

RQ2 left an immediate mechanistic puzzle: why did adaptive Survival-PU trail zero-fit FixedCurve-PU on gradual Benchmark~B whole-stream AP? To answer it, we dissect Stage~1 sensitivity, cross-stage error propagation, and single-factor ablations on our ten-stream representative Benchmark~B subset.

\paragraph{Low Sensitivity Across Stage~1 Maturity Estimators.}
Substituting alternative Stage~1 maturity estimators shifts aggregate accuracy only modestly across the ten representative streams, where we report comparisons descriptively given limited ten-stream power (Table~\ref{tab:rq3_ablations}, full metrics in Table~S-10). Default cohort incidence, quantized Kaplan--Meier cure, global Weibull cure, and FixedCurve-PU span just 1.1 AP points and under half a point in tuned $F_{1,\text{th}}$, with zero survival fallbacks or unreachable bisections. Mirroring RQ2, Causal-4W still leads on AP and trails Survival-PU on tuned $F_{1,\text{th}}$ and post-Platt Brier loss.

[[TAB_RQ3_ABLATIONS]]

\paragraph{Unrevealed-Commit Mechanism Audit Against FixedCurve-PU and Writeback.}
All-pool soft-target accuracy conflates trivial positive writeback ($q_i = 1$ for already revealed defects) with genuine estimation on unrevealed commits. Auditing all 62 DriftLens alarms across the ten representative streams, covering 59,600 unique commits, separates these two effects relative to empirical archive maturity $c^*(a)$ (age-stratified breakdown in Table~S-11):
\begin{itemize}[leftmargin=1.6em,itemsep=0.05em]
  \item \textit{Non-Monotonic Error Propagation Across Stages.} Adaptive cohort incidence estimates empirical maturity substantially better than the fixed 90-day curve, cutting Stage~1 maturity MAE by 22\% at the alarm level and 33\% at the stream level. Better Stage~1 fit does not propagate monotonically downstream, however: FixedCurve-PU edges ahead on Stage~2 prevalence MAE, and both improve over unshrunk inverse-maturity estimates.
  \item \textit{Positive Writeback Versus Unrevealed Soft Targets.} Across the full adaptation pool, writing revealed defects back to label~1 drives most of the Brier improvement over de-weighted teacher scores. Even on strictly unrevealed commits with zero writeback, soft targets still beat de-weighted teacher scores across every age stratum. FixedCurve-PU achieves slightly lower overall unrevealed-commit Brier error by leading on young and mid-age commits, and Survival-PU leads on aged commits.
\end{itemize}

\par\smallskip\noindent
\begin{minipage}{\textwidth}
\fontsize{7.5}{8.9}\selectfont
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.04}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccc@{}}
\toprule
\textbf{Stage 1--3 Audit (62 Alarms)} & \textbf{Control} & \textbf{FixedCurve} & \textbf{Survival-PU} \\
\midrule
Stage~1 Maturity MAE (Alarm / Stream) & -- & 0.1204 / 0.1310 & \textbf{0.0938 / 0.0873} \\
Stage~2 Target Prior MAE (vs.\ HT) & 0.0712 & \textbf{0.0565} & 0.0620 \\
Stage~3 Unrevealed Brier ($y_{\text{obs}}=0$) & 0.1303 (DeWt) & \textbf{0.1200} & 0.1236 \\
Stage~3 Full-Pool Brier (+Writeback) & 0.1134 (+WB) & \textbf{0.1044} & 0.1076 \\
\bottomrule
\addlinespace[2pt]
\multicolumn{4}{@{}p{\textwidth}@{}}{\scriptsize Note: 62 DriftLens alarms (59,600 unique commits; $n=64{,}745$ unrevealed, $n=74{,}400$ full pool). HT = unshrunk Horvitz--Thompson, DeWt = de-weighted teacher, +WB = positive writeback.} \\
\end{tabular*}
\end{minipage}
\par\smallskip

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\textwidth]{fig5_ablations_and_controls_v3.pdf}
  \caption{Ten-stream RQ3 ablations and controls. Panel~(a) compares three maturity estimators, FixedCurve-PU, and single-factor ablations in AP; Panel~(b) contrasts AP and effort recall across teacher and gating controls.}
  \label{fig:rq3_abl}
\end{figure}

\paragraph{Component Ablations and Simplification Candidates.}
Teacher adaptation, monotone bisection, and anchor shrinkage carry Survival-PU's gains, and Stage~2 density ratios emerge as a natural simplification candidate (Figure~\ref{fig:rq3_abl}, Table~\ref{tab:rq3_ablations}). Freezing the teacher to its pre-stream checkpoint cuts AP by over three points and triggers repeated bisection clamping. Removing monotone bisection or anchor shrinkage similarly degrades both AP and calibration, whereas sweeping prior shrinkage $m_0$ and tail cap $\gamma_{\text{tail}}$ confirms low sensitivity across neighboring settings. Conversely, omitting Stage~2 density ratios slightly lifts AP, $F_{1,\text{th}}$, and effort recall at lower compute cost; control runs likewise confirm that Survival-PU beats Teacher-Matched and is largely insensitive to confidence gating.

\rqbox{Summary for RQ3}{All four Stage-1 estimators span just 1.1 AP points with zero fallbacks. Across 62 alarms, cohort incidence estimates Stage-1 maturity far more accurately than FixedCurve-PU, but FixedCurve-PU achieves slightly lower Stage-2 prevalence MAE and unrevealed-commit Brier error, revealing non-monotonic error propagation across stages. Teacher adaptation, monotone bisection, and prior shrinkage carry the primary weight, and density-ratio omission emerges as a simplification candidate.}

% ----------------------------------------------------------------------------
% 6.4 RQ4
% ----------------------------------------------------------------------------
\subsection{RQ4: Stress Tests Across Latency, Monitoring, Noise, and Prompting}
\label{sec:rq4}

Having dissected internal mechanisms, we stress-test operating boundaries across five targeted dimensions: latency scaling and calibration controls (RQ4a), monitor resampling and refresh (RQ4b), semi-synthetic SZZ label noise (RQ4c), window-size sweeps (RQ4d), and a paired Code-LLM case study (RQ4e).

\paragraph{RQ4a: Latency Scaling and Ranking-Versus-Classification Split.}
Scaling finite post-$t_0$ fix delays from zero to triple empirical lag widens the split between ranking and thresholded classification in Table~\ref{tab:rq4a_kappa} and Figure~\ref{fig:rq4_lat_mon}a, with Wait-90 results in Table~S-12. On whole-stream AP, Causal-4W stays modestly ahead of Survival-PU at every lag level, and age-blind Pure-nnPU collapses in effort recall; sanity checks behave as expected, with Frozen Host invariant throughout and Causal-4W coinciding with Window Oracle under post-$t_0$ oracle visibility.

Under heavy latency ($\kappa = 3.0\times$), however, classification diverges sharply. Causal-4W drops steeply on both fixed-threshold $F_1$ and causally tuned $F_{1,\text{th}}$. By contrast, Survival-PU leads Causal-4W by 14.3 points in fixed $F_1$ and 5.6 points in tuned $F_{1,\text{th}}$, and it also outperforms Wait-90.

[[TAB_RQ4A_KAPPA]]

\paragraph{Stage-1 Misspecification Protection Against FixedCurve-PU.}
Adaptive Stage~1 maturity estimation earns its keep at the latency extremes where a fixed 90-day assumption breaks down (Table~S-13). When fixes arrive immediately ($\kappa = 0\times$), FixedCurve-PU's 90-day curve misclassifies clean young commits as heavily censored, leaving FixedCurve-PU 2.2 AP points and four $F_1$ points behind Survival-PU, with Causal-4W leading overall.

At the opposite extreme of triple lag, the fixed 90-day curve overestimates maturity instead. Adaptive Survival-PU consequently beats FixedCurve-PU on fixed $F_1$, tuned $F_{1,\text{th}}$, and post-Platt Brier loss. On whole-stream AP across nonzero lag, however, FixedCurve-PU consistently outranks Survival-PU, finishing about a point ahead of Causal-4W and Survival-PU at $\kappa = 3.0\times$ (Table~\ref{tab:rq4a_posthoc_b2}).

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig6_latency_and_drift_factorial_v3.pdf}
  \caption{Latency scaling and drift-monitor calibration. Panel~(a) scales post-$t_0$ fix delays from zero to triple lag with frozen pre-$t_0$ supervision; Panel~(b) compares the $2 \times 2$ DriftLens factorial against Pote et al.'s reference and RBF-MMD.}
  \label{fig:rq4_lat_mon}
\end{figure}

\paragraph{Raw Versus 90-Day Platt Calibration and Post-Hoc Controls.}
Fitting standard 90-day Platt scaling to a heavily censored calibration pool degrades probability quality rather than repairing it (Table~\ref{tab:rq4a_posthoc_b2}, full metrics in Table~S-14). Whenever median fix lag exceeds three months, the calibration pool $\mathcal{M}_w$ retains heavy residual censoring, reaching 55.3\% in Benchmark~A, 43.7\% in Benchmark~B, and 65.9\% under triple lag. At triple lag, raw probabilities therefore beat Platt scaling on ECE for all five methods and on Brier error for four of five, leaving only PostHoc Logit's Brier error nearly unchanged.

Across all three latency regimes, Survival-PU incurs higher raw Brier and ECE than Causal-4W, and FixedCurve-PU achieves the lowest raw error at triple lag. This confirms Proposition~\ref{prop:bias}: Stage~3 applies positive class reweighting atop prevalence-calibrated targets, pushing raw probabilities upward. Causal-4W's censoring compression and FixedCurve-PU's negative target bias pull scores downward and partially offset that shift. Under the shared Platt calibrator, Survival-PU still retains the lowest post-Platt Brier and ECE.

Alarm-synchronized post-hoc score adjustments likewise fall short of PU retraining on heavy-lag classification (Table~\ref{tab:rq4a_posthoc_b2}). Both controls freeze Causal-4W's weights and adjust scores only at fired alarms: PostHoc Logit shifts logits via monotone bisection to match target prevalence, and PostHoc Ratio removes training class weights, divides by trailing mean maturity, and clips extreme probabilities. Because PostHoc Ratio removes class weights and Survival-PU retains weighted logits, fixed-0.5 $F_1$ compares asymmetric scales; on scale-fair tuned $F_{1,\text{th}}$, Survival-PU still leads PostHoc Ratio by 4.8 points and Causal-4W by 5.6 points across raw and calibrated scores. At zero lag, PostHoc Logit loses whole-stream AP because alarm-specific logit shifts distort cross-window score alignment without hurting per-window AP; PostHoc Ratio achieves the lowest raw Brier error and incurs a mild per-window AP dip solely from upper-bound clipping on 7.9\% of commits.

[[TAB_RQ4A_POSTHOC_B2]]

\paragraph{RQ4b: Unconfounded $2 \times 2$ DriftLens Calibration Factorial.}
Circular block-bootstrap calibration cuts validation-pool null alarm rates nearly fivefold, and rolling reference refresh eliminates missed transitions (Table~\ref{tab:rq4b_monitors}, Figure~\ref{fig:rq4_lat_mon}b). Under matched refresh rules, replacing IID with Circular Block reduces null alarm rates by $4.8\times$, dropping from 20.2\% to 4.2\% under static references and from 16.4\% to 3.4\% under rolling references.

Across twenty controlled-shift runs, rolling reference-and-threshold refresh eliminates missed transitions altogether, cutting miss rates from 10.0\% under Block + Static and 5.0\% under IID + Static to zero. On detected shifts, rolling monitors average 310 to 350 commits of delay, roughly $3\times$ faster than periodic fixed-schedule retraining (Table~\ref{tab:rq4b_monitors}).

Fewer alarms do not guarantee higher AP: default Block + Rolling fires on 32.1\% of windows versus 43.5\% for IID + Rolling and trails it slightly in AP. Even so, Block + Rolling offers the stronger operational balance, cutting null alarms nearly fivefold relative to IID while catching abrupt shifts much faster than periodic schedules. Closed-loop DriftLens refresh and alternative non-Gaussian block monitors, RBF-MMD and PCA-KS, show comparable low null rates and zero missed shifts, with slightly longer detection delays for the kernel and univariate detectors.

[[TAB_RQ4B_MONITORS]]

\paragraph{RQ4c--d: Semi-Synthetic PR-SZZ Noise and Window-Size Sweeps.}
Batch window and PU learners resist independent SZZ label noise far better than streaming Hoeffding and ElasticNet learners, BORB and ORB, in Figure~\ref{fig:rq4_szz_win}a and Table~S-15. Injecting project-averaged B-SZZ error rates ($\alpha_{\text{FP}} = 8.4\%$, $\beta_{\text{FN}} = 47.2\%$) \citep{bludau2026quantifying} into training across five PR-SZZ streams separates model families cleanly on unperturbed evaluation. Batch window and PU learners cluster within 0.4 AP points, with Survival-PU, Wait-90, and Causal-4W effectively tied at the top. By contrast, relative to its clean-label training baseline, BORB drops more than ten AP points to finish six points behind the batch learners, and ORB trails them by ten points. Because independent flips omit feature-dependent label noise, confirming this family split on manually audited SZZ corpora remains important future work.

Survival-PU's abrupt-shift advantage concentrates in narrow and medium windows and fades once wide buffers absorb older matured history (Figure~\ref{fig:rq4_szz_win}b). Across fixed 1,200-commit post-onset horizons on the three controlled shifts inside our ten-stream representative subset, Survival-PU leads deployable methods at $W \in \{200, 400, 600\}$. At the widest 800-commit window, Wait-90 and Causal-4W edge slightly ahead. On full ten-stream AP, Causal-4W stays ahead at every window width.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig7_szz_window_sensitivity_v3.pdf}
  \caption{Label-noise and window-size stress tests. Panel~(a) reports semi-synthetic SZZ perturbation across five PR-SZZ streams; Panel~(b) sweeps window width across fixed 1,200-commit post-onset horizons on three controlled shifts.}
  \label{fig:rq4_szz_win}
\end{figure}

\paragraph{RQ4e: Paired Code-LLM Prompting Case Study.}
Parameter-free in-context learning does not escape alarm-time label censoring (Figure~\ref{fig:rq4f_rt}a, Table~S-16). Across 6,400 audited deepseek-flash evaluations on 320-commit Qt and OpenStack slices, strictly paired six-shot prompts share identical exemplars and differ solely in whether exemplar labels use observable $y_{\text{obs}}(t)$ or eventual $y^*$. Censoring those exemplar labels drops OpenStack $F_1$ by 10.7 points and barely shifts Qt $F_1$.

By contrast, OpenStack AP rises by one point under censored exemplars, echoing Proposition~\ref{prop:bias}'s score-shift effect. Even with eventual labels, domain-trained JIT predictors outscore every prompting arm in AP by wide margins. Because this case study probes a single lightweight model on public historical slices, it illustrates exemplar-censoring sensitivity rather than general LLM capability.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.90\textwidth]{fig8_llm_and_runtime_forensics_v3.pdf}
  \caption{Code-LLM prompting audit and runtime cost. Panel~(a) compares deepseek-flash prompting arms against domain-trained JIT predictors on Qt and OpenStack slices; Panel~(b) reports standalone tabular runtime across 27 natural Benchmark~B streams.}
  \label{fig:rq4f_rt}
\end{figure}

\rqbox{Summary for RQ4}{At triple verification lag, FixedCurve-PU leads deployable AP and raw calibration. By contrast, Survival-PU leads all baselines and post-hoc controls on scale-fair $F_{1,\text{th}}$ and avoids FixedCurve-PU's zero-lag misspecification. Standard 90-day Platt scaling degrades probability quality under 65.9\% pool censoring. Circular block-bootstrap DriftLens cuts null alarms $4.8\times$ and catches abrupt shifts $3\times$ faster than periodic schedules. Batch window and PU learners resist independent SZZ noise better than BORB and ORB, and exemplar censoring cuts deepseek-flash OpenStack $F_1$ by over ten points.}

% ----------------------------------------------------------------------------
% 6.5 RQ5
% ----------------------------------------------------------------------------
\subsection{RQ5: Runtime Telemetry and Equal-Cap Review-Budget Error Analysis}
\label{sec:rq5}

Deployment viability depends jointly on execution overhead and net defect yield under fixed code-review capacity. We therefore audit standalone runtime alongside paired review-budget trade-offs across whole-stream and sprint-level horizons.

\paragraph{Cached Execution Versus Feature-Encoding Runtime.}
Survival-PU adds modest CPU overhead on pre-extracted tabular features, whereas feature encoding dominates multimodal runtime (Figure~\ref{fig:rq4f_rt}b, Table~S-17). When each alarm-triggered method is charged its full anchor and monitoring chain across 27 natural Benchmark~B streams, Survival-PU finishes in 14.4~s per stream, roughly 1.7~s per alarm, on par with Teacher-Matched. Online Hoeffding ORB, ElasticNet BORB, and Frozen Host finish under four seconds per stream; ODaSC requires nearly 70~s due to per-event tree updates and DenStream maintenance.

On multimodal Qt, cached Survival-PU finishes under twenty seconds on CPU LightGBM and under eight seconds on GPU CodeBERT. Including offline GPU CodeBERT batch encoding brings full-history Qt replay to 44.6~s, reflecting batch throughput rather than single-commit webhook latency.

[[TAB_RQ5_PAIRED_BUDGET]]

\paragraph{Paired Whole-Stream Versus Per-Window Equal-Cap Budgets.}
Review-budget granularity determines whether maturity correction yields net defect gains in daily engineering (Table~\ref{tab:rq5_paired_budget}). Directional rescue anecdotes cannot establish net utility, and retrospective whole-stream caps answer a different question from sprint-level capacity. Table~\ref{tab:rq5_paired_budget} reports paired Survival-PU minus Causal-4W differences under fixed-0.5 thresholding, equal-cap whole-stream budgets, and equal-cap per-window budgets alongside five-seed windowed 20\%-LOC recall $\Delta\text{WinR}_{20}$. Under LOC budgets, commits rank by risk density $\hat{p}/\text{LOC}$ and fill greedily without splitting commits, keeping realized LOC consumption closely matched across all eight hosts. Two operational lessons stand out:
\begin{itemize}[leftmargin=1.6em,itemsep=0.05em]
  \item \textit{Unmatched Fixed-Threshold Volume.} Preventing posterior collapse rescues hundreds of defects on Qt and dozens on OpenStack, but only by flagging thousands of extra clean commits on Qt and hundreds on OpenStack (Table~\ref{tab:rq5_paired_budget}).
  \item \textit{Whole-Stream Versus Per-Window Equal Caps.} Echoing Section~\ref{sec:rq2}'s score-scale finding, retrospective whole-stream budgeting penalizes Survival-PU on five of seven alarmed hosts, most steeply on Qt DeepJIT-CNN and OpenStack LightGBM. Switching to operational per-window budgets reverses or removes nearly every steep loss. Qt LightGBM stays strongly positive across commit and LOC caps, both DeepJIT-CNN hosts turn positive on five-seed $\Delta\text{WinR}_{20}$, and Qt Attention-CNN and OpenStack LightGBM break roughly even; only Qt CodeBERT stays negative across both granularities. High-confidence SZZ rescue illustrations and churn skew appear in Table~S-18.
\end{itemize}

\rqbox{Summary for RQ5}{On cached tabular features, ORB and BORB finish under four seconds per stream, Survival-PU takes 14.4~s (44.6~s on Qt with GPU CodeBERT batch encoding), and ODaSC takes nearly 70~s. Under matched caps, retrospective whole-stream budgeting penalizes Survival-PU on five of seven alarmed hosts. Operational per-window budgeting reverses most steep losses on LightGBM and DeepJIT-CNN, leaving only Qt CodeBERT with a net defect-yield deficit.}

% ============================================================================
% 7. Discussion and Observed Operating Boundaries
% ============================================================================
\section{Discussion and Observed Operating Boundaries}
\label{sec:discussion}

We synthesize our findings around the three framing questions from Section~\ref{sec:intro}, separating established benchmark patterns from open operational questions.

\subsection{What Label-Visibility Timing Changes}
\label{sec:disc_timing}

Our audit separates two questions routinely conflated in drift-triggered JIT-SDP \citep{pote2026adaptive}: whether small-window retraining beats freezing, and whether future-fix leakage inflates retrained weights. On long multi-year tabular streams, leaking future fixes inflates whole-stream AP substantially on both MLP and LightGBM. In two other regimes, however, even Window Oracle fails to beat Frozen Host or Causal-4W on neural hosts. One is low-prevalence multimodal Benchmark~A, where frozen linear models lead in line with \citet{zeng2021deep}. The other is the short confirmatory ApacheJIT cohort, where horizon length, higher standalone prevalence, lighter censoring, and project identity shift together. Because follow-up length and repository curation directly govern observable censoring, chronological JIT benchmarks should report archive follow-up horizons (Tables~S-8--S-9) alongside upstream repository splits such as the clean Hadoop partition (Table~S-7).

\subsection{What Three-Stage Maturity Correction Adds Over Simpler Baselines}
\label{sec:disc_increment}

Strict causal benchmarking clarifies exactly where three-stage Survival-PU adds value and where simpler learners suffice. On gradual multi-year streams, Full-LR Adapt-4W, Causal-4W, Wait-90, and zero-fit FixedCurve-PU all outperform Survival-PU on whole-stream AP. Table~\ref{tab:window_ap_scale} shows, however, that cross-window score scaling drives much of the gap against Causal-4W, and per-window ranking stays closely competitive across baselines.

Inside the pipeline, better upstream estimation does not guarantee downstream dominance. Adaptive cohort incidence estimates Stage~1 maturity far more accurately than FixedCurve-PU and improves soft-target Brier over de-weighted teachers on unrevealed commits (Table~S-11). FixedCurve-PU still edges ahead on Stage~2 prevalence error and overall unrevealed Brier, illustrating non-monotonic error propagation across coupled stages.

Under latency extremes, however, adaptive maturity correction earns clear operational separation (Table~\ref{tab:rq4a_posthoc_b2}). At heavy lag, neither alarm-synchronized post-hoc Causal-4W control matches Survival-PU's tuned $F_{1,\text{th}}$ recovery, and adaptive Stage~1 simultaneously avoids FixedCurve-PU's zero-lag misspecification. On probability calibration, Stage~3's positive class weight $\omega_{+,\text{teach}} > 1$ explains why Survival-PU trails Causal-4W and FixedCurve-PU on raw Brier, confirming Proposition~\ref{prop:bias}'s theoretical prediction. Because raw scores still beat 90-day Platt scaling whenever the calibration pool suffers heavy censoring, maturity-aware post-hoc calibration stands out as a high-value open problem.

\subsection{Operational Budgets and Benchmark Design Implications}
\label{sec:disc_implications}

Translating ranking metrics into daily engineering value hinges on how review budgets are scoped (Table~\ref{tab:rq5_paired_budget}). Fixed thresholds hide alert fatigue behind raw recall gains, and retrospective whole-stream caps conflate commit ranking with cross-window score drift. Switching to sprint-level 400-commit caps therefore reverses most steep losses on LightGBM and DeepJIT-CNN, exposing Qt CodeBERT's persistent deficit.

Because shift horizons and fix latencies are rarely known ahead of deployment, Table~\ref{tab:decision_matrix} separates directly observed operating boundaries (Part~I) from exploratory regime hypotheses (Part~II) rather than prescribing a single default learner. For benchmark designers, three concrete practices follow directly from our audit. First, enforce $t_i^{\text{fix}} \le t_{\text{alarm}}$ alongside full archive follow-up disclosure. Second, include frozen linear models, zero-fit maturity curves, post-hoc causal controls, and latency-aware stream learners \citep{cabral2019class,cabral2023borb,song2022odasc} as standard baselines. Third, report whole-stream AP jointly with per-window AP, scale-standardized AP, and operational per-window equal-cap review yields.

\begin{table}[t]
\centering
\caption{Synthesis of directly observed cross-paradigm operating boundaries and exploratory regime hypotheses across evaluated benchmarks.}
\label{tab:decision_matrix}
\fontsize{7.5}{9.0}\selectfont
\renewcommand{\arraystretch}{1.05}
\setlength{\tabcolsep}{2.4pt}
\begin{tabularx}{\textwidth}{L{2.55cm}Y Y}
\toprule
\textbf{Regime / Factor} & \textbf{Observed Benchmark Pattern} & \textbf{Mechanism \& Practical Scope} \\
\midrule
\multicolumn{3}{l}{\textit{Part I: Directly Observed Operating Boundaries Across Evaluated Benchmarks}} \\
\midrule
Multimodal streams (Benchmark~A) & Frozen Full-LR and LApredict lead every deployable neural and tree host in AP; frozen DeepJIT-CNN leads among single multimodal hosts. & Linear skips narrow the neural gap by 8--17 points. Survival-PU recovers Track-1 $F_1$, and per-window budget yield remains host-dependent (Table~\ref{tab:rq5_paired_budget}). \\
Gradual multi-year clusters (25 clusters) & Full-LR Adapt-4W, Causal-4W, Wait-90, and FixedCurve-PU outrank Survival-PU in whole-stream AP; BORB leads tuned $F_{1,\text{th}}$. & Cross-window prior updates shift score scales; within-window standardization flips the mean Causal-4W gap without pushing Survival-PU past Wait-90 or Full-LR. \\
Abrupt controlled shifts (4 splices) & Survival-PU leads full post-onset AP; ElasticNet BORB-Alarm adapts fastest over the initial 1,200 post-onset commits. & Newly shifted defects arrive young and unrevealed, inverting naive single-window rankings. At wide 800-commit windows, matured buffers let Wait-90 and Causal-4W edge ahead. \\
Latency extremes ($\kappa = 0\times, 3\times$) & Causal-4W leads at zero lag and Survival-PU beats FixedCurve-PU across metrics. At triple lag, Survival-PU leads tuned $F_{1,\text{th}}$ and fixed $F_1$, and FixedCurve-PU leads AP and raw calibration. & Adaptive Stage~1 prevents zero-lag misspecification and protects heavy-lag classification over post-hoc controls. Standard 90-day Platt scaling degrades calibration under heavy pool censoring. \\
Semi-synthetic SZZ noise (5 streams) & Batch window and PU learners cluster tightly, beating ElasticNet BORB by six AP points and Hoeffding ORB by ten points. & Batch reweighting absorbs independent flip noise better than per-event updates; feature-dependent real SZZ noise remains untested. \\
Drift monitoring ($2\times 2$ factorial) & Circular block calibration cuts null alarms nearly fivefold; rolling refresh eliminates missed shifts and detects $3\times$ faster than periodic $3W$. & Block resampling preserves sprint autocorrelation (Proposition~\ref{prop:autocorr}), curbing unnecessary censored updates without sacrificing shift sensitivity. \\
\midrule
\multicolumn{3}{l}{\textit{Part II: Exploratory Regime Hypotheses Requiring Prospective Validation}} \\
\midrule
Short confirmatory repos ($N < 3{,}000$) & Online Hoeffding ODaSC leads both standalone and pooled-umbrella evaluations; MLP hindsight optimism fails to replicate. & Shorter horizons, higher standalone prevalence, lighter censoring, and project identity covary alongside the upstream Hadoop split, preventing single-factor attribution. \\
Stage~2 simplification & Dropping Stage~2 density-ratio weights slightly improves ten-stream descriptive metrics at lower compute cost. & Worth testing prospectively across latency extremes and abrupt shifts before adopting as a default. \\
\bottomrule
\end{tabularx}
\end{table}

% ============================================================================
% 8. Threats to Validity
% ============================================================================
\section{Threats to Validity}
\label{sec:threats}

\paragraph{Construct Validity.}
SZZ back-tracing is vulnerable to tangled commits, refactorings, and unlinked fixes \citep{dacosta2017framework,rosa2021evaluating,fan2021impact}, and finite repository snapshots incur end-of-archive censoring whenever fixes occur after the observable archive proxy $T_{\text{end}}$; Benchmark~A likewise replays merged commits by authoring timestamp rather than initial patch uploads. We mitigate these threats by excluding heuristic and retroactive links from observable positives, auditing follow-up across all 34 natural streams, and verifying Benchmark~A stability on 180- and 365-day follow-up slices (Tables~S-8--S-9). To guard against metric artifacts where score-scale drift or commit-size skew distorts practical review value, we pair whole-stream AP with per-window and scale-standardized AP, benchmark LOC-Only overlap, verify paired equal-cap yields (Table~\ref{tab:rq5_paired_budget}), and disclose the clean Hadoop partition (Table~S-7). Our probability calibration audit covers standard 90-day Platt scaling rather than maturity-aware post-hoc alternatives.

\paragraph{Internal Validity.}
The main internal threats are temporal leakage, method assumptions, and post-hoc tuning bias. Every execution branch passed deterministic prefix-invariance checks with zero perturbation error, certifying leak-free pipeline logic rather than upstream Git provenance. Inside Survival-PU, Stage~1 assumes trailing local stationarity, Stage~2 uses heuristic prior shrinkage, and the upper prior cap ($\pi_{\max} = 0.70$) clamps bisection whenever observed prevalence exceeds 70\%, occurring on roughly 30\% of split HDFS* alarms versus under 1\% across natural Benchmark~B streams. We locked Benchmark~B analyses before acquiring ApacheJIT, documented the split rule for short confirmatory repositories, and treated Window Oracle strictly as a counterfactual isolating future-fix visibility. Automated scripts cross-check reported table values and protocol settings against archived result artifacts (Appendix~\ref{app:repro}).

\paragraph{External and Conclusion Validity.}
Our Track-2 stream implementations, ElasticNet BORB, Hoeffding ORB, and Hoeffding ODaSC, represent specific learner configurations rather than entire algorithmic families, and independent synthetic SZZ flips omit feature-dependent real SZZ errors. In addition, RQ4b estimates null alarm rates from resampled validation pools rather than separate stationary repositories, controlled shifts use one-hour splices across which pre-onset fixes continue landing, and RQ4e probes one lightweight model on 640 pre-cutoff public commits. Formal $\alpha = 0.05$ Holm--Bonferroni inference applies strictly to the 25-cluster and 24-cluster Benchmark~B cohorts, separate from exploratory checks in Table~\ref{tab:window_ap_scale}. Confirmatory non-replication of MLP hindsight inflation is reported directly without post-hoc retuning.

% ============================================================================
% 9. Conclusion
% ============================================================================
\section{Conclusion}
\label{sec:conclusion}

Unsupervised drift monitors remove label dependence when deciding \textit{when} to retrain, but not when \textit{supervising} weight updates under multi-month SZZ verification latency. Auditing 34 natural repository streams, eight controlled shifts, and a paired Code-LLM case study reveals that leaking future SZZ fixes significantly inflates multi-year window-oracle ranking over strict causal retraining, even though that neural hindsight optimism does not replicate on shorter confirmatory projects. Once strict causal visibility holds, no single adaptive paradigm dominates every setting. Frozen linear models lead deployable multimodal streams, while simpler causal baselines and zero-fit FixedCurve-PU outperform Survival-PU on gradual multi-year whole-stream ranking (with within-window standardization reversing the mean Causal-4W gap). ElasticNet BORB excels on gradual tuned classification and immediate post-shift recovery, and online Hoeffding ODaSC leads the shorter confirmatory repositories. At the monitoring layer, circular block-bootstrap calibration with rolling reference refresh sharply curbs false-alarm storms while preserving rapid shift detection.

Within these operating boundaries, three-stage Survival-PU earns a precise, targeted role: it leads full post-shift ranking, lifts gradient-boosted soft-target distillation over a step-matched teacher, avoids zero-lag misspecification from hardcoded waiting curves, and protects scale-fair thresholded classification under severe latency where Causal-4W and post-hoc score adjustments falter. Operationally, Survival-PU adds modest CPU overhead on cached tabular features, whereas feature encoding dominates multimodal runtime, and scoping review capacity per sprint rather than across whole streams reverses most steep budget losses on LightGBM and DeepJIT-CNN, leaving only Qt CodeBERT with a net defect-yield deficit. Finally, our calibration audit warns that standard 90-day Platt scaling degrades probability quality whenever calibration pools remain heavily censored. Developing maturity-aware post-hoc calibrators and prospectively validating observable regime-selection criteria stand out as central next steps for live adaptive JIT-SDP.

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
  \item \textbf{Data availability:} The complete replication package, including the immutable core source modules, companion audit and verification scripts, pinned Python dependency lockfile, SHA-256 manifests for all 94 stream bundles, 50 commit-level \texttt{.npz} prediction archives, lossless LLM exchange logs, and Online Resource~1 (\texttt{Online\_Resource\_1.pdf}), is publicly available at \url{https://github.com/789748456v/CausalJIT-Audit-SQJ} with a full offline archival bundle prepared for permanent Zenodo deposit upon acceptance.
  \item \textbf{Author contributions:} Z.L.: Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Writing -- Original Draft. S.X.: Software, Validation, Data Curation, Writing -- Review \& Editing. K.L.: Validation, Investigation, Visualization, Writing -- Review \& Editing. S.M.: Formal Analysis, Investigation, Writing -- Review \& Editing. F.G.: Validation, Visualization, Writing -- Review \& Editing. Y.Z.: Supervision, Conceptualization, Methodology, Project Administration, Writing -- Review \& Editing. All authors read and approved the final manuscript.
\end{itemize}

% ============================================================================
% Compact Appendix A (1 Table at 7.5pt + Proofs + Traceability)
% ============================================================================
\begin{appendices}
\renewcommand{\theHtable}{\thesection.\arabic{table}}

\section{Locked Parameters, Algebraic Proofs, and Code Traceability}
\label{app:repro}

Table~\ref{tab:hyperparams} lists all 33 locked protocol parameters frozen in the configuration module before unlocking the confirmatory cohort. Complete host architecture tables, executed competitor specifications, full-column mirrors of Main tables, and secondary data-funnel, follow-up, mechanism, shift, prompting, runtime, and forensic audits appear in companion \textbf{Online Resource~1} (Tables~S-1--S-18).

\begin{table}[t]
\centering
\caption{Locked protocol parameters of Survival-PU and DriftLens frozen in the configuration module prior to unlocking the confirmatory repositories.}
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
Stg 2 & $\tau_{\text{s}}, \beta_{\text{llm}}$ & $1.15, 0.25$ & \texttt{teacher\_\allowbreak sharpness}, \texttt{llm\_\allowbreak blend\_\allowbreak weight} & Scale, opt.\ LLM$^\dagger$ \\
Stg 3 & $N_{\text{a}}, \gamma, u_0$ & $2{,}000, 1.40, 0.45$ & \texttt{max\_\allowbreak anchor\_\allowbreak commits}, \texttt{drift\_\allowbreak emphasis\_\allowbreak gamma}, \texttt{maturity\_\allowbreak weight\_\allowbreak floor} & Pool, $u_i$ \\
Stg 3 & $p_{\text{r}}, K_{\text{b}}$ & $[0.003, 0.997], 28$ & \texttt{posterior\_\allowbreak p\_\allowbreak min}, \texttt{posterior\_\allowbreak p\_\allowbreak max}, \texttt{bisection\_\allowbreak iters} & Bisection \\
Budg. & $R_{\text{i}}, R_{\text{a}}$ & $90, 75$ & \texttt{lgb\_\allowbreak init\_\allowbreak rounds}, \texttt{lgb\_\allowbreak alarm\_\allowbreak rounds} & GBDT trees \\
Budg. & $E_{\text{t}}, E_{\text{s}}, \alpha_{\text{lr}}$ & $4, 4, 4.5\text{e-}4$ & \texttt{neural\_\allowbreak teacher\_\allowbreak epochs}, \texttt{neural\_\allowbreak student\_\allowbreak epochs}, \texttt{neural\_\allowbreak adapt\_\allowbreak lr} & Epochs, lr \\
Mon. & $W, d, p, \text{cd}$ & $400, 24, 95.0, 1$ & \texttt{window\_\allowbreak size\_\allowbreak default}, \texttt{drift\_\allowbreak pca\_\allowbreak dim}, \texttt{drift\_\allowbreak block\_\allowbreak pct}, \texttt{drift\_\allowbreak cooldown\_\allowbreak windows} & DriftLens \\
\bottomrule
\end{tabularx}
\vspace{2pt}\par\noindent{\scriptsize\raggedright Note: $^\dagger$\texttt{llm\_blend\_weight} ($0.25$) is an optional Stage~3 log-odds blend parameter exercised solely in the auxiliary RQ4e LLM-prior diagnostic (inactive across all primary Benchmark~A/B, RQ3, RQ4a--d, and Phase~C runs); neural teachers warm-start from a fixed 85\%/15\% blend of the current checkpoint and pre-stream anchor. Pre-locked Stage~1--2 scaling constants in Eqs.~\eqref{eq:tail_cap} and~\eqref{eq:teacher_weights}: $a_0 = 45\text{d}$, $\phi_{\min} = 0.65$, $\lambda_g = 0.28$, $\rho_0 = 0.55$, $\epsilon_\pi = 0.03$, $\omega_{\min} = 1.5$, $\omega_{\max} = 4.5$.\par}
\end{table}

\paragraph{Proof of Proposition~\ref{prop:bias}.}
Under $y_{\text{obs}} \mid x \sim \operatorname{Bernoulli}(\bar{c}(x)\eta^*(x))$, setting $\partial \mathbb{E}[\ell \mid x]/\partial p = 0$ yields $\omega_+ \bar{c}(x)\eta^*(x)/p = (1 - \bar{c}(x)\eta^*(x))/(1 - p)$, which rearranges directly to Eq.~\eqref{eq:b2_bayes}. Differentiating Eq.~\eqref{eq:b2_bayes} with respect to $\eta^*$ for constant $\bar{c}_W$ confirms Claim~1's monotonicity, and solving $\hat{p}_{\text{obs}}(x) < \tau_{\text{th}}$ gives Eq.~\eqref{eq:exact_crossing}. Comparing the monotone odds $\hat{p}_{\text{obs}}/(1 - \hat{p}_{\text{obs}})$ at $x_{\text{new}}$ and $x_{\text{old}}$ establishes Claim~2. \qed

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
Core estimators (\texttt{Cohort\allowbreak Incidence\allowbreak Estimator}, \texttt{\_set\_fallback()}, \texttt{estimate\_\allowbreak ht\_\allowbreak shrunk\_\allowbreak prevalence}, and \texttt{compute\_\allowbreak survival\_\allowbreak pu\_\allowbreak posterior}) reside in the survival-estimator module, while host learners, custom soft-target gradient-boosting objectives, and causal stream baselines (\texttt{calibrate\_\allowbreak causal\_\allowbreak stream\_\allowbreak probs\_\allowbreak strict}) reside in the host-model and stream-competitor modules. Protocol sizing (\texttt{compute\_\allowbreak split\_\allowbreak sizes}), locked interpretation rules, and the ten-stream representative subset (\texttt{REPRESENTATIVE\_\allowbreak 10\_\allowbreak STREAMS}) live in the configuration, data-audit, and experiment-runner modules. Automated verifiers in the replication repository cross-check every archived JSON hash, table cell across Main and Online Resource~1, locked parameter, and mechanism invariant; exact module filenames and artifact crosswalks appear in the repository README.
\endgroup

\end{appendices}

\begingroup\raggedright
\bibliography{references}
\endgroup

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
\usepackage[hyphens]{url}
\usepackage{microtype}
\def\UrlBigBreaks{\do\/}
\def\UrlBreaks{\do\/\do-\do.}
\def\UrlNoBreaks{\do\:}
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
\begingroup\raggedright
\bibliography{references}
\endgroup

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

Unsupervised drift monitors such as DriftLens flag representation shifts in Just-in-Time Defect Prediction without defect labels, motivating immediate window retraining. Label-free monitoring decides only when to retrain, however, not how to supervise weight updates. Under multi-month SZZ verification latency, most recently introduced defects remain unfixed at alarm time and appear clean. We audit this causal supervision gap across 34 natural repository streams totaling 347,122 commits, spanning 29 multi-year streams and five shorter confirmatory projects, alongside eight controlled shifts and a Code-LLM case study. Across linear, streaming, and Positive-Unlabeled learners, we evaluate circular block-bootstrap DriftLens alongside Survival-PU, a controlled maturity-aware three-stage testbed trained on soft targets. In multi-year ecosystems, 68.8% to 94.6% of defects inside recent retraining windows remain right-censored at alarm time. Leaking future fixes into matched window-oracle updates significantly inflates Average Precision over causal retraining across multi-year neural and tree models, and block-bootstrap calibration cuts DriftLens false alarms nearly fivefold. On the shorter confirmatory projects, however, that neural hindsight optimism fails to replicate. Once strict causal visibility holds, no single paradigm dominates every regime. Frozen linear models lead deployable multimodal streams, simpler causal baselines and a fixed maturity curve outperform Survival-PU on gradual whole-stream ranking, and streaming Hoeffding trees led by ODaSC excel on the confirmatory projects. By contrast, Survival-PU leads post-shift ranking, avoids hardcoded-lag errors when fixes arrive immediately, and protects thresholded classification under severe latency where standard post-hoc calibration degrades.

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

Every empirical table cell across the **Main Manuscript** (`839` cells across `11` empirical tables) and **Online Resource 1** (`1,054` cells across `16` supplementary/full-column tables; **`1,893` total machine-verified cells**), plus all `33` locked `V3_CONFIG` fields and all key prose comparisons (`35` exact-path claims + `61` relational guards), is verified automatically against the archived JSON artifacts in this repository.

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
│   ├── verify_prose_claims_v3.py                   # [Tier 1] Verifies V3_CONFIG, 35 exact-path claims & 61 relational guards
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

# 3. Verify all 33 V3_CONFIG table fields, 35 exact-path prose claims, 61 relational guards & adversarial self-tests
python scripts_v3/verify_prose_claims_v3.py

# 4. Verify archived pre-unlock gate SHA-256 (766e3fd6...) and run live mechanism Checks 2-14 (41 prefix-invariant branches)
python scripts_v3/run_post_campaign_mechanism_checks_v3.py

# 5. Regenerate all 10 figures (Fig. 1-10), 27 empirical LaTeX tables, manuscript/main.tex, and online_resource.tex
python scripts_v3/generate_conceptual_diagrams_v3.py
python scripts_v3/analyze_and_plot_v3.py
python scripts_v3/build_manuscript_v3.py
```

Compiling `manuscript/main.tex` and `manuscript/online_resource.tex` via `pdflatex -> bibtex -> pdflatex -> pdflatex` reproduces the camera-ready **39-page Springer SQJ Main PDF** (`Section 6 Empirical Results` starting at page 14, i.e., 36% into the PDF) and **11-page companion Online Resource 1 PDF** (`Online_Resource_1.pdf`), both with `0` Errors, `0` Warnings, and `0` Overfull hboxes.

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
