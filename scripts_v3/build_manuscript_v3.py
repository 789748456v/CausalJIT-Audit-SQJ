r"""
Programmatic Manuscript & Replication Documentation Assembler (`scripts_v3/build_manuscript_v3.py`).

Reads the 22 cell-registered empirical LaTeX tables from `build_artifacts/v3_tables_generated.tex` and writes:
  1. `manuscript/main.tex` (Complete Springer *Software Quality Journal* LaTeX manuscript combining
     progressive SQJ academic narrative in Sections 1--9 with full symbol-level implementation
     traceability in Table 2 and Appendix A, synchronized bit-for-bit with frozen `src_v3/*.py`)
  2. `manuscript/N05_SQJ_Manuscript.md` (Synchronized Markdown manuscript summary)
  3. `README.md` (Complete academic replication & verification guide)
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
    for m in re.finditer(r"\\begin\{table\}\[!htb\].*?\\end\{table\}", txt, re.DOTALL):
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
\renewcommand{\arraystretch}{1.08}
\raggedbottom
\hypersetup{bookmarksdepth=2}

\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.82}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{plain}
\newtheorem{proposition}{Proposition}

\newcommand{\rqbox}[2]{%
  \vspace{0.35em}
  \begin{center}
  \setlength{\fboxsep}{6.5pt}%
  \fcolorbox{black!35}{gray!6}{%
    \begin{minipage}{0.945\textwidth}
      \footnotesize\textbf{#1:} #2
    \end{minipage}%
  }
  \end{center}
  \vspace{0.20em}
}

\begin{document}

% ============================================================================
% Title, Authors, & Affiliations (Springer Nature Software Quality Journal)
% ============================================================================
\title[Detecting Drift is Not Safe Retraining]{Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction}

\author[1]{\fnm{Zhijie} \sur{Li}}\email{lizhijie@mails.neu.edu.cn}
\author[1]{\fnm{Shaoyang} \sur{Xu}}\email{xusy3@mails.neu.edu.cn}
\author[1]{\fnm{Ketong} \sur{Liu}}\email{liuketong@mails.neu.edu.cn}
\author[1]{\fnm{Shibo} \sur{Meng}}\email{mengsb1@mails.neu.edu.cn}
\author[2]{\fnm{Feiran} \sur{Gao}}\email{3120260997@bit.edu.cn}
\author*[1]{\fnm{Yuli} \sur{Zhao}}\email{zhaoyl@swc.neu.edu.cn}

\affil*[1]{\orgdiv{Software College}, \orgname{Northeastern University}, \orgaddress{\street{No.~195 Chuangxin Road, Hunnan District}, \city{Shenyang}, \postcode{110169}, \country{China}}}
\affil[2]{\orgdiv{School of Computer Science and Technology}, \orgname{Beijing Institute of Technology}, \orgaddress{\street{No.~5 South Zhongguancun Street, Haidian District}, \city{Beijing}, \postcode{100081}, \country{China}}}

% ============================================================================
% Structured Abstract (~235 Words — Strictly Within Springer 150-250 Limit)
% ============================================================================
\abstract{\textbf{Context:} Unsupervised monitors like DriftLens flag representation shifts in Just-in-Time Defect Prediction without labels, motivating immediate window retraining.
\textbf{Objective:} Monitoring determines when to retrain, not how to supervise updates: under multi-month SZZ latency, recent bug-inducing commits still appear clean. We audit this causal gap and map boundaries across linear, streaming, and Positive-Unlabeled learners.
\textbf{Method:} Across 37 development streams and five confirmatory ApacheJIT repositories---34 natural streams, eight controlled shifts, 347,122 natural commits, and a paired Code-LLM study---we pair block-bootstrap DriftLens with three-stage \textbf{Survival-PU}, combining cohort-incidence maturity, clipped prior shrinkage, and soft-target distillation.
\textbf{Results:} Across scheduled 400-commit boundaries, 68.8\%--94.6\% of defects remain right-censored. Window-oracle supervision significantly inflates whole-stream Average Precision over causal retraining across $K=25$ clusters, adding $+0.0237$ on MLP ($p_{\text{Holm}}=6.63\times 10^{-5}$) and $+0.0476$ on LightGBM ($p_{\text{Holm}}=3.70\times 10^{-5}$). Under strict causal visibility, frozen logistic regression leads Benchmark~A at $0.3653$ AP. On gradual $K=25$ whole-stream AP, \texttt{Full-LR Adapt-4W} leads in mean at $0.5108$, while $B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU} at $0.4942$ ($p_{\text{Holm}}=0.0295$) significantly exceed \texttt{Survival-PU} at $0.4866$; within-window standardization reverses the mean $B_2\text{-4W}$ gap non-significantly. On the confirmatory cohort, MLP hindsight optimism does not replicate, and \texttt{ODaSC} leads unsplit Standalone-3 at $0.7857$ and umbrella-4 at $0.8289$. Conversely, \texttt{Survival-PU} leads full post-shift AP at $0.5764$, avoids $\kappa=0\times$ misspecification, and protects $F_1$ under $3\times$ latency where \texttt{FixedCurve-PU} leads AP and raw Brier.
\textbf{Conclusions:} Chronological splitting permits future-fix leakage; adaptation must match shift regime, latency, calibration, and granularity.}

\keywords{Just-in-Time Defect Prediction, Distribution Shift, Verification Latency, Right-Censoring, Positive-Unlabeled Learning, Survival Analysis, SZZ Evaluation Bias, Large Language Models}

\maketitle

% ============================================================================
% 1. Introduction
% ============================================================================
\section{Introduction}
\label{sec:intro}

Just-in-Time Software Defect Prediction (JIT-SDP) helps quality engineering teams prioritize code review and pre-merge checks at commit time, while design rationale remains fresh in developers' working memory \citep{kamei2013large,hoang2019deepjit,pornprasit2021jitline,marrium2026flame}. In long-lived repositories, however, static commit classifiers gradually degrade as codebases evolve through refactoring, framework migrations, subsystem turnover, and changing development norms \citep{mcintosh2018are,gama2014survey,lu2018learning}. Unlike conventional streaming tasks where ground truth arrives almost immediately, JIT-SDP operates under heavy-tailed \textit{verification latency} \citep{cabral2019class,tabassam2020handling,song2022odasc,song2023humla}. A bug-inducing commit authored at timestamp $t_i$ receives its defect label $y_i^* = 1$ via SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework,rosa2021evaluating,fan2021impact,bludau2026quantifying} only after the fault manifests and a developer merges a bug-fixing commit at $t_i^{\text{fix}}$. In practice, the resulting fix delay $\Delta_i = (t_i^{\text{fix}} - t_i)/86400$ routinely has a median of three to six months.

To respond to representation shifts without waiting months for confirmed labels, recent quality engineering studies have adopted \textit{unsupervised embedding-space drift monitors}. Notably, \citet{pote2026adaptive} coupled bimodal commit encoders---Attention-CNN and CodeBERT---with DriftLens \citep{greco2021driftlens,greco2023driftlens}, tracking the multivariate Gaussian Fr\'echet distance \citep{dowson1982frechet} between pre-stream reference embeddings and incoming commit windows $W_{\text{drift}}$. Because DriftLens inspects latent representations without defect labels, it flags shifts within milliseconds and triggers lightweight retraining on the latest window, escalating to a four-window replay after consecutive alarms. Ordering commits chronologically without random shuffling made retrospective evaluation appear aligned with live continuous integration (CI) deployment \citep{pote2026adaptive}.

Moving unsupervised drift-triggered retraining from archival datasets into live CI pipelines, however, exposes a fundamental causal supervision gap:

\begin{quote}
\small\textit{Unsupervised drift monitoring removes label dependence only when \underline{deciding when to retrain}, not when \underline{supervising how to update model weights}. At the exact wall-clock timestamp $t_{\text{alarm}}$ when DriftLens flags a shifted window $W_{\text{drift}}$, most genuinely bug-inducing commits inside $W_{\text{drift}}$ are only days or weeks old, so $t_i^{\text{fix}} > t_{\text{alarm}}$. Under strictly causal repository visibility, their observable labels are right-censored into apparent clean negatives, yielding $y_{i,\text{obs}}(t_{\text{alarm}}) = 0$.}
\end{quote}

Throughout this study, we use ``safe'' or ``supervision-reliable'' retraining specifically to mean avoiding censoring-induced posterior collapse and thresholded-recall degradation, rather than claiming formal safety guarantees. Our investigation centers on three linked empirical questions. First, we examine \textit{what label-visibility timing changes} once future SZZ fixes are blocked alongside commit-time feature hygiene and archive-horizon follow-up checks. Second, we test \textit{what three-stage maturity correction genuinely adds} over frozen linear predictors \citep{zeng2021deep}, zero-fit maturity priors, alarm-synchronized post-hoc $B_2\text{-4W}$ probability adjustments, and adapted verification-latency stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}. Third, we evaluate \textit{what these performance differences mean operationally} under equal-cap commit and lines-of-code (LOC) review budgets.

Our matched audit reveals clear, regime-specific operating boundaries rather than a universal winner. Across $K=25$ multi-year donor clusters, a Pote-inspired offline window-oracle counterfactual ($B_{1b}^*$) significantly inflates whole-stream Average Precision (AP) over matched causal four-window retraining ($B_2\text{-4W}$). Meanwhile, circular block-bootstrap calibration cuts validation-pool resampled-null alarm rates by $4.8\times$. Once strict causal visibility is enforced, simpler learners win several prominent regimes. Frozen tabular logistic regression leads deployable Benchmark~A. On gradual $K=25$ whole-stream AP, simpler causal baselines---namely $B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU}---significantly outperform \textbf{Survival-PU}, though within-window score standardization reverses the mean $B_2\text{-4W}$ gap non-significantly. On the short confirmatory ApacheJIT cohort, MLP hindsight optimism fails to replicate, and adapted stream learners (\texttt{ODaSC}, \texttt{BORB}, and \texttt{ORB}) lead overall.

Conversely, \texttt{Survival-PU} exhibits targeted, repeatable strengths in three specific operating regimes. It leads full post-shift AP on controlled shifts and improves LightGBM soft-target distillation over a step-matched teacher. It also protects classification performance at \textit{both} latency extremes: it avoids \texttt{FixedCurve-PU}'s misspecification under immediate fixes at $\kappa = 0\times$, and it outperforms $B_2\text{-4W}$, two post-hoc $B_2\text{-4W}$ controls, and \texttt{FixedCurve-PU} on threshold-tuned $F_{1,\text{th}}$ under heavy latency at $\kappa = 3\times$. At the same time, evaluating raw versus calibrated probabilities exposes a clear risk of fitting a standard 90-day Platt calibrator to a heavily censored calibration pool.

\subsection{Research Questions}
\label{sec:rqs}

We operationalize these three empirical questions through five sequential research questions:

\begin{enumerate}[leftmargin=2.3em,label=\textbf{RQ\arabic*:}]
  \item \textbf{Causal Audit of Right-Censoring and Hindsight Optimism.} What fraction of true defects inside recent $1W$ and $4W$ windows remain right-censored at scheduled boundaries and fired alarms? How severely does a Pote-inspired offline window-oracle counterfactual ($B_{1b}^*$) diverge from matched causal retraining and frozen predictors?
  \item \textbf{Cross-Paradigm, Evaluation-Granularity, and Confirmatory Operating Boundaries.} How does \textbf{Survival-PU} compare against Track~1 counterfactuals and Track~2 linear or adapted stream learners across four multimodal and tree hosts, $K=25$ natural donor clusters under whole-stream and per-window AP, four controlled shifts, and five confirmatory ApacheJIT repositories?
  \item \textbf{Maturity-Estimator Sensitivity, Unrevealed-Sample Mechanism Audit, and Ablations.} How sensitive is Survival-PU to alternative maturity estimators such as Kaplan--Meier cure, Weibull cure, or zero-fit \texttt{FixedCurve-PU}? How accurately do Stages~1--3 estimate maturity, target prevalence, and soft targets strictly on unrevealed commits versus a positive-writeback control, and which internal components carry weight?
  \item \textbf{Stress Tests Across Latency Extremes, Monitoring, Noise, and Prompting.} How do adaptive learners---including alarm-synchronized post-hoc $B_2\text{-4W}$ controls under symmetrical raw versus 90-day Platt evaluation---behave across latency multipliers $\kappa \in [0.0, 3.0]$, a $2 \times 2$ DriftLens factorial, independent semi-synthetic PR-SZZ noise \citep{bludau2026quantifying}, window sweeps, and a paired Code-LLM prompting case study?
  \item \textbf{Runtime Cost and Equal-Cap Review-Budget Error Analysis.} What cached versus encoding-inclusive runtime overhead does each paradigm incur, and what are the paired rescued-versus-harmed defect trade-offs under fixed thresholding versus equal-cap whole-stream and per-window review budgets?
\end{enumerate}

\subsection{Summary of Contributions}
\label{sec:contributions}

Ordered by primary evidentiary support, our four contributions are as follows:

\begin{enumerate}[leftmargin=1.8em]
  \item \textbf{Formal and Empirical Causal Supervision Audit.} Building on drift-triggered JIT-SDP \citep{pote2026adaptive}, delayed-feedback modeling \citep{chapelle2014modeling}, and SZZ audits \citep{song2023waiting,bludau2026quantifying}, we formalize future-fix leakage whenever $t_i \le t_{\text{alarm}} < t_i^{\text{fix}}$. We separate uniform-maturity threshold crossing from heterogeneous rank inversion and cross-window score-scale shifts, quantify $K=25$ oracle-versus-causal AP inflation, and audit observable archive-horizon follow-up across all 34 natural streams.
  \item \textbf{Cross-Paradigm Operating Boundaries, Confirmatory Data Audit, and Matched-Budget Evaluation.} Across 37 development streams and five confirmatory repositories, we map where frozen linear models, \texttt{FixedCurve-PU}, $B_2\text{-4W}$, \texttt{BORB}, and \texttt{ODaSC} excel. We disclose upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split while foregrounding the unsplit Standalone-3 replication check, isolate cross-window scale effects via 5-seed $K=25$ standardization ablations, and quantify paired whole-stream versus per-window matched-budget defect yields.
  \item \textbf{Maturity-Calibrated PU Testbed, Unrevealed-Sample Audits, and Symmetrical Calibration Controls.} We instrument gate-free \textbf{Survival-PU} to test what maturity modeling adds over simpler baselines. We document its identifiability assumptions, bisection unreachability rates, unrevealed-commit soft-target accuracy against \texttt{FixedCurve-PU} and positive writeback, symmetrical raw versus 90-day Platt calibration against two post-hoc $B_2\text{-4W}$ controls, and block-bootstrap monitoring trade-offs.
  \item \textbf{Provenance-Audited Replication Suite and Automated Verifiers.} We release a SHA-256-locked benchmark and verification suite covering 94 stream bundles, 41 prefix-invariant execution branches, 50 commit-level prediction archives, lossless prompting logs, and automated cell-, parameter-, and prose-level verifiers.
\end{enumerate}

% ============================================================================
% 2. Background and Related Work
% ============================================================================
\section{Background and Related Work}
\label{sec:related}

To situate our causal audit and cross-paradigm evaluation within the broader software quality literature, Table~\ref{tab:positioning} contrasts static JIT-SDP predictors, unsupervised drift-triggered retraining, passive shift monitors, verification-latency stream learners, canonical PU learning, and SZZ bias audits along three dimensions: update triggers, causal supervision at adaptation time, and observed operating boundaries.

\begin{table}[!htb]
\centering
\caption{Conceptual positioning relative to static JIT-SDP, unsupervised drift retraining, passive shift monitoring, verification-latency stream learners, canonical PU learning, and SZZ evaluation-bias auditing.}
\label{tab:positioning}
\fontsize{6.2}{7.5}\selectfont
\renewcommand{\arraystretch}{1.04}
\setlength{\tabcolsep}{2.2pt}
\begin{tabularx}{\textwidth}{L{2.35cm}L{1.45cm}Y L{1.85cm}L{2.25cm}}
\toprule
\textbf{Paradigm / Study} & \textbf{Update Trigger} & \textbf{Supervision at $t_{\text{alarm}}$ / Step $t$} & \textbf{Unmatured $W_{\text{drift}}$} & \textbf{Empirical Regime / Boundary} \\
\midrule
Static JIT-SDP \citep{kamei2013large,hoang2019deepjit,feng2020codebert,zeng2021deep} ($B_0$, Linear) & None (frozen) & Causal pre-stream prefix ($t \le t_0$) & Not used & \texttt{Full-LR} leads deployable Benchmark~A; stale after shift \\
Pote-inspired oracle \citep{pote2026adaptive} ($B_{1b}^*$) & IID DriftLens & Hindsight $y^*$ ($t_{\text{fix}} > t_{\text{alarm}}$) & Non-causal oracle & Non-deployable counterfactual reference \\
Pote-inspired causal \citep{pote2026adaptive} ($B_2$) & DriftLens & Observable $y_{\text{obs}}(t_{\text{alarm}})$ & Treated as $y=0$ & Competitive gradual AP ($4W$); $F_1$ drops under $3\times$ lag \\
\citet{shah2026detecting} & Shift metrics & Passive monitoring only & Not applicable & Evaluates alert correlation without weight updates \\
Delayed Window ($\text{Wait-}90$) & Block DriftLens & Revealed fixes $\vee$ age $a_i \ge 90\text{d}$ & $<90\text{d}$ unfixed dropped & Reduces $<90\text{d}$ FN noise; lags $W=200$ shifts \\
\citet{cabral2019class} ($\text{ORB}$) & Per-event OHT & $90\text{d}$ clean + fast fixes + OBF Bagging & Excluded until $90\text{d}$ or fix & Lightweight CPU Hoeffding ensemble; sensitive to SZZ noise \\
\citet{cabral2023borb} ($\text{BORB}$) & $200$-c / alarm & $90\text{d}$ clean + fixes + ElasticNet SGD & Excluded until $90\text{d}$ or fix & Best $K=25$ $F_{1,\text{th}}$ \& first-1,200 post-shift AP \\
\citet{song2022odasc} ($\text{ODaSC}$) & Per-event OHT & $15\text{d}$ provisional $0$ + DenStream $\alpha_i$ & Provisional $0$ after $15\text{d}$ & Highest confirmatory AP; higher DenStream CPU cost \\
\citet{chapelle2014modeling} (Delayed FB) & Online updates & Joint $p(x)$ + exp.\ $\lambda(x)$ delay MLE & Soft delayed-event likelihood & Display-ad delayed feedback; inspired global Stage-1 Weibull cure \\
\citet{kiryo2017positive} ($\text{Pure-nnPU}$) & Block DriftLens & Disjoint $P$ vs.\ marginal $U$ under flat $\hat{\pi}$ & Age-blind unlabeled & Trails on $K=25$ AP \& $\kappa=3\times$ effort recall \\
\citet{song2023waiting,bludau2026quantifying} & Offline audit & Eval waiting-time noise / PR-SZZ vs.\ B-SZZ & Static/eval focus & Audits retrospective eval waiting noise \& SZZ FP/FN rates \\
\midrule
\textbf{Survival-PU} & \textbf{Block DriftLens} & \textbf{Cohort $c(a_i)$ + IP $\hat{\pi}_{\text{target}}$ + PU targets} & \textbf{Age-weighted $q_i(\delta^*)$} & \textbf{Best full post-shift AP \& $3\times$-lag $F_1$/$F_{1,\text{th}}$; trails \texttt{FixedCurve-PU} on gradual AP} \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Just-in-Time Defect Prediction, Linear Baselines, and Unsupervised Drift Detection}
\label{sec:rel_jitsdp}

Classical JIT-SDP represents commits with expert change metrics capturing diffusion, size, purpose, history, and developer experience \citep{kamei2013large}. Subsequent work expanded these representations with post-push code-review attributes \citep{mcintosh2018are}, AST and pull-request features \citep{bludau2022feature,bludau2022prszz}, graph fault localization \citep{zheng2026hgfe}, line-level and mobile predictors \citep{pornprasit2021jitline,marrium2026flame}, gradient-boosted decision trees \citep{ke2017lightgbm}, convolutional diff encoders such as DeepJIT \citep{hoang2019deepjit}, and bimodal code transformers such as CodeBERT \citep{feng2020codebert}. Crucially, \citet{zeng2021deep} showed that single-feature lines-added logistic regression, \texttt{LApredict}, often rivals or surpasses deep JIT-SDP models when high-dimensional representations dilute commit-size signals. Likewise, \citet{tantithamthavorn2017mysterious} underscored the importance of leak-free validation protocols. Because post-authoring Gerrit review attributes such as reviewer counts, review duration, and approvals do not exist at initial push, our primary multimodal benchmark restricts tabular inputs to the 13 commit-authoring metrics and isolates 28-metric pre-merge triage in a separate sensitivity analysis.

To trigger model updates without waiting for defect labels, \citet{pote2026adaptive} monitored penultimate neural representations with DriftLens \citep{greco2021driftlens,greco2023driftlens}. Whenever the Gaussian Fr\'echet distance \citep{dowson1982frechet} exceeded a uniform bootstrap percentile, their pipeline fine-tuned on a single $1W$ window or escalated to a four-window $4W$ slice. Concurrently, \citet{shah2026detecting} observed in passive shift monitoring that representation shifts do not invariably induce accuracy drops. Because our benchmark adds causal tabular linear-residual skips and strict commit-time feature filtering rather than re-running Pote et al.'s original repository verbatim, we treat $B_{1b}^*$ strictly as a \textit{Pote-inspired retrospective-oracle counterfactual} that isolates the effect of future-fix label visibility within matched implementations.

\subsection{Verification Latency, Stream Learners, and SZZ Evaluation Bias}
\label{sec:rel_szz}

SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework} introduces two distinct supervision challenges: permanent mislabeling from tangled or refactoring commits \citep{rosa2021evaluating,fan2021impact,keshavarz2022apachejit,bludau2026quantifying} and transient multi-month verification latency \citep{cabral2019class,tabassam2020handling,song2023humla}. Closely related to evaluation validity, \citet{song2023waiting} studied how imposing a fixed waiting time injects label noise into \textit{retrospective performance evaluation} of online JIT-SDP. Our work complements theirs by auditing how future-fix visibility at drift alarm time $t_{\text{alarm}}$ inflates \textit{weight-update supervision} rather than evaluation waiting windows.

Within the online stream-learning literature, \citet{cabral2019class} introduced Oversampling Rate Boosting ($\text{ORB}$) over OzaBagging \citep{oza2005online} of Online Hoeffding Trees \citep{domingos2000mining}. Their approach waits 90 days before treating unfixed commits as clean negatives while scaling Poisson weights via an Oversampling Boost Factor. \citet{cabral2023borb} later developed Batch-Oversampling ($\text{BORB}$), refreshing base learners every 200 commits via skewed resampling over matured pools. Similarly, \citet{song2022odasc} proposed $\text{ODaSC}$, combining a shorter 15-day provisional-negative horizon with delayed positive relabeling weighted by DenStream micro-cluster confidences.

\subsection{Delayed Feedback, Positive-Unlabeled Learning, and Maturity Estimation}
\label{sec:rel_pu}

Outside software engineering, \citet{chapelle2014modeling} formalized the delayed-feedback trade-off in display advertising: young unrevealed events cannot safely be treated as negative, yet waiting for full maturation makes models stale. Their approach jointly fits feature-dependent positive probability $p(x)$ and an exponential delay rate $\lambda(x)$. Learning from confirmed positives and unlabeled mixtures has also been studied via propensity-weighted PU estimators \citep{elkan2008learning,bekker2020learning} and non-negative PU risk ($\text{nnPU}$) \citep{kiryo2017positive}, alongside covariate-shift density ratios \citep{sugiyama2008direct} and kernel two-sample tests \citep{gretton2012kernel}. Standard $\text{nnPU}$, however, assumes a single marginal prior across unlabeled commits regardless of commit age.

Accordingly, we do \textit{not} claim to introduce the general idea of combining delay modeling with PU learning. Rather, our methodological increment is adapting delayed-feedback maturity estimation to drift-triggered, heterogeneous-host JIT-SDP. We evaluate non-parametric risk-eligible cohort incidence \citep{lyndenbell1971method}, quantized Kaplan--Meier cure \citep{kaplan1958nonparametric}, and a global unstratified Weibull mixture-cure MLE \citep{boag1949cure,farewell1982cure} against zero-fit \texttt{FixedCurve-PU} and two alarm-synchronized post-hoc $B_2\text{-4W}$ controls. Our Weibull cure variant contains the exponential law at shape $k=1$ and draws inspiration from delayed-feedback likelihoods, without jointly fitting feature-dependent $p(x)$ and $\lambda(x)$ as in \citet{chapelle2014modeling}.

% ============================================================================
% 3. Theoretical Audit of Censoring Bias & Alarm Calibration
% ============================================================================
\section{Theoretical Analysis of Right-Censored Retraining and Block Calibration}
\label{sec:theory}

Before introducing our adaptation protocol and empirical design, we formalize two mechanisms governing live drift-triggered JIT-SDP. First, we analyze how right-censored labels reshape the Bayes-optimal predictor under uniform versus heterogeneous commit maturity. Second, we show in Proposition~\ref{prop:autocorr} how positive lag autocovariance inflates contiguous window-mean variance relative to independent draws, motivating our empirical alarm-calibration factorial.

\subsection{Posterior Compression and Rank Inversion Under Right-Censored Supervision}
\label{sec:theory_poison}

Consider a chronologically ordered commit stream $\mathcal{S} = \{(x_i, t_i, y_i^*, t_i^{\text{fix}})\}_{i=1}^N$ with non-decreasing timestamps $t_1 \le \dots \le t_N$ measured in seconds. Here $x_i \in \mathcal{X}$ denotes commit-authoring features, $y_i^* \in \{0, 1\}$ is the \textit{archive-horizon defect label}, and $t_i^{\text{fix}} \ge t_i$ is the earliest SZZ bug-fixing timestamp, set to $+\infty$ when $y_i^* = 0$ or unlinked. The resulting fix delay in days is $\Delta_i = (t_i^{\text{fix}} - t_i)/86400 \in [0, +\infty]$. Because upstream datasets lack a separate annotation-completeness timestamp, we measure observation reach using the \textit{observable archive-horizon proxy} $T_{\text{end}} = \max(\max_i t_i, \max_{i:\, t_i^{\text{fix}} < \infty} t_i^{\text{fix}})$ over finite fix timestamps only. Distinguishing \textit{alarm-time right-censoring}, where $t_{\text{alarm}} < t_i^{\text{fix}} \le T_{\text{end}}$, from \textit{end-of-archive censoring}, where $t_i^{\text{fix}} > T_{\text{end}}$, is essential. We therefore audit remaining follow-up $T_{\text{end}} - t_i$ across all 34 natural repository streams and tabulate $\ge 180\text{d}$ and $\ge 365\text{d}$ sensitivity in Section~\ref{sec:benchmarks} and Appendix~\ref{app:audits}. Figure~\ref{fig:causal_timeline} contrasts offline retrospective evaluation against strictly causal CI deployment.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig0a_causal_timeline_v3.pdf}
  \caption{Causal supervision audit at drift alarm $t_{\text{alarm}}$: (a) offline retrospective evaluation ($B_{1b}^*$) leaks future SZZ fixes ($t_{\text{fix}} > t_{\text{alarm}}$) into $W_{\text{drift}}$ despite chronological commit ordering; (b) live causal CI deployment ($t \le t_{\text{alarm}}$) confronts the trade-off between delayed $\text{Wait-}90$ (dropping unfixed commits aged $<90\text{d}$) and naive real-time $B_2\text{-1W/4W}$ (treating unfixed defects as $y=0$), addressed via maturity-weighted Bayesian PU targets.}
  \label{fig:causal_timeline}
\end{figure}

\begin{definition}[Causal Label Visibility and Structural Maturity Factorization]
\label{def:observability}
At alarm timestamp $t_{\text{alarm}} = t_e$ on window $W_{\text{drift}} = \{e - W + 1, \dots, e\}$, causal deployment forbids accessing events after $t_{\text{alarm}}$. For any commit $i \le e$ of age $a_i = (t_{\text{alarm}} - t_i)/86400 \ge 0$ (days), its observable label is
\begin{equation}
  y_{i,\text{obs}}(t_{\text{alarm}}) = \mathbb{I}\!\left(y_i^* = 1 \;\wedge\; t_i^{\text{fix}} \le t_{\text{alarm}}\right) = y_i^* \cdot \mathbb{I}\!\left(\Delta_i \le a_i\right).
  \label{eq:y_obs}
\end{equation}
Assume locally on an adaptation window that (A1) latent defect risk is age-stationary given features, $\Pr(y^* = 1 \mid x_i, a_i) = \eta^*(x_i)$, and (A2) reveal delay among true defects is conditionally independent of $x_i$ given age, $\Pr(\Delta \le a_i \mid y^* = 1, x_i, a_i) = c(a_i) \in [0, 1]$. Then $\Pr(y_{\text{obs}} = 1 \mid x_i, a_i) = c(a_i)\,\eta^*(x_i)$.
\end{definition}

\begin{proposition}[Bayes-Optimal Censored Posterior Under Uniform vs.\ Heterogeneous Maturity]
\label{prop:bias}
Under Assumptions~(A1)--(A2) of Definition~\ref{def:observability}, suppose predictor $f_\theta(x)$ minimizes class-weighted binary cross-entropy $\ell(p, y_{\text{obs}}) = -\omega_+ y_{\text{obs}} \log p - (1 - y_{\text{obs}})\log(1 - p)$ ($\omega_+ > 0$) against $y_{\text{obs}}(t_{\text{alarm}})$ over a window with conditional effective maturity $\bar{c}(x) = \mathbb{E}[c(a) \mid x] \in (0, 1]$. For any $x$ with $\eta^*(x) \in (0, 1)$, the population risk minimizer is
\begin{equation}
  \hat{p}_{B_2}(x) = \frac{\omega_+ \, \bar{c}(x) \, \eta^*(x)}{1 - \bar{c}(x)\,\eta^*(x) + \omega_+\,\bar{c}(x)\,\eta^*(x)} = \frac{\omega_+ \, \bar{c}(x) \, \eta^*(x)}{1 + \big(\omega_+ - 1\big)\,\bar{c}(x)\,\eta^*(x)}.
  \label{eq:b2_bayes}
\end{equation}
Equation~\eqref{eq:b2_bayes} implies two structural regimes:
\begin{enumerate}[leftmargin=1.8em,itemsep=0.10em]
  \item \textbf{Constant-Maturity Order Preservation and Exact Threshold Crossing:} If $\bar{c}(x) \equiv \bar{c}_W \in (0, 1)$ is constant across $x$, then $\partial \hat{p}_{B_2} / \partial \eta^* = \omega_+ \bar{c}_W / [1 + (\omega_+ - 1)\bar{c}_W \eta^*]^2 > 0$: population ranking (AP, ROC-AUC) is strictly preserved. Relative to the uncensored ($\bar{c}_W = 1$) weighted minimizer $\hat{p}_{B_1^*}(x) = \omega_+ \eta^*(x)/[1 + (\omega_+ - 1)\eta^*(x)]$, right-censoring strictly reduces $\hat{p}_{B_2}(x) < \hat{p}_{B_1^*}(x)$ (for $\omega_+ = 1$, $\hat{p}_{B_2}(x) = \bar{c}_W \eta^*(x) < \eta^*(x)$), and forces $\hat{p}_{B_2}(x) < \tau_{\text{th}} \in (0, 1)$ if and only if
  \begin{equation}
    \bar{c}_W < \frac{\tau_{\text{th}}}{\eta^*(x)\,\big[\omega_+(1 - \tau_{\text{th}}) + \tau_{\text{th}}\big]}.
    \label{eq:exact_crossing}
  \end{equation}
  \item \textbf{Heterogeneous-Maturity Rank Inversion:} If two feature subpopulations $x_{\text{old}}, x_{\text{new}}$ have effective maturities $\bar{c}(x_{\text{old}}) = c_{\text{hi}} > c_{\text{lo}} = \bar{c}(x_{\text{new}}) > 0$ with $\eta^*(x_{\text{new}}) > \eta^*(x_{\text{old}})$, then $\hat{p}_{B_2}(x_{\text{new}}) < \hat{p}_{B_2}(x_{\text{old}})$ if and only if $\frac{\eta^*(x_{\text{new}})}{1 - c_{\text{lo}}\eta^*(x_{\text{new}})} < \frac{c_{\text{hi}}}{c_{\text{lo}}} \frac{\eta^*(x_{\text{old}})}{1 - c_{\text{hi}}\eta^*(x_{\text{old}})}$.
\end{enumerate}
\end{proposition}

\begin{proof}
Taking conditional expectation under $y_{\text{obs}} \mid x \sim \operatorname{Bernoulli}(\bar{c}(x)\eta^*(x))$ and setting $\partial \mathbb{E}[\ell \mid x]/\partial p = 0$ on $(0, 1)$ gives $\frac{\omega_+ \bar{c}(x)\eta^*(x)}{p} = \frac{1 - \bar{c}(x)\eta^*(x)}{1 - p}$, yielding Eq.~\eqref{eq:b2_bayes}. Differentiating with respect to $\eta^*$ for constant $\bar{c}_W$ proves Claim~1's monotonicity, while solving $\omega_+ \bar{c}_W \eta^*(x) < \tau_{\text{th}}[1 + (\omega_+ - 1)\bar{c}_W \eta^*(x)]$ yields the exact crossing bound~\eqref{eq:exact_crossing}. Comparing strictly monotone odds $\hat{p}_{B_2}/(1 - \hat{p}_{B_2}) = \omega_+ \bar{c}(x)\eta^*(x)/[1 - \bar{c}(x)\eta^*(x)]$ yields Claim~2.
\end{proof}

\textit{Remark on Scope and Evaluation Granularity.} Right-censoring does not universally worsen raw Brier error whenever $\omega_+ > 1$ inflates uncensored posteriors. For instance, at $\eta^* = 0.5$ and $\omega_+ = 4$, the uncensored weighted minimizer yields $\hat{p}_{B_1^*} = 0.80$, whereas censoring at $\bar{c}_W = 0.40$ pulls $\hat{p}_{B_2}$ back down to $0.50$. Equally important, Claim~1's monotone ranking preservation holds only \textit{within a single window's model}. When predictions from multiple post-alarm windows are concatenated into one \textit{whole-stream} Precision--Recall curve, inter-window score-scale shifts---triggered as estimated priors $\hat{\pi}_{\text{target}}(t)$ or class weights change across alarms---can penalize whole-stream AP even when within-window ranking holds or improves. We test this cross-window comparability effect directly on $K=25$ in Section~\ref{sec:rq2}. Taken together, Proposition~\ref{prop:bias} explains why $B_2\text{-4W}$ can retain competitive whole-stream AP on gradual streams yet lose thresholded recall under heavy latency and lose ranking accuracy when shifted features arrive at young ages.

\subsection{Population Variance of Contiguous vs.\ Independent Draws Under Autocorrelation}
\label{sec:theory_alarm_storm}

Because every triggered alarm exposes the model to right-censored supervision, calibrating the drift monitor against spurious alarms is tightly coupled to update reliability. Software commits arrive in serially correlated development bursts such as feature sprints, subsystem refactorings, and release freezes, violating the i.i.d.\ sampling assumption behind uniform bootstrap calibration.

\begin{proposition}[Population Window-Mean Covariance Under Positive Lag Autocovariance]
\label{prop:autocorr}
Let $\{z_t\}_{t \in \mathbb{Z}} \subset \mathbb{R}^d$ be a weakly stationary embedding sequence with mean $\mu_0$, marginal covariance $\Gamma_0 = \operatorname{Cov}(z_t)$, and lag-$k$ autocovariance $\Gamma_k = \operatorname{Cov}(z_t, z_{t+k})$ satisfying $\Delta_\Gamma \triangleq \sum_{k=1}^{W-1}\frac{W-k}{W}(\Gamma_k + \Gamma_k^\top) \succ 0$. Comparing a contiguous window mean $\bar{z}_{\text{contig}} = \frac{1}{W}\sum_{j=1}^W z_{t_0+j}$ against the mean $\bar{z}_{\text{ind}} = \frac{1}{W}\sum_{j=1}^W \tilde{z}_j$ of $W$ independent draws $\tilde{z}_j$ from the same marginal law, $\operatorname{Cov}(\bar{z}_{\text{contig}}) = \operatorname{Cov}(\bar{z}_{\text{ind}}) + \frac{1}{W}\Delta_\Gamma \succ \operatorname{Cov}(\bar{z}_{\text{ind}})$ and $\mathbb{E}\|\bar{z}_{\text{contig}} - \mu_0\|_2^2 > \mathbb{E}\|\bar{z}_{\text{ind}} - \mu_0\|_2^2$.
\end{proposition}

\begin{proof}
Expanding $\operatorname{Cov}(\frac{1}{W}\sum_{j=1}^W z_{t_0+j}) = \frac{1}{W}\Gamma_0 + \frac{1}{W^2}\sum_{k=1}^{W-1}(W-k)(\Gamma_k + \Gamma_k^\top)$ whereas $\operatorname{Cov}(\bar{z}_{\text{ind}}) = \frac{1}{W}\Gamma_0$ gives difference $\frac{1}{W}\Delta_\Gamma \succ 0$ and trace gap $\frac{1}{W}\operatorname{tr}(\Delta_\Gamma) > 0$.
\end{proof}

In finite calibration pools of size $N_{\text{cal}}$, uniform bootstrap resampling replaces $\Gamma_0$ with the empirical pool covariance $\hat{\Sigma}_{\text{pool}}$ while breaking lag-$k$ sprint serial correlation for $k \ge 1$ \citep{kunsch1989jackknife,politis1992circular}. Although Proposition~\ref{prop:autocorr} characterizes the mean-shift component rather than the full nonlinear Gaussian Fr\'echet statistic, it motivates calibrating DriftLens with circular contiguous blocks of length $b = \max(\min(\lfloor W/6 \rfloor, 80), 25)$. To separate bootstrap sampling from post-alarm updating without confounding percentile or cooldown rules, RQ4b evaluates a clean $2 \times 2$ factorial at fixed $d_{\text{pca}}=24$, $p_{95}$, and $\text{cooldown}=1$. The factorial crosses \texttt{IID} versus \texttt{Circular Block} resampling against \texttt{Static} versus \texttt{Rolling Ref+Th} updating, alongside Pote et al.'s static $p_{98}$ reference with zero cooldown. Under \texttt{Rolling Ref+Th}, each fired alarm refreshes reference statistics on the trailing $3W$ commits and blends the recalibrated $p_{95}$ threshold via $0.70\,\tau_{\text{new}} + 0.30\,\tau_{\text{old}}$.

% ============================================================================
% 4. Methodology: Three-Stage Survival-PU Adaptation Protocol
% ============================================================================
\section{Methodology: Three-Stage Survival-PU Adaptation Protocol}
\label{sec:method}

Proposition~\ref{prop:bias} indicates that supervision-reliable real-time adaptation at alarm timestamp $t_{\text{alarm}}$ must resolve three coupled obstacles using only past repository state $t_i \le t_{\text{alarm}}$. First, it must estimate commit-age reveal maturity $c(a_i)$ without conditioning naively on fast-revealing fixes. Second, it must estimate target defect prevalence $\hat{\pi}_{\text{target}}$ and train a reliable teacher ranking without exploding variance on young commits. Third, it must train on the full adaptation pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ with soft targets that preserve fresh drift signals while preventing posterior collapse.

Figure~\ref{fig:architecture} outlines the resulting gate-free three-stage \textbf{Survival-PU} pipeline, and Table~\ref{tab:hyperparams} lists every locked protocol parameter frozen before unlocking the confirmatory repositories. Exact code symbol mappings are documented in Appendix~\ref{app:repro}. Preliminary heuristic three-state safety gates are excluded from the core method and evaluated solely as an ablation control, \texttt{Survival-PU (+Legacy Gate)}.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig0b_survival_pu_architecture_v3.pdf}
  \caption{Three-stage input--output pipeline of Survival-PU at each DriftLens alarm $t_{\text{alarm}}$: (Stage~1) risk-eligible cohort cumulative incidence $c_i$ on trailing history $\mathcal{H}_{\text{rec}}$; (Stage~2) reliable teacher $T_\phi$ ($85/15$ blend initialization on neural hosts; fresh $75$-tree fit on LightGBM) and clipped inverse-maturity shrunk prevalence $\hat{\pi}_{\text{target}}$ ($\tilde{c}_i = \max(c_i, 0.10)$) on $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$; and (Stage~3) monotone bisection on $\mathcal{B}_{\text{cal}}$ producing maturity- and prior-adjusted soft targets $q_i(\delta^*)$ across $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$.}
  \label{fig:architecture}
\end{figure}

\begin{table}[!htb]
\centering
\caption{Locked protocol parameters of Survival-PU and DriftLens frozen prior to unlocking the confirmatory repositories (exact module and configuration object mappings appear in Appendix~\ref{app:repro}).}
\label{tab:hyperparams}
\fontsize{6.0}{7.2}\selectfont
\renewcommand{\arraystretch}{1.02}
\setlength{\tabcolsep}{1.8pt}
\begin{tabularx}{\textwidth}{lccL{2.95cm}Y}
\toprule
\textbf{Module} & \textbf{Symbol} & \textbf{Default} & \textbf{Config Field} & \textbf{Executed Statistical / Operational Role} \\
\midrule
Stage 1 & $N_{\text{hist}}, q_{\max}$ & $6{,}000,\; 0.85$ & \texttt{max\_\allowbreak history\_\allowbreak cohort}, \texttt{support\_\allowbreak age\_\allowbreak quantile} & Trailing cohort slice $[-N_{\text{hist}}:]$ and $85\%$ age support $a_{\max}$ \\
Stage 1 & $n_{\min}^{\text{risk}}, n_{\min}^{\text{fix}}$ & $120,\; 15$ & \texttt{min\_\allowbreak risk\_\allowbreak set}, \texttt{min\_\allowbreak observed\_\allowbreak fixes} & Risk-set floor ($\ge 25$) and fix floor before $0.88(1 - 2^{-a/90})$ fallback \\
Stage 1 & $[c_{\min}, c_{\max}]$ & $[0.05, 0.96]$ & \texttt{c\_min}, \texttt{c\_max} & Pre- and post-interpolation maturity bounds across Stage~1 estimators \\
Stage 1 & $[\gamma_{\min}, \gamma_{\max}]$ & $[0.60, 0.94]$ & \texttt{tail\_\allowbreak cap\_\allowbreak min}, \texttt{tail\_\allowbreak cap\_\allowbreak max} & Finite-horizon tail-completeness cap bounds (swept in RQ4d) \\
Stage 2 & $\tau_{\text{rel}}, N_{\text{rel}}$ & $0.45,\; 2{,}200$ & \texttt{reliable\_\allowbreak maturity\_\allowbreak thresh}, \texttt{max\_\allowbreak reliable\_\allowbreak commits} & Reliable cohort $\mathcal{R} = \{y_{\text{obs}}=1 \vee c_i \ge 0.45\}$ trailing cap \\
Stage 2 & $[w_{\min}, w_{\max}]$ & $[0.50, 2.50]$ & \texttt{imp\_\allowbreak weight\_\allowbreak min}, \texttt{imp\_\allowbreak weight\_\allowbreak max} & $L_2$-logistic domain density-ratio clipping bounds \\
Stage 2 & $c_{\text{ht}}, m_0, [\pi_{\min}, \pi_{\max}]$ & $0.10, 50, [0.02, 0.70]$ & \texttt{ht\_\allowbreak floor\_\allowbreak c}, \texttt{ht\_\allowbreak shrinkage\_\allowbreak m0}, \texttt{pi\_min}, \texttt{pi\_max} & Inverse-maturity floor $\tilde{c}_i = \max(c_i, 0.10)$ and $n_{\text{eff,heur}}$ shrinkage on $\mathcal{B}_{\text{cal}}$ \\
Stage 2 & $\tau_{\text{sharp}}, \beta_{\text{LLM}}$ & $1.15,\; 0.25$ & \texttt{teacher\_\allowbreak sharpness}, \texttt{llm\_\allowbreak blend\_\allowbreak weight} & Teacher prior-adjusted logit scaling and optional RQ4e LLM blend \\
Stage 3 & $N_{\text{anc}}, \gamma_{\text{drift}}, u_0$ & $2{,}000,\; 1.40,\; 0.45$ & \texttt{max\_\allowbreak anchor\_\allowbreak commits}, \texttt{drift\_\allowbreak emphasis\_\allowbreak gamma}, \texttt{maturity\_\allowbreak weight\_\allowbreak floor} & Non-drift anchor cap, $W_{\text{drift}}$ emphasis, and maturity weight floor $u_i$ \\
Stage 3 & $[p_{\min}, p_{\max}], K_{\text{bis}}$ & $[0.003, 0.997],\; 28$ & \texttt{posterior\_\allowbreak p\_\allowbreak min}, \texttt{posterior\_\allowbreak p\_\allowbreak max}, \texttt{bisection\_\allowbreak iters} & Shared posterior clamp and monotone bisection iterations on $[-8, 8]$ \\
Budgets & $R_{\text{init}}, R_{\text{alm}}$ & $90,\; 75$ & \texttt{lgb\_\allowbreak init\_\allowbreak rounds}, \texttt{lgb\_\allowbreak alarm\_\allowbreak rounds} & LightGBM init ($90$) and fresh per-stage alarm ($75$; Teach-Matched $=150$) rounds \\
Budgets & $E_{\text{teach}}, E_{\text{stud}}, \eta$ & $4,\; 4,\; 4.5\text{e-}4$ & \texttt{neural\_\allowbreak teacher\_\allowbreak epochs}, \texttt{neural\_\allowbreak student\_\allowbreak epochs}, \texttt{neural\_\allowbreak adapt\_\allowbreak lr} & Neural SPU ($4+4$) epochs (init $=12$, $B_2$/Wait $=5$, $\text{nnPU}=6$) \\
Monitor & $W, d_{\text{pca}}, p_{\text{blk}}, \text{cd}$ & $400,\; 24,\; 95.0,\; 1$ & \texttt{window\_\allowbreak size\_\allowbreak default}, \texttt{drift\_\allowbreak pca\_\allowbreak dim}, \texttt{drift\_\allowbreak block\_\allowbreak pct}, \texttt{drift\_\allowbreak cooldown\_\allowbreak windows} & Default window size, PCA dim, percentile, and cooldown ($2\times 2$ factorial) \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Stage 1: Risk-Eligible Cohort Cumulative Incidence and Survival Controls}
\label{sec:stage1}

A naive empirical CDF computed solely over observed bug fixes severely underestimates fix latency because slow-revealing bugs are systematically absent from recent commits. To avoid this right-truncation bias, Stage~1 first slices the trailing $N_{\text{hist}}$ causal commits $\mathcal{H}_{\text{rec}} \subset \{i : t_i \le t_{\text{alarm}}\}$ before computing age quantiles. For each commit in $\mathcal{H}_{\text{rec}}$, it records commit age $a_i = \max((t_{\text{alarm}} - t_i)/86400, 0)$ and observed fix delay $d_i = \max((t_i^{\text{fix}} - t_i)/86400, 0)$ whenever $y_{i,\text{obs}}(t_{\text{alarm}}) = 1$; unrevealed commits and any quarantined retroactive links $t_i^{\text{fix}} < t_i$ receive $d_i = +\infty$.

We evaluate candidate age knots $g_0 = 0 < g_1 < \dots < g_M = a_{\max}$ up to the support quantile $a_{\max} = Q_{q_{\max}}(\{a_i\}_{i \in \mathcal{H}_{\text{rec}}})$, enforcing an effective risk-set floor of $n_{\text{eff}}^{\text{risk}} = \max(\min(n_{\min}^{\text{risk}}, \lfloor (1 - q_{\max})|\mathcal{H}_{\text{rec}}| \rfloor), 25)$. At each knot $g_m$, the \textbf{Risk-Eligible Cohort Cumulative Incidence Estimator} restricts the denominator to commits old enough to have been observed at lag $g_m$, namely $\mathcal{E}(g_m) = \{i \in \mathcal{H}_{\text{rec}} : a_i \ge g_m\}$. Whenever $|\mathcal{E}(g_m)| < n_{\text{eff}}^{\text{risk}}$ for $m > 0$, it carries forward $\hat{G}(g_{m-1})$, and monotonicity is enforced via a running maximum:
\begin{equation}
  \hat{G}(g_m) = \frac{1}{|\mathcal{E}(g_m)|}\sum_{i \in \mathcal{E}(g_m)} \mathbb{I}\!\left(y_{i,\text{obs}} = 1 \;\wedge\; d_i \le g_m\right), \qquad \tilde{G}(g_m) = \max_{0 \le k \le m}\hat{G}(g_k).
  \label{eq:cohort_inc}
\end{equation}

Because $\tilde{G}(a_{\max})$ still reflects finite-horizon observation up to $a_{\max}$, Stage~1 rescales the normalized curve by a bounded tail-completeness factor $\gamma_{\text{tail}} \in [\gamma_{\min}, \gamma_{\max}]$. Letting $\tilde{G}_{\text{half}} = \operatorname{interp}(0.5\,a_{\max}; \{g_m\}, \{\tilde{G}(g_m)\})$, we define the growth ratio $r_g = \tilde{G}(a_{\max}) / \tilde{G}_{\text{half}}$ when $\tilde{G}_{\text{half}} > 10^{-5}$ and $r_g = 1.5$ otherwise. Combining $r_g$ with the age factor $\phi_a = \operatorname{clip}(a_{\max}/(a_{\max}+45),\, 0.65,\, 1.0)$ yields $\gamma_{\text{tail}} = \operatorname{clip}([\gamma_{\max} - 0.28\operatorname{clip}(r_g - 1, 0, 1)]\,\phi_a,\, \gamma_{\min},\, \gamma_{\max})$. Knot maturities are clipped \textit{before} piecewise-linear interpolation across knots:
\begin{equation}
  \begin{aligned}
    c_{\text{knot}}(g_m) &= \operatorname{clip}\!\Big(\gamma_{\text{tail}}\,\frac{\tilde{G}(g_m)}{\tilde{G}(a_{\max})},\; c_{\min},\; c_{\max}\Big), \\
    c_i &= \operatorname{clip}\!\big(\operatorname{interp}(a_i;\, \{g_m\}, \{c_{\text{knot}}(g_m)\}),\; c_{\min},\; c_{\max}\big),
  \end{aligned}
  \label{eq:ci_interp}
\end{equation}
with left boundary $c_{\min}$ and right boundary $c_{\text{knot}}(g_M)$. Whenever historical support is sparse---specifically when observed fixes fall below $n_{\min}^{\text{fix}}$, $|\mathcal{H}_{\text{rec}}| < 200$, $a_{\max} < 30\text{d}$, or $\tilde{G}(a_{\max}) < 10^{-4}$---the estimator triggers a deterministic fallback. This fallback evaluates $0.88(1 - 2^{-g/90})$ on $145$ uniform knots spanning $[0, 720]$ days at 5-day spacing, clips knot values to $[c_{\min}, c_{\max}]$, and linearly interpolates at $a_i$, holding the terminal 720-day value for older commits.

To test whether learned survival curves improve upon classical survival estimators or a zero-fit parametric prior, RQ2 and RQ3 compare three alternatives under the identical Stage~1 interface. The first is right-censored \textbf{Kaplan--Meier Cure}, which quantizes event and censoring times onto a shared $10^{-4}$-day grid of roughly $8.6\text{ s}$ so the risk-set invariant $0 \le e_u \le n_u$ holds identically. The second is L-BFGS-B \textbf{Weibull Mixture-Cure MLE} \citep{boag1949cure,farewell1982cure} fitted on unstratified $\mathcal{H}_{\text{rec}}$, a global delay family containing the exponential law at $k=1$ rather than the feature-dependent joint model of \citet{chapelle2014modeling}. The third is zero-fit \textbf{FixedCurve-PU}, which directly sets $c_{\text{fix}}(a_i) = \operatorname{clip}(1 - 2^{-a_i/90},\, c_{\min},\, c_{\max})$.

\textit{Identifiability Assumptions of Stage~1.} Because the denominator of Eq.~\eqref{eq:cohort_inc} spans all age-eligible commits, $\hat{G}(a)$ estimates the joint probability $\pi F_\Delta(a)$ rather than conditional maturity $c(a) = F_\Delta(a)$ directly. Canceling $\pi$ in Eq.~\eqref{eq:ci_interp} therefore requires two structural assumptions: (I1) trailing-cohort defect prevalence $\pi$ and delay distribution $F_\Delta$ are locally stationary across commit ages inside $\mathcal{H}_{\text{rec}}$, and (I2) $\gamma_{\text{tail}}$ approximates terminal completeness at $a_{\max}$. Under abrupt prevalence shifts across cohorts, $\hat{G}(a)$ can confound lower defect prevalence with slower reveal---an error mode we audit directly against post hoc empirical $c^*(a)$ across 62 fired alarms in Section~\ref{sec:rq3}.

\subsection{Stage 2: Clipped Inverse-Maturity Prevalence on \texorpdfstring{$\mathcal{B}_{\text{cal}}$}{B\_cal} and Reliable Teacher}
\label{sec:stage2}

At each alarm, we assemble the adaptation pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ by combining the trailing $N_{\text{anc}}$ non-drift commits $\mathcal{A}_{\text{anc}}$ with the newly flagged window $W_{\text{drift}}$. To account for covariate shift between historical anchors and the shifted window, an $L_2$-regularized domain logistic classifier separates $\mathcal{A}_{\text{anc}}$ from $W_{\text{drift}}$ to produce bounded importance weights $w_{\text{imp}}(x_i) \in [w_{\min}, w_{\max}]$ \citep{sugiyama2008direct}. On $\mathcal{A}_{\text{anc}}$, multiplying $w_{\text{imp}}(x_i)$ by linear recency in $[0.70, 1.15]$ yields base weights $w_i$, while commits in $W_{\text{drift}}$ receive uniform emphasis $w_i = \gamma_{\text{drift}}$. Weighting by maturity then gives observation-independent sample weights $u_i = w_i[u_0 + (1 - u_0)c_i]$.

Dividing observed labels $y_{i,\text{obs}}$ directly by $c_i$ incurs severe variance on very young commits where $c_i \approx 0.05$. Stage~2 therefore operates on the trailing calibration subset $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$ of size $N_{\text{cal}} = \min(|\mathcal{B}|, \max(3|W_{\text{drift}}|, 900))$ using an inverse-maturity floor $\tilde{c}_i = \max(c_i, c_{\text{ht}})$ with $c_{\text{ht}} = 0.10$. It combines a clipped inverse-maturity plug-in estimate with heuristic variance-inflation shrinkage toward a mature anchor prior $\pi_{\text{anc}}$:
\begin{equation}
  \begin{aligned}
    \hat{\pi}_{\text{IP}} &= \frac{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\,(y_{i,\text{obs}} / \tilde{c}_i)}{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i}, \qquad n_{\text{eff,heur}} = \frac{\big(\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\big)^2}{\sum_{i \in \mathcal{B}_{\text{cal}}} (u_i / \tilde{c}_i)^2}, \\
    \hat{\pi}_{\text{target}} &= \operatorname{clip}\!\left(\max\!\left(\frac{n_{\text{eff,heur}}\,\hat{\pi}_{\text{IP}} + m_0\,\pi_{\text{anc}}}{n_{\text{eff,heur}} + m_0},\; \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4}\right),\; \pi_{\min},\; \pi_{\max}\right).
  \end{aligned}
  \label{eq:ht_prev}
\end{equation}
Here $\pi_{\text{anc}}$ is the inverse-maturity mean over mature history $\{j : c_j \ge 0.55\}$ whenever at least 80 commits qualify, falling back to the initial pre-stream prevalence otherwise, and $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} = \sum_{i \in \mathcal{B}_{\text{cal}}} u_i y_{i,\text{obs}} / \sum_{i \in \mathcal{B}_{\text{cal}}} u_i$. Commits with $c_i < c_{\text{ht}}$ incur conservative downward attenuation by $c_i / c_{\text{ht}}$. Equation~\eqref{eq:ht_prev} also carries a concrete boundary condition: because the outer clamp $[\pi_{\min}, \pi_{\max}] = [0.02, 0.70]$ wraps $\max(\cdot, \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4})$, whenever observed prevalence $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ exceeds $0.70$, the target is capped below the observed positive rate and becomes algebraically unreachable in Stage~3. We audit the empirical frequency of this boundary below.

In parallel, Stage~2 trains a teacher classifier $T_\phi$ on the trailing $N_{\text{rel}}$ commits of the reliable cohort $\mathcal{R} = \{i : y_{i,\text{obs}} = 1 \vee c_i \ge \tau_{\text{rel}}\}$ whenever $|\mathcal{R}| \ge 100$, preventing unfixed low-maturity commits from dominating teacher feature weights. Teacher samples carry weight $w_{\text{rel},i} = w_{\text{imp}}(x_i)\cdot r_{\text{conf},i}\cdot \text{recency}_i$, where $\text{recency}_i \in [0.75, 1.15]$ and $r_{\text{conf},i} = 0.55 + 0.45 c_i$ for young revealed positives with $c_i < \tau_{\text{rel}}$ and $1$ otherwise. Positive class weight is set to $\omega_{+,\text{teach}} = \frac{1}{2}\omega_+ + \frac{1}{2}\operatorname{clip}((1-\hat{\pi}_{\text{target}})/\max(\hat{\pi}_{\text{target}}, 0.03), 1.5, \max(4.5, \omega_+))$.

Host initialization follows each model family's native paradigm. On \textbf{neural hosts}, $T_\phi$ starts from an $85\%/15\%$ parameter blend of the current checkpoint and the initial pre-stream anchor, then trains for $E_{\text{teach}}$ epochs. On \textbf{LightGBM}, the teacher fits a fresh $R_{\text{alm}}$-tree booster without inheriting prior trees or margins. Both hosts emit class-weight-adjusted base log-odds $\eta_i^{(0)} = \tau_{\text{sharp}}(z_{\text{teach}}(x_i) - \log \omega_{+,\text{teach}})$ across $\mathcal{B}$. Crucially, subtracting $\log \omega_{+,\text{teach}}$ removes explicit class reweighting, yet does \textit{not} eliminate selective-cohort inclusion bias on young commits or residual censoring among commits with $c_i \ge 0.45$. Accordingly, Stage~3 treats $q_i(\delta^*)$ strictly as \textit{maturity- and prior-adjusted soft targets} rather than exact true posteriors, and we audit their calibration by commit age in Section~\ref{sec:rq3}.

\subsection{Stage 3: Maturity- and Prior-Adjusted Soft Targets, Linear-Residual Skip, and Callable LightGBM}
\label{sec:stage3}

To transfer matured historical signals onto the entire adaptation pool $\mathcal{B}$, including fresh commits inside $W_{\text{drift}}$, Stage~3 builds soft targets through one shared mapping. It fixes $q_i(\delta) = 1$ whenever $y_{i,\text{obs}} = 1$, and assigns every unrevealed commit with $y_{i,\text{obs}} = 0$ the target
\begin{equation}
  q_i(\delta) = \frac{\tilde{p}_i(\delta)\,(1 - c_i)}{1 - \tilde{p}_i(\delta)\,c_i}, \qquad \tilde{p}_i(\delta) = \operatorname{clip}\!\big(\sigma(\eta_i^{(0)} + \delta),\; p_{\min},\; p_{\max}\big).
  \label{eq:bayes_pu}
\end{equation}
At fixed maturity $c_i$, $q_i(\delta)$ is nondecreasing in $\eta_i^{(0)}$ and strictly increasing on the unclipped interior $\sigma(\eta_i^{(0)} + \delta) \in (p_{\min}, p_{\max})$. Across heterogeneous ages, Eq.~\eqref{eq:bayes_pu} naturally downweights aged unfixed commits, driving $q_i \to 0$ as $c_i \to 1$, relative to fresh unfixed commits where $q_i \to \tilde{p}_i$ as $c_i \to 0$.

Letting $f(\delta) = \sum_{i \in \mathcal{B}_{\text{cal}}} u_i q_i(\delta) / \sum_{i \in \mathcal{B}_{\text{cal}}} u_i$, $K_{\text{bis}}$-step monotone bisection on $[-8, 8]$ targets $f(\delta^*) = \hat{\pi}_{\text{target}}$ whenever $\hat{\pi}_{\text{target}}$ lies inside $[f(-8), f(+8)]$. If $\hat{\pi}_{\text{target}}$ falls outside this reachable interval---for example when alarm-time $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ exceeds $\pi_{\max} = 0.70$ or teacher logits saturate---the solver returns the nearest boundary $\delta^* \in \{-8, +8\}$ and logs an unreachable solve. Auditing every run confirms zero unreachable solves on Benchmark~A and RQ4a--e. Across all 27 Benchmark~B natural streams, only $0.62\%$ of alarms ($1.0 / 161.7$) were unreachable, concentrated in \texttt{apache\_flink} at $1.5\%$ and \texttt{apache\_groovy} at $16.7\%$, where $0.9$ survival fallbacks also occurred. On Confirmatory \texttt{HDFS*}, however, elevated alarm-time $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ triggered unreachable clamping on $28.6\%$ of MLP alarms and $30.0\%$ of LightGBM alarms. Production deployments should therefore raise $\pi_{\max}$ above $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ or project $\hat{\pi}_{\text{target}}$ directly onto $[f(-8), f(+8)]$.

Once solved, float32 soft targets $q_i(\delta^*)$ train the student model across $\mathcal{B}$ under $\mathcal{L}_{\text{SPU}}(z) = \sum_{i \in \mathcal{B}} u_i [\omega_{+,\text{teach}} q_i(\delta^*) \log(1 + e^{-z_i}) + (1 - q_i(\delta^*))\log(1 + e^{z_i})]$. Neural students initialize from $T_\phi$ and train for $E_{\text{stud}}$ epochs, whereas LightGBM fits a fresh $R_{\text{alm}}$-tree booster so total trees $R_{\text{alm}} + R_{\text{alm}}$ match single-stage $2R_{\text{alm}}$-tree \texttt{Teacher-Only (Matched)}. Two host-level mechanisms ensure consistent execution across neural encoders and gradient-boosted trees:

\begin{enumerate}[leftmargin=1.8em,itemsep=0.10em]
  \item \textbf{Causal Tabular Linear-Residual Skip Connection.} High-dimensional diff and message encoders can dilute commit-size predictors such as \texttt{la}, \texttt{nf}, and \texttt{ent} \citep{zeng2021deep}. Our multimodal neural hosts therefore add a residual skip $z(x) = w_{\text{LR}}^\top x_{\text{tab}} + b_{\text{LR}} + f_\theta(x)$ with a zero-initialized output head on $f_\theta$. Scalers and vocabularies fit strictly on $[0, n_{\text{init}})$, while $(w_{\text{LR}}, b_{\text{LR}})$ are fitted by $L_2$-regularized logistic regression on $[0, n_{\text{init}}+n_{\text{val}})$ using $y_{\text{obs}}(t_{\text{start}})$. Pre-stream validation favored a frozen linear skip over a trainable skip, at $0.0586$ versus $0.0575$ validation AP.
  \item \textbf{Analytical Callable Objective for Gradient-Boosted Trees \citep{friedman2001greedy,ke2017lightgbm}.} Rather than duplicating rows $2|\mathcal{B}|$ times, our LightGBM host supplies exact analytical first and second derivatives $(g_i, h_i)$ for continuous soft targets and propagates the initial margin $z_{\text{init}} = \log(\bar{p}_0 / (1 - \bar{p}_0))$ identically to inference. Derivation and verification details appear in Appendix~\ref{app:repro}, Proposition~\ref{prop:lgbm}.
\end{enumerate}

% ============================================================================
% 5. Experimental Design & Statistical Protocol
% ============================================================================
\section{Experimental Design and Evaluation Protocol}
\label{sec:setup}

To separate development analysis from out-of-sample replication, we organize evaluation into three benchmark tiers, freeze all modeling and statistical code before unlocking an initially absent confirmatory cohort, and verify pipeline prefix invariance across every method branch. Throughout the paper, we distinguish three protocol scopes: (i) \textit{protocol-specified development analyses} on Benchmark~A and Benchmark~B, whose comparison families and decision rules were codified in code prior to confirmatory unlock rather than externally timestamp-preregistered before all development work; (ii) \textit{pre-unlock frozen confirmatory evaluation} on five ApacheJIT repositories; and (iii) one \textit{post-unlock split-sizing amendment} for small streams with $N < 4{,}000$, disclosed explicitly below.

\subsection{Provenance-Audited Stream Tiers, Data-Construction Funnels, and Archive Follow-Up}
\label{sec:benchmarks}

Our evaluation spans \textbf{37 development streams}---six Benchmark~A multimodal streams, 27 natural Benchmark~B streams, and four Benchmark~B controlled concatenated shifts across 89 archived bundles---plus \textbf{5 confirmatory ApacheJIT repositories}. Together these comprise 347,122 natural repository commit records alongside a paired Code-LLM prompting case study.

\textbf{Benchmark A: Multimodal Qt and OpenStack (6 Streams).} We cross-linked DeepJIT corpora \citep{hoang2019deepjit}, the commit tables of \citet{mcintosh2018are}, and 16 official Git repositories to assemble 25,704 Qt commits and 13,304 OpenStack commits. This linkage verified Tier~1+2 cryptographic SZZ bug-fixing timestamps for \textbf{96.49\%} of Qt defects ($1{,}761 / 1{,}825$; median lag $163.1\text{d}$) and \textbf{99.32\%} of OpenStack defects ($1{,}616 / 1{,}627$; median lag $178.7\text{d}$), quarantining heuristic Tier-3 matches and unlinked defects to $t_{\text{fix}} = +\infty$. Primary streams retain \textbf{only the 13 commit-authoring metrics} (\texttt{la}, \texttt{ld}, \texttt{nf}, \texttt{nd}, \texttt{ns}, \texttt{ent}, \texttt{ndev}, \texttt{age}, \texttt{nuc}, \texttt{aexp}, \texttt{arexp}, \texttt{asexp}, \texttt{asawr}) and exclude 15 post-authoring Gerrit review fields, which are isolated in pre-merge sensitivity streams. Because raw Qt and OpenStack tables provide only \texttt{author\_date} without Gerrit patch-set-1 upload or merge timestamps, Benchmark~A models \textit{archival merged-commit replay} rather than initial unreviewed patch-set-1 pushes. Empty diffs---$1.02\%$ in Qt and $0.08\%$ in OpenStack---are encoded strictly as empty strings without message substitution. Across two natural ecosystems and four controlled subsystem shifts, we run five seeds of \textbf{Attention-CNN} \citep{pote2026adaptive}, \textbf{CodeBERT} \citep{feng2020codebert}, \textbf{DeepJIT-CNN} \citep{hoang2019deepjit}, and \textbf{LightGBM} \citep{ke2017lightgbm}, together with all 11 Track~2 baselines.

\textbf{Benchmark B: 27 Natural Streams ($K=25$ Donor Clusters) and 4 Controlled Shifts.} Benchmark~B combines 14 CommitGuru streams from \citet{cabral2019class}, eight ApacheJIT development streams \citep{keshavarz2022apachejit}, and five PR-SZZ streams from \citet{bludau2026quantifying}, totaling 300,637 natural commits. PR-SZZ streams are ordered by Git committer date after removing 13 post-closure columns and mapping post-2021H1 sentinels to $+\infty$. Merging Apache Camel across CommitGuru and ApacheJIT, and merging OpenStack Nova and Neutron, maps the 27 natural streams onto \textbf{$K=25$ independent donor clusters}, with a Benchmark-A-disjoint \textbf{$K=24$} sensitivity check. Four \textit{controlled concatenated distribution-shift streams}---Nova$\to$Zulip, Ignite$\to$Cassandra, Airflow$\to$Pulsar, and OpenStack Nova$\to$Neutron---are kept strictly outside $K=25$ natural inference and evaluated over 10 MLP seeds, all Track~2 learners, and 5 LightGBM seeds. To splice a pre-shift prefix and post-shift suffix, all post-onset commit timestamps and finite fix timestamps shift by $\Delta t = t_{\text{pre,last}} - t_{\text{post,first}} + 3{,}600\text{ s}$, leaving a fixed 1-hour gap while preserving $+\infty$. Because $23.7\%\text{--}60.0\%$ of revealed pre-onset fixes continue arriving after the boundary (Appendix Table~\ref{tab:shift_4streams}), these four streams act as controlled cross-project shift benchmarks rather than natural single-repository drift.

\textbf{Phase C: Confirmatory ApacheJIT Cohort ($K=4$ Umbrellas / 5 Repositories) and Upstream Monorepo Disclosure.} Before downloading official \texttt{apachejit\_total.csv} containing 106,674 rows \citep{keshavarz2022apachejit}, we froze all modeling and statistical code and passed a 14-check pre-unlock gate. Ingestion applied only a single chronological cutoff of $\text{author\_date} \le \text{2019-01-01}$, matching Benchmark~B's eight ApacheJIT streams so every evaluated commit retained at least one year of follow-up, yielding 7,477 commits across five repositories.

Auditing raw \texttt{apachejit\_total.csv} (Appendix Table~\ref{tab:confirmatory_funnel}) explains why \texttt{HDFS*} and \texttt{MapReduce*} appear smaller and far more defect-dense than in ApacheJIT Table~2, reaching evaluation prevalences of $89.2\%$ and $78.8\%$. Upstream \texttt{apachejit\_total.csv} assigns exactly $11{,}964$ clean Hadoop monorepo commits---with zero buggy commits---to the umbrella key \texttt{apache/hadoop}, creating class-dependent clean-commit omission inside \texttt{apache/hadoop-hdfs} and \texttt{apache/hadoop-mapreduce}. Those two slices therefore reflect an archive artifact with omitted negatives rather than natural repository base rates. By contrast, \texttt{Spark}, \texttt{Zeppelin}, and \texttt{ZooKeeper} undergo no monorepo clean-commit split and retain $44.6\%\text{--}48.5\%$ cut prevalence. We therefore foreground this post-audit \textbf{Standalone-3} subgroup as the cleaner external validation check while reporting the frozen-protocol \textbf{Umbrella-4} record transparently in Section~\ref{sec:rq2}.

We also disclose one \textit{post-unlock split-sizing exception}. Upon inspecting the unlocked repository sizes before confirmatory training, four repositories had $N < 2{,}000$, violating the $N \ge 4{,}000$ rule's $1{,}500 + 500$ floor. We therefore added an $N < 4{,}000$ sizing branch setting $n_{\text{init}} = \operatorname{clip}(\operatorname{round}(0.25N/25)\cdot 25, 150, 1500)$ and $n_{\text{val}} = \operatorname{clip}(\operatorname{round}(0.08N/25)\cdot 25, 50, 500)$ without altering any development bundle or method parameter; exact per-repository split sizes appear in Appendix Table~\ref{tab:confirmatory_funnel}. Because exact two-sided Wilcoxon inference at $K=4$ has $p_{\min} = 0.125 > 0.05$, Phase~C serves strictly as a directional out-of-sample replication check.

\textbf{Observable Archive-Horizon Follow-Up Audit.} To separate alarm-time censoring from end-of-archive censoring of $y_i^*$, Appendix Table~\ref{tab:followup_audit} audits remaining follow-up $T_{\text{end}} - t_i$ across all \textbf{34 natural repository streams}. Every evaluated commit in Cabral19 and Confirmatory ApacheJIT, along with over $99.3\%$ of Benchmark~B ApacheJIT commits, retains at least 302--389 days of follow-up. Four of five PR-SZZ streams retain at least $343.6$ days, with only \texttt{n60\_angular} dropping to a minimum of $156.9$ days. Benchmark~A natural Qt extends all the way to $T_{\text{end}}$, leaving $15.5\%$ of evaluation commits with under 180 days and $41.0\%$ with under 365 days of follow-up, while OpenStack reaches a minimum of $174.7$ days with $30.4\%$ under 365 days. Appendix Table~\ref{tab:followup_sensitivity} therefore re-evaluates all core Benchmark~A methods on commits with at least 180 and 365 days before $T_{\text{end}}$. Across all six reported slices, \texttt{Full-LR} retains its AP lead and \texttt{Survival-PU} preserves its Track-1 $F_1$ gain over $B_2\text{-4W}$ and $B_0$, while edging \texttt{Full-LR} on Qt $F_1$. Some pairwise AP orderings shift under follow-up restriction: on OpenStack $\ge 365\text{d}$ AP, \texttt{Survival-PU} ($0.3903$) overtakes $B_2\text{-4W}$ ($0.3850$) and $\text{Wait-}90$ ($0.3841$), and on Qt $\ge 365\text{d}$ AP, frozen $B_0$ ($0.2283$) overtakes $B_{1b}^*$ ($0.2260$).

\subsection{Track 1 Within-Host Counterfactuals vs.\ Track 2 Adapted Baselines}
\label{sec:baselines}

Across 29 total methods---\textbf{26 deployable methods} plus three hindsight references verified over 41 execution branches---we adopt a two-track design separating controlled within-host supervision counterfactuals from simpler linear and adapted stream baselines. This separation tracks three fairness axes explicitly: \textit{update compute budget} via \texttt{Teacher-Only (Matched)}, \textit{training sample pool} via $B_{1b}^*$ versus $B_2\text{-4W}$, and \textit{alarm trigger timing} shared across Track~1 versus native per-event or periodic schedules in Track~2.

\begin{itemize}[leftmargin=1.8em,itemsep=0.10em]
  \item \textbf{Track 1 Within-Host Suite (15 Deployable + 3 Hindsight Methods).} This track holds the host architecture and DriftLens trigger fixed. It includes frozen $B_0$, three Pote-inspired hindsight references ($B_{1a}^*$ full-prefix hindsight, $B_{1b}\text{-1W}^*$, and $4W$ window oracle $B_{1b}^*$), and four causal real-time baselines ($B_2$-1W, $B_2$-4W matching $B_{1b}^*$, dynamically reweighted $B_2$-4W-Bal, and $2{,}000$-commit $B_2$-2000). It further compares delayed-window Wait-$30$, Wait-$60$, and Wait-$90$, disjoint-$U$ Pure-nnPU \citep{kiryo2017positive}, host-adapted ODaSC-Host using a 15-day wait and DenStream confidence weights, single-budget Teacher-Only (Exact), step-matched Teacher-Only (Matched), zero-fit FixedCurve-PU, Survival-PU (+Legacy Gate), and gate-free \textbf{Survival-PU}. In RQ4a, we add two alarm-synchronized post-hoc controls on unchanged $B_2\text{-4W}$ weights: \textbf{B2 + PostHoc Logit} and Proposition-1-inspired plug-in \textbf{B2 + PostHoc Ratio}, defined explicitly in Section~\ref{sec:rq4}.
  \item \textbf{Track 2 Linear and Adapted Native-Protocol Stream Competitors (11 Deployable Methods).} This track spans unsupervised LOC-Only ($1/\max(\text{LA}+\text{LD}, 1)$), single-feature LApredict \citep{zeng2021deep}, 3-feature Churn-LR, and all-metric Full-LR in both frozen and $4W$-adaptive forms. It also includes \textit{adapted native-protocol implementations} of ElasticNet SGD batch-oversampling BORB \citep{cabral2023borb}---evaluated at its native 200-commit cadence and as alarm-triggered BORB-Alarm---together with event-ordered River Online Hoeffding Tree Bagging implementations of ORB \citep{cabral2019class} and specification-level ODaSC \citep{song2022odasc} consuming strictly causal $t_{\text{event}} < t_{\text{commit}}$ events. Because \citet{cabral2023borb} also studied other base learners such as MLPMask, our BORB findings apply specifically to this ElasticNet instance; exact constructors appear in Appendix~\ref{app:repro}.
\end{itemize}

\subsection{Endpoints, Three-Branch Causal Calibration, and Statistical Aggregation}
\label{sec:metrics_protocol}

\textbf{Primary and Secondary Endpoints.} Threshold-free \textbf{Average Precision (AP)}, computed across the concatenated evaluation stream, serves as our single primary endpoint and is contrasted explicitly against per-window mean AP in Section~\ref{sec:rq2}. Secondary endpoints include ROC-AUC, fixed-$0.5$ $F_1$, causally tuned $F_{1,\text{th}}$, Recall@20\%Effort alongside top-20\% Jaccard overlap with LOC-Only, and probabilistic error metrics. For probability quality, we track post-Platt Brier score ($\text{Brier}_{\text{cal}}$, which decomposes into calibration and refinement), raw Brier score, Expected Calibration Error ($\text{ECE}_{\text{cal}}$), and calibration slope, decomposed symmetrically in RQ4a.

\textbf{Prefix-Invariant Three-Branch Causal Calibration.} For each evaluation window $w = [s_w, e_w)$ at causal cutoff $t_w = t_{s_w - 1}$, our causal stream calibrator follows a strict three-branch hierarchy. First, if at least 80 prior evaluation-stream commits are aged at least 90 days ($\mathcal{M}_w^{\text{eval}} = \{s_0 \le j < s_w : t_w - t_j \ge 90\text{d}\}$), it calibrates on the trailing up to $1{,}200$ commits of $\mathcal{M}_w^{\text{eval}}$. Second, otherwise, if at least 80 pre-stream initialization commits satisfy the 90-day age rule ($\mathcal{M}_w^{\text{init}}$), it calibrates on the trailing up to $1{,}200$ commits of $\mathcal{M}_w^{\text{init}}$. Third, if neither pool qualifies, it applies a causal prior-odds logit-shift fallback $\hat{p}_{i,\text{cal}} = \sigma(z_i + \operatorname{logit}(\hat{\pi}_0) - \operatorname{median}(z_{\text{init}}))$ with $\hat{\pi}_0 \in [0.02, 0.65]$ and fixed threshold $\tau_{\text{th}}(w) = 0.50$. On the first two branches, letting $\mathcal{M}_w$ denote the selected slice labeled by $y_{j,\text{obs}}(t_w)$, the calibrator fits:
\begin{equation}
  \begin{aligned}
    \hat{p}_{i,\text{cal}} &= \sigma(a_w z_i + b_w), \qquad \text{Brier}_{\text{cal}} = \frac{1}{N_{\text{eval}}}\sum_{i=s_0}^{N-1}\big(\hat{p}_{i,\text{cal}} - y_i^*\big)^2, \\
    \tau_{\text{th}}(w) &= \operatorname*{arg\,max}_{\tau \in \mathcal{G}_w} F_1\!\Big(\big\{\mathbb{I}(\hat{p}_j \ge \tau),\, y_{j,\text{obs}}(t_w)\big\}_{j \in \mathcal{M}_w}\Big).
  \end{aligned}
  \label{eq:causal_cal}
\end{equation}
Here $(a_w, b_w)$ are 1D $L_2$-regularized Platt parameters \citep{platt1999probabilistic}, and $\mathcal{G}_w \subset [0.05, 0.95]$ combines 41 uniform thresholds with 19 score quantiles. Crucially, $F_{1,\text{th}}$ evaluates causal per-commit decisions $\hat{y}_{i,\text{th}} = \mathbb{I}(\hat{p}_i \ge \tau_{\text{th}}(w_i))$ directly on raw un-Platted probabilities $\hat{p}_i = \sigma(z_i)$, decoupling threshold search from Platt scaling; RQ4a also checks Platt-thresholded $F_1$ symmetrically. Because median fix lag exceeds 90 days on many streams, $\mathcal{M}_w$ retains residual right-censoring, which we quantify directly alongside raw versus 90-day Platt calibration in Section~\ref{sec:rq4}.

All 41 method execution branches passed deterministic pipeline prefix-invariance tests, exhibiting zero historical perturbation error under adversarial future feature, label, and simultaneous-timestamp edits. Prefix invariance certifies leak-free \textit{pipeline execution}, not commit-time provenance of upstream merged Git snapshots. Throughout all empirical tables, \textbf{boldface marks the best deployable value per row or column} at displayed precision, excluding hindsight references $B_{1a}^*$ and $B_{1b}^*$; in Table~\ref{tab:bench_b_holm}, bold marks $p_{\text{Holm}} < 0.05$.

\textbf{Aggregation Hierarchy, Multiplicity Control, and Decision Rules.} For each method $m$, stream score $\bar{s}_{r,m}$ averages across random seeds---10 for MLP and 5 for LightGBM---and donor-cluster score $c_{k,m} = |\mathcal{S}_k|^{-1}\sum_{r \in \mathcal{S}_k} \bar{s}_{r,m}$ averages equally across streams within cluster $k$. Formal $\alpha = 0.05$ inference is scoped to \textbf{$K=25$} natural donor clusters and the Benchmark-A-disjoint \textbf{$K=24$} sensitivity check. We use exact two-sided paired Wilcoxon signed-rank tests with the Pratt tie convention, step-down Holm--Bonferroni adjustment \citep{holm1979simple}, unadjusted $10{,}000$-resample cluster-level percentile bootstrap 95\% confidence intervals for the mean difference, and Cohen's $d_z$.

Comparisons are partitioned into three hypothesis families: \textbf{Family~1} tests $B_{1b}^*$, $B_0$, and Survival-PU against $B_2$-4W; \textbf{Family~2} compares Survival-PU against $B_0$, Wait-90, Pure-nnPU, Full-LR, BORB, ORB, and ODaSC; and \textbf{Family~3} compares Survival-PU against FixedCurve-PU, Teacher-Only (Matched), Teacher-Only (Exact), and Survival-PU (+Legacy Gate). Because the bootstrap CI estimates an unadjusted mean difference while $p_{\text{Holm}}$ tests a multiplicity-adjusted signed-rank location shift, an unadjusted CI excluding zero alongside $p_{\text{Holm}} \ge 0.05$ is statistically expected. Under our codified interpretation rules, non-significant contrasts report non-establishment verbatim without claiming statistical equivalence, and numerical proximity never implies equivalence. Diagnostic $p$-values outside Table~\ref{tab:bench_b_holm}, notably in Table~\ref{tab:window_ap_scale}, are exploratory unadjusted two-sided Pratt Wilcoxon tests. Throughout the manuscript, all reported deltas are computed from unrounded means prior to four-decimal display, so subtracting displayed table entries can differ by $0.0001$.

% ============================================================================
% 6. Empirical Results
% ============================================================================
\section{Empirical Results}
\label{sec:results}

We present our empirical findings in sequential RQ order, progressing from the causal supervision audit (RQ1) to cross-paradigm and confirmatory operating boundaries (RQ2), maturity-estimator sensitivity and internal ablations (RQ3), targeted stress tests and monitoring calibration (RQ4), and runtime plus commit-level error analysis (RQ5).

% ----------------------------------------------------------------------------
% 6.1 RQ1
% ----------------------------------------------------------------------------
\subsection{RQ1: Causal Audit of Right-Censoring and Hindsight Optimism}
\label{sec:rq1}

[[TAB_AUDIT_CENSORING]]

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig1_multimodal_audit_v3.pdf}
  \caption{(a) Window-averaged fraction of unrevealed true defects across scheduled $400$-commit ($1W$) and $1{,}600$-commit ($4W$) stream boundaries (RQ1). (b) Six-stream mean AP across all four Benchmark~A host families (+frozen linear skip on neural hosts) relative to frozen tabular \texttt{Full-LR} ($0.3653$), bridging directly into RQ2.}
  \label{fig:rq1_overview}
\end{figure}

\textbf{Severe Right-Censoring Across Multi-Year Cohorts.} Table~\ref{tab:audit_censoring} and Figure~\ref{fig:rq1_overview}(a) track right-censoring in two complementary ways: mean per-window unrevealed defect rates across scheduled 400-commit $1W$ and 1,600-commit $4W$ boundaries, and actual $1W$ censoring rates at seed-42 fired alarms. Across multi-year ecosystems, median SZZ fix delays span three to nine months, leaving most recent defects unrevealed whenever an update occurs.

Across scheduled $1W$ boundaries, \textbf{94.6\%} of window defects in natural Qt remain unfixed, corresponding to a pooled rate of $95.3\%$ across $981$ window-defects. The same severe lag appears across every multi-year cohort in Table~\ref{tab:audit_censoring}, ranging from $68.8\%$ across Cabral's 14 streams to $87.5\%$ in natural OpenStack, with fired-alarm rates matching closely between $65.8\%$ and $95.7\%$. Even escalating to $4W$ windows leaves $45.1\%$ to $80.5\%$ of defects unrevealed. By contrast, the shorter confirmatory ApacheJIT cohort experiences far less censoring---$26.4\%$ across scheduled $1W$ boundaries across all five repositories and $18.0\%$ on unsplit Standalone-3---highlighting how archive cadence shapes alarm-time visibility.

\textbf{Matched Hindsight Optimism vs.\ Gradual Ranking Retention.} To measure how strongly future-fix leakage inflates apparent accuracy, we compare offline window-oracle retraining $B_{1b}^*$ against strictly matched causal four-window retraining $B_2\text{-4W}$ across $K=25$ natural donor clusters (Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}). Hindsight optimism is statistically significant on both hosts. On MLP, $B_{1b}^*$ reaches $0.5241$ AP versus $0.5004$ for $B_2\text{-4W}$, a significant gain of $\Delta\text{AP} = +0.0237$ across 20/0/5 clusters with $p_{\text{Holm}} = 6.63 \times 10^{-5}$. On LightGBM, the gap doubles to $\Delta\text{AP} = +0.0476$: $B_{1b}^*$ reaches $0.5409$ AP versus $0.4933$ for $B_2\text{-4W}$ across 23/0/2 clusters with $p_{\text{Holm}} = 3.70 \times 10^{-5}$. Restricting causal replay to a single $1W$ window depresses accuracy even further on both hosts (Table~\ref{tab:bench_b_clusters}).

\textbf{Why Censoring Hurts Differently Across Metrics and Drift Regimes.} This hindsight gap does not mean naive causal replay always fails. Consistent with Proposition~\ref{prop:bias}, older commits inside a $4W$ buffer retain enough partial maturity for $B_2\text{-4W}$ to beat frozen $B_0$ on full-stream MLP AP across both gradual natural clusters---$0.5004$ versus $0.4566$, with $p_{\text{Holm}} = 0.0041$---and controlled abrupt shifts, at $0.5578$ versus $0.5064$ post-onset AP. Still, causal $B_2\text{-4W}$ trails matched oracle updates in both regimes.

Instead, censoring strikes hardest along two axes predicted by Proposition~\ref{prop:bias}. First, fixed-$0.5$ thresholded $F_1$ collapses on multimodal Benchmark~A, where $B_2\text{-4W}$ and $B_2\text{-1W}$ fall far behind \texttt{Survival-PU} (Table~\ref{tab:bench_a_main}). Second, single-window $B_2\text{-1W}$ falls below frozen $B_0$ on the controlled abrupt shifts. Having quantified both right-censoring and hindsight optimism, we examine in RQ2 whether maturity-calibrated PU adaptation closes this gap across hosts, ecosystems, and an untouched confirmatory cohort---and where simpler baselines prevail (Figure~\ref{fig:rq1_overview}(b)).

\rqbox{Summary for RQ1}{Across scheduled $1W$ boundaries in multi-year ecosystems, 68.8\%--94.6\% of recent defects remain right-censored, with fired alarms showing nearly identical rates. Leaking future SZZ fixes significantly inflates whole-stream AP across $K=25$ clusters on both MLP and LightGBM. Although four-window causal retraining still outperforms freezing in gradual AP, it trails matched oracle updates and suffers severe fixed-threshold $F_1$ degradation on multimodal hosts.}

% ----------------------------------------------------------------------------
% 6.2 RQ2
% ----------------------------------------------------------------------------
\subsection{RQ2: Multi-Host, Cross-Project, and Confirmatory Operating Boundaries}
\label{sec:rq2}

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig2_trajectories_and_skip_v3.pdf}
  \caption{(a) Prequential 3-window rolling AP trajectory on natural Qt (CodeBERT + frozen linear skip). (b) Effect of adding the causal tabular linear-residual skip connection (no skip vs.\ dev-selected frozen linear skip) on natural Qt and OpenStack.}
  \label{fig:rq2_skip}
\end{figure}

[[TAB_BENCH_A_MAIN]]

[[TAB_BENCH_A_SENS]]

\textbf{Benchmark A: Simple Linear Predictors Lead Deployable Multimodal Hosts.} Table~\ref{tab:bench_a_main} and Figure~\ref{fig:rq1_overview}(b) expose a striking operating boundary across the six Benchmark~A streams. Frozen 13-metric logistic regression, \textbf{Full-LR Frozen}, leads every deployable single-host configuration at $0.3653$ AP, closely followed by single-feature \textbf{LApredict Frozen}. Among the four multimodal and tree hosts, the strongest deployable individual model is frozen \texttt{DeepJIT-CNN} at $B_0 = 0.3531$ AP, while only non-deployable full-prefix hindsight \texttt{DeepJIT-CNN} $B_{1a}^*$ at $0.3768$ exceeds \texttt{Full-LR Frozen}. Even simple \textbf{Churn-LR Frozen} beats every window-adapted 4-host mean in Table~\ref{tab:bench_a_main}, and unsupervised \texttt{LOC-Only} reaches $0.6520$ R@20\%Effort, ahead of \texttt{Survival-PU}'s $0.6156$.

Adding our causal tabular linear-residual skip substantially narrows the neural deficit (Figure~\ref{fig:rq2_skip}(b)), lifting 6-stream frozen AP by roughly 8 to 17 points across \texttt{Att-CNN}, \texttt{CodeBERT}, and \texttt{DeepJIT-CNN}. Yet even with this architectural anchor, fine-tuning high-capacity encoders on small drift windows still trails frozen linear models in AP. Within identical Track~1 architectures, however, base-rate drift and censoring erode fixed-$0.5$ $F_1$ on frozen $B_0$ and causal $B_2\text{-4W}$, whereas \texttt{Survival-PU} recovers the strongest Track-1 deployable $F_1$ and $F_{1,\text{th}}$ at lower mean AP (Table~\ref{tab:bench_a_main}).

Sensitivity checks confirm that these main Benchmark~A conclusions hold under alternative feature sets, SZZ tiers, and archive follow-up cutoffs. Across the common-commit sensitivity variants in Table~\ref{tab:bench_a_sens}, $B_2\text{-4W}$ leads deployable AP across the three Qt variants while frozen $B_0$ leads across OpenStack. Restricting saved seed-42 predictions to commits with at least 180 or 365 days before $T_{\text{end}}$ (Appendix Table~\ref{tab:followup_sensitivity}) preserves \texttt{Full-LR}'s AP lead across every slice and maintains \texttt{Survival-PU}'s Track-1 $F_1$ gain over $B_2\text{-4W}$ and $B_0$, while edging \texttt{Full-LR} on Qt $F_1$. Some pairwise AP orderings shift under tighter follow-up cutoffs: for example, \texttt{Survival-PU} overtakes $B_2\text{-4W}$ and $\text{Wait-}90$ on OpenStack $\ge 365\text{d}$ AP, and frozen $B_0$ edges past $B_{1b}^*$ on Qt $\ge 365\text{d}$ AP.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig3_benchmark_b_clusters_v3.pdf}
  \caption{(a) Mean AP ($\pm 1$ SEM) across $K=25$ independent natural donor clusters in Benchmark~B. (b) Post-shift AP across the four separated controlled concatenated-shift streams on both full post-onset and first 1,200 post-onset commit horizons.}
  \label{fig:rq2_bench_b}
\end{figure}

[[TAB_BENCH_B_CLUSTERS]]

[[TAB_BENCH_B_HOLM]]

[[TAB_WINDOW_AP_SCALE]]

\textbf{Benchmark B: Gradual Natural Clusters and Evaluation Granularity.} Tables~\ref{tab:bench_b_clusters} and~\ref{tab:bench_b_holm} along with Figure~\ref{fig:rq2_bench_b}(a) present our powered cross-project inference across $K=25$ natural donor clusters, alongside the Benchmark-A-disjoint $K=24$ check. On primary whole-stream MLP AP, \textbf{Full-LR Adapt-4W} achieves the highest deployable mean at $0.5108$, followed by $B_2\text{-4W}$ at $0.5004$, $\text{Wait-}90$, and zero-fit \textbf{FixedCurve-PU}.

Within this landscape, \texttt{Survival-PU} scores $0.4866$ AP. It significantly outperforms canonical \texttt{Pure-nnPU} at $0.4461$, gaining $\Delta\text{AP} = +0.0405$ across 18/0/7 clusters with $p_{\text{Holm}} = 0.0020$, as well as Hoeffding \texttt{ORB} (Table~\ref{tab:bench_b_holm}). After Holm correction, it does not differ significantly from frozen $B_0$, \texttt{ODaSC}, ElasticNet \texttt{BORB}, or \texttt{Full-LR Adapt-4W}. Crucially, however, \textbf{\texttt{Survival-PU} significantly trails both $B_2\text{-4W}$ by $\Delta\text{AP} = -0.0138$ with $p_{\text{Holm}} = 0.0056$ and $\text{Wait-}90$ with $p_{\text{Holm}} = 0.0069$ on primary whole-stream AP}.

To test whether this whole-stream deficit reflects weaker within-window discrimination or cross-window score-scale drift (Proposition~\ref{prop:bias}), Table~\ref{tab:window_ap_scale} evaluates all seven core methods across 5 seeds on the same $K=25$ clusters, spanning $531$ operational 400-commit windows with zero single-class windows excluded. All $p$-values in Table~\ref{tab:window_ap_scale} are exploratory and unadjusted. Moving from concatenated whole-stream AP to unweighted per-window AP reverses the sign against $B_2\text{-4W}$ into a small non-significant mean edge: \textbf{\texttt{Survival-PU} reaches $0.5246$ versus $0.5192$ for $B_2\text{-4W}$} ($p=0.0569$), with positive-weighted Win-AP pointing in the same direction. Under per-window AP, \texttt{Survival-PU} remains close to \texttt{FixedCurve-PU}, $\text{Wait-}90$, and \texttt{Full-LR Adapt-4W}.

Applying \textbf{controlled scale removal}---strictly monotone within-window z-score or percentile-rank standardization on the \textit{exact same predictions} before computing concatenated whole-stream AP---yields a matching reversal. Relative to $B_2\text{-4W}$, standardization flips \texttt{Survival-PU}'s whole-stream mean deficit into small non-significant leads under both Z-Norm and Rank-Norm, reaching $0.5027$ versus $0.4989$ under Rank-Norm ($p=0.0877$). Under Rank-Norm, gaps against \texttt{FixedCurve-PU} and $\text{Wait-}90$ shrink near zero, whereas a smaller gap persists under Z-Norm (Table~\ref{tab:window_ap_scale}). Inter-window score scaling therefore accounts for a major part of \texttt{Survival-PU}'s whole-stream deficit against $B_2\text{-4W}$, though within-window gains over $B_2\text{-4W}$ remain modest and do not surpass $\text{Wait-}90$, \texttt{FixedCurve-PU}, or \texttt{Full-LR}. Because standardization removes inter-window location, scale, and quantile differences together, the specific training step generating those shifts is not isolated separately.

Secondary endpoints on $K=25$ exhibit a clear metric split (Table~\ref{tab:bench_b_clusters}). ElasticNet \texttt{BORB} and Hoeffding \texttt{ODaSC} achieve the highest causally tuned $F_{1,\text{th}}$ overall, while \texttt{Survival-PU} leads Track-1 methods. On post-Platt Brier loss, \texttt{FixedCurve-PU} and \texttt{Survival-PU} are numerically close at four-decimal display ($0.1525$), ahead of \texttt{Full-LR Adapt-4W} and $B_2\text{-4W}$.

\textbf{Benchmark B: Codified Component Controls in Family 3.} Testing whether adaptive survival estimation and Stage~3 distillation justify their complexity reveals negative and host-dependent outcomes (Table~\ref{tab:bench_b_holm}). On $K=25$ primary AP, \textbf{zero-fit FixedCurve-PU achieved significantly higher AP than cohort-estimated Survival-PU on both MLP---at $0.4942$ versus $0.4866$, with $\Delta\text{AP} = -0.0077$ across 6/0/19 clusters and $p_{\text{Holm}} = 0.0295$---and LightGBM, at $0.4680$ versus $0.4619$ with $p_{\text{Holm}} = 0.0367$}. The Benchmark-A-disjoint $K=24$ check confirms the same direction. Because our benchmark does not isolate finite-sample survival variance from parametric prior shape, we quote our codified decision rule verbatim: ``\textit{An incremental benefit over FixedCurve\_PU was not established on this cohort.}''

Comparing \texttt{Survival-PU} against step-matched \texttt{Teacher-Only (Matched)} on $K=25$ MLP likewise shows no established gain ($p_{\text{Holm}} = 0.9157$). Quoting our codified rule verbatim: ``\textit{An incremental benefit over Teacher\_Only\_Matched was not established on this cohort.}'' On LightGBM, by contrast, Stage~3 PU distillation significantly improves AP over both \texttt{Teacher-Only (Matched)} by $+3.79$ points ($p_{\text{Holm}} = 9.87 \times 10^{-4}$) and \texttt{Teacher-Only (Exact)}, although causal $B_2\text{-4W}$ still achieves the highest deployable LightGBM AP overall. Legacy three-state gating adds no established benefit over gate-free \texttt{Survival-PU}.

\textbf{Benchmark B: Controlled Concatenated Distribution Shifts Across Four Streams.} Under sharp cross-project or cross-subsystem regime changes, method rankings reverse with horizon length (Table~\ref{tab:bench_b_clusters}, Figure~\ref{fig:rq2_bench_b}(b), and Appendix Table~\ref{tab:shift_4streams}). Over the \textbf{full post-onset horizon}, \texttt{Survival-PU} achieves the highest observed deployable mean AP at \textbf{0.5764}, closely followed by Track-1 \texttt{ODaSC-Host} at $0.5727$, $B_2\text{-4W-Bal}$, and $\text{Wait-}30$, whereas single-window $B_2\text{-1W}$, age-blind \texttt{Pure-nnPU}, and \texttt{ORB} fall below frozen $B_0$.

Over the \textbf{immediate first 1,200 post-onset commits}, however, ElasticNet batch-oversampling \textbf{BORB-Alarm} at $0.5931$ and \textbf{BORB-Native} at $0.5734$ adapt fastest, surpassing \texttt{Survival-PU} at $0.5591$ as well as $\text{Wait-}30$ and \texttt{ODaSC-Host}. Appendix Table~\ref{tab:shift_4streams} also highlights notable stream heterogeneity: \texttt{OS Nova$\to$Neutron} yields substantially lower post-onset AP across all adaptive methods than the three cross-project shifts evaluated in RQ4d.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig4_confirmatory_boundary_v3.pdf}
  \caption{(a) Out-of-sample AP across the five confirmatory ApacheJIT repositories (asterisks mark Hadoop monorepo subprojects whose clean commits reside in \texttt{apache/hadoop}). (b) Exploratory association plotting evaluation horizon $N_{\text{eval}}$ against the matched oracle-minus-causal hindsight gap $\Delta\text{AP}(B_{1b}^* - B_2\text{-4W})$ on MLP across the 27 natural Benchmark~B streams ($22/27 > 0$) and five confirmatory repositories ($0/5 > 0$, separating Standalone-3 from Hadoop Monorepo-2).}
  \label{fig:rq2_confirmatory}
\end{figure}

[[TAB_CONFIRMATORY]]

\textbf{Phase C: Confirmatory Non-Replication on MLP — Standalone-3 vs.\ Frozen Umbrella-4.} Table~\ref{tab:confirmatory} and Figure~\ref{fig:rq2_confirmatory} report out-of-sample results on the five confirmatory ApacheJIT repositories. Because upstream \texttt{apachejit\_total.csv} assigns $11{,}964$ clean Hadoop monorepo commits to \texttt{apache/hadoop}, evaluation prevalence is inflated to $89.2\%$ on \texttt{HDFS*} and $78.8\%$ on \texttt{MapReduce*} (Table~\ref{tab:confirmatory_funnel}). The \textbf{three standalone repositories}---\texttt{Spark}, \texttt{Zeppelin}, and \texttt{ZooKeeper}, free of monorepo splitting---therefore provide the cleaner external check. On Standalone-3, \textbf{MLP hindsight optimism did not replicate}: $B_{1b}^*$ scores $0.5951$ AP, still trailing causal $B_2\text{-4W}$ at $0.5985$ and frozen $B_0$ at $0.6565$. Meanwhile, Hoeffding \textbf{ODaSC} leads clearly at $0.7857$, followed by ElasticNet \textbf{BORB} and Hoeffding \textbf{ORB}, all well above every window-retrained MLP variant. On Standalone-3 LightGBM, frozen $B_0$ likewise leads both $B_{1b}^*$ and $B_2\text{-4W}$.

The frozen pre-unlock \textbf{umbrella-4 MLP AP} record shows the identical direction. Across all four umbrella ecosystems, window-oracle $B_{1b}^*$ reaches $0.6801$ AP, trailing causal $B_2\text{-4W}$ at $0.6858$ by $\Delta\text{AP} = -0.0057$ across 0/0/4 umbrella wins with exact two-sided $p = 0.125$. It also trails frozen $B_0$ at $0.7244$ by $\Delta\text{AP} = -0.0443$ across 1/0/3 wins with exact two-sided $p = 0.375$. Again, \textbf{ODaSC} leads at $0.8289$, ahead of \textbf{BORB} and \textbf{ORB} (Table~\ref{tab:confirmatory}). On LightGBM umbrella-4, $B_{1b}^*$ stays slightly and non-significantly above $B_2\text{-4W}$. Because short horizon length, higher prevalence, low $18.0\%$ Standalone-3 censoring, alarm count, and project identity all covary on this cohort, the pattern favoring \texttt{ODaSC} remains an exploratory hypothesis rather than a validated selection rule.

\rqbox{Summary for RQ2}{Frozen \texttt{Full-LR} leads every deployable host on Benchmark~A, while \texttt{Survival-PU} recovers Track-1 $F_1$. On gradual $K=25$ whole-stream MLP AP, \texttt{Full-LR Adapt-4W} leads in mean, while $B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU} significantly outperform \texttt{Survival-PU}. Within-window standardization on identical 5-seed predictions reverses the mean deficit against $B_2\text{-4W}$ non-significantly, and per-window AP narrows gaps across causal methods. On controlled shifts, \texttt{Survival-PU} leads full post-onset AP whereas \texttt{BORB-Alarm} leads early recovery. On the confirmatory cohort, MLP hindsight optimism fails to replicate and \texttt{ODaSC} leads both Standalone-3 and Umbrella-4.}

% ----------------------------------------------------------------------------
% 6.3 RQ3
% ----------------------------------------------------------------------------
\subsection{RQ3: Maturity-Estimator Sensitivity, Mechanism Audit, and Component Ablations}
\label{sec:rq3}

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig5_ablations_and_controls_v3.pdf}
  \caption{(a) Ten-stream mean AP across three causal maturity estimators, zero-fit \texttt{FixedCurve-PU}, and single-factor internal ablations. (b) Paired AP and Recall@20\%Effort across gate-free Survival-PU, step-matched teacher controls, and legacy three-state gating.}
  \label{fig:rq3_abl}
\end{figure}

[[TAB_RQ3_ABLATIONS]]

Table~\ref{tab:rq3_ablations} and Figure~\ref{fig:rq3_abl} dissect maturity-estimator sensitivity and internal component contributions across the ten-stream representative subset, comprising seven natural streams and three controlled shifts. Because seven natural streams have exact $p_{\min} = 0.015625 > 0.0125$, these contrasts are reported descriptively.

\textbf{Low Sensitivity Across Stage~1 Maturity Estimators.} Substituting alternative Stage~1 estimators changes aggregate performance only modestly (Table~\ref{tab:rq3_ablations}). Across default cohort incidence, $10^{-4}$-day Kaplan--Meier cure, global Weibull mixture-cure MLE, and zero-fit \texttt{FixedCurve-PU}, all four variants span just $1.09$ AP points and $0.45$ $F_{1,\text{th}}$ points, with zero survival fallbacks and zero unreachable bisection solves. On this ten-stream subset, causal $B_2\text{-4W}$ leads descriptive AP while trailing \texttt{Survival-PU} on both $F_{1,\text{th}}$ and $\text{Brier}_{\text{cal}}$.

\textbf{Unrevealed-Commit Mechanism Audit vs.\ FixedCurve-PU and Positive Writeback.} Because Eq.~\eqref{eq:bayes_pu} fixes $q_i = 1$ on already-revealed positives, all-pool error conflates label writeback with genuine maturity estimation on unrevealed commits. Appendix Table~\ref{tab:mechanism_audit} therefore audits all 62 seed-42 DriftLens alarms across the ten representative streams, covering $74{,}400$ pool evaluations and $59{,}600$ unique commits, comparing \texttt{Survival-PU} directly against \texttt{FixedCurve-PU}. Reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is computed over all archive-labeled positives $\mathcal{P}$ per stream, treating unlinked defects as unrevealed at finite ages; dispersion terms report sample standard deviations.

\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Stage~1 Maturity vs.\ Stage~2 Prevalence: Non-Monotonic Error Propagation.} Adaptive cohort incidence estimates empirical maturity $c^*(a)$ more accurately than the fixed 90-day curve, lowering alarm-level MAE from $0.1204$ for \texttt{FixedCurve-PU} to $0.0938$ for \texttt{Survival-PU} and cutting stream-level MAE from $0.1310$ to $0.0873$. Crucially, however, lower Stage-1 error does \textit{not} translate monotonically into lower downstream error: on Stage-2 target prevalence, \texttt{FixedCurve-PU} achieves slightly lower alarm-level MAE at $0.0565$ than \texttt{Survival-PU} at $0.0620$, though both improve over unshrunk $\hat{\pi}_{\text{IP}}$.
  \item \textit{Positive Writeback vs.\ Strictly Unrevealed Soft Targets.} Across all $74{,}400$ pool evaluations, writing back observed positives to the de-weighted teacher alone reduces Brier error from $0.1525$ to $0.1134$, confirming that writeback drives most of the all-pool reduction. Restricting strictly to \textbf{unrevealed commits} with $y_{\text{obs}} = 0$ ($64{,}745$ evaluations and zero writeback), soft targets still improve over the de-weighted teacher in every age stratum, lowering Brier from $0.1303$ to $0.1236$ for \texttt{Survival-PU} with near-zero bias. Mirroring Stage~2, \texttt{FixedCurve-PU} achieves slightly lower overall unrevealed Brier at $0.1200$ with negative bias $-0.0358$, leading on young and mid-age commits while \texttt{Survival-PU} leads on aged $\ge 90\text{d}$ commits (Table~\ref{tab:mechanism_audit}). This cross-stage trade-off is consistent with \texttt{FixedCurve-PU}'s competitiveness on gradual streams, though Brier and bias do not directly determine AP and the exact source of \texttt{FixedCurve-PU}'s $K=25$ AP edge remains unisolated.
\end{itemize}

\textbf{Component Ablations and Simplification Candidates.} Table~\ref{tab:rq3_ablations} shows that freezing the teacher to the initial pre-stream checkpoint produces the steepest AP drop---over three points---while also triggering $4.2$ unreachable bisection alarms per stream, confounding teacher staleness with solver clamping. Removing monotone bisection or anchor shrinkage degrades both AP and $\text{Brier}_{\text{cal}}$. Conversely, omitting Stage-2 covariate density ratios slightly improves descriptive AP, $F_{1,\text{th}}$, and effort recall while trimming runtime, identifying density-ratio weighting as an unverified simplification candidate. On this subset, \texttt{Survival-PU} also beats step-matched $2\times$ \texttt{Teacher-Only}, while legacy gating alters AP negligibly.

\rqbox{Summary for RQ3}{All four Stage-1 maturity estimators span just $1.09$ AP points with zero solver fallbacks. Across 62 fired alarms, adaptive cohort incidence estimates Stage-1 maturity more accurately than \texttt{FixedCurve-PU}, yet \texttt{FixedCurve-PU} attains slightly lower Stage-2 prior MAE and strictly unrevealed soft-target Brier, demonstrating non-monotonic error propagation across stages. Teacher adaptation, bisection, and shrinkage carry the primary weight, whereas covariate density-ratio weighting emerges as a simplification candidate.}

% ----------------------------------------------------------------------------
% 6.4 RQ4
% ----------------------------------------------------------------------------
\subsection{RQ4: Stress Tests Across Latency Extremes, Monitoring, Noise, and Prompting}
\label{sec:rq4}

We subject our findings to five targeted stress tests: RQ4a evaluates future-only latency scaling with post-hoc $B_2$ controls and raw versus Platt calibration; RQ4b runs an unconfounded $2 \times 2$ DriftLens monitoring factorial; RQ4c injects independent semi-synthetic PR-SZZ label noise; RQ4d sweeps window sizes and key hyperparameters; and RQ4e conducts a paired Code-LLM prompting case study.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig6_latency_and_drift_factorial_v3.pdf}
  \caption{(a) Post-$t_0$ verification-latency scaling ($\kappa \in [0.0, 3.0]$ with frozen pre-$t_0$ supervision, keeping $B_0$ invariant). (b) $2 \times 2$ DriftLens bootstrap/refresh factorial ($p_{95}$, cooldown $1$) plus Pote-inspired reference ($p_{98}$, cooldown $0$) and RBF-MMD.}
  \label{fig:rq4_lat_mon}
\end{figure}

[[TAB_RQ4A_KAPPA]]

[[TAB_RQ4A_FIXEDCURVE]]

[[TAB_RQ4A_POSTHOC_B2]]

[[TAB_RQ4B_MONITORS]]

\textbf{RQ4a: Future-Only Verification-Latency Scaling and Stage-1 Misspecification.} Tables~\ref{tab:rq4a_kappa}--\ref{tab:rq4a_posthoc_b2} and Figure~\ref{fig:rq4_lat_mon}(a) scale finite post-$t_0$ fix delays by $\kappa \in [0.0, 3.0]$ while preserving pre-$t_0$ supervision and unlinked defects. Frozen $B_0$ therefore stays invariant across every row, and post-$t_0$ oracle visibility satisfies exact $B_2\text{-4W} = B_{1b}^* = 0.5980$ AP parity.

Table~\ref{tab:rq4a_kappa} exposes a sharp ranking-versus-classification split as latency grows. Across the ten-stream AP trajectory, $B_2\text{-4W}$ stays slightly above \texttt{Survival-PU} from $\kappa = 0\times$ through $3\times$. On classification quality at $\kappa = 3.0\times$, however, $B_2\text{-4W}$'s fixed-$0.5$ $F_1$ drops by $15.78$ points from $0.5403$ to $0.3825$, and its causally tuned $F_{1,\text{th}}$ falls sharply as well. By contrast, \textbf{Survival-PU sustains $F_1 = 0.5259$ at $\kappa = 3.0\times$}---over 14 points above $B_2\text{-4W}$---and holds \textbf{causally tuned $F_{1,\text{th}} = 0.5022$ versus $0.4465$ for $B_2\text{-4W}$}, also outperforming $\text{Wait-}90$. Meanwhile, age-blind \texttt{Pure-nnPU} collapses in effort recall.

Table~\ref{tab:rq4a_fixedcurve} resolves when adaptive Stage-1 cohort incidence adds value over zero-fit \texttt{FixedCurve-PU}'s hardcoded 90-day half-life:
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Bidirectional Misspecification Protection.} When bug fixes reveal immediately at $\kappa = 0.00\times$, \texttt{FixedCurve-PU}'s fixed 90-day prior treats clean young commits as censored. It consequently trails adaptive \texttt{Survival-PU} across all four endpoints in Table~\ref{tab:rq4a_fixedcurve}, including a $2.16$-point AP gap of $0.5544$ versus $0.5760$ and a four-point drop in $F_1$, while $B_2\text{-4W}$ leads overall. At the opposite extreme of $\kappa = 2.00\times$ and $3.00\times$, the fixed 90-day curve overestimates maturity, allowing adaptive \texttt{Survival-PU} to outperform \texttt{FixedCurve-PU} on fixed-$0.5$ $F_1$, threshold-tuned $F_{1,\text{th}}$, and post-Platt Brier loss.
  \item \textit{Where Zero-Fit \texttt{FixedCurve-PU} Leads Pure Ranking.} Across $\kappa \in [0.25\times, 3.0\times]$, \texttt{FixedCurve-PU} consistently achieves higher whole-stream AP than \texttt{Survival-PU}. In fact, it achieves the highest deployable AP under heavy latency, reaching $0.5547$ at $\kappa = 3.0\times$ and surpassing both $B_2\text{-4W}$ at $0.5466$ and \texttt{Survival-PU} at $0.5438$.
\end{itemize}

\textbf{Symmetrical Raw vs.\ 90-Day Platt Calibration and Post-Hoc $B_2$ Controls.} Auditing the 90-day calibration pool $\mathcal{M}_w$ confirms heavy residual censoring whenever median fix lag exceeds 90 days---reaching $55.3\%$ in Benchmark~A, $43.8\%$ in Benchmark~B, and $65.9\%$ at $\kappa = 3\times$. To separate training-time PU updates from post-hoc probability adjustment, Table~\ref{tab:rq4a_posthoc_b2} evaluates all five methods symmetrically under both \textbf{Raw} and \textbf{Shared 90-Day Platt} probabilities using the identical 90-day causal threshold search. Both post-hoc controls leave $B_2\text{-4W}$'s weights untouched and update strictly at fired DriftLens alarms using causal history:

\begin{enumerate}[leftmargin=1.6em,itemsep=0.04em]
  \item \textit{\texttt{B2 + PostHoc Logit}} initializes $\delta_0$ on $z_{\text{init}}$ to match $\hat{\pi}_{\text{init}}$, then at each fired alarm solves via 28-step bisection on the trailing window's $B_2$ logits for scalar $\delta_k \in [-8, 8]$ satisfying $\text{mean}(\sigma(z_{B_2,\text{past}} + \delta_k)) = \hat{\pi}_{\text{target}}$.
  \item \textit{\texttt{B2 + PostHoc Ratio}} implements a Proposition~\ref{prop:bias}-inspired plug-in ratio control. It removes $B_2\text{-4W}$'s training class weight via $r_{\text{dewt}}(x) = \sigma(z_{B_2}(x) - \log\omega_{\text{base}})$, divides by the alarm-updated trailing-$4W$ mean maturity $\bar{c}_{4W} = \max(\text{mean}_{j \in W_{4W}} \hat{c}(a_j), 0.10)$, and clips the ratio to $[10^{-4}, 1-10^{-4}]$.
\end{enumerate}

Table~\ref{tab:rq4a_posthoc_b2} yields three crisp mechanistic findings:
\begin{enumerate}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Neither post-hoc $B_2$ control matches PU retraining's heavy-latency classification recovery.} At $\kappa = 0\times$, \texttt{PostHoc Logit} loses whole-stream AP via alarm-to-alarm scale shifts while preserving \texttt{WinAP}, whereas plug-in \texttt{PostHoc Ratio} incurs a smaller whole-stream drop and attains the lowest raw Brier error. Its modest \texttt{WinAP} drop at $\kappa=0\times$ stems entirely from upper-bound clipping at $1-10^{-4}$, clipping $7.9\%$ of commits across $71.8\%$ of windows at $\kappa=0\times$ and elevating within-window ties; removing that cap restores $B_2\text{-4W}$'s exact \texttt{WinAP} across all three latency levels while leaving fixed-$0.5$ $F_1$ unchanged. At $\kappa = 3\times$, both \texttt{PostHoc Logit} ($F_1 = 0.3593$) and \texttt{PostHoc Ratio} ($F_1 = 0.3216$) still trail \texttt{Survival-PU} ($F_1 = 0.5259$) and \texttt{FixedCurve-PU} substantially, while within-window AP stays narrow across all five methods. Because \texttt{PostHoc Ratio} strips training class weights to emit unweighted probabilities whereas \texttt{Survival-PU} and $B_2\text{-4W}$ emit class-weighted scores, fixed-$0.5$ $F_1$ is asymmetric across scales. On the scale-fair threshold-tuned metric $F_{1,\text{th}}$, however, \texttt{Survival-PU} still leads \texttt{PostHoc Ratio} by $4.79$ points and $B_2\text{-4W}$ by $5.57$ points (Table~\ref{tab:rq4a_posthoc_b2}).
  \item \textit{Why \texttt{Survival-PU}'s raw probabilities appear less calibrated than $B_2\text{-4W}$ and \texttt{FixedCurve-PU}.} Across all three $\kappa$ levels, \texttt{Survival-PU} incurs higher raw Brier and ECE than $B_2\text{-4W}$, while \texttt{FixedCurve-PU} achieves the best raw calibration at $\kappa=3\times$ (Table~\ref{tab:rq4a_posthoc_b2}). This directly closes Proposition~\ref{prop:bias}'s Remark: Stage~3 applies class weight $\omega_{+,\text{teach}} > 1$ \textit{on top of} already prevalence-calibrated soft targets $q_i(\delta^*)$, pushing raw student probabilities upward. Conversely, $B_2\text{-4W}$'s censoring compression and \texttt{FixedCurve-PU}'s negative target bias ($-0.0358$ in Table~\ref{tab:mechanism_audit}) pull raw outputs downward, partially canceling $\omega_+ > 1$'s upward shift.
  \item \textit{Cautionary finding on 90-day Platt calibration under severe lag.} At $\kappa = 3\times$, raw probabilities outperform 90-day Platt calibration on ECE for all five methods and on Brier for four of five, leaving \texttt{PostHoc Logit} nearly unchanged. This serves primarily as a warning against fitting standard 90-day Platt scaling to a $65.9\%$-censored calibration pool. Secondarily, when that shared calibrator is applied, \texttt{Survival-PU} retains the lowest final post-Platt Brier and ECE, incurring smaller Raw-to-Platt increases than $B_2\text{-4W}$ and \texttt{FixedCurve-PU} though not uniformly smaller across all metrics relative to the two post-hoc controls. Crucially, \texttt{Survival-PU}'s $F_{1,\text{th}}$ lead holds whether causal thresholding runs on raw scores or on 90-day Platted scores ($0.4898$ vs.\ $0.4814$ for \texttt{FixedCurve-PU} and $0.4385$ for $B_2\text{-4W}$).
\end{enumerate}

\textbf{RQ4b: Unconfounded $2 \times 2$ DriftLens Calibration Factorial.} Table~\ref{tab:rq4b_monitors} and Figure~\ref{fig:rq4_lat_mon}(b) hold percentile and cooldown fixed at $p_{95}$ and $\text{cooldown}=1$, measuring validation-pool resampled-null alarm rates via 20 block-bootstrap windows of length 50. Replacing \texttt{IID} with \texttt{Circular Block} cuts this \textbf{validation-pool resampled-null alarm rate by $4.8\times$ under matched refresh rules}, dropping from $20.2\%$ to $4.2\%$ under static references and from $16.4\%$ to $3.4\%$ under rolling references.

Across the four controlled-shift streams evaluated over five seeds---20 runs in total---enabling \texttt{Rolling Ref+Th} eliminates missed transitions entirely, reducing them from $5.0\%\text{--}10.0\%$ under static monitoring ($1/20$ for \texttt{IID + Static} and $2/20$ for \texttt{Block + Static}) to $0.0\%$. Conditional detection delays average $310$ commits for \texttt{IID + Rolling Ref+Th} and $350$ commits for \texttt{Block + Rolling Ref+Th}, versus $1{,}050$ commits for periodic $3W$ retraining. Crucially, \textbf{fewer alarms do not automatically improve AP}: default \texttt{Block + Rolling} reaches $0.5543$ SPU AP at a $32.1\%$ 10-stream alarm rate, slightly trailing \texttt{IID + Rolling} at $0.5628$ and $43.5\%$ alarm rate as well as periodic $3W$. Even so, block-calibrated DriftLens remains operationally preferable to fixed periodic schedules because it detects abrupt shifts \textbf{$3\times$ faster} while keeping stationary validation-pool alarm rates an order of magnitude lower (Table~\ref{tab:rq4b_monitors}).

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig7_szz_window_sensitivity_v3.pdf}
  \caption{(a) $2 \times 2$ frozen-prediction semi-synthetic SZZ perturbation across five PR-SZZ streams ($\alpha_{\text{FP}}=8.42\%, \beta_{\text{FN}}=47.15\%$). (b) Matched-horizon fixed 1,200-commit post-onset AP across $W \in \{200, 400, 600, 800\}$ on the three representative controlled concatenated-shift streams (Nova$\to$Zulip, Ignite$\to$Cassandra, Airflow$\to$Pulsar).}
  \label{fig:rq4_szz_win}
\end{figure}

[[TAB_RQ4D_SZZ]]

\textbf{RQ4c: Frozen-Prediction $2 \times 2$ Semi-Synthetic PR-SZZ Noise Perturbation.} To probe combined latency and permanent mislabeling, Table~\ref{tab:rq4d_szz} and Figure~\ref{fig:rq4_szz_win}(a) inject independent conditional flip noise into five PR-SZZ streams using project-averaged B-SZZ error rates from \citet{bludau2026quantifying}: clean commits flip to buggy at rate $\alpha_{\text{FP}} = 8.42\%$ with empirical delays, while buggy commits permanently lose their fix at rate $\beta_{\text{FN}} = 47.15\%$. Under this independent perturbation model, training on noisy labels and evaluating on unperturbed PR-SZZ labels produces a sharp family-level split. Batch window-replay and maturity-weighted methods cluster tightly within $0.0034$ AP, led descriptively by \textbf{Survival-PU} alongside $\text{Wait-}90$ and $B_2\text{-4W}$ (Table~\ref{tab:rq4d_szz}). Online stream learners degrade substantially more, with ElasticNet \texttt{BORB} falling from $0.4224$ to $0.3187$ and Hoeffding \texttt{ORB} dropping even further. Because independent synthetic flips omit feature- or age-dependent SZZ error correlation, generalizing this robustness requires validation on manually verified SZZ corpora.

\textbf{RQ4d: Matched-Horizon Window Sweeps and Hyperparameter Sensitivity.} Figure~\ref{fig:rq4_szz_win}(b) sweeps window size $W \in \{200, 400, 600, 800\}$ on the fixed 1,200-commit post-onset horizon across the three representative controlled shift streams in our ten-stream subset: Nova$\to$Zulip, Ignite$\to$Cassandra, and Airflow$\to$Pulsar. Here \texttt{Survival-PU} leads deployable methods at narrow and medium windows of $W = 200$, $400$, and $600$. \textbf{At $W=800$, however, $\text{Wait-}90$ at $0.6078$ and $B_2\text{-4W}$ at $0.5989$ exceed \texttt{Survival-PU} at $0.5974$} once wide windows span older matured history. Across full-stream ten-stream AP for all $W$, $B_2\text{-4W}$ stays ahead of \texttt{Survival-PU}. Sweeping anchor shrinkage $m_0 \in \{0, 25, 50, 100\}$ and tail cap $\gamma_{\text{tail}} \in [0.65, 0.94]$ confirms low sensitivity across both AP and calibration.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig8_llm_and_runtime_forensics_v3.pdf}
  \caption{(a) Paired prompting case study (\texttt{deepseek-flash} on 320-commit Qt and OpenStack slices) vs.\ domain-trained JIT predictors (initial checkpoints for CodeBERT/LightGBM, 0 alarms on slice). (b) Standalone wall-clock runtime on pre-extracted features across 27 natural Benchmark~B streams.}
  \label{fig:rq4f_rt}
\end{figure}

\textbf{RQ4e: Paired Code-LLM Prompting Case Study.} To test whether parameter-free in-context learning bypasses weight-retraining censoring, we conducted a paired prompting case study using \texttt{deepseek-flash} on 320-commit slices of Qt and OpenStack. Across prompt regimes and three exemplar seeds, this covers $640$ distinct commits and $6{,}400$ logical evaluations (Figure~\ref{fig:rq4f_rt}(a); full protocol and results in Appendix Table~\ref{tab:rq4f_llm}). Comparing strictly paired 6-shot prompts that share identical exemplars and order---differing solely in observable $y_{\text{obs}}(t)$ versus eventual $y^*$---shows that exemplar censoring depresses OpenStack $F_1$ by $10.70$ points while barely shifting Qt $F_1$. Conversely, \textbf{OpenStack AP is slightly higher under censored 6-shot at $0.4277 \pm 0.0152$ SD across three exemplar seeds than under paired oracle 6-shot at $0.4177 \pm 0.0299$ SD}, echoing Proposition~\ref{prop:bias}'s probability-shift behavior. Because this case study covers one lightweight model on 640 historical public commits where zero DriftLens alarms fired, it shows only that exemplar label visibility alters prompting classification on this model rather than establishing general Code-LLM claims; domain-trained JIT predictors outperform every prompting arm in AP (Table~\ref{tab:rq4f_llm}).

\rqbox{Summary for RQ4}{At $\kappa=3.0\times$, \texttt{FixedCurve-PU} achieves top deployable AP and best raw calibration, whereas \texttt{Survival-PU} outperforms $B_2\text{-4W}$, both post-hoc $B_2$ controls, and \texttt{FixedCurve-PU} on scale-fair $F_{1,\text{th}}$ while also beating \texttt{FixedCurve-PU} across all metrics at $\kappa=0\times$. Fitting 90-day Platt scaling to a $65.9\%$-censored pool degrades calibration relative to raw scores across methods. Circular block bootstrap cuts validation-pool null alarms $4.8\times$ and detects shifts $3\times$ faster than periodic $3W$ schedules. Batch window and PU learners resist independent SZZ noise better than \texttt{BORB} and \texttt{ORB}, and exemplar censoring lowers \texttt{deepseek-flash} OpenStack $F_1$ substantially.}

% ----------------------------------------------------------------------------
% 6.5 RQ5
% ----------------------------------------------------------------------------
\subsection{RQ5: Runtime Telemetry and Paired Equal-Cap Budget Error Analysis}
\label{sec:rq5}

[[TAB_RQ5_RUNTIME]]

[[TAB_RQ5_PAIRED_BUDGET]]

\textbf{Cached Execution vs.\ Feature-Encoding Runtime Telemetry.} Table~\ref{tab:rq5_runtime} and Figure~\ref{fig:rq4f_rt}(b) report phase-decomposed wall-clock runtimes on pre-extracted tabular features across 27 natural Benchmark~B streams, averaging $5.99$ alarms per stream and charging each alarm-triggered method its full standalone anchor and monitoring chain. Standalone \texttt{Survival-PU} completes in $14.43\text{ s}$ per stream, or roughly $1.66\text{ s}$ per alarm, matching step-matched \texttt{Teacher-Only}. Online Hoeffding \texttt{ORB}, ElasticNet \texttt{BORB}, and frozen $B_0$ run fastest at roughly $2$ to $4\text{ s}$ per stream, whereas \texttt{ODaSC} requires nearly $70\text{ s}$ due to per-event tree updates and DenStream micro-cluster maintenance. On multimodal Qt, cached \texttt{Survival-PU} runs in $18.19\text{ s}$ on CPU LightGBM across $20.2$ alarms and in $7.26\text{ s}$ on RTX~4080 GPU CodeBERT across $1.2$ alarms. \textbf{Adding offline GPU CodeBERT batch feature encoding---$37.32\text{ s}$ on Qt and $19.32\text{ s}$ on OpenStack---brings total batch processing on Qt to $44.58\text{ s}$}, distinct from unbatched CI webhook latency.

\textbf{Paired Whole-Stream vs.\ Per-Window Equal-Cap Budget Net Gain.} Directional rescue anecdotes cannot establish net review yield, and retrospective whole-stream budgeting answers a different operational question than sprint-level review capacity. Table~\ref{tab:rq5_paired_budget} therefore compares \texttt{Survival-PU} minus $B_2\text{-4W}$ on seed-42 commit predictions across three inspection regimes: unmatched fixed-$0.50$ thresholding, equal-cap retrospective whole-stream budgets at Top-10\% commits and 20\% LOC, and equal-cap operational per-window 400-commit budgets, corroborated by 5-seed windowed 20\%-LOC recall $\Delta\text{WinR}_{20}$. Under 20\%-LOC budgets, commits are ranked by risk density $\hat{p}/\text{LOC}$ with ties favoring smaller LOC and selected greedily without splitting commits up to the 20\% cap. This yields exact Top-10\% commit counts and closely matched realized LOC consumption across every host ($20.0\%$ vs.\ $20.0\%$ on seven of eight rows and $20.0\%$ vs.\ $19.8\%$ on OpenStack LightGBM).

\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Fixed Threshold at $0.50$ with Unmatched Volume.} Preventing posterior collapse rescues hundreds of defects on Qt and dozens on OpenStack, yet only by flagging $+1{,}838$ to $+3{,}211$ extra clean commits on Qt and $+130$ to $+357$ on OpenStack (Table~\ref{tab:rq5_paired_budget}).
  \item \textit{Whole-Stream Retrospective vs.\ Per-Window Operational Budgets.} Echoing Section~\ref{sec:rq2}'s granularity finding, retrospective whole-stream selection penalizes \texttt{Survival-PU} steeply on five of seven alarmed hosts, notably on Qt DeepJIT-CNN and OpenStack LightGBM. Switching to \textbf{operational per-window 400-commit budgets} eliminates or reverses nearly every steep whole-stream loss in Table~\ref{tab:rq5_paired_budget}. Qt LightGBM stays strongly positive across both commit and LOC caps, Qt and OpenStack DeepJIT-CNN turn positive on windowed 20\%-LOC recall across five seeds, and Qt Att-CNN and OpenStack LightGBM become near-even. By contrast, Qt CodeBERT remains negative across both whole-stream and per-window budgets. Qualitative Tier-1 SZZ rescue illustrations and FN/FP churn distributions are archived in Appendix Table~\ref{tab:rq5_forensics}.
\end{itemize}

\rqbox{Summary for RQ5}{On cached features, \texttt{ORB} and \texttt{BORB} run fastest under four seconds per stream, \texttt{Survival-PU} takes $14.43\text{ s}$ ($44.58\text{ s}$ including Qt CodeBERT batch encoding), and \texttt{ODaSC} takes nearly $70\text{ s}$. Under equal budget caps, retrospective whole-stream selection penalizes \texttt{Survival-PU} on five of seven alarmed hosts, whereas operational per-window 400-commit budgeting reverses most steep losses on LightGBM and DeepJIT-CNN while remaining host-dependent and negative on Qt CodeBERT.}

% ============================================================================
% 7. Discussion, Observed Operating Boundaries, and Limitations
% ============================================================================
\section{Discussion, Observed Operating Boundaries, and Limitations}
\label{sec:discussion}

We synthesize our findings around the three framing questions of this study, avoiding verbatim repetition of Section~\ref{sec:results}.

\subsection{What Label-Visibility Timing Changes}
\label{sec:disc_timing}

Our audit separates two questions often conflated in drift-triggered JIT-SDP \citep{pote2026adaptive}: whether small-window retraining outperforms freezing ($B_{1b}^*$ vs.\ $B_0$), and whether future-fix leakage inflates retrained weights ($B_{1b}^*$ vs.\ $B_2\text{-4W}$). On long multi-year tabular streams ($K=25$), leaking future SZZ fixes inflates whole-stream AP significantly on both MLP and LightGBM. In two other regimes, however, even window-oracle $B_{1b}^*$ fails to beat frozen $B_0$ or causal $B_2\text{-4W}$ on MLP: low-prevalence multimodal Benchmark~A, where frozen \texttt{Full-LR} leads in line with \citet{zeng2021deep}, and the short confirmatory ApacheJIT cohort, where horizon length, higher Standalone-3 prevalence ($44.6\%\text{--}48.5\%$), low $1W$ censoring, and project identity shift together. Chronological evaluations should therefore disclose end-of-archive follow-up horizons (Appendix Tables~\ref{tab:followup_audit}--\ref{tab:followup_sensitivity}) alongside upstream repository-key splits such as \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} partition (Table~\ref{tab:confirmatory_funnel}).

\subsection{What Complex Maturity Correction Adds Over Simpler Causal Baselines}
\label{sec:disc_increment}

Once strict causal visibility is enforced, our multi-track controls pinpoint where three-stage \texttt{Survival-PU} delivers genuine value and where simpler baselines suffice:
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Whole-Stream vs.\ Within-Window Granularity.} On gradual $K=25$ streams, \texttt{Full-LR Adapt-4W}, $B_2\text{-4W}$, $\text{Wait-}90$, and \texttt{FixedCurve-PU} outperform \texttt{Survival-PU} on whole-stream AP. Yet our 5-seed comparability diagnostic in Table~\ref{tab:window_ap_scale} shows that within-window score standardization reverses the mean whole-stream deficit against $B_2\text{-4W}$, while keeping per-window AP close across $\text{Wait-}90$, \texttt{FixedCurve-PU}, and \texttt{Full-LR}.
  \item \textit{Non-Monotonic Cross-Stage Error Propagation.} On strictly unrevealed commits in Table~\ref{tab:mechanism_audit}, adaptive cohort incidence estimates Stage-1 maturity $\hat{c}(a)$ more accurately than \texttt{FixedCurve-PU} and lowers soft-target Brier over de-weighted teachers. Even so, \texttt{FixedCurve-PU} attains slightly lower Stage-2 prior MAE and unrevealed Brier, showing that better Stage-1 estimation does not automatically reduce downstream error across coupled stages.
  \item \textit{Heavy Latency, Scale-Fair Controls, and Calibration Caution.} Under $\kappa=3\times$ latency (Table~\ref{tab:rq4a_posthoc_b2}), neither post-hoc $B_2$ control matches \texttt{Survival-PU}'s threshold-tuned $F_{1,\text{th}}$ recovery, and adaptive Stage~1 simultaneously protects against \texttt{FixedCurve-PU}'s $\kappa=0\times$ misspecification. On raw probability error, Stage~3's positive class weight $\omega_{+,\text{teach}} > 1$ explains why \texttt{Survival-PU} trails $B_2\text{-4W}$ and \texttt{FixedCurve-PU}, whose downward censoring or target bias partially cancels $\omega_+ > 1$ in line with Proposition~\ref{prop:bias}. More broadly, because raw scores beat 90-day Platt calibration under $65.9\%$ pool censoring, our calibration audit serves primarily as a warning against blind 90-day Platt scaling and highlights maturity-aware post-hoc calibration as an open priority.
\end{itemize}

\subsection{Equal-Cap Inspection Cost, Observed Patterns vs.\ Exploratory Hypotheses, and Scope Limits}
\label{sec:disc_implications}

Translating ranking shifts into code-review impact hinges on budget granularity (Table~\ref{tab:rq5_paired_budget}). Preventing fixed-$0.5$ recall collapse rescues hundreds of defects only by flagging thousands of clean commits, and retrospective whole-stream budgeting penalizes \texttt{Survival-PU} on five of seven alarmed Benchmark~A hosts. Under realistic per-window 400-commit sprint budgets with closely matched realized LOC consumption, however, steep losses disappear or turn positive on Qt LightGBM, Qt DeepJIT-CNN, and OpenStack DeepJIT-CNN, while remaining negative on Qt CodeBERT.

Because practitioners cannot observe future latency multipliers or shift horizons in advance, Table~\ref{tab:decision_matrix} separates \textbf{Part~I: Directly Observed Empirical Patterns} from \textbf{Part~II: Exploratory Hypotheses}, which are not validated selection rules. For benchmark designers, three concrete practices follow directly. First, enforce $t_i^{\text{fix}} \le t_{\text{alarm}}$ alongside archive-horizon follow-up disclosure. Second, include linear models, zero-fit maturity priors, post-hoc controls, and adapted stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}. Third, pair whole-stream AP with per-window AP, scale-normalized AP, and operational per-window equal-cap review yields.

\begin{table}[!htb]
\centering
\caption{Synthesis separating (Part~I) directly observed empirical patterns across our evaluated cohorts from (Part~II) exploratory hypotheses and prospective considerations (not validated deployment selection rules).}
\label{tab:decision_matrix}
\fontsize{6.0}{7.3}\selectfont
\renewcommand{\arraystretch}{1.04}
\setlength{\tabcolsep}{1.9pt}
\begin{tabularx}{\textwidth}{L{2.30cm}L{2.05cm}Y Y}
\toprule
\textbf{Evaluated Cohort / Factor} & \textbf{Top Observed Methods} & \textbf{Observed Empirical Evidence (Part I)} & \textbf{Scope Boundary / Exploratory Hypothesis (Part II)} \\
\midrule
\multicolumn{4}{l}{\textit{Part I: Directly Observed Empirical Patterns Within Evaluated Benchmarks}} \\
\midrule
Merged-commit archival multimodal streams (Bench~A, 13 authoring metrics) & \textbf{Full-LR Frozen}, \textbf{LApredict Frozen} & \texttt{Full-LR} ($0.3653$ AP) and \texttt{LApredict} ($0.3601$) lead all deployable neural/tree hosts (best single host: \texttt{DeepJIT-CNN} $B_0 = 0.3531$). & Within Track-1 hosts, \texttt{SPU} recovers fixed-$0.5$ $F_1$ ($0.2907$); operational per-window equal-cap budget yield is host-dependent (Table~\ref{tab:rq5_paired_budget}). \\
Gradual multi-year tabular streams ($K=25$ clusters) & \textbf{Full-LR Adapt-4W}, \textbf{B2-4W}, \textbf{FixedCurve-PU}, \textbf{BORB} & Whole-stream AP: \texttt{Full-LR} ($0.5108$, $p_{\text{Holm}}=0.0639$), $B_2\text{-4W}$ ($0.5004$), $\text{Wait-}90$ ($0.4982$), \texttt{FixedCurve-PU} ($0.4942 > 0.4866$ \texttt{SPU}, $p_{\text{Holm}}=0.0295$). \texttt{BORB} leads $F_{1,\text{th}}$ ($0.5144$). & On 5-seed predictions, within-window standardization reverses the whole-stream mean gap vs.\ $B_2\text{-4W}$ (Rank-Norm: $0.5027$ vs.\ $0.4989$; Z-Norm: $0.4990$ vs.\ $0.4968$); per-window AP also shows a small non-significant edge ($0.5246$ vs.\ $0.5192$, unadjusted $p=0.0569$; Table~\ref{tab:window_ap_scale}). \\
Controlled concatenated shifts (4 streams) & \textbf{Survival-PU} (full) / \textbf{BORB-Alarm} (first 1.2k) & Full post-onset AP: \texttt{SPU} ($0.5764$) $>$ \texttt{ODaSC-Host} ($0.5727$). First 1,200c: \texttt{BORB-Alarm} ($0.5931$) $>$ \texttt{SPU} ($0.5591$). & Concatenated 1-hour splices ($23.7\%\text{--}60.0\%$ cross-boundary fixes) differ from natural drift; at $W=800$ on 3 shifts, $\text{Wait-}90$ ($0.6078$) overtakes \texttt{SPU} ($0.5974$). \\
Latency extremes ($\kappa=0\times, 3\times$) & \textbf{Survival-PU} ($F_1$/$F_{1,\text{th}}$), \textbf{FixedCurve-PU} ($3\times$ AP/raw Br.), $B_2\text{-4W}$ ($0\times$ AP) & $\kappa=0\times$: $B_2\text{-4W}$ ($0.5974$) $>$ \texttt{SPU} ($0.5760$) $>$ Fix ($0.5544$). $\kappa=3\times$: \texttt{SPU} leads tuned $F_{1,\text{th}}$ ($0.5022$) and $F_1$ ($0.5259$); Fix leads AP ($0.5547$) and raw Brier/ECE ($0.1693 / 0.0973$). & Raw scores beat 90d Platt under $65.9\%$ pool censoring (\texttt{SPU} is most resilient post-Platt); feature-dependent post-hoc calibration remains untested, and density-ratio omission remains an unvalidated simplification candidate not evaluated under extreme $\kappa$. \\
Independent semi-synthetic SZZ noise (5 PR-SZZ streams) & \textbf{Survival-PU}, \textbf{Wait-90}, \textbf{B2-4W} & Independent $\alpha_{\text{FP}}=8.42\%, \beta_{\text{FN}}=47.15\%$: batch methods cluster ($0.3768\text{--}0.3802$) well above \texttt{BORB} ($0.3187$) and \texttt{ORB} ($0.2782$). & Scoped to independent synthetic flips over automated PR-SZZ labels; feature- or age-dependent real SZZ noise remains untested. \\
\midrule
\multicolumn{4}{l}{\textit{Part II: Exploratory Hypotheses \& Prospective Considerations (Not Validated Selection Rules)}} \\
\midrule
Confirmatory repos ($N < 3{,}000$; Standalone-3 primary vs.\ Umbrella-4) & \textbf{ODaSC} (OHT), \textbf{BORB} (ENet), \textbf{ORB} (OHT) & Standalone-3 (unsplit): \texttt{ODaSC} $=0.7857$, $B_{1b}^* = 0.5951 < B_2\text{-4W} = 0.5985 < B_0 = 0.6565$. Frozen Umbrella-4: \texttt{ODaSC} $=0.8289$, $B_{1b}^* = 0.6801 < B_2\text{-4W} = 0.6858$ (LightGBM $0.7020$ vs.\ $0.6976$ n.s.). & \textit{Unvalidated hypothesis:} \texttt{HDFS*}/\texttt{MapReduce*} omit $11{,}964$ clean commits into \texttt{apache/hadoop}; horizon, prevalence, censoring, and project identity remain confounded on Standalone-3. \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{Scope Limitations.} Six limitations bound our conclusions:
\begin{enumerate}[leftmargin=1.6em,itemsep=0.06em]
  \item Benchmark~A orders merged Git revisions by \texttt{author\_date} rather than live Gerrit patch-set-1 uploads, and archive horizons $T_{\text{end}}$ are observable proxies rather than certified collection cutoffs.
  \item Stage~1 assumes local trailing-cohort stationarity (I1--I2), Stage~2 relies on heuristic shrinkage, and Eq.~\eqref{eq:ht_prev}'s outer $\pi_{\max}=0.70$ clamp causes unreachable bisection when alarm-time $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ exceeds $0.70$, occurring on $28.6\%\text{--}30.0\%$ of \texttt{HDFS*} alarms.
  \item Standard 90-day causal Platt calibration retains $27\%\text{--}66\%$ censoring under multi-month lag and degrades post-Platt calibration at $\kappa=3\times$ (Section~\ref{sec:rq4}).
  \item RQ4b's null alarm rate resamples the calibration validation pool rather than independent stationary production streams.
  \item Controlled concatenated shifts impose a 1-hour splice in which $23.7\%\text{--}60.0\%$ of revealed pre-onset fixes arrive after onset.
  \item RQ4e is a single-model (\texttt{deepseek-flash}), 640-commit prompting case study on public commits predating LLM pretraining cutoffs.
\end{enumerate}

% ============================================================================
% 8. Threats to Validity
% ============================================================================
\section{Threats to Validity}
\label{sec:threats}

\textbf{Construct Validity.} SZZ back-tracing is susceptible to tangled commits, refactorings, and unlinked fixes \citep{dacosta2017framework,rosa2021evaluating,fan2021impact}, while archival snapshots also incur end-of-archive censoring when $t_i^{\text{fix}} > T_{\text{end}}$. We mitigate construct threats by quarantining Tier-3 and retroactive links to $+\infty$ and auditing remaining follow-up across all 34 natural streams alongside Benchmark~A $\ge 180\text{d}$ and $\ge 365\text{d}$ sensitivity (Appendix Tables~\ref{tab:followup_audit}--\ref{tab:followup_sensitivity}). We also disclose upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split (Table~\ref{tab:confirmatory_funnel}), pair whole-stream AP with per-window and scale-normalized AP, report \texttt{LOC-Only} overlap, and verify paired equal-cap budget net TP counts (Table~\ref{tab:rq5_paired_budget}).

\textbf{Internal Validity.} All 41 method execution branches passed deterministic pipeline prefix-invariance tests with zero historical perturbation error, although prefix invariance certifies leak-free code execution rather than commit-time provenance of upstream Git exports. We separated development analyses on $K=25$ from the pre-unlock frozen confirmatory cohort, disclosed our post-unlock $N < 4{,}000$ sizing amendment explicitly, and treated $B_{1b}^*$ strictly as a Pote-inspired counterfactual rather than a verbatim original-repository reproduction. Automated verifiers further audit all $1{,}452$ empirical table cells across $22$ tables, locked parameters, exact-path prose claims, and relational overclaim guards (Appendix~\ref{app:repro}).

\textbf{External and Conclusion Validity.} Our adapted Track-2 implementations---ElasticNet-\texttt{BORB}, OHT-\texttt{ORB}, and OHT-\texttt{ODaSC}---reflect specific base learners rather than entire algorithmic families, and independent synthetic SZZ flips cannot capture feature-dependent real SZZ errors. Formal $\alpha=0.05$ Holm--Bonferroni inference is scoped strictly to $K=25$ and $K=24$ natural donor clusters, distinct from the exploratory unadjusted $p$-values in Table~\ref{tab:window_ap_scale}. Failed MLP directional replication on Phase~C, across both unsplit Standalone-3 and frozen Umbrella-4, is reported directly without post hoc retuning.

% ============================================================================
% 9. Conclusion
% ============================================================================
\section{Conclusion}
\label{sec:conclusion}

Unsupervised drift monitors remove label dependence when deciding \textit{when} to flag shifts, not when \textit{supervising} model updates under multi-month SZZ verification latency. Across multi-year ecosystems, 68.8\%--94.6\% of defects inside scheduled 400-commit windows remain unrevealed at update time. Consequently, leaking future fixes inflates whole-stream AP significantly across $K=25$ donor clusters on both MLP ($+0.0237$, $p_{\text{Holm}} = 6.63 \times 10^{-5}$) and LightGBM ($+0.0476$, $p_{\text{Holm}} = 3.70 \times 10^{-5}$). Meanwhile, circular block-bootstrap calibration cuts validation-pool null alarms $4.8\times$ and detects abrupt shifts $3\times$ faster than periodic schedules.

Our cross-paradigm audit shows that no single adaptive strategy dominates every regime. Frozen linear models lead deployable Benchmark~A. On gradual $K=25$ whole-stream AP, simpler causal baselines---$B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU}---significantly outperform cohort-fitted \texttt{Survival-PU}, reversing the mean $B_2\text{-4W}$ gap non-significantly under within-window score standardization. ElasticNet \texttt{BORB} leads $K=25$ $F_{1,\text{th}}$ and early post-shift recovery, while Hoeffding \texttt{ODaSC} leads the short confirmatory cohort at $0.7857$ on Standalone-3 and $0.8289$ on Umbrella-4, where MLP hindsight optimism fails to replicate.

Within these operating boundaries, \texttt{Survival-PU} delivers targeted gains on full post-shift AP at $0.5764$, LightGBM soft-target distillation, and classification protection at both latency extremes. Specifically, it avoids \texttt{FixedCurve-PU}'s $\kappa = 0\times$ misspecification and beats $B_2\text{-4W}$, two post-hoc $B_2$ controls, and \texttt{FixedCurve-PU} on threshold-tuned $F_{1,\text{th}}$ at $0.5022$ under $\kappa = 3\times$ latency. Finally, our calibration audit warns that standard 90-day Platt scaling degrades probability estimates when calibration pools suffer heavy censoring; developing maturity-aware post-hoc calibration and prospectively validating observable deployment selection rules remain central priorities for adaptive JIT-SDP.

\backmatter

% ============================================================================
% Declarations (Springer Nature / Software Quality Journal Standard)
% ============================================================================
\section*{Declarations}
\begin{itemize}[leftmargin=1.5em,itemsep=0.12em]
  \item \textbf{Funding:} This research received no external funding.
  \item \textbf{Conflict of interest:} The authors declare that they have no conflict of interest.
  \item \textbf{Competing interests:} The authors declare no competing financial or non-financial interests that are relevant to the content of this article.
  \item \textbf{Ethics approval and consent to participate:} Not applicable; this study analyzes publicly available open-source software repository histories without human-subject experimentation.
  \item \textbf{Consent for publication:} Not applicable.
  \item \textbf{Availability of data, code, and artifacts:} The complete replication package---including immutable \texttt{src\_v3/} modules, \texttt{scripts\_v3/} post-processing, companion audit, and verification runners, pinned \texttt{requirements\_\allowbreak lock\_\allowbreak v3.txt}, \texttt{LICENSE} (MIT), SHA-256 manifests for all 94 stream bundles and CodeBERT caches, 50 commit-level \texttt{.npz} prediction archives, lossless archived LLM exchange logs, and automated verifiers---is publicly available at \url{https://github.com/789748456v/CausalJIT-Audit-SQJ} and accompanied by a full offline archival bundle for permanent Zenodo deposit upon acceptance.
  \item \textbf{Author contributions:} Z.L.: Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Writing -- Original Draft. S.X.: Software, Validation, Data Curation, Writing -- Review \& Editing. K.L.: Validation, Investigation, Visualization, Writing -- Review \& Editing. S.M.: Formal Analysis, Investigation, Writing -- Review \& Editing. F.G.: Validation, Visualization, Writing -- Review \& Editing. Y.Z.: Supervision, Conceptualization, Methodology, Project Administration, Writing -- Review \& Editing. All authors read and approved the final manuscript.
\end{itemize}

% ============================================================================
% Appendices
% ============================================================================
\begin{appendices}
\renewcommand{\theHtable}{\thesection.\arabic{table}}

\section{Executed Host Architectures, Soft-Target GBDT Proof, and Traceability}
\label{app:repro}

Table~\ref{tab:backbone_specs} documents the exact executed constructors and training budgets in \texttt{src\_v3/\allowbreak models\_\allowbreak and\_\allowbreak baselines\_\allowbreak v3.py}. All neural models optimize with \texttt{AdamW} (weight decay $10^{-4}$, batch size $128$, gradient-norm clip $1.0$, deterministic CuDNN). Token vocabularies ($8{,}000$ tokens), TF-IDF/SVD projections ($48$-D msg + $64$-D diff = $112$-D SVD), and $z$-score scalers fit \textbf{strictly on $[0, n_{\text{init}})$}, whereas initial host weights and optional tabular linear-residual skips (\texttt{init\_\allowbreak linear\_\allowbreak skip\_\allowbreak from\_\allowbreak data}) train on the causal pre-stream prefix $[0, n_{\text{init}}+n_{\text{val}})$ using observable labels $y_{\text{obs}}(t_{\text{start}})$. For LightGBM, DriftLens computes $24$-D PCA Gaussian Fr\'echet distances on the concatenation of the leading $\min(D, 48)$ standardized features and the initial booster's exact raw margin $x_{i,1:\min(D,48)} \oplus [f_{\text{raw}}(x_i) + z_{\text{init}}]$.

\begin{proposition}[Analytical Margin Derivatives for Soft-Target GBDT]
\label{prop:lgbm}
For $\mathcal{L}_{\text{SPU},i}(z_i) = u_i[\omega_+ q_i \log(1 + e^{-z_i}) + (1 - q_i)\log(1 + e^{z_i})]$ with $q_i \in [0, 1]$, $u_i > 0$, and margin $z_i = z_{\text{init}} + \sum_{m=1}^M T_m(x_i)$, the exact first and second derivatives with respect to $z_i$ are $g_i = u_i[(1 + (\omega_+ - 1)q_i)\sigma(z_i) - \omega_+ q_i]$ and $h_i = u_i(1 + (\omega_+ - 1)q_i)\sigma(z_i)(1 - \sigma(z_i)) > 0$. Passing $z_{\text{init}}$ as the training initial margin and adding $z_{\text{init}}$ to raw tree margins at inference preserves exact split-gain margins for continuous $q_i \in [0, 1]$ without dataset duplication.
\end{proposition}

\begin{proof}
Differentiating $\mathcal{L}_{\text{SPU},i}(z_i)$ via $\frac{d}{dz}\log(1+e^{-z}) = \sigma(z)-1$ and $\frac{d}{dz}\log(1+e^z) = \sigma(z)$ yields $g_i$ and $h_i = \partial g_i/\partial z_i$. Because LightGBM excludes training initial scores from raw tree sums at prediction time, adding $z_{\text{init}}$ restores $z_i$ (verified in our pre-unlock gate to match native hard-label LightGBM within $1.19 \times 10^{-7}$ max float32 host margin error).
\end{proof}

\begin{table}[!htb]
\centering
\caption{Executed architectural, DriftLens monitoring-space, and optimization specifications across all five host predictors (\texttt{src\_v3/models\_and\_baselines\_v3.py}).}
\label{tab:backbone_specs}
\fontsize{5.8}{7.0}\selectfont
\renewcommand{\arraystretch}{1.02}
\setlength{\tabcolsep}{1.6pt}
\begin{tabularx}{\textwidth}{L{1.55cm}L{1.85cm}Y L{1.95cm}L{2.55cm}}
\toprule
\textbf{Host Model} & \textbf{Causal Input Features} & \textbf{Architecture \& Linear-Residual Skip} & \textbf{DriftLens Space ($24$-D PCA)} & \textbf{Executed Init \& Per-Alarm Budgets} \\
\midrule
Attention-CNN \citep{pote2026adaptive} & $64$ msg + $128$ diff tokens + $13$ authoring metrics & $64$-D dual emb.; $k\in\{1,2,3\}$ 1D-CNN ($48$ filters $\to 144$-D); $6$-head self/cross-attn; $192\to 64$-D ($\text{drop}=0.30$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-\allowbreak ReLU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=7{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ ($B_{1/2}$/Wait/ODaSC-H), $6$ ep.\ ($\text{nnPU}$), $4{+}4$ ep.\ ($\text{SPU}$/Fixed), or step-matched \texttt{max\_\allowbreak opt\_\allowbreak steps} \\
CodeBERT \citep{feng2020codebert} & $768$-D msg + $768$-D diff (\texttt{codebert-\allowbreak base}) + $13$ metrics & Dual $128$-D LayerNorm-\allowbreak GELU adapters + $4$-head bimodal self-attn + $|a_m-a_c| \oplus (a_m\odot a_c) \to 160 \to 64$-D ($\text{drop}=0.25$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-\allowbreak GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=8{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ ($B_{1/2}$/Wait/ODaSC-H), $6$ ep.\ ($\text{nnPU}$), $4{+}4$ ep.\ ($\text{SPU}$/Fixed), or step-matched \texttt{max\_\allowbreak opt\_\allowbreak steps} \\
DeepJIT-CNN \citep{hoang2019deepjit} & $64$ msg + $128$ diff tokens + $13$ metrics & $64$-D dual emb.; parallel $k\in\{1,2,3\}$ 1D-CNNs ($64$ filters) + $48$-D metric MLP $\to 64$-D ($\text{drop}=0.25$) + frozen Tab-13 LR skip & $64$-D penultimate LayerNorm-\allowbreak GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=7{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ ($B_{1/2}$/Wait/ODaSC-H), $6$ ep.\ ($\text{nnPU}$), $4{+}4$ ep.\ ($\text{SPU}$/Fixed), or step-matched \texttt{max\_\allowbreak opt\_\allowbreak steps} \\
Tabular MLP (Benchmark B/C) & Standardized commit-time / PR-SZZ metrics ($D$-dim) & LayerNorm-\allowbreak GELU ($D\to 128$) + residual block ($128\to 128$) $\to 64$-D head ($\text{drop}=0.20$) & $64$-D penultimate LayerNorm-\allowbreak GELU vector $z_i$ & AdamW ($\text{bs}=128$); Init: $12$ ep., $\text{lr}=8{\times}10^{-4}$; Alarm ($\text{lr}=4.5{\times}10^{-4}$): $5$ ep.\ ($B_{1/2}$/Wait/ODaSC-H), $6$ ep.\ ($\text{nnPU}$), $4{+}4$ ep.\ ($\text{SPU}$/Fixed), or step-matched \texttt{max\_\allowbreak opt\_\allowbreak steps} \\
LightGBM \citep{ke2017lightgbm} & $112$-D msg+diff SVD + $13$ metrics (or $D$ tab.) & Callable-objective GBDT (Prop.~\ref{prop:lgbm}), $\text{num\_leaves}=15$, $\text{lr}=0.05$, subsample $0.85$, colsample $0.85$, $\ell_1=0.1, \ell_2=1.0$ & $[x_{i,1:\min(D,48)}, z_i]$ with exact \texttt{init\_score} margin $z_i$ & Init: $90$ rounds; Alarm: fresh $75$-tree fit ($B_{1/2}$/Wait/ODaSC-H/Teach-Exact), fresh $75{+}75$ fits ($\text{SPU}$/Fixed), fresh $150$-tree fit (\texttt{Teach-\allowbreak Matched}); 2 threads \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{Executed Competitor Specifications (\texttt{src\_v3/\allowbreak stream\_\allowbreak competitors\_\allowbreak v3.py}).}
\begingroup\raggedright
\begin{itemize}[leftmargin=1.6em,itemsep=0.08em]
  \item \texttt{ORB\_\allowbreak OHT\_\allowbreak Native} \citep{cabral2019class}: Ensemble of $10$ \texttt{river.\allowbreak tree.\allowbreak Hoeffding\allowbreak TreeClassifier} learners ($\text{grace\_period}=200$, $\delta = 10^{-7}$, $\tau = 0.05$, \texttt{leaf\_\allowbreak prediction="nba"}) trained via OzaBagging \citep{oza2005online} on strict $t_{\text{event}} < t_{\text{commit}}$ verification-latency events ($\text{waiting\_days} = 90.0$ clean maturity events plus revealed fix events), with exact feature-tuple duplicate clean-count noise filtering ($\text{count} > 5$ skipped prior to $p_1$ updates, matching \texttt{jitsdp/\allowbreak baseline.py}), EMA decay $0.99$, prediction-rate window $100$, $th=0.40, l_0=10, l_1=12, m=1.5$, and cold-start $0.0$ output until both classes are observed.
  \item \texttt{BORB\_\allowbreak Native} \& \texttt{BORB\_\allowbreak Alarm} \citep{cabral2023borb}: Single \texttt{SGDClassifier} (\texttt{loss=\allowbreak "log\_loss"}, \texttt{penalty=\allowbreak "elasticnet"}, $\alpha=0.02$, \texttt{l1\_ratio}$=0.15$, \texttt{learning\_\allowbreak rate=\allowbreak "optimal"}) refreshed every $200$ commits (\texttt{BORB\_\allowbreak Native}) or strictly post-window upon DriftLens alarms (\texttt{BORB\_\allowbreak Alarm}), running $20$ inner skewed-resampling \texttt{partial\_fit} iterations ($\text{max\_sample\_size} = 2{,}000$, $th=0.40, l_0=10, l_1=12, m=1.5$) over all historical revealed positives ($t_{\text{fix}} \le t$) and unfixed commits aged $\ge 90.0$ days.
  \item \texttt{ODaSC\_\allowbreak OHT\_\allowbreak Native} \citep{song2022odasc}: Ensemble of $10$ \texttt{river} Hoeffding trees ($\text{grace\_period}=200, \delta=10^{-7}, \tau=0.05$) consuming two-stage $t_{\text{event}} < t_{\text{commit}}$ events with $\text{waiting\_days} = 15.0$ provisional-negative horizon and \texttt{SongDenStream\allowbreak ConfidenceTracker} one-sided micro-cluster label confidences ($\delta_{\text{thresh}} = 0.85$, radius $\epsilon_r$ initialized from the $15$th pairwise distance percentile on $[0, \min(s_0, 250))$) across all $D$ features.
  \item \texttt{Pure\_nnPU} \citep{kiryo2017positive}: Disjoint historical confirmed-positive set $P$ and $4W$ unlabeled marginal pool $U$ trained with non-negative PU risk ($\beta = 0.0, \gamma = 1.0$; $6$ neural epochs or unit pseudo-Hessian GBDT steps) using clipped inverse-maturity shrunk prior $\hat{\pi}_{\text{target}}$ on $4W$.
\end{itemize}
\endgroup

\textbf{Code-to-Manuscript Traceability Mapping (\texttt{src\_v3/} and \texttt{scripts\_v3/}).} The following mapping links protocol concepts in Sections~\ref{sec:method}--\ref{sec:threats} to their implementation symbols and archived artifacts:
\begingroup\raggedright
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Stage 1--3 Estimators \& Training:} Risk-eligible cohort incidence is implemented by \texttt{Cohort\allowbreak Incidence\allowbreak Estimator} and its $145$-knot fallback \texttt{\_set\_fallback()} in \texttt{src\_v3/\allowbreak survival\_\allowbreak estimators\_\allowbreak v3.py}; clipped inverse-maturity shrunk prevalence by \texttt{estimate\_\allowbreak ht\_\allowbreak shrunk\_\allowbreak prevalence}; shared Bayesian posterior by \texttt{compute\_\allowbreak survival\_\allowbreak pu\_\allowbreak posterior}; neural/GBDT host fitting by \texttt{train\_\allowbreak supervised\_\allowbreak v3} and \texttt{LightGBM\allowbreak HostV3} in \texttt{src\_v3/\allowbreak models\_\allowbreak and\_\allowbreak baselines\_\allowbreak v3.py}; and three-branch causal calibration by \texttt{calibrate\_\allowbreak causal\_\allowbreak stream\_\allowbreak probs\_\allowbreak strict} in \texttt{src\_v3/\allowbreak stream\_\allowbreak competitors\_\allowbreak v3.py}.
  \item \textit{Data Bundles, Split Rules, \& Codified Constants:} Stream sizing is implemented by \texttt{compute\_\allowbreak split\_\allowbreak sizes} and confirmatory unlocking via \texttt{data\_\allowbreak audit\_\allowbreak v3.py -{-}unlock-\allowbreak confirmatory}; codified interpretation rules and forensic rules reside in \texttt{PRE\_\allowbreak REGISTERED\_\allowbreak INTERPRETATION\_\allowbreak BOUNDARIES} and \texttt{V3\_CONFIG.\allowbreak forensic\_rules} (\texttt{Tier1\_\allowbreak McIntosh\_\allowbreak CSV}) in \texttt{src\_v3/\allowbreak config\_\allowbreak v3.py}; natural multimodal bundles are \texttt{qt\_\allowbreak natural\_\allowbreak 25k} and \texttt{openstack\_\allowbreak natural\_\allowbreak 13k} (with \texttt{premerge28} sensitivity); the ten-stream subset is \texttt{REPRESENTATIVE\_\allowbreak 10\_\allowbreak STREAMS} in \texttt{src\_v3/\allowbreak run\_\allowbreak v3\_\allowbreak experiments.py} (containing shift streams \texttt{abrupt\_\allowbreak nova\_\allowbreak zulip}, \texttt{abrupt\_\allowbreak ignite\_\allowbreak cassandra}, \texttt{abrupt\_\allowbreak airflow\_\allowbreak pulsar}; Benchmark~B's fourth shift is \texttt{abrupt\_\allowbreak deepjit\_\allowbreak nova\_\allowbreak neutron\_\allowbreak pure13}). Historical RQ4 artifact filenames \texttt{rq4d\_\allowbreak szz\_\allowbreak perturbation\_\allowbreak v3.json}, \texttt{rq4e\_\allowbreak sensitivity\_\allowbreak sweeps\_\allowbreak v3.json}, and \texttt{rq4f\_\allowbreak llm\_\allowbreak audit\_\allowbreak v3.json} correspond to manuscript subsections RQ4c, RQ4d, and RQ4e.
  \item \textit{Automated Verification Suites \& Companion Controls:} Pre-unlock gate archive (\texttt{766e3fd6...}) and live post-campaign Checks 2--14 run via \texttt{scripts\_v3/\allowbreak run\_\allowbreak post\_\allowbreak campaign\_\allowbreak mechanism\_\allowbreak checks\_\allowbreak v3.py}; the matched RQ4a \texttt{FixedCurve-PU} control (\texttt{rq4a\_\allowbreak fixedcurve\_\allowbreak control\_\allowbreak v3.json}) runs via \texttt{scripts\_v3/\allowbreak run\_\allowbreak rq4a\_\allowbreak fixedcurve\_\allowbreak control\_\allowbreak v3.py}; all companion data-funnel, follow-up sensitivity, unrevealed-commit Stage 1--3, symmetrical Raw/Platt \& PostHoc Logit/Ratio (with clipping/tie-rate diagnostics), 5-seed window-AP/scale-removal, and operational equal-cap budget audits (\texttt{reviewer\_\allowbreak mechanism\_\allowbreak audits\_\allowbreak v3.json}, exact $0.00$ parity with \texttt{rq4a\_\allowbreak latency\_\allowbreak intervention\_\allowbreak v3.json}) run via \texttt{scripts\_v3/\allowbreak run\_\allowbreak reviewer\_\allowbreak mechanism\_\allowbreak audits\_\allowbreak v3.py}; all $1{,}452$ empirical table cells across $22$ tables and provenance hashes across all $15$ JSON outputs verify via \texttt{src\_v3/\allowbreak verify\_\allowbreak manuscript\_\allowbreak v3.py}; and all $33$ \texttt{V3\_CONFIG} table fields, exact-path prose claims, and quantifier/relational claim guards verify via \texttt{scripts\_v3/\allowbreak verify\_\allowbreak prose\_\allowbreak claims\_\allowbreak v3.py}.
\end{itemize}
\endgroup

\section{Additional Data-Funnel, Follow-Up, Mechanism, Shift, Prompting, and Forensics Audits}
\label{app:audits}
\setcounter{table}{0}

This appendix consolidates seven supplementary audit tables referenced across Sections~\ref{sec:benchmarks}--\ref{sec:rq5}. Tables~\ref{tab:confirmatory_funnel}--\ref{tab:followup_sensitivity} document the confirmatory ApacheJIT filtering funnel---reconciling upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split---alongside archive-horizon follow-up across all 34 natural streams and Benchmark~A sensitivity at $\ge 180\text{d}$ and $\ge 365\text{d}$ cutoffs.

[[TAB_CONFIRMATORY_FUNNEL]]

[[TAB_FOLLOWUP_AUDIT]]

[[TAB_FOLLOWUP_SENSITIVITY]]

Table~\ref{tab:mechanism_audit} audits Stage~1--3 intermediate accuracy across all 62 fired DriftLens alarms on the ten representative streams, covering $74{,}400$ pool evaluations and $59{,}600$ unique commits. It separates strictly unrevealed commits ($y_{\text{obs}}=0$) from the all-pool positive-writeback control while comparing \texttt{Survival-PU} against \texttt{FixedCurve-PU}. Reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is evaluated over all archive-labeled positives $\mathcal{P}$ per stream, counting unlinked fixes as unrevealed at finite ages, and averaged per alarm before aggregating across alarms or streams; all $\pm$ dispersion terms report sample SD.

[[TAB_MECHANISM_AUDIT]]

Table~\ref{tab:shift_4streams} disaggregates all four Benchmark~B controlled concatenated shifts alongside cross-boundary fix-reveal rates, Table~\ref{tab:rq4f_llm} presents the full paired Code-LLM prompting study, and Table~\ref{tab:rq5_forensics} archives the Tier-1 SZZ commit illustrations discussed in Section~\ref{sec:rq5}.

[[TAB_SHIFT_4STREAMS]]

[[TAB_RQ4F_LLM]]

\textbf{RQ4e Paired Prompting Protocol Details.} The 6,400 logical evaluations in Table~\ref{tab:rq4f_llm} queried \texttt{deepseek-flash} at \texttt{temperature=0.0} with strict JSON schema output across the first 320 evaluation commits of natural Qt and OpenStack. Across $6{,}819$ total HTTP attempts, $419$ transient transport errors were retried and only two fell back to $0.50$, yielding a $99.97\%$ valid-response rate, $100\%$ JSON compliance on $6{,}398$ responses, and $99.95\%$ non-empty diff coverage. Zero-shot prompts supplied the commit message, truncated code diff, and commit-time metrics. Paired 6-shot arms sampled identical historical $1W$ exemplars across three random seeds differing solely in $y_{\text{obs}}(t)$ versus $y^*$, whereas the descriptive causal-mature arm drew aged $\ge 90\text{d}$ exemplars, reaching $0.5072$ $F_1$ on OpenStack and $0.2984$ on Qt.

[[TAB_RQ5_FORENSICS]]

\textbf{RQ5 Qualitative Commit Illustrations and Churn Skew.} Applying our pre-declared Tier-1 forensic filter---requiring lag $\ge 14\text{d}$, $B_0 \ge 0.45$, $B_2\text{-4W} \le 0.35$, and $\text{SPU} \ge 0.50$---on natural Qt CodeBERT seed 42 yields $22$ rescue illustrations (Table~\ref{tab:rq5_forensics}), where Alarm~\#1 at commit $9{,}900$ experienced $88.19\%$ $4W$ censoring; zero alarms fired on OpenStack CodeBERT seed 42. Across full evaluation sets, Tier-1 false-negative candidates skew smaller than all defects: $436$ Qt FNs have median $32$ versus $69$ LOC across all $981$ defects, and $477$ OpenStack FNs have median $74$ versus $89$ LOC. Conversely, high-score false positives skew larger than clean commits, with medians of $65$ versus $9$ LOC on Qt ($1{,}620$ FPs) and $146$ versus $19$ LOC on OpenStack ($18$ FPs), spanning $2$-LOC faults (\texttt{qtmultimedia:4bb75559}) to $570{,}767$-LOC imports (\texttt{qtwebkit:d441d6f3}).

\end{appendices}

\bibliography{references}

\end{document}
"""
    replacements = {
        "[[TAB_AUDIT_CENSORING]]": tmap["tab:audit_censoring"],
        "[[TAB_BENCH_A_MAIN]]": tmap["tab:bench_a_main"],
        "[[TAB_BENCH_A_SENS]]": tmap["tab:bench_a_sens"],
        "[[TAB_BENCH_B_CLUSTERS]]": tmap["tab:bench_b_clusters"],
        "[[TAB_BENCH_B_HOLM]]": tmap["tab:bench_b_holm"],
        "[[TAB_WINDOW_AP_SCALE]]": tmap["tab:window_ap_scale"],
        "[[TAB_CONFIRMATORY]]": tmap["tab:confirmatory"],
        "[[TAB_RQ3_ABLATIONS]]": tmap["tab:rq3_ablations"],
        "[[TAB_RQ4A_KAPPA]]": tmap["tab:rq4a_kappa"],
        "[[TAB_RQ4A_FIXEDCURVE]]": tmap["tab:rq4a_fixedcurve"],
        "[[TAB_RQ4A_POSTHOC_B2]]": tmap["tab:rq4a_posthoc_b2"],
        "[[TAB_RQ4B_MONITORS]]": tmap["tab:rq4b_monitors"],
        "[[TAB_RQ4D_SZZ]]": tmap["tab:rq4d_szz"],
        "[[TAB_RQ4F_LLM]]": tmap["tab:rq4f_llm"],
        "[[TAB_RQ5_RUNTIME]]": tmap["tab:rq5_runtime"],
        "[[TAB_RQ5_PAIRED_BUDGET]]": tmap["tab:rq5_paired_budget"],
        "[[TAB_RQ5_FORENSICS]]": tmap["tab:rq5_forensics"],
        "[[TAB_CONFIRMATORY_FUNNEL]]": tmap["tab:confirmatory_funnel"],
        "[[TAB_FOLLOWUP_AUDIT]]": tmap["tab:followup_audit"],
        "[[TAB_FOLLOWUP_SENSITIVITY]]": tmap["tab:followup_sensitivity"],
        "[[TAB_MECHANISM_AUDIT]]": tmap["tab:mechanism_audit"],
        "[[TAB_SHIFT_4STREAMS]]": tmap["tab:shift_4streams"],
    }
    for k, v in replacements.items():
        assert k in tex, f"Placeholder {k} missing in main.tex template!"
        tex = tex.replace(k, v)
    return tex


def build_markdown_manuscript() -> str:
    return """# Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction

> **Synchronized Manuscript Summary (`manuscript/main.tex` / `manuscript/main.pdf` is the canonical camera-ready Springer `sn-jnl` LaTeX source verified across all 1,452 empirical table cells across 22 tables and 15 JSON outputs by `src_v3/verify_manuscript_v3.py` and across `V3_CONFIG` parameter bindings + exact-path-validated prose claims by `scripts_v3/verify_prose_claims_v3.py`).**

## Structured Abstract

- **Context:** Unsupervised monitors like DriftLens flag representation shifts in Just-in-Time Defect Prediction without labels, motivating immediate window retraining.
- **Objective:** Monitoring determines when to retrain, not how to supervise updates: under multi-month SZZ latency, recent bug-inducing commits still appear clean. We audit this causal gap and map boundaries across linear, streaming, and Positive-Unlabeled learners.
- **Method:** Across 37 development streams and five confirmatory ApacheJIT repositories—34 natural streams, eight controlled shifts, 347,122 natural commits, and a paired Code-LLM study—we pair block-bootstrap DriftLens with three-stage **Survival-PU**, combining cohort-incidence maturity, clipped prior shrinkage (`c_ht = 0.10`), and soft-target distillation.
- **Results:** Across scheduled 400-commit boundaries, 68.8%–94.6% of defects remain right-censored. Window-oracle supervision significantly inflates whole-stream Average Precision over causal retraining across $K=25$ clusters, adding $+0.0237$ on MLP ($p_{\\text{Holm}}=6.63\\times 10^{-5}$) and $+0.0476$ on LightGBM ($p_{\\text{Holm}}=3.70\\times 10^{-5}$). Under strict causal visibility, frozen logistic regression leads Benchmark A at $0.3653$ AP. On gradual $K=25$ whole-stream AP, `Full-LR Adapt-4W` leads in mean at $0.5108$, while `B2-4W`, `Wait-90`, and zero-fit `FixedCurve-PU` at $0.4942$ ($p_{\\text{Holm}}=0.0295$) significantly exceed `Survival-PU` at $0.4866$; within-window standardization reverses the mean `B2-4W` gap non-significantly. On the confirmatory cohort, MLP hindsight optimism does not replicate, and `ODaSC` leads unsplit Standalone-3 at $0.7857$ and umbrella-4 at $0.8289$. Conversely, `Survival-PU` leads full post-shift AP at $0.5764$, avoids $\\kappa=0\\times$ misspecification, and protects $F_1$ under $3\\times$ latency where `FixedCurve-PU` leads AP and raw Brier.
- **Conclusions:** Chronological splitting permits future-fix leakage; adaptation must match shift regime, latency, calibration, and granularity.

---

## Core Empirical & Statistical Findings

1. **RQ1 — Causal Supervision Audit:**
   - Across scheduled 400-commit ($1W$) boundaries in multi-year cohorts, 68.8%–94.6% of window defects are right-censored (`Qt`: 94.6% scheduled, 95.3% pooled across 981 window-defects, 95.7% at LightGBM alarms; `OpenStack`: 87.5% / 84.4%; `Cabral-14`: 68.8% / 68.2%; `ApacheJIT-8`: 70.4% / 65.8%; `PR-SZZ-5`: 78.8% / 78.9%).
   - Across $K=25$ independent donor clusters, offline window-oracle supervision (`B1b_4W_Oracle`) significantly inflates AP over matched causal `B2_4W` on both MLP (`0.5241` vs. `0.5004`, $+0.0237$, 20/0/5, $p_{\\text{Holm}}=6.63\\times 10^{-5}$) and LightGBM (`0.5409` vs. `0.4933`, $+0.0476$, 23/0/2, $p_{\\text{Holm}}=3.70\\times 10^{-5}$). Causal `B2-4W` improves AP over `B0` on both natural streams (`0.5004` vs. `0.4566`) and controlled shifts (`0.5578` vs. `0.5064`), whereas single-window `B2-1W` (`0.5011`) falls below `B0` on the controlled shifts.

2. **RQ2 — Simple Linear Winners on Benchmark A, $K=25$ Cluster Trade-Offs, & Confirmatory Non-Replication:**
   - **Benchmark A (6 Commit-Time Qt/OpenStack Streams):** Simple frozen tabular logistic regression (`Full-LR Frozen = 0.3653` AP, `0.3939` $F_1$, `0.3951` $F_{1,\\text{th}}$) and `LApredict Frozen` (`0.3601` AP) outperform every deployable single-host configuration (best deployable single host: `DeepJIT-CNN B0 = 0.3531`, `Att-CNN B0 = 0.3525`; only non-deployable full-hindsight `DeepJIT-CNN B1a* = 0.3768` exceeds `Full-LR`), and `LOC-Only` reaches `0.6520` R@20%Effort. Within the four multimodal/GBDT hosts, `Survival-PU` recovers the highest Track-1 $F_1$ (`0.2907` vs. `0.1650` `B2-4W`) and Track-1 $F_{1,\\text{th}}$ (`0.3043`).
   - **Benchmark B ($K=25$ Natural Donor Clusters):** `Full-LR Adapt-4W` (`0.5108`, $p_{\\text{Holm}}=0.0639$) attains top mean AP, while `B2-4W` (`0.5004`, $p_{\\text{Holm}}=0.0056$), `Wait-90` (`0.4982`, $p_{\\text{Holm}}=0.0069$), and zero-fit `FixedCurve-PU` (`0.4942`, $p_{\\text{Holm}}=0.0295$) significantly exceed `Survival-PU` (`0.4866`, which beats `Pure-nnPU = 0.4461`, $p_{\\text{Holm}}=0.0020$, and `ORB = 0.4468`, $p_{\\text{Holm}}=0.0279$). On 5-seed predictions (`42..46`), within-window standardization reverses the whole-stream mean gap vs. `B2-4W` (Rank-Norm: `0.5027` vs. `0.4989`; Z-Norm: `0.4990` vs. `0.4968`), and per-window AP shows a small non-significant edge (`0.5246` vs. `0.5192`, unadjusted $p=0.0569$) while remaining numerically close to `Wait-90`, `FixedCurve-PU`, and `Full-LR Adapt-4W`. ElasticNet `BORB` leads $F_{1,\\text{th}}$ (`0.5144`), while `FixedCurve-PU` (`0.15247`) and `Survival-PU` (`0.15255`) are numerically close at `0.1525` on post-Platt Brier loss. On LightGBM, Stage 3 beats `Teacher-Only (Matched)` (`0.4619` vs. `0.4240`, $p_{\\text{Holm}}=9.87\\times 10^{-4}$), while `B2-4W` leads overall (`0.4933`).
   - **Controlled Abrupt Shifts (4 Separated Streams):** `Survival-PU` achieves the highest full post-onset mean AP (`0.5764`, ahead of `ODaSC-Host = 0.5727`), whereas `BORB-Alarm` (`0.5931`) and `BORB-Native` (`0.5734`) lead the immediate first 1,200 post-onset commits (`Survival-PU = 0.5591`).
   - **Phase C Untouched Confirmatory Cohort (5 ApacheJIT Repos, Standalone-3 Primary vs. Frozen Umbrella-4):** Disclosing upstream `apachejit_total.csv`'s `11,964`-commit clean `apache/hadoop` split, MLP hindsight optimism did **not** replicate on unsplit Standalone-3 (`B1b* = 0.5951` vs. `B2-4W = 0.5985`; `B0 = 0.6565`; `ODaSC = 0.7857`) or frozen Umbrella-4 (`B1b* = 0.6801` vs. `B2-4W = 0.6858`, 0/0/4, exact $p=0.125$; `ODaSC = 0.8289`).

3. **RQ3, RQ4, & RQ5 — Sensitivity, Unrevealed Mechanism Audit, Symmetrical Raw/Platt Controls, DriftLens Factorial, SZZ Noise, Audited `deepseek-flash`, & Equal-Cap Budget Error Analysis:**
   - Risk-eligible cohort incidence (`0.5543`), $10^{-4}$d Kaplan–Meier (`0.5624`), global Weibull cure (`0.5652`), and `FixedCurve-PU` (`0.5639`) span `1.09` AP points across 10 representative streams with zero solver fallbacks; on 62 alarms (`59,600` unique commits; SD across alarms/streams), cohort incidence improves Stage-1 $\\hat{c}(a)$ MAE over `FixedCurve-PU` (`0.0938` vs. `0.1204`) yet `FixedCurve-PU` attains slightly lower Stage-2 $\\hat{\\pi}_{\\text{target}}$ MAE (`0.0565` vs. `0.0620`) and unrevealed-only (`y_obs=0`) Brier (`0.1200` vs. `0.1236`; both improving over de-weighted teachers at `0.1303`).
   - At $\\kappa=3.0\\times$ future latency (`RQ4a`), zero-fit `FixedCurve-PU` attains top deployable AP (`0.5547`) and lowest raw Brier/ECE (`0.1693 / 0.0973`), whereas `Survival-PU` outperforms `B2-4W` (`0.3825` $F_1$, `0.4465` $F_{1,\\text{th}}$), alarm-synchronized `PostHoc Logit` (`0.3593 / 0.4420`), Proposition-1-inspired plug-in `PostHoc Ratio` (`0.3216 / 0.4543`), and `FixedCurve-PU` (`0.5132 / 0.4918`) on $F_1$ (`0.5259`) and $F_{1,\\text{th}}$ (`0.5022` raw; `0.4898` Platt) while achieving the lowest final post-Platt Brier (`0.1945`) and ECE (`0.1603`) among the five tested methods under the `65.9%`-censored 90-day calibrator (which worsens ECE across all five methods and Brier across four of five; `PostHoc Logit`: `0.2020 -> 0.2018`), and beats `FixedCurve-PU` across all reported metrics at $\\kappa=0.0\\times$ (`0.5760` vs. `0.5544` AP; `0.5364` vs. `0.4964` $F_1$; `0.1631` vs. `0.1687` Brier). Holding $(p_{95}, \\text{cd}=1)$ fixed (`RQ4b`), circular block bootstrap lowers validation-pool resampled-null alarm rates $4.8\\times$ (`20.2% -> 4.2%` static; `16.4% -> 3.4%` rolling; `32.1%` vs. `43.5%` 10-stream alarm rate) with `0/20` missed transitions under `Rolling Ref+Th` (`310` commits conditional delay for `IID + Rolling`, `350` commits for `Block + Rolling`). Under independent semi-synthetic SZZ noise (`RQ4c`), batch methods (`0.3768–0.3802`) outperform `BORB` (`0.3187`) and `ORB` (`0.2782`). Across RQ4d's 3 representative controlled shifts, `Survival-PU` leads fixed-1,200 post-onset AP at $W \\in \\{200, 400, 600\\}$ and trails `Wait-90` (`0.6078`) and `B2-4W` (`0.5989`) at $W=800$ (`0.5974`).
   - Across 6,400 logical `deepseek-flash` evaluations (`6,819` HTTP attempts), paired 6-shot censoring reduces OpenStack $F_1$ (`0.4946 -> 0.3876`) while slightly increasing AP (`0.4177 -> 0.4277`), and domain-trained JIT models lead AP (`0.3401–0.5788`).
   - Cached execution of `Survival-PU` takes `14.43 s`/stream on Benchmark B (`7.26 s` cached + `37.32 s` CodeBERT GPU encoding = `44.58 s` total batch processing on Qt), while `ORB` (`2.25 s`) and `BORB` (`3.70 s`) run fastest on CPU.
"""


def build_n05_readme() -> str:
    return """# Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction

Official replication package and automated verification suite for the manuscript submitted to *Software Quality Journal* (Springer Nature).

**Authors:** Zhijie Li¹ · Shaoyang Xu¹ · Ketong Liu¹ · Shibo Meng¹ · Feiran Gao² · Yuli Zhao¹\\*  
¹ Software College, Northeastern University, Shenyang 110169, China  
² School of Computer Science and Technology, Beijing Institute of Technology, Beijing 100081, China  
\\* Corresponding author: `zhaoyl@swc.neu.edu.cn`

---

## Study Overview

Unsupervised drift monitors such as DriftLens detect embedding-space shifts in Just-in-Time Software Defect Prediction (JIT-SDP) without defect labels, motivating immediate window retraining. Yet label-free monitoring eliminates label dependence only when deciding *when* to retrain—not when *supervising* weight updates. Under multi-month SZZ verification latency, most genuinely bug-inducing commits inside newly flagged windows remain unfixed at alarm time ($t_i^{\\mathrm{fix}} > t_{\\mathrm{alarm}}$) and appear clean under strictly causal deployment.

This replication package accompanies our causal supervision audit and cross-paradigm operating-boundary study spanning **37 development streams** and **5 confirmatory ApacheJIT repositories** (**34 natural repository streams** and **8 controlled-shift streams**; **347,122 natural commits**), **29 evaluated methods** (26 deployable + 3 hindsight references across 41 prefix-invariant execution branches), and a **6,400-call paired Code-LLM prompting case study** (`deepseek-flash`).

### Key Headline Findings

- **Severe Alarm-Time Right-Censoring & Hindsight Optimism (RQ1):** Across scheduled 400-commit ($1W$) boundaries in multi-year ecosystems, `68.8%–94.6%` of window defects remain right-censored (`65.8%–95.7%` at fired alarms; `45.1%–80.5%` even at $4W$), whereas the shorter confirmatory cohort exhibits lower censoring (`26.4%` across 5 repositories; `18.0%` on unsplit Standalone-3). Leaking future SZZ fixes into matched window-oracle updates (`B1b*` vs. causal `B2-4W`) significantly inflates whole-stream Average Precision (AP) across $K=25$ independent donor clusters on both MLP (`+0.0237`, `20/0/5`, $p_{\\mathrm{Holm}}=6.63\\times 10^{-5}$) and LightGBM (`+0.0476`, `23/0/2`, $p_{\\mathrm{Holm}}=3.70\\times 10^{-5}$).
- **Cross-Paradigm Operating Boundaries, Granularity, & Confirmatory Non-Replication (RQ2):** No single adaptive paradigm dominates across regimes. Frozen 13-metric logistic regression (`Full-LR Frozen = 0.3653` AP) leads every deployable multimodal/tree host on Benchmark A. On gradual $K=25$ whole-stream AP, `Full-LR Adapt-4W` (`0.5108`, $p_{\\mathrm{Holm}}=0.0639$) leads in mean while `B2-4W` (`0.5004`, $p_{\\mathrm{Holm}}=0.0056$), `Wait-90` (`0.4982`, $p_{\\mathrm{Holm}}=0.0069$), and zero-fit `FixedCurve-PU` (`0.4942`, $p_{\\mathrm{Holm}}=0.0295$) significantly exceed `Survival-PU` (`0.4866`)—though within-window score standardization on identical 5-seed predictions reverses the mean `B2-4W` gap non-significantly (`Rank-Norm: 0.5027 vs. 0.4989`, unadjusted $p=0.0877$). On the short confirmatory cohort, disclosing upstream `apachejit_total.csv`'s `11,964`-commit clean `apache/hadoop` split, MLP hindsight optimism does not replicate on unsplit Standalone-3 (`0.5951 vs. 0.5985`; `ODaSC = 0.7857`) or frozen Umbrella-4 (`0.6801 vs. 0.6858`; `ODaSC = 0.8289`). Conversely, `Survival-PU` leads full post-shift AP (`0.5764`) and LightGBM Stage-3 distillation (`+0.0379` over `Teacher-Only Matched`, $p_{\\mathrm{Holm}}=9.87\\times 10^{-4}$).
- **Unrevealed-Sample Audit & Non-Monotonic Cross-Stage Error Propagation (RQ3):** Across 62 fired DriftLens alarms (`74,400` pool evaluations / `59,600` unique commits), adaptive cohort incidence improves Stage-1 maturity $\\hat{c}(a)$ MAE over `FixedCurve-PU` (`0.0938` vs. `0.1204`) and reduces strictly unrevealed ($y_{\\mathrm{obs}}=0$) soft-target Brier over de-weighted teachers (`0.1303 -> 0.1236`), yet `FixedCurve-PU` attains slightly lower Stage-2 prior MAE (`0.0565` vs. `0.0620`) and unrevealed Brier (`0.1200`).
- **Extreme Latency, Post-Hoc Controls, Block Calibration, & Budget Granularity (RQ4–RQ5):** Under $\\kappa=3.0\\times$ latency, `FixedCurve-PU` leads AP (`0.5547`) and raw Brier/ECE (`0.1693 / 0.0973`), while `Survival-PU` beats `B2-4W`, alarm-synchronized `PostHoc Logit`, Proposition-1-inspired plug-in `PostHoc Ratio`, and `FixedCurve-PU` on scale-fair threshold-tuned $F_{1,\\mathrm{th}}$ (`0.5022` raw; `0.4898` Platt) and avoids `FixedCurve-PU`'s $\\kappa=0\\times$ misspecification (`0.5760 vs. 0.5544` AP). Standard 90-day Platt calibration worsens calibration relative to raw probabilities under `65.9%` calibration-pool censoring. Circular block-bootstrap DriftLens cuts validation-pool null alarms `4.8x` and detects abrupt shifts `3x` faster than periodic `3W` retraining (`350` vs. `1,050` commits; `0/20` missed). Operational per-window 400-commit budgets reverse most steep whole-stream net-TP losses on Benchmark A while remaining host-dependent.

---

## Manuscript-to-Artifact Crosswalk

Every empirical table cell (`1,452` cells across `22` empirical tables), locked configuration field (`33` fields in `V3_CONFIG`), and key prose comparison (`34` exact-path claims + `60` relational/anti-misstatement guards) is verified automatically against the archived JSON artifacts in this repository.

| Manuscript Element | Backing Artifact File(s) in This Repository | Generator / Verifier Script |
| :--- | :--- | :--- |
| **Table 1** (Paradigm Positioning) | Conceptual positioning table in `manuscript/main.tex` | `scripts_v3/build_manuscript_v3.py` |
| **Table 2** (Locked Protocol Parameters) | `src_v3/config_v3.py` (`V3_CONFIG`, 33 locked fields) | `scripts_v3/verify_prose_claims_v3.py` |
| **Table 3** (RQ1 Verification Latency & Censoring Audit) | `results_v3/data_provenance_audit_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 4** (RQ2 Benchmark A Multi-Host & Track-2 Audit) | `results_v3/benchmark_a_multihost_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 5** (RQ2 Benchmark A Provenance/Feature Sensitivity) | `results_v3/benchmark_a_multihost_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 6** (RQ2 Benchmark B $K=25$ / $K=24$ / LightGBM / Shift) | `results_v3/benchmark_b_main_v3.json`, `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 7** (RQ2 Holm–Bonferroni Paired Inference, $K=25$) | `results_v3/benchmark_b_main_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 8** (RQ2 5-Seed Per-Window AP & Scale Removal) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_f_window_vs_whole_ap`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 9** (RQ2 Untouched Confirmatory ApacheJIT Cohort) | `results_v3/confirmatory_cohort_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 10** (RQ3 Maturity Estimators & Component Ablations) | `results_v3/rq3_ablations_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 11** (RQ4a Future-Only Latency Scaling, $\\kappa \\in [0, 3]$) | `results_v3/rq4a_latency_intervention_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 12** (RQ4a Stage-1 Control: `FixedCurve-PU` vs. `Survival-PU`) | `results_v3/rq4a_fixedcurve_control_v3.json` | `scripts_v3/run_rq4a_fixedcurve_control_v3.py` |
| **Table 13** (RQ4a Raw vs. 90d Platt & Post-Hoc $B_2$ Controls) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_d_calibration_and_posthoc`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 14** (RQ4b $2\\times 2$ DriftLens Calibration Factorial) | `results_v3/rq4b_factorial_monitors_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 15** (RQ4c $2\\times 2$ Semi-Synthetic PR-SZZ Perturbation) | `results_v3/rq4d_szz_perturbation_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 16** (RQ5 Phase-Decomposed Wall-Clock Runtime) | `results_v3/manuscript_derived_aggregates_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table 17** (RQ5 Paired Whole-Stream vs. Per-Window Budgets) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_g_paired_net_gain`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table 18** (Synthesis: Observed Patterns vs. Hypotheses) | Synthesized from Tables 3–17 | `scripts_v3/build_manuscript_v3.py` |
| **Table A1** (Executed Host Architectures & Budgets) | `src_v3/models_and_baselines_v3.py` | `scripts_v3/build_manuscript_v3.py` |
| **Table B1** (Confirmatory ApacheJIT Data-Construction Funnel) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_a_apachejit_funnel`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B2** (34-Stream Archive Follow-Up Proxy Audit) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_b_followup_cutoffs`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B3** (Benchmark A $\\ge 180$d / $\\ge 365$d Sensitivity) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_b_followup_cutoffs`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B4** (62-Alarm Unrevealed Stage 1–3 Mechanism Audit) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_c_stage123_mechanisms`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B5** (Per-Stream Breakdown of 4 Controlled Shifts) | `results_v3/reviewer_mechanism_audits_v3.json` (`audit_e_controlled_shifts`) | `scripts_v3/run_reviewer_mechanism_audits_v3.py` |
| **Table B6** (RQ4e Audited Code-LLM Prompting Study) | `results_v3/rq4f_llm_audit_v3.json`, `logs/llm_raw_exchange_v3.jsonl.gz` | `scripts_v3/analyze_and_plot_v3.py` |
| **Table B7** (RQ5 Pre-Declared Qt Tier-1 SZZ Forensics) | `results_v3/rq5_natural_forensics_v3.json` | `scripts_v3/analyze_and_plot_v3.py` |
| **Fig. 1–2** (Causal Timeline & Three-Stage Architecture) | `figures/fig0{a_causal_timeline,b_survival_pu_architecture}_v3.{pdf,png}` | `scripts_v3/generate_conceptual_diagrams_v3.py` |
| **Fig. 3–10** (Empirical RQ1–RQ5 Figures) | `figures/fig{1..8}_*_v3.{pdf,png}` | `scripts_v3/analyze_and_plot_v3.py` |

> **Artifact Naming Conventions:** To preserve exact SHA-256 provenance locks across all immutable `src_v3/` modules and archived `results_v3/*_v3.json` files, internal identifiers retain their frozen execution names: `v3` marks the locked V3 experimental campaign; `n60_*` denotes the five PR-SZZ streams from Bludau & Pretschner (2026); historical JSON filenames `rq4d_szz_perturbation_v3.json`, `rq4e_sensitivity_sweeps_v3.json`, and `rq4f_llm_audit_v3.json` map to manuscript subsections **RQ4c**, **RQ4d**, and **RQ4e**; and `CAUSALJIT_BASE_DIR` is an optional environment variable overriding auto-detected repository root paths in post-processing scripts.

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
│   ├── run_v3_llm_and_forensics.py                 # Audited Code-LLM prompting (RQ4e) & Tier-1 forensics (RQ5) runner
│   ├── verify_mechanisms_and_protocol_v3.py        # Historical pre-unlock 14-check gate (archived SHA-256 766e3fd6...)
│   └── verify_manuscript_v3.py                     # [Tier 1] SHA-256 provenance, .npz, BibTeX & 1,452-cell verifier
├── scripts_v3/                                     # [Portable companion audits, plotting, assembly & verification]
│   ├── run_post_campaign_mechanism_checks_v3.py    # [Tier 1] Verifies pre-unlock lock + runs live Checks 2-14 (41 branches)
│   ├── verify_prose_claims_v3.py                   # [Tier 1] Verifies V3_CONFIG, 34 exact-path claims & 60 relational guards
│   ├── generate_conceptual_diagrams_v3.py          # [Tier 1] Regenerates conceptual vector/PNG diagrams (Fig. 1-2)
│   ├── analyze_and_plot_v3.py                      # [Tier 1] Regenerates Fig. 3-10, 22 empirical tables & 1,452-cell registry
│   ├── build_manuscript_v3.py                      # [Tier 1] Assembles manuscript/main.tex, Markdown summary & README.md
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
│   ├── *_v3.json                                   # All 15 provenance-stamped JSON outputs & 1,452-cell registry
│   ├── prerun_gate_archive/                        # Frozen pre-unlock 14-check gate report (SHA-256 766e3fd6...)
│   ├── commit_predictions/                         # 50 commit-level seed-0 .npz archives (385,620 commits)
│   ├── per_cell_a/ & per_stream_b/                 # Per-cell Benchmark A & per-stream Benchmark B archived summaries
├── logs/
│   ├── llm_raw_exchange_v3.jsonl.gz                # Lossless archive of all 6,400 Code-LLM evaluations (6,819 HTTP calls)
│   └── llm_api_audit_v3.jsonl                      # Per-call latency, token, JSON-compliance & retry audit ledger
├── figures/                                        # Vector PDF & 300-DPI PNG figures (Fig. 1-10)
└── manuscript/                                     # Camera-ready Springer sn-jnl LaTeX source, BibTeX, PDF & Markdown summary
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

# 1. Verify all 15 JSON provenance hashes, 50 .npz archives (385,620 commits), 48 citations & 1,452 table cells
python src_v3/verify_manuscript_v3.py

# 2. Verify all 33 V3_CONFIG table fields, 34 exact-path prose claims, 60 relational guards & adversarial self-tests
python scripts_v3/verify_prose_claims_v3.py

# 3. Verify archived pre-unlock gate SHA-256 (766e3fd6...) and run live mechanism Checks 2-14 (41 prefix-invariant branches)
python scripts_v3/run_post_campaign_mechanism_checks_v3.py

# 4. Regenerate all 10 figures (Fig. 1-10), 22 empirical LaTeX tables, and manuscript/main.tex
python scripts_v3/generate_conceptual_diagrams_v3.py
python scripts_v3/analyze_and_plot_v3.py
python scripts_v3/build_manuscript_v3.py
```

Compiling `manuscript/main.tex` via `pdflatex -> bibtex -> pdflatex -> pdflatex` reproduces the camera-ready 51-page Springer SQJ PDF with `0` Errors, `0` Warnings, and `0` Overfull hboxes.

### Tier 2 — Tabular Benchmark & Companion Audit Re-Execution

Because all 31 Benchmark B `.npz` bundles, all 5 confirmatory `.npz` bundles, all 6 Benchmark A `*_tab.npz` bundles, and all 50 seed-0 commit prediction `.npz` archives are included directly in this repository, you can re-execute the tabular campaigns and companion audits offline without extra downloads:

```bash
# Tabular Benchmark B (K=25), RQ3 ablations, RQ4a-d stress tests & confirmatory cohort
python src_v3/run_v3_experiments.py --phase phase_b
python src_v3/run_v3_experiments.py --phase mechanisms
python src_v3/run_v3_experiments.py --phase confirmatory

# Matched RQ4a FixedCurve-PU Stage-1 control (0.00 parity with rq4a_latency_intervention_v3.json)
python scripts_v3/run_rq4a_fixedcurve_control_v3.py

# Companion Audits B-G (follow-up sensitivity, 62-alarm Stage 1-3, PostHoc Logit/Ratio + clipping, 5-seed WinAP, budgets)
python scripts_v3/run_reviewer_mechanism_audits_v3.py
# Fast PostHoc Ratio clipping/tie-rate diagnostic only (~13 min on CPU/GPU):
python scripts_v3/run_reviewer_mechanism_audits_v3.py --update-ratio-clipping-only
```

> **Note on Audit A (Upstream ApacheJIT Funnel):** Re-running Audit A's raw CSV row counts requires the upstream `apachejit_total.csv` (`data_sync/confirmatory_apachejit_raw/`, included in the Tier 3 Zenodo archive).

### Tier 3 — Full Multimodal Re-Execution & Raw-to-Bundle Reconstruction (Full Zenodo Archive)

Re-executing Benchmark A neural hosts (`Attention-CNN`, `CodeBERT`, `DeepJIT-CNN`), rebuilding `.npz` bundles from raw corpora (`data_sync/`), or re-extracting `768`-D `microsoft/codebert-base` embeddings requires the complete `1.55 GB` offline Zenodo replication archive (too large for GitHub's repository limits; available via the corresponding author `zhaoyl@swc.neu.edu.cn` and scheduled for permanent Zenodo DOI deposit upon acceptance):

```bash
# Optional offline CodeBERT feature extraction (local HF cache of microsoft/codebert-base required)
python src_v3/build_codebert_cache_v3.py
# Rebuild all 89 development + 5 confirmatory bundles from raw data_sync/
python src_v3/data_audit_v3.py --rebuild-all --unlock-confirmatory
# Full development campaign (Phase A + B + mechanisms) & Tier-1 commit forensics
python scripts_v3/run_parallel_campaign_v3.py --workers 8
python src_v3/run_v3_llm_and_forensics.py --mode rq5
```

### Tier 4 — Optional Live Code-LLM Prompting Audit (RQ4e)

Offline inspection and verification of all `6,400` logical `deepseek-flash` evaluations (`6,819` HTTP attempts) in `logs/llm_raw_exchange_v3.jsonl.gz` requires **no** API key or network access. Re-issuing live API calls requires placing a DeepSeek key at `../../.llm_api/.deepseek.key` relative to the repository root and running:

```bash
python src_v3/run_v3_llm_and_forensics.py --mode rq4f
```

### Historical Pre-Unlock Gate Audit Record

Prior to unlocking the five confirmatory ApacheJIT repositories (`HDFS`, `MAPREDUCE`, `SPARK`, `ZEPPELIN`, `ZOOKEEPER`), `src_v3/verify_mechanisms_and_protocol_v3.py` was executed on the locked pre-unlock filesystem (`check_1` enforces `assert current_fs_locked` and therefore intentionally raises `AssertionError` once confirmatory data are present). Its exact SHA-256-sealed output is preserved at `results_v3/prerun_gate_archive/prerun_gate_verification_v3.json` (`766e3fd64c45a3718ad42c86c44f0b75703d139929d1a72dbd44e0019f492307`) and verified automatically in Tier 1.

---

## Citation

If you use this replication package, causal benchmark suite, or verification tools in your research, please cite:

```bibtex
@article{li2026causaljit,
  title   = {Detecting Drift is Not Safe Retraining: A Causal Supervision Audit and Operating Boundaries for Adaptive Just-in-Time Defect Prediction},
  author  = {Li, Zhijie and Xu, Shaoyang and Liu, Ketong and Meng, Shibo and Gao, Feiran and Zhao, Yuli},
  journal = {Software Quality Journal},
  year    = {2026},
  note    = {Under review. Replication package: \\url{https://github.com/789748456v/CausalJIT-Audit-SQJ}}
}
```

Please also credit the upstream datasets and foundational studies integrated into our benchmark:
- Pote et al. (2026), *Software Quality Journal* (unsupervised DriftLens JIT-SDP paradigm)
- McIntosh & Kamei (2018), *IEEE TSE* & Hoang et al. (2019), *MSR* (Qt / OpenStack multimodal JIT corpora)
- Cabral et al. (2019), *ICSE* & Cabral et al. (2023), *EMSE* (`ORB`, `BORB`, and 14 CommitGuru streams)
- Song et al. (2022), *IEEE TSE* (`ODaSC` verification-latency stream learner)
- Keshavarz & Nagappan (2022), *MSR* (`ApacheJIT` corpus)
- Bludau & Pretschner (2026), *Software Quality Journal* (`PR-SZZ` streams and SZZ error rates)

---

## License

- **Code & Scripts (`src_v3/`, `scripts_v3/`):** Licensed under the [MIT License](LICENSE).
- **Structured Results, Manifests, Tables, & Figures (`results_v3/`, `figures/`, `manuscript/`):** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), respecting upstream open-source project licenses (Apache-2.0 / GPL / LGPL / CC-BY-4.0).
"""


def main() -> None:
    tables_tex_path = os.path.join(BUILD_DIR, "v3_tables_generated.tex")
    tmap = load_generated_tables(tables_tex_path)
    assert len(tmap) == 22, f"Expected 22 generated tables, got {len(tmap)}: {list(tmap.keys())}"

    ms_dir = os.path.join(BASE_DIR, "manuscript")
    os.makedirs(ms_dir, exist_ok=True)

    tex_txt = build_main_tex(tmap)
    with open(os.path.join(ms_dir, "main.tex"), "w", encoding="utf-8") as f:
        f.write(tex_txt)

    md_txt = build_markdown_manuscript()
    with open(os.path.join(ms_dir, "N05_SQJ_Manuscript.md"), "w", encoding="utf-8") as f:
        f.write(md_txt)

    rm_txt = build_n05_readme()
    with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(rm_txt)

    print("Assembled manuscript/main.tex, manuscript/N05_SQJ_Manuscript.md, and README.md successfully.")


if __name__ == "__main__":
    main()
