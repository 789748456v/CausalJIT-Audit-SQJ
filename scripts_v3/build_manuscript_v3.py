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
\abstract{\textbf{Context:} Unsupervised monitors like DriftLens flag representation shifts in Just-in-Time Defect Prediction (JIT-SDP) without labels, motivating immediate window retraining.
\textbf{Objective:} Monitoring detects shifts without supervising updates: under multi-month SZZ verification latency, recent bug-inducing commits still appear clean. We audit this causal gap and map boundaries across linear, streaming, and Positive-Unlabeled (PU) learners.
\textbf{Method:} Across 37 development streams and five confirmatory ApacheJIT repositories (34 natural and 8 controlled-shift streams; 347,122 natural commits; paired Code-LLM study), we pair block-bootstrap DriftLens with three-stage \textbf{Survival-PU} as a maturity-correction testbed combining cohort-incidence maturity, clipped prior shrinkage, and PU distillation.
\textbf{Results:} At scheduled 400-commit boundaries, 68.8\%--94.6\% of defects are right-censored, and window-oracle supervision inflates whole-stream Average Precision (AP) over causal retraining across $K=25$ clusters ($+0.0237$ MLP, $p_{\text{Holm}}=6.63\times 10^{-5}$; $+0.0476$ LightGBM, $p_{\text{Holm}}=3.70\times 10^{-5}$). Yet frozen logistic regression leads Benchmark~A ($0.3653$ AP); on gradual $K=25$ whole-stream AP, \texttt{Full-LR Adapt-4W} ($0.5108$) leads in mean while $B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU} ($0.4942$, $p_{\text{Holm}}=0.0295$) significantly exceed \texttt{Survival-PU} ($0.4866$), though within-window standardization reverses the mean $B_2\text{-4W}$ gap (non-significantly). On the confirmatory cohort, MLP hindsight optimism does not replicate (\texttt{ODaSC} leads unsplit Standalone-3 at $0.7857$ and umbrella-4 at $0.8289$). \texttt{Survival-PU} leads full post-shift AP ($0.5764$), avoids $\kappa=0\times$ misspecification, and protects $F_1$ under $3\times$ latency where \texttt{FixedCurve-PU} leads AP and raw Brier.
\textbf{Conclusions:} Chronological splitting permits future-fix leakage; adaptation must match shift regime, latency, calibration, and granularity.}

\keywords{Just-in-Time Defect Prediction, Distribution Shift, Verification Latency, Right-Censoring, Positive-Unlabeled Learning, Survival Analysis, SZZ Evaluation Bias, Large Language Models}

\maketitle

% ============================================================================
% 1. Introduction
% ============================================================================
\section{Introduction}
\label{sec:intro}

Just-in-Time Software Defect Prediction (JIT-SDP) helps quality engineering teams prioritize code review and pre-merge verification at commit time while design rationale remains fresh in developers' working memory \citep{kamei2013large,hoang2019deepjit,pornprasit2021jitline,marrium2026flame}. In long-lived software repositories, however, static commit classifiers degrade as architectures evolve through refactoring, framework migrations, subsystem turnover, and changing team norms \citep{mcintosh2018are,gama2014survey,lu2018learning}. Unlike conventional streaming domains where ground truth arrives moments after prediction, JIT-SDP operates under heavy-tailed \textit{verification latency} \citep{cabral2019class,tabassam2020handling,song2022odasc,song2023humla}: a bug-inducing commit authored at timestamp $t_i$ is labeled defective ($y_i^* = 1$) via SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework,rosa2021evaluating,fan2021impact,bludau2026quantifying} only after the fault manifests and a developer merges a bug-fixing commit at $t_i^{\text{fix}}$. In practice, the resulting fix delay $\Delta_i = (t_i^{\text{fix}} - t_i)/86400$ routinely has a median of three to six months.

To react to representation shifts without waiting months for confirmed defect labels, recent software quality engineering research has turned to \textit{unsupervised embedding-space drift monitors}. Notably, \citet{pote2026adaptive} (\textit{Software Quality Journal}) coupled bimodal commit encoders (Attention-CNN and CodeBERT) with DriftLens \citep{greco2021driftlens,greco2023driftlens}, tracking the multivariate Gaussian Fr\'echet distance \citep{dowson1982frechet} between pre-stream reference embeddings and incoming commit windows $W_{\text{drift}}$. Because DriftLens inspects latent representations without defect labels, it detects shifts within milliseconds and triggers lightweight retraining on the latest window (escalating to a four-window replay upon consecutive alarms). Ordering commits chronologically without random shuffling made retrospective evaluation appear aligned with continuous integration (CI) deployment \citep{pote2026adaptive}.

Transitioning unsupervised drift-triggered retraining from archival datasets to live CI pipelines, however, exposes a fundamental causal supervision gap:

\begin{quote}
\small\textit{Unsupervised drift monitoring eliminates label dependence only from \underline{deciding when to retrain}, not from \underline{supervising how to update model weights}. At the exact wall-clock timestamp $t_{\text{alarm}}$ when DriftLens flags a shifted window $W_{\text{drift}}$, most genuinely bug-inducing commits inside $W_{\text{drift}}$ are only days or weeks old ($t_i^{\text{fix}} > t_{\text{alarm}}$). Under strictly causal repository visibility, their observable labels are right-censored into apparent clean negatives ($y_{i,\text{obs}}(t_{\text{alarm}}) = 0$).}
\end{quote}

Throughout this study, we use ``safe'' or ``supervision-reliable'' retraining specifically to denote avoiding censoring-induced posterior collapse and thresholded-recall degradation rather than claiming formal safety guarantees. Our investigation centers on three linked empirical questions: (1) \textit{what label-visibility timing changes} when future SZZ fixes are blocked alongside commit-time feature hygiene and archive-horizon follow-up checks; (2) \textit{what three-stage maturity correction actually adds} over frozen linear predictors \citep{zeng2021deep}, zero-fit maturity priors, alarm-synchronized post-hoc $B_2\text{-4W}$ probability corrections, and adapted verification-latency stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}; and (3) \textit{what these differences mean under equal-cap commit and LOC-effort review budgets}.

Our matched audit establishes clear, regime-specific boundaries rather than a universal winner. Across $K=25$ multi-year donor clusters, a Pote-inspired offline window-oracle counterfactual ($B_{1b}^*$) significantly inflates whole-stream Average Precision (AP) over matched causal four-window retraining ($B_2\text{-4W}$), while circular block-bootstrap calibration cuts validation-pool resampled-null alarm rates $4.8\times$. Once strict causal visibility is enforced, simpler learners win several key regimes: frozen tabular logistic regression leads deployable Benchmark~A, simpler causal baselines ($B_2\text{-4W}$, $\text{Wait-}90$, and zero-fit \texttt{FixedCurve-PU}) significantly outperform \textbf{Survival-PU} on gradual $K=25$ whole-stream AP (reversing the mean $B_2\text{-4W}$ gap non-significantly under within-window score standardization), and adapted stream learners (\texttt{ODaSC}, \texttt{BORB}, \texttt{ORB}) lead the short confirmatory cohort where MLP hindsight optimism fails to replicate. Conversely, \texttt{Survival-PU} demonstrates targeted, repeatable strengths on full post-shift AP, LightGBM soft-target distillation, and classification protection at \textit{both} latency extremes---avoiding \texttt{FixedCurve-PU}'s misspecification at $\kappa = 0\times$ and outperforming $B_2\text{-4W}$, two post-hoc $B_2\text{-4W}$ controls, and \texttt{FixedCurve-PU} on threshold-tuned $F_{1,\text{th}}$ at $\kappa = 3\times$ (where raw scores also expose the risk of applying a censored 90-day Platt calibrator).

\subsection{Research Questions}
\label{sec:rqs}

We operationalize these three empirical questions through five sequential research questions:

\begin{enumerate}[leftmargin=2.3em,label=\textbf{RQ\arabic*:}]
  \item \textbf{(Causal Audit of Right-Censoring and Hindsight Optimism):} What fraction of true defects inside recent $1W$ ($400$-commit) and $4W$ ($1{,}600$-commit) windows remain right-censored across scheduled boundaries and fired alarms, and how severely does a Pote-inspired offline window-oracle counterfactual ($B_{1b}^*$) diverge from matched causal retraining ($B_2\text{-1W/4W}$) and frozen predictors ($B_0$)?
  \item \textbf{(Cross-Paradigm, Evaluation-Granularity, and Confirmatory Operating Boundaries):} How does \textbf{Survival-PU} compare against Track~1 counterfactuals and Track~2 linear or adapted stream learners across four multimodal/tree hosts, $K=25$ natural donor clusters (whole-stream, per-window, and scale-normalized AP), four controlled concatenated shifts, and five confirmatory ApacheJIT repositories (both unsplit Standalone-3 and frozen umbrella-4)?
  \item \textbf{(Maturity-Estimator Sensitivity, Unrevealed-Sample Mechanism Audit, and Ablations):} How sensitive is Survival-PU to Kaplan--Meier cure, Weibull cure, or zero-fit \texttt{FixedCurve-PU}; how accurately do Stages~1--3 estimate maturity $\hat{c}(a)$, $\hat{\pi}_{\text{target}}$, and soft targets strictly on unrevealed commits ($y_{\text{obs}}=0$) versus a positive-writeback control; and which internal components carry weight?
  \item \textbf{(Stress Tests Across Latency Extremes, Monitoring, Noise, and Prompting):} How do adaptive learners---including alarm-synchronized post-hoc logit-shift and Proposition-1-inspired plug-in ratio $B_2\text{-4W}$ controls under symmetrical Raw vs.\ 90-day Platt evaluation---behave across $\kappa \in [0.0, 3.0]$, a $2 \times 2$ DriftLens factorial, independent semi-synthetic PR-SZZ noise \citep{bludau2026quantifying}, window sweeps ($W \in \{200..800\}$), and a paired Code-LLM prompting case study?
  \item \textbf{(Runtime Cost and Equal-Cap Whole-Stream vs.\ Per-Window Error Analysis):} What cached versus encoding-inclusive runtime overhead does each paradigm incur, and what are the paired rescued-versus-harmed defect trade-offs under fixed thresholding versus equal-cap whole-stream and per-window operational commit/LOC review budgets?
\end{enumerate}

\subsection{Summary of Contributions}
\label{sec:contributions}

Ordered by primary evidentiary support, our four contributions are:

\begin{enumerate}[leftmargin=1.8em]
  \item \textbf{Formal and Empirical Causal Supervision Audit:} Building on drift-triggered JIT-SDP \citep{pote2026adaptive}, delayed-feedback modeling \citep{chapelle2014modeling}, and SZZ audits \citep{song2023waiting,bludau2026quantifying}, we formalize future-fix leakage when $t_i \le t_{\text{alarm}} < t_i^{\text{fix}}$, distinguish uniform-maturity threshold crossing from heterogeneous rank inversion and cross-window AP scale effects, quantify $K=25$ oracle-versus-causal AP inflation, and audit observable archive-horizon follow-up across all 34 natural streams.
  \item \textbf{Cross-Paradigm Operating Boundaries, Confirmatory Data Audit, and Matched-Budget Evaluation:} Across 37 development streams and five confirmatory repositories, we map where frozen linear models, \texttt{FixedCurve-PU}, $B_2\text{-4W}$, \texttt{BORB}, and \texttt{ODaSC} excel; disclose upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split while elevating the unsplit Standalone-3 replication check; isolate cross-window scale effects via 5-seed $K=25$ scale-removal ablations; and quantify paired whole-stream versus per-window matched-budget net defect yields.
  \item \textbf{Structured Maturity-Calibrated PU Testbed, Unrevealed-Sample Mechanism Audits, and Symmetrical Calibration Controls:} We instrument gate-free \textbf{Survival-PU} to test what maturity modeling adds over simpler baselines, documenting its identifiability assumptions, bisection unreachability rates, unrevealed-commit ($y_{\text{obs}}=0$) soft-target errors vs.\ \texttt{FixedCurve-PU} and fair positive writeback, symmetrical Raw vs.\ 90-day Platt calibration against two alarm-synchronized post-hoc $B_2\text{-4W}$ controls at $\kappa \in \{0\times, 1\times, 3\times\}$, and block-bootstrap monitoring trade-offs.
  \item \textbf{Provenance-Audited Replication Suite and Automated Verifiers:} We deliver a SHA-256-locked benchmark and verification package covering 94 stream bundles, 41 prefix-invariant execution branches, 50 commit-level prediction archives, lossless archived prompting logs, and automated cell-, parameter-, and prose-level verifiers.
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

Classical JIT-SDP represents commits with expert change metrics capturing diffusion, size, purpose, history, and developer experience \citep{kamei2013large}. Later work expanded this foundation with post-push code-review attributes \citep{mcintosh2018are}, AST and pull-request features \citep{bludau2022feature,bludau2022prszz}, graph fault localization \citep{zheng2026hgfe}, line-level and mobile predictors \citep{pornprasit2021jitline,marrium2026flame}, gradient-boosted decision trees \citep{ke2017lightgbm}, convolutional diff encoders (DeepJIT) \citep{hoang2019deepjit}, and bimodal code transformers (CodeBERT) \citep{feng2020codebert}. Crucially, \citet{zeng2021deep} demonstrated that single-feature lines-added logistic regression (\texttt{LApredict}) frequently rivals or surpasses deep JIT-SDP models when high-dimensional neural representations dilute commit-size signals, while \citet{tantithamthavorn2017mysterious} underscored the need for leak-free validation protocols. In commit-time CI deployment, post-authoring Gerrit review fields (number of reviewers, review duration, approval counts) do not exist at initial push; our primary multimodal benchmark therefore restricts tabular inputs to the 13 commit-authoring metrics while isolating 28-metric pre-merge triage in a dedicated sensitivity analysis.

To trigger model updates without waiting for defect labels, \citet{pote2026adaptive} monitored penultimate neural representations with DriftLens \citep{greco2021driftlens,greco2023driftlens}, fine-tuning on single ($1W$) or escalated four-window ($4W$) slices whenever the Gaussian Fr\'echet distance \citep{dowson1982frechet} exceeded a uniform bootstrap percentile. Concurrently, \citet{shah2026detecting} observed in passive shift monitoring that representation shifts do not invariably induce accuracy drops. Because our benchmark adds causal tabular linear-residual skips and strict commit-time feature filtering rather than re-running Pote et al.'s original repository verbatim, we treat $B_{1b}^*$ strictly as a \textit{Pote-inspired retrospective-oracle counterfactual} quantifying the isolated effect of future-fix label visibility within matched implementations.

\subsection{Verification Latency, Stream Learners, and SZZ Evaluation Bias}
\label{sec:rel_szz}

SZZ back-tracing \citep{sliwerski2005when,dacosta2017framework} introduces two distinct supervision challenges: permanent mislabeling from tangled or refactoring commits \citep{rosa2021evaluating,fan2021impact,keshavarz2022apachejit,bludau2026quantifying} and transient multi-month verification latency \citep{cabral2019class,tabassam2020handling,song2023humla}. Closely related to evaluation validity, \citet{song2023waiting} studied how imposing a fixed waiting time introduces label noise into \textit{retrospective performance evaluation} of online JIT-SDP; our work is complementary, auditing how future-fix visibility at drift alarm time $t_{\text{alarm}}$ inflates \textit{weight-update supervision} rather than evaluation waiting windows. Within the online stream-learning literature, \citet{cabral2019class} introduced Oversampling Rate Boosting ($\text{ORB}$) over OzaBagging \citep{oza2005online} of Online Hoeffding Trees \citep{domingos2000mining}, waiting $90$ days before treating unfixed commits as clean negatives while scaling Poisson weights via an Oversampling Boost Factor (OBF). \citet{cabral2023borb} later developed Batch-Oversampling ($\text{BORB}$), refreshing base learners every $200$ commits via skewed resampling over historical matured pools, and \citet{song2022odasc} proposed $\text{ODaSC}$, pairing a shorter $15$-day provisional-negative horizon with delayed positive relabeling weighted by DenStream micro-cluster confidences.

\subsection{Delayed Feedback, Positive-Unlabeled Learning, and Maturity Estimation}
\label{sec:rel_pu}

Outside software engineering, \citet{chapelle2014modeling} formalized the core delayed-feedback trade-off in display advertising---young unrevealed events cannot safely be labeled negative, yet waiting for full maturation makes models stale---via joint parametric modeling of feature-dependent positive probability $p(x)$ and exponential delay rate $\lambda(x)$. Learning from confirmed positives and unlabeled mixtures has also been studied through propensity-weighted PU estimators \citep{elkan2008learning,bekker2020learning} and non-negative PU risk ($\text{nnPU}$) \citep{kiryo2017positive}, alongside covariate-shift density ratios \citep{sugiyama2008direct} and kernel two-sample tests \citep{gretton2012kernel}. Standard $\text{nnPU}$, however, assumes a single marginal prior across unlabeled commits regardless of age. Accordingly, we do \textit{not} claim to introduce the general concept of combining survival/delay modeling with PU learning; rather, our methodological increment is adapting delayed-feedback maturity estimation to drift-triggered, heterogeneous-host JIT-SDP---evaluating non-parametric risk-eligible cohort incidence \citep{lyndenbell1971method}, quantized Kaplan--Meier cure \citep{kaplan1958nonparametric}, and a global unstratified Weibull mixture-cure MLE \citep{boag1949cure,farewell1982cure} (a delay family containing the exponential law at shape $k=1$, inspired by delayed-feedback/cure likelihoods rather than jointly fitting feature-dependent $p(x)$ and $\lambda(x)$ as in \citet{chapelle2014modeling}) against zero-fit \texttt{FixedCurve-PU} and two alarm-synchronized post-hoc $B_2\text{-4W}$ controls.

% ============================================================================
% 3. Theoretical Audit of Censoring Bias & Alarm Calibration
% ============================================================================
\section{Theoretical Analysis of Right-Censored Retraining and Block Calibration}
\label{sec:theory}

Before introducing our adaptation protocol and empirical design, we formalize two mechanisms governing live drift-triggered JIT-SDP: (1) how right-censored labels reshape the Bayes-optimal predictor under uniform versus heterogeneous commit maturity, and (2) how positive lag autocovariance inflates contiguous window-mean variance relative to independent draws (Proposition~\ref{prop:autocorr}), motivating our empirical alarm-calibration factorial.

\subsection{Posterior Compression and Rank Inversion Under Right-Censored Supervision}
\label{sec:theory_poison}

Consider a chronologically ordered commit stream $\mathcal{S} = \{(x_i, t_i, y_i^*, t_i^{\text{fix}})\}_{i=1}^N$ ($t_1 \le \dots \le t_N$, timestamps in seconds) observed up to an \textit{observable archive-horizon proxy} $T_{\text{end}} = \max(\max_i t_i, \max_{i:\, t_i^{\text{fix}} < \infty} t_i^{\text{fix}})$ (using finite fix timestamps only, since upstream datasets do not carry a separate annotation-completeness timestamp), where $x_i \in \mathcal{X}$ denotes commit-authoring features, $y_i^* \in \{0, 1\}$ is the \textit{archive-horizon defect label}, $t_i^{\text{fix}} \ge t_i$ is the earliest SZZ bug-fixing timestamp ($t_i^{\text{fix}} = +\infty$ if $y_i^* = 0$ or unlinked), and $\Delta_i = (t_i^{\text{fix}} - t_i)/86400 \in [0, +\infty]$ is the fix delay in days. Distinguishing \textit{alarm-time right-censoring} ($t_{\text{alarm}} < t_i^{\text{fix}} \le T_{\text{end}}$) from \textit{end-of-archive censoring} ($t_i^{\text{fix}} > T_{\text{end}}$) is critical; we audit remaining follow-up $T_{\text{end}} - t_i$ across all 34 natural repository streams and tabulate full $\ge 180\text{d}/\ge 365\text{d}$ sensitivity in Section~\ref{sec:benchmarks} and Appendix~\ref{app:audits}. Figure~\ref{fig:causal_timeline} contrasts offline retrospective evaluation ($B_{1b}^*$) against strictly causal CI deployment ($t \le t_{\text{alarm}}$).

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

\textit{Remark (Scope of Proposition~\ref{prop:bias} and Within-Window vs.\ Whole-Stream AP).} Right-censoring does not universally worsen raw Brier error when $\omega_+ > 1$ inflates uncensored posteriors (e.g., at $\eta^* = 0.5, \omega_+ = 4$, $\hat{p}_{B_1^*} = 0.80$ vs.\ $\hat{p}_{B_2} = 0.50$ at $\bar{c}_W = 0.40$). Equally important, Claim~1's monotone ranking preservation holds \textit{within a single window's model}; when predictions across post-alarm windows are concatenated into a single \textit{whole-stream} Precision--Recall curve, inter-window score-scale shifts (such as re-anchoring output scales as estimated priors $\hat{\pi}_{\text{target}}(t)$ or class weights vary) can penalize whole-stream AP even when within-window ranking holds or improves---a cross-window comparability diagnostic we evaluate on $K=25$ in Section~\ref{sec:rq2}. Thus, Proposition~\ref{prop:bias} explains why $B_2\text{-4W}$ can retain competitive whole-stream AP on gradual streams while losing thresholded recall under heavy latency ($\kappa \to 3\times$) and ranking accuracy when shifted features appear at young ages ($c_{\text{lo}} \ll c_{\text{hi}}$).

\subsection{Population Variance of Contiguous vs.\ Independent Draws Under Autocorrelation}
\label{sec:theory_alarm_storm}

Because every triggered alarm exposes the model to right-censored supervision, calibrating the drift monitor to avoid spurious alarms is tightly coupled to update safety. Software commits arrive in serially correlated development bursts (feature sprints, subsystem refactorings, and release freezes), violating the i.i.d.\ sampling assumption of standard uniform bootstrap calibration.

\begin{proposition}[Population Window-Mean Covariance Under Positive Lag Autocovariance]
\label{prop:autocorr}
Let $\{z_t\}_{t \in \mathbb{Z}} \subset \mathbb{R}^d$ be a weakly stationary embedding sequence with mean $\mu_0$, marginal covariance $\Gamma_0 = \operatorname{Cov}(z_t)$, and lag-$k$ autocovariance $\Gamma_k = \operatorname{Cov}(z_t, z_{t+k})$ satisfying $\Delta_\Gamma \triangleq \sum_{k=1}^{W-1}\frac{W-k}{W}(\Gamma_k + \Gamma_k^\top) \succ 0$. Comparing a contiguous window mean $\bar{z}_{\text{contig}} = \frac{1}{W}\sum_{j=1}^W z_{t_0+j}$ against the mean $\bar{z}_{\text{ind}} = \frac{1}{W}\sum_{j=1}^W \tilde{z}_j$ of $W$ independent draws $\tilde{z}_j$ from the same marginal law, $\operatorname{Cov}(\bar{z}_{\text{contig}}) = \operatorname{Cov}(\bar{z}_{\text{ind}}) + \frac{1}{W}\Delta_\Gamma \succ \operatorname{Cov}(\bar{z}_{\text{ind}})$ and $\mathbb{E}\|\bar{z}_{\text{contig}} - \mu_0\|_2^2 > \mathbb{E}\|\bar{z}_{\text{ind}} - \mu_0\|_2^2$.
\end{proposition}

\begin{proof}
Expanding $\operatorname{Cov}(\frac{1}{W}\sum_{j=1}^W z_{t_0+j}) = \frac{1}{W}\Gamma_0 + \frac{1}{W^2}\sum_{k=1}^{W-1}(W-k)(\Gamma_k + \Gamma_k^\top)$ whereas $\operatorname{Cov}(\bar{z}_{\text{ind}}) = \frac{1}{W}\Gamma_0$ gives difference $\frac{1}{W}\Delta_\Gamma \succ 0$ and trace gap $\frac{1}{W}\operatorname{tr}(\Delta_\Gamma) > 0$.
\end{proof}

In finite calibration pools of size $N_{\text{cal}}$, uniform bootstrap resampling replaces $\Gamma_0$ with the empirical pool covariance $\hat{\Sigma}_{\text{pool}}$ while breaking lag-$k$ sprint serial correlation ($k \ge 1$) \citep{kunsch1989jackknife,politis1992circular}. Although Proposition~\ref{prop:autocorr} characterizes the mean-shift component rather than the full nonlinear Gaussian Fr\'echet statistic, it motivates calibrating DriftLens via circular contiguous blocks ($b = \max(\min(\lfloor W/6 \rfloor, 80), 25)$). To separate bootstrap sampling from post-alarm updating without confounding percentile or cooldown settings, our RQ4b experiment evaluates a clean $2 \times 2$ factorial at fixed $(d_{\text{pca}}=24, p_{95}, \text{cooldown}=1)$ crossing $\{\text{IID}, \text{Circular Block}\}$ with $\{\text{Static}, \text{Rolling Ref+Th}\}$, where Rolling Ref+Th refreshes reference statistics on the trailing $3W$ commits upon alarms and blends the recalibrated $p_{95}$ threshold ($0.70\,\tau_{\text{new}} + 0.30\,\tau_{\text{old}}$), alongside Pote et al.'s static $(p_{98}, \text{cooldown}=0)$ reference.

% ============================================================================
% 4. Methodology: Three-Stage Survival-PU Adaptation Protocol
% ============================================================================
\section{Methodology: Three-Stage Survival-PU Adaptation Protocol}
\label{sec:method}

Proposition~\ref{prop:bias} indicates that safe real-time adaptation at alarm timestamp $t_{\text{alarm}}$ requires resolving three coupled obstacles using only past repository state ($t_i \le t_{\text{alarm}}$): (1) estimating commit-age reveal maturity $c(a_i)$ without conditioning naively on fast-revealing fixes; (2) estimating target defect prevalence $\hat{\pi}_{\text{target}}$ and a reliable teacher ranking without exploding variance on young commits; and (3) training on the full adaptation pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ with soft targets that preserve fresh drift signals while preventing posterior collapse.

Figure~\ref{fig:architecture} summarizes the resulting gate-free three-stage \textbf{Survival-PU} pipeline, and Table~\ref{tab:hyperparams} consolidates every locked protocol parameter frozen prior to unlocking the confirmatory repositories (exact code symbol mappings appear in Appendix~\ref{app:repro}). Preliminary heuristic three-state safety gates are excluded from the core method and isolated strictly as an ablation control (\texttt{Survival-PU (+Legacy Gate)}).

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

A naive empirical CDF over observed bug fixes ($y_{i,\text{obs}} = 1$) severely underestimates fix latency because slow-revealing bugs are systematically absent from recent commits. To avoid this right-truncation bias, Stage~1 slices the trailing $N_{\text{hist}}$ causal commits $\mathcal{H}_{\text{rec}} \subset \{i : t_i \le t_{\text{alarm}}\}$ \textit{before} computing age quantiles, recording commit age $a_i = \max((t_{\text{alarm}} - t_i)/86400, 0)$ and observed fix delay $d_i = \max((t_i^{\text{fix}} - t_i)/86400, 0)$ solely when $y_{i,\text{obs}}(t_{\text{alarm}}) = 1$ (with $d_i = +\infty$ otherwise, quarantining any retroactive $t_i^{\text{fix}} < t_i$ to $+\infty$). On candidate age knots $g_0 = 0 < g_1 < \dots < g_M = a_{\max}$ up to the support quantile $a_{\max} = Q_{q_{\max}}(\{a_i\}_{i \in \mathcal{H}_{\text{rec}}})$, with effective risk-set floor $n_{\text{eff}}^{\text{risk}} = \max(\min(n_{\min}^{\text{risk}}, \lfloor (1 - q_{\max})|\mathcal{H}_{\text{rec}}| \rfloor), 25)$, the \textbf{Risk-Eligible Cohort Cumulative Incidence Estimator} restricts each knot's denominator to commits old enough to be observed at lag $g_m$, $\mathcal{E}(g_m) = \{i \in \mathcal{H}_{\text{rec}} : a_i \ge g_m\}$ (carrying forward $\hat{G}(g_{m-1})$ when $|\mathcal{E}(g_m)| < n_{\text{eff}}^{\text{risk}}$ for $m > 0$), and enforces monotonicity via its running maximum:
\begin{equation}
  \hat{G}(g_m) = \frac{1}{|\mathcal{E}(g_m)|}\sum_{i \in \mathcal{E}(g_m)} \mathbb{I}\!\left(y_{i,\text{obs}} = 1 \;\wedge\; d_i \le g_m\right), \qquad \tilde{G}(g_m) = \max_{0 \le k \le m}\hat{G}(g_k).
  \label{eq:cohort_inc}
\end{equation}

Because $\tilde{G}(a_{\max})$ still reflects finite-horizon observation up to $a_{\max}$, Stage~1 rescales the normalized curve by a bounded tail-completeness factor $\gamma_{\text{tail}} \in [\gamma_{\min}, \gamma_{\max}]$. Letting $\tilde{G}_{\text{half}} = \operatorname{interp}(0.5\,a_{\max}; \{g_m\}, \{\tilde{G}(g_m)\})$, the growth ratio is $r_g = \tilde{G}(a_{\max}) / \tilde{G}_{\text{half}}$ when $\tilde{G}_{\text{half}} > 10^{-5}$ and $r_g = 1.5$ otherwise; combining this with age factor $\phi_a = \operatorname{clip}(a_{\max}/(a_{\max}+45),\, 0.65,\, 1.0)$ yields $\gamma_{\text{tail}} = \operatorname{clip}([\gamma_{\max} - 0.28\operatorname{clip}(r_g - 1, 0, 1)]\,\phi_a,\, \gamma_{\min},\, \gamma_{\max})$. Knot maturities are clipped \textit{prior} to piecewise-linear interpolation across knots:
\begin{equation}
  \begin{aligned}
    c_{\text{knot}}(g_m) &= \operatorname{clip}\!\Big(\gamma_{\text{tail}}\,\frac{\tilde{G}(g_m)}{\tilde{G}(a_{\max})},\; c_{\min},\; c_{\max}\Big), \\
    c_i &= \operatorname{clip}\!\big(\operatorname{interp}(a_i;\, \{g_m\}, \{c_{\text{knot}}(g_m)\}),\; c_{\min},\; c_{\max}\big),
  \end{aligned}
  \label{eq:ci_interp}
\end{equation}
with left boundary $c_{\min}$ and right boundary $c_{\text{knot}}(g_M)$. Whenever historical support is insufficient (observed fixes $< n_{\min}^{\text{fix}}$, $|\mathcal{H}_{\text{rec}}| < 200$, $a_{\max} < 30\text{d}$, or $\tilde{G}(a_{\max}) < 10^{-4}$), the estimator activates a deterministic fallback that evaluates $0.88(1 - 2^{-g/90})$ on $145$ uniform knots spanning $[0, 720]$ days ($5\text{-day}$ spacing), clips those knot values to $[c_{\min}, c_{\max}]$, and piecewise-linearly interpolates at $a_i$ (holding the terminal $720\text{-day}$ knot value for $a_i > 720\text{d}$).

To test whether learned survival curves improve upon classical survival estimators or a zero-fit parametric prior, RQ2 and RQ3 compare three alternatives under the identical Stage~1 interface: (i) right-censored \textbf{Kaplan--Meier Cure}, quantizing event and censoring times onto a shared $10^{-4}$-day ($\approx 8.6\text{ s}$) grid so the risk-set invariant $0 \le e_u \le n_u$ holds identically; (ii) L-BFGS-B \textbf{Weibull Mixture-Cure MLE} \citep{boag1949cure,farewell1982cure} fitted on unstratified $\mathcal{H}_{\text{rec}}$ (a global Weibull delay family containing the exponential special case at shape $k=1$, inspired by delayed-feedback/cure likelihoods rather than jointly fitting feature-dependent $p(x)$ and $\lambda(x)$ as in \citet{chapelle2014modeling}); and (iii) zero-fit \textbf{FixedCurve-PU}, setting $c_{\text{fix}}(a_i) = \operatorname{clip}(1 - 2^{-a_i/90},\, c_{\min},\, c_{\max})$.

\textit{Identifiability Assumptions of Stage~1.} Because the denominator of Eq.~\eqref{eq:cohort_inc} includes all age-eligible commits, $\hat{G}(a)$ estimates the joint probability $\pi F_\Delta(a)$ rather than conditional maturity $c(a) = F_\Delta(a)$ directly. Canceling $\pi$ in Eq.~\eqref{eq:ci_interp} therefore relies on two structural assumptions: (I1) trailing-cohort defect prevalence $\pi$ and delay distribution $F_\Delta$ are locally stationary across commit ages inside $\mathcal{H}_{\text{rec}}$, and (I2) $\gamma_{\text{tail}}$ approximates terminal completeness at $a_{\max}$. Under abrupt prevalence shifts across cohorts, $\hat{G}(a)$ can confound lower defect prevalence with slower reveal---an effect we audit directly against post hoc empirical $c^*(a)$ across 62 fired alarms in Section~\ref{sec:rq3} and Table~\ref{tab:mechanism_audit}.

\subsection{Stage 2: Clipped Inverse-Maturity Prevalence on \texorpdfstring{$\mathcal{B}_{\text{cal}}$}{B\_cal} and Reliable Teacher}
\label{sec:stage2}

At each alarm, we construct the adaptation pool $\mathcal{B} = \mathcal{A}_{\text{anc}} \cup W_{\text{drift}}$ by pairing the trailing $N_{\text{anc}}$ non-drift commits $\mathcal{A}_{\text{anc}}$ with the newly flagged window $W_{\text{drift}}$. To account for covariate shift between historical anchors and the shifted window, an $L_2$-regularized domain logistic classifier discriminates $\mathcal{A}_{\text{anc}}$ from $W_{\text{drift}}$ to yield bounded importance weights $w_{\text{imp}}(x_i) \in [w_{\min}, w_{\max}]$ \citep{sugiyama2008direct}. Multiplying $w_{\text{imp}}(x_i)$ on $\mathcal{A}_{\text{anc}}$ by linear recency in $[0.70, 1.15]$ while assigning uniform emphasis $w_i = \gamma_{\text{drift}}$ on $W_{\text{drift}}$ yields base weights $w_i$ and observation-independent sample weights $u_i = w_i[u_0 + (1 - u_0)c_i]$.

Directly dividing observed labels $y_{i,\text{obs}}$ by $c_i$ to estimate target prevalence incurs extreme variance on very young commits ($c_i \approx 0.05$). Stage~2 therefore operates on the trailing calibration subset $\mathcal{B}_{\text{cal}} \subset \mathcal{B}$ of size $N_{\text{cal}} = \min(|\mathcal{B}|, \max(3|W_{\text{drift}}|, 900))$ using an inverse-maturity floor $\tilde{c}_i = \max(c_i, c_{\text{ht}})$ ($c_{\text{ht}} = 0.10$), combining a clipped inverse-maturity plug-in estimator with heuristic variance-inflation shrinkage toward a mature anchor prior $\pi_{\text{anc}}$:
\begin{equation}
  \begin{aligned}
    \hat{\pi}_{\text{IP}} &= \frac{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\,(y_{i,\text{obs}} / \tilde{c}_i)}{\sum_{i \in \mathcal{B}_{\text{cal}}} u_i}, \qquad n_{\text{eff,heur}} = \frac{\big(\sum_{i \in \mathcal{B}_{\text{cal}}} u_i\big)^2}{\sum_{i \in \mathcal{B}_{\text{cal}}} (u_i / \tilde{c}_i)^2}, \\
    \hat{\pi}_{\text{target}} &= \operatorname{clip}\!\left(\max\!\left(\frac{n_{\text{eff,heur}}\,\hat{\pi}_{\text{IP}} + m_0\,\pi_{\text{anc}}}{n_{\text{eff,heur}} + m_0},\; \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4}\right),\; \pi_{\min},\; \pi_{\max}\right).
  \end{aligned}
  \label{eq:ht_prev}
\end{equation}
Here $\pi_{\text{anc}}$ is the inverse-maturity mean over mature history $\{j : c_j \ge 0.55\}$ when at least $80$ commits qualify (otherwise the initial pre-stream anchor prevalence), $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} = \sum_{i \in \mathcal{B}_{\text{cal}}} u_i y_{i,\text{obs}} / \sum_{i \in \mathcal{B}_{\text{cal}}} u_i$, and commits with $c_i < c_{\text{ht}}$ incur conservative downward attenuation $c_i / c_{\text{ht}}$. Note a concrete mathematical boundary condition in Eq.~\eqref{eq:ht_prev}: because the outer clamp $[\pi_{\min}, \pi_{\max}] = [0.02, 0.70]$ wraps $\max(\cdot, \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} + 10^{-4})$, whenever $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} > 0.70$, the target is capped at $0.70 < \bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ and becomes algebraically unreachable in Stage~3 (because Eq.~\eqref{eq:bayes_pu} sets $q_i = 1$ on revealed positives and $q_i \ge 0$ elsewhere). We audit the exact empirical frequency of this unreachable regime across all experiments in Section~\ref{sec:stage3}.

In parallel, Stage~2 trains a teacher classifier $T_\phi$ on the trailing $N_{\text{rel}}$ commits of the selective reliable cohort $\mathcal{R} = \{i : y_{i,\text{obs}} = 1 \vee c_i \ge \tau_{\text{rel}}\}$ (when $|\mathcal{R}| \ge 100$) so that unfixed commits with low maturity do not dominate teacher feature weights. Teacher samples carry weight $w_{\text{rel},i} = w_{\text{imp}}(x_i)\cdot r_{\text{conf},i}\cdot \text{recency}_i$, where $r_{\text{conf},i} = 0.55 + 0.45 c_i$ if $y_{i,\text{obs}} = 1 \wedge c_i < \tau_{\text{rel}}$ (else $1$) and $\text{recency}_i \in [0.75, 1.15]$, with positive class weight $\omega_{+,\text{teach}} = \frac{1}{2}\omega_+ + \frac{1}{2}\operatorname{clip}((1-\hat{\pi}_{\text{target}})/\max(\hat{\pi}_{\text{target}}, 0.03), 1.5, \max(4.5, \omega_+))$. Host initialization follows each model family's native paradigm: on \textbf{neural hosts}, $T_\phi$ initializes from an $85\%/15\%$ parameter blend of the current checkpoint and the initial pre-stream anchor and trains for $E_{\text{teach}}$ epochs; on \textbf{LightGBM}, the teacher fits a fresh $R_{\text{alm}}$-tree booster without inheriting prior trees or margins. Either host emits class-weight-adjusted base log-odds $\eta_i^{(0)} = \tau_{\text{sharp}}(z_{\text{teach}}(x_i) - \log \omega_{+,\text{teach}})$ across $\mathcal{B}$. Crucially, subtracting $\log \omega_{+,\text{teach}}$ removes explicit class reweighting but does \textit{not} eliminate selective-cohort inclusion bias on young commits or residual censoring among commits with $c_i \ge 0.45$; accordingly, Stage~3 treats $q_i(\delta^*)$ strictly as \textit{maturity- and prior-adjusted soft targets} rather than exact true posteriors (audited by commit age in Section~\ref{sec:rq3}).

\subsection{Stage 3: Maturity- and Prior-Adjusted Soft Targets, Linear-Residual Skip, and Callable LightGBM}
\label{sec:stage3}

To transfer knowledge from matured history onto the entire adaptation pool $\mathcal{B}$---including fresh commits in $W_{\text{drift}}$---Stage~3 constructs maturity- and prior-adjusted soft targets via a single shared mapping $(\eta_i^{(0)}, \delta, y_{i,\text{obs}}, c_i)$ setting $q_i(\delta) = 1$ whenever $y_{i,\text{obs}} = 1$ and, for unrevealed commits ($y_{i,\text{obs}} = 0$),
\begin{equation}
  q_i(\delta) = \frac{\tilde{p}_i(\delta)\,(1 - c_i)}{1 - \tilde{p}_i(\delta)\,c_i}, \qquad \tilde{p}_i(\delta) = \operatorname{clip}\!\big(\sigma(\eta_i^{(0)} + \delta),\; p_{\min},\; p_{\max}\big).
  \label{eq:bayes_pu}
\end{equation}
Conditional on equal maturity $c_i$ among unrevealed commits ($y_{i,\text{obs}} = 0$), $q_i(\delta)$ is nondecreasing in $\eta_i^{(0)}$ and strictly increasing on the unclipped interior $\sigma(\eta_i^{(0)} + \delta) \in (p_{\min}, p_{\max})$; across heterogeneous ages, it downweights aged unfixed commits ($q_i \to 0$ as $c_i \to 1$) relative to fresh unfixed commits ($q_i \to \tilde{p}_i$ as $c_i \to 0$). Letting $f(\delta) = \sum_{i \in \mathcal{B}_{\text{cal}}} u_i q_i(\delta) / \sum_{i \in \mathcal{B}_{\text{cal}}} u_i$, monotone bisection ($K_{\text{bis}}$ iterations on $[-8, 8]$) targets $f(\delta^*) = \hat{\pi}_{\text{target}}$ on $\mathcal{B}_{\text{cal}}$ whenever $\hat{\pi}_{\text{target}} \in [f(-8), f(+8)]$. When $\hat{\pi}_{\text{target}}$ falls outside this reachable interval---determined by the alarm-time weighted observed positive rate $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ and teacher logit saturation inside $\mathcal{B}_{\text{cal}}$ (e.g., if $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} > \pi_{\max} = 0.70$)---the target is not matched exactly; instead, the solver returns the nearest boundary $\delta^* \in \{-8, +8\}$, achieving endpoint mean $f(\delta^*)$ with residual $|f(\delta^*) - \hat{\pi}_{\text{target}}|$, and logs an unreachable solve. Auditing all runs with total-alarm denominators confirms $0$ unreachable solves on Benchmark~A and RQ4a--e; across all 27 Benchmark~B natural streams, $1.0 / 161.7$ alarms ($0.62\%$) were unreachable, concentrated in \texttt{apache\_flink} ($0.1 / 6.5 = 1.5\%$) and \texttt{apache\_groovy} ($0.9 / 5.4 = 16.7\%$, coinciding with $0.9$ survival fallbacks); on Confirmatory \texttt{HDFS*}, high alarm-time $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ triggered unreachable clamping on $0.4 / 1.4 = 28.6\%$ of MLP alarms and $0.6 / 2.0 = 30.0\%$ of LightGBM alarms (illustrating that production implementations should raise $\pi_{\max}$ above $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}}$ or project $\hat{\pi}_{\text{target}}$ onto $[f(-8), f(+8)]$). Float32 soft targets $q_i(\delta^*)$ then train the student across $\mathcal{B}$ under $\mathcal{L}_{\text{SPU}}(z) = \sum_{i \in \mathcal{B}} u_i [\omega_{+,\text{teach}} q_i(\delta^*) \log(1 + e^{-z_i}) + (1 - q_i(\delta^*))\log(1 + e^{z_i})]$, initializing neural students from $T_\phi$ ($E_{\text{stud}}$ epochs) and fitting fresh $R_{\text{alm}}$-tree LightGBM boosters (matched in total trees $R_{\text{alm}} + R_{\text{alm}}$ against single-stage $2R_{\text{alm}}$-tree \texttt{Teacher-Only (Matched)}).

Two host-level mechanisms ensure consistent execution across multimodal neural encoders and gradient-boosted trees:
\begin{enumerate}[leftmargin=1.8em,itemsep=0.10em]
  \item \textbf{Causal Tabular Linear-Residual Skip Connection (Multimodal Neural Backbones):} Because high-dimensional diff and message encoders can dilute commit-size predictors (\texttt{la}, \texttt{nf}, \texttt{ent}) \citep{zeng2021deep}, neural hosts support $z(x) = w_{\text{LR}}^\top x_{\text{tab}} + b_{\text{LR}} + f_\theta(x)$, where $(w_{\text{LR}}, b_{\text{LR}})$ are fitted via $L_2$-regularized logistic regression on $[0, n_{\text{init}}+n_{\text{val}})$ using $y_{\text{obs}}(t_{\text{start}})$ (while scalers and token vocabularies fit strictly on $[0, n_{\text{init}})$) and $f_\theta$'s output head is zero-initialized. Pre-stream validation selected a frozen linear skip over a trainable skip ($0.0586$ vs.\ $0.0575$ validation AP).
  \item \textbf{Analytical Callable Objective for Gradient-Boosted Trees \citep{friedman2001greedy,ke2017lightgbm}:} Rather than duplicating rows $2|\mathcal{B}|$ times, our LightGBM host supplies exact continuous-target first and second derivatives ($g_i, h_i$) with initial margin $z_{\text{init}} = \log(\bar{p}_0 / (1 - \bar{p}_0))$ propagated identically to inference (derivation and verification details in Appendix~\ref{app:repro}, Proposition~\ref{prop:lgbm}).
\end{enumerate}

% ============================================================================
% 5. Experimental Design & Statistical Protocol
% ============================================================================
\section{Experimental Design and Evaluation Protocol}
\label{sec:setup}

To separate development analysis from out-of-sample replication, we structured benchmark evaluation into three tiers, froze all modeling and statistical code prior to unlocking an initially untouched confirmatory cohort, and verified pipeline prefix invariance across every method execution branch. Throughout this study, we distinguish: (i) \textit{protocol-specified development-cohort analyses} on Benchmark~A and Benchmark~B ($K=25$, whose comparison families and decision rules were codified in our codebase prior to confirmatory unlock rather than externally timestamp-preregistered before all development exploration); (ii) \textit{pre-unlock frozen confirmatory evaluation} on five initially absent ApacheJIT repositories; and (iii) one \textit{post-unlock split-sizing amendment} for small streams ($N < 4{,}000$), disclosed explicitly below.

\subsection{Provenance-Audited Stream Tiers, Data-Construction Funnels, and Archive Follow-Up}
\label{sec:benchmarks}

Our evaluation spans \textbf{37 development streams} (6 Benchmark~A multimodal streams, 27 natural Benchmark~B streams, and 4 Benchmark~B controlled concatenated-shift streams; 89 archived stream bundles total) plus \textbf{5 confirmatory ApacheJIT repositories}, totaling 347,122 natural repository commit records alongside a paired Code-LLM prompting case study:

\begin{enumerate}[leftmargin=1.8em,itemsep=0.14em]
  \item \textbf{Benchmark A — Multimodal Qt (25,704 Commits) and OpenStack (13,304 Commits; 6 Streams):} Cross-linking DeepJIT corpora \citep{hoang2019deepjit}, the tables of \citet{mcintosh2018are}, and 16 official Git repositories, we verified Tier~1+2 cryptographic SZZ bug-fixing timestamps for \textbf{96.49\%} ($1{,}761/1{,}825$; median lag $163.1\text{d}$) of Qt defects and \textbf{99.32\%} ($1{,}616/1{,}627$; median $178.7\text{d}$) of OpenStack defects, quarantining heuristic Tier-3 matches and unlinked defects to $t_{\text{fix}} = +\infty$. Primary streams retain \textbf{only the 13 commit-authoring metrics} (\texttt{la}, \texttt{ld}, \texttt{nf}, \texttt{nd}, \texttt{ns}, \texttt{ent}, \texttt{ndev}, \texttt{age}, \texttt{nuc}, \texttt{aexp}, \texttt{arexp}, \texttt{asexp}, \texttt{asawr}), removing 15 post-authoring Gerrit review fields (isolated in pre-merge sensitivity streams). \textit{Archival merged-commit caveat:} raw Qt/OpenStack tables provide only \texttt{author\_date} without Gerrit patch-set-1 upload or merge timestamps; thus Benchmark~A models \textit{archival merged-commit replay}, not initial unreviewed patch-set-1 pushes. Empty diffs (Qt: 262 commits, $1.02\%$; OpenStack: 10 commits, $0.08\%$) are encoded strictly as empty strings without message substitution. Across two natural ecosystems and four controlled subsystem shifts, we evaluate \textbf{Attention-CNN} \citep{pote2026adaptive}, \textbf{CodeBERT} \citep{feng2020codebert}, \textbf{DeepJIT-CNN} \citep{hoang2019deepjit}, and \textbf{LightGBM} \citep{ke2017lightgbm} (5 seeds) plus all 11 Track~2 baselines.
  \item \textbf{Benchmark B — 27 Natural Streams ($K=25$ Independent Donor Clusters, 300,637 Commits) plus 4 Controlled Concatenated Shifts:} Combines 14 CommitGuru streams from \citet{cabral2019class}, eight ApacheJIT development streams \citep{keshavarz2022apachejit}, and five PR-SZZ streams from \citet{bludau2026quantifying} (ordered by Git committer date after removing 13 post-closure/target columns and mapping post-2021H1 sentinels to $+\infty$). Merging Apache Camel (shared by CommitGuru and ApacheJIT) and OpenStack Nova/Neutron maps the 27 natural streams onto \textbf{$K=25$ independent donor clusters} (and Benchmark-A-disjoint \textbf{$K=24$}). Four \textit{controlled concatenated distribution-shift streams} (Nova$\to$Zulip, Ignite$\to$Cassandra, Airflow$\to$Pulsar, and OpenStack Nova$\to$Neutron) are kept strictly outside $K=25$ natural inference (10 MLP seeds, all Track~2 learners, 5 LightGBM seeds). \textit{Exact splice timeline construction:} concatenating a pre-shift prefix and post-shift suffix shifts all post-onset commit timestamps and finite fix timestamps by $\Delta t = t_{\text{pre,last}} - t_{\text{post,first}} + 3{,}600\text{ s}$ (a fixed 1-hour boundary gap; $+\infty$ remains $+\infty$), while $23.7\%\text{--}60.0\%$ of revealed pre-onset fixes continue arriving after the onset boundary (Appendix Table~\ref{tab:shift_4streams}); these streams are therefore controlled cross-project distribution-shift benchmarks rather than natural single-repo concept drift.
  \item \textbf{Phase C — Confirmatory ApacheJIT Cohort ($K=4$ Umbrella Ecosystems / 5 Repositories, 7,477 Commits) and Upstream Monorepo Disclosure:} Prior to downloading official \texttt{apachejit\_total.csv} ($106{,}674$ rows) \citep{keshavarz2022apachejit}, all modeling and statistical code was frozen and passed a 14-check pre-unlock gate. \textit{Data-construction funnel and upstream monorepo split (Table~\ref{tab:confirmatory_funnel}):} our ingestion applied only a single chronological cutoff ($\text{author\_date} \le \text{2019-01-01}$) matching Benchmark~B's eight ApacheJIT streams so every evaluated commit retained $\ge 1$ year of follow-up. Auditing raw \texttt{apachejit\_total.csv} resolves why \texttt{HDFS*} ($N=2{,}850$, $76.7\%$ overall, $89.2\%$ eval [$1{,}718/1{,}925$]) and \texttt{MapReduce*} ($N=1{,}318$, $63.5\%$ overall, $78.8\%$ eval) have smaller sizes and elevated defect rates relative to ApacheJIT Table~2 ($10{,}359$ and $5{,}833$): upstream \texttt{apachejit\_total.csv} places exactly $11{,}964$ clean Hadoop monorepo commits ($0$ buggy; matching $(10{,}359-2{,}907)+(5{,}833-1{,}321)=7{,}452+4{,}512=11{,}964$) under \texttt{project="apache/hadoop"}, creating class-dependent clean-commit omission in \texttt{apache/hadoop-hdfs} and \texttt{apache/hadoop-mapreduce}. Consequently, those two slices reflect an archive distribution with omitted negatives rather than natural repository base rates. By contrast, \texttt{Spark} ($1{,}273$, $46.5\%$), \texttt{Zeppelin} ($1{,}350$, $44.6\%$), and \texttt{ZooKeeper} ($686$, $48.5\%$) have no monorepo clean-commit split; we therefore foreground this post-audit \textbf{Standalone-3} subgroup as the more interpretable external validation check while retaining the frozen-protocol \textbf{Umbrella-4} record transparently (Section~\ref{sec:rq2}). \textit{Post-unlock split-sizing exception:} upon inspecting unlocked sizes (prior to confirmatory training), four repositories had $N < 2{,}000$, violating the $N \ge 4{,}000$ rule's $1{,}500+500$ floor; adding an $N < 4{,}000$ branch ($n_{\text{init}} = \operatorname{clip}(\operatorname{round}(0.25N/25)\cdot 25, 150, 1500)$, $n_{\text{val}} = \operatorname{clip}(\operatorname{round}(0.08N/25)\cdot 25, 50, 500)$) yielded $(700, 225, 1{,}925)$ on \texttt{HDFS*}, $(325, 100, 893)$ on \texttt{MapReduce*}, $(325, 100, 848)$ on \texttt{Spark}, $(350, 100, 900)$ on \texttt{Zeppelin}, and $(175, 50, 461)$ on \texttt{ZooKeeper} without modifying any development bundle or method parameter. At $K=4$, exact two-sided Wilcoxon $p_{\min} = 0.125 > 0.05$, scoping Phase~C strictly as a directional out-of-sample replication check.
\end{enumerate}

\textbf{Observable Archive-Horizon Follow-Up Audit (Appendix Tables~\ref{tab:followup_audit}--\ref{tab:followup_sensitivity}).} To separate alarm-time censoring from end-of-archive censoring of $y_i^*$, Table~\ref{tab:followup_audit} audits remaining follow-up $T_{\text{end}} - t_i$ across all \textbf{34 natural repository streams} ($2$ Benchmark~A + $14$ Cabral19 + $8$ ApacheJIT Dev + $5$ PR-SZZ + $5$ Confirmatory). Every evaluated commit in Cabral19 (min $366.0\text{d}$) and Confirmatory ApacheJIT (min $388.8\text{d}$) and $99.3\%+$ of Benchmark~B ApacheJIT (min $301.9\text{d}$, $\le 0.7\%$ $<365\text{d}$) has $\ge 300\text{--}366\text{d}$ of follow-up; four of five PR-SZZ streams have $\ge 343.6\text{d}$ (only \texttt{n60\_angular} has min $156.9\text{d}$, $2.4\%$ $<180\text{d}$, $22.7\%$ $<365\text{d}$). In contrast, Benchmark~A natural Qt extends to $T_{\text{end}}$ (min $0.0\text{d}$; $15.5\%$ $<180\text{d}$, $41.0\%$ $<365\text{d}$; OpenStack min $174.7\text{d}$, $30.4\%$ $<365\text{d}$). Table~\ref{tab:followup_sensitivity} tabulates complete AP and $F_1$ across all core methods restricted to $\ge 180\text{d}$ and $\ge 365\text{d}$ follow-up: across all reported slices, \texttt{Full-LR} retains its AP lead and \texttt{Survival-PU} retains its Track-1 $F_1$ improvement over $B_2\text{-4W}$ and $B_0$ (also edging \texttt{Full-LR} on Qt $F_1$), whereas some pairwise AP orderings shift under follow-up restriction (e.g., on OpenStack $\ge 365\text{d}$ AP, \texttt{Survival-PU} [$0.3903$] $>$ $B_2\text{-4W}$ [$0.3850$] $>$ $\text{Wait-}90$ [$0.3841$]; on Qt $\ge 365\text{d}$ AP, $B_0$ [$0.2283$] $>$ $B_{1b}^*$ [$0.2260$]).

\subsection{Track 1 Within-Host Counterfactuals vs.\ Track 2 Adapted Baselines}
\label{sec:baselines}

Across 29 total methods (\textbf{26 deployable methods} plus three hindsight references, verified across 41 method execution branches), we maintain a two-track design separating controlled within-host supervision counterfactuals from simpler baselines and adapted stream learners, while explicitly distinguishing three fairness axes: \textit{update compute budget} (matched via \texttt{Teacher-Only (Matched)}), \textit{training sample pool} (matched via $B_{1b}^*$ vs.\ $B_2\text{-4W}$), and \textit{alarm trigger timing} (shared across Track~1 vs.\ native per-event/periodic updates in Track~2):

\begin{itemize}[leftmargin=1.8em,itemsep=0.10em]
  \item \textbf{Track 1 Within-Host Suite (15 Deployable + 3 Hindsight Methods):} Frozen $B_0$; Pote-inspired hindsight references $B_{1a}^*$ (full-prefix hindsight), $B_{1b}\text{-1W}^*$, and $B_{1b}^*$ ($4W$ window oracle); real-time censored $B_2$-1W, $B_2$-4W (exact $1W/4W$ parity with $B_{1b}^*$), dynamically reweighted $B_2$-4W-Bal, and $2{,}000$-commit $B_2$-2000; delayed windows Wait-$30/60/90$; disjoint-$U$ Pure-nnPU \citep{kiryo2017positive}; host-adapted ODaSC-Host ($15\text{d}$ wait + DenStream confidence weights); single-budget Teacher-Only (Exact) and optimizer-step-matched Teacher-Only (Matched); zero-fit FixedCurve-PU; Survival-PU (+Legacy Gate); and gate-free \textbf{Survival-PU}. In RQ4a, we additionally evaluate two alarm-synchronized post-hoc controls on unchanged $B_2\text{-4W}$ weights: \textbf{B2 + PostHoc Logit} and Proposition-1-inspired plug-in \textbf{B2 + PostHoc Ratio} (explicit formulas in Section~\ref{sec:rq4}).
  \item \textbf{Track 2 Linear and Adapted Native-Protocol Stream Competitors (11 Deployable Methods):} Unsupervised LOC-Only ($1/\max(\text{LA}+\text{LD}, 1)$); single-feature LApredict \citep{zeng2021deep}, 3-feature Churn-LR, and all-metric Full-LR (frozen and $4W$-adaptive); \textit{adapted native-protocol implementations} of ElasticNet SGD batch-oversampling BORB \citep{cabral2023borb} (native $200$-commit cadence and post-window BORB-Alarm; note EMSE 2023 evaluated multiple base learners including MLPMask, so our BORB conclusions apply specifically to this ElasticNet instance) and event-ordered River Online Hoeffding Tree Bagging implementations of ORB \citep{cabral2019class} and specification-level ODaSC \citep{song2022odasc} consuming strictly $t_{\text{event}} < t_{\text{commit}}$ causal events (exact constructors in Appendix~\ref{app:repro}).
\end{itemize}

\subsection{Endpoints, Three-Branch Causal Calibration, and Statistical Aggregation}
\label{sec:metrics_protocol}

\textbf{Primary and Secondary Endpoints.} Threshold-free \textbf{Average Precision (AP)}, computed across the concatenated evaluation stream (whole-stream AP; contrasted explicitly against per-window mean AP in Section~\ref{sec:rq2}), serves as our single primary endpoint. Secondary endpoints comprise ROC-AUC, fixed-$0.5$ $F_1$, causally tuned $F_{1,\text{th}}$, Recall@20\%Effort (tracking top-20\% Jaccard overlap with LOC-Only), and probabilistic error metrics---post-Platt Brier score ($\text{Brier}_{\text{cal}}$, which decomposes into calibration plus refinement/resolution), raw Brier score, Expected Calibration Error ($\text{ECE}_{\text{cal}}$), and calibration slope (decomposed in RQ4a).

\textbf{Prefix-Invariant Three-Branch Causal Calibration.} For each evaluation window $w = [s_w, e_w)$ at causal cutoff $t_w = t_{s_w - 1}$, our causal stream calibrator (exact symbol in Appendix~\ref{app:repro}) executes a strict three-branch hierarchy: (1) when at least $80$ prior evaluation-stream commits are aged $\ge 90\text{d}$ ($\mathcal{M}_w^{\text{eval}} = \{s_0 \le j < s_w : t_w - t_j \ge 90\text{d}\}$), it calibrates on the trailing $\le 1{,}200$ commits of $\mathcal{M}_w^{\text{eval}}$; (2) otherwise, when at least $80$ pre-stream initialization commits are aged $\ge 90\text{d}$ ($\mathcal{M}_w^{\text{init}}$), it uses the trailing $\le 1{,}200$ commits of $\mathcal{M}_w^{\text{init}}$; or (3) otherwise, it applies a causal prior-odds logit-shift fallback $\hat{p}_{i,\text{cal}} = \sigma(z_i + \operatorname{logit}(\hat{\pi}_0) - \operatorname{median}(z_{\text{init}}))$ ($\hat{\pi}_0 \in [0.02, 0.65]$) and fixed $\tau_{\text{th}}(w) = 0.50$. On branches (1)--(2), letting $\mathcal{M}_w$ denote the selected slice labeled by $y_{j,\text{obs}}(t_w)$:
\begin{equation}
  \begin{aligned}
    \hat{p}_{i,\text{cal}} &= \sigma(a_w z_i + b_w), \qquad \text{Brier}_{\text{cal}} = \frac{1}{N_{\text{eval}}}\sum_{i=s_0}^{N-1}\big(\hat{p}_{i,\text{cal}} - y_i^*\big)^2, \\
    \tau_{\text{th}}(w) &= \operatorname*{arg\,max}_{\tau \in \mathcal{G}_w} F_1\!\Big(\big\{\mathbb{I}(\hat{p}_j \ge \tau),\, y_{j,\text{obs}}(t_w)\big\}_{j \in \mathcal{M}_w}\Big),
  \end{aligned}
  \label{eq:causal_cal}
\end{equation}
where $(a_w, b_w)$ are 1D $L_2$-regularized Platt parameters \citep{platt1999probabilistic}, $\mathcal{G}_w \subset [0.05, 0.95]$ combines $41$ uniform thresholds and $19$ score quantiles, and $F_{1,\text{th}}$ evaluates causal per-commit decisions $\hat{y}_{i,\text{th}} = \mathbb{I}(\hat{p}_i \ge \tau_{\text{th}}(w_i))$ directly on raw (un-Platted) probabilities $\hat{p}_i = \sigma(z_i)$---so thresholding is decoupled from Platt scaling (and RQ4a additionally verifies Platt-thresholded $F_1$ symmetrically). Because median fix latency exceeds $90$ days on many streams, $\mathcal{M}_w$ itself retains residual right-censoring; we directly quantify this calibration-pool censoring rate and report Raw (No-Platt) vs.\ 90-day Platt calibration metrics symmetrically for all five methods in Section~\ref{sec:rq4}. All 41 method execution branches passed deterministic pipeline prefix-invariance testing ($0.0$ historical change under adversarial future feature, label, and simultaneous-timestamp perturbations); note that prefix invariance certifies leak-free \textit{pipeline execution}, not commit-time provenance of upstream merged Git snapshots. Throughout all empirical tables, \textbf{boldface denotes the best deployable value in each row or column} at displayed precision (excluding $B_{1a}^*, B_{1b}^*$; in Table~\ref{tab:bench_b_holm}, bold denotes $p_{\text{Holm}} < 0.05$).

\textbf{Aggregation Hierarchy, Multiplicity Control, and Decision Rules.} For each method $m$, stream score $\bar{s}_{r,m}$ averages across random seeds ($10$ MLP / $5$ LightGBM); donor-cluster score $c_{k,m} = |\mathcal{S}_k|^{-1}\sum_{r \in \mathcal{S}_k} \bar{s}_{r,m}$ averages equally across member streams of ecosystem cluster $k \in \{1,\dots,K\}$. Formal $\alpha = 0.05$ exact two-sided paired Wilcoxon signed-rank tests (Pratt tie convention via exact integer subset-sum DP) with step-down Holm--Bonferroni adjustment \citep{holm1979simple}, unadjusted $10{,}000$-resample cluster-level percentile bootstrap 95\% confidence intervals for the mean paired difference, and Cohen's $d_z$ are scoped to \textbf{$K=25$} (and Benchmark-A-disjoint \textbf{$K=24$}) across three families: \textbf{Family~1} ($B_{1b}^*$ vs.\ $B_2$-4W, $B_0$ vs.\ $B_2$-4W, Survival-PU vs.\ $B_2$-4W), \textbf{Family~2} (Survival-PU vs.\ $B_0$, Wait-90, Pure-nnPU, Full-LR, BORB, ORB, ODaSC), and \textbf{Family~3} (Survival-PU vs.\ FixedCurve-PU, Teacher-Only (Matched), Teacher-Only (Exact), Survival-PU (+Legacy Gate)). Because the bootstrap CI estimates the unadjusted mean difference whereas $p_{\text{Holm}}$ tests the multiplicity-adjusted signed-rank location shift, an unadjusted CI excluding zero alongside $p_{\text{Holm}} \ge 0.05$ is statistically expected. Under our codified interpretation rules, non-significant contrasts report verbatim non-establishment without claiming statistical equivalence (and numerical proximity never implies equivalence). All diagnostic $p$-values outside Table~\ref{tab:bench_b_holm}'s Holm-adjusted families (notably Table~\ref{tab:window_ap_scale}) are exploratory, unadjusted two-sided Pratt Wilcoxon tests. Throughout the paper, all reported differences and deltas are computed from unrounded means prior to four-decimal display (so subtracting rounded table values may differ by $0.0001$).

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

\textbf{Severe Right-Censoring Across Multi-Year Cohorts.} Cohort-level averages reveal substantial right-censoring across multi-year software ecosystems, with considerable variation between individual repositories. Table~\ref{tab:audit_censoring} and Figure~\ref{fig:rq1_overview}(a) distinguish two complementary censoring estimands: (i) mean per-window unrevealed defect proportions across all scheduled $400$-commit ($1W$) and $1{,}600$-commit ($4W$) stream boundaries, and (ii) seed-42 fired-alarm $1W$ censoring rates. Across multi-year ecosystems, median SZZ fix latency spans $106.5$ to $275.0$ days. Consequently, across scheduled $1W$ boundaries, \textbf{94.6\%} of window defects in natural Qt (pooled: $95.3\%$ across $981$ window-defects; $95.7\%$ across 20 LightGBM alarms and $100.0\%$ at CodeBERT's alarm), \textbf{87.5\%} in natural OpenStack ($84.4\%$ at LightGBM's alarm), \textbf{68.8\%} across Cabral's 14 streams ($68.2\%$ across 102 seed-42 MLP alarms), \textbf{70.4\%} across ApacheJIT's eight development streams ($65.8\%$ across 40 alarms), and \textbf{78.8\%} across Bludau and Pretschner's five PR-SZZ streams ($78.9\%$ across 23 alarms) remain unfixed. Even across escalated $4W$ windows, $45.1\%$ to $80.5\%$ of defects remain unrevealed in these multi-year cohorts, whereas the slower-cadence, shorter confirmatory ApacheJIT repositories exhibit markedly lower censoring ($26.4\%$ scheduled $1W$, $26.7\%$ at alarms, and $15.2\%$ at $4W$ across all five repositories; and even lower on unsplit Standalone-3: $18.0\%$ scheduled $1W$, $17.4\%$ at alarms, and $10.9\%$ at $4W$).

\textbf{Matched Hindsight Optimism vs.\ Gradual Ranking Retention.} How strongly does leaking those future SZZ fixes into alarm-triggered updates inflate apparent accuracy? Comparing offline window-oracle retraining ($B_{1b}^*$) against strictly matched causal four-window retraining ($B_2\text{-4W}$) across $K=25$ natural donor clusters (full breakdowns in Section~\ref{sec:rq2}, Tables~\ref{tab:bench_b_clusters}--\ref{tab:bench_b_holm}) reveals statistically significant hindsight optimism on both neural and tree hosts. On MLP, $B_{1b}^*$ reaches $0.5241$ AP versus $0.5004$ for $B_2\text{-4W}$ ($\Delta\text{AP} = +0.0237$, 20/0/5 wins, $p_{\text{Holm}} = 6.63 \times 10^{-5}$, $d_z = 0.906$), dropping further to $0.4704$ under single-window $B_2\text{-1W}$. On LightGBM, the hindsight gap widens to $+4.76$ AP points: $B_{1b}^*$ reaches $0.5409$ AP versus $0.4933$ for $B_2\text{-4W}$ ($\Delta\text{AP} = +0.0476$, 23/0/2 wins, $p_{\text{Holm}} = 3.70 \times 10^{-5}$, $d_z = 0.986$; $0.4529$ for $B_2\text{-1W}$).

\textbf{Why Censoring Hurts Differently Across Metrics and Drift Regimes.} Crucially, this hindsight gap does not mean that naive causal replay always fails catastrophically. Consistent with Proposition~\ref{prop:bias}(1), older commits inside a $4W$ replay buffer supply partial reveal maturity, allowing $B_2\text{-4W}$ to improve full-stream MLP AP over frozen $B_0$ on both gradual natural clusters ($0.5004$ vs.\ $0.4566$, $p_{\text{Holm}} = 0.0041$) and controlled abrupt shifts ($0.5578$ vs.\ $0.5064$ post-onset AP), while remaining below matched oracle updates ($B_{1b}^* = 0.5241$ natural; $0.6137$ shifts). The penalty manifests sharply along two other axes predicted by Proposition~\ref{prop:bias}: fixed-$0.5$ thresholded $F_1$ collapses on multimodal Benchmark~A ($0.1650$ for $B_2\text{-4W}$ and $0.1070$ for $B_2\text{-1W}$, compared with $0.2907$ for $\text{Survival-PU}$), and single-window $B_2\text{-1W}$ falls below frozen $B_0$ on the controlled abrupt shifts ($0.5011$ vs.\ $0.5064$ post-onset AP). Having established both the prevalence of right-censoring and the magnitude of hindsight optimism, we turn in RQ2 to evaluating whether maturity-calibrated PU adaptation closes this gap across hosts, ecosystems, and an untouched confirmatory cohort---and where simpler baselines prevail (previewed in Figure~\ref{fig:rq1_overview}(b)).

\rqbox{Summary for RQ1}{Across scheduled $400$-commit boundaries in multi-year cohorts, 68.8\%--94.6\% of window defects are right-censored ($65.8\%\text{--}95.7\%$ at fired alarms). Offline window-oracle supervision significantly inflates AP over matched causal $B_2\text{-4W}$ across $K=25$ donor clusters on both MLP ($+0.0237$, $p_{\text{Holm}} = 6.63\times 10^{-5}$) and LightGBM ($+0.0476$, $p_{\text{Holm}} = 3.70\times 10^{-5}$). Causal $B_2\text{-4W}$ improves AP over frozen $B_0$ on both natural streams and controlled shifts, yet remains below matched oracle updates ($B_{1b}^*$) and loses fixed-$0.5$ thresholded $F_1$; single-window $B_2\text{-1W}$ falls below $B_0$ on the controlled shifts.}

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

\textbf{Benchmark A: Simple Linear Predictors Lead Deployable Multimodal Hosts.} Evaluating Track~2 simple baselines alongside all four Track~1 multimodal and tree hosts (Table~\ref{tab:bench_a_main}, Figure~\ref{fig:rq1_overview}(b)) reveals a striking operating boundary on commit-time Qt and OpenStack streams. Across the six Benchmark~A streams, frozen 13-metric logistic regression (\textbf{Full-LR Frozen: $0.3653$ AP, $0.3939$ $F_1$, $0.3951$ $F_{1,\text{th}}$, $0.6728$ R@20\%, $0.1106$ $\text{Brier}_{\text{cal}}$}) and single-feature \textbf{LApredict Frozen} ($0.3601$ AP, $0.3853$ $F_1$, $0.1105$ $\text{Brier}_{\text{cal}}$) outperform \textit{every deployable single-host configuration} (best deployable individual hosts: frozen \texttt{DeepJIT-CNN} $B_0 = 0.3531$, frozen \texttt{Att-CNN} $B_0 = 0.3525$, frozen \texttt{LightGBM} $B_0 = 0.3356$, and \texttt{CodeBERT} $B_2\text{-1W} = 0.3268$ / $B_0 = 0.3208$), as well as the 4-host window-oracle $B_{1b}^*$ ($0.3134$ AP) and full-prefix hindsight $B_{1a}^*$ ($0.3468$ AP) means. Only non-deployable single-host full-prefix hindsight \texttt{DeepJIT-CNN} $B_{1a}^*$ ($0.3768$) exceeds \texttt{Full-LR Frozen}. Meanwhile, \textbf{Churn-LR Frozen} ($0.3397$ AP, $0.3736$ $F_1$) exceeds all window-adapted 4-host means ($0.2959\text{--}0.3234$, spanning $\text{Wait-}90 = 0.2959$ through $\text{Pure-nnPU} = 0.3234$, including $B_{1b}^* = 0.3134$) while trailing frozen \texttt{DeepJIT-CNN} ($0.3531$), \texttt{Att-CNN} ($0.3525$), and the 4-host $B_0$ mean ($0.3405$), and trivial $\text{LOC-Only}$ reaches $0.6520$ R@20\%Effort (exceeding $\text{Survival-PU}$'s $0.6156$).

Adding our causal tabular linear-residual skip substantially narrows the encoder deficit: on the two natural streams plotted in Figure~\ref{fig:rq2_skip}(b), it lifts $B_0$ AP from $0.1379 \to 0.2898$ (Att-CNN), $0.1984 \to 0.2604$ (CodeBERT), and $0.2084 \to 0.3000$ (DeepJIT-CNN)---and across all six Benchmark~A streams from $0.1849 \to 0.3525$, $0.2447 \to 0.3208$, and $0.2697 \to 0.3531$. Even with this architectural boost, however, periodic fine-tuning of high-capacity multimodal encoders on small drift windows still trails frozen linear models in AP. Restricted strictly to identical Track~1 host architectures, base-rate drift and censoring erode fixed-$0.5$ $F_1$ on $B_0$ ($0.1342$) and $B_2\text{-4W}$ ($0.1650$), whereas $\text{Survival-PU}$ recovers the highest Track-1 deployable $F_1$ ($0.2907$, close to $B_{1b}^* = 0.2937$) and $F_{1,\text{th}}$ ($0.3043$) at lower 4-host mean AP ($0.2984$ vs.\ $0.3405$ for $B_0$). Across the feature and SZZ-provenance sensitivity variants in Table~\ref{tab:bench_a_sens} (evaluated on identical common commit IDs: $N=18{,}191$ on Qt and $N=3{,}699$ on OpenStack), $B_2\text{-4W}$ attains the top deployable AP across the three Qt variants ($0.1603\text{--}0.1912$) while frozen $B_0$ leads the three OpenStack variants ($0.3332\text{--}0.3571$). Restricting saved seed-42 predictions to commits with $\ge 180\text{d}$ or $\ge 365\text{d}$ before the archive-horizon proxy $T_{\text{end}}$ (Appendix Table~\ref{tab:followup_sensitivity}) leaves the main conclusions intact---\texttt{Full-LR} retains its AP lead across all six reported slices and \texttt{Survival-PU} retains its Track-1 $F_1$ recovery over $B_2\text{-4W}$ and $B_0$ (also edging \texttt{Full-LR} on Qt $F_1$)---whereas some pairwise AP rankings shift under restriction (e.g., \texttt{Survival-PU} overtakes $B_2\text{-4W}$ and $\text{Wait-}90$ on OpenStack $\ge 365\text{d}$ AP, and $B_0$ overtakes $B_{1b}^*$ on Qt $\ge 365\text{d}$ AP).

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig3_benchmark_b_clusters_v3.pdf}
  \caption{(a) Mean AP ($\pm 1$ SEM) across $K=25$ independent natural donor clusters in Benchmark~B. (b) Post-shift AP across the four separated controlled concatenated-shift streams on both full post-onset and first 1,200 post-onset commit horizons.}
  \label{fig:rq2_bench_b}
\end{figure}

[[TAB_BENCH_B_CLUSTERS]]

[[TAB_BENCH_B_HOLM]]

[[TAB_WINDOW_AP_SCALE]]

\textbf{Benchmark B: Gradual Natural Clusters ($K=25$ Primary AP, Multi-Seed Per-Window AP, Controlled Scale Removal, and Secondary Endpoints).} Tables~\ref{tab:bench_b_clusters}--\ref{tab:window_ap_scale} and Figure~\ref{fig:rq2_bench_b}(a) report our powered cross-project inference across $K=25$ independent natural donor clusters ($K=24$ Benchmark-A-disjoint sensitivity). On primary whole-stream MLP AP, \textbf{Full-LR Adapt-4W} achieves the highest deployable mean ($0.5108$), followed by $B_2\text{-4W}$ ($0.5004$), $\text{Wait-}90$ ($0.4982$), and zero-fit \textbf{FixedCurve-PU} ($0.4942$). Within this landscape, $\text{Survival-PU}$ ($0.4866$) significantly outperforms canonical $\text{Pure-nnPU}$ ($0.4461$, $\Delta\text{AP} = +0.0405$, 18/0/7, $p_{\text{Holm}} = 0.0020$) and Hoeffding $\text{ORB}$ ($0.4468$, $\Delta\text{AP} = +0.0398$, 19/0/6, $p_{\text{Holm}} = 0.0279$), and does not differ significantly after Holm correction from $B_0$ ($0.4566$, $p_{\text{Holm}} = 0.2673$), $\text{ODaSC}$ ($0.4626$, $p_{\text{Holm}} = 0.1652$), ElasticNet $\text{BORB}$ ($0.4684$, $p_{\text{Holm}} = 0.2673$), or $\text{Full-LR Adapt-4W}$ ($0.5108$, $p_{\text{Holm}} = 0.0639$). However, \textbf{$\text{Survival-PU}$ trails $B_2\text{-4W}$ ($\Delta\text{AP} = -0.0138$, $p_{\text{Holm}} = 0.0056$) and $\text{Wait-}90$ ($\Delta\text{AP} = -0.0117$, $p_{\text{Holm}} = 0.0069$) significantly on primary whole-stream AP}.

To test whether this whole-stream deficit reflects weaker within-window discrimination or cross-window score-comparability differences (Proposition~\ref{prop:bias} Remark), Table~\ref{tab:window_ap_scale} evaluates all seven core methods across 5 seeds ($42\text{--}46$) on the exact same $K=25$ clusters ($531$ operational $400$-commit windows; $0$ single-class windows excluded; diagnostic $p$-values below are exploratory and unadjusted per Section~\ref{sec:metrics_protocol}). Moving from whole-stream AP ($B_2\text{-4W} = 0.5015$ vs.\ $\text{Survival-PU} = 0.4863$, $\Delta = -0.0152$, $5/1/19$, $p=0.0024$) to unweighted per-window AP reverses the sign against $B_2\text{-4W}$ into a small non-significant mean advantage (\textbf{$\text{Survival-PU} = 0.5246 > B_2\text{-4W} = 0.5192$}, $\Delta = +0.0054$, 95\% CI $[+0.0009, +0.0101]$, $16/1/8$, $p=0.0569 > 0.05$; positive-weighted Win-AP: $0.5478$ vs.\ $0.5442$, $\Delta = +0.0036$, $17/1/7$, $p=0.0930$), while leaving $\text{Survival-PU}$ numerically close to $\text{FixedCurve-PU}$ ($0.5246$, $p=0.9895$) and $\text{Wait-}90$ ($0.5251$, $p=0.8222$) and slightly behind $\text{Full-LR Adapt-4W}$ ($0.5270$, $p=0.6150$). Applying \textbf{controlled scale removal}---strictly monotone within-window z-score or percentile-rank standardization on the \textit{exact same predictions} before computing concatenated whole-stream AP---reverses $\text{Survival-PU}$'s $-0.0152$ mean whole-stream deficit relative to $B_2\text{-4W}$ under both Z-Norm ($0.4990$ vs.\ $0.4968$, $+0.0023$, $17/1/7$, $p=0.1954$) and Rank-Norm ($0.5027$ vs.\ $0.4989$, $+0.0038$, $16/1/8$, $p=0.0877$). Under Rank-Norm, mean gaps against $\text{FixedCurve-PU}$ ($-0.0008$, $p=0.6244$) and $\text{Wait-}90$ ($-0.0009$, $p=0.9053$) shrink near zero (under Z-Norm, a smaller gap persists vs.\ $\text{Wait-}90$ [$-0.0048$, $p=0.0265$] and $\text{FixedCurve-PU}$ [$-0.0026$, $p=0.2149$]). This cross-window comparability diagnostic supports that inter-window score scaling plays a major role in $\text{Survival-PU}$'s whole-stream deficit vs.\ $B_2\text{-4W}$ (while within-window gains over $B_2\text{-4W}$ remain modest and do not surpass $\text{Wait-}90$, $\text{FixedCurve-PU}$, or $\text{Full-LR}$); because standardization removes inter-window location, scale, and quantile differences jointly, the specific training mechanism generating those shifts is not isolated separately.

Secondary endpoints exhibit a clear metric split. ElasticNet $\text{BORB}$ ($0.5144$) and Hoeffding $\text{ODaSC}$ ($0.4951$) achieve the highest causally tuned $F_{1,\text{th}}$ overall (with $\text{Survival-PU}$ leading Track-1 methods at $0.4795$). On post-Platt Brier loss, $\text{FixedCurve-PU}$ ($0.15247$) and $\text{Survival-PU}$ ($0.15255$) are numerically close at four-decimal display ($0.1525$), ahead of $\text{Full-LR Adapt-4W}$ ($0.1531$) and $B_2\text{-4W}$ ($0.1568$).

\textbf{Benchmark B: Codified Component Controls (Family 3).} Testing whether cohort survival estimation and Stage~3 distillation justify their added complexity reveals important negative and host-dependent findings. On $K=25$ primary AP, \textbf{zero-fit FixedCurve-PU achieved significantly higher AP than cohort-estimated Survival-PU on both MLP ($0.4942$ vs.\ $0.4866$, $\Delta\text{AP} = -0.0077$, 6/0/19, $p_{\text{Holm}} = 0.0295$; $K=24$: $p_{\text{Holm}} = 0.0317$) and LightGBM ($0.4680$ vs.\ $0.4619$, $p_{\text{Holm}} = 0.0367$)}. The specific mechanism behind this difference (e.g., finite-sample survival-curve variance versus parametric prior shape) is not isolated by the benchmark design; quoting our codified decision rule verbatim: ``\textit{An incremental benefit over FixedCurve\_PU was not established on this cohort.}''

Comparing $\text{Survival-PU}$ ($0.4866$) against optimizer-step-matched $\text{Teacher-Only (Matched)}$ ($0.4727$) on $K=25$ MLP similarly yields $p_{\text{Holm}} = 0.9157$ ($K=24$: $p_{\text{Holm}} = 0.9495$); quoting our codified rule verbatim: ``\textit{An incremental benefit over Teacher\_Only\_Matched was not established on this cohort.}'' On the LightGBM host ($K=25$), by contrast, Stage~3 PU distillation delivers a large, significant gain over both $\text{Teacher-Only (Matched)}$ ($0.4619$ vs.\ $0.4240$, $+0.0379$, 19/0/6, $p_{\text{Holm}} = 9.87 \times 10^{-4}$) and $\text{Teacher-Only (Exact)}$ ($0.4186$, $p_{\text{Holm}} = 0.0135$), although causal $B_2\text{-4W}$ achieves the highest deployable $K=25$ LightGBM AP overall ($0.4933$). Finally, an incremental benefit from legacy three-state gating was not established ($\text{Survival-PU (+Legacy Gate)} = 0.4873$ vs.\ $0.4866$, $p_{\text{Holm}} = 0.8988$).

\textbf{Benchmark B: Controlled Concatenated Distribution Shifts Depend on Evaluation Horizon (4 Separated Streams).} Under sharp cross-project or cross-subsystem regime changes (Table~\ref{tab:bench_b_clusters}, rightmost column; Figure~\ref{fig:rq2_bench_b}(b); full per-stream breakdown across all four shifts in Appendix Table~\ref{tab:shift_4streams}), method rankings reverse depending on horizon length. Over the \textbf{full post-onset horizon}, $\text{Survival-PU}$ achieves the highest observed deployable mean AP (\textbf{0.5764}), closely followed by Track-1 $\text{ODaSC-Host}$ ($0.5727$), $B_2\text{-4W-Bal}$ ($0.5708$), $\text{Wait-}30$ ($0.5684$), and $\text{FixedCurve-PU}$ ($0.5652$), whereas single-window $B_2\text{-1W}$ ($0.5011$), age-blind $\text{Pure-nnPU}$ ($0.5017$), and $\text{ORB}$ ($0.4928$) fall below frozen $B_0$ ($0.5064$). Over the \textbf{immediate first 1,200 post-onset commits}, however, ElasticNet batch-oversampling \textbf{BORB-Alarm ($0.5931$)} and \textbf{BORB-Native ($0.5734$)} adapt fastest, surpassing $\text{Survival-PU}$ ($0.5591$), $\text{Wait-}30$ ($0.5587$), and $\text{ODaSC-Host}$ ($0.5586$). Per-stream inspection (Table~\ref{tab:shift_4streams}) also shows substantial stream heterogeneity: the fourth shift stream (\texttt{OS Nova$\to$Neutron}) exhibits lower AP across all adaptive methods ($0.4214$ post-onset / $0.4358$ first-1,200 for $\text{Survival-PU}$) than the three cross-project shifts in RQ4d ($0.4836\text{--}0.6799$).

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig4_confirmatory_boundary_v3.pdf}
  \caption{(a) Out-of-sample AP across the five confirmatory ApacheJIT repositories (asterisks mark Hadoop monorepo subprojects whose clean commits reside in \texttt{apache/hadoop}). (b) Exploratory association plotting evaluation horizon $N_{\text{eval}}$ against the matched oracle-minus-causal hindsight gap $\Delta\text{AP}(B_{1b}^* - B_2\text{-4W})$ on MLP across the 27 natural Benchmark~B streams ($22/27 > 0$) and five confirmatory repositories ($0/5 > 0$, separating Standalone-3 from Hadoop Monorepo-2).}
  \label{fig:rq2_confirmatory}
\end{figure}

[[TAB_CONFIRMATORY]]

\textbf{Phase C: Confirmatory Non-Replication on MLP — Standalone-3 Primary Interpretation vs.\ Frozen Umbrella-4 Record.} Table~\ref{tab:confirmatory} and Figure~\ref{fig:rq2_confirmatory} report out-of-sample results on the five confirmatory ApacheJIT repositories. Because upstream \texttt{apachejit\_total.csv} omits $11{,}964$ clean Hadoop monorepo commits into \texttt{apache/hadoop} (elevating \texttt{HDFS*} and \texttt{MapReduce*} evaluation prevalence to $89.2\%$ and $78.8\%$; Table~\ref{tab:confirmatory_funnel}), those two slices reflect an archive distribution with class-dependent clean-commit omission rather than natural repository base rates. Accordingly, the \textbf{three standalone repositories} (\texttt{Spark}, \texttt{Zeppelin}, \texttt{ZooKeeper}---a post-audit 3-repo subgroup free of monorepo splitting) provide the cleaner external check: here \textbf{MLP hindsight optimism did not replicate}, with $B_{1b}^*$ ($0.5951$) still trailing $B_2\text{-4W}$ ($0.5985$) and frozen $B_0$ ($0.6565$), while Hoeffding \textbf{ODaSC ($0.7857$)}, ElasticNet \textbf{BORB ($0.7001$)}, and Hoeffding \textbf{ORB ($0.6790$)} outperform all window-retrained MLP variants ($0.5968\text{--}0.6033$); on Standalone-3 LightGBM, $B_0$ ($0.6211$) leads $B_{1b}^*$ ($0.6173$) and $B_2\text{-4W}$ ($0.6121$).

For transparent protocol completeness, the frozen pre-unlock \textbf{umbrella-4 MLP AP} record shows the identical direction: window-oracle $B_{1b}^*$ ($0.6801$) trailed causal $B_2\text{-4W}$ ($0.6858$, $\Delta\text{AP} = -0.0057$, 0/0/4 umbrella wins, exact two-sided $p = 0.125$, $p_{\text{Holm}} = 0.375$) and trailed frozen $B_0$ ($0.7244$, $\Delta\text{AP} = -0.0443$, 1/0/3 umbrella wins: Hadoop $+0.0072$, Spark $-0.0033$, Zeppelin $-0.1455$, ZooKeeper $-0.0354$; exact two-sided $p = 0.375$), while \textbf{ODaSC ($0.8289$)}, \textbf{BORB ($0.7530$)}, and \textbf{ORB ($0.7453$)} led (on LightGBM umbrella-4, $B_{1b}^* = 0.7020$ stayed slightly above $B_2\text{-4W} = 0.6976$, 3/0/1 wins, $p = 0.375$; on the two Hadoop slices alone, MLP clustered at $0.9116\text{--}0.9587$ with $B_{1b}^* = 0.9353 < B_2\text{-4W} = 0.9476$). Because horizon length ($N_{\text{eval}} = 461\text{--}1{,}925$), prevalence, low $1W$ censoring ($18.0\%$ on Standalone-3; $26.4\%$ across all five repositories), alarm count, and project identity covary here, treating ``short, high-prevalence repos favor ODaSC'' as a validated deployment rule would be premature; it remains an exploratory hypothesis.

\rqbox{Summary for RQ2}{Frozen \texttt{Full-LR} ($0.3653$ AP) leads deployable Benchmark~A, while \texttt{Survival-PU} recovers Track-1 $F_1$ ($0.2907$). On $K=25$ whole-stream MLP AP, \texttt{Full-LR Adapt-4W} ($0.5108$, $p_{\text{Holm}}=0.0639$) leads in mean AP while $B_2\text{-4W}$ ($0.5004$, $p_{\text{Holm}}=0.0056$), $\text{Wait-}90$ ($0.4982$, $p_{\text{Holm}}=0.0069$), and \texttt{FixedCurve-PU} ($0.4942$, $p_{\text{Holm}}=0.0295$) significantly exceed \texttt{Survival-PU} ($0.4866$). On 5-seed predictions, within-window standardization reverses the whole-stream mean gap vs.\ $B_2\text{-4W}$ (Rank-Norm: $0.5027$ vs.\ $0.4989$; Z-Norm: $0.4990$ vs.\ $0.4968$), and per-window AP edges $B_2\text{-4W}$ non-significantly ($0.5246$ vs.\ $0.5192$, unadjusted $p=0.0569$) while remaining numerically close to \texttt{Full-LR}, $\text{Wait-}90$, and \texttt{FixedCurve-PU}. \texttt{Survival-PU} leads 4-shift full post-onset AP ($0.5764$), whereas \texttt{BORB-Alarm} leads the first 1,200 commits ($0.5931$). On the confirmatory cohort, MLP hindsight optimism did not replicate on unsplit Standalone-3 ($0.5951$ vs.\ $0.5985$; \texttt{ODaSC} $=0.7857$) or frozen Umbrella-4 ($0.6801$ vs.\ $0.6858$; \texttt{ODaSC} $=0.8289$).}

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

Table~\ref{tab:rq3_ablations} and Figure~\ref{fig:rq3_abl} dissect maturity-estimator sensitivity and internal component contributions across the ten-stream representative subset (seven independent natural streams plus three controlled cross-project shifts; $K=7$ natural contrasts have exact $p_{\min} = 0.015625 > 0.0125$ and are reported descriptively).

\textbf{Low Sensitivity Across Stage~1 Maturity Estimators.} Replacing risk-eligible cohort cumulative incidence ($0.5543$ AP, $0.5395$ $F_{1,\text{th}}$, $0.6920$ R@20\%Effort) with $10^{-4}$-day Kaplan--Meier cure ($0.5624$ AP, $0.5434$ $F_{1,\text{th}}$), global unstratified Weibull mixture-cure MLE ($0.5652$ AP, $0.5389$ $F_{1,\text{th}}$; a Weibull delay family containing the exponential case at $k=1$, inspired by delayed-feedback/cure likelihoods \citep{chapelle2014modeling}), or zero-fit 90-day $\text{FixedCurve-PU}$ ($0.5639$ AP, $0.5391$ $F_{1,\text{th}}$) spans a narrow descriptive range of $1.09$ AP points and $0.45$ $F_{1,\text{th}}$ points, with zero survival fallbacks and zero unreachable bisection solves across all four variants. On this 10-stream mix, $B_2\text{-4W}$ reaches $0.5705$ AP at lower $F_{1,\text{th}}$ ($0.5132$) and higher $\text{Brier}_{\text{cal}}$ ($0.1749$ vs.\ $0.1698$ for $\text{Survival-PU}$).

\textbf{Unrevealed-Commit Stage~1--3 Mechanism Audit vs.\ FixedCurve-PU and Positive Writeback (Table~\ref{tab:mechanism_audit}).} Because Eq.~\eqref{eq:bayes_pu} sets $q_i = 1$ deterministically on already-revealed positives ($y_{i,\text{obs}}=1$), evaluating all pool commits conflates label writeback with genuine maturity correction on unrevealed commits. Appendix Table~\ref{tab:mechanism_audit} audits all 62 seed-42 DriftLens alarms on the 10 representative streams ($74{,}400$ pool evaluations spanning $59{,}600$ unique commits, comparing \texttt{Survival-PU} and \texttt{FixedCurve-PU} head-to-head). Here retrospective reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is defined over all archive-labeled positives $\mathcal{P} = \{j : y_j^* = 1\}$ in each stream (counting unlinked $t_j^{\text{fix}} = +\infty$ as unrevealed at every finite age $a$), evaluated on $\mathcal{B}_{\text{cal}}$ commit ages and averaged per alarm before aggregating across alarms ($n=62$) or streams ($n=10$); all $\pm$ dispersion terms in this audit and Table~\ref{tab:mechanism_audit} report sample standard deviations (SD):
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Stage 1 $\hat{c}(a)$ vs.\ Stage 2 $\hat{\pi}_{\text{target}}$ (Non-Monotonic Cross-Stage Error Propagation):} Adaptive cohort incidence recovers empirical maturity $c^*(a)$ more accurately than the fixed 90-day prior: alarm-level $\hat{c}(a)$ MAE $= 0.0938 \pm 0.0714$ SD (stream-level $0.0873 \pm 0.0403$ SD; $0.0803$ natural vs.\ $0.1257$ shifts) compared with \texttt{FixedCurve-PU}'s $0.1204 \pm 0.0639$ SD (stream-level $0.1310 \pm 0.0594$ SD). Crucially, however, more accurate Stage~1 maturity estimation does \textit{not} automatically yield lower downstream error: on Stage~2 target prevalence, \texttt{FixedCurve-PU} achieves slightly lower alarm-level MAE ($0.0565 \pm 0.0368$ SD) than \texttt{Survival-PU} ($0.0620 \pm 0.0452$ SD, bias $+0.006$, median $n_{\text{eff,heur}} = 125.0$, median anchor weight $28.6\%$; both improve over unshrunk $\hat{\pi}_{\text{IP}} = 0.0712$).
  \item \textit{Disentangling Positive Writeback from Unrevealed-Commit ($y_{\text{obs}}=0$) Soft Targets:} Across all $74{,}400$ pool evaluations ($13.0\%$ observed positives), applying fair positive writeback ($y_{\text{obs}}=1 \to 1.0$) to the de-weighted teacher alone lowers Brier from $0.1525$ to $0.1134$, confirming that writeback accounts for most of the raw all-pool reduction. Restricting strictly to \textbf{unrevealed commits ($y_{\text{obs}} = 0$, $64{,}745$ evaluations, zero writeback)}, soft targets still improve modestly over the de-weighted teacher in every age stratum ($0.1303 \to 0.1236$ for \texttt{Survival-PU}, alarm-level $\Delta = -0.0065 \pm 0.0355$ SD, stream-level $-0.0063 \pm 0.0142$ SD, near-zero bias $+0.0027$). Yet mirroring Stage~2, \texttt{FixedCurve-PU} achieves slightly lower overall unrevealed soft-target Brier ($0.1200$, at negative bias $-0.0358$; lower on young/mid strata while \texttt{Survival-PU} is lower on aged $\ge 90\text{d}$ commits: $0.1181$ vs.\ $0.1194$). This cross-stage trade-off---where better Stage-1 $\hat{c}(a)$ MAE does not translate monotonically into lower Stage-2 prior MAE or Stage-3 unrevealed Brier---is consistent with \texttt{FixedCurve-PU}'s competitiveness on gradual streams, although Brier and signed bias do not directly determine ranking AP and the specific source of \texttt{FixedCurve-PU}'s $K=25$ AP advantage remains unisolated.
\end{itemize}

\textbf{Component Ablations and Simplification Candidates.} Freezing the teacher to the initial pre-stream checkpoint produces the steepest drop ($0.5543 \to 0.5229$ AP, $-3.14$ points), co-occurring with $4.2$ unreachable bisection alarms/stream ($35$ across 5 seeds on \texttt{apache\_flink}, $7$ on \texttt{n60\_airflow}, confounding teacher staleness with clamping). Removing bisection ($0.5436$ AP) or anchor shrinkage $m_0 = 0$ ($0.5447$ AP) degrades both AP and $\text{Brier}_{\text{cal}}$, whereas omitting Stage-2 covariate density ratios slightly improves descriptive AP ($0.5558$ vs.\ $0.5543$), $F_{1,\text{th}}$ ($0.5406$), and R@20\%Eff ($0.6935$) at lower runtime ($8.93\text{ s}$ vs.\ $9.25\text{ s}$), making it an unverified simplification candidate. On this 10-stream subset, \texttt{Survival-PU} ($0.5543$ AP) exceeds step-matched $2\times$ \texttt{Teacher-Only} ($0.5293$) while legacy gating yields $0.5521$.

\rqbox{Summary for RQ3}{Cohort incidence, Kaplan--Meier cure, global Weibull cure, and \texttt{FixedCurve-PU} span $1.09$ AP points with zero fallbacks. Across 62 alarms ($59{,}600$ unique commits; $\pm$ = sample SD), cohort incidence improves Stage-1 $\hat{c}(a)$ MAE over \texttt{FixedCurve-PU} ($0.0938 \pm 0.0714$ vs.\ $0.1204 \pm 0.0639$), yet \texttt{FixedCurve-PU} achieves slightly lower Stage-2 $\hat{\pi}_{\text{target}}$ MAE ($0.0565$ vs.\ $0.0620$) and unrevealed-only ($y_{\text{obs}}=0$) Brier ($0.1200$ vs.\ $0.1236$; both improving over de-weighted teachers at $0.1303$), showing non-monotonic error propagation across stages. Teacher adaptation, bisection, and shrinkage matter most; density-ratio omission ($0.5558$ AP) is a simplification candidate.}

% ----------------------------------------------------------------------------
% 6.4 RQ4
% ----------------------------------------------------------------------------
\subsection{RQ4: Stress Tests Across Latency Extremes, Monitoring, Noise, and Prompting}
\label{sec:rq4}

We subject our findings to five targeted stress tests: future-only latency scaling with symmetrical Raw/Platt and alarm-synchronized post-hoc $B_2$ controls (RQ4a), an unconfounded $2 \times 2$ DriftLens monitoring factorial (RQ4b), independent semi-synthetic PR-SZZ label noise (RQ4c), matched-horizon window and hyperparameter sweeps (RQ4d), and a paired Code-LLM prompting case study (RQ4e).

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

\textbf{RQ4a: Future-Only Verification-Latency Scaling ($\kappa \in [0.0, 3.0]$), Stage-1 Misspecification, and Post-Hoc $B_2$ Control.} Tables~\ref{tab:rq4a_kappa}--\ref{tab:rq4a_posthoc_b2} and Figure~\ref{fig:rq4_lat_mon}(a) scale finite post-$t_0$ bug-fix delays by $\kappa \in [0.0, 3.0]$ while preserving unlinked $+\infty$ defects and pre-$t_0$ supervision ($B_0 = 0.4872$ AP invariant across all rows; oracle post-$t_0$ parity: $B_2\text{-4W} = B_{1b}^* = 0.5980$ AP).

Comparing $B_2\text{-4W}$, $\text{Wait-}90$, and $\text{Survival-PU}$ in Table~\ref{tab:rq4a_kappa} confirms a sharp ranking-versus-classification split as latency grows: while $B_2\text{-4W}$ stays ahead of $\text{Survival-PU}$ in 10-stream AP ($0.5974 \to 0.5466$ vs.\ $0.5760 \to 0.5438$), at $\kappa = 3.0\times$ $B_2\text{-4W}$'s fixed-$0.5$ $F_1$ drops by $15.78$ points ($0.5403 \to 0.3825$; $B_2\text{-1W} \to 0.2582$) and its causally tuned $F_{1,\text{th}}$ drops from $0.5755 \to 0.4465$ ($\text{Wait-}90$: $F_1 = 0.4790$, $F_{1,\text{th}} = 0.4736$). In contrast, \textbf{Survival-PU sustains $F_1 = 0.5259$ at $\kappa = 3.0\times$} ($+14.34$ points over $B_2\text{-4W}$) \textbf{and causally tuned $F_{1,\text{th}} = 0.5022$ vs.\ $0.4465$ for $B_2\text{-4W}$} ($+5.57$ points; $+2.86$ over $\text{Wait-}90$), while age-blind $\text{Pure-nnPU}$ collapses in effort recall ($\text{R@20\%Eff} = 0.3410$).

Table~\ref{tab:rq4a_fixedcurve} resolves when adaptive Stage-1 cohort incidence is justified relative to zero-fit \texttt{FixedCurve-PU}'s hardcoded 90-day half-life:
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Bidirectional Misspecification Protection ($\kappa = 0\times$ and $\kappa \in \{2\times, 3\times\}$):} When bug fixes reveal immediately ($\kappa = 0.00\times$), \texttt{FixedCurve-PU}'s hardcoded 90-day prior treats clean young commits as censored, lagging adaptive \texttt{Survival-PU} across all four endpoints: $0.5544$ vs.\ $0.5760$ AP ($-2.16$ points; while $B_2\text{-4W}$ leads at $0.5974$), $0.4964$ vs.\ $0.5364$ $F_1$, $0.5593$ vs.\ $0.5643$ $F_{1,\text{th}}$, and $0.1687$ vs.\ $0.1631$ Brier. At severe latency ($\kappa = 2.00\times, 3.00\times$), the hardcoded 90-day prior overestimates maturity, and adaptive \texttt{Survival-PU} outperforms \texttt{FixedCurve-PU} on $F_1$ ($0.5348$ vs.\ $0.5268$ at $2\times$; $0.5259$ vs.\ $0.5132$, $+1.28$ points at $3\times$), $F_{1,\text{th}}$ ($0.5211$ vs.\ $0.5123$; $0.5022$ vs.\ $0.4918$), and $\text{Brier}_{\text{cal}}$ ($0.1853$ vs.\ $0.1875$; $0.1945$ vs.\ $0.1966$).
  \item \textit{Where Zero-Fit \texttt{FixedCurve-PU} Leads Pure Ranking ($\kappa \in [1.5\times, 3.0\times]$ AP):} Across $\kappa \in [0.25\times, 3.0\times]$, \texttt{FixedCurve-PU} retains higher AP than \texttt{Survival-PU}, attaining the highest deployable AP at $\kappa \in \{1.5\times, 2.0\times, 3.0\times\}$ ($0.5637$, $0.5603$, and $0.5547$ at $\kappa = 3.0\times$, surpassing both $B_2\text{-4W} = 0.5466$ and $\text{Survival-PU} = 0.5438$).
\end{itemize}

\textbf{Symmetrical Raw vs.\ 90-Day Platt Calibration and Alarm-Synchronized Post-Hoc $B_2$ Controls (Table~\ref{tab:rq4a_posthoc_b2}).} Auditing the 90-day calibration pool $\mathcal{M}_w$ confirms heavy residual censoring whenever median fix lag exceeds 90 days ($55.3\%$ Benchmark~A, $43.8\%$ Benchmark~B, $27.3\%$ Confirmatory, and $5.0\% \to 46.2\% \to 65.9\%$ across $\kappa = 0\times, 1\times, 3\times$). To separate training-time PU updates from post-hoc score correction and decouple Platt scaling from causal thresholding, Table~\ref{tab:rq4a_posthoc_b2} evaluates all five methods symmetrically under both \textbf{Raw (No-Platt)} and \textbf{Shared 90-Day Platt} probabilities using the identical 90-day causal threshold search ($F_{1,\text{th}}$ on raw probabilities; Platt-thresholded $F_1$ reported below). Both post-hoc controls keep $B_2\text{-4W}$'s trained weights untouched and update strictly at fired DriftLens alarms $t_{\text{alarm}}$ using causal history $t \le t_{\text{alarm}}$:
\begin{enumerate}[leftmargin=1.6em,itemsep=0.04em]
  \item \textit{\texttt{B2 + PostHoc Logit}:} initializes $\delta_0$ on $z_{\text{init}}$ to match $\hat{\pi}_{\text{init}}$, then at each fired alarm solves via 28-step bisection on the trailing evaluated window's $B_2$ logits for scalar $\delta_k \in [-8, 8]$ satisfying $\text{mean}(\sigma(z_{B_2,\text{past}} + \delta_k)) = \hat{\pi}_{\text{target}}$ (Eq.~\eqref{eq:ht_prev} on trailing $4W$), outputting $z_{\text{logit}} = z_{B_2} + \delta_k$ until the next alarm.
  \item \textit{\texttt{B2 + PostHoc Ratio} (Proposition~\ref{prop:bias}-inspired plug-in ratio control):} removes $B_2\text{-4W}$'s constant training class weight $\omega_{\text{base}}$ via $r_{\text{dewt}}(x) = \sigma(z_{B_2}(x) - \log\omega_{\text{base}})$, divides by the alarm-updated trailing-$4W$ causal mean maturity $\bar{c}_{4W} = \max(\text{mean}_{j \in W_{4W}} \hat{c}(a_j), 0.10)$ (initialized at $\bar{c}_{\text{init}}$ pre-first-alarm), and outputs $\hat{\eta}_{\text{ratio}}(x) = \operatorname{clip}(r_{\text{dewt}}(x)/\bar{c}_{4W}, 10^{-4}, 1-10^{-4})$.
\end{enumerate}
Table~\ref{tab:rq4a_posthoc_b2} yields three crisp mechanistic conclusions:
\begin{enumerate}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Neither tested post-hoc $B_2$ control matches PU retraining's heavy-latency classification recovery.} At $\kappa = 0\times$, \texttt{PostHoc Logit} loses whole-stream AP ($0.5974 \to 0.5377$) via alarm-to-alarm scale shifts while preserving \texttt{WinAP} ($0.6153$), whereas plug-in \texttt{PostHoc Ratio} incurs a smaller AP drop ($0.5974 \to 0.5808$, $-0.0166$) and achieves the lowest raw Brier ($0.1708$). Its \texttt{WinAP} drop at $\kappa=0\times$ ($0.6153 \to 0.5977$) stems entirely from upper-bound clipping at $1-10^{-4}$, clipping $7.9\%$ of commits across $71.8\%$ of windows at $\kappa=0\times$ (raising within-window tied commits from $7.1\%$ to $16.1\%$; $4.8\% / 2.3\%$ clipped across $64.2\% / 47.3\%$ of windows at $\kappa=1\times / 3\times$). Removing that upper clip restores \texttt{B2-4W}'s exact \texttt{WinAP} ($0.6153 / 0.5863 / 0.5693$), shifts $\kappa=0\times$ whole-stream AP to $0.5904$ via cross-window $\bar{c}_{4W}$ changes, and leaves fixed-$0.5$ $F_1$ unchanged ($0.5428 / 0.4375 / 0.3216$). At $\kappa = 3\times$, both \texttt{PostHoc Logit} ($F_1 = 0.3593$, $F_{1,\text{th}} = 0.4420$) and plug-in \texttt{PostHoc Ratio} ($F_1 = 0.3216$, $F_{1,\text{th}} = 0.4543$) trail \texttt{Survival-PU} ($F_1 = 0.5259$, $F_{1,\text{th}} = 0.5022$) and \texttt{FixedCurve-PU} ($0.5132 / 0.4918$), while within-window AP differences remain small ($0.5657\text{--}0.5735$). Crucially, because \texttt{PostHoc Ratio} explicitly removes $\log\omega_{\text{base}}$ to estimate unweighted $\hat{\eta}_{\text{ratio}}$ whereas \texttt{Survival-PU} and $B_2\text{-4W}$ emit class-weighted probabilities ($\omega_+ > 1$), fixed-$0.5$ $F_1$ is asymmetric across scales; the scale-fair comparison is threshold-tuned $F_{1,\text{th}}$, where \texttt{Survival-PU} still leads \texttt{PostHoc Ratio} by $+4.79$ points and $B_2\text{-4W}$ by $+5.57$ points.
  \item \textit{Why \texttt{Survival-PU}'s raw probabilities appear less calibrated than $B_2\text{-4W}$ and \texttt{FixedCurve-PU} (closing Proposition~\ref{prop:bias}).} Across all three $\kappa$ levels, \texttt{Survival-PU} shows higher raw Brier and ECE than $B_2\text{-4W}$ (e.g., $0.2004 / 0.1641$ vs.\ $0.1700 / 0.1071$ at $\kappa=1\times$), while \texttt{FixedCurve-PU} attains the lowest raw Brier ($0.1693$) and ECE ($0.0973$) at $\kappa=3\times$. This follows directly from Stage~3 and Proposition~\ref{prop:bias}'s Remark: Stage~3 applies positive class weight $\omega_{+,\text{teach}} > 1$ \textit{on top of} already prevalence-calibrated soft targets $q_i(\delta^*)$, pushing raw student probabilities upward. Conversely, $B_2\text{-4W}$'s censoring compression ($\bar{c}_W < 1$) and \texttt{FixedCurve-PU}'s negative target bias ($-0.0358$, Table~\ref{tab:mechanism_audit}) pull outputs downward, partially canceling $\omega_+ > 1$'s upward shift in raw Brier/ECE.
  \item \textit{Cautionary finding on 90-day Platt calibration under severe lag.} At $\kappa = 3\times$, raw probabilities outperform 90-day Platt calibration on ECE for all five methods and on Brier for four of five (\texttt{PostHoc Logit} stays nearly unchanged: $0.2020 \to 0.2018$). This serves primarily as a warning against fitting standard 90-day Platt scaling to a $65.9\%$-censored calibration pool, and secondarily shows that \texttt{Survival-PU} is more resilient when that calibrator is applied---retaining the lowest post-Platt Brier ($0.1945$) and ECE ($0.1603$), with smaller Raw$\to$Platt increases ($+0.0123 / +0.0379$) than $B_2\text{-4W}$ ($+0.0279 / +0.0626$) and \texttt{FixedCurve-PU} ($+0.0274 / +0.0681$), though not uniformly smaller across metrics and post-hoc controls (\texttt{PostHoc Logit}: $-0.0002 / +0.0216$; \texttt{PostHoc Ratio}: $+0.0151 / +0.0202$). Meanwhile, \texttt{Survival-PU}'s $F_{1,\text{th}}$ lead holds whether causal thresholding runs on raw scores ($0.5022$) or 90-day Platted scores ($0.4898$ vs.\ $0.4814$ Fix / $0.4543$ Ratio / $0.4385$ $B_2$).
\end{enumerate}

\textbf{RQ4b: Unconfounded $2 \times 2$ DriftLens Calibration Factorial.} Table~\ref{tab:rq4b_monitors} and Figure~\ref{fig:rq4_lat_mon}(b) hold percentile and cooldown fixed at $(p_{95}, \text{cooldown}=1)$, measuring resampled-null alarm rates via $20$ block-bootstrap windows (block length $50$) drawn from the same pre-stream validation pool that calibrates initial thresholds. Replacing \texttt{IID} with \texttt{Circular Block} cuts this \textbf{validation-pool resampled-null policy alarm rate $4.8\times$ under matched refresh rules} ($20.2\% \to 4.2\%$ static; $16.4\% \to 3.4\%$ rolling; $5.9\times$ combined). Enabling Rolling Ref+Th achieves zero missed transitions across the four controlled-shift streams evaluated over five seeds ($4 \times 5 = 20$ runs), reducing missed transitions from $5.0\%\text{--}10.0\%$ ($1/20$ under \texttt{IID + Static}; $2/20$ under \texttt{Block + Static}) to $0.0\%$ ($0/20$ under either Rolling Ref+Th configuration), with conditional detection delays of $310$ commits for \texttt{IID + Rolling Ref+Th} and $350$ commits for \texttt{Block + Rolling Ref+Th} (vs.\ $1{,}050$ commits for periodic $3W$ and $74.7\%$ 10-stream alarm rate for the Pote-inspired static $(p_{98}, \text{cd}=0)$ reference). Crucially, \textbf{fewer alarms do not automatically improve AP}: default \texttt{Block + Rolling} ($0.5543$ SPU AP, $32.1\%$ 10-stream alarm rate) slightly trails \texttt{IID + Rolling} ($0.5628$, $43.5\%$ alarm rate) and \texttt{Periodic Every 3W} ($0.5621$, $31.7\%$ alarm rate). Even so, DriftLens remains operationally preferable to fixed periodic schedules because it detects abrupt shifts \textbf{$3\times$ faster} ($350$ vs.\ $1{,}050$ commits) while holding stationary validation-pool alarm rates an order of magnitude lower ($3.4\%$ vs.\ $33.3\%$). Block calibration therefore governs an alarm-rate/compute versus detection-delay trade-off rather than boosting AP directly.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig7_szz_window_sensitivity_v3.pdf}
  \caption{(a) $2 \times 2$ frozen-prediction semi-synthetic SZZ perturbation across five PR-SZZ streams ($\alpha_{\text{FP}}=8.42\%, \beta_{\text{FN}}=47.15\%$). (b) Matched-horizon fixed 1,200-commit post-onset AP across $W \in \{200, 400, 600, 800\}$ on the three representative controlled concatenated-shift streams (Nova$\to$Zulip, Ignite$\to$Cassandra, Airflow$\to$Pulsar).}
  \label{fig:rq4_szz_win}
\end{figure}

[[TAB_RQ4D_SZZ]]

\textbf{RQ4c: Frozen-Prediction $2 \times 2$ Semi-Synthetic PR-SZZ Noise Perturbation.} To probe combined latency and permanent mislabeling, Table~\ref{tab:rq4d_szz} and Figure~\ref{fig:rq4_szz_win}(a) inject independent conditional flip noise over automated PR-SZZ reference labels across five streams: clean reference commits ($y_{\text{ref}} = 0$) flip to buggy at rate $\alpha_{\text{FP}} = 8.42\%$ (assigned synthetic reveal delays drawn from empirical stream latencies), and buggy reference commits ($y_{\text{ref}} = 1$) lose their fix permanently ($y_{\text{pert}} = 0, t_{\text{fix}} = +\infty$) at rate $\beta_{\text{FN}} = 47.15\%$---parameterized from project-averaged B-SZZ error rates in \citet{bludau2026quantifying} (Tables~4--5). Under this specific independent perturbation model (which does not capture feature- or age-dependent SZZ error correlation), training on noisy labels and evaluating on unperturbed PR-SZZ reference labels produces a family-level separation: window-replay and maturity-weighted batch methods cluster within $0.0034$ AP---\textbf{Survival-PU ($0.3802$)}, $\text{Wait-}90$ ($0.3795$, $+0.0007$ descriptive gap), and $B_2\text{-4W}$ ($0.3768$)---whereas online stream learners degrade substantially more (ElasticNet $\text{BORB}$: $0.4224 \to 0.3187$; Hoeffding $\text{ORB}$: $0.4243 \to 0.2782$). Generalizing this robustness beyond independent synthetic noise requires future validation on manually verified SZZ subsets.

\textbf{RQ4d: Matched-Horizon Window Sweeps and Hyperparameter Sensitivity.} Figure~\ref{fig:rq4_szz_win}(b) evaluates window size $W \in \{200, 400, 600, 800\}$ on the matched fixed-1,200 post-onset horizon across the three representative controlled shift streams in our ten-stream subset (Nova$\to$Zulip, Ignite$\to$Cassandra, and Airflow$\to$Pulsar; full 4-shift comparison including \texttt{OS Nova$\to$Neutron} in Table~\ref{tab:shift_4streams}). Here $\text{Survival-PU}$ leads deployable methods at narrow and medium windows---$W=200$ ($0.5810$ vs.\ $0.5347$ $\text{Wait-}90$ and $0.5296$ $B_2\text{-4W}$), $W=400$ ($0.5966$ vs.\ $0.5758$ $B_2\text{-4W}$; averaging these three shifts vs.\ $0.5591$ across all four shifts), and $W=600$ ($0.5801$ vs.\ $0.5727$ $\text{Wait-}90$). \textbf{At $W=800$, however, $\text{Wait-}90$ ($0.6078$) and $B_2\text{-4W}$ ($0.5989$) exceed $\text{Survival-PU}$ ($0.5974$)} once wide windows span older matured history (and across full-stream 10-stream AP for $W \in \{200..800\}$, $B_2\text{-4W}$ leads at $0.5521\text{--}0.5705$ vs.\ $0.5320\text{--}0.5543$ for $\text{Survival-PU}$). Sweeping anchor shrinkage $m_0 \in \{0, 25, 50, 100\}$ ($0.5447\text{--}0.5567$ AP) and tail cap $\gamma_{\text{tail}} \in [0.65, 0.94]$ ($\text{Brier}_{\text{cal}} \in [0.1692, 0.1716]$) confirms low sensitivity.

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.98\textwidth]{fig8_llm_and_runtime_forensics_v3.pdf}
  \caption{(a) Paired prompting case study (\texttt{deepseek-flash} on 320-commit Qt and OpenStack slices) vs.\ domain-trained JIT predictors (initial checkpoints for CodeBERT/LightGBM, 0 alarms on slice). (b) Standalone wall-clock runtime on pre-extracted features across 27 natural Benchmark~B streams.}
  \label{fig:rq4f_rt}
\end{figure}

\textbf{RQ4e: Paired Code-LLM Prompting Case Study (Does In-Context Learning Bypass the Causal Wall?).} To probe whether parameter-free in-context learning (ICL) sidesteps weight-retraining censoring, we conducted a paired prompting case study using \texttt{deepseek-flash} on 320-commit slices of Qt and OpenStack ($640$ distinct commits evaluated across prompt regimes and three exemplar seeds, totaling $6{,}400$ logical evaluations; full table, model settings, and sampling protocol in Appendix~\ref{app:audits}, Table~\ref{tab:rq4f_llm}; overview in Figure~\ref{fig:rq4f_rt}(a)). Comparing strictly paired 6-shot prompts sharing identical exemplar commits and order but differing solely in observable $y_{\text{obs}}(t)$ versus eventual $y^*$ ($0.424$ flipped positives/prompt on Qt; $1.165$ on OpenStack) shows that exemplar censoring depresses thresholded $F_1$ on OpenStack from $0.4946$ to $0.3876$ ($-10.70$ points; Qt: $0.2938 \to 0.2856$), whereas \textbf{OpenStack AP is slightly higher under censored 6-shot ($0.4277 \pm 0.0152$ SD across three exemplar seeds) than paired oracle 6-shot ($0.4177 \pm 0.0299$ SD; Qt: $0.2282 \pm 0.0109$ vs.\ $0.2378 \pm 0.0166$ SD)}, mirroring Proposition~\ref{prop:bias}'s probability-shift pattern. Because this case study covers one lightweight model, 640 commits where zero DriftLens alarms fired, and historical public repositories that may overlap LLM pretraining corpora, it demonstrates only that exemplar label visibility alters prompting classification on this model rather than establishing general Code-LLM ICL claims; on both slices, domain-trained JIT predictors outperform every prompting regime in AP ($0.3401\text{--}0.5788$).

\rqbox{Summary for RQ4}{At $\kappa=3.0\times$, \texttt{FixedCurve-PU} achieves top deployable AP ($0.5547$) and raw Brier/ECE ($0.1693 / 0.0973$), whereas \texttt{Survival-PU} outperforms $B_2\text{-4W}$, both post-hoc $B_2$ controls, and \texttt{FixedCurve-PU} on threshold-tuned $F_{1,\text{th}}$ ($0.5022$ raw; $0.4898$ Platt) and fixed-$0.5$ $F_1$ ($0.5259$), while also beating \texttt{FixedCurve-PU} across all reported metrics at $\kappa=0\times$. Applying 90-day Platt calibration to the $65.9\%$-censored pool worsens calibration relative to raw scores across methods (with \texttt{Survival-PU} most resilient post-Platt). Block bootstrap cuts validation-pool null alarms $4.8\times$ and detects shifts $3\times$ faster than periodic $3W$ ($350$ vs.\ $1{,}050$ commits; $0/20$ missed). Batch window/PU methods resist independent SZZ noise better than \texttt{BORB}/\texttt{ORB}, and exemplar censoring depresses \texttt{deepseek-flash} OpenStack $F_1$ ($0.4946 \to 0.3876$).}

% ----------------------------------------------------------------------------
% 6.5 RQ5
% ----------------------------------------------------------------------------
\subsection{RQ5: Runtime Telemetry and Paired Equal-Cap Budget Error Analysis}
\label{sec:rq5}

[[TAB_RQ5_RUNTIME]]

[[TAB_RQ5_PAIRED_BUDGET]]

\textbf{Cached Execution vs.\ Feature-Encoding Runtime Telemetry.} Table~\ref{tab:rq5_runtime} and Figure~\ref{fig:rq4f_rt}(b) report phase-decomposed wall-clock runtimes on pre-extracted tabular features across 27 natural Benchmark~B streams ($5.99$ alarms/stream), charging every alarm-triggered method its standalone dependency chain ($3{,}474.1\text{ ms}$ anchor + $173.2\text{ ms}$ monitor). Standalone $\text{Survival-PU}$ executes in $14.43\text{ s}$/stream ($\approx 1.66\text{ s}$/alarm), matching step-matched $\text{Teacher-Only}$ ($14.37\text{ s}$). Online Hoeffding $\text{ORB}$ ($2.25\text{ s}$), frozen $B_0$ ($3.71\text{ s}$), and ElasticNet $\text{BORB}$ ($3.70\text{ s}$) run fastest, whereas Hoeffding $\text{ODaSC}$ requires $69.87\text{ s}$/stream due to per-event tree updates and DenStream micro-cluster radius maintenance. On multimodal Qt ($25{,}704$ commits), cached $\text{Survival-PU}$ execution takes $18.19\text{ s}$ on CPU LightGBM ($20.2$ alarms; seed-averaged alarm-step $p_{50}=0.80\text{ s}$, $p_{95}=1.07\text{ s}$, max $=1.14\text{ s}$) and $7.26\text{ s}$ on RTX~4080 GPU CodeBERT ($1.2$ alarms; $p_{50}=p_{95}=\text{max}=1.32\text{ s}$). \textbf{Adding offline GPU CodeBERT batch feature encoding ($37.32\text{ s}$ on Qt; $19.32\text{ s}$ on OpenStack) yields $44.58\text{ s}$ total batch processing on Qt}---distinct from unbatched CI webhook latency.

\textbf{Paired Whole-Stream vs.\ Per-Window Equal-Cap Budget Net Gain (Table~\ref{tab:rq5_paired_budget}).} Showing only directional rescue stories cannot establish net review yield, and allocating budgets across a multi-year retrospective stream answers a different question than matching sprint-level operational capacity. Table~\ref{tab:rq5_paired_budget} therefore reports a seed-42 commit-level diagnostic audit comparing \texttt{Survival-PU} minus $B_2\text{-4W}$ across (i) unmatched fixed-$0.50$ thresholding, (ii) equal-cap retrospective whole-stream budgets (Top-10\% commits and 20\% LOC), and (iii) equal-cap operational per-window $400$-commit budgets (Top-10\% commits and 20\% LOC per window), corroborated by 5-seed multi-seed windowed 20\%-LOC recall ($\Delta\text{WinR}_{20}$). Under 20\%-LOC budgets, commits are ranked by descending risk density $\hat{p}/\text{LOC}$ (ties: smaller LOC) and selected greedily without splitting commits while $\sum \text{LOC} \le 0.20\,\text{LOC}_{\text{total}}$, yielding equal budget caps with closely matched realized consumption ($20.0\% / 20.0\%$ on 7/8 rows and $20.0\% / 19.8\%$ on OS LightGBM; Top-10\% commit counts are exact):
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Fixed Threshold ($0.50$, Unmatched Volume):} Preventing posterior collapse rescues $+418$ to $+512$ defects on Qt and $+26$ to $+146$ on OpenStack, at the cost of $+1{,}838$ to $+3{,}211$ extra clean commits on Qt ($+130$ to $+357$ on OpenStack).
  \item \textit{Whole-Stream Retrospective vs.\ Per-Window Operational Budgets:} Directly paralleling Section~\ref{sec:rq2}'s granularity finding, retrospective whole-stream selection penalizes \texttt{Survival-PU} heavily on Qt DeepJIT-CNN ($-54$ Top-10\%, $-123$ 20\%-LOC), Qt Att-CNN ($-38$ LOC), OS Att-CNN ($-33 / -20$), and OS LightGBM ($-67 / -33$). Switching to \textbf{operational per-window 400-commit budgets} eliminates or reverses nearly every steep whole-stream loss: Qt LightGBM remains strongly positive ($+124$ Win-10\%, $+31$ Win-20\%L; 5-seed $\Delta\text{WinR}_{20} = +0.0250$), Qt DeepJIT-CNN flips to $+15$ Win-20\%L ($-1$ Win-10\%; 5-seed $+0.0248$), OS DeepJIT-CNN stays positive ($+1 / +10$; 5-seed $+0.0198$), and Qt Att-CNN ($+8 / +4$; 5-seed $-0.0001$) and OS LightGBM ($+1 / -3$; 5-seed $+0.0025$) turn near-even, while Qt CodeBERT remains negative ($-38 / -22$; 5-seed $-0.0028$). Qualitative Tier-1 SZZ rescue illustrations ($22$ filtered Qt CodeBERT commits, $B_2\text{-4W} \in [0.0518, 0.0871]$ vs.\ $\text{SPU} \in [0.9543, 0.9754]$, plus FN/FP churn distributions) are archived in Appendix Table~\ref{tab:rq5_forensics}.
\end{itemize}

\rqbox{Summary for RQ5}{On cached features, \texttt{ORB} ($2.25\text{ s}$) and \texttt{BORB} ($3.70\text{ s}$) run fastest, \texttt{Survival-PU} takes $14.43\text{ s}$/stream ($44.58\text{ s}$ including Qt CodeBERT batch encoding), and \texttt{ODaSC} takes $69.87\text{ s}$. Under equal budget caps (exact Top-10\% commit counts; closely matched realized 20\%-LOC consumption), retrospective whole-stream selection penalizes \texttt{Survival-PU} on 5/7 alarmed hosts, whereas operational per-window 400-commit budgeting reverses most steep losses (corroborated by 5-seed $\Delta\text{WinR}_{20}$: $+0.0250$ Qt LightGBM, $+0.0248$ Qt DeepJIT-CNN, $+0.0198$ OS DeepJIT-CNN) while remaining host-dependent (negative on Qt CodeBERT).}

% ============================================================================
% 7. Discussion, Observed Operating Boundaries, and Limitations
% ============================================================================
\section{Discussion, Observed Operating Boundaries, and Limitations}
\label{sec:discussion}

We synthesize our findings around the three questions framing this study, avoiding verbatim repetition of Section~\ref{sec:results}.

\subsection{What Label-Visibility Timing Changes}
\label{sec:disc_timing}

Our audit separates two questions often conflated in drift-triggered JIT-SDP \citep{pote2026adaptive}: \textit{whether small-window retraining beats freezing} ($B_{1b}^*$ vs.\ $B_0$) and \textit{whether future-fix leakage inflates retrained weights} ($B_{1b}^*$ vs.\ $B_2\text{-4W}$). On long multi-year tabular streams ($K=25$), leaking future SZZ fixes significantly inflates whole-stream AP on both MLP and LightGBM. On two other regimes, however, even window-oracle $B_{1b}^*$ fails to surpass frozen $B_0$ or causal $B_2\text{-4W}$ on MLP: low-prevalence multimodal Benchmark~A (where frozen \texttt{Full-LR} leads, echoing \citet{zeng2021deep}) and the short confirmatory ApacheJIT cohort (where Standalone-3 prevalence is much higher at $44.6\%\text{--}48.5\%$, while horizon length, initialization scale, low $1W$ censoring, and project identity shift jointly). Finally, chronological evaluation must disclose end-of-archive follow-up limits (Appendix Tables~\ref{tab:followup_audit}--\ref{tab:followup_sensitivity}) alongside upstream repository-key artifacts such as \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split (Table~\ref{tab:confirmatory_funnel}).

\subsection{What Complex Maturity Correction Adds Over Simpler Causal Baselines}
\label{sec:disc_increment}

Once strict causal visibility ($t \le t_{\text{alarm}}$) is enforced, our multi-track controls pinpoint where three-stage \texttt{Survival-PU} delivers genuine value and where simpler baselines suffice:
\begin{itemize}[leftmargin=1.6em,itemsep=0.06em]
  \item \textit{Whole-Stream vs.\ Within-Window Granularity:} On gradual $K=25$ streams, \texttt{Full-LR Adapt-4W}, $B_2\text{-4W}$, $\text{Wait-}90$, and \texttt{FixedCurve-PU} lead whole-stream AP over \texttt{Survival-PU}. Yet our 5-seed cross-window comparability diagnostic (Table~\ref{tab:window_ap_scale}) reverses the mean whole-stream deficit against $B_2\text{-4W}$ via within-window standardization, while leaving per-window means close across $\text{Wait-}90$, \texttt{FixedCurve-PU}, and \texttt{Full-LR}.
  \item \textit{Non-Monotonic Cross-Stage Error Propagation:} On strictly unrevealed commits ($y_{\text{obs}}=0$, Table~\ref{tab:mechanism_audit}), adaptive cohort incidence estimates Stage-1 maturity $\hat{c}(a)$ more accurately than \texttt{FixedCurve-PU} and reduces soft-target Brier over de-weighted teachers. However, \texttt{FixedCurve-PU} attains slightly lower Stage-2 prior MAE and unrevealed Brier, showing that better Stage-1 estimation does not automatically lower downstream error across coupled stages.
  \item \textit{Heavy Latency, Scale-Fair Controls, and Calibration Caution:} Under $\kappa=3\times$ latency (Table~\ref{tab:rq4a_posthoc_b2}), neither post-hoc $B_2$ control matches \texttt{Survival-PU}'s threshold-tuned $F_{1,\text{th}}$ recovery, and adaptive Stage~1 simultaneously protects against \texttt{FixedCurve-PU}'s $\kappa=0\times$ misspecification. On probabilistic accuracy, Stage~3's class weight $\omega_{+,\text{teach}} > 1$ explains why \texttt{Survival-PU}'s raw outputs trail $B_2\text{-4W}$ and \texttt{FixedCurve-PU} (whose downward censoring/target bias partially cancels $\omega_+ > 1$, matching Proposition~\ref{prop:bias}). More broadly, because raw probabilities outperform 90-day Platt calibration under $65.9\%$ pool censoring, our calibration result serves primarily as a warning against blind 90-day Platt scaling and highlights maturity-aware post-hoc calibration as a key open problem.
\end{itemize}

\subsection{Equal-Cap Inspection Cost, Observed Patterns vs.\ Exploratory Hypotheses, and Scope Limits}
\label{sec:disc_implications}

Translating ranking shifts into code-review impact depends sharply on budget granularity (Table~\ref{tab:rq5_paired_budget}): preventing fixed-$0.5$ recall collapse rescues hundreds of defects only by flagging thousands of clean commits, and retrospective whole-stream budgeting penalizes \texttt{Survival-PU} on 5/7 alarmed Benchmark~A hosts. Under realistic per-window 400-commit sprint budgets (equal budget caps with closely matched realized LOC consumption), however, steep losses vanish or turn positive on Qt LightGBM, Qt DeepJIT-CNN, and OpenStack DeepJIT-CNN (corroborated by 5-seed $\Delta\text{WinR}_{20}$), while remaining negative on Qt CodeBERT.

Because deployers cannot observe future latency multipliers ($\kappa$) or shift horizons beforehand, Table~\ref{tab:decision_matrix} separates \textbf{Part~I: Directly Observed Empirical Patterns} from \textbf{Part~II: Exploratory Hypotheses (Not Validated Selection Rules)}. For benchmark designers, three practices follow directly: enforce $t_i^{\text{fix}} \le t_{\text{alarm}}$ with archive-horizon follow-up disclosure; include linear baselines, zero-fit maturity priors, post-hoc controls, and adapted stream learners \citep{cabral2019class,cabral2023borb,song2022odasc}; and pair whole-stream AP with per-window AP, scale-normalized AP, and operational per-window equal-cap budget net TP counts.

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

\textbf{Scope Limitations.} Six limitations bound our conclusions: (1) Benchmark~A orders merged Git revisions by \texttt{author\_date} rather than live Gerrit patch-set-1 uploads, and archive observation horizons $T_{\text{end}}$ are observable proxies rather than certified collection cutoffs; (2) Stage~1 assumes local trailing-cohort stationarity (I1--I2), Stage~2 uses heuristic shrinkage, and Eq.~\eqref{eq:ht_prev}'s outer $\pi_{\max}=0.70$ clamp causes unreachable bisection when alarm-time $\bar{y}_{\text{obs},\mathcal{B}_{\text{cal}}} > 0.70$ ($28.6\%\text{--}30.0\%$ of alarms on \texttt{HDFS*}); (3) 90-day causal Platt calibration retains $27\%\text{--}66\%$ censoring under multi-month lag and worsens post-Platt calibration at $\kappa=3\times$ (Section~\ref{sec:rq4}); (4) RQ4b's null alarm rate resamples the calibration validation pool rather than independent stationary production streams; (5) controlled concatenated shifts impose a 1-hour splice with $23.7\%\text{--}60.0\%$ cross-boundary fixes; and (6) RQ4e is a single-model (\texttt{deepseek-flash}), 640-commit prompting case study on public commits predating LLM pretraining cutoffs.

% ============================================================================
% 8. Threats to Validity
% ============================================================================
\section{Threats to Validity}
\label{sec:threats}

\textbf{Construct Validity.} SZZ back-tracing is susceptible to tangled commits, refactorings, and unlinked fixes \citep{dacosta2017framework,rosa2021evaluating,fan2021impact}, while archival collections also experience end-of-archive censoring ($t_i^{\text{fix}} > T_{\text{end}}$). We mitigated construct threats by quarantining Tier-3 and retroactive links ($t_{\text{fix}} < t_{\text{commit}}$) to $+\infty$, auditing remaining follow-up across all 34 natural repository streams plus full Benchmark~A $\ge 180\text{d}/\ge 365\text{d}$ sensitivity (Appendix Tables~\ref{tab:followup_audit}--\ref{tab:followup_sensitivity}), disclosing upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split (Table~\ref{tab:confirmatory_funnel}), verifying paired whole-stream and per-window equal-cap budget net TP counts (Table~\ref{tab:rq5_paired_budget}), and reporting \texttt{LOC-Only} overlap alongside whole-stream, per-window, and scale-normalized AP.

\textbf{Internal Validity.} All 41 method execution branches passed deterministic pipeline prefix-invariance testing ($0.0$ historical perturbation error), though prefix invariance certifies leak-free code execution rather than upstream commit-time feature provenance. We separated protocol-specified development analyses ($K=25$) from the pre-unlock frozen confirmatory cohort, disclosed the post-unlock $N < 4{,}000$ sizing amendment and its exact splits, treated $B_{1b}^*$ strictly as a Pote-inspired counterfactual rather than a verbatim original-repo reproduction, and verified all $1{,}452$ empirical table cells across $22$ tables, locked parameters, exact-path prose comparisons, and quantifier/relational claim guards automatically (Appendix~\ref{app:repro}).

\textbf{External and Conclusion Validity.} Our adapted Track-2 implementations (ElasticNet-\texttt{BORB}, OHT-\texttt{ORB}, OHT-\texttt{ODaSC}) reflect specific base learners rather than entire algorithmic paradigms, and independent synthetic SZZ flips do not capture feature-dependent real SZZ errors. Formal $\alpha=0.05$ Holm--Bonferroni inference is scoped strictly to $K=25$ and $K=24$ natural donor clusters (distinct from the exploratory unadjusted $p$-values in Table~\ref{tab:window_ap_scale}; Section~\ref{sec:metrics_protocol}), and failed MLP directional replication on Phase~C (both unsplit Standalone-3 and frozen Umbrella-4) is reported directly without post hoc retuning.

% ============================================================================
% 9. Conclusion
% ============================================================================
\section{Conclusion}
\label{sec:conclusion}

Unsupervised drift monitors remove label dependence when deciding \textit{when} to flag shifts, not when \textit{supervising} model updates under multi-month SZZ verification latency. Across multi-year ecosystems, 68.8\%--94.6\% of defects inside scheduled 400-commit windows remain unrevealed at update time. Consequently, leaking future fixes inflates whole-stream AP significantly across $K=25$ donor clusters on both MLP ($+0.0237$, $p_{\text{Holm}} = 6.63 \times 10^{-5}$) and LightGBM ($+0.0476$, $p_{\text{Holm}} = 3.70 \times 10^{-5}$), while circular block-bootstrap calibration cuts validation-pool null alarms $4.8\times$ and detects abrupt shifts $3\times$ faster than periodic schedules.

Our cross-paradigm audit demonstrates that no single adaptive strategy dominates all regimes. Frozen linear models lead deployable Benchmark~A; simpler causal baselines ($B_2\text{-4W}$, $\text{Wait-}90$, \texttt{FixedCurve-PU}) significantly outperform cohort-fitted \texttt{Survival-PU} on gradual $K=25$ whole-stream AP (reversing the mean $B_2\text{-4W}$ gap non-significantly under within-window score standardization); ElasticNet \texttt{BORB} leads $K=25$ $F_{1,\text{th}}$ and early post-shift recovery; and Hoeffding \texttt{ODaSC} leads the short confirmatory cohort (Standalone-3: $0.7857$; Umbrella-4: $0.8289$), where MLP hindsight optimism fails to replicate. Within these clear operating boundaries, \texttt{Survival-PU} offers targeted advantages on full post-shift AP ($0.5764$), LightGBM soft-target distillation, and classification resilience across both latency extremes---preventing \texttt{FixedCurve-PU}'s $\kappa = 0\times$ misspecification and beating $B_2\text{-4W}$, two post-hoc $B_2$ controls, and \texttt{FixedCurve-PU} on threshold-tuned $F_{1,\text{th}}$ ($0.5022$) at $\kappa = 3\times$. Finally, our calibration audit warns that standard 90-day Platt scaling degrades probabilities when calibration pools suffer heavy censoring; developing maturity-aware post-hoc calibration and prospectively validating deployment-observable selection rules remain central priorities for adaptive JIT-SDP.

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

This appendix consolidates seven supplementary audit tables referenced in Sections~\ref{sec:benchmarks}--\ref{sec:rq5}. Tables~\ref{tab:confirmatory_funnel}--\ref{tab:followup_sensitivity} document the confirmatory ApacheJIT filtering funnel (reconciling upstream \texttt{apachejit\_total.csv}'s $11{,}964$-commit clean \texttt{apache/hadoop} split), observable archive-horizon follow-up windows across all 34 natural repository streams, and complete Benchmark~A follow-up sensitivity across $\ge 180\text{d}$ and $\ge 365\text{d}$ cutoffs.

[[TAB_CONFIRMATORY_FUNNEL]]

[[TAB_FOLLOWUP_AUDIT]]

[[TAB_FOLLOWUP_SENSITIVITY]]

Table~\ref{tab:mechanism_audit} audits Stage~1--3 intermediate accuracy across all 62 fired DriftLens alarms on the 10-stream subset ($74{,}400$ pool evaluations across $59{,}600$ unique commits), separating strictly unrevealed commits ($y_{\text{obs}}=0$) from the all-pool positive-writeback control and comparing \texttt{Survival-PU} against \texttt{FixedCurve-PU}. Reference maturity $c^*(a) = |\mathcal{P}|^{-1}\sum_{j \in \mathcal{P}} \mathbb{I}(\Delta_j \le a)$ is computed over all archive-labeled positives $\mathcal{P} = \{j : y_j^* = 1\}$ in each stream (counting unlinked $t_j^{\text{fix}} = +\infty$ as unrevealed), evaluated on $\mathcal{B}_{\text{cal}}$ ages and averaged per alarm before aggregating across alarms ($n=62$) or streams ($n=10$), with all $\pm$ dispersion values reporting sample SD.

[[TAB_MECHANISM_AUDIT]]

Table~\ref{tab:shift_4streams} disaggregates all four Benchmark~B controlled concatenated shifts alongside cross-boundary fix-reveal fractions, Table~\ref{tab:rq4f_llm} presents the complete paired Code-LLM prompting case study (RQ4e), and Table~\ref{tab:rq5_forensics} archives the qualitative Tier-1 SZZ commit illustrations referenced in Section~\ref{sec:rq5}.

[[TAB_SHIFT_4STREAMS]]

[[TAB_RQ4F_LLM]]

\textbf{RQ4e Paired Prompting Protocol Details.} The 6,400 logical evaluations in Table~\ref{tab:rq4f_llm} queried \texttt{deepseek-flash} (\texttt{temperature=0.0}, JSON schema output, $6{,}819$ HTTP attempts including $419$ retried transient transport errors and $2$ final $0.50$ fallbacks; $99.97\%$ valid-response rate, $100\%$ JSON compliance on $6{,}398$ responses, $99.95\%$ non-empty diff coverage) across the first 320 evaluation commits of natural Qt and OpenStack. Zero-shot prompts passed commit message, truncated code diff, and commit-time metrics; paired 6-shot arms sampled identical historical 1W exemplars across three random seeds differing solely in $y_{\text{obs}}(t)$ versus $y^*$, while the descriptive causal-mature arm sampled aged ($\ge 90\text{d}$) exemplars ($F_1 = 0.5072$ OS / $0.2984$ Qt).

[[TAB_RQ5_FORENSICS]]

\textbf{RQ5 Qualitative Commit Illustrations and Churn Skew.} Applying our pre-declared Tier-1 forensic filter (lag $\ge 14\text{d}$, $B_0 \ge 0.45$, $B_2\text{-4W} \le 0.35$, $\text{SPU} \ge 0.50$) on natural Qt (CodeBERT, seed 42; Alarm~\#1 at commit $9{,}900$ had $88.19\%$ $4W$ censoring) yields $22$ rescue illustrations (Table~\ref{tab:rq5_forensics}; $0$ alarms on OpenStack CodeBERT seed 42). Across full evaluation sets, Tier-1 false-negative candidates skew smaller (Qt: $436$ FNs, median $32$ vs.\ $69$ LOC across all $981$ defects; OpenStack: $477$ FNs, median $74$ vs.\ $89$ LOC) while high-score false positives skew larger (Qt: $1{,}620$ FPs, median $65$ vs.\ $9$ LOC clean; OpenStack: $18$ FPs, median $146$ vs.\ $19$ LOC), ranging from $2$-LOC FNs (\texttt{qtmultimedia:4bb75559}) to $570{,}767$-LOC imports (\texttt{qtwebkit:d441d6f3}).

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

- **Context:** Unsupervised monitors like DriftLens flag representation shifts in Just-in-Time Defect Prediction (JIT-SDP) without labels, motivating immediate window retraining.
- **Objective:** Monitoring detects shifts without supervising updates: under multi-month SZZ verification latency, recent bug-inducing commits still appear clean. We audit this causal gap and map boundaries across linear, streaming, and Positive-Unlabeled (PU) learners.
- **Method:** Across 37 development streams and five confirmatory ApacheJIT repositories (34 natural and 8 controlled-shift streams; 347,122 natural commits; paired Code-LLM study), we pair block-bootstrap DriftLens with three-stage **Survival-PU** as a maturity-correction testbed combining cohort-incidence maturity, clipped prior shrinkage (`c_ht = 0.10`), and PU distillation.
- **Results:** At scheduled 400-commit boundaries, 68.8%–94.6% of defects are right-censored, and window-oracle supervision inflates whole-stream Average Precision (AP) over causal retraining across $K=25$ clusters ($+0.0237$ MLP, $p_{\\text{Holm}}=6.63\\times 10^{-5}$; $+0.0476$ LightGBM, $p_{\\text{Holm}}=3.70\\times 10^{-5}$). Yet frozen logistic regression leads Benchmark A ($0.3653$ AP); on gradual $K=25$ whole-stream AP, `Full-LR Adapt-4W` ($0.5108$) leads in mean while `B2-4W`, `Wait-90`, and zero-fit `FixedCurve-PU` ($0.4942$, $p_{\\text{Holm}}=0.0295$) significantly exceed `Survival-PU` ($0.4866$), though within-window standardization reverses the mean `B2-4W` gap (non-significantly). On the confirmatory cohort, MLP hindsight optimism does not replicate (`ODaSC` leads unsplit Standalone-3 at $0.7857$ and umbrella-4 at $0.8289$). `Survival-PU` leads full post-shift AP ($0.5764$), avoids $\\kappa=0\\times$ misspecification, and protects $F_1$ under $3\\times$ latency where `FixedCurve-PU` leads AP and raw Brier.
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

Compiling `manuscript/main.tex` via `pdflatex -> bibtex -> pdflatex -> pdflatex` reproduces the camera-ready 53-page Springer SQJ PDF with `0` Errors, `0` Warnings, and `0` Overfull hboxes.

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
