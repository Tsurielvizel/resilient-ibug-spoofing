# Resilient I-Bug Navigation under Intermittent Signal Spoofing

**Authors:** Tsuriel Vizel & Avraham Guez  
**Supervisor:** Dr. Andrey Zhitnikov (Technion – Israel Institute of Technology)

> **Agent Quick-Start Context:** Read this file first to avoid scanning the entire repository. The active working directory with the latest code, data, plots, and LaTeX manuscripts is `22_10/`.

---

## 1. Project Overview
This project studies minimalist intensity-based bug navigation (**I-Bug**) under **intermittent additive signal spoofing** ($S(p,t) = m(p-p_t) + u(t)\frac{\alpha_e}{\|p-p_e\|_2}$, where $m(p-p_t) = \|p-p_t\|_2^{-1}$ and $u(t) \in \{0,1\}$). Without adding any hardware, odometry, or global coordinates, we interpose a **kinematics-based temporal anomaly filter** (using the radial Lipschitz bound $\Delta I_{\max}$) and evaluate three navigation strategies:
1. **Naive (`:Naive`)**: Unmodified I-Bug baseline (corrupted by gradient deflection and $i_H$ hit-threshold poisoning).
2. **Cautious (`:Cautious`)**: Freezes motion ($\Delta p_k = 0$), pose, and memory ($i_H, i_{\max}, i_{\text{mem}}$) upon detecting a positive intensity spike ($\mathcal{A}_{\text{spike}}$), and resumes only after detecting a drop ($\mathcal{A}_{\text{drop}}$) followed by $N_{\text{confirm}} = 3$ stable steps. **Theorem 1** proves step-by-step active-trajectory preservation and conditional finite-time convergence:
   $$N_{\text{cautious}} \le N_{\text{ideal}} + \left\lceil \frac{N_{\text{ideal}}}{T_{\text{off}} - N_{\text{confirm}}} \right\rceil (T_{\text{on}} + N_{\text{confirm}}).$$
3. **Inertial (`:Inertial`)**: Freezes heading and memory updates during spoofing bursts while continuing open-loop forward motion (`MoveToTower`) or physical boundary following (`FollowObstacle`). **Proposition 1** proves that memory freezing alone cannot guarantee universal convergence due to **topological phase-locking (resonance)** when $T_{\text{orbit}} = m \cdot T_{\text{cycle}}$.

---

## 2. Repository Map (`22_10/` is Canonical)

### Simulation Code (Julia) & Plotting (Python)
* `22_10/CoreLogic.jl`: Core 2D continuous kinematic simulator, environment generator (`generate_random_env`), sensor model, anomaly filter (`ANOMALY_THRESHOLD = 2.0`, `SENSOR_SAMPLE_DISP = 0.002`, `ENEMY_CONFIRM_STEPS = 3`), and the three navigation strategies.
  * Arena: $\mathcal{W} = [0, 20]^2$, $p_0 = (2, 1)$, $p_t = (18, 18)$, $R_{\text{reach}} = 0.5$, $r_c = \Delta s = 0.3$.
  * Adversary: $\alpha_e = 1.0$, $T_{\text{delay}} = 20$, $T_{\text{cycle}} = 40$, $T_{\text{on}} = 10$ ($25\%$ duty cycle), $T_{\text{off}} = 30$, $N_{\max} = 2,000$.
* `22_10/BatchLogger.jl`: Multi-threaded batch runner for paired trials across randomized obstacle environments; outputs `simulation_results.csv` (git-ignored due to size).
* `22_10/benign_baseline.jl`: Re-runs the exact same $1,000,000$ seeds from `simulation_results.csv` with the adversary disabled ($u(t) \equiv 0$) to produce `benign_results.csv` and verify Theorem 1's step bound across all runs.
* `22_10/Viewer.jl` & `22_10/main.jl`: Interactive GUI/visualization and entry point.
* `22_10/graph.py`: Generates publication plots (`success_rates.png`, `steps_distribution.png`). `simulation_trap.png` shows the representative double-failure scenario (Naive poisoned at $i_H = 1.230$; Inertial resonant-trapped at $i_H = 0.167$; Cautious succeeds in $137$ steps).

### LaTeX Manuscripts (in `22_10/`)
* `draft of Adversery tower I-Bug.tex` (compile with `pdflatex`): Full single-column academic paper (~9 pages).
* `draft_robust_ibug_short_cscml.tex` (compile with `pdflatex`): 2-column short conference paper in English (CSCML format).
* `draft_robust_ibug_short_cscml_he.tex` (compile with `xelatex`): 2-column short conference paper in Hebrew.
* `draft_robust_ibug_short_accessible_en.tex` (compile with `pdflatex`) & `accessible_paper_online_en.html`: Illustrated, accessible English edition (with notation glossary, TikZ diagrams, and live simulation GIFs `anomaly_filter_anim.gif` & `simulation_trap_anim.gif`).
* `draft_robust_ibug_short_accessible_he.tex` (compile with `xelatex`) & `accessible_paper_online.html`: Illustrated, accessible Hebrew edition with notation glossary, TikZ diagrams, and live simulation GIFs.

*(Note: `22_9/`, `projectB21_8/`, `final files/`, and `25_10/` are historical snapshots/backups; always edit in `22_10/`.)*

---

## 3. Summary of 1,000,000 Paired Simulation Results

| Strategy | Success Rate (95% CI) | Mean Steps (Succ.) | Median [IQR] (Succ.) | Mean Steps Capped (All Runs) |
| :--- | :---: | :---: | :---: | :---: |
| *Benign I-Bug (no attack)* | *100.00%* | *132.6* | *126.0 [108, 150]* | *132.6* |
| **Naive** (Baseline I-Bug) | 54.68% ± 0.10% | 199.3 | 153.0 [121, 205] | 1015.4 |
| **Cautious** (Proposed) | **100.00%** ([99.9996, 100]%) | 193.0 (max 578) | 178.0 [160, 215] | 193.0 |
| **Inertial** (Proposed) | 99.61% ± 0.01% (3,877 fails) | **153.4** (-20.5%) | **135.0** [115, 168] | **160.6** (-16.8%) |

