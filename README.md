# Resilient I-Bug Navigation under Intermittent Signal Spoofing

**Authors:** Tsuriel Vizel & Avraham Guez  
**Supervisor:** Dr. Andrey Zhitnikov  
*Faculty of Computer Science, Technion – Israel Institute of Technology*

---

## 1. Abstract & Project Overview

This repository provides the official implementation, empirical data analysis, and manuscripts for the study of minimalist intensity-based bug navigation (**I-Bug**) under **intermittent additive signal spoofing**:

$$S(p, t) = m(p - p_t) + u(t) \frac{\alpha_e}{\|p - p_e\|_2}, \quad m(p - p_t) = \frac{1}{\|p - p_t\|_2}, \quad u(t) \in \{0, 1\}$$

Without requiring hardware changes, odometry, or absolute GPS coordinates, we introduce a **kinematics-based temporal anomaly filter** based on the radial Lipschitz bound $\Delta I_{\max}$. We evaluate three navigation paradigms:

1. **Naive (`:Naive`)**: Unmodified I-Bug baseline. Severely vulnerable to gradient deflection and hit-threshold ($i_H$) poisoning.
2. **Cautious (`:Cautious`)**: Temporarily halts locomotion ($\Delta p_k = 0$) and memory updates upon detecting spoofing bursts ($\mathcal{A}_{\text{spike}}$), resuming only after detecting an attack drop ($\mathcal{A}_{\text{drop}}$) followed by $N_{\text{confirm}} = 3$ consecutive verified steps. **Theorem 1** guarantees step-by-step active trajectory invariance and conditional finite-time convergence:
   $$N_{\text{cautious}} \le N_{\text{ideal}} + \left\lceil \frac{N_{\text{ideal}}}{T_{\text{off}} - N_{\text{confirm}}} \right\rceil (T_{\text{on}} + N_{\text{confirm}})$$
3. **Inertial (`:Inertial`)**: Freezes sensor memory during attacks while continuing open-loop dead-reckoning motion along the boundary. **Proposition 1** establishes that memory freezing alone cannot guarantee convergence due to **topological phase-locking (resonance)** when $T_{\text{orbit}} = m \cdot T_{\text{cycle}}$.

---

## 2. Repository Structure

```text
.
├── src/                          # Core simulation and analysis source code
│   ├── CoreLogic.jl              # Simulation engine, kinematics, sensor model, and anomaly filter
│   ├── BatchLogger.jl            # High-throughput multi-threaded headless batch evaluation
│   ├── Viewer.jl                 # Interactive GLMakie visualization and simulation replay navigator
│   ├── main.jl                   # Primary executable entry-point
│   ├── benign_baseline.jl        # Verification of no-attack equivalence and theoretical step bounds
│   └── graph.py                  # Statistical analysis and figure generation
├── docs/                         # Manuscripts, technical reports, and visual demonstrations
│   ├── draft_robust_ibug_short_cscml.pdf          # 3-page English conference paper (CSCML format)
│   ├── draft of Adversery tower I-Bug.pdf         # Comprehensive academic technical manuscript
│   ├── draft_robust_ibug_short_accessible_en.pdf  # Illustrated edition with notation glossary and diagrams
│   ├── accessible_paper_online_en.html            # Standalone interactive web edition with animations
│   └── figures/                                   # High-resolution publication plots and simulation animations
└── .gitignore                    # Git tracking rules
```

---

## 3. Large-Scale Empirical Evaluation (1,000,000 Paired Mazes)

Evaluated across $1,000,000$ procedurally generated random obstacle mazes in paired trials ($N_{\max} = 2,000$ step horizon, $T_{\text{cycle}} = 40$, $T_{\text{on}} = 10$ with $25\%$ adversary duty cycle):

| Strategy | Success Rate (95% CI) | Mean Steps (Successful) | Median [IQR] (Successful) | Mean Steps Capped (All Runs) |
| :--- | :---: | :---: | :---: | :---: |
| *Benign I-Bug (no attack)* | *100.00%* | *132.6* | *126.0 [108, 150]* | *132.6* |
| **Naive** (Standard Baseline) | 54.68% ± 0.10% | 199.3 | 153.0 [121, 205] | 1015.4 |
| **Cautious** (Proposed) | **100.00%** ([99.9996, 100]%) | 193.0 (max 578) | 178.0 [160, 215] | 193.0 |
| **Inertial** (Proposed) | 99.61% ± 0.01% (3,877 fails) | **153.4** (-20.5%) | **135.0** [115, 168] | **160.6** (-16.8%) |

*Note: In the absence of an adversary ($u(t) \equiv 0$), the anomaly filter experiences exactly zero false positives (0 FP), and all three strategies produce identical trajectories ($100.00\%$ success rate, $132.6$ average steps).*

---

## 4. Getting Started

### Prerequisites
* [Julia](https://julialang.org/) (v1.9 or later)
* Required Julia packages: `GLMakie`, `DataFrames`, `CSV`

To install dependencies:
```julia
using Pkg
Pkg.add(["GLMakie", "DataFrames", "CSV"])
```

### Running the Interactive Simulation & Replay Viewer
To launch the interactive replay navigator (GLMakie GUI) from the repository root:
```bash
julia --project=. src/main.jl
```

### Running Batch Evaluations
To run headless batch simulations across random mazes:
```julia
include("src/CoreLogic.jl")
include("src/BatchLogger.jl")
run_batch_simulations(1000) # Evaluates 1,000 paired environments
```

---

## 5. Citation & Acknowledgments
If you use this benchmark, simulator, or defense algorithm in your research, please cite our technical paper located in `docs/draft_robust_ibug_short_cscml.pdf`.
