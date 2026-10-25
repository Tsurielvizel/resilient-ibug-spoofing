# Resilient I-Bug Navigation under Intermittent Signal Spoofing (`22_10` Active Workspace)

See the root [`../README.md`](../README.md) for the complete repository overview, architecture, mathematical guarantees, and simulation summary.

## Quick Commands (in `22_10/`)
* **Compile English papers:**
  ```powershell
  pdflatex -interaction=nonstopmode "draft of Adversery tower I-Bug.tex"
  pdflatex -interaction=nonstopmode "draft_robust_ibug_short_cscml.tex"
  pdflatex -interaction=nonstopmode "draft_robust_ibug_short_accessible_en.tex"
  ```
* **Compile Hebrew papers (requires `xelatex`):**
  ```powershell
  xelatex -interaction=nonstopmode "draft_robust_ibug_short_cscml_he.tex"
  xelatex -interaction=nonstopmode "draft_robust_ibug_short_accessible_he.tex"
  ```
* **Run Julia simulations:**
  ```powershell
  julia --project=. main.jl
  julia -t auto benign_baseline.jl
  ```

