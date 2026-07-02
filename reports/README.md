# Report

`report.tex` — the project write-up (Math and Machine Learning Praktikum,
University of Leipzig). Self-contained: all figures live in `figures/` and the
bibliography is `references.bib`.

## Build

No TeX toolchain is assumed in the dev environment. Two easy options:

**Overleaf** — upload `report.tex`, `references.bib`, and the `figures/` folder
(or the whole `reports/` directory). It compiles as-is with the default
pdfLaTeX + BibTeX setting.

**Local** — needs a TeX distribution (e.g. `brew install --cask mactex-no-gui`,
or BasicTeX + `tlmgr install siunitx natbib booktabs caption`):

```bash
cd reports
latexmk -pdf report.tex          # preferred, runs bibtex automatically
# or, manually:
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

## Figures

Generated figures are reproducible from the repo:

- `cf_v2_vs_v3_1e582bc0.png`, `far_side_by_model.png` — `python src/analysis/report_figures.py`
- `latent_embedding_v2.png` — `python src/analysis/latent_embedding.py`
- `eda_shot_heatmap.png`, `rasterize_example.png`, `v2_training_curves.png` —
  copied from `results/` and `models/checkpoints/baseline_v2/`.

## Status

First full draft covering v1 → v2 → v3/v3b, validation, and the latent-embedding
analysis. Sections most likely to want expansion before submission: Related Work,
and any module-required front matter (declaration of authorship, etc.).
