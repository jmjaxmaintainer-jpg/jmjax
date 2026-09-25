#!/bin/sh
# Assemble and build the arXiv version from the JCGS sources.
#   sh dev/paper-arxiv/build.sh
# Output: dev/paper-arxiv/build/arxiv-paper.pdf and
#         dev/paper-arxiv/arxiv-source.zip (flat folder for arXiv upload,
#         with the .bbl, since arXiv does not run BibTeX reliably).
set -e
cd "$(dirname "$0")"
J=../paper-jcgs
mkdir -p build
cp arxiv-paper.tex build/
for f in sec0-abstract sec1-intro sec2-model sec3-absorbable sec4-rotation \
         sec5-blocked sec6-simulation sec7-generality sec8-realdata \
         sec9-discussion declarations appA-proofs supp-body; do
  cp "$J/$f.tex" build/
done
cp "$J/refs.bib" "$J/agsm.bst" build/
cp ../figures/fig1_ridge.pdf ../figures/fig2_gain.pdf ../figures/fig3_alpha.pdf build/
cd build
pdflatex -interaction=nonstopmode arxiv-paper.tex >/dev/null || true
bibtex arxiv-paper >/dev/null || true
pdflatex -interaction=nonstopmode arxiv-paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode arxiv-paper.tex >/dev/null || true
grep -E "^! |undefined|Overfull" arxiv-paper.log || true
zip -qX ../arxiv-source.zip arxiv-paper.tex arxiv-paper.bbl agsm.bst \
  sec*.tex declarations.tex appA-proofs.tex supp-body.tex \
  fig1_ridge.pdf fig2_gain.pdf fig3_alpha.pdf
