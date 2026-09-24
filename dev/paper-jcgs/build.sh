#!/bin/sh
# Build the JCGS paper and its supplement; each cross-references the other.
set -e
cd "$(dirname "$0")"
for i in 1 2; do
  pdflatex -interaction=nonstopmode jcgs-paper.tex >/dev/null || true
  pdflatex -interaction=nonstopmode jcgs-supplement.tex >/dev/null || true
  bibtex jcgs-paper >/dev/null || true
done
pdflatex -interaction=nonstopmode jcgs-paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode jcgs-supplement.tex >/dev/null || true
pdflatex -interaction=nonstopmode jcgs-paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode jcgs-supplement.tex >/dev/null || true
grep -H -E "^! |undefined|multiply defined" jcgs-paper.log jcgs-supplement.log || true
