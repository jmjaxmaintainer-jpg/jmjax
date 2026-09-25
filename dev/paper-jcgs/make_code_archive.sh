#!/bin/sh
# Build the JCGS supplementary code archive from a tagged release.
#   sh dev/paper-jcgs/make_code_archive.sh [TAG] [RESULTS_DIR]
# TAG defaults to v0.3.0; RESULTS_DIR (where the scripts wrote the result
# files kept outside the repository) defaults to ~/Documents/R/jmjax_results.
# Output: RESULTS_DIR/jcgs_code_archive_<TAG>_<timestamp>/jmjax-jcgs-code.zip
set -e
cd "$(git rev-parse --show-toplevel)"
TAG=${1:-v0.3.0}
RES=${2:-$HOME/Documents/R/jmjax_results}
VER=$(echo "$TAG" | sed 's/^v//')
OUT="$RES/jcgs_code_archive_${TAG}_$(date +%Y%m%d_%H%M%S)"
STAGE="$OUT/jmjax-jcgs-code"
mkdir -p "$STAGE/results"
# Package source and scripts at the tag; the manuscripts are submitted
# separately, so leave them out.
git archive --prefix="jmjax-$VER/" "$TAG" -- . \
  ':(exclude)dev/paper' ':(exclude)dev/paper-jcgs' | tar -x -C "$STAGE"
cp dev/paper-jcgs/code-archive-README.md "$STAGE/README.md"
for f in study_realdata_rotate.csv pilot_q1_intercept.csv; do
  cp "$RES/$f" "$STAGE/results/"
done
(cd "$OUT" && zip -qrX jmjax-jcgs-code.zip jmjax-jcgs-code)
ls -l "$OUT/jmjax-jcgs-code.zip"
