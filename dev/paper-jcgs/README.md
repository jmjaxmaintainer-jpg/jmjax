# Methods paper: JCGS submission version

This folder is the Journal of Computational and Graphical Statistics version
of the methods paper. It uses the journal's LaTeX template: 12pt, spacing
1.8, `agsm` references, and an `\anon` switch for the anonymized version.
The reviewed general version is in `dev/paper/`, frozen at git tag
`methods-paper-v1`.

## Build

    sh dev/paper-jcgs/build.sh

This builds `jcgs-paper.pdf` and `jcgs-supplement.pdf`, and runs each twice
because the two documents cross-reference each other (`xr-hyper`). Figures
are read from `dev/figures/`. Build on a machine with Latin Modern
(`lmodern`), as the template expects. Without it, the preamble falls back
to Computer Modern.

## Differences from `dev/paper/`

- **Abstract:** cut to 199 words (JCGS limit 200). The keywords were
  replaced with ones that do not appear in the title, as the template asks.
- **Moved to the supplement:** the detail listed below. The main text keeps
  a summary of each, with a reference to the supplement section.
  - the Gaussian check of Section 5 (S1);
  - the random-intercept-only runs to convergence (S2);
  - the stress-grid table (S3);
  - the calibration table and the α bias study (S4);
  - the real-data random-intercept-only fits (S5);
  - the reproducibility table, formerly Appendix B (S6).
- **Text:** tightened throughout, with no change to any number or claim.
  An independent check compared the two versions number by number.
- **Added:** disclosure statement, generative-AI declaration, data
  availability statement and supplementary material list
  (`declarations.tex`), plus references for LKJ and HSAUR3.

## Before submission

- [ ] Replace AFFILIATION and FUNDING on the title page (`jcgs-paper.tex`).
- [ ] Confirm the disclosure statement and edit the generative-AI
      declaration (`declarations.tex`).
- [ ] Build the code archive: the jmjax package source, plus the dev
      scripts and result files named in Section S6.
- [ ] Upload the PDF, then the LaTeX sources as one zip: the `.tex` files,
      `refs.bib`, `jcgs-paper.bbl`, `agsm.bst` and the figures. Upload the
      supplement PDF and the code archive as supplementary files.
- [ ] Anonymized version, only if choosing double-anonymous review: set
      `\anon` to 0. The name `jmjax` would still identify the author.
