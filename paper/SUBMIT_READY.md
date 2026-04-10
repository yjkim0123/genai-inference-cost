# JSA Submission Readiness Report

## Target Journal
- **Journal:** Journal of Systems Architecture (JSA)
- **Publisher:** Elsevier
- **SCIE / IF:** Yes / 5.6
- **Open Access:** Non-OA (free submission)
- **Submission portal:** https://www.editorialmanager.com/jsa/

## Output Files
| File | Description |
|------|-------------|
| `main_jsa.tex` | elsarticle (review, 12pt, authoryear) formatted manuscript |
| `main_jsa.pdf` | Compiled PDF (40 pages, review double-spaced with line numbers) |
| `refs.bib` | Bibliography (shared with IEEE version) |

## Author Information
- **Yongjun Kim** (sole author, corresponding author)
- Department of Software, Ajou University, Suwon 16499, South Korea
- Email: yjkim0123@ajou.ac.kr
- ORCID: 0000-0003-4234-4883

## Changes from IEEE Access Format → JSA (elsarticle)

### Document Class & Structure
- [x] `\documentclass{ieeeaccess}` → `\documentclass[review,12pt,authoryear]{elsarticle}`
- [x] `\begin{frontmatter}...\end{frontmatter}` wrapping title/abstract/keywords
- [x] Line numbers enabled (`\linenumber`)
- [x] `\journal{Journal of Systems Architecture}` set

### Author Block
- [x] Sole author: Yongjun Kim (removed Eun Jung Sun as co-author per instructions)
- [x] `\author[ajou]{...}` with `\corref`, `\fnref` for ORCID
- [x] `\ead{yjkim0123@ajou.ac.kr}`
- [x] `\affiliation[ajou]{...}` with structured organization/city/postcode/country

### Citation Style
- [x] `\cite{...}` → `\citep{...}` / `\citet{...}` (natbib authoryear)
- [x] `\bibliographystyle{IEEEtran}` → `\bibliographystyle{elsarticle-harv}`
- [x] All inline `Author et al.~\cite{...}` → `\citet{...}` for textual citations

### IEEE-Specific Removals
- [x] Removed `\usepackage{cite}` (elsarticle uses natbib)
- [x] Removed `\usepackage{balance}`
- [x] Removed `\markboth{...}`
- [x] Removed `\begin{IEEEkeywords}...\end{IEEEkeywords}` → `\begin{keyword}...\sep...\end{keyword}`
- [x] Removed `\maketitle` (handled by frontmatter)
- [x] Removed all `\begin{IEEEbiography}...\end{IEEEbiography}` blocks
- [x] Removed `ieeeaccess.cls` dependency

### Elsevier-Required Sections
- [x] **Highlights** (5 items in `\begin{highlights}...\end{highlights}`)
- [x] **Keywords** (6 keywords, `\sep`-separated)
- [x] **CRediT authorship contribution statement**
- [x] **Declaration of competing interest**
- [x] **Data availability**
- [x] **Acknowledgments**

### Compilation
- [x] pdflatex → bibtex → pdflatex → pdflatex → pdflatex (4 passes, all clean)
- [x] 0 errors, 1 bibtex warning (empty pages in salimans2022progressive — cosmetic only)
- [x] All cross-references resolved
- [x] All `\input{../results/tex/...}` values rendered correctly

## Highlights (as included)
1. A unified measurement dataset of 4,133 energy records across 7 generative AI models and 3 hardware platforms is constructed
2. SVR with log-transformed features achieves R² = 0.923 for within-platform inference energy prediction using only 8 design-time features
3. Leave-one-hardware-out cross-validation reveals a fundamental generalization gap (R² < 0.1) for cross-platform prediction
4. SHAP analysis uncovers that autoregressive models are hardware-bound while diffusion models are workload-bound
5. Practical design guidelines and an analytical cost formula are provided for hardware-aware inference cost estimation

## Keywords
Energy modeling, Generative AI, Inference cost prediction, Hardware-aware design, Cross-platform estimation, SHAP analysis

## Pre-Submission Checklist
- [x] elsarticle document class with review option
- [x] Sole author with complete affiliation and ORCID
- [x] Abstract within limits
- [x] Highlights (5 items, each < 85 characters)
- [x] Keywords provided
- [x] Figures as PDF (via `../figures/*.pdf`)
- [x] Tables compile cleanly (via `\input{../results/tex/*.tex}`)
- [x] CRediT statement included
- [x] COI declaration included
- [x] Data availability statement included
- [x] Bibliography uses elsarticle-harv style (natbib authoryear)
- [x] Line numbers enabled for review
- [x] No IEEE-specific commands remaining

## Notes for Submission
1. Upload `main_jsa.tex` + `refs.bib` + figures folder + results/tex folder to Editorial Manager
2. Alternatively, upload the compiled `main_jsa.pdf` directly
3. The cover letter from the IEEE submission may need updating for JSA
4. Consider whether the title needs adjustment for JSA's systems architecture scope
