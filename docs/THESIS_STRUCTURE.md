# Thesis restructure — chapter split + reconciliation (2026-06-13)

Goal: pull **Testing** out of the "UML Design & Testing" chapter and make it stand
alone, add a separate **Results** chapter, and make the whole thesis describe the
*real* shipped system (supervised Normal-vs-Disease gate + B2+B3 ensemble), not the
old VAE / single-B2 story.

## New / edited chapter files

| File | Status | Becomes chapter |
|------|--------|-----------------|
| `Midterm_Chapter4_Implementation.tex` | EDITED (VAE→gate, B2→B2+B3, tables, numbers) | 4 — Implementation |
| `Midterm_Chapter4.tex` | EDITED — same fixes, **plus** has the "Selected Code Listings" appendix | 4 (+ appendix) |
| `Chapter_UML_Design.tex` | NEW — reconstructed from PDF, §5.5 testing removed | 5 — UML Design |
| `Chapter_Results.tex` | NEW — quantitative half of the old testing chapter | 6 — Results |
| `Chapter_Testing.tex` | NEW — app-testing half | 7 — Testing and Evaluation |
| `Chapter5_Testing.tex` | SUPERSEDED by the two files above — safe to delete | — |

> **Pick ONE Chapter 4 file.** `Midterm_Chapter4.tex` and
> `Midterm_Chapter4_Implementation.tex` are near-duplicates; both are now fixed and
> consistent. Use whichever your root `.tex` already `\input`s and delete/ignore the
> other so they don't drift apart again.

## Recommended chapter order (root .tex)

```latex
\input{Chapter1_Introduction}      % (your existing files — names may differ)
\input{Chapter2_LiteratureReview}
\input{Chapter3_Design}
\input{Midterm_Chapter4_Implementation}   % 4  Implementation
\input{Chapter_UML_Design}                % 5  UML Design
\input{Chapter_Results}                   % 6  Results
\input{Chapter_Testing}                   % 7  Testing and Evaluation
\input{Chapter6_Cost}                     % 8  Cost Analysis
\input{Chapter7_TimePlan}                 % 9  Time Plan
\input{Chapter8_Conclusions}              % 10 Conclusions
% \bibliography / References, then \appendix (Selected Code Listings, etc.)
```

## Still TODO (need you / a device)

1. **Send the root `.tex`** (the `\documentclass` + `\input` list) so the new
   chapters get wired in and chapter numbers/cross-refs resolve. Until then the new
   files compile only when `\input` into a document.
2. **UML diagram images** — `Chapter_UML_Design.tex` references
   `Figures/usecase_*.png`, `sequence_diagram.png`, `class_diagram.png` as
   placeholders. Point them at your real figure files, and ideally **regenerate the
   diagrams** so they show the *gate* (not "VAE / Anomaly Detection").
3. **Other chapters still mention the VAE as the shipped gate** — at minimum the
   **Abstract**, **Chapter 2 (Literature Review)**, and **Chapter 8 (Conclusions)**.
   Decide whether to reframe them as "explored then replaced" too (recommended, for
   consistency) — send those files and I'll do it.
4. **Device-only data** (cannot be done headless):
   - On-device `[TIMING]` numbers → fill `tab:inference_time` in `Chapter_Testing.tex`.
   - Score-CAM result screenshots → `Figures/scorecam_{acne,eczema,tinea}.png`.
5. **Related-work table** in `Chapter_Results.tex` still has placeholder citations —
   swap in the real studies from your literature review.
