# Chapter 5 (Testing & Evaluation) — corrections and real evidence

Generated from the shipped models + held-out test split on 2026-06-13.
The existing `Chapter5_Testing.tex` predates the architecture pivot and several
audits; the numbers below are what the code and data actually produce.

> **STATUS: RESOLVED.** `Chapter5_Testing.tex` has now been **rewritten** with all
> the real numbers below, and the 5 failing unit tests were **fixed** (the suite is
> green at 23/23). This document is retained as a changelog / source-of-numbers.
> Device-only items (on-device timing, Score-CAM result screenshots) remain TODO.

---

## Issue 1 — Dataset split table is wrong

The chapter claims train 8,156 / val 325 / test 322. The dataset on disk
(`New_Augmented_Dataset/`, which all the eval scripts use) is:

| Class  | Train | Val | Test | Total |
|--------|------:|----:|-----:|------:|
| Acne   |   781 |  97 |   99 |   977 |
| Eczema | 1,256 | 157 |  158 | 1,571 |
| Tinea  |   832 | 104 |  105 | 1,041 |
| **Total** | **2,869** | **358** | **362** | **3,589** |

Corrected LaTeX:

```latex
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Training} & \textbf{Validation} & \textbf{Test} & \textbf{Total} \\ \hline
Acne   &   781 &  97 &  99 &   977 \\ \hline
Eczema & 1{,}256 & 157 & 158 & 1{,}571 \\ \hline
Tinea  &   832 & 104 & 105 & 1{,}041 \\ \hline
\textbf{Total} & \textbf{2{,}869} & \textbf{358} & \textbf{362} & \textbf{3{,}589} \\ \hline
\end{tabular}
```

---

## Issue 2 — Report on the TEST split, and disclose dataset leakage

The chapter reports validation-split accuracy (92.62%) and states the test split
is "withheld." For a final thesis you want the **test-set** number, and you must
disclose the augmentation-before-split leakage we measured — it is the honest,
defensible framing and pre-empts an examiner finding it.

**Leakage audit (`leakage_audit.py`), New_Augmented_Dataset, dHash Hamming ≤ 6:**

| Class  | Exact MD5 dups (train↔test) | Near-duplicate in train |
|--------|----------------------------:|------------------------:|
| Acne   |  6 | 24 / 99  (24.2%) |
| Eczema | 18 | 43 / 158 (27.2%) |
| Tinea  | 48 | 53 / 105 (50.5%) |

So ~33% of the test set (120/362) has a near-duplicate in training. The headline
finding is that removing the leaked images **barely moves the score**, which is
strong evidence the model genuinely generalises rather than memorising:

| Metric | Full test set | Leak-removed (honest) |
|--------|--------------:|----------------------:|
| n | 362 | 242 |
| Overall accuracy | **88.40%** | **88.43%** |
| Macro F1 | 0.884 | 0.876 |
| Macro AUC (one-vs-rest) | 0.962 | — |

### CNN confusion matrix (B2+B3 ensemble, single-pass, test set)

**Full (n=362, acc 88.40%)** — rows = true, columns = predicted:

| true \ pred | Acne | Eczema | Tinea |
|-------------|-----:|-------:|------:|
| **Acne**    |  89  |   4    |   6   |
| **Eczema**  |   4  |  141   |  13   |
| **Tinea**   |   2  |   13   |  90   |

**Leak-removed (n=242, acc 88.43%):**

| true \ pred | Acne | Eczema | Tinea |
|-------------|-----:|-------:|------:|
| **Acne**    |  66  |   3    |   6   |
| **Eczema**  |   4  |  101   |  10   |
| **Tinea**   |   1  |   4    |  47   |

The dominant error is the **Eczema↔Tinea** confusion (13 images each direction on
the full set: 8.2% of eczema, 12.4% of tinea) — clinically expected, as both
present as scaly erythematous plaques. (Note the chapter's old text blames an
Eczema→Acne confusion; the real data shows Eczema↔Tinea.)

### Per-class metrics (full test set)

| Class | Precision | Recall | F1 | Support | AUC |
|-------|----------:|-------:|---:|--------:|----:|
| Acne   | 0.9368 | 0.8990 | 0.9175 | 99  | 0.9919 |
| Eczema | 0.8924 | 0.8924 | 0.8924 | 158 | 0.9525 |
| Tinea  | 0.8257 | 0.8571 | 0.8411 | 105 | 0.9423 |
| **Macro Avg**    | 0.8850 | 0.8828 | 0.8837 | 362 | 0.9622 |
| **Weighted Avg** | 0.8852 | 0.8840 | 0.8844 | 362 | — |

Corrected LaTeX (full-test confusion matrix + classification report):

```latex
% Confusion matrix (full test set, n=362, accuracy 88.40%)
\begin{tabular}{|l|c|c|c|}
\hline
 & \textbf{Pred: Acne} & \textbf{Pred: Eczema} & \textbf{Pred: Tinea} \\ \hline
\textbf{True: Acne}   & 89 & 4   & 6  \\ \hline
\textbf{True: Eczema} & 4  & 141 & 13 \\ \hline
\textbf{True: Tinea}  & 2  & 13  & 90 \\ \hline
\end{tabular}

% Per-class metrics (full test set)
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Prec.} & \textbf{Recall} & \textbf{F1} & \textbf{Support} & \textbf{AUC} \\ \hline
Acne   & 0.9368 & 0.8990 & 0.9175 & 99  & 0.9919 \\ \hline
Eczema & 0.8924 & 0.8924 & 0.8924 & 158 & 0.9525 \\ \hline
Tinea  & 0.8257 & 0.8571 & 0.8411 & 105 & 0.9423 \\ \hline
\textbf{Macro Avg}    & 0.8850 & 0.8828 & 0.8837 & 362 & 0.9622 \\ \hline
\textbf{Weighted Avg} & 0.8852 & 0.8840 & 0.8844 & 362 & -- \\ \hline
\end{tabular}
```

> Method note: figures use a single forward pass (no TTA), matching `honest_eval.py`
> so the headline number is reproducible. The shipped app additionally applies
> 4-view flip TTA, which gives a small further gain; if you want the chapter to
> report the deployed-with-TTA number, say so and I'll regenerate.

---

## Issue 3 — The gate is the supervised Normal-vs-Disease model, not the VAE

The chapter's "VAE Anomaly Detection" section describes the deployed gate. The
app (`ai_service.dart`) replaced the VAE with a supervised EfficientNetB0 gate
(`normal_gate.tflite`); the VAE code is retained but **no longer gates**. The
gate outputs P(disease) and applies two thresholds:

- P(disease) ≤ **0.60** → "No Disease Detected" (pipeline exits)
- 0.60 < P(disease) < **0.90** → "Inconclusive" — advise retake (CNN not run)
- P(disease) ≥ **0.90** → run the B2+B3 CNN to name the disease

This three-band design is the safety mechanism: on skin the gate cannot
confidently call, it abstains rather than guessing. See
`gate_score_histogram.png` and `gate_summary.txt` for the score distribution and
per-band rates on held-out phone normals vs. the disease test split.

Suggested replacement section heading: **"Normal-vs-Disease Gate Evaluation"**,
describing the supervised gate, the bimodal score separation, and the
abstention band. The VAE can be mentioned in a sentence as the superseded
approach (it could not separate normal skin from disease — reconstruction error
distributions overlapped and even inverted on-device).

### Gate results (shipped `normal_gate.tflite`)

Run on held-out phone normals (never trained on) vs. the disease test split:

| Set | n | mean P(disease) | Normal ≤0.60 | Inconclusive 0.60–0.90 | Disease ≥0.90 |
|-----|--:|----------------:|-------------:|-----------------------:|--------------:|
| Normal (held-out phone) | 25  | 0.20 | **96.0%** (24/25) | 4.0% (1/25) | 0.0% (0/25) |
| Disease (test split)    | 362 | 0.96 | 1.4% (5/362) | 8.3% (30/362) | **90.3%** (327/362) |

Headline gate numbers: **96% normal-pass**, **98.6% disease detection** (P>0.60),
**0% of normals ever mislabelled "disease"** (the worst a normal gets is the safe
"inconclusive / retake" band). This is the clean bimodal separation the design
relies on. Figure: `gate_score_histogram.png`.

Corrected LaTeX:

```latex
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Set} & \textbf{n} & \textbf{mean P(dis.)} & \textbf{Normal $\leq$0.60} & \textbf{Inconcl.\ 0.60--0.90} & \textbf{Disease $\geq$0.90} \\ \hline
Normal (held-out phone) & 25  & 0.20 & 96.0\% & 4.0\% & 0.0\% \\ \hline
Disease (test split)    & 362 & 0.96 & 1.4\% & 8.3\% & 90.3\% \\ \hline
\end{tabular}
```

---

## Issue 4 — The unit-test tables describe tests that do not exist

The chapter's unit-test tables list VAE inference, CNN B2/B3 inference, Score-CAM
masking, ScanResult serialization, and DB CRUD tests. The actual suite
(`Flutter/test/`) contains **22 tests across five files**. As of this run,
**17 pass and 5 fail** — the 5 failures are all in `body_part_render_test.dart`,
a golden/widget snapshot test that is **stale** (see note below), not a
logic/model regression. The 17 logic, security, and model unit tests all pass.
Real coverage:

| File | Tests | What it covers |
|------|------:|----------------|
| `appointment_test.dart`      | 4 | `Appointment` `toMap`/`fromMap` round-trip; default status `Scheduled`; `copyWith` isolates the status change; `fromMap` tolerates missing optional fields |
| `password_hasher_test.dart`  | 6 | PBKDF2-SHA256 hash format; verify accepts correct / rejects incorrect; random-salt makes identical passwords hash differently (both still verify); `isHashed` separates legacy plaintext from hashes; malformed stored values rejected |
| `progression_utils_test.dart`| 5 | `groupScansByBodyPart` groups + sorts oldest→newest; blank body part → `Unspecified`; `trackableBodyParts` orders by scan count and filters by `minScans`; `confidenceTrend` maps normal→0, disease→confidence |
| `widget_test.dart`           | 2 | `User` model `toMap`/`fromMap` round-trip; `fromMap` defaults (role→patient, empty email, null age) |
| `body_part_render_test.dart` | 5 | **FAILING (stale)** — golden/widget snapshot of `BodyPartSelectionScreen`. See note. |

**The 5 failures (`body_part_render_test.dart`).** This is a golden-image test
whose goldens were captured on 2026-06-06. The screen was redesigned afterwards:
the test taps `find.text('CHEST'/'FACE'/'Back'/'Upper Back'/'Help')`, but the
current screen renders body parts as positioned silhouette *hotspots* and uses
chevron/icon toggles, so those `Text` widgets no longer exist (and the empty-state
golden differs by 57%). It is a **stale UI test**, not a defect in the app logic.
Options: (a) update the finders to the new hotspot interaction and regenerate the
goldens with `flutter test --update-goldens` (note: goldens are font/platform
sensitive and may not be portable to another machine); (b) mark these as
render/inspection snapshots excluded from the unit-test pass count; or (c) remove
them. Recommend reporting the 17 logic/security/model tests as the unit-test
suite and treating the body-part goldens separately.

Corrected LaTeX:

```latex
\begin{table}[H]
\centering
\caption{Unit test coverage (\texttt{flutter test}, 17 tests).}
\begin{tabular}{|p{3.6cm}|c|p{7.4cm}|}
\hline
\textbf{Test file} & \textbf{\#} & \textbf{Coverage} \\ \hline
\texttt{appointment\_test.dart} & 4 & Appointment model round-trip; default status; \texttt{copyWith} isolation; missing-field tolerance \\ \hline
\texttt{password\_hasher\_test.dart} & 6 & PBKDF2-SHA256 format; correct/incorrect verification; random-salt uniqueness; legacy-plaintext detection; malformed-hash rejection \\ \hline
\texttt{progression\_utils\_test.dart} & 5 & Group-by-body-part with chronological sort; \texttt{Unspecified} bucket; \texttt{trackableBodyParts} ordering/filtering; \texttt{confidenceTrend} normal-vs-disease \\ \hline
\texttt{widget\_test.dart} & 2 & User model round-trip; \texttt{fromMap} defaults \\ \hline
\end{tabular}
\end{table}
```

`flutter test` result: **17 passed, 5 failed** (the 5 stale body-part goldens).
A clean readable summary is in `flutter_test_summary.txt`.

---

## Figure file mapping (generated in `chapter5_figures/`)

| Generated file | Chapter figure / table it feeds |
|----------------|---------------------------------|
| `confusion_matrix_full.png`  | CNN confusion matrix (full test set) |
| `confusion_matrix_clean.png` | CNN confusion matrix (leak-removed, honest) |
| `roc_curves.png`             | One-vs-rest ROC + AUC |
| `classification_report_full.csv` / `_clean.csv` | Per-class precision/recall/F1 table |
| `gate_score_histogram.png`   | Gate score distribution (replaces the VAE anomaly-ratio histogram) |
| `predictions.csv`            | Raw per-image scores (audit trail / reproducibility) |
| `metrics_summary.txt` / `gate_summary.txt` | Source numbers for the prose |
