# Bias & Robustness Audit — Urdu Code-Switch Emotion Detector

This report documents 7 evaluations of the fine-tuned XLM-R emotion classifier
on held-out test data (2931 rows). The model is not retrained or
modified anywhere in this audit — it is evaluated and documented only.

**Baseline reference (test set):** accuracy 0.6285, macro F1 0.4305.
All findings below report deltas or patterns relative to this baseline.

## Executive Summary

Across 7 audits, the model shows the clearest weaknesses on Roman Urdu negation
handling, rare-class reliability (Fear, Surprise), and cross-language consistency
for ambiguous emotions. It is reasonably stable under synthetic text noise and
shows only a modest (10%) sensitivity to gender-term swaps on neutral sentences.
Confidence calibration is weak: 23.0% of all errors are
high-confidence (>0.8), meaning the model is often wrong while appearing certain.

## Code-Switch Findings

On this test set, only the High (>50% English-word-ratio) group had a usable
sample size (n=2931); Low and Medium groups had no
test examples. Macro F1 for the High group was 0.431
versus the overall baseline of 0.431. Because no Low or Medium group exists
in this test split, this audit cannot establish whether heavier code-mixing
helps or hurts performance — the dataset is simply too dominated by Roman Urdu
(per NB02's finding of 89.5% mostly-Roman-Urdu sentences) for a 3-way comparison.

## Gender Findings

On a sample of 50 neutral test sentences containing a swappable
gender term, swapping the term changed the model's prediction in 10.0% of cases,
with an average confidence shift of 0.0509. Since all base sentences were
neutral, the fair/expected result is 0% change — gendered terms should not
carry emotional valence alone. A 10% change rate is a real, non-trivial finding,
not noise to explain away, though it is a Medium-severity issue by the stated
threshold (5–15%) rather than a High-severity one.

## Keyword Dependency Findings

On 6 hand-written negation pairs spanning 5 emotion classes, the model correctly
changed its prediction under negation in 50.0% of pairs (3/6) — termed the
"negation pass rate." The pairs that failed (anger, fear, surprise) were all
Roman Urdu phrasings; the pairs that passed were predominantly English or
simpler constructions. This is a small, illustrative sample (n=6), but the
pattern suggests the model may rely more on surface keyword presence than
sentence-level negation specifically when the negation is expressed in Roman
Urdu rather than English.

## Rare-Class Findings

Fear had 33 test examples, with 28 misclassified
(84.8%); its most frequent wrong prediction was
fear → neutral. Surprise had 31 test examples, with
18 misclassified (58.1%);
its most frequent wrong prediction was surprise → neutral.
Reading the 10 highest-confidence failures for each class (saved in
rare_class_failures.csv), several Fear errors read as anxious complaint or
concern rather than clear-cut fear (e.g. requests for help, frustration with
policy/lockdown rules), and several Surprise errors read as sarcasm or
rhetorical complaint rather than genuine surprise. This suggests at least part
of the rare-class error rate reflects genuine ambiguity in how these emotions
are expressed in this corpus, not purely a model weakness — though with
33 and 31 test examples respectively, both classes' metrics
should be treated as imprecise estimates regardless of cause.

## OOD Findings

On 3 hand-written semantic triplets (same emotion expressed in pure English,
pure Roman Urdu, and code-switched form), only 1 of 3 triplets received a
consistent predicted label across all three language variants. This is a
small, illustrative sample, not a statistically powered claim about the
model's general language robustness — but within this small sample, the model
showed clear inconsistency on Anger and Sad triplets specifically, with
predictions shifting across neutral, sad, happy, and surprise depending on
which language the same underlying emotion was expressed in.

## Noise Robustness Findings

On synthetic noise variants (letter elongation, extra punctuation, vowel-drop
typos) generated from 5 base sentences, 80.0% of predictions
remained unchanged from the clean original. Note that this dataset's raw text
already contains organic spelling variation (mixed casing, abbreviated
transliteration, per NB02), so this stability rate measures robustness to
noise *beyond* what training data implicitly already contained, not robustness
to a wholly novel phenomenon.

## Reliability (Confidence Calibration) Findings

Mean confidence on correct predictions was 0.7370 versus
0.6418 on incorrect predictions, a gap of
0.0953. 23.0% of
all 1089 test-set errors were high-confidence (>0.8) mistakes. Per
the stated threshold (High if >20%), the model is poorly calibrated — it is
often confidently wrong, not just occasionally wrong, which matters directly
for any deployed setting where confidence scores might be shown to end users
or used to gate human review.

## Overall Limitations

This model performs adequately on its majority classes (Neutral, Happy, Anger)
but should not be trusted at face value on Fear or Surprise, where both data
volume and apparent label ambiguity limit reliability. Negation handling is
inconsistent, particularly in Roman Urdu phrasing. Confidence scores from this
model should not be treated as a reliable proxy for correctness given the
23.0% high-confidence error rate — any production use
should pair predictions with human review for Fear/Surprise outputs and avoid
surfacing raw confidence scores as a certainty signal to end users.
