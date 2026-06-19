# Local repetition-only training summary

Full Lichtheim2Model unchanged.
Task: repetition-only.
Dataset: 1200 English word items.
Device: local CPU.
Loss reduction: sum.
Seed: 0.

## Run A — 200 epochs, zero_error_radius=0.0

Output dir:
outputs/repetition_only/local_full_lr001_zer_0.0/20260617_202724

- lr: 0.01
- eval_radius: 0.1
- avg BCE: 30.3700 -> 17.9240
- phoneme_accuracy: 0.4490
- exact_match: 29/1200
- paper_like_output_word_acc: 0/1200
- mean_positive_output: 0.3232
- mean_negative_output: 0.0206
- output_positive_threshold_acc: 0.0464
- output_negative_threshold_acc: 0.9498

## Run B — 200 epochs, zero_error_radius=0.1

Output dir:
outputs/repetition_only/local_full_lr001_zer_0.1/20260617_213128

- lr: 0.01
- eval_radius: 0.1
- avg BCE: 23.9721 -> 13.9513
- phoneme_accuracy: 0.5127
- exact_match: 64/1200
- paper_like_output_word_acc: 0/1200
- mean_positive_output: 0.3627
- mean_negative_output: 0.0523
- output_positive_threshold_acc: 0.0707
- output_negative_threshold_acc: 0.8838

## Run C — 500 epochs, zero_error_radius=0.1

Output dir:
outputs/repetition_only/local_500_lr001_zer_0.1_seed0/20260617_223816

- lr: 0.01
- eval_radius: 0.1
- avg BCE: 23.9721 -> 11.8462
- phoneme_accuracy: 0.4823
- exact_match: 55/1200
- paper_like_output_word_acc: 0/1200
- mean_positive_output: 0.3642
- mean_negative_output: 0.0459
- output_positive_threshold_acc: 0.0712
- output_negative_threshold_acc: 0.8920

## Run D — 200 epochs, zero_error_radius=0.1, seed=1

Output dir:
outputs/repetition_only/local_full_lr001_zer_0.1_seed1/20260618_022902

- lr: 0.01
- eval_radius: 0.1
- avg BCE: 22.9952 -> 13.9873
- phoneme_accuracy: 0.5156
- exact_match: 64/1200
- paper_like_output_word_acc: 0/1200
- mean_positive_output: 0.3754
- mean_negative_output: 0.0545
- output_positive_threshold_acc: 0.0804
- output_negative_threshold_acc: 0.8615

## Compact results table

| Run | epochs | seed | zero_error_radius | BCE | phoneme_accuracy | exact_match | paper_like_output_word_acc | mean_positive_output | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | 200 | 0 | 0.0 | 30.3700 -> 17.9240 | 0.4490 | 29/1200 | 0/1200 | 0.3232 | Partial argmax learning, weak positives |
| B | 200 | 0 | 0.1 | 23.9721 -> 13.9513 | 0.5127 | 64/1200 | 0/1200 | 0.3627 | Best current run |
| C | 500 | 0 | 0.1 | 23.9721 -> 11.8462 | 0.4823 | 55/1200 | 0/1200 | 0.3642 | Lower BCE, no better repetition |
| D | 200 | 1 | 0.1 | 22.9952 -> 13.9873 | 0.5156 | 64/1200 | 0/1200 | 0.3754 | Reproducible seed check |

## Main message

The model learns partial repetition under argmax metrics, especially with `zero_error_radius=0.1`.
The result is reproducible across two seeds, both reaching around 51% phoneme argmax accuracy and 64/1200 exact-match words.
However, the candidate paper-like all-units-within-radius word accuracy remains 0/1200.
Positive motor activations remain too weak, around 0.36–0.38 on average, and only about 7–8% of positive target units exceed 0.9.

The 500-epoch run shows that BCE can keep decreasing without improving phoneme-level or word-level accuracy.
Therefore, loss alone is not sufficient for evaluating repetition learning.

## Open questions

- [Open #5] What is the exact role of the zero-error radius in the original implementation?
- [Open #9] Is our all-units-within-radius metric the right approximation of the paper’s Figure 2 scoring?
- [Open #2] Should motor silence be supervised during the whole input phase?
- [Open #6] How much does the English 39-dim one-hot representation differ from the paper’s Japanese 21-bit distributed mora representation?
- Should the next debugging step target the loss imbalance, the scoring convention, or a simpler dorsal-only repetition setup?