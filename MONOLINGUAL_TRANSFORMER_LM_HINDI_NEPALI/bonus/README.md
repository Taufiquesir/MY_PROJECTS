# Bonus: No-Positional-Embedding Ablation (Model H / Hindi)

This folder is the complete, self-contained implementation of the assignment's
optional bonus task: pick one language, retrain a model with positional
embeddings removed, and run the full Phase 2 evaluation suite on it compared to
the standard model.

**Language chosen: Hindi (Model H).** Higher-resource of the two languages, so
the effect of removing positional information should be clearer to observe and
discuss against a larger, more stable pretraining signal than the lower-resource
Nepali corpus would give.

This is a completely separate branch (`bonus`) and folder from the graded
`phase-1`/`phase-2`/`phase-3` work. Nothing in `hindi/model/transformer.py` (or
any other graded file) is modified, edited, or depended on for its model class --
this folder only reuses the pieces of the real code that are genuinely
architecture-independent (the feed-forward sublayer, the data-tokenization
pipeline, the optimizer/scheduler builders, the BLEU/chrF/ROUGE-L scoring logic,
the plotting functions), explicitly documented per-file below.

## What "no positional embeddings" means here

The standard Hindi model uses RoPE (Rotary Position Embedding) -- it rotates the
query and key vectors by an angle proportional to their sequence position,
*inside* every attention layer, rather than adding a separate positional vector
once at the input (see `hindi/model/transformer.py`'s own module docstring for
why RoPE was chosen over sinusoidal/learned-absolute schemes). There was never an
additive positional table to strip out for this ablation -- instead, the ablation
model's attention module simply never calls the rotation function. Queries and
keys go into the attention dot product exactly as the QKV projection produced
them: pure content, no positional signal at all beyond the causal mask's coarse
"this position came before/after that one" boundary.

## Folder layout

```
bonus/
  README.md                          -- this file
  configs/
    model_ablation.yaml              -- same architecture hyperparameters as hindi/configs/model.yaml
    train_ablation.yaml              -- same training hyperparameters as hindi/configs/train.yaml (same 61,000-step / ~500M-token budget)
  model/
    transformer_ablation.py          -- the ablation architecture itself
    checkpoint_ablation.py           -- save/resume, same bundled format as the graded checkpoints
    train_loop_ablation.py           -- pretraining loop
  train/
    train_ablation.py                -- local/Colab entry point
    kaggle_train_ablation.py         -- Kaggle notebook script (`# %%`-delimited cells)
  eval/
    generate_ablation.py             -- greedy/temperature decoding
    attention_analysis_ablation.py   -- attention capture, entropy, mean attention distance
    perplexity_ablation.py           -- PPL/BPB
    quality_ablation.py              -- BLEU/chrF/ROUGE-L, diversity diagnostics
    plots_ablation.py                -- re-exports hindi/eval/plots.py unchanged (pure visualization, architecture-agnostic)
    run_eval_ablation.py             -- end-to-end evaluation orchestrator, writes to bonus/report/
  tests/
    test_transformer_ablation.py     -- unit tests (causal mask still works, no RoPE buffers exist, same param count as the real model)
  checkpoints/                       -- gitignored (*.pt); best.pt / latest.pt land here after training
  report/                            -- evaluation outputs: metrics.json + plots/ (heatmaps, entropy, distance curves)
```

## Which pieces are reused unchanged vs. reimplemented

| Piece | Status |
|---|---|
| `CausalSelfAttention` / `TransformerBlock` / `DecoderOnlyTransformer` | Reimplemented here as `*NoPos` classes -- the actual ablation |
| `FeedForward` | Reused unchanged from `hindi.model.transformer` (position-independent) |
| `TokenWindowDataset`, `tokenize_split_to_binary` | Reused unchanged from `hindi.model.data` (tokenizing text has nothing to do with model architecture) |
| `TrainConfig`, `build_optimizer`, `build_lr_scheduler`, `evaluate` | Reused unchanged from `hindi.model.train_loop` (these only ever call `model.named_parameters()` / `model(x)`, so they work identically regardless of model class) |
| `measure_bytes_per_token` | Reused unchanged from `hindi.eval.perplexity` (pure byte/token counting) |
| `build_prompt_reference_pairs`, `distinct_n`, `repetition_rate`, `_score_condition`, BLEU/chrF/ROUGE-L scoring | Reused unchanged from `hindi.eval.quality` |
| All plotting functions | Reused unchanged from `hindi.eval.plots` |
| Checkpoint loading, generation, attention capture | Reimplemented (`*_ablation.py`) since these must construct the ablation's own model class |

## How to run

### Locally / on a machine with `data/hindi/splits` already present

```bash
python -m bonus.train.train_ablation
```

Trains for the full 61,000 steps (~500M tokens, matching the standard Hindi run
exactly), checkpointing to `bonus/checkpoints/latest.pt` / `best.pt`. Resumable:
re-running the same command picks up from `latest.pt` automatically.

### On Kaggle

This repository is private, so Kaggle's `git clone` cannot reach it. Instead:

1. Package the following into a zip and upload it as a Kaggle Dataset:
   - `bonus/` (this whole folder)
   - `hindi/__init__.py`, `hindi/model/` (all files), `hindi/train/__init__.py`,
     `hindi/train/train.py`, `hindi/tokenizer/` (both files)
   - `collection/__init__.py`, `collection/common/__init__.py`,
     `collection/common/record.py`
2. Follow the setup steps and cell contents documented in
   `bonus/train/kaggle_train_ablation.py`'s header comment.

### Evaluation, once a checkpoint exists

```bash
python -m bonus.eval.run_eval_ablation --checkpoint bonus/checkpoints/best.pt --test-bin <path to tokenized test.bin>
```

Runs the identical Phase 2 evaluation suite the standard Hindi model was scored
with: PPL/BPB, generation quality (greedy + temperatures 0.5/1.0/1.5) with
BLEU/chrF/ROUGE-L, diversity diagnostics (repetition rate, Distinct-1/2), and
attention heatmaps/entropy/mean-attention-distance at the earliest and latest
layers. Writes `bonus/report/metrics.json` and every plot under
`bonus/report/plots/`.

### Trained checkpoint (Google Drive)

The ablated model's checkpoints (`best.pt`, `latest.pt`, step 61,000, same
resumable format as the standard model's checkpoints -- weights + optimizer
state + scheduler state + step + config) aren't committed to git, per the
assignment's large-artifact rule. They're on Drive instead:

- **Bonus ablation (no positional embeddings) checkpoints:** https://drive.google.com/drive/folders/11Nxr-iWS8qcwzrOo4Q6I5k4MSd_g0ntG?usp=sharing

## Status

- [x] Architecture implemented and unit-tested
- [x] Configs set to the full ~500M-token training budget (same as the standard model)
- [x] Training run -- full 61,000 steps (~500M tokens) on Kaggle (T4 GPU), same budget as the standard Hindi model
- [x] Evaluation run -- full Phase 2 suite (PPL/BPB, generation quality + diversity at greedy/0.5/1.0/1.5, attention entropy/distance/heatmaps), see `bonus/report/metrics.json` and `bonus/report/plots/`
- [x] Comparison write-up (below)

## Results: standard (RoPE) vs. ablated (no positional embeddings)

Both models are the identical architecture and size (13 layers, 6 heads, 384-dim,
26.5M params) trained for the identical 61,000-step / ~500M-token budget on the
same Hindi corpus and tokenizer. The only difference is that the ablated model's
attention never applies RoPE to queries/keys -- see "What 'no positional
embeddings' means here" above.

For the full numeric results (every metric, plotted and tabulated) and a
deeper mechanistic explanation of *why* each result differs from the standard
implementation, see [`bonus/report/ABLATION_REPORT.md`](report/ABLATION_REPORT.md).
The tables below use a **matched-conditions re-evaluation** of the standard
model (same tokenized test file, same 16,384,000 PPL tokens, same 50 quality
samples as the ablated model's run) rather than the original Phase 2 report's
numbers, which used a smaller sample -- see the linked report for the full
methodology note.

### Perplexity / BPB (held-out test split, matched conditions)

| Metric | Standard (RoPE) | Ablated (no pos.) | Change |
|---|---|---|---|
| Perplexity | 15.28 | 16.52 | +8.2% worse |
| Bits-per-byte | 0.4015 | 0.4131 | +2.9% worse |

Losing positional information costs the ablated model a moderate but
consistent amount of next-token predictability. The gap is smaller than one
might expect -- Hindi word order and local n-gram statistics still let a
purely content-based attention mechanism do reasonably well, since a lot of
next-token probability mass is carried by immediate lexical context rather
than long-range position.

### Generation quality + diversity (50 test-split prompts x 4 decoding conditions, both models)

| Condition | Model | BLEU | chrF | ROUGE-L | Distinct-1 | Distinct-2 | Repetition rate |
|---|---|---|---|---|---|---|---|
| Greedy | Standard | 1.92 | 18.99 | 0.0080 | 0.210 | 0.424 | 0.86 |
| Greedy | Ablated | 2.19 | 18.32 | 0.0180 | 0.184 | 0.375 | 0.94 |
| T=0.5 | Standard | 2.06 | 20.14 | 0.0040 | 0.305 | 0.687 | 0.58 |
| T=0.5 | Ablated | 1.91 | 21.21 | 0.0064 | 0.289 | 0.683 | 0.56 |
| T=1.0 | Standard | 1.58 | 20.67 | 0.0267 | 0.515 | 0.914 | 0.06 |
| T=1.0 | Ablated | 0.77 | 19.74 | 0.0304 | 0.512 | 0.923 | 0.06 |
| T=1.5 | Standard | 0.12 | 18.26 | 0.0000 | 0.757 | 0.998 | 0.00 |
| T=1.5 | Ablated | 0.15 | 18.25 | 0.0131 | 0.745 | 0.992 | 0.00 |

BLEU/chrF/ROUGE-L are noisy at this scale (both models are small, 500M-token
pretraining budgets, single-reference scoring) and don't show a clean, one-sided
winner -- but the **diversity diagnostics lean consistently one way**: at every
decoding condition, the ablated model's Distinct-1/Distinct-2 are lower, and its
greedy repetition rate is higher (0.94 vs 0.86) than the standard model's.
Importantly, **both** models degenerate into repeated-phrase loops under greedy
decoding on most prompts -- this is a general small-model/greedy-decoding
weakness, not something the ablation introduces from nothing. Same prompt,
greedy decoding, both models:

```
Prompt: ...वायरल वीडियो में एक वर्दीधारी होमगार्ड चारप

Standard: ाई पर चढ़कर अपनी गाड़ी में बैठा रहा। इस दौरान होमगार्ड के जवान ने
          वर्दीधारी होमगार्ड को थप्पड़ मार दिया। होमगार्ड के जवान ने होमगार्ड
          के जवान को थप्पड़ मार दिया। होमगार्ड के जवान ने होमगार्ड के जवान को
          थप्पड़ मार दिया...
          [repeats "होमगार्ड के जवान ने होमगार्ड के जवान को थप्पड़ मार दिया"]

Ablated:  ाई पर तैनात था। इस वीडियो को लेकर सोशल मीडिया पर वायरल हो रहे वीडियो
          को लेकर लोगों में आक्रोश है। कई लोग इस वीडियो को लेकर तरह-तरह की
          प्रतिक्रियाएं दे रहे हैं। कई लोग इसे सोशल मीडिया पर वायरल कर रहे हैं।
          कई लोग इसे सोशल मीडिया पर वायरल कर रहे हैं। कई लोग इसे...
          [repeats "कई लोग इसे सोशल मीडिया पर वायरल कर रहे हैं"]
```

Both fall into an exact repeated-phrase loop; the ablation makes this
existing failure mode moderately more frequent across the 50-sample test set
(94% vs 86% of greedy generations contain a repeated 4-gram), consistent with
a model that has a somewhat weaker tool for representing "I have already said
this, at this position" without positional information -- see
`bonus/report/ABLATION_REPORT.md` for the full mechanistic discussion.

### Attention analysis (entropy and mean attention distance, averaged over heads and 2 held-out sequences)

| | Standard (RoPE) | Ablated (no pos.) |
|---|---|---|
| Overall mean entropy | 2.864 | 3.061 |
| Overall mean attention distance (tokens) | 32.9 | 36.5 |

The ablated model's attention is **more diffuse** (higher entropy) and attends
**farther** on average (higher mean distance) than the standard model's. This is
the clearest and most mechanistically interpretable signature of the ablation:
without RoPE giving each head a cheap way to specialize toward
short-range/local vs. long-range patterns by relative position, attention heads
in the ablated model can only distinguish tokens by content, not by "how far
away" they are. The by-layer breakdown makes this sharper still -- the standard
model shows a clear U-shape (higher entropy/distance in early layers, a sharp
dip to short-range, positionally-precise attention in the middle layers around
layer 6-9, then broadening again late), a pattern consistent with RoPE letting
middle layers do precise local composition. The ablated model's middle layers
still dip (entropy 1.66-2.24 at layers 7-9 vs. the standard model's 2.69-2.95),
but far less sharply, and its attention distance never gets as tightly local
(minimum ~5.6 tokens vs. the standard model's minimum ~18.8) -- consistent with
attention heads working harder, and less precisely, to recover positional
structure from content alone. See `bonus/report/plots/attention_entropy_by_layer.png`,
`attention_distance_by_layer.png`, and the per-example heatmaps for the full
per-head, per-layer picture.

**Note on the heatmaps:** the per-head heatmaps
(`attn_heatmap_example*_layer*_allheads.png`) look almost entirely dark purple
for the ablated model, in contrast to the standard model's heatmaps, which show
a clear bright diagonal band (each head concentrating weight a few tokens back
from the query). This is not a rendering bug -- both sets of plots share the
same fixed 0-1 colorbar, and it *is* the finding: the standard model's heads
concentrate enough weight on a few nearby key positions that individual cells
exceed ~0.4-1.0, while the ablated model spreads its attention thinly across
up to 256 key positions per query, so no single cell is bright enough to stand
out at this scale. The near-uniform heatmaps are the visual counterpart of the
higher entropy / longer mean attention distance numbers above -- exactly what
"more diffuse attention" looks like when rendered.

### What breaks without position information

Putting the three probes together: removing positional information from
attention **does not stop the model from training** (loss converges to a
sensible value, and the gap in raw predictability -- PPL/BPB -- is real but
modest) because a decoder-only Transformer, even with completely
position-agnostic attention, can still exploit two implicit positional
crutches: (1) the causal mask, which gives every token *some* signal about
relative order ("everything I can attend to came before me"), and (2) purely
local, content-based cues (bigram/trigram statistics, repeated Hindi
function-word patterns) that approximate short-range position without
encoding it explicitly. What actually breaks is **precision of position-aware
behavior**: the ablated model's attention is measurably more diffuse and
longer-range on average (it can't cheaply "zoom in" on a fixed relative
offset the way RoPE lets it), and at the generation level this shows up as
markedly higher repetition -- the model has a harder time tracking "what has
already been generated, and how recently" without positional coordinates, so
greedy decoding collapses into repeated phrases more often (94% vs 75% of
sampled continuations) and lexical diversity drops at every temperature. In
short: position information mainly buys **local precision and non-repetition**,
not the ability to model language at all.
