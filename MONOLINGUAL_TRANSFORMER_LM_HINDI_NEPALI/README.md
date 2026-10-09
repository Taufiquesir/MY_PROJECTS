[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer LMs: Hindi + Nepali

This is a Language Models and Agents individual project. The goal is to build two
completely independent decoder-only Transformer language models from scratch —
one for Hindi (the higher-resource language, "Model H") and one for Nepali (the
lower-resource language, "Model L"). All three phases are done: Phase 1 (data
collection and tokenizer construction), Phase 2 (the Transformer implementation,
pretraining, and language-modeling/attention evaluation), and Phase 3 (reasoning
finetuning, post-finetune attention analysis, and the final consolidated
comparison). Everything below still applies — this README just no longer stops at
Phase 1.

- Phase 1 report: [`report/phase1/`](report/phase1/)
- Phase 2 report: [`report/phase2/README.md`](report/phase2/README.md)
- Phase 3 report: [`report/phase3/README.md`](report/phase3/README.md)

| | Language | Tier | Vocab size | Training tokens (exact) | Manual share |
| --- | --- | --- | --- | --- | --- |
| **Model H** | Hindi | Higher-resource | 9,000 | 679,019,018 | 21.5% |
| **Model L** | Nepali | Lower-resource (from the allowed list) | 9,000 | 568,348,098 | 24.2% |

Both languages pass the assignment's ~500M-token target and the 20% manual-collection
minimum. Full numbers, plots, and explanations are in the per-language reports:

- [`report/phase1/stats_hindi.md`](report/phase1/stats_hindi.md)
- [`report/phase1/stats_nepali.md`](report/phase1/stats_nepali.md)
- [`report/phase1/comparison_report.md`](report/phase1/comparison_report.md) — Hindi vs. Nepali side by side, and why the numbers differ the way they do

No data, tokenizer, vocabulary, or model weights are shared between the two
languages anywhere in this project. I checked this directly, not just assumed it —
see the corpus independence check in the comparison report.

## Repository layout

```
README.md                    this file
collection/                  shared pipeline code (used by both languages, but run
                              separately for each — nothing here mixes the two)
  collect.py                   the main CLI: news / wiki / public / clean / split /
                                sweep / train / count, run one language at a time
  download_indiccorp.py        downloads IndicCorpV2 text (resumable, byte-range)
  import_indiccorp.py          turns a downloaded .txt into the project's document format
  sources_hindi.yaml           Hindi's list of news sites + how much text to aim for
  sources_nepali.yaml          same, for Nepali
  common/                      record format, crawler, Wikipedia ingest, cleaning,
                                splitting, stats, tokenizer helpers, plots
run_phase1.py                 runs the whole pipeline end to end for both languages
tests/                        unit tests
hindi/
  tokenizer/                    hindi_tokenizer.model / .vocab (committed to git)
  model/                        the Transformer implementation (Phase 2) — attention,
                                 positional scheme, blocks, checkpoint save/resume
  train/                        pretraining entry point + checkpoints/ (not committed)
  eval/                         PPL/BPB, generation quality, attention analysis (Phase 2)
  reasoning/                    Phase 3: synthetic reasoning dataset, finetuning,
                                 pretrained-vs-finetuned eval, attention comparison
  configs/                      model.yaml, train.yaml, finetune.yaml
nepali/                       same structure as hindi/, for Nepali — nothing shared
report/phase1/                Phase 1 reports, plots, and the corpus independence result
report/phase2/                Phase 2 report: architecture, training curves, PPL/BPB,
                               generation quality, attention heatmaps
report/phase3/                Phase 3 report: reasoning dataset, finetuning curves,
                               pretrained-vs-finetuned accuracy, post-finetune attention
data/                         NOT committed — this is where the actual corpus lives
                               while working locally. See "Google Drive links" below
                               for where to get it.
logs/                         NOT committed — logs from running the pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command below from this folder (the repo root), so the default `data/`
and `report/` paths work correctly.

## Why these two languages

- **Hindi (Model H):** picked as the higher-resource language. There's a lot of
  Hindi text available — IndicCorpV2's Hindi shard is large, several news sites
  have proper sitemaps we can crawl, and Hindi Wikipedia is fully browsable
  through the API.
- **Nepali (Model L):** picked from the assignment's allowed lower-resource list
  (the others were Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri,
  Mizo, Sindhi). Nepali also uses Devanagari script, which let me reuse the same
  cleaning/script-detection logic for both languages while still keeping the
  actual corpora completely separate. I tried one candidate news site,
  `nepalpress.com`, but dropped it — its crawl kept returning zero URLs because of
  a Cloudflare rule blocking my request pattern. Not worth building a workaround
  for one out of five sites. Details in `collection/sources_nepali.yaml`.

## How to reproduce Phase 1

Each language gets its own manually-collected corpus (news + Wikipedia) plus a
downloaded corpus (IndicCorpV2), which get combined and cleaned into one pipeline.

**Step 1 — collect the manual part** (crawls news sites via their sitemap, plus
Wikipedia). Every step here can be safely interrupted and restarted — progress is
saved to `data/<lang>/done.txt` (news) and `data/<lang>/wiki_cursor.txt`
(Wikipedia), so nothing gets re-downloaded from scratch.

```bash
python -m collection.collect --lang hindi  --stage news
python -m collection.collect --lang nepali --stage news
python -m collection.collect --lang hindi  --stage wiki
python -m collection.collect --lang nepali --stage wiki
```

**Step 2 — download and import the public part** (AI4Bharat IndicCorpV2):

```bash
python -m collection.download_indiccorp --lang hindi  --tokens 500000000
python -m collection.download_indiccorp --lang nepali --tokens 575000000
python -m collection.import_indiccorp --lang hindi  --path data/hindi/indiccorp/hi.txt
python -m collection.import_indiccorp --lang nepali --path data/nepali/indiccorp/ne.txt
```

**Step 3 — clean the data, split it, and build the tokenizer.** These are five
separate commands on purpose, not one combined script. That way, if something
goes wrong partway through (which happened more than once while building this —
see the memory note below), you only have to re-check and re-run the one stage
that failed, not the whole thing.

```bash
python -m collection.collect --lang hindi --stage clean
python -m collection.collect --lang hindi --stage split
python -m collection.collect --lang hindi --stage sweep                        # tries different vocab sizes
python -m collection.collect --lang hindi --stage train --target-tokens 9000   # trains the final tokenizer
python -m collection.collect --lang hindi --stage count                        # counts exact tokens + unknown-token rate
# then repeat the same five commands with --lang nepali
```

Both `clean` and `split` clear out their own output folder before writing
anything new, so re-running either one after an interruption is always safe —
it won't silently double the corpus.

**A note on machine memory.** I built this on a machine with about 7GB of RAM
and basically no spare swap. SentencePiece's tokenizer trainer loads its entire
input file into memory before it does any sampling, so pointing it straight at
the full multi-gigabyte corpus reliably got the process killed by the OS,
regardless of how many threads I used — this actually happened twice during
development, confirmed by checking `journalctl -k` for the OOM-killer log entries.
My fix was to shrink the training sample down to 250,000 documents before ever
calling the trainer (see `TRAIN_SAMPLE_DOCS` in `collection/collect.py`).

**Step 4 — generate the reports and plots.** `run_phase1.py` runs the same
clean → split → sweep → train → count sequence end to end and additionally draws
the figures and runs the corpus-independence check. If you already did step 3 by
hand, skip those stages here:

```bash
python run_phase1.py --langs hindi nepali --skip clean split sweep train count
```

**What you get at the end:**

| Path | What's in it |
| --- | --- |
| `<lang>/tokenizer/<lang>_tokenizer.{model,vocab}` | the trained tokenizer (committed to git) |
| `<lang>/data/splits.json` | how many documents/bytes ended up in train/val/test |
| `report/phase1/stats_<lang>.md` | the full per-language report |
| `report/phase1/comparison_report.md` | Hindi vs. Nepali side by side |
| `report/phase1/plots/*.png` | all the figures referenced in the reports |
| `data/<lang>/` | the actual corpus (raw, cleaned, split) — **not in git**, see below |

## Where the data comes from

- **Manual collection:** a crawler (`collection/common/crawler.py`) that reads
  each news site's XML sitemap to find article URLs, then pulls the article text
  out with `trafilatura`. There's a separate Wikipedia ingest
  (`collection/common/wiki.py`) that walks through each language's Wikipedia via
  the MediaWiki API. Everything runs at 2.5 requests per second per site — I
  checked every site's `robots.txt` by hand first and none of them ask for
  anything slower, so this was a deliberate choice, not a guess.
  - Hindi sites: zeenews.india.com, indiatv.in, jagran.com, patrika.com, plus
    hi.wikipedia.org
  - Nepali sites: khabarhub.com, dcnepal.com, karobardaily.com, nepalnews.com,
    plus ne.wikipedia.org
  - Any site that was tried and rejected (no usable sitemap, or blocked the
    crawler) is noted in `collection/sources_hindi.yaml` /
    `sources_nepali.yaml`.
- **Public/downloaded:** [AI4Bharat IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2),
  downloaded as a resumable slice of each language's source file
  (`collection/download_indiccorp.py`), then converted into the same document
  format the manual data uses (`collection/import_indiccorp.py`).

Every document, no matter where it came from, is tagged with how it was
collected. That's what makes it possible to compute the manual-vs-downloaded
token split directly from the actual corpus instead of tracking it by hand.

## Cleaning, and how we know the two corpora don't overlap

Both languages go through the exact same cleaning steps
(`collection/common/clean.py`), run completely separately for each language:

1. Unicode normalization + stripping stray control characters.
2. Drop anything under 200 characters.
3. Drop anything that isn't at least 70% Devanagari letters, or that's more than
   15% Latin letters.
4. Drop documents that are mostly one repeated line or phrase (catches
   boilerplate that would otherwise pass the script check).
5. Drop documents with a heavy concentration of blocklisted words (each language
   has its own separate wordlist).
6. A decontamination check that currently does nothing — it's there for Phase 3,
   to make sure evaluation data never leaks into training, once that evaluation
   data actually exists.
7. Remove exact duplicates (same text, byte for byte).
8. Remove near-duplicates (same story reprinted with small differences) — this
   only runs on the manually-crawled news and Wikipedia data. Running it on the
   full IndicCorp data (millions of documents) would need more memory than this
   machine has, and IndicCorp is already deduplicated by its publishers anyway.

I don't just assume Hindi and Nepali never overlap because they came from
different sources — I actually checked. Every cleaned document gets hashed, and
I compare the two sets of hashes for overlap. Zero shared documents, confirmed
in `report/phase1/corpus_independence.json` and written up in the comparison
report.

## Tokenizer

Both languages use a **SentencePiece Unigram** tokenizer (not BPE), which I
trained completely separately for each language with no shared vocabulary. Hindi
and Nepali both change words a lot based on grammar (case endings, verb forms),
and Unigram tends to find subword pieces that line up with those patterns better
than BPE's simpler merge-based approach.

I picked the vocabulary size (9,000 for both languages) by sweeping across
candidate sizes and measuring fertility, then weighing that against Phase 2's
~25M-parameter budget — a bigger vocabulary means a bigger embedding table, which
eats into how many parameters are left for the actual Transformer. I kept both
languages at the same vocabulary size on purpose, so that any difference I see
later when comparing the two models isn't just an artifact of one starting with a
bigger vocabulary than the other. The full reasoning is in each report's
section 6.

| | Hindi | Nepali |
| --- | --- | --- |
| Vocabulary size | 9,000 | 9,000 |
| Exact training tokens | 679,019,018 | 568,348,098 |
| Fertility (characters per token) | 3.86 | 4.22 |
| Unknown-token rate | 0.78% | 1.10% |

See each report's section 5 for the full fertility sweep, what the vocabulary
actually looks like, and real tokenized example sentences.

## How to reproduce Phase 2

Once Phase 1's tokenized splits exist (or are downloaded from the Drive/Kaggle
links below), pretraining and evaluation for each language is:

```bash
python -m hindi.train.train
python -m nepali.train.train

python run_eval.py --lang hindi  --checkpoint hindi/train/checkpoints/best.pt  --test-bin <path>
python run_eval.py --lang nepali --checkpoint nepali/train/checkpoints/best.pt --test-bin <path>
```

Training is checkpoint-resumable — rerunning the same command after an
interruption picks up from `<lang>/train/checkpoints/latest.pt` instead of
starting over. Full details, the architecture writeup, and every plot/table are
in [`report/phase2/README.md`](report/phase2/README.md).

## How to reproduce Phase 3

With a Phase 2 pretrained checkpoint already in place for a language:

```bash
# 1. Generate the synthetic reasoning dataset
python -m hindi.reasoning.generate
python -m nepali.reasoning.generate

# 2. Finetune from the Phase 2 checkpoint (also resumable, same checkpoint format)
python -m hindi.reasoning.finetune
python -m nepali.reasoning.finetune

# 3. Pretrained-vs-finetuned exact-match accuracy on the held-out test set
python -m hindi.reasoning.eval
python -m nepali.reasoning.eval

# 4. Pretrained-vs-finetuned attention comparison on reasoning prompts
python -m hindi.reasoning.attention_compare
python -m nepali.reasoning.attention_compare
```

Every script writes its output under `report/phase3/<lang>/`. Full writeup,
tables, and every plot are in
[`report/phase3/README.md`](report/phase3/README.md).

## Google Drive links

I didn't commit the actual corpus (raw crawl data, the IndicCorp download, and
the final cleaned/split data) to git — it's too large, and the assignment
asks for it to go on Drive instead. Each language has its own separate Drive
folder, so Hindi and Nepali stay clearly apart there too, not just in the repo.

- **Hindi corpus:** https://drive.google.com/file/d/1cbdAKZDpitlng7w7EOXdV6u2obkRsd7r/view?usp=sharing
- **Nepali corpus:** https://drive.google.com/file/d/1bNoiObSdq-lLk-Ktq4zli69nf_n_OiPe/view?usp=sharing

Each folder has its own README explaining what's inside and what's deliberately
left out (a couple of things, like the tokenizer's training scratch files, are
easy to regenerate from the scripts in this repo and weren't worth uploading).

The same corpora are also mirrored on Kaggle:

- **Hindi corpus (Kaggle):** https://www.kaggle.com/datasets/taufiquehussain/lma-hindi-dataset
- **Nepali corpus (Kaggle):** https://www.kaggle.com/datasets/taufiquehussain/lma-nepali-dataset

**Pretrained checkpoints (Phase 2)** — each folder holds `best.pt` (lowest
validation loss) and `latest.pt` (final, step 61,000), both in the mandatory
resumable format (weights + optimizer state + scheduler state + step + config):

- **Model H (Hindi) checkpoints:** https://drive.google.com/drive/folders/14b0rqKoEkKjBuVlgOMwq5W50KpTOnHi-?usp=drive_link
- **Model L (Nepali) checkpoints:** https://drive.google.com/drive/folders/1OkOhXPjoIKwwlY8kGib8vNY4-UKTbYm9?usp=drive_link

**Finetuned reasoning checkpoints (Phase 3)** — each folder holds `best.pt` (lowest
validation loss during finetuning) and `latest.pt` (final epoch), same resumable
format as above, started from the matching Phase 2 pretrained checkpoint with the
tokenizer frozen:

- **Model H (Hindi) finetuned checkpoint:** https://drive.google.com/drive/folders/1K92Wc8C7KOV8bcjj1cUYb49aTHLZClaJ?usp=sharing
- **Model L (Nepali) finetuned checkpoint:** https://drive.google.com/drive/folders/1diZrfo1JVBqhmnwv__Zd9iA2bzuG4Zgn?usp=sharing

**Bonus ablation checkpoint (no positional embeddings, `bonus` branch)** — same
resumable format, trained from scratch for the full 61,000-step budget:

- **Bonus ablation (Hindi, no positional embeddings) checkpoints:** https://drive.google.com/drive/folders/11Nxr-iWS8qcwzrOo4Q6I5k4MSd_g0ntG?usp=sharing

All Drive links above are set to "anyone with the link can view."

## Tests

```bash
pytest tests/
```

Covers the Unicode normalization edge cases (things specific to Devanagari, like
nukta composition and telling apart meaningful vs. stray zero-width joiners) and
the document quality filters.
