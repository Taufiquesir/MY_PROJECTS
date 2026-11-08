# My Projects

A collection of my personal and academic projects, spanning systems programming,
machine learning and image processing.

| Project | Description | Tech |
|---|---|---|
| [POSIX Shell](POSIX_SHELL_IMPLEMENTATION/) | A Unix shell built from scratch — process control, pipelines, I/O redirection, signals and job control | C++, POSIX, GNU readline |
| [SMS Spam Classifier](SMS_SPAM_CLASSIFIER/) | Web app that classifies SMS messages as spam or ham, with a trained model and live demo | Python, scikit-learn, Streamlit |
| [Crop & Weed Detection](CROP_WEED_DETECTION/) | Object-detection system distinguishing crops from weeds to enable targeted pesticide use | YOLOv3, Darknet, OpenCV |
| [Monolingual Transformer LMs](MONOLINGUAL_TRANSFORMER_LM_HINDI_NEPALI/) | Two independent decoder-only Transformer language models built from scratch, for Hindi and Nepali | Python, PyTorch, SentencePiece |
| [Haze Removal via Dark Channel Prior](DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR/) | Classical single-image dehazing with three transmission-refinement methods, plus a temporally regularised video extension | Python, NumPy, SciPy, OpenCV |
| [Parallel LLM Inference Pipeline](PARALLEL_LLM_INFERENCE_PIPELINE/) | GPT-2 split across two worker processes over gRPC, with concurrency, backpressure and crash recovery by replay (distributed systems) | Python, PyTorch, gRPC |

---

## POSIX Shell Implementation

A fully functional Unix shell in C++, written without using `system()`, `popen()`
or the C++ `filesystem` library.

**Highlights**
- `ls -l` output is **byte-identical** to the system `ls`, including block totals and column alignment
- Pipelines of arbitrary length that never hang — all pipe descriptors are closed in the parent
- Async-signal-safe `SIGCHLD` handling for reaping background jobs
- Built-in commands support redirection by saving and restoring the shell's own file descriptors
- Compiles clean under `-Wall -Wextra` with **zero warnings**

**Built-ins:** `cd`, `echo`, `pwd`, `ls`, `pinfo`, `search`, `history`, `jobs`

```bash
cd POSIX_SHELL_IMPLEMENTATION
make
./myshell
```

Documentation: [`SHELL_BOOK.pdf`](POSIX_SHELL_IMPLEMENTATION/SHELL_BOOK.pdf) is a
52-page implementation study covering every module and system call.
[`TESTING_GUIDE.md`](POSIX_SHELL_IMPLEMENTATION/TESTING_GUIDE.md) walks through
every test case with explanations.

---

## SMS Spam Classifier

A machine-learning web app that classifies SMS messages as spam or legitimate.
Text is cleaned and vectorised with TF-IDF, then classified by a model trained on
a labelled SMS dataset.

```bash
cd SMS_SPAM_CLASSIFIER
pip install -r requirements.txt
streamlit run app.py
```

---

## Crop & Weed Detection

An intelligent weed-detection system for agriculture. Weeds compete with crops
for resources, and blanket pesticide spraying leaves residue and harms the
environment — this model enables targeted application instead.

Built on **YOLOv3** with transfer learning from pretrained convolutional weights
in the Darknet framework, trained on a Kaggle dataset of 1,300 labelled sesame
crop and weed images with YOLO-format bounding boxes.

---

## Monolingual Transformer LMs (Hindi and Nepali)

Two completely independent decoder-only Transformer language models, one for Hindi
(higher-resource) and one for Nepali (lower-resource), built from scratch with no
data, tokenizer or weights shared between them. Covers data collection and
tokenizer construction, the Transformer implementation and pretraining, and
reasoning finetuning with an attention analysis.

See [`MONOLINGUAL_TRANSFORMER_LM_HINDI_NEPALI/README.md`](MONOLINGUAL_TRANSFORMER_LM_HINDI_NEPALI/README.md).

---

## Haze Removal via Dark Channel Prior

Classical image dehazing implemented from first principles, with no learned
component. The Dark Channel Prior pipeline exposes every intermediate output (dark
channel, atmospheric light, raw and refined transmission, recovered image), and
three transmission-refinement methods are compared on accuracy and runtime: soft
matting, a guided filter and a fast guided filter. A video extension then studies
why per-frame dehazing flickers and tests four temporal-smoothing variants.

**Highlights**
- Benchmarked on RESIDE SOTS: **19.82 dB** (indoor) and **17.03 dB** (outdoor) with default settings
- Fast guided filter matches the guided filter within 0.04 dB at about 60% of the time; soft matting is roughly 100x slower with no gain
- Video: smoothing the atmospheric light cuts flicker from **2.04 to 1.58** (clean video = 1.0); smoothing the transmission, even with optical-flow warping, does not help
- Sky is a measurable failure mode (r = -0.50 between sky fraction and PSNR); a sky-aware transmission floor gains 2.4 dB on sky-heavy images
- Every reported number regenerates from one script, and a re-run reproduced the results

```bash
cd DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Documentation: two project reports in [`reports/`](DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR/reports/),
the final deck in [`presentation/`](DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR/presentation/), and
[`IMPLEMENTATION_LOG.md`](DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR/IMPLEMENTATION_LOG.md) with every
design decision and measurement. The datasets are downloaded separately; see the
project README.

---

## Parallel LLM Inference Pipeline

A distributed-systems project: GPT-2 small is cut into two halves, each running in
its own process and talking over gRPC. The goal is token-for-token identical output
to the single-process model, under concurrent users and when a worker is killed
mid-sentence. Covers RPC, per-request KV-cache state, ordering and idempotency,
bounded queues and backpressure, heartbeats, epoch fencing and recovery by replay.

Status: planning and concept study complete; implementation in progress. See
[`PARALLEL_LLM_INFERENCE_PIPELINE/README.md`](PARALLEL_LLM_INFERENCE_PIPELINE/README.md)
for the phase-by-phase implementation plan.

---

## Repository layout

```
MY_PROJECTS
├── POSIX_SHELL_IMPLEMENTATION/
├── SMS_SPAM_CLASSIFIER/
├── CROP_WEED_DETECTION/
├── MONOLINGUAL_TRANSFORMER_LM_HINDI_NEPALI/
├── DIP_HAZE_REMOVAL_DARK_CHANNEL_PRIOR/
├── PARALLEL_LLM_INFERENCE_PIPELINE/
└── README.md
```
