# Project 38 — Parallel LLM Inference with a Fault-Tolerant Pipeline

**Team 4:** MD TAUFIQUE HUSSAIN
**Plan written:** 24 Oct 2026

This file has four parts:

1. [What the project is](#1-what-the-project-is-in-plain-words): the idea in plain words
2. [Business perspective](#2-business-perspective-why-anyone-would-pay-for-this): why companies care about this problem
3. [The phase-by-phase plan](#3-the-phase-by-phase-plan): how to build it, one small working step at a time
4. [Reference sheets](#4-reference-sheets): repo layout, message format, test matrix, risks, team split

---

## 1. What the project is (in plain words)

### 1.1 The one-sentence version

Take GPT-2 small (12 layers), **cut it in half**, run each half in a **separate process**, make the two halves talk over **gRPC**, and prove that the split system writes **exactly the same text** as the normal model, even when several users send prompts at once and even when one half **crashes in the middle of a sentence**.

### 1.2 The picture

```
             prompt                       token IDs + step + epoch
  Client  ─────────────►  Coordinator  ──────────────────────────►  Worker A (Stage 1)
          ◄─────────────    (brain)                                  embeddings + blocks 0–5
          streamed tokens      ▲                                      KV cache for its 6 layers
                               │                                            │
                               │ next token                    activations  │ (768 floats per token)
                               │                                            ▼
                               └──────────────────────────────────  Worker B (Stage 2)
                                                                     blocks 6–11 + final LN + LM head
                                                                     KV cache for its 6 layers
                                                                     picks next token (greedy)
```

- **Coordinator** = the control logic. It hands out request IDs, runs the token-by-token loop, sends heartbeats, detects crashes, restarts workers and replays prompts. It never touches model weights.
- **Workers** = the muscle. Each holds half the weights plus a per-request KV cache. They stay simple: check the message labels, run layers, reply.

### 1.3 Why it is a *distributed systems* project, not an ML project

The model is tiny and pretrained, so you do no ML research at all. Every hard part is a classic distributed-systems problem:

| Problem that appears once you split the model | Concept from the course | Where you solve it |
|---|---|---|
| Halves must exchange tensors | RPC, serialization | Phase 3 |
| Each worker holds state (KV cache) that the other needs to be consistent with | Distributed state, ordering | Phases 2–4 |
| Many users at once | Concurrency, isolation, pipelining | Phase 4 |
| A slow stage could flood the other with work | Bounded queues, backpressure | Phase 5 |
| You cannot tell "crashed" from "slow" | Timeouts, heartbeats, failure detection | Phase 6 |
| Crash wipes the KV cache | Recovery by replay | Phase 6 |
| Old messages arrive after a restart; retries create duplicates | Epoch fencing, idempotency | Phases 4 + 6 |
| Is distribution worth it? | Communication overhead, latency vs throughput | Phase 8 |

### 1.4 The five ideas you must understand before coding

1. **Prefill vs decode.** Prefill runs the whole prompt through the model at once and produces the first new token. Decode then runs one token at a time. Step *n+1* needs the token from step *n*, so a single request can never keep both workers busy.
2. **KV cache.** Each attention layer saves the Keys and Values of every past token, so a decode step only computes the one new token. This cache is **the state** of your distributed system: it is split across workers, it is per-request, it must grow in order, and it is lost on a crash.
3. **Positions.** The new token at decode step *k* sits at position `prompt_len + k - 1`, not 0. Getting this wrong gives plausible-looking but wrong text. Most "it almost matches" bugs come from here.
4. **Greedy decoding = determinism.** Always pick the highest-scoring token (argmax). Same input, same output, so you can compare token IDs exactly with a reference run.
5. **Replay.** Because the model is deterministic, re-running `prompt + tokens generated so far` as one big prefill rebuilds *exactly* the lost KV cache. That is why crash recovery can be exact.

---

## 2. Business perspective (why anyone would pay for this)

### 2.1 The real-world problem

Production LLMs (70B to 400B+ parameters) **do not fit on one GPU**. A 70B model in 16-bit needs about 140 GB just for weights, while a top GPU has about 80 GB. So every serious LLM provider splits models across devices, and the moment they do, they inherit all the problems this project studies. Your GPT-2 setup is a **scale model** of what runs inside:

- **Inference platforms**: OpenAI, Anthropic, Google, and cloud APIs (AWS Bedrock, Azure OpenAI)
- **Open-source serving engines**: vLLM, NVIDIA TensorRT-LLM / Triton, Hugging Face TGI, DeepSpeed-Inference, all of which support pipeline parallelism
- **Decentralised inference**: Petals runs big models across volunteers' GPUs over the internet, where workers crash and disappear all the time
- **On-prem and edge**: splitting a model between a phone and a server, or across several cheaper GPUs instead of one expensive one

### 2.2 How each part of the project maps to business value

| What you build | Business metric it affects | Why a company cares |
|---|---|---|
| Model split across workers | **Hardware cost** | Run big models on several cheap GPUs instead of a few very expensive ones |
| Concurrency + micro-batching | **Throughput (tokens/sec per GPU) → cost per million tokens** | GPU hours are the main cost of an AI product; idle GPUs (pipeline bubbles) burn money |
| Time to first token, per-token latency | **User experience** | Users feel latency directly; chat products target first token in under ~1 s |
| Bounded queues + backpressure | **Stability under traffic spikes** | Better to reply "busy, retry" (HTTP 429) than to crash with out-of-memory and drop *every* user |
| Heartbeats, restart, replay | **Availability / SLA** (e.g. 99.9%) | Hardware fails constantly at scale; a crash must not lose users' half-finished answers |
| Epochs, step IDs, exactly-once delivery | **Correctness and trust** | A user must never see duplicated words or another user's text, which would be a privacy incident |
| Token-for-token equality with single process | **Reproducibility / auditability** | Regulated industries (finance, health, legal) need to prove the system behaves as tested |
| Bytes-transferred measurement | **Network cost and capacity planning** | Decides whether stages can live on separate machines or need fast interconnects |

### 2.3 The honest trade-off (put this in your report)

For a model as small as GPT-2, the pipeline **will be slower** than one process: serialization and network time outweigh compute. That is expected, not a failure. The business lesson is that **you only split a model when you must** (it does not fit) **or when concurrency fills the pipeline** (throughput gain). Your experiments E1 and E2 show exactly this crossover, which is the same reasoning an infrastructure team does before buying hardware.

### 2.4 Your pitch in two lines (for viva or demo)

> "Large models must be split across machines, and splitting turns inference into a distributed system. We built a two-stage GPT-2 pipeline that serves concurrent users, survives a worker being killed mid-sentence with byte-identical output, never delivers a token twice, and we measured exactly what distribution costs in latency and bytes."

---

## 3. The phase-by-phase plan

**Golden rule:** every phase ends with something that **runs and is tested** before you start the next one. Never build the distributed version before the single-process version is proven correct.

```
Phase 0  Setup                       ─┐
Phase 1  Reference (single process)   │  "Make it correct"
Phase 2  In-process split             ─┘
Phase 3  Two processes over gRPC     ─┐
Phase 4  Concurrency + ordering       │  "Make it distributed"
Phase 5  Bounded queues/backpressure ─┘
Phase 6  Failure detection + recovery ─  "Make it survive"
Phase 7  Micro-batching (optional)    ─  "Make it fast"
Phase 8  Benchmarks                  ─┐
Phase 9  Report, README, demo        ─┘  "Prove it"
```

Suggested timing assumes about 7–8 weeks; stretch or squeeze to your deadline. Phases 6 and 8 are the most important for marks, so protect their time.

---

### Phase 0 — Setup (2–3 days)

**Goal:** the same code runs on every machine with the same library versions.

**Tasks**
- Create the GitHub repo with the folder layout from [§4.1](#41-suggested-repo-layout).
- Python 3.11 virtual environment. Install `torch` (CPU build is fine), `transformers`, `grpcio`, `grpcio-tools`, `pytest`, `pandas`, `matplotlib`, `psutil`.
- **Pin versions on day 1:** `pip freeze > requirements.txt` and commit it. The Hugging Face KV-cache format has changed between versions; pinning avoids surprises.
- Download GPT-2 once (`GPT2LMHeadModel.from_pretrained("gpt2")`) and save it locally so workers load from disk, not the internet.
- Write `prompts.json` with the 20 test prompts of varied length and topic.

**Done when:** `python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"` works on every laptop, and the requirements file is committed.

**Learn in this phase:** read the GPT-2 block code (`ln_1 → attn → +residual → ln_2 → mlp → +residual`).

---

### Phase 1 — Single-process reference with your own KV cache (4–5 days)

**Goal:** a ground-truth generator you fully understand and control.

**Why your own loop?** You will later need to run "blocks 0–5 only" with "my own cache dict". If you depend on `model.generate()` internals, you cannot split it. So write a manual loop and prove it matches `generate()`.

**Recommended approach (robust to library changes):** write your own small GPT-2 forward pass that **reuses Hugging Face's weights** but **manages the KV cache yourself** as plain tensors:

```python
# model_core.py  (sketch)
def embed(token_ids, positions):            # wte + wpe
    return wte(token_ids) + wpe(positions)

def run_block(block, h, cache):             # cache = {"k": Tensor|None, "v": Tensor|None}
    x = block.ln_1(h)
    q, k, v = block.attn.c_attn(x).split(768, dim=-1)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)   # [B, 12, T, 64]
    if cache["k"] is not None:
        k = torch.cat([cache["k"], k], dim=2)
        v = torch.cat([cache["v"], v], dim=2)
    cache["k"], cache["v"] = k, v
    a = causal_attention(q, k, v)           # mask so token i sees only <= i
    h = h + block.attn.c_proj(merge_heads(a))
    h = h + block.mlp(block.ln_2(h))
    return h

def lm_head(h):                             # ln_f + tied embedding matrix
    return ln_f(h) @ wte.weight.T
```

All cache handling then lives in **one module** (a mitigation already listed in your risk table), and a `transformers` upgrade cannot break it.

**Tasks**
- Implement `generate_reference(prompt_ids, max_new_tokens=50)` using prefill, then a decode loop with argmax.
- Run all 20 prompts × 50 tokens and save the token IDs to `reference_outputs.json`. **This is your ground truth for the whole project (Test 1).**
- Test: your manual loop equals `model.generate(do_sample=False)` for every prompt.

**Done when:** `pytest tests/test_reference.py` passes 20/20.

**Common bugs:** positions not offset during decode; causal mask wrong during prefill; forgetting `ln_f` before the LM head; dtype not float32.

---

### Phase 2 — In-process split: Stage 1 + Stage 2 as two classes (2–3 days)

**Goal:** split the model **logically** before you split it **physically**. Any mismatch here is a model bug, not a network bug, which makes debugging much easier.

**Tasks**
- `Stage1`: owns `wte, wpe, blocks[0:6]` and `kv_cache: dict[request_id → list of 6 layer caches]`.
  `forward(request_id, token_ids, positions) → hidden [1, T, 768]`
- `Stage2`: owns `blocks[6:12], ln_f, lm_head` and its own `kv_cache`.
  `forward(request_id, hidden) → next_token_id`
- `release(request_id)` on both deletes the cache entry.
- Make the **split point configurable** (`--split 6`) for the load-balance experiment later (e.g. a 7/5 split).
- Add serialization helpers now: `to_bytes(tensor) → (bytes, shape, dtype)` and back. Test the round trip, since you will need it in Phase 3.

**Done when:** in one Python process, `Stage1 → to_bytes → from_bytes → Stage2` reproduces all 20 references exactly, including two requests interleaved step by step (an early isolation test).

---

### Phase 3 — Two real processes over gRPC (5–6 days)

**Goal:** the same thing, now over the network: Test 2 (one request at a time).

**Tasks**
1. Write `proto/pipeline.proto` (see [§4.2](#42-message-format-protobuf-sketch)) and generate code with `python -m grpc_tools.protoc`.
2. **Worker server** (`worker.py --stage A|B --port … --epoch …`) wraps Stage1 or Stage2 behind gRPC methods `Forward`, `Release`, `Heartbeat`.
3. **Choose the data path.** Recommended: **coordinator → A, then A calls B, then B's token returns through A to the coordinator.** Activations then travel only between A and B, as in your architecture figure, and byte counting is clean. (Alternative: the coordinator relays activations. It is simpler but doubles the traffic, so avoid it.)
4. **Coordinator** (`coordinator.py`): assigns UUIDs, runs the loop *prefill (step 0) → decode step 1, 2, …*, keeps `prompt_ids` and `generated_ids` per request (needed for replay later), and calls `Release` at the end.
5. **Client** (`client.py`): sends a prompt, prints tokens as they stream back. Use a server-streaming RPC between client and coordinator.
6. **Launcher** (`launcher.py`): starts both workers with `subprocess.Popen`, waits until they answer a heartbeat, and starts the coordinator.
7. Set a **deadline on every RPC** (`timeout=2.0` for decode, longer for prefill) and **turn off gRPC automatic retries** (the concept guide warns that hidden retries break idempotency).
8. Count bytes: log the size of every activation payload, split into prefill and decode.

**Done when:** Test 2 passes, with all 20 prompts through the real two-process pipeline identical to the reference. Then try it once across two laptops on the same Wi-Fi/LAN.

**Sanity check for bytes:** decode is about 3 KB per step (`768 × 4`); a 100-token prefill is about 300 KB.

---

### Phase 4 — Concurrency, ordering and duplicate handling (5–6 days)

**Goal:** many users at once, never mixed up, never out of order: Tests 3 and 6.

**Tasks**
- **Coordinator:** run each request as its own task (`asyncio` with `grpc.aio`, or a thread per request). Requests proceed independently, so while B works on request 1's step, A can work on request 2's. That is how pipelining happens naturally.
- **Worker ordering rules** (per `request_id`):

  | Incoming message | Worker action |
  |---|---|
  | `epoch < my_epoch` | Discard and reply `STALE_EPOCH` (pre-crash message) |
  | `step == expected_step` | Process, append to cache, `expected_step += 1`, **store the result** |
  | `step < expected_step` | Duplicate: do **not** touch the cache; return the stored result (idempotent) |
  | `step > expected_step` | Gap: reply `OUT_OF_ORDER`; the coordinator resends the missing step |

- Keep `last_result[request_id] = (step, token)` so duplicates can be answered without recomputing.
- **Coordinator side:** apply the same rule to results, so a `(request_id, step)` is delivered to the client **exactly once**.
- **Thread safety:** use a lock per request (or one worker loop thread) so two messages for the same request never touch its cache at the same time. PyTorch on CPU already uses many cores, so set `torch.set_num_threads()` sensibly for both workers on one laptop.
- **Idle-timeout cleanup:** delete caches not touched for N seconds (abandoned clients must not leak memory).
- **Fault-injection hooks** for Test 6: a test-only RPC or flag that resends an old step or an old epoch on purpose.

**Done when:**
- Test 3: 10 prompts sent concurrently, each identical to its own reference.
- Test 6: injected duplicate and old-epoch messages are all discarded and the output is unchanged.

---

### Phase 5 — Bounded queues and backpressure (3–4 days)

**Goal:** overload causes slower service or polite rejection, never a crash or unbounded memory: Test 7.

**Tasks**
- Each worker has an input queue with capacity 8 (configurable). Full queue → reply `BUSY` immediately.
- Coordinator on `BUSY`: exponential backoff (10 ms, 20, 40, … capped), then retry. Retries are safe because of Phase 4's idempotency.
- **Admission control:** a `max_in_flight` limit on the coordinator. New client requests beyond it get a clear `RESOURCE_EXHAUSTED` error ("server busy, try later").
- Add an artificial `--delay-ms` option to Worker B to simulate a slow stage.
- Record worker memory (RSS via `psutil`) over time.

**Done when:** Test 7 flooding beyond capacity gives clients BUSY or rejection errors, no process crashes, and the memory graph stays flat.

---

### Phase 6 — Failure detection and crash recovery (7–8 days, the heart of the project)

**Goal:** kill a worker mid-sentence, and the final text is still identical with no token delivered twice: Tests 4 and 5.

**6a. Detection**
- The coordinator sends `Heartbeat` to each worker every 500 ms with a short deadline.
- 3 missed heartbeats in a row, **or** a `Forward` call failing past its deadline → worker declared failed.
- Log the timestamp of detection (needed for the recovery-time metric).

**6b. Recovery procedure**
1. **Fence:** pause all request loops (a "recovering" flag or lock). No new steps go out.
2. **Restart:** the launcher starts a fresh process for the failed stage with `epoch = old_epoch + 1`, which loads its weights from disk. Wait for its first heartbeat.
3. **Epoch update:** the coordinator records the new epoch, and every later message carries it. (Simplest design: both workers move to the new epoch, matching the proposal where *both stages replay*.)
4. **Replay:** for every unfinished request, send `REPLAY` with `prompt_ids + generated_ids` and positions `0 … L-1`. Each worker **clears** its old cache entry for that request, runs it as one prefill, and sets `expected_step` to the next step. Worker B returns the next token, which **is** the next token the client has not seen yet.
5. **Resume** the decode loops. Already-delivered tokens are never resent, because the coordinator only forwards steps greater than the last delivered step.
6. Late messages from the old epoch are discarded by the Phase 4 rule.

**6c. Crash-test tooling**
- `kill_at_step.py --worker B --step 17`: sends `SIGKILL` to the worker's PID when the coordinator reaches that step (or kill from a test hook).
- Randomised crash test: loop N times with a random worker and random step, and assert equality every time. This catches race conditions that a single test misses.

**Watch out for**
- A request that was *in the middle* of a step when the crash happened. The coordinator must not count that step as delivered unless the token actually came back.
- Requests that **finish** during recovery, and requests that **arrive** during recovery (queue or reject them).
- Replay length grows with sequence length, so replay prefill needs the longer deadline.

**Done when:** Tests 4 and 5 pass for early, middle and late crash steps, over many randomised runs.

**Optional optimisation (good for extra discussion):** if only B crashed, A's cache is still valid. A could resend only activations instead of recomputing. Compare the recovery time of both approaches.

---

### Phase 7 — Micro-batching (stretch goal, drop it first if time is short)

**Goal:** fill pipeline bubbles and cut per-message overhead.

**Tasks**
- Worker pulls up to `k = 4` ready decode steps (from different requests) or waits at most a few ms, then stacks them into `[k, 1, 768]`.
- Different requests have different cache lengths, so you need **left-padding plus an attention mask**, and per-request position IDs.
- Split the outputs back per request and keep all the ordering and epoch rules per item.
- **Correctness check:** tokens identical to the reference. Logits may differ slightly because batching changes the order of additions; compare logits within 1e-4 and document any rare argmax flip honestly.

**Done when:** E3 shows the throughput difference (on vs off) and tokens still match.

---

### Phase 8 — Benchmarks and experiments (5–6 days)

**Goal:** turn the system into numbers and graphs. Log raw data to CSV and generate *every* graph from those files with a script, so every number in the report can be traced to its source.

**What to log per token:** `request_id, step, epoch, t_submit, t_token, bytes_A→B, stage_A_ms, stage_B_ms`.

| # | Experiment | What to vary | What to plot |
|---|---|---|---|
| E1 | Baseline vs pipeline | single process / 2 processes on 1 laptop / 2 laptops | TTFT, per-token latency (mean, p95) |
| E2 | Concurrency scaling | 1, 2, 4, 8 concurrent requests | Throughput rising then flattening; p95 latency rising |
| E3 | Micro-batching | on/off at 4 and 8 requests | Throughput, worker idle time |
| E4 | Communication cost | prompt length 16, 64, 256 | Bytes per prefill vs per decode step |
| E5 | Failure recovery | kill A vs B at early, middle, late step | Recovery time = detection + restart + replay; replayed tokens |
| E6 | Overload | arrival rate above capacity; queue size 4, 8, 16 | Rejected requests, latency, memory |
| (bonus) | Split point | 6/6 vs 7/5 | Per-stage compute time (B has the big LM head) |

Repeat each experiment at least 5 times and report mean and p95.

**Expected story:** pipeline slower than single process for one request (E1). Throughput improves with concurrency and then saturates (E2). Decode bytes are tiny compared with prefill or replay bytes (E4). Recovery cost grows with how late the crash happens (E5). Memory stays bounded under overload (E6).

---

### Phase 9 — Report, README and live demo (4–5 days)

- **README:** install, download the model, start with `launcher.py`, run tests, run benchmarks.
- **Report:** architecture, message format, ordering rules, recovery protocol (draw a timeline), failure assumptions (crash-stop, reliable coordinator), results with graphs, overhead analysis, limitations (coordinator is a single point of failure), future work (continuous batching, keeping A's cache on B-only crashes, float16 activations).
- **Demo script (rehearse it):**
  1. Single process vs pipeline: same text side by side.
  2. Five clients at once, each streaming its own correct text.
  3. `kill -9` Worker B mid-sentence: the client pauses, the log shows *detected → restarted (epoch 2) → replayed*, the text continues, and the final output equals the reference.
  4. Flood the system: clients get "busy", nothing crashes, memory graph flat.

---

## 4. Reference sheets

### 4.1 Suggested repo layout

```
project38/
├── proto/pipeline.proto          # gRPC messages and services
├── src/
│   ├── model_core.py             # your GPT-2 forward + KV cache (Phase 1)
│   ├── stages.py                 # Stage1 / Stage2 classes (Phase 2)
│   ├── serialization.py          # tensor <-> bytes
│   ├── worker.py                 # gRPC server for stage A or B
│   ├── coordinator.py            # request loop, heartbeats, recovery
│   ├── client.py                 # CLI client
│   ├── launcher.py               # start / kill / restart processes
│   └── metrics.py                # CSV logging
├── tests/
│   ├── test_reference.py         # Test 1 + manual loop vs generate()
│   ├── test_pipeline.py          # Test 2
│   ├── test_concurrency.py       # Test 3
│   ├── test_crash.py             # Tests 4, 5 (+ randomised)
│   ├── test_duplicates.py        # Test 6
│   └── test_overload.py          # Test 7
├── bench/                        # E1–E6 scripts, raw CSVs, plot scripts
├── data/prompts.json, reference_outputs.json
├── requirements.txt              # pinned!
└── README.md
```

### 4.2 Message format (protobuf sketch)

```protobuf
enum Kind   { PREFILL = 0; DECODE = 1; REPLAY = 2; RELEASE = 3; HEARTBEAT = 4; }
enum Status { OK = 0; BUSY = 1; OUT_OF_ORDER = 2; STALE_EPOCH = 3; DUPLICATE = 4; }

message Tensor { bytes data = 1; repeated int32 shape = 2; string dtype = 3; }

message StageRequest {
  string request_id = 1;
  int32  step       = 2;     // 0 = prefill, 1.. = decode
  int64  epoch      = 3;     // incarnation of the receiving worker
  Kind   kind       = 4;
  repeated int32 token_ids = 5;   // to Worker A
  repeated int32 positions = 6;
  Tensor hidden     = 7;     // to Worker B
}

message StageReply { Status status = 1; int32 next_token = 2; int32 step = 3; int64 epoch = 4; }

service Stage {
  rpc Forward(StageRequest) returns (StageReply);
  rpc Release(StageRequest) returns (StageReply);
  rpc Heartbeat(StageRequest) returns (StageReply);
}
```

### 4.3 Test matrix (from the proposal)

| # | Test | Passes when | Built in |
|---|---|---|---|
| 1 | 20 prompts × 50 tokens, single process | Saved as ground truth | Phase 1 |
| 2 | Same prompts through pipeline, one at a time | Identical tokens | Phase 3 |
| 3 | 10 prompts concurrently | Each identical to its own reference | Phase 4 |
| 4 | Kill Worker B at a random step | Identical; no token delivered twice | Phase 6 |
| 5 | Kill Worker A at a random step | Identical; no token delivered twice | Phase 6 |
| 6 | Inject duplicate and old-epoch messages | All discarded; output unchanged | Phase 4 |
| 7 | Flood beyond queue capacity | BUSY or rejection; no crash; memory bounded | Phase 5 |

### 4.4 Top risks and what to do

| Risk | What to do |
|---|---|
| HF cache format changes | Own KV cache in `model_core.py` and pinned versions |
| Output "almost" matches | Check positions first, then the causal mask, then `ln_f`, then dtype |
| Race conditions in recovery | Pause all loops during recovery; per-request locks; randomised crash tests run many times |
| gRPC hidden retries | Disable them; every retry goes through your own step and epoch logic |
| Both workers on one laptop fight for CPU | Set `torch.set_num_threads` per worker; mention it in the report |
| Schedule slips | Drop Phase 7 first; never cut Phase 6 or Phase 8 |

### 4.5 Work breakdown (MD TAUFIQUE HUSSAIN)

| Phase | MD TAUFIQUE HUSSAIN |
|---|---|
| 0–2 | `model_core.py`, reference, Stage classes, `.proto`, gRPC skeleton, launcher, serialization |
| 3–4 | Worker ordering and epoch logic, coordinator loop, concurrency, client streaming |
| 5–6 | Recovery protocol (replay, fencing), heartbeats, crash tooling, backpressure |
| 7–9 | Micro-batching, benchmarks, plots, README, report, demo script |

### 4.6 Your next three actions

1. Phase 0: repo, virtual environment, pinned `requirements.txt`, GPT-2 saved locally.
2. Write `prompts.json` (20 prompts).
3. Start `model_core.py` and get the manual loop to match `generate()` on one prompt. Once that works, everything else builds on it.
