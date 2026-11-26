# autoresearch on this Mac: notes

These notes cover Karpathy's autoresearch (<https://github.com/karpathy/autoresearch>): how the loop works, which Mac
fork we use, what ran on this machine, and what to watch out for. The H&M version of the loop is drafted in
[`hm_program.md`](hm_program.md).

## How the loop works

autoresearch is a fixed experiment harness plus a Markdown brief that turns a coding agent into an overnight researcher.
Three files matter:

| File | Edited by | Role |
|---|---|---|
| `prepare.py` | nobody (fixed) | One-time data download and BPE tokenizer training, plus runtime utilities: the dataloader, the constants (`MAX_SEQ_LEN`, `TIME_BUDGET = 300`, `EVAL_TOKENS`) and the scorer `evaluate_bpb`. This is the ground truth, and the agent may not touch it. |
| `train.py` | **the agent** | The whole GPT model, the optimizer and the training loop, in one file. Anything goes: architecture, hyperparameters, batch size, model size. |
| `program.md` | **the human** | The agent's instructions. You program the research org in Markdown instead of editing Python. |

The rules that make it work:

- **Fixed time budget.** Each run trains for exactly 5 minutes of wall time, not counting startup and eval. Runs are
  comparable whatever the agent changes: a bigger model just gets fewer steps. The loop finds the best model *for this
  machine in 5 minutes*, so results don't transfer to other hardware.
- **One metric.** `val_bpb`, validation bits per byte. Lower is better. It doesn't depend on vocab size, so tokenizer
  and architecture changes compare fairly. Memory is a soft constraint, and simplicity is a stated tie-breaker. Deleting
  code at equal score is a win.
- **Keep or discard through git.** The agent works on a branch `autoresearch/<tag>`. Each step is:
  1. edit `train.py`;
  2. commit;
  3. run `uv run train.py > run.log 2>&1`;
  4. grep `val_bpb` and `peak_vram_mb`.

  If the score improved, the branch advances. If not, `git reset` back. A crash gets one or two cheap fixes, then it's
  logged and skipped. A run past 10 minutes is killed.
- **Results log.** `results.tsv`, untracked, tab-separated:
  `commit  val_bpb  memory_gb  status(keep|discard|crash)  description`. The kept rows trace the improvement curve.
- **Never stop.** Once it's going, the agent doesn't ask whether to continue. At about 12 runs an hour that's about 100
  experiments a night on an H100. On this Mac it's fewer (see below).

## Choosing a Mac fork

Upstream needs an NVIDIA GPU: it loads FlashAttention-3 kernels, uses `torch.compile`, and has `cuda` hard-coded. The
README links two macOS forks. I cloned both and diffed them against upstream `228791f` (2026-03-25).

| | `miolini/autoresearch-macos` (PyTorch MPS) | `trevin-creator/autoresearch-mlx` (MLX) |
|---|---|---|
| Last code change | 2026-03-07. Hasn't merged upstream's commits since 03-08. | 2026-07-02: a fix that bounds eval memory, a bf16 fix, and `rigor.py` |
| Diff vs upstream | Small. About 22 lines in `prepare.py`, about 126 lines in `train.py` (FA3 replaced with SDPA, device detection) | `prepare.py` is a faithful 1:1 port. torch was swapped for mlx/numpy, and the constants are the same except `EVAL_TOKENS`. `train.py` is a full MLX rewrite with the same structure and hyperparameter names. `program.md` has only small wording changes. |
| Things to watch | Runs in fp32 on MPS, with no autocast and no `torch.compile`. Reports `peak_vram_mb` as 0 on MPS, so it can't log memory. Drops upstream's NaN fast-fail. Sliding windows use a dense T×T mask. | AdamW only: there's no Muon port yet. `mfu_percent` is always 0. Also no NaN check, only a `loss > 100` check. |
| Dependencies | torch 2.6.0 | mlx ≥ 0.30 (0.31.0 installed). No torch at all. |

**I picked the MLX fork.** It is recent and maintained. Its `prepare.py`, the fixed half of the harness, is a
line-for-line port. It reports real peak memory (`mx.get_peak_memory`). It already fixed the final-eval out-of-memory
crash on smaller Macs. The MPS fork is the smaller diff, but it's eight months stale and can't report memory. The MLX
`train.py` is a rewrite, but that is unavoidable when porting to another framework. I read it: same model structure, no
network calls, no subprocesses. Its only subprocess is `rigor.py` calling `uv run train.py`.

## What ran

- **Machine:** Apple M2 Pro, 16 GB unified memory. MLX reports `max_recommended_working_set_size` as 11.45 GB and the
  max buffer as 8 GB. Python 3.12, `uv` 0.13.10 (installed with Homebrew).
- **Clone:** `~/git/autoresearch-mac`, from upstream fork commit `766a25f`. The local branch `m2pro-16gb` holds the
  config change (commit `4c2213a`).
- **`uv sync`:** OK.
- **`uv run prepare.py`:** downloaded 10 train shards plus the pinned val shard (about 91 MB each, 1.0 GB in
  `~/.cache/autoresearch/`). Trained the 8192-token BPE tokenizer in 42 s. Total time 132 s.
- **First `train.py` run with the fork's defaults** (DEPTH 4, seq 2048, batch 16, 64K tokens per step, `SSSL`): I killed
  it. The machine already had about 20 GB of swap in use from open apps (Chrome, Slack, Tana, Spotify, the Claude
  apps). The run pushed the swap to its limit. Step 0 took 26 s, and step 1 hadn't finished after another 30 s. Log:
  `~/git/autoresearch-mac/run_default_killed.log`.
- **Shrunk config, as the upstream README suggests for small machines:**

  | Setting | Fork default | This Mac |
  |---|---|---|
  | `MAX_SEQ_LEN` (prepare.py) | 2048 | **512** |
  | `EVAL_TOKENS` (prepare.py) | 3 × 524288 | **1 × 524288** |
  | `WINDOW_PATTERN` | `SSSL` | **`L`** |
  | `TOTAL_BATCH_SIZE` | 2^16 | **2^15** |
  | `DEVICE_BATCH_SIZE` | 16 | **32** (16K tokens per forward pass, 2 grad-accum steps) |
  | `FINAL_EVAL_BATCH_SIZE` | 256 | **64** |
  | `DEPTH` | 4 | 4 (unchanged) |

  Changing `prepare.py` is a one-time setup decision made by the human. The agent still must not edit it. Because
  `MAX_SEQ_LEN` and `EVAL_TOKENS` changed, our `val_bpb` numbers can't be compared with the fork's published numbers,
  only with each other.

### Baseline result (shrunk config, one 5-minute run)

```
val_bpb:          1.669670
training_seconds: 300.5
total_seconds:    305.4
peak_vram_mb:     3990.6
total_tokens_M:   17.1
num_steps:        521
num_params_M:     11.5
depth:            4
```

- **val_bpb 1.669670.** For reference, the fork's published baselines on bigger Macs at seq 2048 are 1.81–2.67 before
  the loop, and 1.29–1.35 after a night of it. Those aren't directly comparable, because the sequence length and eval
  set are different.
- **Peak memory 3.9 GB** (MLX peak). The process's max RSS was 0.39 GB. On Apple Silicon the GPU allocations don't
  show up as RSS.
- **Wall time 306 s** end to end. It trained at about 55–59K tokens/s, about 0.58 s per step. Eval took 3.7 s, and
  start-up took under 1 s.

That's about 11 runs an hour, or about 80 overnight, if the agent's own thinking time between runs is short.

## Problems and caveats

- **The machine's memory is the real constraint.** 16 GB of RAM with about 20 GB of swap already in use before the run
  is not a clean benchmark box. For an overnight loop, close Chrome, Slack and similar apps. Treat 4 GB of MLX peak as
  the soft cap and about 8 GB as the hard limit, or the run starts swapping and the 5-minute budget measures disk I/O
  instead of the model. The agent will happily make the model bigger, so put the cap in `program.md`.
- **Noise.** The fork's README and `rigor.py` say that re-running the same `train.py` moves `val_bpb` by about 0.03
  (non-deterministic GPU reductions). Many single-run "wins" are inside that band. If we run the loop, use
  `uv run rigor.py run "<desc>"`: it runs 3 seeds and keeps a change only with bootstrap confidence. The alternative is
  to tell the agent to ignore deltas under 0.03.
- **The fork has no Muon.** Upstream's Muon+AdamW is replaced by AdamW only. The loop can rediscover the gap, but
  results won't line up with H100 runs.
- **Swap pressure varies over time**, so throughput can drift if other apps wake up. That makes the fixed time budget
  less fair between runs. That's one more reason to run the loop with the machine otherwise idle.
- **Not started:** the autonomous agent loop. Ask the user before you start it. It is meant to run for hours without
  asking, editing and committing on a branch of `~/git/autoresearch-mac`.

## How to start the loop later (once the user agrees)

```bash
cd ~/git/autoresearch-mac
git checkout -b autoresearch/<tag> m2pro-16gb
# then start a coding agent in that folder with permissions set to allow edits and runs, and prompt:
#   "Have a look at program.md and let's kick off a new experiment. Do the setup first."
```

Add to `program.md` before you start: a memory cap (peak_vram_mb ≤ 6000), the `rigor.py` keep rule, and "never edit
prepare.py".

## Applying it to H&M

See [`hm_program.md`](hm_program.md). The mapping:

| autoresearch | H&M version |
|---|---|
| `prepare.py` | `hm_prepare.py`: CSV→parquet, the last-week holdout split, the MAP@12 scorer, a time and memory watchdog |
| `train.py` | `hm_train.py`: candidate generation + features + LightGBM ranker |
| `val_bpb` | `map12` on 2020-09-16..22 (higher is better) |
| 5 min of training | ≤ 15 min for the whole run, hard kill at 18 min or 12 GB RSS |
| `results.tsv` | the same idea, plus `cand_recall`, `peak_rss_gb` and `wall_s` columns |

One difference matters. A single validation week is small, so a hundred keep/discard decisions will overfit it. The H&M
brief adds a human-only `check` mode, which repeats the full pipeline one week earlier, and a `submit` mode for a real
Kaggle LB check. Neither `hm_prepare.py` nor `hm_train.py` exists yet. Building them is the next step once the H&M
data is local and the main session has a first two-stage ranker.
