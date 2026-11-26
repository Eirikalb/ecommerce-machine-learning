# autoresearch: H&M Personalized Fashion Recommendations

This is a draft brief, written in the style of Karpathy's autoresearch `program.md`. It applies the same loop to the
H&M recommender. Read `NOTES.md` in this folder for how the original loop works. **Status:** design only. The files it
names (`hm_prepare.py`, `hm_train.py`, `pyproject.toml`) do not exist yet. Build them, and check that one manual run
works, before you point an agent at this file.

The task: for each customer, predict 12 articles they will buy in the next 7 days. The metric is MAP@12. We only have
purchases (no clicks, no views). The best simple baseline from `hm_fashion_exploration.ipynb` (repurchase +
bought-together + popular) scores **MAP@12 0.0271** on the last-week holdout. The top Kaggle solutions are two-stage:
recall roughly 50–200 candidates per customer, then rank them with a GBDT. The 1st place scored about 0.037 on the private LB.

## Files

The repo is deliberately small. Only three files matter:

| File | Who edits it | What it holds |
|---|---|---|
| `hm_prepare.py` | nobody (fixed) | One-time data prep, the time split, the MAP@12 scorer, budget enforcement and the summary printout |
| `hm_train.py` | **the agent** | Candidate generation, features, the LightGBM ranker, and the final top-12 lists |
| `hm_program.md` | the human | This brief |

### `hm_prepare.py` (fixed, never edited by the agent)

- **One-time prep** (`uv run hm_prepare.py`): reads the three Kaggle CSVs from `$HM_DATA_DIR` and writes compact
  parquet files to `~/.cache/hm/`:
  - `transactions.parquet`: `t_dat` as date, `cid` int32, `article_id` int32, `price` float32, `sales_channel_id` int8.
  - `articles.parquet` and `customers.parquet`.
  - `customer_ids.parquet`, which maps `cid` back to the hash. Only the submission mode uses it.

  This takes the raw ~3.5 GB CSV down to roughly 0.5 GB in memory.
- **Constants:**
  ```python
  HOLDOUT_START = "2020-09-16"   # last week in the data: 2020-09-16 .. 2020-09-22
  HOLDOUT_DAYS  = 7
  K             = 12
  TIME_BUDGET   = 900            # seconds of wall clock for a whole hm_train.py run
  MEM_SOFT_GB   = 10.0           # what a run should stay under
  MEM_HARD_GB   = 12.0           # the watchdog kills the run above this
  SEED          = 42
  ```
- **`load_split(mode="val")`** returns a `Split` with:
  - `tx`: only the transactions **before** `holdout_start`. `hm_train.py` never sees holdout rows.
  - `articles` and `customers`.
  - `holdout_start`: a date.
  - `target_customers`: the `cid`s to predict for.

  It also starts the budget watchdog: a daemon thread that reads `resource.getrusage().ru_maxrss` and the elapsed time
  once a second. If peak RSS goes over `MEM_HARD_GB`, or the elapsed time goes over `1.2 × TIME_BUDGET`, it prints
  `BUDGET_KILL: <reason>` and calls `os._exit(3)`.

  Modes:
  - `val` (the agent's only mode): holdout = 2020-09-16..22. `target_customers` = everyone who bought in that week
    (about 68k). Scoring is only over them, so predicting only for them gives the same score. Membership of this list
    must never be used as a feature.
  - `check` (human only): the same thing shifted one week back (holdout 2020-09-09..15, tx before 2020-09-09). This is
    how we catch overfitting to the one holdout week (see "Guarding the holdout").
  - `submit` (human only): tx is all data. `target_customers` is all 1.37M customers. Writes `submission.csv` for the
    real hidden week (2020-09-23..29), which the human uploads to Kaggle.
- **`evaluate_map12(preds)`**: `preds` is `{cid: [article_id, ...]}`. It loads the holdout answers itself, deduplicates
  and truncates each list to 12, and scores MAP@12 over the customers who bought in the holdout week. A customer with no
  prediction scores 0. This is the same `apk`/`mapk` as the notebook, so the numbers line up with 0.0271.
- **`candidate_recall(cands)`**: `cands` is `{cid: iterable of article_id}`. It returns the share of holdout purchases
  (customer, article) that appear in the candidate set, plus the mean number of candidates per customer. It is a
  diagnostic, not the metric. It tells the agent whether a run is limited by recall or by ranking.
- **`report(map12, recall, n_cands)`** prints the fixed summary block (below), using the wall time and peak RSS the
  watchdog tracked.

### `hm_train.py` (the agent edits this)

Everything below is fair game: candidate sources, how many to take from each, features, the training-week windows, the
negative sampling, the LightGBM objective (`lambdarank`, `binary`, `xendcg`) and its parameters, ensembling a few
rankers, and the fallback fill. The file must:

1. Call `split = load_split("val")` first, and use `split.holdout_start` for every date cut. Never hard-code dates. That
   way the same file runs unchanged in `check` and `submit` mode.
2. Build ranker training data only from weeks before `holdout_start`. For example, use weeks t-1..t-4 as label weeks,
   with features computed from data strictly before each label week.
3. Produce `preds` (12 articles for every `cid` in `split.target_customers`) and `cands` (the candidate set that was
   ranked).
4. End with `report(evaluate_map12(preds), *candidate_recall(cands))`.

**The starting `hm_train.py`** should be the main session's first working two-stage ranker. If that isn't ready, start
from a port of the notebook baseline: candidates from repurchase, bought-together and popular, plus a small LightGBM
LambdaRank on a handful of features. Its first run is the baseline row in the results log.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**, e.g. `hm-nov27`. The branch `autoresearch/<tag>` must not exist yet.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from the current main.
3. **Read the in-scope files:** this brief, `hm_prepare.py` (to learn the API and the rules it enforces) and
   `hm_train.py` (the file you modify).
4. **Check that the data exists:** `~/.cache/hm/transactions.parquet` must be there. If it isn't, tell the human to run
   `uv run hm_prepare.py`.
5. **Initialize `results.tsv`** with just the header row. Leave it untracked by git.
6. **Confirm and go.**

## Experimentation

Each experiment is one run of `hm_train.py`:
`uv run hm_train.py > run.log 2>&1`. Never use `tee`, and don't let the output flood your context.

**What you CAN do:**
- Edit `hm_train.py`. That is the only file you edit.

**What you CANNOT do:**
- Edit `hm_prepare.py`, the scorer, the split, the budget constants or the watchdog.
- Read holdout rows in any other way. Don't open the parquet files directly with a date filter that reaches into the
  holdout week, and don't call `load_split` with any mode other than `"val"`.
- Use `target_customers` membership, or anything derived from the holdout week, as a feature or to filter candidates.
- Install packages. You can use what `pyproject.toml` already has: `pandas`, `numpy`, `pyarrow`, `polars`, `lightgbm`,
  `scipy`.
- Use the GPU or network.

**The goal: the highest `map12`.** Unlike nanochat, a GBDT pipeline doesn't stop by itself when the time is up, so the
budget is a ceiling rather than a fixed length:

- **Time:** the whole run (load, candidates, features, training, inference, scoring) should fit in `TIME_BUDGET`
  (15 min) of wall time. The watchdog kills it at 18 min, and that counts as a crash. A faster run with the same score
  is a win.
- **Memory:** stay under 10 GB peak RSS on this 16 GB Mac. The watchdog kills the run at 12 GB. The usual ways to blow
  up memory are cross-joining customers × candidates × weeks, or `object` dtype columns. Keep ids as int32, features as
  float32, and drop each intermediate frame once you've used it.
- **Determinism:** pass `seed=SEED` and `deterministic=True` to LightGBM, and seed any sampling. Two runs of the same
  commit should agree to the 5th decimal.

**When to keep a change:**
- Keep it if `map12` improves by **≥ 0.0002**. That's about 0.7% of the baseline, and above what changing the sampling
  seed moves it.
- For a smaller gain, re-run with `SEED` overridden to 2 other values (`HM_SEED` env var). Keep it only if the mean of
  the 3 runs beats the current best. This is the idea behind `rigor.py` in the MLX fork.
- Otherwise discard it.
- **Simplicity:** apply the same rule as upstream. A 0.0002 gain that adds 60 lines of fragile feature code is probably
  not worth it. Deleting a candidate source or a feature with no loss is a win.

**Where the known wins are** (from the public top solutions), roughly in order of payoff:
1. Candidate recall:
   - repurchase, ranked by recency and frequency;
   - bought-together pairs, with time decay;
   - popular items by age bucket or by recent week;
   - same `product_code` as the customer's past items (other colours or sizes);
   - item2item similarity from co-purchase.

   Watch `cand_recall` against `n_cands`.
2. Features:
   - customer × item recency, count and time-decayed count;
   - each candidate source and its rank inside that source, as features;
   - article popularity trend (last week vs 4 weeks ago);
   - price relative to the customer's usual spend;
   - customer age vs the article's buyer-age profile;
   - article attributes.
3. Training data: more label weeks, negative downsampling, dropping customers with no positives from training.
4. The objective and its parameters, then a small ensemble.

The first run must establish the baseline: run `hm_train.py` as is.

## Output format

`report()` prints:

```
---
map12:            0.027100
cand_recall:      0.1840
n_cands:          63.2
wall_seconds:     412.7
peak_rss_gb:      6.3
n_target:         68984
holdout_start:    2020-09-16
```

If the watchdog killed the run, the log contains `BUDGET_KILL: memory` or `BUDGET_KILL: time` and no `map12:` line. Read
the results with:

```
grep "^map12:\|^cand_recall:\|^wall_seconds:\|^peak_rss_gb:\|^BUDGET_KILL" run.log
```

## Logging results

Log every run to `results.tsv` (tab-separated, untracked by git). It has 7 columns:

```
commit	map12	cand_recall	peak_rss_gb	wall_s	status	description
```

1. Short git commit hash (7 chars).
2. `map12` to 6 decimals. Use 0.000000 for a crash or a budget kill.
3. `cand_recall` to 4 decimals. Use 0.0000 for a crash.
4. Peak RSS in GB to 1 decimal. Use 0.0 for a crash.
5. Wall seconds, rounded to whole seconds.
6. Status: `keep`, `discard`, `crash` or `budget` (killed by the watchdog).
7. A short description with no tabs.

Example:

```
commit	map12	cand_recall	peak_rss_gb	wall_s	status	description
a1b2c3d	0.027900	0.1840	6.3	413	keep	baseline two-stage: repurchase+pairs+popular, lambdarank 300 trees
b2c3d4e	0.028600	0.2120	7.1	455	keep	add same-product_code candidates (top 10)
c3d4e5f	0.028500	0.2120	7.1	460	discard	num_leaves 63
d4e5f6a	0.000000	0.0000	0.0	1080	budget	8 label weeks without downsampling (memory)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Edit `hm_train.py` to try one idea.
3. Commit it.
4. Run `uv run hm_train.py > run.log 2>&1`.
5. Grep the results (command above). If they're empty, run `tail -n 50 run.log` for the traceback. Fix a trivial bug
   and re-run. If the idea itself is broken, log it as `crash` and move on.
6. Log the run in `results.tsv`.
7. If `map12` improved by the keep rule, keep the commit: the branch advances.
8. If not, `git reset --hard` to the last kept commit.

**Timeout:** if a run passes 20 minutes and the watchdog hasn't fired, kill it yourself and log it as `budget`.

**NEVER STOP:** once the loop has started, don't stop to ask whether to continue. The human may be asleep. At about 3–4
runs an hour, an overnight session gives roughly 30 experiments. When you run out of ideas, re-read the list of known
wins, then:
- look at which candidate source supplies the hits;
- look at where the misses are: customers with no history vs with history;
- try combining near-misses.

## Guarding the holdout (for the human)

A hundred keep/discard decisions on one week of data will overfit that week. autoresearch on val_bpb doesn't have this
problem as badly, because its validation shard is huge. Here the agent never sees any number except the `val` week.
The human:

- Every ~10 keeps, or each morning, runs the current best commit in `check` mode. That repeats the full pipeline one
  week earlier. If `check` stops rising while `val` keeps rising, the loop is fitting the week, so tighten the keep
  threshold or reset.
- Uses `submit` mode only on milestones, and uploads `submission.csv` to Kaggle by hand. The Kaggle private LB is the
  real, untouched test. The leaderboard reference points are about 0.0271 for our baseline, 0.0295+ for a good single
  ranker, and about 0.037 for 1st place.

## Hardware notes

Apple M2 Pro, 16 GB unified memory, no CUDA, so LightGBM runs on CPU threads only. Assume about 12 GB is really
available, since the OS and the agent use some. That only holds with other apps closed. During the nanochat baseline
(see `NOTES.md`), Chrome, Slack and the rest already had about 20 GB in swap, and a 5–6 GB job pushed the machine into
thrashing.

To keep a run inside the budget:
- Use the last 8–12 weeks of transactions for features.
- Use 1–4 label weeks for training.
- Take 50–100 candidates per customer.
- If the training frame goes over about 20M rows, downsample negatives (for example to 1:30).

The 1st-place style setup (hundreds of features, about 30M training rows) won't fit in 15 min on this machine. The loop
should find the best pipeline for this budget, not copy theirs.
