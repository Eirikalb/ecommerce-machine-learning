# Plan: replicate the top H&M solutions locally

Target: get from the notebook's best baseline (MAP@12 **0.0271**, repurchase + bought-together + popular) to the level of the winning solutions with their technique.

| Reference | Score |
|---|---|
| 1st place, best single LightGBM | CV 0.0441 / LB 0.0367 |
| 1st place, ensemble of 5 LightGBM + 7 CatBoost | LB 0.0371 |
| Open-source two-stage rebuild (6 recall channels + LambdaMART, built on Apple Silicon) | private 0.0360 |
| 45th place (silver), 2 recall strategies | private 0.0300 |

## What the winners did (the short version)

- **Two stages.** First generate about 100 candidates per customer, then rank them with gradient-boosted trees and keep the top 12.
- **Recall carried most of the gain.** Recently popular items and the customer's own repurchases matter most. The 1st-place write-ups say they started with ~600 candidates, then used a GBDT to cut them to ~130. Image and text features did not help in their tests.
- **Interaction features beat content features.** These cover customer × article, customer × product type, and how recent and how frequent each pair is. Because ~50% of customers have no purchases in the last 3 months, they added cumulative long-window features for them.
- **Validation.** Train on about 6 weekly folds and validate on the last week. 100 candidates per customer gave a stable CV-to-LB match.
- **Engineering.** Label-encode the categoricals and cache intermediate features in a feature store, so nothing is recomputed.

## Steps

1. **Local baseline parity.** Run `hm_fashion_exploration.ipynb` and match the Kaggle numbers: 31,788,324 purchases and 0.0271.
2. **Fixed harness, `hm/prepare.py`.** Load the CSVs once and write parquet with integer IDs. Build the weekly folds (anchor dates, truth per week) and one MAP@12 scorer. This file stays frozen afterwards, which is also what autoresearch needs.
3. **Recall, `hm/recall.py`.** Add the sources one at a time and log recall@100 for each:
   repurchase (last N weeks), popular overall and by age bucket, popular within the customer's top product types/departments, bought-together (item-item co-purchase), same `product_code` siblings, ItemCF, then embeddings (word2vec over purchase sequences, ProNE/ALS).
4. **Features, `hm/features.py`.** Customer, article and customer × article recency/frequency/price features, plus each candidate's source and rank within that source. Everything is computed strictly before the fold's anchor date.
5. **Ranker, `hm/train.py`.** Use an LGBM binary classifier first (that is what 1st place used), then `LGBMRanker` with lambdarank, then CatBoost. Train on 4–6 weeks of folds and validate on the last week.
6. **Ensemble and full run.** Blend the rankers and score every customer. Compare the result with the reference table above.

## Constraints on this Mac (16 GB RAM)

- 100 candidates × ~1.37M customers × several weeks is too big for pandas in memory. Use polars or DuckDB on parquet, compact dtypes, and for each training week only the customers who bought something. Score the full customer base in chunks.
- Iterate on a 10–20% customer sample and confirm that the sample score follows the full score before relying on it.

## Link to autoresearch

Once step 2 exists, the loop maps directly: `prepare.py` is the frozen harness, `train.py` is recall + features + ranker, the single metric is MAP@12 on the held-out week, and each run gets a fixed time budget. The setup is being worked on separately; see `autoresearch/` on the `autoresearch-setup` branch.

## Sources

- [1st place write-up (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution)
- [1st place summary slides](https://speakerdeck.com/kfujikawa/h-and-m-personalized-fashion-recommendation)
- [HaokunLiu-Vincent/hm-recsys](https://github.com/HaokunLiu-Vincent/hm-recsys)
- [Wp-Zhang/H-M-Fashion-RecSys (45th)](https://github.com/Wp-Zhang/H-M-Fashion-RecSys)
