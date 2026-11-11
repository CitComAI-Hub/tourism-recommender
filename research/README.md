# Research scripts

Evaluation of the recommender on real visitor behaviour (Flickr YFCC100M
photos). Data lives in `data/raw/yfcc/` and is not versioned.

| Script | What it measures |
|---|---|
| `yfcc_scan.py` | Streams the YFCC100M metadata (65 GB SQLite on AWS Open Data) and keeps the European geotagged rows. |
| `eval_service.py` | **Current reference.** Exports the pseudonymised Valencia photos and evaluates the production `Recommender` (5 folds by user, leave-one-out on tourists). |
| `transfer_service.py` | Transfer to another TEF city (Milano) with the production code: Valencia's model 0.253 hit@10, no model 0.300, Milano's own model 0.660. |
| `protocol.py` | Evaluation protocol for any city: weights chosen on validation, users and time splits, bootstrap and McNemar against popularity. |
| `traits.py` | Onboarding signals simulated with real tourists: likes as anchors are the only one with value; themes, popularity affinity, radius and similarity add nothing. |
| `run_tef_protocol.sh` | Held-out confirmation on the TEF cities (run once, frozen code). 2026-11-11 at 7ed2a4a: service beats popularity in all 5 cities with data (+0.17..+0.46 hit@10, users split); Mechelen too small (22 cases). |
| `external_check.py` | Learnt popularity vs **official visitor counts** (`external_visitors.csv`: Valencia yearbook 2024, 29 places; Comune di Milano open data 2004-2017, CC BY, 23 places). Spearman: Flickr 0.70 Valencia, 0.86 Milano; Wikipedia 0.79 / 0.71; content-predicted (masked) 0.67 / 0.73. Museums under-photographed, towers and squares over-photographed, in both cities. |
| `external_blend.py` | Wikipedia mixed into the Flickr popularity: closer to the official counts (beta 0.25: 0.75 / 0.88), but production hit@10 0.754 -> 0.733 (beyond the 0.01 margin): `wikipedia_weight` off by default. |
| `context.py` | Context variants on the production formula, chosen on validation: second-order and decayed next-POI, category-smoothed transitions, distance from the last visit, hour of day, weekend, trip length. None beats the formula (Valencia +0.000, Milano +0.001 users split). |
| `cards.py` | Onboarding cards: adaptive / near-position cards get more likes (2.58 vs 2.27 Valencia, 3.5 vs 3.3 Milano) but no more hit@10; popular cards stay (Milano adaptive -0.005..-0.009). |
| `one_config.py` | One ranking for every city, weights from the other city (leave-city-out): -0.015 Valencia, -0.010 Milano, n.s. -> the per-city tuning adds little (no overfitting to one city). A learnt linear ranker does not beat the formula (-0.010 / -0.021). |
| `item_cold_start.py` | Popularity predicted from the POI embedding vs Wikipedia page views. |
| `loo_eval.py`, `onboarding_sim.py`, `transfer_eval.py` (superseded by `transfer_service.py`) | Design experiments. They compare against the previous recommender (styles, bandit onboarding), so they run at commit `74f33b6`, before it was removed, and use fixed assignment radii. |

Run from the repository root: `PYTHONPATH=research python research/<script>.py`.

Results with real OSM shapes (`eval_service.py`, 486 cases): service 0.677
hit@10, popularity only 0.547, service with Wikipedia popularity 0.455,
random 0.043.
