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
| `item_cold_start.py` | Popularity predicted from the POI embedding vs Wikipedia page views. |
| `loo_eval.py`, `onboarding_sim.py`, `transfer_eval.py` (superseded by `transfer_service.py`) | Design experiments. They compare against the previous recommender (styles, bandit onboarding), so they run at commit `74f33b6`, before it was removed, and use fixed assignment radii. |

Run from the repository root: `PYTHONPATH=research python research/<script>.py`.

Results with real OSM shapes (`eval_service.py`, 486 cases): service 0.677
hit@10, popularity only 0.547, service with Wikipedia popularity 0.455,
random 0.043.
