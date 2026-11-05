# PMV IA Turismo (Valencia)

## Goal
Build a modular, scalable minimum viable product (MVP) that combines
**recommendation** and **itinerary planning** for tourists in Valencia, with
interchangeable components.

## MVP scope
- POI ingestion (OpenStreetMap enriched with Wikidata/Wikipedia/Commons).
- Content-based recommendation for cold-start users, driven by an active
  onboarding.
- Itinerary planning with time, distance and opening-hours constraints.
- A small API to query POIs, recommendations and itineraries.
- Reproducible offline evaluation.

## Modules (decoupled layers)
1. **Ingestion/ETL**
2. **POI catalog**
3. **Recommendation** (onboarding + ranking)
4. **Planning**
5. **API**
6. **Evaluation**

## Principles
- File-based configuration (YAML/JSON) to swap components.
- Clear interfaces between modules.
- Incremental evolution.

## Setup (Linux)
1. Create a virtual environment:
	python3 -m venv .venv
2. Activate it:
	source .venv/bin/activate
3. Install dependencies:
	pip install -r requirements.txt
4. Install the package in editable mode:
	pip install -e .
5. Secrets: copy `.env.example` to `.env` and fill in `ORS_API_KEY`.
   `.env` is git-ignored; keys never go into `configs/base.yaml`.

The embedding dependencies (`sentence-transformers`, `torch`) live in the
`embeddings` extra: `pip install -e ".[embeddings]"`. They are needed to
regenerate the POI embeddings and, at runtime, to use the **free-text
interests** typed during onboarding (without them the API ignores that text
and logs a warning). The repository ships the catalog
(`data/processed/poi.jsonld`) and its embeddings, so the rest works without
re-ingesting.

## Tests
	pip install -e ".[dev]"
	pytest

## CLI
The CLI runs individual pipeline steps.

- Ingestion: `python -m pmv_turismo.cli --config configs/base.yaml --step ingestion`
- Catalog: `python -m pmv_turismo.cli --config configs/base.yaml --step catalog`
- Embeddings: `python -m pmv_turismo.cli --config configs/base.yaml --step embeddings`
- Recommendation: `python -m pmv_turismo.cli --config configs/base.yaml --step recommendation`
  (needs `recommendation.liked_pois` in the config; skipped when empty)
- Planning: `python -m pmv_turismo.cli --config configs/base.yaml --step planning`
- Evaluation: `python -m pmv_turismo.cli --config configs/base.yaml --step evaluation`
- Everything: `python -m pmv_turismo.cli --config configs/base.yaml --step all`

## Recommendation
`docs/flow.md` walks through the whole flow step by step. In short:

**Onboarding (active learning).** The user picks at least 2 categories and can
optionally describe their interests in free text ("Roman ruins and
prehistory"). Then they swipe through rounds of POI cards (6, 4 and 4 by
default, `NEXT_PUBLIC_ONBOARDING_ROUND_SIZES`), served by `/poi/sample`:
- Round 1: representative and well-known POIs of the chosen categories,
  spread out with MMR.
- Later rounds: a Bayesian estimate of P(like) per POI (Beta posterior
  smoothed over embedding similarity), ranked optimistically (mean +
  uncertainty, UCB): liked regions are refined, unexplored ones probed,
  disliked ones dropped.
- The free-text interests are embedded with the same model as the POIs and
  add a similarity term in every round (`active_learning_interest_weight`).
- If a round comes back short, the category pool is exhausted and no more
  rounds are requested.

**Ranking.** Each candidate gets a weighted score (weights normalised to 1):
semantic affinity to the user's likes (multi-interest: up to 3 clusters, minus
similarity to dislikes) blended with the free-text interests
(`interest_text_weight`), popularity, distance and a bonus for the categories
chosen in onboarding (`category_weight`). MMR then diversifies the top-k.
Three user-facing styles (`explorer`, `conservative`, `hidden_gems`) override
the weights in `recommendation.styles`. POIs that are closed for the walk
there plus the whole visit are filtered out.

Users who reject every onboarding card still get recommendations: the ranking
falls back to their interests, categories, popularity and distance.

## Offline evaluation
`--step evaluation` writes `data/processed/evaluation.json` and
`evaluation_report.txt`. Opening hours are disabled so results do not depend
on the clock. It has two parts.

**Leave-one-out over likes.** For every profile with at least two likes, each
like is held out in turn, the recommender runs with the rest and we check
where the held-out POI lands. Profiles are the stored ones in
`user_profiles.json` (if any) plus deterministic synthetic profiles built from
categories (`evaluation.synthetic_users`, `evaluation.seed`). It compares the
styles against `semantic_mean`, `semantic_multi`, `popularity` and `random`
and reports hit@k, nDCG@k, intra-list diversity (ILD), novelty, coverage,
dislike leak and mean distance. Synthetic likes are category-driven and the
embedding text includes the category, so semantic methods are favoured: use
it to compare variants, not as a product metric.

**Hand-labelled personas** (`data/eval/personas.json`, 18 tourist profiles).
Each persona has the categories it would pick in onboarding, the free text it
would type as interests and the POIs it would want to visit. Relevance was
judged by an LLM (Claude) from each POI's name and description, not from the
embeddings, so the labels do not share the recommender's blind spots. Two
protocols, scored with precision@k capped by the number of relevant POIs
left:
- `direct_p@k`: 3 random relevant likes and 3 random dislikes, then recommend.
- `onboarding_p@k`: the persona swipes through the active-learning rounds
  (`evaluation.onboarding_rounds`) with 10% noise. It also reports `found`
  (relevant POIs the persona ends up with: onboarding likes plus top-k hits)
  and `no_likes` (share of personas that finish without any like).

Baselines: `popularity_in_categories` (the most popular POIs of the chosen
categories, no personalisation) and `random`; the `:no_interests` row scores
the default style without the free text. Labels come from a single annotator
and each persona's categories were written together with its relevant POIs,
which favours the category signal: treat the numbers as an approximation, not
as a substitute for real users.

## Configuration
The main configuration is [configs/base.yaml](configs/base.yaml). It defines:
- Area (OSM relation or bbox)
- Tourism tag filters
- Overpass endpoints
- Enabled modules
- Recommendation weights, styles (`recommendation.styles`) and onboarding
  parameters (`recommendation.active_learning_*`)
- Evaluation parameters (`evaluation`)

Environment variables (or `.env`): `ORS_API_KEY`, `GOOGLE_PLACES_API_KEY`,
`PMV_CONFIG` (config path for the API), `PMV_CORS_ORIGINS` (comma-separated
allowed origins; `*` by default) and `PMV_HTTP_CONTACT` (see ingestion).

### Ingestion sources
Sources can be toggled in [configs/base.yaml](configs/base.yaml):
- `ingestion.osm.enabled`: OpenStreetMap (Overpass), the list of POIs.
- Wikidata, Wikipedia and Wikimedia Commons: descriptions, photos and
  popularity (always on).
- `ingestion.heritage.enabled`: the immovable property of the *Inventario
  General del Patrimonio Cultural Valenciano* (BIC and BRL), published by the
  Generalitat Valenciana on dadesobertes.gva.es under CC BY 4.0. It marks
  which POIs are protected heritage.

Licences: OpenStreetMap data is ODbL (© OpenStreetMap contributors),
Wikipedia text CC BY-SA, the heritage inventory CC BY 4.0 (© Generalitat
Valenciana). Every entity names its sources in `dataProvider`.

### Catalog format (Smart Data Models)
`data/processed/poi.jsonld` is a list of NGSI-LD entities in *keyValues* form
that validate against the official
[`dataModel.PointOfInterest`](https://github.com/smart-data-models/dataModel.PointOfInterest)
schema (v0.3.1); `tests/test_smart_data_models.py` checks it with a local copy
of the schemas. Details:

- `id`: `urn:ngsi-ld:PointOfInterest:<OSM type>:<OSM id>`.
- `source`: URL of the OpenStreetMap object; `refSeeAlso`: official website,
  Wikidata and Wikipedia URLs.
- `address`: `streetAddress`, `postalCode`, `addressLocality`, `addressCountry`.
- `openingHours`: OSM/schema.org text (`"Tu-Su 10:00-19:00; Mo off"`).
- `relevance`: 0 = most popular in its category, high values = lesser known
  (standard semantics: 0 is the most relevant).

Project extensions (not part of the standard): `subcategory` (OSM value),
`duration` (visit minutes), `durationEstimated` and `openingHoursEstimated`
(the value is an estimate, not from the source), `descriptionGenerated` (the
description was built from structured data, see below), and
`heritageDesignation` (`BIC` or `BRL`) with `heritageInventoryId` (the
inventory code) for protected heritage.

### Ingestion pipeline and data quality
1. **OSM (Overpass):** named elements inside the municipality. The
   description comes from `description:es` > `description` > `description:ca`
   > `description:en`; the address includes house number and postcode; OSM
   images are only accepted from Wikimedia Commons.
2. **Deduplication** by name and distance. The best record of each group
   becomes the primary one (with Wikidata, way/relation before node,
   information boards last); the others only fill gaps.
3. `tourism=information` elements that are not tourist offices (boards, maps,
   guideposts) are dropped: they are not destinations. So are falla
   monuments (`artwork_type=falla`): they stand only in March.
4. **Protected heritage:** each POI (except artworks) is matched to the
   closest inventory entry within `heritage.max_distance_m` whose official
   name contains every distinctive word of the OSM name, allowing
   Valencian/Spanish variants ("Torres dels Serrans" = "Torres de Serranos").
   Matches are written to `data/processed/heritage_matches.csv` for review.
5. **Wikidata links are checked** before use: the item must be located
   within 1.5 km of the POI (10 km for beaches, parks and natural areas). An
   item without coordinates is accepted only if its label names the POI, it is
   not a person, and the POI is not a monument to someone. OSM sometimes
   links a namesake (the artwork "Marqués del Túria" links the noble title,
   the bullring links Alicante's); those links are ignored.
6. **Enrichment** (parallel, `ingestion.workers`): Wikipedia article (Spanish
   preferred), description when OSM's is too short (<80 characters; the
   extract is cut at a full sentence), photo from Wikidata (P18) or from
   Commons only when the file title names the place, and popularity = page
   views over the last `pageview_days` days summed over the Wikipedia
   editions in `pageview_languages`. Requests are batched (50 Wikidata
   entities, 20 extracts or 50 articles per call), rate-limited per host and
   retried on 429/5xx.
7. **Generated descriptions:** a POI that no source describes gets one built
   from fixed Spanish templates over its OSM tags and heritage status: kind,
   material, artist, year, inscription, period, architecture, BIC/BRL
   protection ("Escultura de bronce, obra de Nassio Bayarri (1975).").
   Nothing is inferred and no language model is used; tag values without a
   known translation are left out. It is kept only if it states at least
   `generated_description_min_facts` facts beyond the kind of place (museums
   and other destinations in their own right are always kept), and is
   flagged `descriptionGenerated: true`. The rest are dropped.
8. **Categories:** `category` is one of the canonical categories used by the
   frontend (`museum`, `historic`, `beach`...) and `subcategory` keeps the OSM
   value (`bridge`, `castle`, `gallery`...).
9. **Relevance:** inverse popularity percentile within the category (against
   the whole catalog when the category has fewer than 5 POIs).
10. **Duration and opening hours:** the sources have no visit duration, so it
   is estimated deterministically per subcategory (`durationEstimated: true`).
   Without OSM opening hours a conservative schedule is generated
   (`synthetic: true`); outdoor places (bridges, artworks, parks...) are 24/7.

Each run writes `data/processed/ingestion_report.json` with drops per reason
and a quality summary (image, address and real-opening-hours coverage,
description length, generated descriptions, protected heritage). To comply with the Wikimedia policy, set
`PMV_HTTP_CONTACT` (contact URL or e-mail), which is added to the User-Agent.

If any request still fails after retries, ingestion stops with an error and
**does not overwrite** `poi.jsonld` (unless `ingestion.allow_partial: true`):
a Wikimedia rate limit cannot silently empty the catalog.

Regenerate the embeddings after re-ingesting (`--step embeddings`).

### Planning with routing
Planning uses OpenRouteService for real distances and times. Configure in
[configs/base.yaml](configs/base.yaml):
- `ORS_API_KEY` in `.env` (not in the YAML)
- `parameters.planning.routing.profile` (`foot-walking` by default)
- Key-less alternative: `provider: osrm` (public OSRM server)

Run:
- `python -m pmv_turismo.cli --config configs/base.yaml --step planning`

The planner:
- Uses POIs with valid coordinates.
- Takes `max_stops_per_day`, `max_total_minutes`, `visit_time_minutes`,
  `max_candidates` and `start_location` from the request or the config. In
  the app they come from the pace chosen in onboarding; the planner screen
  has no settings of its own.
- Visit duration per stop: the request's `visit_time_minutes` if the user set
  one; otherwise the POI's `duration`, falling back to the config's
  `visit_time_minutes`.
- Computes the routing matrix and runs a score-aware greedy: the next stop
  maximises `score / (1 + travel_minutes / travel_penalty_minutes)`. With
  equal scores it is nearest-neighbour.
- Skips stops that do not fit the time budget or are closed on arrival
  (Europe/Madrid time zone; `respect_opening_hours`).
- The API completes the lightweight candidates sent by the frontend (id +
  location) with duration, opening hours and score from the catalog and the
  user's recommendations.

## API
Start it with:
- `uvicorn pmv_turismo.api:app --reload`

On startup the API loads the embedding model in the background (about 7 s on
CPU and ~1.5 GB of RAM, torch included) so the first onboarding with free-text interests does
not wait for it; afterwards each text costs ~70 ms and is cached.

Endpoints:
- `/health`
- `/poi`
- `/poi/sample` (onboarding rounds; params `categories`, `limit`,
  `liked_pois`, `disliked_pois`, `seen_pois`, `interests`)
- `/embedding/{poi_id}`
- `/user/onboarding` (POST)
- `/user/preferences` (GET/POST)
- `/recommendations` (GET/POST)
- `/itinerary` (GET/POST)
- `/run/ingestion` (POST)
- `/run/recommendations` (POST)

Documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Profile fields: `name`, `preferred_categories`, `recommendation_style`
(`explorer`, `conservative` or `hidden_gems`) and `interests` (free text, up
to 300 characters).

Example (POST /recommendations):
{
	"user_id": "u123"
}

Inline example (POST /recommendations):
{
	"profile": {
		"name": "tourist",
		"preferred_categories": ["museum", "park", "historic"],
		"interests": "Roman ruins and prehistory"
	},
	"liked_pois": ["urn:ngsi-ld:PointOfInterest:way:4790961"],
	"top_n": 30,
	"category": "museum,park",
	"limit": 10
}

Example (POST /itinerary):
{
	"user_id": "u123",
	"max_stops_per_day": 6,
	"max_total_minutes": 420,
	"visit_time_minutes": 60,
	"max_candidates": 20,
	"start_location": {"lat": 39.47, "lon": -0.38}
}

Inline example (POST /itinerary) with explicit candidates:
{
	"candidates": [
		{"id": "poi1", "location": {"coordinates": [-0.38, 39.47]}, "category": ["museum"]},
		{"id": "poi2", "location": {"coordinates": [-0.39, 39.48]}, "category": ["park"]}
	],
	"start_location": {"lat": 39.47, "lon": -0.38}
}

## Frontend (Next.js)
- Requirements: Node 18+ and pnpm (`corepack enable` or install pnpm).
- Variables (`frontend/.env`, see `.env.example`):
  - `NEXT_PUBLIC_API_URL` (`http://127.0.0.1:8000` by default)
  - `NEXT_PUBLIC_ONBOARDING_ROUND_SIZES` (cards per onboarding round, `6,4,4`)
  - `NEXT_PUBLIC_CARTO_API_KEY` (optional; CARTO Voyager map tiles, plain OSM
    tiles without it)
- Install and run:
  1) `cd frontend`
  2) `pnpm install`
  3) `pnpm dev`
  4) Open `http://localhost:3000`
- Build/production: `pnpm build` and `pnpm start` in `frontend`.
- Remote images must come from a host listed in `images.remotePatterns` in
  `frontend/next.config.mjs` (Wikimedia hosts are already there).

### Multi-user flow
`user_id` may only contain letters, digits, `_` and `-` (max. 64). User data
is stored in `data/processed/users/` and is not versioned.

1) Onboarding: POST `/user/onboarding` stores the profile, likes, dislikes,
   start location and planner preferences.
2) Personalised recommendations:
	- POST `/recommendations` with `user_id` (reads the stored profile) or an
	  inline `profile`.
	- Optional filters: `limit` and `category` (comma-separated).
	- With `user_id` the result is saved to
	  `data/processed/users/{user_id}/recommendations.jsonld`.
3) Itinerary:
	- POST `/itinerary` with `user_id` (uses the stored recommendations) or
	  inline `candidates`.
	- Optional: `max_stops_per_day`, `max_total_minutes`, `visit_time_minutes`,
	  `max_candidates`, `start_location`, `start_at` (ISO 8601).
	- With `user_id` the result is saved to
	  `data/processed/users/{user_id}/itinerary.jsonld`.
4) Read results back:
	- GET `/recommendations?user_id=...` and GET `/itinerary?user_id=...`.

Example of saving preferences (POST /user/preferences):
{
	"user_id": "u123",
	"profile": {
		"name": "tourist",
		"preferred_categories": ["museum", "park", "historic"],
		"interests": "sacred art and baroque churches"
	}
}
