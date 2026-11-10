# Tourism recommender (Valencia)

Recommends Points of Interest to tourists: a semantic model of the POI
descriptions calibrated with what real visitors do, a short onboarding for new
users, explanations for every result and free-text search. Itinerary planning
is a separate service.

## How it recommends
- **Popularity** per POI from real visits (geotagged photos). POIs without
  visits get a popularity **predicted from their description** (embedding +
  category); without a behaviour model, Wikipedia page views are used.
- **Proximity** to the places the user liked or to their position
  (`log popularity - distance / 0.3 km`).
- **Similarity** on embeddings **adapted to co-visits**: a projection learnt
  so that "similar" means "visited by the same people", available for every
  POI, also those nobody photographed.
- **Onboarding:** the user picks categories and answers cards with the best
  known places first (rounds of 6, 4, 4).
- **Free text** is a search over the descriptions, not part of the ranking.
- Every result carries its `reasons` (popular, near a liked place, people who
  visit X also go here) and where its popularity comes from.

Measured on real tourists (Flickr YFCC100M, Valencia; `research/eval_service.py`):
a held-out visit lands in the top 10 for 68% of cases, against 55% for the
most visited places and 46% for the same ranking with Wikipedia popularity.

## Setup (Linux)
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -e ".[embeddings,dev]"
	cp .env.example .env    # optional settings

The embedding dependencies (`sentence-transformers`, `torch`) are in the
`embeddings` extra. They are needed to regenerate the POI embeddings and for
the free-text search; without them `/search` answers 503 and the rest works.
The repository ships the catalog (`data/processed/poi.jsonld`) and its
embeddings.

## Tests
	pytest

## Offline pipeline (CLI)
	python -m pmv_turismo.cli --step ingestion    # OSM + Wikidata/Wikipedia/Commons + GVA heritage
	python -m pmv_turismo.cli --step embeddings   # POI embeddings
	python -m pmv_turismo.cli --step geometry     # real OSM shapes of the POIs
	python -m pmv_turismo.cli --step behaviour    # behaviour model from photos
	python -m pmv_turismo.cli --step all

`--config` selects another configuration (default `configs/base.yaml`).

## Behaviour model
`--step behaviour` reads `behaviour.photos_path`, a JSON Lines file (gzip
allowed) of photos `{"user", "lat", "lon", "ts"}` from any source, counts a
photo as a visit to every POI whose real OSM shape it falls in
(`poi_geometry.json`; within 40 m for points and lines) and writes
`data/processed/behaviour_model.npz`: per-POI visitor counts, the popularity
regressor and the adapted projection. It holds aggregates only; user ids
should be pseudonymised before training. Without the file the service falls
back to Wikipedia popularity and plain embeddings.

The Valencia model is trained on Flickr YFCC100M photos (2004-2014):
`research/yfcc_scan.py` streams the YFCC100M metadata from AWS Open Data and
keeps the European rows; `research/eval_service.py` exports the Valencia
photos, pseudonymised. The data and the model are not versioned. The terms of
use of the YFCC100M metadata for a deployed service are still to be
confirmed.

## Configuration
The main configuration is [configs/base.yaml](configs/base.yaml). It defines:
- Area (OSM relation or bbox)
- Tourism tag filters
- Overpass endpoints
- Recommendation weights (`recommendation`)
- Behaviour model paths and source (`behaviour`)

Environment variables (or `.env`): `GOOGLE_PLACES_API_KEY`,
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
   description comes from the tags in the order of
   `ingestion.description_languages` (Valencia: `description:es` >
   `description:en` > `description` > `description:ca`), and its language is
   recorded in `descriptionLanguage`; the address includes house number and postcode; OSM
   images are only accepted from Wikimedia Commons.
2. **Deduplication** by name and distance. The best record of each group
   becomes the primary one (with Wikidata, way/relation before node,
   information boards last); the others only fill gaps.
3. **Artwork collections** (`ingestion.artwork_collections`): the artworks
   inside a given area become one POI for the whole collection, with the
   area's OSM relation as identity and a Wikidata item for enrichment. In
   Valencia, the 65 sculptures of the UPV Vera campus (median of one Flickr
   visitor each) become the "Campus Escultòric" (Q30133840).
   `tourism=information` elements that are not tourist offices (boards, maps,
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
6. **Enrichment** (parallel, `ingestion.workers`): Wikipedia article (in the order of
   `description_languages`), description when OSM's is too short (<80
   characters; the extract is cut at a full sentence), photo from OSM
   (Commons only) or Wikidata (P18), never from a search by name (in the
   manual audit that returned a photo of something else 4 times out of 5),
   and popularity = page
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

## API
Stateless: the client sends the user's feedback with each request and the
server stores nothing about users.

- `GET /health`: catalog size and behaviour model in use.
- `GET /pois`: catalog; filters `category` (comma-separated), `limit` and a
  bounding box (`min_lat`, `min_lon`, `max_lat`, `max_lon`).
- `GET /pois/{id}`
- `POST /onboarding/cards`: `{"limit", "seen", "liked", "disliked",
  "categories"}` -> next cards.
- `POST /recommendations`: `{"liked", "disliked", "categories", "location":
  {"lat", "lon"}, "top_n", "at"}` -> `{"results": [...]}`, each POI with
  `rank`, `score`, `popularity`, `popularitySource` (`visits`, `predicted` or
  `wikipedia`), `distanceKm` and `reasons`. `at` (ISO 8601 with timezone)
  sets the time for the opening-hours filter; now by default.
- `GET /search?q=...`: free-text search over the descriptions.

Run it with `uvicorn pmv_turismo.api:app --port 8000`. `PMV_CONFIG` selects
the configuration and `PMV_CORS_ORIGINS` the allowed origins (`*` by
default; the API never uses credentials).

## Frontend (Next.js)
- Requirements: Node 18+ and pnpm.
- Variables (`frontend/.env`):
  - `NEXT_PUBLIC_API_URL` (`http://127.0.0.1:8000` by default)
  - `NEXT_PUBLIC_ONBOARDING_ROUND_SIZES` (cards per onboarding round, `6,4,4`)
  - `NEXT_PUBLIC_CARTO_API_KEY` (optional; CARTO Voyager map tiles, plain OSM
    tiles without it)
- `cd frontend && pnpm install && pnpm dev`, then open
  `http://localhost:3000`. Production: `pnpm build && pnpm start`.
- The user's name, categories, likes, dislikes and position are kept in the
  browser (`localStorage`) and sent with each request.
