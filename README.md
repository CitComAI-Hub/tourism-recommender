# PMV IA Turismo (Valencia)

## Objetivo
Construir un PMV modular y escalable que combine **recomendación** y **planificación** de itinerarios turísticos para Valencia, con componentes intercambiables.

## Alcance del PMV (MVP)
- Ingesta básica de POI (OSM u otra fuente).
- Recomendación content-based con perfiles sintéticos.
- Planificación de itinerarios con restricciones de tiempo/distancia/horarios.
- API mínima para consultar POI, recomendaciones y planificación.
- Evaluación offline reproducible.

## Módulos (capas desacopladas)
1. **Ingesta/ETL**
2. **Catálogo de POI**
3. **Recomendación**
4. **Planificación**
5. **API**
6. **Evaluación**

## Principios
- Configuración por archivos (YAML/JSON) para intercambiar componentes.
- Interfaces claras entre módulos.
- Evolución incremental (paso a paso).

## Setup (Linux)
1. Crear entorno virtual:
	python3 -m venv .venv
2. Activar entorno:
	source .venv/bin/activate
3. Instalar dependencias:
	pip install -r requirements.txt
4. Instalar el paquete en modo editable:
	pip install -e .
5. Configurar secretos: copiar `.env.example` a `.env` y rellenar `ORS_API_KEY`.
   El `.env` está ignorado por git; las claves nunca van en `configs/base.yaml`.

Las dependencias de embeddings (`sentence-transformers`, `torch`) solo hacen falta
para regenerarlos: `pip install -e ".[embeddings]"`. El repositorio incluye el
catálogo (`data/processed/poi.jsonld`) y sus embeddings para poder arrancar sin
reingestar.

## Tests
	pip install -e ".[dev]"
	pytest

## CLI
El CLI permite ejecutar pasos concretos del pipeline.

Ejemplos:
- Ingesta: python -m pmv_turismo.cli --config configs/base.yaml --step ingestion
- Catálogo: python -m pmv_turismo.cli --config configs/base.yaml --step catalog
- Embeddings: python -m pmv_turismo.cli --config configs/base.yaml --step embeddings
- Recomendación: python -m pmv_turismo.cli --config configs/base.yaml --step recommendation
  (requiere `recommendation.liked_pois` en la config; si está vacío se omite)
- Planificación: python -m pmv_turismo.cli --config configs/base.yaml --step planning
- Evaluación: python -m pmv_turismo.cli --config configs/base.yaml --step evaluation

- Todo: python -m pmv_turismo.cli --config configs/base.yaml --step all

## Evaluación offline
`--step evaluation` hace leave-one-out sobre los likes: para cada perfil con al
menos dos likes, oculta uno, recomienda con el resto y mira en qué posición
aparece el oculto. Compara los estilos configurados con varios baselines
(`semantic_mean`, `semantic_multi`, `popularity` y `random`) y reporta hit@k,
nDCG@k, diversidad intra-lista (ILD), novedad, cobertura, fuga de dislikes y
distancia media.

Los perfiles son los guardados en `user_profiles.json` (si hay) más perfiles
sintéticos deterministas generados por categoría (`evaluation.synthetic_users`,
`evaluation.seed`). Para que el resultado no dependa de la hora, la evaluación
desactiva el filtro de horarios. Salida: `data/processed/evaluation.json` y
`evaluation_report.txt`.

Ojo: los likes sintéticos se basan en categorías y el texto de los embeddings
incluye la categoría, así que los métodos semánticos salen favorecidos. Sirve
para comparar variantes entre sí, no como métrica de producto.

## Configuración
La configuración principal está en [configs/base.yaml](configs/base.yaml). Ahí se define:
- Área (por relación OSM o bbox)
- Filtros de tags turísticos
- Endpoints Overpass
- Activación de módulos
- Pesos de la recomendación y de los estilos (`recommendation.styles`)
- Parámetros de la evaluación (`evaluation`)

Variables de entorno (o `.env`): `ORS_API_KEY`, `GOOGLE_PLACES_API_KEY`,
`PMV_CONFIG` (ruta de la config para la API) y `PMV_CORS_ORIGINS` (orígenes
permitidos separados por comas; por defecto `*`).

### Fuentes de ingesta
Puedes activar o desactivar fuentes fácilmente en [configs/base.yaml](configs/base.yaml):
- `ingestion.osm.enabled`: OSM (Overpass)
- `ingestion.opendatasoft.enabled`: OpenDataSoft (monumentos Valencia)

### Pipeline de ingesta y calidad de datos
1. **OSM (Overpass):** elementos con nombre dentro del municipio. La descripción
   se toma de `description:es` > `description` > `description:ca` > `description:en`;
   la dirección incluye número y código postal; las imágenes de OSM solo se
   aceptan si vienen de Wikimedia Commons.
2. **Deduplicación** por nombre y distancia. De cada grupo se queda como
   principal el mejor registro (con Wikidata, way/relation antes que node,
   paneles informativos al final); los demás solo rellenan huecos.
3. Se descartan los `tourism=information` que no son oficinas de turismo
   (paneles, mapas, postes): no son destinos.
4. **Enriquecimiento** (en paralelo, `ingestion.workers`): artículo de
   Wikipedia (preferencia por español), descripción si la de OSM es demasiado
   corta (<80 caracteres; el extracto se corta en una frase completa), foto de
   Wikidata (P18) o de Commons solo si el título del fichero nombra el lugar,
   y popularidad = visitas de los últimos `pageview_days` días sumadas en las
   ediciones de Wikipedia de `pageview_languages`. Las peticiones van en lotes
   (50 entidades de Wikidata, 20 extractos o 50 artículos por llamada), con
   límite de ritmo por servidor y reintentos ante 429/5xx.
5. **Categorías:** `category` es una de las categorías canónicas que usa el
   frontend (`museum`, `historic`, `beach`...) y `subcategory` conserva el
   valor de OSM (`bridge`, `castle`, `gallery`...).
6. **Relevancia:** percentil inverso de popularidad dentro de la categoría
   (contra todo el catálogo si la categoría tiene menos de 5 POIs).
7. **Duración y horarios:** no hay dato de origen sobre la duración, así que
   se estima de forma determinista por subcategoría (`durationEstimated: true`).
   Si no hay horario en OSM se genera uno conservador (`synthetic: true`);
   los lugares al aire libre (puentes, obras de arte, parques...) son 24/7.

Cada ejecución escribe `data/processed/ingestion_report.json` con los
descartes por motivo y un resumen de calidad (cobertura de imágenes,
direcciones, horarios reales, longitud de las descripciones). Para cumplir la
política de Wikimedia, define `PMV_HTTP_CONTACT` (URL o e-mail de contacto),
que se añade al User-Agent.

Si alguna petición falla tras los reintentos, la ingesta termina con error y
**no sobrescribe** `poi.jsonld` (salvo `ingestion.allow_partial: true`): un
límite de peticiones de Wikimedia no puede vaciar el catálogo en silencio.

Tras reingestar hay que regenerar los embeddings (`--step embeddings`).

### Planificación con ruteo
La planificación usa OpenRouteService para distancias/tiempos reales. Configura en [configs/base.yaml](configs/base.yaml):
- `ORS_API_KEY` en `.env` (no en el YAML)
- `parameters.planning.routing.profile` (por defecto `foot-walking`)
- Alternativa sin clave: `provider: osrm` (servidor público de OSRM)

Ejecuta:
- `python -m pmv_turismo.cli --config configs/base.yaml --step planning`

El planner actual:
- Usa POI con coordenadas válidas.
- Toma los parámetros configurados: `max_stops_per_day`, `max_total_minutes`, `visit_time_minutes`, `max_candidates` y `start_location`.
- Duración de cada parada: el `visit_time_minutes` de la petición si el usuario lo fija; si no, la `duration` del POI y, en su defecto, el `visit_time_minutes` de la config.
- Calcula la matriz de rutas y aplica un greedy que tiene en cuenta la puntuación: la siguiente parada maximiza `score / (1 + minutos_trayecto / travel_penalty_minutes)`. Con puntuaciones iguales equivale a ir al más cercano.
- Descarta paradas que no caben en el presupuesto de tiempo o que están cerradas a la hora de llegada (zona horaria Europe/Madrid; `respect_opening_hours`).
- La API completa los candidatos ligeros que envía el frontend (id + ubicación) con duración, horarios y puntuación del catálogo y de las recomendaciones del usuario.

## API
Para levantar la API:
- `uvicorn pmv_turismo.api:app --reload`

Endpoints:
- `/health`
- `/poi`
- `/poi/sample` (rondas del onboarding con active learning)
- `/user/onboarding` (POST)
- `/recommendations` (GET/POST)
- `/itinerary` (GET/POST)
- `/run/ingestion` (POST)
- `/run/recommendations` (POST)
- `/run/planning` (POST)
- `/user/preferences` (GET/POST)

Documentación:
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Ejemplo de perfil (POST /run/recommendations):
{
	"profile": {
		"name": "turista",
		"preferred_categories": ["museum", "park", "historic"]
	}
}

Ejemplo (POST /recommendations):
{
	"user_id": "u123"
}

Ejemplo inline (POST /recommendations):
{
	"profile": {
		"name": "turista",
		"preferred_categories": ["museum", "park", "historic"]
	},
	"top_n": 30,
	"category": "museum,park",
	"limit": 10
}

Ejemplo (POST /itinerary):
{
	"user_id": "u123",
	"max_stops_per_day": 6,
	"max_total_minutes": 420,
	"visit_time_minutes": 60,
	"max_candidates": 20,
	"start_location": {"lat": 39.47, "lon": -0.38}
}

Ejemplo inline (POST /itinerary) con candidatos explícitos:
{
	"candidates": [
		{"id": "poi1", "location": {"coordinates": [-0.38, 39.47]}, "category": ["museum"]},
		{"id": "poi2", "location": {"coordinates": [-0.39, 39.48]}, "category": ["park"]}
	],
	"start_location": {"lat": 39.47, "lon": -0.38}
}

## Frontend (Next.js)
- Requisitos: Node 18+ y pnpm (usa `corepack enable` o instala pnpm).
- Variables: `NEXT_PUBLIC_API_URL` (por defecto `http://127.0.0.1:8000`).
- Instalación y desarrollo:
  1) `cd frontend`
  2) `pnpm install`
  3) `pnpm dev`
  4) Abrir `http://localhost:3000`
- Build/producción: `pnpm build` y `pnpm start` en `frontend`.

### Flujo multiusuario (frontend)
Los `user_id` solo pueden contener letras, dígitos, `_` y `-` (máx. 64). Los datos
de usuario se guardan en `data/processed/users/` y no se versionan.

1) Guardar perfil del usuario (opcional si se envía inline en cada llamada):
	- POST `/user/preferences` con `user_id` y `profile`.
2) Obtener recomendaciones personalizadas:
	- POST `/recommendations` con `user_id` (lee su perfil guardado) o con `profile` inline.
	- Filtros opcionales: `limit` y `category` (coma) para acotar la respuesta.
	- El resultado se guarda en `data/processed/users/{user_id}/recommendations.jsonld` si se envía `user_id`.
3) Generar itinerario:
	- POST `/itinerary` con `user_id` (usa sus recomendaciones guardadas) o con `candidates` inline.
	- Parámetros opcionales: `max_stops_per_day`, `max_total_minutes`, `visit_time_minutes`, `max_candidates`, `start_location`, `start_at` (ISO 8601).
	- El resultado se guarda en `data/processed/users/{user_id}/itinerary.jsonld` si se envía `user_id`.
4) Consultar resultados:
	- GET `/recommendations?user_id=...` y GET `/itinerary?user_id=...`.

Ejemplo de guardar preferencias (POST /user/preferences):
{
	"user_id": "u123",
	"profile": {
		"name": "turista",
		"preferred_categories": ["museum", "park", "historic"]
	}
}
