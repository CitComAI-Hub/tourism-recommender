# Flujo del Algoritmo (Paso a Paso)
El sistema opera en dos grandes fases: la construcción del perfil (Onboarding Activo) y la generación de la ruta (Ranking y Planificación).

## Fase A: Onboarding Activo (Conociendo al usuario)
Filtro Inicial (Opcional): El usuario abre la app y selecciona categorías de alto nivel (ej. "Arte", "Naturaleza", "Gastronomía").

### Ronda 1 (Exploración Global - 6 tarjetas):
El sistema toma el catálogo filtrado y agrupa los POIs en clusters usando K-Means sobre sus embeddings (representaciones semánticas).Selecciona el "medoide" (el POI más representativo) de cada cluster.Muestra al usuario 6 tarjetas muy diversas entre sí. El usuario hace swipe (Me gusta / No me gusta).

### Ronda 2 y 3 (Refinamiento - 4+4 tarjetas):
Con los primeros likes, el sistema no hace un promedio simple. Agrupa los likes en sub-intereses ($K_u$). Calcula una puntuación para las tarjetas no vistas combinando Afinidad (lo que parece gustarle) e Incertidumbre (zonas del mapa de gustos que el sistema aún desconoce). Muestra nuevas tarjetas enfocadas en afinar esas regiones prometedoras o dudosas.

Fase B: Ranking y Planificación (Generando la recomendación)

Scoring Híbrido (Puntuación Global): Para los POIs elegibles, el sistema calcula una nota final sumando tres factores: Semántica (50%): ¿Se parece a alguno de los sub-intereses del usuario? (Busca la máxima similitud, no el promedio). Calidad (30%): Basado en el percentil de relevancia del POI dentro de su categoría. Proximidad Geoespacial (20%): Penaliza los lugares que están demasiado lejos (decaimiento espacial).

Re-ranking con MMR (Diversificación): El sistema revisa el "Top-K" generado. Si los 3 primeros lugares son iglesias góticas, el MMR (Maximal Marginal Relevance) penaliza las siguientes iglesias y "empuja" hacia arriba otras categorías (ej. un parque o un museo) para garantizar variedad.

Planificación de Itinerario: El Top-K diversificado se envía a un motor de rutas para ordenarlo lógicamente en el mapa, calculando tiempos de desplazamiento y duración de visitas (usando el campo duration del JSON).