# Algorithm flow (step by step)
The system works in two phases: building the profile (active onboarding) and
producing the recommendations and the route (ranking and planning).

## Phase A: Active onboarding (getting to know the user)
**Categories and interests.** The user picks at least 2 high-level categories
(e.g. museums, parks, places of worship) and can optionally type what they
are into ("Roman ruins and prehistory"). The text is embedded with the same
model as the POIs, so it can be compared with them directly.

### Round 1 (global exploration, 6 cards)
The catalog is filtered by the chosen categories. Each POI gets a quality
that mixes how representative it is (mean similarity to the rest of the pool)
with its popularity, plus its similarity to the typed interests when there
are any. A greedy MMR selection picks 6 cards that score well but differ from
each other, so every swipe (like / dislike) tells the system about a
different region of the catalog.

### Rounds 2 and 3 (refinement, 4 + 4 cards)
For every unseen POI the system estimates the probability that the user
likes it: each answer spreads to similar POIs (a Beta posterior smoothed over
embedding similarity). Cards are ranked optimistically, by that probability
plus its uncertainty (UCB), with a small popularity term and the similarity
to the typed interests:
- regions with likes have a high probability and get refined;
- regions not explored yet stay uncertain and get probed;
- regions with dislikes end up low and certain, and are dropped.

MMR keeps each round from repeating near-identical POIs. If a round comes
back short, the selected categories have run out of POIs and onboarding ends
there.

## Phase B: Ranking and planning (producing the recommendation)
**Candidate filter.** A POI is a candidate if it has an embedding, is not
disliked, and stays open for the walk there plus the whole visit.

**Hybrid score.** Each candidate gets a weighted sum (weights normalised to 1):
- *Semantics:* similarity to the closest of the user's interests. Likes are
  grouped into up to 3 clusters so a user who likes both art and nature is
  not averaged into neither; similarity to dislikes is subtracted. When the
  user typed interests, their similarity makes up half of this term.
- *Popularity:* from the POI's popularity percentile within its category;
  the style decides whether popular places or hidden gems are favoured.
- *Proximity:* distance decay from the user's start location.
- *Preferred categories:* a bonus for POIs in the categories picked in
  onboarding.

A user who rejected every card still gets recommendations: the score then
relies on the typed interests, the categories, popularity and distance.

**MMR re-ranking (diversification).** If the top places are three Gothic
churches, MMR penalises the next similar ones and pushes other kinds of POIs
(a park, a museum) up to keep the list varied. The style sets how strong this
is: `explorer` diversifies most, `conservative` least.

**Itinerary planning.** The places the user accepts go to a routing engine
(OpenRouteService or OSRM), which orders them on the map with a score-aware
greedy, accounting for travel times, visit durations (the POI's `duration`)
and opening hours. The pace (stops per day, total time, visit time) comes
from onboarding.
