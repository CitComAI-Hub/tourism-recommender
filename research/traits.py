"""Which onboarding questions carry information? Simulated with real tourists.

The onboarding happens at the start of the trip, so every later visit is a
"future" visit. For each tourist with >= 3 visited POIs:

- the last two visits are held out (scored; never shown, never used as input);
- the earlier visits ("known") stand for what the user tells the onboarding:
  their themes (category chips), whether they go for famous or lesser-known
  places, how far they roam. Known visits are excluded from every ranking, so
  all rows compete for the same slots;
- the user's position is their first photo in the city (where they start: the
  GPS or the address given in the onboarding).

Part 1, reliability: is a trait stable within a user? Visits split in two
halves (alternating in time); Spearman across users for popularity affinity and
radius; theme agreement against a random user and against the user whose
first half is spatially closest (controls for "same area, same kind of place").

Part 2, value: hit@10 of the held-out visits with each signal, weights chosen on
a validation fold (5 folds by user, like protocol.py), paired tests.

Part 3, cards: 14 onboarding cards, popular-first (production) or spread over
themes and areas, chosen without knowing the held-out visits; the oracle likes a
card if it is a known visit and gives no answer otherwise (a first version that
counted the rest as dislikes and excluded them leaked: it removed 14 popular
places the user did not visit). Likes used as distance anchors (production) or
as themes.

Usage: PYTHONPATH=research venv/bin/python research/traits.py <config> <lat0> <lat1> <lon0> <lon1>
"""
from __future__ import annotations

import collections
import math
import sys

import numpy as np
from protocol import City, paired
from scipy.stats import spearmanr

from pmv_turismo.recommendation import LOG_DISTANCE_OFFSET_KM, Recommender

K = 10
CARDS = 14
THEME = {"historic": "heritage", "place_of_worship": "religious", "artwork": "art", "arts_centre": "art",
         "museum": "museum", "park": "nature", "garden": "nature", "nature_reserve": "nature", "beach": "nature",
         "zoo": "family", "aquarium": "family", "attraction": "attraction", "marketplace": "attraction"}
GRID_DIST = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
GRID_THEME = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
GRID_POP = (0.0, 0.15, 0.3, 0.5)
GRID_RAD = (0.0, 0.25, 0.5)
SIM_PRODUCTION = 0.15


def themes_of(poi: dict) -> set[str]:
    return {THEME.get(c, "other") for c in poi.get("category") or []}


class Data:
    def __init__(self, city: City):
        self.city = city
        self.n = len(city.pois)
        self.themes = [themes_of(p) for p in city.pois]
        self.theme_names = sorted({t for s in self.themes for t in s})
        self.T = np.array([[t in s for t in self.theme_names] for s in self.themes], dtype=float)
        self.D = np.array([[0.0]])
        self.card_cache: dict = {}
        self.first = {}
        for u, ps in city.by_user.items():
            p = min(ps, key=lambda p: p["ts"])
            self.first[u] = (p["lat"], p["lon"])

    def cases(self, users):
        """(user, known POI indices oldest first, held-out index)."""
        out = []
        for u in sorted(users):
            v = self.city.visits.get(u, {})
            if u not in self.city.tourists or len(v) < 3:
                continue
            seq = sorted(v, key=v.get)
            for held in seq[-2:]:
                out.append((u, seq[:-2], held))
        return out


def pop_affinity(rec: Recommender, idx) -> float:
    pct = np.argsort(np.argsort(rec.popularity, kind="stable"), kind="stable") / max(len(rec.popularity) - 1, 1)
    return float(np.mean(pct[list(idx)]))


def roam_km(rec: Recommender, start, idx) -> float:
    return float(np.median(rec._distances_km(*start)[list(idx)]))


def level(x: float, cuts) -> int:
    return -1 if x < cuts[0] else (1 if x > cuts[1] else 0)


def cards(d: Data, rec: Recommender, kind: str, blocked: set[int]) -> list[int]:
    order = [i for i in np.argsort(-rec.popularity, kind="stable") if i not in blocked]
    if kind == "popular":
        return order[:CARDS]
    # spread: round robin over themes (most popular theme first), at least 300 m
    # from the cards already chosen
    by_theme = collections.defaultdict(list)
    for i in order:
        for t in d.themes[i]:
            by_theme[t].append(i)
    theme_order = sorted(by_theme, key=lambda t: -rec.popularity[by_theme[t][0]])
    chosen: list[int] = []
    for _ in range(CARDS):
        for t in theme_order:
            if len(chosen) == CARDS:
                break
            for i in by_theme[t]:
                if i in chosen:
                    continue
                if chosen and d.D[i, chosen].min() < 0.3:
                    continue
                chosen.append(i)
                break
        if len(chosen) == CARDS:
            break
    return chosen[:CARDS] if len(chosen) >= CARDS else (chosen + [i for i in order if i not in chosen])[:CARDS]


def rank(d: Data, rec: Recommender, case, *, position=True, anchors=(), dist_w=1.0, themes=None,
         theme_w=0.0, pop_mult=1.0, sim_to=(), sim_w=0.0, exclude=(), pop_only=False) -> list[int]:
    u, known, held = case
    pop = rec.popularity * pop_mult
    if pop_only:
        score = pop.copy()
    else:
        dists = [rec._distances_from(i) for i in anchors]
        if position:
            dists.append(rec._distances_km(*d.first[u]))
        nearest = np.min(dists, axis=0) if dists else np.zeros(d.n)
        score = pop - dist_w * np.log(nearest + LOG_DISTANCE_OFFSET_KM)
    score = (score - score.min()) / max(float(np.ptp(score)), 1e-12)
    if themes is not None and theme_w:
        mask = np.array([bool(themes & d.themes[i]) for i in range(d.n)], dtype=float)
        score = score + theme_w * mask
    if sim_to and sim_w:
        s = (rec.adapted @ rec.adapted[list(sim_to)].T).mean(axis=1)
        score = score + sim_w * (s - s.min()) / max(float(np.ptp(s)), 1e-12)
    ex = set(known) | set(exclude)
    return [i for i in np.argsort(-score, kind="stable") if i not in ex][:K]


def methods(d: Data, rec: Recommender, w: dict, cuts: dict):
    """name -> function(case) -> top 10."""
    def themes_known(c):
        return set().union(*(d.themes[i] for i in c[1]))

    def pmult(c):
        return 1.0 + w["pop"] * level(pop_affinity(rec, c[1]), cuts["pop"])

    def dw(c):
        return w["dist"] * (1.0 - w["rad"] * level(roam_km(rec, d.first[c[0]], c[1]), cuts["rad"]))

    def card_feedback(c, kind):
        """Cards are chosen without knowing the held-out visits. The oracle likes a
        card that is a known visit; any other card gets no answer (not visiting a
        place is not disliking it, and a held-out visit must not leak)."""
        key = (id(rec), kind)
        if key not in d.card_cache:
            d.card_cache[key] = cards(d, rec, kind, set())
        shown = d.card_cache[key]
        liked = [i for i in shown if i in c[1]]
        return shown, liked

    def card_rows(kind):
        def anchors_sim(c, position):
            shown, liked = card_feedback(c, kind)
            d.likes[kind].append(len(liked))
            return rank(d, rec, c, position=position, anchors=liked, dist_w=w["dist"], sim_to=liked,
                        sim_w=SIM_PRODUCTION)

        def themes_pos(c):
            shown, liked = card_feedback(c, kind)
            th = set().union(*(d.themes[i] for i in liked)) if liked else None
            return rank(d, rec, c, dist_w=w["dist"], themes=th, theme_w=w["theme_cards"])
        return anchors_sim, themes_pos

    pop_anchor_sim, pop_themes = card_rows("popular")
    spr_anchor_sim, spr_themes = card_rows("spread")
    return {
        "popularity": lambda c: rank(d, rec, c, pop_only=True),
        "position": lambda c: rank(d, rec, c, dist_w=w["dist"]),
        "known as anchors (no position)": lambda c: rank(d, rec, c, position=False, anchors=c[1], dist_w=w["dist"]),
        "position + known as anchors": lambda c: rank(d, rec, c, anchors=c[1], dist_w=w["dist"]),
        "position + known anchors + similarity": lambda c: rank(d, rec, c, anchors=c[1], dist_w=w["dist"],
                                                               sim_to=c[1], sim_w=SIM_PRODUCTION),
        "position + themes": lambda c: rank(d, rec, c, dist_w=w["dist"], themes=themes_known(c), theme_w=w["theme"]),
        "position + popularity affinity": lambda c: rank(d, rec, c, dist_w=w["dist"], pop_mult=pmult(c)),
        "position + radius": lambda c: rank(d, rec, c, dist_w=dw(c)),
        "position + all three traits": lambda c: rank(d, rec, c, dist_w=dw(c), pop_mult=pmult(c),
                                                     themes=themes_known(c), theme_w=w["theme"]),
        "popular cards: no position, likes anchors+sim": lambda c: pop_anchor_sim(c, False),
        "popular cards: position + likes anchors+sim": lambda c: pop_anchor_sim(c, True),
        "popular cards: position + liked themes": pop_themes,
        "spread cards: position + likes anchors+sim": lambda c: spr_anchor_sim(c, True),
        "spread cards: position + liked themes": spr_themes,
    }


def hits(fn, cases) -> np.ndarray:
    return np.array([c[2] in fn(c) for c in cases], dtype=float)


def choose(d: Data, rec: Recommender, cases, cuts) -> dict:
    w = {"dist": 1.0, "theme": 0.0, "theme_cards": 0.0, "pop": 0.0, "rad": 0.0}
    w["dist"] = max(GRID_DIST, key=lambda x: hits(methods(d, rec, {**w, "dist": x}, cuts)["position"], cases).mean())
    w["theme"] = max(GRID_THEME, key=lambda x: hits(
        methods(d, rec, {**w, "theme": x}, cuts)["position + themes"], cases).mean())
    w["theme_cards"] = max(GRID_THEME, key=lambda x: hits(
        methods(d, rec, {**w, "theme_cards": x}, cuts)["spread cards: position + liked themes"], cases).mean())
    w["pop"] = max(GRID_POP, key=lambda x: hits(
        methods(d, rec, {**w, "pop": x}, cuts)["position + popularity affinity"], cases).mean())
    w["rad"] = max(GRID_RAD, key=lambda x: hits(
        methods(d, rec, {**w, "rad": x}, cuts)["position + radius"], cases).mean())
    return w


def tercile_cuts(d: Data, rec: Recommender, cases) -> dict:
    users = {c[0]: c[1] for c in cases}
    pa = [pop_affinity(rec, k) for k in users.values()]
    ro = [roam_km(rec, d.first[u], k) for u, k in users.items()]
    return {"pop": np.quantile(pa, [1 / 3, 2 / 3]), "rad": np.quantile(ro, [1 / 3, 2 / 3])}


def reliability(d: Data, rec: Recommender) -> None:
    city = d.city
    rows = []
    for u in sorted(city.tourists):
        v = city.visits.get(u, {})
        if len(v) < 4:
            continue
        seq = sorted(v, key=v.get)
        a, b = seq[0::2], seq[1::2]
        rows.append((u, a, b))
    print(f"\n== Part 1: reliability (tourists with >= 4 visits: {len(rows)})")
    if len(rows) < 10:
        return
    pa = np.array([[pop_affinity(rec, a), pop_affinity(rec, b)] for _, a, b in rows])
    ro = np.array([[roam_km(rec, d.first[u], a), roam_km(rec, d.first[u], b)] for u, a, b in rows])
    for name, x in (("popularity affinity", pa), ("radius from start", ro)):
        r = spearmanr(x[:, 0], x[:, 1])
        print(f"   {name:22s} split-half Spearman {r.statistic:+.3f} (p={r.pvalue:.2g})")
    TA = np.array([d.T[a].mean(axis=0) for _, a, _ in rows])
    TB = np.array([d.T[b].mean(axis=0) for _, _, b in rows])

    def cos(x, y):
        return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
    same = np.mean([cos(TA[i], TB[i]) for i in range(len(rows))])
    rng = np.random.default_rng(0)
    perm = [np.mean([cos(TA[i], TB[j]) for i, j in enumerate(rng.permutation(len(rows)))]) for _ in range(200)]
    cent = np.array([[np.mean([math.degrees(rec.lat[i]) for i in a]), np.mean([math.degrees(rec.lon[i]) for i in a])]
                     for _, a, _ in rows])
    near = []
    for i in range(len(rows)):
        dd = np.hypot(cent[:, 0] - cent[i, 0], (cent[:, 1] - cent[i, 1]) * math.cos(math.radians(cent[i, 0])))
        dd[i] = np.inf
        near.append(cos(TA[int(np.argmin(dd))], TB[i]))
    print(f"   themes: same user {same:.3f}; random user {np.mean(perm):.3f} "
          f"(p={np.mean(np.array(perm) >= same):.3g}); spatially nearest user {np.mean(near):.3f}")


def main():
    cfg_name, la0, la1, lo0, lo1 = sys.argv[1], *map(float, sys.argv[2:6])
    city = City(cfg_name, {"lat": (la0, la1), "lon": (lo0, lo1)})
    d = Data(city)
    users = sorted(city.visits)
    rng = np.random.default_rng(0)
    order = list(users)
    rng.shuffle(order)
    fold = {u: n % 5 for n, u in enumerate(order)}
    d.hidden_of = {}
    for u in users:
        v = city.visits[u]
        d.hidden_of[u] = sorted(v, key=v.get)[-2:] if len(v) >= 3 else []
    print(f"{cfg_name}: POIs {d.n}, themes {d.theme_names}, onboarding cases {len(d.cases(users))}")

    full = Recommender(city.pois, city.index, city.model(city.photos)[0], city.cfg)
    d.D = np.array([full._distances_from(i) for i in range(d.n)])
    reliability(d, full)

    results = collections.defaultdict(list)
    chosen = []
    d.likes = {"popular": [], "spread": []}
    for f in range(5):
        v = (f + 1) % 5
        test_users = [u for u in users if fold[u] == f]
        val_users = [u for u in users if fold[u] == v]
        val_train = [p for u in users if fold[u] not in (f, v) for p in city.by_user[u]]
        test_train = [p for u in users if fold[u] != f for p in city.by_user[u]]
        rec_val = Recommender(city.pois, city.index, city.model(val_train)[0], city.cfg)
        val_cases = d.cases(val_users)
        cuts = tercile_cuts(d, rec_val, val_cases)
        w = choose(d, rec_val, val_cases, cuts)
        chosen.append(w)
        rec = Recommender(city.pois, city.index, city.model(test_train)[0], city.cfg)
        d.likes = {"popular": [], "spread": []}
        for name, fn in methods(d, rec, w, cuts).items():
            results[name].append(hits(fn, d.cases(test_users)))
        print(f"   fold {f}: {w}", flush=True)
    rows = {k: np.concatenate(v) for k, v in results.items()}
    n = len(rows["position"])
    print(f"\n== Part 2+3: onboarding, cases {n} (95% CI ~ +-{1.96 * math.sqrt(0.25 / n):.3f})")
    print(f"   likes per user on the last fold: popular cards {np.mean(d.likes['popular']):.2f}, "
          f"spread cards {np.mean(d.likes['spread']):.2f} (of {CARDS})")
    for k, h in rows.items():
        print(f"   {k:48s} {h.mean():.3f}")
    print()
    for a, b in (("position", "popularity"),
                 ("position + known as anchors", "position"),
                 ("position + known anchors + similarity", "position + known as anchors"),
                 ("position + themes", "position"),
                 ("position + popularity affinity", "position"),
                 ("position + radius", "position"),
                 ("position + all three traits", "position"),
                 ("popular cards: position + likes anchors+sim", "position"),
                 ("popular cards: position + liked themes", "position"),
                 ("spread cards: position + likes anchors+sim", "position"),
                 ("spread cards: position + liked themes", "position"),
                 ("spread cards: position + liked themes", "popular cards: position + likes anchors+sim")):
        t = paired(rows[a], rows[b])
        print(f"   {a} vs {b}: {t['diff']:+.3f} CI95 [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] p={t['mcnemar_p']:.2g}")


if __name__ == "__main__":
    main()
