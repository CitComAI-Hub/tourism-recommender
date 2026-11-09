"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { Loader2 } from "lucide-react";
import { useUser } from "@/lib/user-context";
import { getPOIs, getRecommendations, searchPOIs, type Location, type POI } from "@/lib/api";
import { AppHeader } from "@/components/explore/app-header";
import { MapView } from "@/components/explore/map-view";
import { RecommendationsList } from "@/components/explore/recommendations-list";
import { POIDetailSheet } from "@/components/explore/poi-detail-sheet";
import { normalizeImageUrl } from "@/components/ui/optimized-image";
import { matchesPoiCategory } from "@/lib/poi-categories";
import { useI18n } from "@/lib/i18n";

const MAX_RECOMMENDATIONS = 30;
const SEARCH_DEBOUNCE_MS = 350;

export default function ExplorePage() {
  const router = useRouter();
  const { t } = useI18n();
  const {
    profile,
    onboardingComplete,
    isHydrated,
    likedPOIs,
    dislikedPOIs,
    startLocation,
    useCurrentLocation,
    like,
    dislike,
    clearFeedback,
  } = useUser();

  const [recommendations, setRecommendations] = useState<POI[]>([]);
  const [catalog, setCatalog] = useState<POI[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedPOI, setSelectedPOI] = useState<POI | null>(null);
  const [hoveredPOI, setHoveredPOI] = useState<POI | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [showMap, setShowMap] = useState(true);
  const [userLocation, setUserLocation] = useState<Location | null>(startLocation);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<POI[] | null>(null);

  useEffect(() => {
    if (!isHydrated) return;
    if (!onboardingComplete) router.replace("/onboarding");
  }, [isHydrated, onboardingComplete, router]);

  // Position: the browser's, if allowed; otherwise the one chosen in onboarding.
  useEffect(() => {
    setUserLocation(startLocation);
    if (!useCurrentLocation || !navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => setUserLocation({ lat: coords.latitude, lon: coords.longitude }),
      () => undefined,
      { enableHighAccuracy: true, maximumAge: 300000, timeout: 10000 }
    );
  }, [startLocation, useCurrentLocation]);

  // The catalog is small (hundreds of POIs): load it once to show the liked places.
  useEffect(() => {
    getPOIs().then(setCatalog).catch(() => setCatalog([]));
  }, []);

  // Every like, dislike or move re-ranks the list.
  useEffect(() => {
    if (!isHydrated || !onboardingComplete) return;
    let cancelled = false;
    setIsLoading(true);
    setError(false);
    getRecommendations({
      liked: likedPOIs,
      disliked: dislikedPOIs,
      categories: profile?.preferred_categories ?? [],
      location: userLocation,
      top_n: MAX_RECOMMENDATIONS,
      at: new Date().toISOString(),
    })
      .then((recs) => !cancelled && setRecommendations(recs))
      .catch(() => {
        if (!cancelled) {
          setRecommendations([]);
          setError(true);
        }
      })
      .finally(() => !cancelled && setIsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [isHydrated, onboardingComplete, likedPOIs, dislikedPOIs, profile, userLocation]);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      setSearchResults(null);
      return;
    }
    const timer = setTimeout(() => {
      searchPOIs(q, MAX_RECOMMENDATIONS).then(setSearchResults).catch(() => setSearchResults([]));
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query]);

  const likedItems = useMemo(() => {
    const byId = new Map([...catalog, ...recommendations].map((p) => [p.id, p]));
    return likedPOIs.map((id) => byId.get(id)).filter((p): p is POI => Boolean(p));
  }, [catalog, recommendations, likedPOIs]);

  const isSearching = searchResults !== null;
  const shown = (isSearching ? searchResults : recommendations).filter(
    (poi) => !activeFilter || matchesPoiCategory(poi.category, activeFilter)
  );

  if (!isHydrated || !onboardingComplete) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  const toggleLike = (poi: POI) => (likedPOIs.includes(poi.id) ? clearFeedback(poi.id) : like(poi.id));

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <AppHeader />

      <main className="flex-1 flex flex-col lg:flex-row min-h-0">
        <div className={`${showMap ? "h-[40vh] lg:h-[calc(100vh-64px)]" : "h-0"} lg:flex-1 lg:min-h-0 transition-all`}>
          <MapView
            pois={[...shown, ...likedItems.filter((p) => !shown.some((s) => s.id === p.id))]}
            likedPOIs={likedItems}
            selectedPOI={selectedPOI}
            hoveredPOI={hoveredPOI}
            onSelectPOI={setSelectedPOI}
            isVisible={showMap}
            userLocation={userLocation}
          />
        </div>

        <div className="flex-1 lg:w-[420px] lg:flex-none overflow-hidden min-h-0 lg:h-[calc(100vh-64px)] lg:max-h-[calc(100vh-64px)] flex flex-col bg-background border-l border-border">
          <RecommendationsList
            recommendations={shown}
            likedItems={likedItems}
            isLoading={isLoading && !isSearching}
            activeFilter={activeFilter}
            onFilterChange={setActiveFilter}
            query={query}
            onQueryChange={setQuery}
            isSearching={isSearching}
            onSelectPOI={setSelectedPOI}
            onHoverPOI={setHoveredPOI}
            onLike={toggleLike}
            onDislike={(poi) => dislike(poi.id)}
            onUnlike={(poi) => clearFeedback(poi.id)}
            showMap={showMap}
            onToggleMap={() => setShowMap(!showMap)}
          />
          {error && (
            <p role="alert" className="px-4 pb-4 text-sm text-destructive">
              {t("recommendations.service_error")}
            </p>
          )}
        </div>
      </main>

      <POIDetailSheet
        poi={selectedPOI}
        isLiked={selectedPOI ? likedPOIs.includes(selectedPOI.id) : false}
        onClose={() => setSelectedPOI(null)}
        onLike={toggleLike}
        onDislike={(poi) => {
          dislike(poi.id);
          setSelectedPOI(null);
        }}
      />

      {/* Preload the hovered POI's hero image so the detail sheet opens without a flash. */}
      {hoveredPOI?.image && (
        <div aria-hidden className="absolute w-0 h-0 overflow-hidden">
          <Image
            key={`preload-${hoveredPOI.id}`}
            src={normalizeImageUrl(hoveredPOI.image)}
            alt=""
            width={800}
            height={400}
            sizes="(max-width: 768px) 100vw, 50vw"
          />
        </div>
      )}
    </div>
  );
}
