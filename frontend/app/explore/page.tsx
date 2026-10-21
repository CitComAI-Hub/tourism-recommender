"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/lib/user-context";
import { getRecommendations, getPOIs, type Location, type POI } from "@/lib/api";
import { AppHeader } from "@/components/explore/app-header";
import { MapView } from "@/components/explore/map-view";
import { RecommendationsList } from "@/components/explore/recommendations-list";
import { POIDetailSheet } from "@/components/explore/poi-detail-sheet";
import { normalizeImageUrl } from "@/components/ui/optimized-image";
import { matchesPoiCategory } from "@/lib/poi-categories";
import Image from "next/image";
import { Loader2 } from "lucide-react";

const MAX_RECOMMENDATIONS = 30;

function dedupeById(pois: POI[]): POI[] {
  const seen = new Set<string>();
  const result: POI[] = [];
  for (const poi of pois) {
    const key = poi.id || `poi-${result.length}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(poi);
  }
  return result.slice(0, MAX_RECOMMENDATIONS);
}

export default function ExplorePage() {
  const router = useRouter();
  const {
    userId,
    profile,
    onboardingComplete,
    isHydrated,
    likedPOIs,
    acceptedRecommendations,
    plannerPreferences,
    addAcceptedRecommendation,
    removeAcceptedRecommendation,
  } = useUser();

  const [recommendations, setRecommendations] = useState<POI[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPOI, setSelectedPOI] = useState<POI | null>(null);
  const [hoveredPOI, setHoveredPOI] = useState<POI | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [showMap, setShowMap] = useState(true);
  const [userLocation, setUserLocation] = useState<Location | null>(
    plannerPreferences.start_location
  );

  useEffect(() => {
    if (plannerPreferences.start_location) {
      setUserLocation(plannerPreferences.start_location);
    }
    if (!plannerPreferences.use_current_location || !navigator.geolocation) return;

    navigator.geolocation.getCurrentPosition(
      ({ coords }) => {
        setUserLocation({ lat: coords.latitude, lon: coords.longitude });
      },
      () => {
        // Keep the persisted planning location when browser permission is unavailable.
      },
      { enableHighAccuracy: true, maximumAge: 300000, timeout: 10000 }
    );
  }, [plannerPreferences.start_location, plannerPreferences.use_current_location]);

  useEffect(() => {
    if (!isHydrated) return;
    if (!onboardingComplete) {
      router.replace("/onboarding");
      return;
    }

    const fetchRecommendations = async () => {
      setIsLoading(true);
      try {
        const data = await getRecommendations({
          user_id: userId || undefined,
          profile: profile || undefined,
          limit: MAX_RECOMMENDATIONS,
          requested_at: new Date().toISOString(),
          user_mobility: plannerPreferences.mobility,
        });
        let recs = Array.isArray(data) ? data : [];

        // Fallback: if no recs, load generic POIs to show algo
        if (recs.length === 0) {
          try {
            const pois = await getPOIs({ limit: MAX_RECOMMENDATIONS });
            if (Array.isArray(pois)) {
              recs = pois;
            }
          } catch (e) {
            console.warn("Fallback POIs failed", e);
          }
        }

        setRecommendations(dedupeById(recs));
      } catch (error) {
        console.error("Error fetching recommendations:", error);
        // Use mock data for demo
        setRecommendations(dedupeById(getMockRecommendations()));
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecommendations();
  }, [userId, profile, onboardingComplete, isHydrated, router]);

  const handleToggleAccept = (poi: POI) => {
    const isAccepted = acceptedRecommendations.some(p => p.id === poi.id);
    if (isAccepted) {
      removeAcceptedRecommendation(poi.id);
    } else {
      addAcceptedRecommendation(poi);
    }
  };

  const safeRecommendations = Array.isArray(recommendations)
    ? recommendations
    : [];

  const likedSet = new Set(likedPOIs);
  const rankedRecommendations = [...safeRecommendations].sort((a, b) => {
    const aLiked = likedSet.has(a.id) ? 1 : 0;
    const bLiked = likedSet.has(b.id) ? 1 : 0;
    if (aLiked !== bLiked) return bLiked - aLiked;
    return (b.rating ?? 0) - (a.rating ?? 0);
  });

  const filteredRecommendations = activeFilter
    ? rankedRecommendations.filter((poi) => matchesPoiCategory(poi.category, activeFilter))
    : rankedRecommendations;

  if (!isHydrated || !onboardingComplete) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <AppHeader
        acceptedCount={acceptedRecommendations.length}
        onPlanClick={() => router.push("/planner")}
      />

      <main className="flex-1 flex flex-col lg:flex-row min-h-0">
        {/* Map Section */}
        <div
          className={`${showMap ? "h-[40vh] lg:h-[calc(100vh-64px)]" : "h-0"} lg:flex-1 lg:min-h-0 transition-all`}
        >
          <MapView
            pois={filteredRecommendations}
            acceptedPOIs={acceptedRecommendations}
            selectedPOI={selectedPOI}
            hoveredPOI={hoveredPOI}
            onSelectPOI={setSelectedPOI}
            isVisible={showMap}
            userLocation={userLocation}
          />
        </div>

        {/* Recommendations Panel */}
        <div className="flex-1 lg:w-[420px] lg:flex-none overflow-hidden min-h-0 lg:h-[calc(100vh-64px)] lg:max-h-[calc(100vh-64px)] flex flex-col bg-background border-l border-border">
          <RecommendationsList
            recommendations={filteredRecommendations}
            likedPOIs={likedPOIs}
            acceptedRecommendations={acceptedRecommendations}
            isLoading={isLoading}
            activeFilter={activeFilter}
            onFilterChange={setActiveFilter}
            onSelectPOI={setSelectedPOI}
            onHoverPOI={setHoveredPOI}
            onToggleAccept={handleToggleAccept}
            showMap={showMap}
            onToggleMap={() => setShowMap(!showMap)}
          />
        </div>
      </main>

      {/* POI Detail Sheet */}
      <POIDetailSheet
        poi={selectedPOI}
        isAccepted={selectedPOI ? acceptedRecommendations.some(p => p.id === selectedPOI.id) : false}
        onClose={() => setSelectedPOI(null)}
        onToggleAccept={handleToggleAccept}
      />

      {/* Hidden preloader: preload the hero image of the hovered POI so 
          it's already cached gracefully when the detail sheet opens.
          Sizes must match the OptimizedImage sizes in poi-detail-sheet.tsx */}
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

// Mock data for when API is unavailable
function getMockRecommendations(): POI[] {
  return [
    {
      id: "1",
      name: "Ciudad de las Artes y las Ciencias",
      description: "Complejo arquitectónico, cultural y de entretenimiento diseñado por Santiago Calatrava y Félix Candela.",
      category: ["attraction", "artwork", "museum"],
      location: { coordinates: [-0.3524, 39.4541] },
      rating: 4.8,
      duration: 180,
    },
    {
      id: "2",
      name: "La Lonja de la Seda",
      description: "Obra maestra del gótico civil valenciano, declarada Patrimonio de la Humanidad por la UNESCO.",
      category: ["attraction", "place_of_worship"],
      location: { coordinates: [-0.3780, 39.4744] },
      rating: 4.7,
      duration: 60,
    },
    {
      id: "3",
      name: "Jardines del Turia",
      description: "Uno de los parques urbanos más grandes de España, construido en el antiguo cauce del río Turia.",
      category: ["park", "nature_reserve"],
      location: { coordinates: [-0.3700, 39.4780] },
      rating: 4.6,
      duration: 120,
    },
    {
      id: "4",
      name: "Mercado Central",
      description: "Mercado de estilo modernista, uno de los más grandes y antiguos de Europa dedicado a la alimentación.",
      category: ["marketplace", "attraction"],
      location: { coordinates: [-0.3785, 39.4735] },
      rating: 4.5,
      duration: 90,
    },
    {
      id: "5",
      name: "Playa de la Malvarrosa",
      description: "Popular playa urbana con paseo marítimo, restaurantes y ambiente animado.",
      category: ["nature_reserve", "attraction"],
      location: { coordinates: [-0.3220, 39.4780] },
      rating: 4.4,
      duration: 180,
    },
    {
      id: "6",
      name: "Catedral de Valencia",
      description: "Catedral gótica que alberga el Santo Cáliz, considerado el Santo Grial.",
      category: ["place_of_worship", "attraction"],
      location: { coordinates: [-0.3755, 39.4755] },
      rating: 4.6,
      duration: 60,
    },
    {
      id: "7",
      name: "Bioparc Valencia",
      description: "Zoológico de inmersión que recrea ecosistemas africanos con más de 100 especies.",
      category: ["zoo", "park"],
      location: { coordinates: [-0.4065, 39.4785] },
      rating: 4.7,
      duration: 240,
    },
    {
      id: "8",
      name: "Barrio del Carmen",
      description: "Barrio histórico con calles medievales, arte urbano, tiendas vintage y vida nocturna.",
      category: ["attraction", "artwork", "marketplace"],
      location: { coordinates: [-0.3810, 39.4778] },
      rating: 4.5,
      duration: 120,
    },
  ];
}
