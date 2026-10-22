"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/lib/user-context";
import { generateItinerary, type Itinerary, VALENCIA_CENTER } from "@/lib/api";
import { PlannerHeader } from "@/components/planner/planner-header";
import { SelectedPlaces } from "@/components/planner/selected-places";
import { ItineraryView } from "@/components/planner/itinerary-view";
import { PlannerMap } from "@/components/planner/planner-map";
import { Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useI18n } from "@/lib/i18n";

// The plan uses the pace chosen in onboarding; there is no in-planner settings UI.
const DEFAULT_SETTINGS = {
  maxStopsPerDay: 6,
  maxTotalMinutes: 480, // 8 hours
  visitTimeMinutes: 60,
  startLocation: VALENCIA_CENTER,
};

export default function PlannerPage() {
  const router = useRouter();
  const {
    onboardingComplete,
    isHydrated,
    userId,
    acceptedRecommendations,
    removeAcceptedRecommendation,
    clearAcceptedRecommendations,
    plannerPreferences,
  } = useUser();
  const { toast } = useToast();
  const { t } = useI18n();

  const [itinerary, setItinerary] = useState<Itinerary | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const settings = useMemo(
    () => ({
      maxStopsPerDay: plannerPreferences?.max_stops_per_day ?? DEFAULT_SETTINGS.maxStopsPerDay,
      maxTotalMinutes: plannerPreferences?.max_total_minutes ?? DEFAULT_SETTINGS.maxTotalMinutes,
      visitTimeMinutes: plannerPreferences?.visit_time_minutes ?? DEFAULT_SETTINGS.visitTimeMinutes,
      startLocation: plannerPreferences?.start_location ?? DEFAULT_SETTINGS.startLocation,
    }),
    [plannerPreferences]
  );
  const [activeView, setActiveView] = useState<"places" | "itinerary">("places");

  useEffect(() => {
    if (isHydrated && !onboardingComplete) {
      router.replace("/onboarding");
    }
  }, [isHydrated, onboardingComplete, router]);

  const handleGenerateItinerary = async () => {
    if (acceptedRecommendations.length === 0) return;

    setIsGenerating(true);
    try {
      const candidates = acceptedRecommendations.map((poi) => ({
        id: poi.id,
        location: poi.location,
        category: poi.category,
        name: poi.name,
        duration: poi.duration,
      }));

      const result = await generateItinerary({
        user_id: userId || undefined,
        candidates,
        max_stops_per_day: settings.maxStopsPerDay,
        max_total_minutes: settings.maxTotalMinutes,
        visit_time_minutes: settings.visitTimeMinutes,
        start_location: settings.startLocation,
      });

      setItinerary(result);
      setActiveView("itinerary");
    } catch (error) {
      console.error("Error generating itinerary:", error);
      toast({
        title: t("planner.toast.error_title"),
        description: t("planner.toast.error_desc"),
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClearPlan = () => {
    clearAcceptedRecommendations();
    setItinerary(null);
    setActiveView("places");
  };

  if (!isHydrated || !onboardingComplete) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <PlannerHeader
        onBack={() => router.push("/explore")}
        onClear={handleClearPlan}
        hasPlaces={acceptedRecommendations.length > 0}
      />

      <main className="flex-1 flex flex-col lg:flex-row">
        {/* Map */}
        <div className="h-[35vh] lg:h-auto lg:flex-1">
          <PlannerMap
            places={acceptedRecommendations}
            itinerary={itinerary}
            startLocation={settings.startLocation}
          />
        </div>

        {/* Content Panel */}
        <div className="flex-1 lg:w-[420px] lg:flex-none overflow-hidden flex flex-col bg-background border-l border-border">
          {/* Tabs */}
          <div className="flex border-b border-border">
            <button
              onClick={() => setActiveView("places")}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                activeView === "places"
                  ? "text-primary border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t("planner.tabs.places")} ({acceptedRecommendations.length})
            </button>
            <button
              onClick={() => setActiveView("itinerary")}
              disabled={!itinerary}
              className={`flex-1 py-3 text-sm font-medium transition-colors ${
                activeView === "itinerary"
                  ? "text-primary border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground disabled:opacity-50"
              }`}
            >
              {t("planner.tabs.itinerary")}
            </button>
          </div>

          {/* Content */}
          {activeView === "places" ? (
            <SelectedPlaces
              places={acceptedRecommendations}
              onRemove={removeAcceptedRecommendation}
              onGenerate={handleGenerateItinerary}
              isGenerating={isGenerating}
            />
          ) : (
            <ItineraryView
              itinerary={itinerary}
              places={acceptedRecommendations}
              startLocation={settings.startLocation}
            />
          )}
        </div>
      </main>
    </div>
  );
}
