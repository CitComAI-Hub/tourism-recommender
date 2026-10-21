"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/lib/user-context";
import { generateItinerary, type Itinerary, VALENCIA_CENTER } from "@/lib/api";
import { PlannerHeader } from "@/components/planner/planner-header";
import { SelectedPlaces } from "@/components/planner/selected-places";
import { ItineraryView } from "@/components/planner/itinerary-view";
import { PlannerMap } from "@/components/planner/planner-map";
import { PlannerSettings } from "@/components/planner/planner-settings";
import { Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useI18n } from "@/lib/i18n";

interface PlannerSettingsInterface {
  maxStopsPerDay: number;
  maxTotalMinutes: number;
  visitTimeMinutes: number;
  startLocation: { lat: number; lon: number };
}

const isSameSettings = (a: PlannerSettingsInterface, b: PlannerSettingsInterface) =>
  a.maxStopsPerDay === b.maxStopsPerDay &&
  a.maxTotalMinutes === b.maxTotalMinutes &&
  a.visitTimeMinutes === b.visitTimeMinutes &&
  a.startLocation.lat === b.startLocation.lat &&
  a.startLocation.lon === b.startLocation.lon;

const DEFAULT_SETTINGS: PlannerSettingsInterface = {
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
    setPlannerPreferences,
  } = useUser();
  const { toast } = useToast();
  const { t } = useI18n();

  const [itinerary, setItinerary] = useState<Itinerary | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState<PlannerSettingsInterface>(() => ({
    maxStopsPerDay: plannerPreferences?.max_stops_per_day ?? DEFAULT_SETTINGS.maxStopsPerDay,
    maxTotalMinutes:
      plannerPreferences?.max_total_minutes ?? DEFAULT_SETTINGS.maxTotalMinutes,
    visitTimeMinutes:
      plannerPreferences?.visit_time_minutes ?? DEFAULT_SETTINGS.visitTimeMinutes,
    startLocation: plannerPreferences?.start_location ?? DEFAULT_SETTINGS.startLocation,
  }));
  const [showSettings, setShowSettings] = useState(false);
  const [activeView, setActiveView] = useState<"places" | "itinerary">("places");

  // Persist planner settings changes into user context
  useEffect(() => {
    if (!plannerPreferences) return;
    const next = {
      maxStopsPerDay: plannerPreferences?.max_stops_per_day ?? DEFAULT_SETTINGS.maxStopsPerDay,
      maxTotalMinutes:
        plannerPreferences?.max_total_minutes ?? DEFAULT_SETTINGS.maxTotalMinutes,
      visitTimeMinutes:
        plannerPreferences?.visit_time_minutes ?? DEFAULT_SETTINGS.visitTimeMinutes,
      startLocation:
        plannerPreferences?.start_location ?? DEFAULT_SETTINGS.startLocation,
    };

    setSettings((prev) => (isSameSettings(prev, next) ? prev : next));
  }, [plannerPreferences]);

  useEffect(() => {
    const nextPrefs = {
      max_stops_per_day: settings.maxStopsPerDay,
      max_total_minutes: settings.maxTotalMinutes,
      visit_time_minutes: settings.visitTimeMinutes,
      start_location: settings.startLocation,
    };

    if (
      plannerPreferences &&
      plannerPreferences.max_stops_per_day === nextPrefs.max_stops_per_day &&
      plannerPreferences.max_total_minutes === nextPrefs.max_total_minutes &&
      plannerPreferences.visit_time_minutes === nextPrefs.visit_time_minutes &&
      plannerPreferences.start_location?.lat === nextPrefs.start_location.lat &&
      plannerPreferences.start_location?.lon === nextPrefs.start_location.lon
    ) {
      return;
    }

    setPlannerPreferences(nextPrefs);
  }, [settings, setPlannerPreferences]);

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
        onSettings={() => setShowSettings(true)}
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

      {/* Settings Sheet */}
      <PlannerSettings
        open={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSettingsChange={setSettings}
      />
    </div>
  );
}
