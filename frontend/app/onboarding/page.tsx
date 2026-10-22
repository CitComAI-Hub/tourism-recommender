"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { WelcomeStep } from "@/components/onboarding/welcome-step";
import { NameStep } from "@/components/onboarding/name-step";
import { CategoriesStep } from "@/components/onboarding/categories-step";
import { POIsStep } from "@/components/onboarding/pois-step";
import { LocationStep } from "@/components/onboarding/location-step";
import { PreferencesStep } from "@/components/onboarding/preferences-step";
import { CompletedStep } from "@/components/onboarding/completed-step";
import { useUser } from "@/lib/user-context";
import {
  saveOnboardingSelections,
  type Location,
  type PlannerPreferences,
} from "@/lib/api";
import { sanitizePoiCategoryIds } from "@/lib/poi-categories";

const STEPS = [
  "welcome",
  "name",
  "categories",
  "pois",
  "location",
  "preferences",
  "completed",
] as const;
type Step = (typeof STEPS)[number];

export default function OnboardingPage() {
  const router = useRouter();
  const {
    setUserId,
    setProfile,
    addLikedPOI,
    setPlannerPreferences,
    completeOnboarding,
  } = useUser();
  const [currentStep, setCurrentStep] = useState<Step>("welcome");
  const [name, setName] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [interests, setInterests] = useState("");
  const [likedPOIs, setLikedPOIs] = useState<string[]>([]);
  const [dislikedPOIs, setDislikedPOIs] = useState<string[]>([]);
  const [useCurrentLocation, setUseCurrentLocation] = useState(true);
  const [customLocation, setCustomLocation] = useState<Location | null>(null);
  const [maxStopsPerDay, setMaxStopsPerDay] = useState(5);
  const [maxTotalMinutes, setMaxTotalMinutes] = useState(480);
  const [visitTimeMinutes, setVisitTimeMinutes] = useState(60);
  const [recommendationStyle, setRecommendationStyle] = useState<
    "explorer" | "conservative" | "hidden_gems"
  >("explorer");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const stepIndex = STEPS.indexOf(currentStep);
  const progressSteps = STEPS.length - 2;
  const currentProgressStep = Math.max(0, stepIndex - 1);
  const progress =
    currentStep === "welcome" || currentStep === "completed"
      ? 0
      : ((currentProgressStep + 1) / progressSteps) * 100;

  const handleNext = () => {
    const nextIndex = stepIndex + 1;
    if (nextIndex < STEPS.length) {
      setCurrentStep(STEPS[nextIndex]);
    }
  };

  const handleBack = () => {
    const prevIndex = stepIndex - 1;
    if (prevIndex >= 0) {
      setCurrentStep(STEPS[prevIndex]);
    }
  };

  const handleComplete = async () => {
    setIsSubmitting(true);
    const userId = `user_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const safeSelectedCategories = sanitizePoiCategoryIds(selectedCategories);
    const profile = {
      name,
      preferred_categories: safeSelectedCategories,
      recommendation_style: recommendationStyle,
      interests: interests.trim(),
    };
    const plannerPrefs: PlannerPreferences = {
      max_stops_per_day: maxStopsPerDay,
      max_total_minutes: maxTotalMinutes,
      visit_time_minutes: visitTimeMinutes,
      start_location: customLocation,
      use_current_location: useCurrentLocation,
      mobility: "walking",
    };

    let finalUserId = userId;
    try {
      await saveOnboardingSelections({
        user_id: userId,
        profile,
        liked_pois: likedPOIs,
        disliked_pois: dislikedPOIs,
        start_location: customLocation,
        use_current_location: useCurrentLocation,
        planner_preferences: plannerPrefs,
      });
    } catch (error) {
      console.error("Error saving preferences:", error);
      finalUserId = `user_${Date.now()}`;
    }

    setUserId(finalUserId);
    setProfile(profile);
    likedPOIs.forEach((poi) => addLikedPOI(poi));
    setPlannerPreferences(plannerPrefs);
    completeOnboarding();
    setCurrentStep("completed");
    setIsSubmitting(false);
  };

  const handleStartExploring = () => {
    router.push("/explore");
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Progress Bar */}
      {currentStep !== "welcome" && currentStep !== "completed" && (
        <div className="fixed top-0 left-0 right-0 h-1 bg-secondary z-50">
          <motion.div
            className="h-full bg-primary"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center p-6">
        <AnimatePresence mode="wait">
          {currentStep === "welcome" && (
            <WelcomeStep key="welcome" onNext={handleNext} />
          )}
          {currentStep === "name" && (
            <NameStep
              key="name"
              name={name}
              onNameChange={setName}
              onNext={handleNext}
              onBack={handleBack}
            />
          )}
          {currentStep === "categories" && (
            <CategoriesStep
              key="categories"
              selectedCategories={selectedCategories}
              onCategoriesChange={setSelectedCategories}
              interests={interests}
              onInterestsChange={setInterests}
              onNext={handleNext}
              onBack={handleBack}
              isSubmitting={isSubmitting}
            />
          )}
          {currentStep === "pois" && (
            <POIsStep
              key="pois"
              selectedCategories={selectedCategories}
              interests={interests}
              likedPOIs={likedPOIs}
              onLikedPOIsChange={setLikedPOIs}
              onDislikedPOIsChange={setDislikedPOIs}
              onNext={handleNext}
              onBack={handleBack}
            />
          )}
          {currentStep === "location" && (
            <LocationStep
              key="location"
              useCurrentLocation={useCurrentLocation}
              customLocation={customLocation}
              onUseCurrentLocationChange={setUseCurrentLocation}
              onCustomLocationChange={setCustomLocation}
              onNext={handleNext}
              onBack={handleBack}
            />
          )}
          {currentStep === "preferences" && (
            <PreferencesStep
              key="preferences"
              maxStopsPerDay={maxStopsPerDay}
              maxTotalMinutes={maxTotalMinutes}
              visitTimeMinutes={visitTimeMinutes}
              onMaxStopsChange={setMaxStopsPerDay}
              onMaxTotalMinutesChange={setMaxTotalMinutes}
              onVisitTimeChange={setVisitTimeMinutes}
              onNext={handleComplete}
              onBack={handleBack}
              isSubmitting={isSubmitting}
              recommendationStyle={recommendationStyle}
              onRecommendationStyleChange={setRecommendationStyle}
            />
          )}
          {currentStep === "completed" && (
            <CompletedStep
              key="completed"
              name={name}
              onStartExploring={handleStartExploring}
            />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
