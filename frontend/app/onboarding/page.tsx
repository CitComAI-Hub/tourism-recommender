"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { WelcomeStep } from "@/components/onboarding/welcome-step";
import { NameStep } from "@/components/onboarding/name-step";
import { CategoriesStep } from "@/components/onboarding/categories-step";
import { POIsStep } from "@/components/onboarding/pois-step";
import { LocationStep } from "@/components/onboarding/location-step";
import { CompletedStep } from "@/components/onboarding/completed-step";
import { useUser } from "@/lib/user-context";
import type { Location } from "@/lib/api";
import { sanitizePoiCategoryIds } from "@/lib/poi-categories";

const STEPS = [
  "welcome",
  "name",
  "categories",
  "pois",
  "location",
  "completed",
] as const;
type Step = (typeof STEPS)[number];

export default function OnboardingPage() {
  const router = useRouter();
  const { setProfile, setFeedback, setLocationPreference, completeOnboarding } = useUser();
  const [currentStep, setCurrentStep] = useState<Step>("welcome");
  const [name, setName] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [likedPOIs, setLikedPOIs] = useState<string[]>([]);
  const [dislikedPOIs, setDislikedPOIs] = useState<string[]>([]);
  const [useCurrentLocation, setUseCurrentLocation] = useState(true);
  const [customLocation, setCustomLocation] = useState<Location | null>(null);

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

  // Everything stays in this browser: the API is stateless.
  const handleComplete = () => {
    setProfile({ name, preferred_categories: sanitizePoiCategoryIds(selectedCategories) });
    setFeedback(likedPOIs, dislikedPOIs);
    setLocationPreference(customLocation, useCurrentLocation);
    completeOnboarding();
    setCurrentStep("completed");
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
              onNext={handleNext}
              onBack={handleBack}
              isSubmitting={false}
            />
          )}
          {currentStep === "pois" && (
            <POIsStep
              key="pois"
              selectedCategories={selectedCategories}
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
              onNext={handleComplete}
              onBack={handleBack}
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
