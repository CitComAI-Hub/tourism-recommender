"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Clock, MapPin, Footprints, Loader2, Compass, ShieldCheck, Gem } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";

interface PreferencesStepProps {
  maxStopsPerDay: number;
  maxTotalMinutes: number;
  visitTimeMinutes: number;
  onMaxStopsChange: (value: number) => void;
  onMaxTotalMinutesChange: (value: number) => void;
  onVisitTimeChange: (value: number) => void;
  onNext: () => void;
  onBack: () => void;
  isSubmitting: boolean;
  recommendationStyle: "explorer" | "conservative" | "hidden_gems";
  onRecommendationStyleChange: (value: "explorer" | "conservative" | "hidden_gems") => void;
}

const PACE_OPTIONS = [
  {
    id: "relaxed",
    labelKey: "onboarding.preferences.pace.relaxed.label",
    descriptionKey: "onboarding.preferences.pace.relaxed.desc",
    stops: 3,
    visitTime: 90,
    totalMinutes: 360,
    icon: "relaxed",
  },
  {
    id: "balanced",
    labelKey: "onboarding.preferences.pace.balanced.label",
    descriptionKey: "onboarding.preferences.pace.balanced.desc",
    stops: 5,
    visitTime: 60,
    totalMinutes: 480,
    icon: "balanced",
  },
  {
    id: "intensive",
    labelKey: "onboarding.preferences.pace.intensive.label",
    descriptionKey: "onboarding.preferences.pace.intensive.desc",
    stops: 8,
    visitTime: 45,
    totalMinutes: 600,
    icon: "intensive",
  },
];

const TIME_OPTIONS = [
  { labelKey: "onboarding.preferences.time_options.half.label", value: 240, descriptionKey: "onboarding.preferences.time_options.half.desc" },
  { labelKey: "onboarding.preferences.time_options.full.label", value: 480, descriptionKey: "onboarding.preferences.time_options.full.desc" },
  { labelKey: "onboarding.preferences.time_options.long.label", value: 600, descriptionKey: "onboarding.preferences.time_options.long.desc" },
];

export function PreferencesStep({
  maxStopsPerDay,
  maxTotalMinutes,
  visitTimeMinutes,
  onMaxStopsChange,
  onMaxTotalMinutesChange,
  onVisitTimeChange,
  onNext,
  onBack,
  isSubmitting,
  recommendationStyle,
  onRecommendationStyleChange,
}: PreferencesStepProps) {
  const { t } = useI18n();
  const selectedPace =
    PACE_OPTIONS.find((p) => p.stops === maxStopsPerDay)?.id || "balanced";

  const handlePaceSelect = (pace: (typeof PACE_OPTIONS)[0]) => {
    onMaxStopsChange(pace.stops);
    onVisitTimeChange(pace.visitTime);
    onMaxTotalMinutesChange(pace.totalMinutes);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-md w-full"
    >
      <button
        type="button"
        onClick={onBack}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-8 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">{t("onboarding.preferences.back")}</span>
      </button>

      <div className="text-center mb-8">
        <h2 className="font-serif text-3xl md:text-4xl text-foreground mb-4">
          {t("onboarding.preferences.title")}
        </h2>
        <p className="text-muted-foreground">
          {t("onboarding.preferences.subtitle")}
        </p>
      </div>

      <div className="space-y-3 mb-8">
        {PACE_OPTIONS.map((pace, index) => (
          <motion.button
            key={pace.id}
            type="button"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => handlePaceSelect(pace)}
            className={cn(
              "w-full p-4 rounded-2xl border-2 flex items-start gap-4 transition-all text-left",
              selectedPace === pace.id
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            )}
          >
            <div
              className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0",
                selectedPace === pace.id
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-muted-foreground"
              )}
            >
              {pace.id === "relaxed" && <Clock className="w-5 h-5" />}
              {pace.id === "balanced" && <MapPin className="w-5 h-5" />}
              {pace.id === "intensive" && <Footprints className="w-5 h-5" />}
            </div>
            <div className="flex-1">
                <p className="font-medium text-foreground mb-1">{t(pace.labelKey)}</p>
                <p className="text-sm text-muted-foreground mb-2">
                  {t(pace.descriptionKey)}
                </p>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>
                    {t(`onboarding.preferences.pace.${pace.id}.stops`, {
                      stops: pace.stops,
                    })}
                  </span>
                  <span className="w-1 h-1 rounded-full bg-muted-foreground" />
                  <span>
                    {t(`onboarding.preferences.pace.${pace.id}.total`, {
                      hours: Math.floor(pace.totalMinutes / 60),
                    })}
                  </span>
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mb-8"
      >
        <p className="text-sm font-medium text-foreground mb-3">
          {t("onboarding.preferences.time_available")}
        </p>
        <div className="flex gap-2">
          {TIME_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => onMaxTotalMinutesChange(option.value)}
              className={cn(
                "flex-1 p-3 rounded-xl border text-center transition-all",
                maxTotalMinutes === option.value
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <p className="text-sm font-medium text-foreground">
                {t(option.labelKey)}
              </p>
              <p className="text-xs text-muted-foreground">
                {t(option.descriptionKey)}
              </p>
            </button>
          ))}
        </div>
      </motion.div>

      <div className="mb-8">
          <p className="text-sm font-medium text-foreground mb-3">
            {t("onboarding.preferences.recommendation_style.title")}
          </p>
          <div className="grid grid-cols-2 gap-2">
            {([
              ["explorer", Compass],
              ["conservative", ShieldCheck],
              ["hidden_gems", Gem],
            ] as const).map(([value, Icon]) => (
              <button
                key={value}
                type="button"
                onClick={() => onRecommendationStyleChange(value)}
                className={cn(
                  "p-3 rounded-xl border text-left transition-all",
                  recommendationStyle === value
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                )}
              >
                <Icon className="w-5 h-5 mb-1" />
                <span className="block text-sm font-medium">{t(`onboarding.preferences.recommendation_style.${value}.label`)}</span>
                <span className="block text-xs text-muted-foreground">{t(`onboarding.preferences.recommendation_style.${value}.desc`)}</span>
              </button>
            ))}
          </div>
        </div>

      <Button
        size="lg"
        onClick={onNext}
        disabled={isSubmitting}
        className="w-full py-6 rounded-full"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            {t("onboarding.preferences.creating")}
          </>
        ) : (
          t("onboarding.preferences.complete")
        )}
      </Button>
    </motion.div>
  );
}
