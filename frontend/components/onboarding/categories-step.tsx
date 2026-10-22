"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import { POI_CATEGORY_META, sanitizePoiCategoryIds } from "@/lib/poi-categories";

interface CategoriesStepProps {
  selectedCategories: string[];
  onCategoriesChange: (categories: string[]) => void;
  interests: string;
  onInterestsChange: (interests: string) => void;
  onNext: () => void;
  onBack: () => void;
  isSubmitting: boolean;
}

export function CategoriesStep({
  selectedCategories,
  onCategoriesChange,
  interests,
  onInterestsChange,
  onNext,
  onBack,
  isSubmitting,
}: CategoriesStepProps) {
  const { t } = useI18n();
  const safeSelectedCategories = sanitizePoiCategoryIds(selectedCategories);

  const toggleCategory = (categoryId: string) => {
    const normalizedCategory = sanitizePoiCategoryIds([categoryId])[0];
    if (!normalizedCategory) return;

    if (safeSelectedCategories.includes(normalizedCategory)) {
      const nextSelection = safeSelectedCategories.filter((id) => id !== normalizedCategory);
      onCategoriesChange(sanitizePoiCategoryIds(nextSelection));
    } else {
      const nextSelection = [...safeSelectedCategories, normalizedCategory];
      onCategoriesChange(sanitizePoiCategoryIds(nextSelection));
    }
  };

  const canContinue = safeSelectedCategories.length >= 2;

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-2xl w-full"
    >
      {/* Back Button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-8 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">{t("onboarding.categories.back")}</span>
      </button>

      {/* Content */}
      <div className="text-center mb-8">
        <h2 className="font-serif text-4xl text-foreground mb-4">
          {t("onboarding.categories.title")}
        </h2>
        <p className="text-muted-foreground">
          {t("onboarding.categories.subtitle")}
        </p>
      </div>

      {/* Categories Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-8">
        {POI_CATEGORY_META.map((category, index) => {
          const isSelected = safeSelectedCategories.includes(category.id);
          const Icon = category.icon;

          return (
            <motion.button
              key={category.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => toggleCategory(category.id)}
              className={cn(
                "relative flex flex-col items-center justify-center p-4 rounded-xl border-2 transition-all duration-200",
                isSelected
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border bg-card hover:border-primary/50 text-foreground"
              )}
            >
              {isSelected && (
                <div className="absolute top-2 right-2">
                  <Check className="w-4 h-4 text-primary" />
                </div>
              )}
              <Icon className="w-6 h-6 mb-2" />
              <span className="text-sm font-medium">
                {t(`onboarding.categories.labels.${category.id}`)}
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Free-text interests: embedded by the backend to steer rounds and ranking */}
      <div className="mb-6">
        <label htmlFor="interests" className="block text-sm font-medium text-foreground mb-2">
          {t("onboarding.categories.interests_label")}
        </label>
        <Textarea
          id="interests"
          value={interests}
          onChange={(e) => onInterestsChange(e.target.value)}
          placeholder={t("onboarding.categories.interests_placeholder")}
          maxLength={300}
          rows={2}
          className="resize-none"
        />
      </div>

      {/* Selection Count */}
      <div className="text-center mb-6">
        <span className="text-sm text-muted-foreground">
          {t("onboarding.categories.selected", { count: safeSelectedCategories.length })}
          {safeSelectedCategories.length < 2 &&
            t("onboarding.categories.min_hint", { min: 2 })}
        </span>
      </div>

      {/* Continue Button */}
      <Button
        size="lg"
        onClick={onNext}
        disabled={!canContinue || isSubmitting}
        className="w-full py-6 rounded-full"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            {t("onboarding.categories.saving")}
          </>
        ) : (
          t("onboarding.categories.create")
        )}
      </Button>
    </motion.div>
  );
}
