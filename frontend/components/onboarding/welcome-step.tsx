"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { MapPin, Compass, Calendar } from "lucide-react";
import { useI18n } from "@/lib/i18n";

interface WelcomeStepProps {
  onNext: () => void;
}

export function WelcomeStep({ onNext }: WelcomeStepProps) {
  const { t } = useI18n();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, x: -50 }}
      transition={{ duration: 0.3 }}
      className="max-w-2xl w-full text-center"
    >
      {/* Logo/Brand */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        className="mb-8"
      >
        <div className="w-20 h-20 mx-auto rounded-2xl bg-primary flex items-center justify-center mb-6">
          <MapPin className="w-10 h-10 text-primary-foreground" />
        </div>
        <div className="flex items-center justify-center gap-3 mb-2">
          <h1 className="font-serif text-5xl md:text-6xl text-foreground">
            {t("onboarding.welcome.title")}
          </h1>
        </div>
        <p className="text-muted-foreground text-lg">
          {t("onboarding.welcome.subtitle")}
        </p>
      </motion.div>

      {/* Features */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
        className="grid md:grid-cols-3 gap-6 mb-12"
      >
        <div className="p-6 rounded-xl bg-card border border-border">
          <div className="w-12 h-12 mx-auto rounded-xl bg-primary/10 flex items-center justify-center mb-4">
            <Compass className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-medium text-foreground mb-2">
            {t("onboarding.welcome.features.personalized.title")}
          </h3>
          <p className="text-sm text-muted-foreground">
            {t("onboarding.welcome.features.personalized.desc")}
          </p>
        </div>

        <div className="p-6 rounded-xl bg-card border border-border">
          <div className="w-12 h-12 mx-auto rounded-xl bg-primary/10 flex items-center justify-center mb-4">
            <MapPin className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-medium text-foreground mb-2">
            {t("onboarding.welcome.features.discover.title")}
          </h3>
          <p className="text-sm text-muted-foreground">
            {t("onboarding.welcome.features.discover.desc")}
          </p>
        </div>

        <div className="p-6 rounded-xl bg-card border border-border">
          <div className="w-12 h-12 mx-auto rounded-xl bg-primary/10 flex items-center justify-center mb-4">
            <Calendar className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-medium text-foreground mb-2">
            {t("onboarding.welcome.features.plan.title")}
          </h3>
          <p className="text-sm text-muted-foreground">
            {t("onboarding.welcome.features.plan.desc")}
          </p>
        </div>
      </motion.div>

      {/* CTA Button */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.4 }}
      >
        <Button
          size="lg"
          onClick={onNext}
          className="px-12 py-6 text-lg rounded-full"
        >
          {t("onboarding.welcome.cta")}
        </Button>
        <p className="text-sm text-muted-foreground mt-4">
          {t("onboarding.welcome.hint")}
        </p>
      </motion.div>
    </motion.div>
  );
}
