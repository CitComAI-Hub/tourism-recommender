"use client";

import React from "react"

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useI18n } from "@/lib/i18n";

interface NameStepProps {
  name: string;
  onNameChange: (name: string) => void;
  onNext: () => void;
  onBack: () => void;
}

export function NameStep({ name, onNameChange, onNext, onBack }: NameStepProps) {
  const { t } = useI18n();
  const canContinue = name.trim().length >= 2;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && canContinue) {
      onNext();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-md w-full"
    >
      {/* Back Button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-8 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">{t("onboarding.name.back")}</span>
      </button>

      {/* Content */}
      <div className="text-center mb-8">
        <h2 className="font-serif text-4xl text-foreground mb-4">
          {t("onboarding.name.title")}
        </h2>
        <p className="text-muted-foreground">
          {t("onboarding.name.subtitle")}
        </p>
      </div>

      {/* Input */}
      <div className="space-y-6">
        <Input
          type="text"
          placeholder={t("onboarding.name.placeholder")}
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="text-center text-xl py-6 border-2 focus:border-primary"
          autoFocus
        />

        <Button
          size="lg"
          onClick={onNext}
          disabled={!canContinue}
          className="w-full py-6 rounded-full"
        >
          {t("onboarding.name.continue")}
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </motion.div>
  );
}
