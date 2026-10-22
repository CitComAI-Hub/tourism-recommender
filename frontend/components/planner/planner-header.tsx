"use client";

import { Button } from "@/components/ui/button";
import { ArrowLeft, Trash2 } from "lucide-react";
import { useI18n } from "@/lib/i18n";

interface PlannerHeaderProps {
  onBack: () => void;
  onClear: () => void;
  hasPlaces: boolean;
}

export function PlannerHeader({
  onBack,
  onClear,
  hasPlaces,
}: PlannerHeaderProps) {
  const { t } = useI18n();
  return (
    <header className="sticky top-0 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border">
      <div className="flex items-center justify-between h-16 px-4 lg:px-6">
        {/* Back */}
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          {t("planner.back")}
        </Button>

        {/* Title */}
        <h1 className="font-serif text-xl text-foreground">{t("planner.title")}</h1>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {hasPlaces && (
            <Button variant="ghost" size="icon" onClick={onClear}>
              <Trash2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
