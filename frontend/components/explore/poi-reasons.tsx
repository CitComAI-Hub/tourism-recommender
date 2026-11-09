"use client";

import { Flame, MapPin, Users, Sparkles } from "lucide-react";
import type { POI } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

/** Why the recommender put this place in the list (sent by the API with each result). */
export function PoiReasons({ poi, className, compact = false }: { poi: POI; className?: string; compact?: boolean }) {
  const { t } = useI18n();
  const reasons = poi.reasons ?? [];
  const lines = reasons.map((reason) => {
    if (reason.type === "popular") return { icon: Flame, text: t("recommendations.reasons.popular") };
    if (reason.type === "near")
      return {
        icon: MapPin,
        text: reason.poi
          ? t("recommendations.reasons.near_named", { km: reason.km, poi: reason.poi })
          : t("recommendations.reasons.near", { km: reason.km }),
      };
    return { icon: Users, text: t("recommendations.reasons.similar", { poi: reason.poi }) };
  });
  if (!compact && poi.popularitySource === "predicted") {
    lines.push({ icon: Sparkles, text: t("recommendations.popularity_predicted") });
  }
  if (lines.length === 0) return null;
  const shown = compact ? lines.slice(0, 1) : lines;
  return (
    <ul className={cn("space-y-1 text-xs text-muted-foreground", className)}>
      {shown.map(({ icon: Icon, text }) => (
        <li key={text} className="flex items-center gap-1.5">
          <Icon className="h-3.5 w-3.5 shrink-0 text-primary" aria-hidden="true" />
          <span className="line-clamp-1">{text}</span>
        </li>
      ))}
    </ul>
  );
}
