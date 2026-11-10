"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PoiCategoryBadge } from "@/components/poi-category-badge";
import { OptimizedImage } from "@/components/ui/optimized-image";
import { Check, Heart, ThumbsDown, Clock, MapPin, Navigation, X } from "lucide-react";
import { PoiReasons } from "@/components/explore/poi-reasons";
import type { POI } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import { useState } from "react";

const DAY_ORDER = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"] as const;

// Image comes from ingestion (Commons) via `poi.image`. No client fetch fallback to avoid mismatches.

function parseOpeningSegments(raw: string) {
  return raw
    .split(";")
    .map((segment) => segment.trim())
    .filter(Boolean)
    .map((segment) => {
      const parts = segment.split(/\s+/);
      let dayToken: string | undefined;
      let hoursToken: string | undefined;
      if (parts.length === 2) {
        [dayToken, hoursToken] = parts;
      } else {
        hoursToken = parts[0];
      }
      if (!hoursToken || !hoursToken.includes("-")) return null;
      const [startStr, endStr] = hoursToken.split("-");
      const [sh, sm] = startStr.split(":").map(Number);
      const [eh, em] = endStr.split(":").map(Number);
      if ([sh, sm, eh, em].some((v) => Number.isNaN(v))) return null;
      const days: string[] = [];
      if (dayToken) {
        if (dayToken.includes("-")) {
          const [a, b] = dayToken.split("-");
          const ai = DAY_ORDER.indexOf(a as any);
          const bi = DAY_ORDER.indexOf(b as any);
          if (ai !== -1 && bi !== -1) {
            if (ai <= bi) days.push(...DAY_ORDER.slice(ai, bi + 1));
            else days.push(...DAY_ORDER.slice(ai), ...DAY_ORDER.slice(0, bi + 1));
          }
        } else if (DAY_ORDER.includes(dayToken as any)) {
          days.push(dayToken);
        }
      }
      if (days.length === 0) days.push(...DAY_ORDER);
      return { days, start: sh * 60 + sm, end: eh * 60 + em };
    })
    .filter(Boolean) as Array<{ days: string[]; start: number; end: number }>;
}

function isOpenNow(raw?: string) {
  if (!raw) return { status: "unknown" as const, raw };
  const segments = parseOpeningSegments(raw);
  if (segments.length === 0) return { status: "unknown" as const, raw };
  const now = new Date();
  const minuteOfDay = now.getHours() * 60 + now.getMinutes();
  const dayCode = DAY_ORDER[(now.getDay() + 6) % 7];
  const open = segments.some((seg) => seg.days.includes(dayCode) && seg.start <= minuteOfDay && minuteOfDay < seg.end);
  return { status: open ? "open" : "closed", raw };
}

interface POIDetailSheetProps {
  poi: POI | null;
  isLiked: boolean;
  isVisited: boolean;
  onToggleVisited: (poi: POI) => void;
  onClose: () => void;
  onLike: (poi: POI) => void;
  onDislike: (poi: POI) => void;
}

export function POIDetailSheet({
  poi,
  isLiked,
  isVisited,
  onToggleVisited,
  onClose,
  onLike,
  onDislike,
}: POIDetailSheetProps) {
  const isOpen = !!poi;
  const { t } = useI18n();

  const categories = poi?.category || [];
  const description = poi?.description || t("poi.fallback_desc");
  const descriptionId = poi && description ? `poi-desc-${poi.id || "detail"}` : undefined;
  const heroImage = poi?.image || null;
  const [showFullDesc, setShowFullDesc] = useState(false);

  const opening =
    typeof poi?.openingHours === "string"
      ? poi.openingHours
      : poi?.openingHours?.value || (poi as any)?.opening_hours;
  const openingStatus = isOpenNow(opening);

  const handleGetDirections = () => {
    if (!poi) return;
    const url = `https://www.google.com/maps/dir/?api=1&destination=${poi.location.coordinates[1]},${poi.location.coordinates[0]}`;
    window.open(url, "_blank");
  };

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        side="bottom"
        className="h-[85vh] md:h-[72vh] rounded-t-3xl p-0 [&>button:last-child]:hidden z-[1000] overflow-y-auto"
        {...(descriptionId ? { "aria-describedby": descriptionId } : {})}
      >
        {poi && (
          <>
            {/* Header image: provided by ingestion (Commons); fallback icon */}
            <div className="relative h-48 min-h-[12rem] overflow-hidden rounded-t-3xl bg-gradient-to-br from-primary/20 to-accent/20">
              {heroImage ? (
                <OptimizedImage
                  src={heroImage}
                  alt={poi.name}
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="object-cover"
                  wrapperClassName="absolute inset-0 h-full w-full"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <MapPin className="w-16 h-16 text-primary/30" />
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-black/25 to-transparent" aria-hidden="true" />

              {/* Close Button */}
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-4 right-4 bg-background/80 backdrop-blur rounded-full"
                onClick={onClose}
              >
                <X className="w-5 h-5" />
              </Button>
            </div>

            {/* Content */}
            <div className="p-6 pb-24 -mt-6 bg-background rounded-t-3xl relative">
              <SheetHeader className="text-left mb-6">
                <div className="flex items-start justify-between gap-4">
                  <SheetTitle className="font-serif text-2xl text-foreground">
                    {poi.name}
                  </SheetTitle>
                </div>
              </SheetHeader>

              {/* Categories */}
              <div className="flex flex-wrap gap-2 mb-4">
                {categories.map((cat) => (
                  <PoiCategoryBadge key={cat} category={cat} className="text-sm px-2.5 py-0.5" />
                ))}
              </div>

              {opening && (
                <div className="flex items-center gap-2 mb-4 text-sm text-muted-foreground">
                  <Clock className="w-4 h-4" />
                  <span className="truncate">{opening}</span>
                  {openingStatus.status !== "unknown" && (
                    <Badge
                      variant={openingStatus.status === "open" ? "default" : "secondary"}
                      className={cn(
                        "ml-1",
                        openingStatus.status === "open"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      )}
                    >
                      {openingStatus.status === "open"
                        ? t("poi.open_now")
                        : t("poi.closed_now")}
                    </Badge>
                  )}
                </div>
              )}

              {/* Info Row */}
              {poi.duration && (
                <div className="flex items-center gap-4 mb-6 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    <span>
                      {Math.floor(poi.duration / 60)}h{" "}
                      {poi.duration % 60 > 0 && `${poi.duration % 60}min`}
                    </span>
                  </div>
                </div>
              )}

              {(poi.reasons?.length || poi.popularitySource === "predicted") && (
                <div className="mb-6 rounded-xl border border-border p-3">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {t("poi.why")}
                  </p>
                  <PoiReasons poi={poi} className="text-sm" />
                </div>
              )}

              {/* Description */}
              <div className="text-muted-foreground leading-relaxed mb-8 space-y-2">
                <p id={descriptionId}>
                  {showFullDesc || description.length <= 220
                    ? description
                    : `${description.slice(0, 220)}…`}
                </p>
                {poi.descriptionGenerated && (
                  <p className="text-xs italic">{t("poi.generated_desc")}</p>
                )}
                {description.length > 220 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="px-0 text-primary"
                    onClick={() => setShowFullDesc((v) => !v)}
                  >
                    {showFullDesc ? t("common.show_less") : t("common.show_more")}
                  </Button>
                )}
              </div>

              {/* Actions */}
              <Button
                variant="outline"
                aria-pressed={isVisited}
                className={cn("w-full mb-3 rounded-full", isVisited && "border-primary text-primary")}
                onClick={() => onToggleVisited(poi)}
              >
                <Check className="w-4 h-4 mr-2" />
                {isVisited ? t("poi.visited") : t("poi.visit")}
              </Button>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  className="flex-1 py-6 rounded-full bg-transparent"
                  onClick={handleGetDirections}
                >
                  <Navigation className="w-4 h-4 mr-2" />
                  {t("poi.direction")}
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-auto w-14 rounded-full"
                  aria-label={t("poi.dislike")}
                  title={t("poi.dislike")}
                  onClick={() => onDislike(poi)}
                >
                  <ThumbsDown className="w-4 h-4" />
                </Button>
                <Button
                  className={cn("flex-1 py-6 rounded-full", isLiked && "bg-primary text-primary-foreground")}
                  onClick={() => onLike(poi)}
                >
                  <Heart className={cn("w-4 h-4 mr-2", isLiked && "fill-current")} />
                  {isLiked ? t("poi.liked") : t("poi.like")}
                </Button>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
