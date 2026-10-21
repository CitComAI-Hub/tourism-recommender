"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { OptimizedImage } from "@/components/ui/optimized-image";
import { Check, Plus, Clock, Star, MapPin, Navigation, X } from "lucide-react";
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
  isAccepted: boolean;
  onClose: () => void;
  onToggleAccept: (poi: POI) => void;
}

export function POIDetailSheet({
  poi,
  isAccepted,
  onClose,
  onToggleAccept,
}: POIDetailSheetProps) {
  const isOpen = !!poi;
  const { t } = useI18n();

  const categories = poi?.category || [];
  const description = poi?.description || t("planner.detail.fallback_desc");
  const descriptionId = poi && description ? `poi-desc-${poi.id || "detail"}` : undefined;
  const heroImage = poi?.image || null;
  const [showFullDesc, setShowFullDesc] = useState(false);

  const opening = poi?.openingHours?.value || (poi as any)?.opening_hours;
  const openingStatus = isOpenNow(opening);

  const handleGetDirections = () => {
    if (!poi) return;
    const url = `https://www.google.com/maps/dir/?api=1&destination=${poi.location.coordinates[1]},${poi.location.coordinates[0]}`;
    window.open(url, "_blank");
  };

  const handleToggle = () => {
    if (poi) {
      onToggleAccept(poi);
    }
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
                  {poi.rating && (
                    <div className="flex items-center gap-1 text-sm bg-accent/10 text-accent px-2 py-1 rounded-full">
                      <Star className="w-4 h-4 fill-accent" />
                      {poi.rating}
                    </div>
                  )}
                </div>
              </SheetHeader>

              {/* Categories */}
              <div className="flex flex-wrap gap-2 mb-4">
                {categories.map((cat) => (
                  <Badge key={cat} variant="secondary" className="capitalize">
                    {cat}
                  </Badge>
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
                        ? t("planner.detail.open_now")
                        : t("planner.detail.closed_now")}
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

              {/* Description */}
              <div className="text-muted-foreground leading-relaxed mb-8 space-y-2">
                <p id={descriptionId}>
                  {showFullDesc || description.length <= 220
                    ? description
                    : `${description.slice(0, 220)}…`}
                </p>
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
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  className="flex-1 py-6 rounded-full bg-transparent"
                  onClick={handleGetDirections}
                >
                  <Navigation className="w-4 h-4 mr-2" />
                  {t("planner.detail.direction")}
                </Button>
                <Button
                  className={cn(
                    "flex-1 py-6 rounded-full",
                    isAccepted && "bg-primary text-primary-foreground"
                  )}
                  onClick={handleToggle}
                >
                  {isAccepted ? (
                    <>
                      <Check className="w-4 h-4 mr-2" />
                      {t("planner.detail.in_plan")}
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4 mr-2" />
                      {t("planner.detail.add_plan")}
                    </>
                  )}
                </Button>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
