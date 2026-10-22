"use client";

import { Button } from "@/components/ui/button";
import { PoiCategoryBadge } from "@/components/poi-category-badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { X, Clock, Loader2, Sparkles, MapPin } from "lucide-react";
import type { POI } from "@/lib/api";
import { useI18n } from "@/lib/i18n";

interface SelectedPlacesProps {
  places: POI[];
  onRemove: (id: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

export function SelectedPlaces({
  places,
  onRemove,
  onGenerate,
  isGenerating,
}: SelectedPlacesProps) {
  const { t } = useI18n();
  if (places.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
          <MapPin className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="font-medium text-foreground mb-2">{t("planner.selected.empty_title")}</h3>
        <p className="text-sm text-muted-foreground max-w-xs">
          {t("planner.selected.empty_desc")}
        </p>
      </div>
    );
  }

  const totalDuration = places.reduce((acc, p) => acc + (p.duration || 60), 0);
  const hours = Math.floor(totalDuration / 60);
  const minutes = totalDuration % 60;

  return (
    <div className="flex flex-col h-full">
      {/* Summary */}
      <div className="p-4 border-b border-border bg-secondary/30">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            {places.length === 1
              ? t("planner.selected.summary.singular", { count: places.length })
              : t("planner.selected.summary.plural", { count: places.length })}
          </span>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>
              ~{hours}h {minutes > 0 && `${minutes}min`}
            </span>
          </div>
        </div>
      </div>

      {/* Places List */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {places.map((place, index) => (
            <div
              key={place.id}
              className="flex items-start gap-3 p-3 rounded-xl bg-card border border-border"
            >
              {/* Order Number */}
              <div className="w-7 h-7 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm font-medium shrink-0">
                {index + 1}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-foreground line-clamp-1">
                  {place.name}
                </h4>
                <div className="flex items-center gap-2 mt-1">
                  {(Array.isArray(place.category) ? place.category : []).slice(0, 2).map((cat) => (
                    <PoiCategoryBadge key={cat} category={cat} />
                  ))}
                  {place.duration && (
                    <span className="text-xs text-muted-foreground">
                      {Math.round(place.duration / 60)}h
                    </span>
                  )}
                </div>
              </div>

              {/* Remove */}
              <Button
                variant="ghost"
                size="icon"
                className="shrink-0 h-8 w-8"
                onClick={() => onRemove(place.id)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Generate Button */}
      <div className="p-4 border-t border-border">
        <Button
          onClick={onGenerate}
          disabled={isGenerating || places.length === 0}
          className="w-full py-6 rounded-full"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              {t("planner.selected.generating")}
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 mr-2" />
              {t("planner.selected.generate")}
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
