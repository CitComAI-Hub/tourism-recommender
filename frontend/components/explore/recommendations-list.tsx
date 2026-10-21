"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { OptimizedImage } from "@/components/ui/optimized-image";
import {
  Check,
  Plus,
  Map,
  List,
  Clock,
  Star,
  Loader2,
  Heart,
  ListFilter,
} from "lucide-react";
import type { POI } from "@/lib/api";
import { POI_CATEGORY_FILTERS, POI_CATEGORY_META } from "@/lib/poi-categories";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";

interface RecommendationsListProps {
  recommendations: POI[];
  likedPOIs?: string[];
  acceptedRecommendations: POI[];
  isLoading: boolean;
  activeFilter: string | null;
  onFilterChange: (filter: string | null) => void;
  onSelectPOI: (poi: POI) => void;
  onHoverPOI: (poi: POI | null) => void;
  onToggleAccept: (poi: POI) => void;
  showMap: boolean;
  onToggleMap: () => void;
}

export function RecommendationsList({
  recommendations,
  likedPOIs = [],
  acceptedRecommendations,
  isLoading,
  activeFilter,
  onFilterChange,
  onSelectPOI,
  onHoverPOI,
  onToggleAccept,
  showMap,
  onToggleMap,
}: RecommendationsListProps) {
  const { t } = useI18n();
  const safeRecommendations = Array.isArray(recommendations)
    ? recommendations
    : [];

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-serif text-2xl text-foreground">{t("recommendations.heading")}</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleMap}
            className="lg:hidden"
          >
            {showMap ? <List className="w-4 h-4" /> : <Map className="w-4 h-4" />}
          </Button>
        </div>

        {/* Filters */}
        <div className="w-full min-w-0 overflow-x-auto">
          <div className="flex w-max min-w-full gap-2 pb-2">
            {POI_CATEGORY_FILTERS.map((filter) => (
              <Badge
                key={filter.id || "all"}
                variant={activeFilter === filter.id ? "default" : "outline"}
                className={cn(
                  "shrink-0 cursor-pointer whitespace-nowrap transition-colors",
                  activeFilter === filter.id
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-secondary"
                )}
                onClick={() => onFilterChange(filter.id)}
              >
                {(() => {
                  const Icon = filter.id
                    ? POI_CATEGORY_META.find((category) => category.id === filter.id)?.icon
                    : ListFilter;
                  return Icon ? <Icon className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" /> : null;
                })()}
                {t(filter.key)}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4 space-y-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          ) : safeRecommendations.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                {t("recommendations.empty")}
              </p>
            </div>
          ) : (
            safeRecommendations.map((poi, idx) => {
              const isAccepted = acceptedRecommendations.some(
                (p) => p.id === poi.id
              );
              const isLiked = likedPOIs.includes(poi.id);
              const visitMinutes = poi.duration ?? 60;
              return (
                <div
                  key={poi.id || `poi-${idx}`}
                  className={cn(
                    "group relative p-4 rounded-xl border transition-all cursor-pointer",
                    isAccepted
                      ? "border-primary/30 bg-primary/5"
                      : isLiked
                        ? "border-border bg-card shadow-none"
                        : "border-border bg-card hover:border-primary/20 hover:shadow-md"
                  )}
                  onClick={() => onSelectPOI(poi)}
                  onMouseEnter={() => onHoverPOI(poi)}
                  onMouseLeave={() => onHoverPOI(null)}
                >
                  <div className="flex gap-3">
                    {/* Thumbnail */}
                    {poi.image && (
                      <OptimizedImage
                        src={poi.image}
                        alt={poi.name}
                        fill
                        sizes="64px"
                        className="object-cover rounded-lg"
                        wrapperClassName="relative shrink-0 w-16 h-16 rounded-lg"
                        fallbackIconSize={18}
                      />
                    )}

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <h3 className="font-medium text-foreground line-clamp-1 flex-1 min-w-0">
                          {poi.name}
                        </h3>
                        <div className="flex items-center gap-2 shrink-0">
                          {isLiked && (
                            <Heart className="w-4 h-4 fill-primary text-primary" />
                          )}
                          {poi.rating && (
                            <div className="flex items-center gap-1 text-sm text-muted-foreground">
                              <Star className="w-3.5 h-3.5 fill-accent text-accent" />
                              {poi.rating}
                            </div>
                          )}
                        </div>
                      </div>

                      <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                        {poi.description}
                      </p>

                      <div className="flex items-center gap-3">
                        {/* Categories */}
                        <div className="flex gap-1 flex-wrap">
                          {(poi.category || []).slice(0, 2).map((cat) => (
                            <Badge
                              key={cat}
                              variant="secondary"
                              className="text-xs px-2 py-0"
                            >
                              {cat.charAt(0).toUpperCase() + cat.slice(1)}
                            </Badge>
                          ))}
                        </div>

                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Clock className="w-3 h-3" />
                          {visitMinutes >= 60
                            ? `${Math.round(visitMinutes / 60)}h`
                            : `${visitMinutes}min`}
                        </div>

                      </div>
                    </div>

                    {/* Accept Button */}
                    <Button
                      size="icon"
                      variant={isAccepted ? "default" : "outline"}
                      className={cn(
                        "shrink-0 rounded-full h-10 w-10 transition-all",
                        isAccepted && "bg-primary text-primary-foreground"
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleAccept(poi);
                      }}
                    >
                      {isAccepted ? (
                        <Check className="w-4 h-4" />
                      ) : (
                        <Plus className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
