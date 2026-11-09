"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PoiCategoryBadge } from "@/components/poi-category-badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { OptimizedImage } from "@/components/ui/optimized-image";
import { Input } from "@/components/ui/input";
import { PoiReasons } from "@/components/explore/poi-reasons";
import { Map, List, Clock, Loader2, Heart, ListFilter, Search, ThumbsDown, X } from "lucide-react";
import type { POI } from "@/lib/api";
import { POI_CATEGORY_FILTERS, POI_CATEGORY_META } from "@/lib/poi-categories";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";

interface RecommendationsListProps {
  recommendations: POI[];
  likedItems: POI[];
  isLoading: boolean;
  activeFilter: string | null;
  onFilterChange: (filter: string | null) => void;
  query: string;
  onQueryChange: (query: string) => void;
  isSearching: boolean;
  onSelectPOI: (poi: POI) => void;
  onHoverPOI: (poi: POI | null) => void;
  onLike: (poi: POI) => void;
  onDislike: (poi: POI) => void;
  onUnlike: (poi: POI) => void;
  showMap: boolean;
  onToggleMap: () => void;
}

export function RecommendationsList({
  recommendations,
  likedItems,
  isLoading,
  activeFilter,
  onFilterChange,
  query,
  onQueryChange,
  isSearching,
  onSelectPOI,
  onHoverPOI,
  onLike,
  onDislike,
  onUnlike,
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
          <h2 className="font-serif text-2xl text-foreground">
            {isSearching ? t("recommendations.search_results") : t("recommendations.heading")}
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleMap}
            className="lg:hidden"
          >
            {showMap ? <List className="w-4 h-4" /> : <Map className="w-4 h-4" />}
          </Button>
        </div>

        {/* Free-text search (semantic, over the descriptions) */}
        <div className="relative mb-3">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder={t("recommendations.search_placeholder")}
            className="rounded-full pl-9 pr-9"
            aria-label={t("recommendations.search_placeholder")}
          />
          {query && (
            <button
              type="button"
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
              onClick={() => onQueryChange("")}
              aria-label={t("common.clear")}
            >
              <X className="h-4 w-4" />
            </button>
          )}
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
          {!isSearching && likedItems.length > 0 && (
            <div className="mb-2">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {t("recommendations.liked_heading")}
              </p>
              <div className="flex flex-wrap gap-2">
                {likedItems.map((poi) => (
                  <span
                    key={poi.id}
                    className="inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/5 py-1 pl-3 pr-1 text-sm"
                  >
                    <button type="button" className="max-w-[12rem] truncate" onClick={() => onSelectPOI(poi)}>
                      {poi.name}
                    </button>
                    <button
                      type="button"
                      className="rounded-full p-1 text-muted-foreground hover:text-foreground"
                      onClick={() => onUnlike(poi)}
                      aria-label={t("common.remove")}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          ) : safeRecommendations.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                {isSearching ? t("recommendations.search_empty") : t("recommendations.empty")}
              </p>
            </div>
          ) : (
            safeRecommendations.map((poi, idx) => {
              const isLiked = likedItems.some((p) => p.id === poi.id);
              const visitMinutes = poi.duration ?? 60;
              return (
                <div
                  key={poi.id || `poi-${idx}`}
                  className={cn(
                    "group relative p-4 rounded-xl border transition-all cursor-pointer",
                    isLiked
                      ? "border-primary/30 bg-primary/5"
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
                        </div>
                      </div>

                      <PoiReasons poi={poi} compact className="mb-1" />
                      <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                        {poi.description}
                      </p>

                      <div className="flex items-center gap-3">
                        {/* Categories */}
                        <div className="flex gap-1 flex-wrap">
                          {(poi.category || []).slice(0, 2).map((cat) => (
                            <PoiCategoryBadge key={cat} category={cat} />
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

                    {/* Feedback: each answer re-ranks the list */}
                    <div className="flex shrink-0 flex-col gap-2">
                      <Button
                        size="icon"
                        variant={isLiked ? "default" : "outline"}
                        className="h-9 w-9 rounded-full"
                        aria-label={t("poi.like")}
                        title={t("poi.like")}
                        onClick={(e) => {
                          e.stopPropagation();
                          onLike(poi);
                        }}
                      >
                        <Heart className={cn("w-4 h-4", isLiked && "fill-current")} />
                      </Button>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-9 w-9 rounded-full text-muted-foreground"
                        aria-label={t("poi.dislike")}
                        title={t("poi.dislike")}
                        onClick={(e) => {
                          e.stopPropagation();
                          onDislike(poi);
                        }}
                      >
                        <ThumbsDown className="w-4 h-4" />
                      </Button>
                    </div>
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
