"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Image from "next/image";
import { motion, AnimatePresence, type PanInfo } from "framer-motion";
import { Button } from "@/components/ui/button";
import { OptimizedImage, normalizeImageUrl } from "@/components/ui/optimized-image";
import { ArrowLeft, Heart, X, Star, Clock, MapPin, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { POI } from "@/lib/api";
import { getRandomPOIs } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { poiCategoryLabel } from "@/components/poi-category-badge";

interface POIsStepProps {
  selectedCategories: string[];
  interests: string;
  likedPOIs: string[];
  onLikedPOIsChange: (pois: string[]) => void;
  onDislikedPOIsChange?: (pois: string[]) => void;
  onNext: () => void;
  onBack: () => void;
}

const DEFAULT_ROUND_SIZES = [6, 4, 4] as const;
const ROUND_SIZES: readonly number[] = (() => {
  const configured = process.env.NEXT_PUBLIC_ONBOARDING_ROUND_SIZES
    ?.split(",")
    .map((value) => Number.parseInt(value.trim(), 10))
    .filter((value) => Number.isInteger(value) && value > 0);
  return configured?.length ? configured : DEFAULT_ROUND_SIZES;
})();

export function POIsStep({
  selectedCategories,
  interests,
  likedPOIs,
  onLikedPOIsChange,
  onDislikedPOIsChange,
  onNext,
  onBack,
}: POIsStepProps) {
  const { t, locale } = useI18n();
  const [pois, setPois] = useState<POI[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [direction, setDirection] = useState<"left" | "right" | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dislikedPOIs, setDislikedPOIs] = useState<string[]>([]);
  const [currentRound, setCurrentRound] = useState(0);
  // Set when a round comes back short: the category pool has no more POIs.
  const [isExhausted, setIsExhausted] = useState(false);

  const loadRound = useCallback(
    async (roundIndex: number, append: boolean, seenIds: string[]) => {
      if (selectedCategories.length === 0) {
        setPois([]);
        setCurrentIndex(0);
        return;
      }

      const isFirstRound = !append;
      if (isFirstRound) {
        setIsLoading(true);
      } else {
        setIsLoadingMore(true);
      }
      setError(null);

      try {
        const requested = ROUND_SIZES[roundIndex] ?? 4;
        const sample = await getRandomPOIs({
          categories: selectedCategories,
          limit: requested,
          likedPOIs,
          dislikedPOIs,
          seenPOIs: seenIds,
          interests,
        });

        const deduped = sample.filter(
          (poi, index, self) => self.findIndex((p) => p.id === poi.id) === index
        );

        for (let i = 0; i < Math.min(2, deduped.length); i++) {
          const imgUrl = deduped[i]?.image;
          if (imgUrl) {
            const resolved = normalizeImageUrl(imgUrl);
            const preload = new window.Image();
            preload.src = `/_next/image?url=${encodeURIComponent(resolved)}&w=640&q=75`;
          }
        }

        if (append) {
          setPois((prev) => {
            const prevIds = new Set(prev.map((poi) => poi.id));
            const uniqueNew = deduped.filter((poi) => !prevIds.has(poi.id));
            return [...prev, ...uniqueNew];
          });
        } else {
          setPois(deduped);
          setCurrentIndex(0);
        }
        setCurrentRound(roundIndex);
        setIsExhausted(deduped.length < requested);
      } catch (err) {
        console.error("Error fetching sample POIs", err);
        setError(t("common.error"));
        if (!append) {
          setPois([]);
        }
      } finally {
        if (isFirstRound) {
          setIsLoading(false);
        } else {
          setIsLoadingMore(false);
        }
        setDirection(null);
      }
    },
    [selectedCategories, interests, likedPOIs, dislikedPOIs, t]
  );

  // The rounds restart from scratch, so earlier answers must not steer them
  // (they would turn round 1 into a refinement round over unseen cards).
  useEffect(() => {
    setDislikedPOIs([]);
    onDislikedPOIsChange?.([]);
    onLikedPOIsChange([]);
    setCurrentRound(0);
    setIsExhausted(false);
    void loadRound(0, false, []);
  }, [selectedCategories, interests, onDislikedPOIsChange, onLikedPOIsChange]);

  useEffect(() => {
    if (isLoading || isLoadingMore) return;
    if (selectedCategories.length === 0) return;
    if (currentIndex < pois.length) return;

    const nextRound = currentRound + 1;
    if (nextRound >= ROUND_SIZES.length || isExhausted) {
      return;
    }
    void loadRound(nextRound, true, pois.map((poi) => poi.id));
  }, [
    currentIndex,
    currentRound,
    isExhausted,
    isLoading,
    isLoadingMore,
    loadRound,
    pois.length,
    selectedCategories.length,
  ]);

  const currentPOI = useMemo(() => pois[currentIndex], [pois, currentIndex]);
  const isLiked = currentPOI ? likedPOIs.includes(currentPOI.id) : false;
  const hasMoreRounds = !isExhausted && currentRound < ROUND_SIZES.length - 1;
  const isFinished = currentIndex >= pois.length && pois.length > 0 && !hasMoreRounds && !isLoadingMore;

  // Upcoming POIs whose images we want to preload through next/image pipeline
  const prefetchPOIs = useMemo(() => {
    const upcoming: POI[] = [];
    for (let i = 1; i <= 3; i++) {
      const poi = pois[currentIndex + i];
      if (poi?.image) upcoming.push(poi);
    }
    return upcoming;
  }, [currentIndex, pois]);

  const handleSwipe = (swipeDirection: "left" | "right") => {
    if (isAnimating || isFinished || !currentPOI) return;

    setIsAnimating(true);
    setDirection(swipeDirection);

    if (swipeDirection === "right") {
      if (!likedPOIs.includes(currentPOI.id)) {
        onLikedPOIsChange([...likedPOIs, currentPOI.id]);
      }
    } else {
      if (!dislikedPOIs.includes(currentPOI.id)) {
        const next = [...dislikedPOIs, currentPOI.id];
        setDislikedPOIs(next);
        onDislikedPOIsChange?.(next);
      }
    }

    setTimeout(() => {
      setCurrentIndex((prev) => prev + 1);
      setDirection(null);
      setIsAnimating(false);
    }, 300);
  };

  const handleDragEnd = (
    _: MouseEvent | TouchEvent | PointerEvent,
    info: PanInfo
  ) => {
    const threshold = 100;
    if (info.offset.x > threshold) {
      handleSwipe("right");
    } else if (info.offset.x < -threshold) {
      handleSwipe("left");
    }
  };

  // Cards shown so far plus the rounds still to come; shrinks to the real
  // count when the selected categories run out of POIs.
  const totalItems = hasMoreRounds
    ? pois.length + ROUND_SIZES.slice(currentRound + 1).reduce((a, b) => a + b, 0)
    : pois.length;

  const progress = totalItems
    ? Math.min((currentIndex / totalItems) * 100, 100)
    : 0;

  const savedSuffix = likedPOIs.length !== 1 ? (locale === "es" ? "s" : "s") : "";
  const continueLabel = isFinished
    ? t("onboarding.pois.cta_finished")
    : likedPOIs.length > 0
      ? t("onboarding.pois.cta_continue_count", { count: likedPOIs.length, suffix: savedSuffix })
      : t("onboarding.pois.cta_skip");

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-md w-full mx-auto"
    >
      <button
        type="button"
        onClick={onBack}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-4 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">{t("onboarding.pois.back")}</span>
      </button>

      <div className="text-center mb-4">
        <h2 className="font-serif text-2xl md:text-3xl text-foreground mb-2 text-balance">
          {isFinished
            ? t("onboarding.pois.title_done")
            : t("onboarding.pois.title_question")}
        </h2>
        <p className="text-sm text-muted-foreground">
          {isFinished
            ? t("onboarding.pois.subtitle_done", {
              count: likedPOIs.length,
              suffix: likedPOIs.length !== 1 ? (locale === "es" ? "es" : "s") : "",
            })
            : t("onboarding.pois.subtitle_swipe")}
        </p>
      </div>

      <div className="h-1 bg-secondary rounded-full mb-6 overflow-hidden">
        <motion.div
          className="h-full bg-primary"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {(isLoading || (isLoadingMore && !currentPOI)) && (
        <div className="relative h-[420px] mb-6 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
        </div>
      )}

      {!isLoading && selectedCategories.length === 0 && (
        <div className="relative h-[420px] mb-6 flex items-center justify-center">
          <p className="text-center text-sm text-muted-foreground">
            {t("onboarding.pois.empty")}
          </p>
        </div>
      )}

      {!isLoading && !isFinished && currentPOI && (
        <div className="relative h-[420px] mb-6">
          {/* Hidden preloaders: render upcoming images off-screen via next/image so the
              /_next/image optimized version is already cached when the user swipes */}
          <div aria-hidden className="absolute w-0 h-0 overflow-hidden">
            {prefetchPOIs.map((poi) => (
              <Image
                key={`prefetch-${poi.id}`}
                src={normalizeImageUrl(poi.image!)}
                alt=""
                width={448}
                height={420}
                sizes="(max-width: 768px) 100vw, 448px"
              />
            ))}
          </div>
          <AnimatePresence mode="popLayout">
            <motion.div
              key={currentPOI.id}
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{
                scale: 1,
                opacity: 1,
                x:
                  direction === "left"
                    ? -400
                    : direction === "right"
                      ? 400
                      : 0,
                rotate:
                  direction === "left"
                    ? -15
                    : direction === "right"
                      ? 15
                      : 0,
              }}
              exit={{
                scale: 0.95,
                opacity: 0,
              }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              drag={!isAnimating ? "x" : false}
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={0.8}
              onDragEnd={handleDragEnd}
              className="absolute inset-0 cursor-grab active:cursor-grabbing"
            >
              <div className="relative w-full h-full rounded-3xl overflow-hidden shadow-xl">
                <OptimizedImage
                  src={currentPOI.image || "/placeholder.svg"}
                  alt={currentPOI.name}
                  fill
                  sizes="(max-width: 768px) 100vw, 448px"
                  priority
                  className="object-cover pointer-events-none"
                  wrapperClassName="absolute inset-0 w-full h-full"
                />

                <div className="absolute inset-0 bg-gradient-to-t from-foreground/95 via-foreground/30 to-transparent" />

                <AnimatePresence>
                  {direction === "right" && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      className="absolute inset-0 bg-primary/40 flex items-center justify-center"
                    >
                      <motion.div
                        initial={{ scale: 0.5 }}
                        animate={{ scale: 1 }}
                        className="w-24 h-24 rounded-full bg-primary flex items-center justify-center shadow-2xl"
                      >
                        <Heart className="w-12 h-12 text-primary-foreground fill-primary-foreground" />
                      </motion.div>
                    </motion.div>
                  )}
                  {direction === "left" && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      className="absolute inset-0 bg-muted-foreground/40 flex items-center justify-center"
                    >
                      <motion.div
                        initial={{ scale: 0.5 }}
                        animate={{ scale: 1 }}
                        className="w-24 h-24 rounded-full bg-muted flex items-center justify-center shadow-2xl"
                      >
                        <X className="w-12 h-12 text-muted-foreground" />
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="absolute bottom-0 left-0 right-0 p-5 text-card">
                  {currentPOI.rating && (
                    <div className="inline-flex items-center gap-1 px-2.5 py-1 bg-card/20 backdrop-blur-sm rounded-full text-sm mb-3">
                      <Star className="w-3.5 h-3.5 fill-accent text-accent" />
                      <span className="font-medium">{currentPOI.rating}</span>
                    </div>
                  )}

                  <h3 className="font-serif text-2xl font-medium leading-tight mb-2">
                    {currentPOI.name}
                  </h3>

                  {currentPOI.description && (
                    <p className="text-sm text-card/80 mb-3 line-clamp-2">
                      {currentPOI.description}
                    </p>
                  )}

                  <div className="flex items-center gap-4 text-xs text-card/70">
                    {currentPOI.duration && (
                      <span className="flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        {currentPOI.duration >= 60
                          ? `${Math.floor(currentPOI.duration / 60)}h${currentPOI.duration % 60 > 0
                            ? ` ${currentPOI.duration % 60}m`
                            : ""
                          }`
                          : `${currentPOI.duration}m`}
                      </span>
                    )}
                    {currentPOI.category?.[0] && (
                      <span className="flex items-center gap-1 capitalize">
                        <MapPin className="w-3.5 h-3.5" />
                        {poiCategoryLabel(t, currentPOI.category[0])}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      )}

      {isFinished && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative h-[420px] mb-6 flex flex-col items-center justify-center text-center p-6 rounded-3xl border-2 border-dashed border-border bg-secondary/30"
        >
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
            <Heart className="w-8 h-8 text-primary" />
          </div>
          <p className="text-lg font-medium text-foreground mb-2">
            {likedPOIs.length > 0
              ? t("onboarding.pois.saved_count", {
                count: likedPOIs.length,
                suffix: likedPOIs.length !== 1 ? (locale === "es" ? "s" : "s") : "",
              })
              : t("onboarding.pois.saved_empty")}
          </p>
          <p className="text-sm text-muted-foreground">
            {likedPOIs.length > 0
              ? t("onboarding.pois.saved_hint")
              : t("onboarding.pois.saved_empty_hint")}
          </p>
        </motion.div>
      )}

      {error && (
        <p className="text-center text-sm text-destructive mb-4">{error}</p>
      )}

      {!isLoading && pois.length > 0 && !isFinished && (
        <div className="flex items-center justify-center gap-6 mb-6">
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleSwipe("left")}
            disabled={isAnimating}
            className="w-16 h-16 rounded-full bg-card border-2 border-border shadow-lg flex items-center justify-center text-muted-foreground hover:border-destructive hover:text-destructive transition-colors disabled:opacity-50"
          >
            <X className="w-7 h-7" />
          </motion.button>

          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleSwipe("right")}
            disabled={isAnimating}
            className="w-20 h-20 rounded-full bg-primary shadow-lg flex items-center justify-center text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <Heart className="w-9 h-9" />
          </motion.button>
        </div>
      )}

      {!isLoading && pois.length > 0 && !isFinished && (
        <p className="text-center text-sm text-muted-foreground mb-6">
          {t("onboarding.pois.progress", { current: Math.min(currentIndex + 1, totalItems), total: totalItems })}
        </p>
      )}

      <Button
        size="lg"
        onClick={onNext}
        variant={isFinished ? "default" : "outline"}
        className={cn("w-full py-6 rounded-full", !isFinished && "bg-transparent")}
      >
        {continueLabel}
      </Button>
    </motion.div>
  );
}
