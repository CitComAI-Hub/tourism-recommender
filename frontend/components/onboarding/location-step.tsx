"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ArrowLeft,
  Navigation,
  Loader2,
  Check,
  Search,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getPOIs, type Location } from "@/lib/api";
import { useI18n } from "@/lib/i18n";

interface SearchResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
}

interface Bounds {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
}

// About 10 km: the API ignores a position farther than that from every place
// (someone planning the trip from home).
const MARGIN_DEG = 0.09;

function catalogBounds(points: [number, number][]): Bounds | null {
  if (points.length === 0) return null;
  const lons = points.map((p) => p[0]);
  const lats = points.map((p) => p[1]);
  return {
    minLat: Math.min(...lats) - MARGIN_DEG,
    maxLat: Math.max(...lats) + MARGIN_DEG,
    minLon: Math.min(...lons) - MARGIN_DEG,
    maxLon: Math.max(...lons) + MARGIN_DEG,
  };
}

function inside(bounds: Bounds | null, lat: number, lon: number): boolean {
  if (!bounds) return true;
  return lat >= bounds.minLat && lat <= bounds.maxLat && lon >= bounds.minLon && lon <= bounds.maxLon;
}

interface LocationStepProps {
  useCurrentLocation: boolean;
  customLocation: Location | null;
  onUseCurrentLocationChange: (use: boolean) => void;
  onCustomLocationChange: (location: Location | null) => void;
  onNext: () => void;
  onBack: () => void;
}

export function LocationStep({
  useCurrentLocation,
  customLocation,
  onUseCurrentLocationChange,
  onCustomLocationChange,
  onNext,
  onBack,
}: LocationStepProps) {
  const { t, locale } = useI18n();
  const [isLoadingLocation, setIsLoadingLocation] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedAddress, setSelectedAddress] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUseCurrentLocation = async () => {
    setIsLoadingLocation(true);
    setLocationError(null);
    setSelectedAddress(null);
    setSearchQuery("");

    try {
      const position = await new Promise<GeolocationPosition>(
        (resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, {
            enableHighAccuracy: true,
            timeout: 10000,
          });
        }
      );

      const { latitude: lat, longitude: lon } = position.coords;
      if (!inside(bounds, lat, lon)) {
        setLocationError(t("onboarding.location.far"));
        onUseCurrentLocationChange(false);
        return;
      }
      onCustomLocationChange({ lat, lon });
      onUseCurrentLocationChange(true);
    } catch {
      setLocationError(t("onboarding.location.error"));
      onUseCurrentLocationChange(false);
    } finally {
      setIsLoadingLocation(false);
    }
  };

  const searchAddress = async (query: string) => {
    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      // viewbox = left,top,right,bottom (lon,lat): the catalog's area
      const area = bounds
        ? `&viewbox=${bounds.minLon},${bounds.maxLat},${bounds.maxLon},${bounds.minLat}&bounded=1`
        : "";
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}${area}&limit=5&addressdetails=1`,
        {
          headers: {
            "Accept-Language": locale,
          },
        }
      );
      const data = await response.json();
      setSearchResults(data);
      setShowResults(true);
    } catch (error) {
      console.error("Error searching address:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setSelectedAddress(null);

    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    searchTimeout.current = setTimeout(() => {
      searchAddress(value);
    }, 300);
  };

  const handleSelectResult = (result: SearchResult) => {
    const shortName = result.display_name.split(",").slice(0, 2).join(", ");
    setSelectedAddress(shortName);
    setSearchQuery(shortName);
    setShowResults(false);
    onCustomLocationChange({
      lat: Number.parseFloat(result.lat),
      lon: Number.parseFloat(result.lon),
    });
    onUseCurrentLocationChange(false);
    setLocationError(null);
  };

  const handleClearSearch = () => {
    setSearchQuery("");
    setSearchResults([]);
    setSelectedAddress(null);
    setShowResults(false);
    if (!useCurrentLocation) {
      onCustomLocationChange(null);
    }
    inputRef.current?.focus();
  };

  useEffect(() => {
    getPOIs()
      .then((pois) => setBounds(catalogBounds(pois.map((p) => p.location.coordinates))))
      .catch(() => setBounds(null));
  }, []);

  useEffect(() => {
    if (navigator.permissions) {
      navigator.permissions.query({ name: "geolocation" }).then((result) => {
        if (result.state === "granted" && !customLocation) {
          handleUseCurrentLocation();
        }
      });
    }
  }, []);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const hasValidLocation = useCurrentLocation || customLocation !== null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="max-w-md w-full"
    >
      <button
        type="button"
        onClick={onBack}
        className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-8 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="text-sm">{t("onboarding.location.back")}</span>
      </button>

      <div className="text-center mb-8">
        <h2 className="font-serif text-3xl md:text-4xl text-foreground mb-4">
          {t("onboarding.location.title")}
        </h2>
        <p className="text-muted-foreground">
          {t("onboarding.location.subtitle")}
        </p>
      </div>

      <motion.button
        type="button"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        onClick={handleUseCurrentLocation}
        disabled={isLoadingLocation}
        className={cn(
          "w-full p-4 rounded-2xl border-2 flex items-center gap-4 transition-all mb-4",
          useCurrentLocation && !locationError
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50"
        )}
      >
        <div
          className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center",
            useCurrentLocation && !locationError
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-muted-foreground"
          )}
        >
          {isLoadingLocation ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : useCurrentLocation && !locationError ? (
            <Check className="w-5 h-5" />
          ) : (
            <Navigation className="w-5 h-5" />
          )}
        </div>
        <div className="text-left flex-1">
          <p className="font-medium text-foreground">{t("onboarding.location.current")}</p>
          <p className="text-sm text-muted-foreground">
            {isLoadingLocation
              ? t("onboarding.location.getting")
              : t("onboarding.location.use_gps")}
          </p>
        </div>
      </motion.button>

      {locationError && (
        <p className="text-sm text-destructive mb-4 text-center">{locationError}</p>
      )}

      <div className="flex items-center gap-4 my-6">
        <div className="flex-1 h-px bg-border" />
        <span className="text-sm text-muted-foreground">
          {t("onboarding.location.or_search")}
        </span>
        <div className="flex-1 h-px bg-border" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative"
      >
        <div className="flex gap-3 items-center p-4 rounded-2xl border-2 border-border focus-within:border-primary transition-colors">
          <Search className="w-5 h-5 text-muted-foreground" />
          <Input
            ref={inputRef}
            placeholder={t("onboarding.location.placeholder")}
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="border-none shadow-none focus-visible:ring-0 p-0"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={handleClearSearch}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        <AnimatePresence>
          {showResults && searchResults.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="absolute left-0 right-0 mt-2 bg-popover border rounded-2xl shadow-lg max-h-64 overflow-y-auto z-10"
            >
              {searchResults.map((result) => (
                <button
                  key={result.place_id}
                  type="button"
                  onClick={() => handleSelectResult(result)}
                  className="w-full text-left px-4 py-3 hover:bg-muted transition-colors"
                >
                  <p className="font-medium text-foreground">
                    {result.display_name.split(",")[0]}
                  </p>
                  <p className="text-sm text-muted-foreground line-clamp-1">
                    {result.display_name}
                  </p>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {isSearching && (
        <p className="text-sm text-muted-foreground mt-3">{t("onboarding.location.searching")}</p>
      )}

      {selectedAddress && (
        <div className="mt-4 p-3 rounded-xl bg-secondary text-sm text-foreground">
          <p className="font-medium">{t("onboarding.location.selected_label")}</p>
          <p className="text-muted-foreground">{selectedAddress}</p>
        </div>
      )}

      <Button
        size="lg"
        onClick={onNext}
        disabled={!hasValidLocation}
        className="w-full mt-8 py-6 rounded-full"
      >
        {hasValidLocation
          ? t("onboarding.name.continue")
          : t("onboarding.location.select_prompt")}
      </Button>
    </motion.div>
  );
}
