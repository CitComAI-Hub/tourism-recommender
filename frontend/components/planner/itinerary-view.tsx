"use client";

import { PoiCategoryBadge } from "@/components/poi-category-badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Clock, MapPin, Route, Navigation, Flag } from "lucide-react";
import type { Itinerary, POI, Location } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";

interface ItineraryViewProps {
  itinerary: Itinerary | null;
  places: POI[];
  startLocation?: Location | null;
}

export function ItineraryView({ itinerary, places, startLocation }: ItineraryViewProps) {
  const { t } = useI18n();
  if (!itinerary) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
          <Route className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="font-medium text-foreground mb-2">
          {t("planner.itinerary.empty_title")}
        </h3>
        <p className="text-sm text-muted-foreground max-w-xs">
          {t("planner.itinerary.empty_desc")}
        </p>
      </div>
    );
  }

  const formatDuration = (minutes?: number) => {
    if (minutes === undefined || isNaN(minutes)) return "-";
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours === 0) return `${mins}min`;
    if (mins === 0) return `${hours}h`;
    return `${hours}h ${mins}min`;
  };

  const totalMinutes = (itinerary as any).total_duration_minutes ?? (itinerary as any).totalMinutes ?? 0;
  const totalDistanceKm = (itinerary as any).total_distance_km ?? (itinerary as any).totalDistanceKm;
  const legs = (itinerary as any).legs as Array<{ distance_m?: number; duration_s?: number }> | undefined;
  const startToFirstLeg = legs && legs.length > 0 ? legs[0] : undefined;

  const toLatLon = (coords?: [number, number]) => {
    if (!coords || coords.length < 2) return null;
    const [lon, lat] = coords;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    return { lat, lon };
  };

  const haversineKm = (a: { lat: number; lon: number }, b: { lat: number; lon: number }) => {
    const R = 6371;
    const dLat = ((b.lat - a.lat) * Math.PI) / 180;
    const dLon = ((b.lon - a.lon) * Math.PI) / 180;
    const lat1 = (a.lat * Math.PI) / 180;
    const lat2 = (b.lat * Math.PI) / 180;
    const sinDLat = Math.sin(dLat / 2);
    const sinDLon = Math.sin(dLon / 2);
    const h = sinDLat * sinDLat + Math.cos(lat1) * Math.cos(lat2) * sinDLon * sinDLon;
    const c = 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
    return R * c;
  };

  const startToFirst = (() => {
    if (!startLocation || !itinerary.stops.length) return null;
    const firstName =
      (itinerary.stops[0] as any).name || (itinerary.stops[0] as any).poi?.name || t("planner.itinerary.first_stop");
    if (startToFirstLeg && typeof startToFirstLeg.duration_s === "number") {
      return { minutes: Math.round(startToFirstLeg.duration_s / 60), firstName };
    }
    const firstCoords = toLatLon((itinerary.stops[0] as any).location?.coordinates || (itinerary.stops[0] as any).poi?.location?.coordinates);
    if (!firstCoords) return null;
    const km = haversineKm(startLocation, firstCoords);
    const walkingSpeedKmh = 4.5;
    const minutes = Math.round((km / walkingSpeedKmh) * 60);
    return { minutes, firstName };
  })();

  const handleOpenMaps = () => {
    const stops = itinerary.stops;
    if (stops.length === 0) return;

    const waypoints = stops
      .map((s) => {
        const coords = (s as any).location?.coordinates || (s as any).poi?.location?.coordinates;
        if (!coords || coords.length < 2) return null;
        return `${coords[1]},${coords[0]}`;
      })
      .filter(Boolean)
      .join("/");

    const url = `https://www.google.com/maps/dir/${waypoints}`;
    window.open(url, "_blank");
  };

  return (
    <div className="flex flex-col h-full">
      {/* Summary */}
      <div className="p-4 border-b border-border bg-primary/5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium text-foreground">{t("planner.itinerary.title")}</h3>
            <p className="text-sm text-muted-foreground">
              {t("planner.itinerary.stops", { count: itinerary.stops.length })}
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-sm text-foreground">
              <Clock className="w-4 h-4" />
              {formatDuration(totalMinutes)}
            </div>
            {typeof totalDistanceKm === "number" && !isNaN(totalDistanceKm) && (
              <p className="text-xs text-muted-foreground">
                {totalDistanceKm.toFixed(1)} km
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-[15px] top-8 bottom-8 w-0.5 bg-border" />

            {/* Stops */}
            <div className="space-y-4">
              {startLocation && startToFirst && (
                <div className="relative flex gap-4">
                  <div className="relative z-10 flex flex-col items-center">
                    <div className="w-8 h-8 rounded-full bg-emerald-700 text-white flex items-center justify-center text-sm font-medium">
                      <Flag className="w-4 h-4" />
                    </div>
                    <div className="mt-2 text-xs text-muted-foreground text-center">
                      {formatDuration(startToFirst.minutes)}
                    </div>
                  </div>
                  <div className="flex-1 pb-4">
                    <div className="p-4 rounded-xl bg-card border border-border">
                      <div className="flex items-center gap-2 mb-2 text-sm text-primary font-medium">
                        <MapPin className="w-3.5 h-3.5" />
                        {t("planner.itinerary.start_title")}
                      </div>
                      <p className="text-sm text-foreground mb-1">{t("planner.itinerary.start_location")}</p>
                      <p className="text-xs text-muted-foreground">
                        {t("planner.itinerary.start_to", { name: startToFirst.firstName })}
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {itinerary.stops.map((stop, index) => {
                const place = places.find((p) => p.id === stop.id) || ("poi" in stop ? (stop as any).poi : undefined);
                const isLast = index === itinerary.stops.length - 1;
                const nextLeg = legs && legs[index + 1] ? legs[index + 1] : undefined;
                const inboundForNextStop = itinerary.stops[index + 1] as any;
                const travelToNextMinutes = !isLast
                  ? (nextLeg?.duration_s !== undefined
                      ? Math.round((nextLeg.duration_s || 0) / 60)
                      : inboundForNextStop?.travelMinutes ?? inboundForNextStop?.travel_minutes)
                  : undefined;

                // Normalize fields to tolerate backend shapes that wrap POI data under stop.poi
                const categories = Array.isArray((stop as any).category)
                  ? (stop as any).category
                  : Array.isArray((stop as any).poi?.category)
                    ? (stop as any).poi.category
                    : [];
                const name = (stop as any).name || (stop as any).poi?.name || t("planner.itinerary.unnamed");
                const arrival = (stop as any).arrival_time || "--";
                const departure = (stop as any).departure_time || "--";
                const durationMinutes = (stop as any).duration_minutes || (stop as any).visitMinutes;

                return (
                  <div key={stop.id || index} className="relative flex gap-4">
                    {/* Timeline Node */}
                    <div className="relative z-10 flex flex-col items-center">
                      <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                        {stop.order}
                      </div>
                      {!isLast && travelToNextMinutes !== undefined && (
                        <div className="mt-2 text-xs text-muted-foreground text-center">
                          {formatDuration(travelToNextMinutes)}
                        </div>
                      )}
                    </div>

                    {/* Content Card */}
                    <div className="flex-1 pb-4">
                      <div className="p-4 rounded-xl bg-card border border-border">
                        {/* Time */}
                        <div className="flex items-center gap-2 mb-2 text-sm text-primary font-medium">
                          <Clock className="w-3.5 h-3.5" />
                          {arrival} - {departure}
                        </div>

                        {/* Name */}
                        <h4 className="font-medium text-foreground mb-1">
                          {name}
                        </h4>

                        {/* Categories */}
                        <div className="flex flex-wrap gap-1 mb-2">
                          {categories.slice(0, 2).map((cat: string) => (
                            <PoiCategoryBadge key={cat} category={cat} />
                          ))}
                        </div>

                        {/* Duration */}
                        {durationMinutes && (
                          <p className="text-xs text-muted-foreground">
                            {t("planner.itinerary.visit_time", { duration: formatDuration(durationMinutes) })}
                          </p>
                        )}

                        {/* Description */}
                        {place?.description && (
                          <p className="text-sm text-muted-foreground mt-2 line-clamp-2">
                            {place.description}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </ScrollArea>

      {/* Open in Maps */}
      <div className="p-4 border-t border-border">
        <Button
          variant="outline"
          onClick={handleOpenMaps}
          className="w-full py-6 rounded-full bg-transparent"
        >
          <Navigation className="w-4 h-4 mr-2" />
          {t("planner.itinerary.open_maps")}
        </Button>
      </div>
    </div>
  );
}
