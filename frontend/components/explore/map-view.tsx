"use client";

import { useEffect, useRef, useState } from "react";
import type { Location, POI } from "@/lib/api";
import { VALENCIA_CENTER } from "@/lib/api";
import { useI18n } from "@/lib/i18n";

interface MapViewProps {
  pois: POI[];
  acceptedPOIs: POI[];
  selectedPOI: POI | null;
  hoveredPOI: POI | null;
  onSelectPOI: (poi: POI) => void;
  isVisible: boolean;
  userLocation?: Location | null;
}

declare global {
  interface Window {
    L: typeof import("leaflet");
  }
}

export function MapView({
  pois,
  acceptedPOIs,
  selectedPOI,
  hoveredPOI,
  onSelectPOI,
  isVisible,
  userLocation,
}: MapViewProps) {
  const { t } = useI18n();
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<import("leaflet").Map | null>(null);
  const markersRef = useRef<Map<string, { marker: import("leaflet").Marker; minZoom: number }>>(new Map());
  const userMarkerRef = useRef<import("leaflet").Marker | null>(null);
  const invalidateTimeoutRef = useRef<number | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const safePois = Array.isArray(pois) ? pois : [];
  const poiColor = "#c77f32"; // matches default POI markers

  // Load Leaflet
  useEffect(() => {
    if (typeof window === "undefined") return;

    const loadLeaflet = async () => {
      // Add CSS
      if (!document.querySelector('link[href*="leaflet.css"]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

      // Add JS
      if (!window.L) {
        await new Promise<void>((resolve) => {
          const script = document.createElement("script");
          script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
          script.onload = () => resolve();
          document.head.appendChild(script);
        });
      }

      setIsLoaded(true);
    };

    loadLeaflet();
  }, []);

  // Initialize map
  useEffect(() => {
    if (!isLoaded || !mapRef.current || mapInstanceRef.current) return;

    const L = window.L;
    const map = L.map(mapRef.current).setView(
      [VALENCIA_CENTER.lat, VALENCIA_CENTER.lon],
      13
    );

    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> | <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    // Ensure initial sizing when mounted
    invalidateTimeoutRef.current = window.setTimeout(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    }, 100);
    // Invalidate size on resize (window + container)
    const handleResize = () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    };
    window.addEventListener("resize", handleResize);

    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => handleResize()) : null;
    if (ro && mapRef.current) ro.observe(mapRef.current);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (ro && mapRef.current) ro.unobserve(mapRef.current);
      if (invalidateTimeoutRef.current) {
        window.clearTimeout(invalidateTimeoutRef.current);
      }
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [isLoaded]);

  // Ensure map resizes correctly when it becomes visible
  useEffect(() => {
    if (!isLoaded || !isVisible || !mapInstanceRef.current) return;
    const timer = setTimeout(() => {
      mapInstanceRef.current?.invalidateSize();
    }, 250);

    return () => clearTimeout(timer);
  }, [isLoaded, isVisible]);

  // Update markers
  useEffect(() => {
    if (!isLoaded || !mapInstanceRef.current) return;

    const L = window.L;
    const map = mapInstanceRef.current;

    const zoomMin = Math.max(map.getMinZoom(), 8);
    const zoomMax = Math.min(map.getMaxZoom(), 18);

    // Spread visibility by rank so markers actually differ even if relevance values are close
    const rankedPois = [...safePois]
      .map((poi, idx) => ({ poi, idx }))
      .sort((a, b) => {
        const ra = Number.isFinite(a.poi.relevance) ? Number(a.poi.relevance) : 1;
        const rb = Number.isFinite(b.poi.relevance) ? Number(b.poi.relevance) : 1;
        return ra - rb;
      });
    const rankToZoom = new Map<string, number>();
    const denom = Math.max(1, rankedPois.length - 1);
    rankedPois.forEach(({ poi }, rank) => {
      const t = rank / denom; // 0 best, 1 worst by rank
      const minZoom = Math.round(zoomMin + t * (zoomMax - zoomMin));
      rankToZoom.set(poi.id, minZoom);
    });
    const computeMinZoom = (poi: POI) => rankToZoom.get(poi.id) ?? zoomMax;

    const applyVisibility = () => {
      const zoom = map.getZoom();
      markersRef.current.forEach(({ marker, minZoom }) => {
        const shouldShow = zoom >= minZoom;
        const isShown = map.hasLayer(marker);
        if (shouldShow && !isShown) {
          marker.addTo(map);
        } else if (!shouldShow && isShown) {
          marker.remove();
        }
      });
    };

    // Clear existing markers
    markersRef.current.forEach(({ marker }) => marker.remove());
    markersRef.current.clear();
    userMarkerRef.current?.remove();
    userMarkerRef.current = null;

    if (userLocation && Number.isFinite(userLocation.lat) && Number.isFinite(userLocation.lon)) {
      const userIcon = L.divIcon({
        className: "custom-marker",
        html: `
          <div style="
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #2d6a4f, #1b4d33);
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            transform: translate(-50%, -50%);
          ">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 10c0 5-8 12-8 12S4 15 4 10a8 8 0 1 1 16 0Z" />
              <circle cx="12" cy="10" r="2.5" />
            </svg>
          </div>
        `,
        iconSize: [36, 36],
        iconAnchor: [18, 18],
      });
      userMarkerRef.current = L.marker([userLocation.lat, userLocation.lon], { icon: userIcon })
        .addTo(map)
        .bindTooltip("Tu ubicación", { direction: "top", offset: [0, -14] });
    }

    // Add new markers
    safePois.forEach((poi) => {
      const coords = poi.location?.coordinates;
      if (!coords || coords.length < 2) {
        return;
      }
      const isAccepted = acceptedPOIs.some((p) => p.id === poi.id);
      const isSelected = selectedPOI?.id === poi.id;
      const isHovered = hoveredPOI?.id === poi.id;

      const markerColor = isAccepted ? "#2d6a4f" : poiColor;
      const markerSize = isSelected || isHovered ? 36 : 28;

      const icon = L.divIcon({
        className: "custom-marker",
        html: `
          <div style="
            width: ${markerSize}px;
            height: ${markerSize}px;
            background: ${markerColor};
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            transform: translate(-50%, -50%);
            ${isSelected ? "transform: translate(-50%, -50%) scale(1.2);" : ""}
          ">
            ${isAccepted ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3"><polyline points="20,6 9,17 4,12"></polyline></svg>` : ""}
          </div>
        `,
        iconSize: [markerSize, markerSize],
        iconAnchor: [markerSize / 2, markerSize / 2],
      });

      const marker = L.marker([coords[1], coords[0]], { icon }).on("click", () => onSelectPOI(poi));

      marker.bindTooltip(poi.name, {
        direction: "top",
        offset: [0, -14],
      });

      const minZoom = computeMinZoom(poi);
      markersRef.current.set(poi.id, { marker, minZoom });
    });

    applyVisibility();
    map.on("zoomend", applyVisibility);

    return () => {
      map.off("zoomend", applyVisibility);
    };
  }, [isLoaded, safePois, acceptedPOIs, selectedPOI, hoveredPOI, onSelectPOI, userLocation]);

  // Center on selected POI
  useEffect(() => {
    if (!selectedPOI || !mapInstanceRef.current) return;

    mapInstanceRef.current.setView(
      [selectedPOI.location.coordinates[1], selectedPOI.location.coordinates[0]],
      15,
      { animate: true }
    );
  }, [selectedPOI]);

  return (
    <div className="relative w-full h-full bg-muted">
      <div ref={mapRef} className="absolute inset-0" />
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-background/95 backdrop-blur rounded-lg p-3 shadow-lg border border-border">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full border-2 border-white shadow" style={{ background: poiColor }} />
            <span className="text-muted-foreground">{t("recommendations.legend.recommended")}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-primary border-2 border-white shadow" />
            <span className="text-muted-foreground">{t("recommendations.legend.in_plan")}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
