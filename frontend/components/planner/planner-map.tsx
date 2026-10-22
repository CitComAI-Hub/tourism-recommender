"use client";

import { useEffect, useRef, useState } from "react";
import type { POI, Itinerary, Location } from "@/lib/api";
import { TILE_URL, TILE_ATTRIBUTION } from "@/lib/map-tiles";

interface PlannerMapProps {
  places: POI[];
  itinerary: Itinerary | null;
  startLocation: Location;
}

declare global {
  interface Window {
    L: typeof import("leaflet");
  }
}

export function PlannerMap({ places, itinerary, startLocation }: PlannerMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<import("leaflet").Map | null>(null);
  const markersRef = useRef<import("leaflet").Marker[]>([]);
  const polylineRef = useRef<import("leaflet").Polyline | null>(null);
  const invalidateTimeoutRef = useRef<number | null>(null);
  const routeAbortRef = useRef<AbortController | null>(null);
  const routeCacheRef = useRef<Map<string, Array<[number, number]>>>(new Map());
  const [isLoaded, setIsLoaded] = useState(false);
  const [routePoints, setRoutePoints] = useState<Array<[number, number]> | null>(null);
  const [routeStatus, setRouteStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  const toLatLon = (coordinates?: [number, number]) => {
    if (!coordinates || coordinates.length < 2) return null;
    const [lon, lat] = coordinates;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    return { lat, lon };
  };

  const normalizeStart = () => {
    if (startLocation && Number.isFinite(startLocation.lat) && Number.isFinite(startLocation.lon)) {
      return startLocation;
    }
    return null;
  };

  const decodePolyline6 = (encoded: string): Array<[number, number]> => {
    let index = 0;
    const len = encoded.length;
    let lat = 0;
    let lon = 0;
    const coords: Array<[number, number]> = [];

    while (index < len) {
      let b;
      let shift = 0;
      let result = 0;
      do {
        b = encoded.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const deltaLat = (result & 1) ? ~(result >> 1) : result >> 1;
      lat += deltaLat;

      shift = 0;
      result = 0;
      do {
        b = encoded.charCodeAt(index++) - 63;
        result |= (b & 0x1f) << shift;
        shift += 5;
      } while (b >= 0x20);
      const deltaLon = (result & 1) ? ~(result >> 1) : result >> 1;
      lon += deltaLon;

      coords.push([lat / 1e6, lon / 1e6]);
    }

    return coords;
  };

  useEffect(() => {
    const start = normalizeStart();
    if (!start || !itinerary || !itinerary.stops?.length) {
      setRoutePoints(null);
      setRouteStatus("idle");
      routeAbortRef.current?.abort();
      return;
    }

    // If backend already provides a polyline (OSRM), use it directly to avoid refetch
    if (typeof (itinerary as any).polyline === "string") {
      try {
        const decoded = decodePolyline6((itinerary as any).polyline as string);
        setRoutePoints(decoded);
        setRouteStatus("ready");
        routeAbortRef.current?.abort();
        return;
      } catch (err) {
        console.error("Failed to decode backend polyline", err);
        // fallback to live fetch
      }
    }

    const coords: Array<[number, number]> = [];
    coords.push([start.lat, start.lon]);
    itinerary.stops.forEach((stop) => {
      const c = stop.location?.coordinates;
      if (c && c.length >= 2 && Number.isFinite(c[0]) && Number.isFinite(c[1])) {
        coords.push([c[1], c[0]]);
      }
    });

    if (coords.length < 2) {
      setRoutePoints(null);
      setRouteStatus("idle");
      return;
    }

    const coordQuery = coords
      .map(([lat, lon]) => `${lon},${lat}`)
      .join(";");

    const cacheKey = coordQuery;
    if (routeCacheRef.current.has(cacheKey)) {
      setRoutePoints(routeCacheRef.current.get(cacheKey) || null);
      setRouteStatus("ready");
      return;
    }

    const fetchRoute = async () => {
      routeAbortRef.current?.abort();
      const controller = new AbortController();
      routeAbortRef.current = controller;
      setRouteStatus("loading");
      setRoutePoints(null);
      try {
        const res = await fetch(
          `https://router.project-osrm.org/route/v1/foot/${coordQuery}?overview=full&geometries=polyline6`,
          { signal: controller.signal }
        );
        if (!res.ok) throw new Error("route fetch failed");
        const data = await res.json();
        const geometry = data?.routes?.[0]?.geometry;
        if (typeof geometry === "string") {
          const decoded = decodePolyline6(geometry);
          routeCacheRef.current.set(cacheKey, decoded);
          setRoutePoints(decoded);
          setRouteStatus("ready");
        } else {
          setRoutePoints(null);
          setRouteStatus("error");
        }
      } catch (err) {
        if ((err as any)?.name === "AbortError") return;
        console.error("Route fetch error", err);
        setRoutePoints(null);
        setRouteStatus("error");
      }
    };

    fetchRoute();

    return () => {
      routeAbortRef.current?.abort();
    };
  }, [itinerary, startLocation]);

  // Load Leaflet
  useEffect(() => {
    if (typeof window === "undefined") return;

    const loadLeaflet = async () => {
      if (!document.querySelector('link[href*="leaflet.css"]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(link);
      }

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
    if (typeof window === "undefined" || !window.L) return;

    const L = window.L;
    const start = normalizeStart();
    const mapCenter: [number, number] = start ? [start.lat, start.lon] : [0, 0];

    const map = L.map(mapRef.current).setView(mapCenter, start ? 13 : 2);

    L.tileLayer(TILE_URL, {
      attribution: TILE_ATTRIBUTION,
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    // Fix size after first render
    invalidateTimeoutRef.current = window.setTimeout(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    }, 100);

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

  // Update markers and route
  useEffect(() => {
    if (!isLoaded || !mapInstanceRef.current) return;
    if (typeof window === "undefined" || !window.L) return;

    const L = window.L;
    const map = mapInstanceRef.current;

    const start = normalizeStart();
    if (!start || !Number.isFinite(start.lat) || !Number.isFinite(start.lon)) return;

    // Clear existing markers and polyline
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];
    if (polylineRef.current) {
      polylineRef.current.remove();
      polylineRef.current = null;
    }

    const pointsToShow = itinerary?.stops || places;
    if (pointsToShow.length === 0) return;

    // Add start location marker
    const startIcon = L.divIcon({
      className: "custom-marker",
      html: `
        <div style="
          width: 36px;
          height: 36px;
          background: linear-gradient(135deg, #2d6a4f, #1b4d33);
          border: 3px solid white;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.25);
          display: flex;
          align-items: center;
          justify-content: center;
          transform: translate(-50%, -50%);
        ">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 3v18" />
            <path d="M6 4l10 3-10 3" />
          </svg>
        </div>
      `,
      iconSize: [36, 36],
      iconAnchor: [18, 18],
    });

    const startMarker = L.marker([start.lat, start.lon], {
      icon: startIcon,
    }).addTo(map);
    startMarker.bindTooltip("Punto de inicio", { direction: "top", offset: [0, -14] });
    markersRef.current.push(startMarker);

    // Add numbered markers for each stop
    const coordinates: [number, number][] = [[start.lat, start.lon]];

    pointsToShow.forEach((point, index) => {
      const order = "order" in point ? point.order : index + 1;
      const name = (point as any).name || (point as any).poi?.name || "Sin nombre";
      const coordsRaw = (point as any).location?.coordinates || (point as any).poi?.location?.coordinates;
      const coords = toLatLon(coordsRaw);
      if (!coords) return;

      coordinates.push([coords.lat, coords.lon]);

      const icon = L.divIcon({
        className: "custom-marker",
        html: `
          <div style="
            width: 32px;
            height: 32px;
            background: ${itinerary ? "#2d6a4f" : "#c77f32"};
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 12px;
            transform: translate(-50%, -50%);
          ">
            ${order}
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const marker = L.marker([coords.lat, coords.lon], { icon })
        .addTo(map)
        .bindTooltip(name, { direction: "top", offset: [0, -14] });

      markersRef.current.push(marker);
    });

    // Draw route line (prefer routed polyline if available)
    const routedCoords =
      routeStatus === "ready" && routePoints?.length
        ? (routePoints.map(([lat, lon]) => [lat, lon]) as Array<[number, number]>)
        : null;

    if (routedCoords && routedCoords.length > 1) {
      polylineRef.current = L.polyline(routedCoords, {
        color: "#2d6a4f",
        weight: 5,
        opacity: 0.85,
      }).addTo(map);
    }

    // Fit bounds
    const boundsCoords = routedCoords && routedCoords.length > 1 ? routedCoords : coordinates;
    if (boundsCoords.length > 0) {
      const bounds = L.latLngBounds(boundsCoords as [number, number][]);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [isLoaded, places, itinerary, startLocation, routePoints, routeStatus]);

  return (
    <div className="relative w-full h-full bg-muted z-0">
      <div ref={mapRef} className="absolute inset-0 z-0" />

      {itinerary && (
        <div className="absolute top-4 left-4 flex flex-col gap-2 max-w-xs">
          {routeStatus === "error" && (
            <div className="bg-destructive/90 text-destructive-foreground rounded-lg p-3 shadow-lg border border-destructive">
              <div className="text-xs font-medium">No se pudo cargar la ruta</div>
              <div className="text-[11px] opacity-80">Reintenta o ajusta el punto de inicio.</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
