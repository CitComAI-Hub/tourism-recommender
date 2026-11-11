// Same origin in production (the API container serves the app under BASE_PATH);
// set NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api for `pnpm dev`.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/api`;

export interface Location {
  lat: number;
  lon: number;
}

export interface POILocation {
  coordinates: [number, number]; // [lon, lat]
}

export type Reason =
  | { type: "popular" }
  | { type: "near"; km: number; poi: string | null }
  | { type: "similar"; poi: string }
  | { type: "next"; poi: string }
  | { type: "interest" };

export interface POI {
  id: string;
  name: string;
  description?: string;
  descriptionGenerated?: boolean;
  category: string[];
  location: POILocation;
  image?: string;
  duration?: number;
  openingHours?: string | { value: string }; // string per Smart Data Models
  // Set on recommendation results
  rank?: number;
  score?: number;
  distanceKm?: number | null;
  popularitySource?: "visits" | "predicted" | "wikipedia";
  reasons?: Reason[];
}

export interface UserProfile {
  name: string;
  preferred_categories: string[];
  /** What the user says they like, in their own words. */
  interests?: string;
}

export interface Feedback {
  liked: string[];
  disliked: string[];
  categories?: string[];
  interests?: string;
  /** Places already visited, oldest first. */
  visited?: string[];
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path} failed (${response.status})`);
  return response.json();
}

export async function getPOIs(params?: { limit?: number; category?: string }): Promise<POI[]> {
  const query = new URLSearchParams();
  if (params?.limit) query.set("limit", params.limit.toString());
  if (params?.category) query.set("category", params.category);
  const qs = query.toString();
  const response = await fetch(`${API_BASE_URL}/pois${qs ? `?${qs}` : ""}`);
  if (!response.ok) throw new Error("Failed to fetch POIs");
  return response.json();
}

/** Next onboarding cards: the best-known places the user has not answered yet. */
export function getOnboardingCards(params: Feedback & { seen: string[]; limit: number }): Promise<POI[]> {
  return postJSON<POI[]>("/onboarding/cards", params);
}

export async function getRecommendations(
  request: Feedback & { location?: Location | null; top_n?: number; at?: string }
): Promise<POI[]> {
  const data = await postJSON<{ results: POI[] }>("/recommendations", {
    ...request,
    location: request.location ?? undefined,
  });
  return data.results ?? [];
}

/** Free-text search over the descriptions; [] when the text model is not available. */
export async function searchPOIs(q: string, limit = 20): Promise<POI[]> {
  const response = await fetch(`${API_BASE_URL}/search?${new URLSearchParams({ q, limit: String(limit) })}`);
  if (response.status === 503) return [];
  if (!response.ok) throw new Error("Search failed");
  return response.json();
}

// Valencia center coordinates
export const VALENCIA_CENTER: Location = {
  lat: 39.4699,
  lon: -0.3763,
};
