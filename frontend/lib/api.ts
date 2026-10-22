const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface Location {
  lat: number;
  lon: number;
}

export interface POILocation {
  coordinates: [number, number]; // [lon, lat]
}

export interface POI {
  id: string;
  name: string;
  description?: string;
  category: string[];
  location: POILocation;
  image?: string;
  rating?: number;
  duration?: number;
  openingHours?: string | { value: string }; // string per Smart Data Models
  opening_hours?: string;
  relevance?: number;
}

export interface UserProfile {
  name: string;
  preferred_categories: string[];
  recommendation_style?: "explorer" | "conservative" | "hidden_gems";
  interests?: string;
}

export interface UserPreferences {
  user_id: string;
  profile: UserProfile;
}

export interface PlannerPreferences {
  max_stops_per_day: number;
  max_total_minutes: number;
  visit_time_minutes: number;
  start_location?: Location | null;
  use_current_location?: boolean;
  mobility?: "walking" | "transit" | "car";
}

export interface OnboardingSelections {
  user_id: string;
  profile: UserProfile;
  liked_pois: string[];
  disliked_pois: string[];
  start_location: Location | null;
  use_current_location: boolean;
  planner_preferences: PlannerPreferences;
}

export interface RecommendationRequest {
  user_id?: string;
  profile?: UserProfile;
  top_n?: number;
  liked_pois?: string[];
  disliked_pois?: string[];
  user_location?: Location;
  user_mobility?: "walking" | "transit" | "car";
  requested_at?: string;
  category?: string;
  limit?: number;
}

export interface ItineraryCandidate {
  id: string;
  location: POILocation;
  category: string[];
  name?: string;
}

export interface ItineraryRequest {
  user_id?: string;
  candidates?: ItineraryCandidate[];
  max_stops_per_day?: number;
  max_total_minutes?: number;
  visit_time_minutes?: number;
  max_candidates?: number;
  start_location?: Location;
}

export interface ItineraryStop {
  id: string;
  name: string;
  location: POILocation;
  category: string[];
  arrival_time?: string;
  departure_time?: string;
  duration_minutes?: number;
  order: number;
}

export interface Itinerary {
  stops: ItineraryStop[];
  total_duration_minutes: number;
  total_distance_km?: number;
}

// API Functions

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function getPOIs(params?: { limit?: number; category?: string }): Promise<POI[]> {
  const query = new URLSearchParams();
  if (params?.limit) query.set("limit", params.limit.toString());
  if (params?.category) query.set("category", params.category);

  const qs = query.toString();
  const response = await fetch(`${API_BASE_URL}/poi${qs ? `?${qs}` : ""}`);
  if (!response.ok) throw new Error("Failed to fetch POIs");
  return response.json();
}

export async function getRandomPOIs(params: {
  categories?: string[];
  limit?: number;
  likedPOIs?: string[];
  dislikedPOIs?: string[];
  seenPOIs?: string[];
  interests?: string;
}): Promise<POI[]> {
  const query = new URLSearchParams();
  if (params.categories?.length) {
    query.set("categories", params.categories.join(","));
  }
  if (params.limit) {
    query.set("limit", params.limit.toString());
  }
  if (params.likedPOIs?.length) {
    query.set("liked_pois", params.likedPOIs.join(","));
  }
  if (params.dislikedPOIs?.length) {
    query.set("disliked_pois", params.dislikedPOIs.join(","));
  }
  if (params.seenPOIs?.length) {
    query.set("seen_pois", params.seenPOIs.join(","));
  }
  if (params.interests?.trim()) {
    query.set("interests", params.interests.trim());
  }
  const qs = query.toString();
  const response = await fetch(`${API_BASE_URL}/poi/sample${qs ? `?${qs}` : ""}`);
  if (!response.ok) throw new Error("Failed to fetch sample POIs");
  return response.json();
}

export async function saveUserPreferences(
  preferences: UserPreferences
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/user/preferences`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preferences),
  });
  if (!response.ok) throw new Error("Failed to save preferences");
}

export async function getUserPreferences(
  userId: string
): Promise<UserPreferences | null> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/user/preferences?user_id=${userId}`
    );
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

export async function getRecommendations(
  request: RecommendationRequest
): Promise<POI[]> {
  const response = await fetch(`${API_BASE_URL}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error("Failed to get recommendations");
  const data = await response.json();
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.results)) return data.results;
  return [];
}

export async function getUserRecommendations(userId: string): Promise<POI[]> {
  const response = await fetch(
    `${API_BASE_URL}/recommendations?user_id=${userId}`
  );
  if (!response.ok) throw new Error("Failed to get user recommendations");
  const data = await response.json();
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.results)) return data.results;
  return [];
}

export async function generateItinerary(
  request: ItineraryRequest
): Promise<Itinerary> {
  const response = await fetch(`${API_BASE_URL}/itinerary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error("Failed to generate itinerary");
  return response.json();
}

export async function getUserItinerary(userId: string): Promise<Itinerary> {
  const response = await fetch(`${API_BASE_URL}/itinerary?user_id=${userId}`);
  if (!response.ok) throw new Error("Failed to get user itinerary");
  return response.json();
}

export async function saveOnboardingSelections(payload: OnboardingSelections): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/user/onboarding`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("Failed to store onboarding data");
}

// Fetch OSRM-like polyline/legs from backend (extension point; not used yet)
export interface RouteLeg {
  distance_m?: number;
  duration_s?: number;
  geometry?: string;
}

export interface ItineraryWithRoute extends Itinerary {
  legs?: RouteLeg[];
  polyline?: string;
}

// Available categories for preferences
export const AVAILABLE_CATEGORIES = [
  { id: "museum", label: "Museos", icon: "building-columns" },
  { id: "park", label: "Parques", icon: "trees" },
  { id: "historic", label: "Lugares Historicos", icon: "landmark" },
  { id: "beach", label: "Playas", icon: "umbrella-beach" },
  { id: "restaurant", label: "Restaurantes", icon: "utensils" },
  { id: "nightlife", label: "Vida Nocturna", icon: "moon" },
  { id: "shopping", label: "Compras", icon: "shopping-bag" },
  { id: "art", label: "Arte", icon: "palette" },
  { id: "nature", label: "Naturaleza", icon: "leaf" },
  { id: "architecture", label: "Arquitectura", icon: "building" },
] as const;

// Valencia center coordinates
export const VALENCIA_CENTER: Location = {
  lat: 39.4699,
  lon: -0.3763,
};
