import type { LucideIcon } from "lucide-react";
import {
  TreePine,
  LandPlot,
  Flower2,
  Fish,
  Info,
  ShoppingBag,
  Palette,
  Leaf,
  PawPrint,
  Landmark,
  Church,
  Castle,
  Waves,
} from "lucide-react";

export const POI_CATEGORY_IDS = [
  "museum",
  "park",
  "attraction",
  "artwork",
  "place_of_worship",
  "nature_reserve",
  "garden",
  "marketplace",
  "information",
  "aquarium",
  "zoo",
  "arts_centre",
  "historic",
  "beach",
] as const;

export type PoiCategoryId = (typeof POI_CATEGORY_IDS)[number];

export const POI_CATEGORY_META: Array<{ id: PoiCategoryId; icon: LucideIcon }> = [
  { id: "museum", icon: Landmark },
  { id: "park", icon: TreePine },
  { id: "attraction", icon: LandPlot },
  { id: "artwork", icon: Palette },
  { id: "place_of_worship", icon: Church },
  { id: "nature_reserve", icon: Leaf },
  { id: "garden", icon: Flower2 },
  { id: "marketplace", icon: ShoppingBag },
  { id: "information", icon: Info },
  { id: "aquarium", icon: Fish },
  { id: "zoo", icon: PawPrint },
  { id: "arts_centre", icon: Palette },
  { id: "historic", icon: Castle },
  { id: "beach", icon: Waves },
];

export const POI_CATEGORY_FILTERS = [
  { id: null, key: "recommendations.filters.all" },
  ...POI_CATEGORY_IDS.map((id) => ({ id, key: `recommendations.filters.${id}` })),
] as const;

export function normalizePoiCategory(value?: string | null): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

export function isValidPoiCategoryId(value?: string | null): value is PoiCategoryId {
  const normalized = normalizePoiCategory(value);
  return normalized !== null && POI_CATEGORY_IDS.includes(normalized as PoiCategoryId);
}

export function sanitizePoiCategoryIds(values?: string[] | null): PoiCategoryId[] {
  const sanitized = new Set<PoiCategoryId>();

  for (const value of values ?? []) {
    const normalized = normalizePoiCategory(value);
    if (normalized && isValidPoiCategoryId(normalized)) {
      sanitized.add(normalized);
    }
  }

  return [...sanitized];
}

export function matchesPoiCategory(
  poiCategories?: string[] | null,
  targetFilter?: string | null
): boolean {
  if (!targetFilter) return true;

  const normalizedTarget = normalizePoiCategory(targetFilter);
  if (!normalizedTarget) return false;

  const categories = Array.isArray(poiCategories) ? poiCategories : [];
  return categories.some((category) => normalizePoiCategory(category) === normalizedTarget);
}
