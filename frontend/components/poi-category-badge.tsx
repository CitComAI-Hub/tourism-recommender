"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useI18n } from "@/lib/i18n";
import { POI_CATEGORY_META, isValidPoiCategoryId, normalizePoiCategory } from "@/lib/poi-categories";

interface PoiCategoryBadgeProps {
  category: string;
  className?: string;
}

/** Translated label for a category id; unknown ids are humanised ("foo_bar" -> "Foo bar"). */
export function poiCategoryLabel(t: (key: string) => string, category: string): string {
  const id = normalizePoiCategory(category);
  if (isValidPoiCategoryId(id)) return t(`onboarding.categories.labels.${id}`);
  const raw = (id ?? category).replace(/_/g, " ");
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

/** Category as icon + translated label instead of the raw id ("place_of_worship"). */
export function PoiCategoryBadge({ category, className }: PoiCategoryBadgeProps) {
  const { t } = useI18n();
  const id = normalizePoiCategory(category);
  const Icon = POI_CATEGORY_META.find((meta) => meta.id === id)?.icon;
  const label = poiCategoryLabel(t, category);

  return (
    <Badge variant="secondary" className={cn("gap-1 text-xs px-2 py-0", className)}>
      {Icon && <Icon className="h-3 w-3" aria-hidden="true" />}
      {label}
    </Badge>
  );
}
