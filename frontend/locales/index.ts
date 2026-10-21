export const DEFAULT_LOCALE = "es" as const;
export const SUPPORTED_LOCALES = ["es", "en"] as const;

import { es } from "./es";
import { en } from "./en";

export const messages = {
  es,
  en,
};
