"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { messages, DEFAULT_LOCALE, SUPPORTED_LOCALES } from "@/locales";

type Locale = (typeof SUPPORTED_LOCALES)[number];

type Values = Record<string, string | number | boolean>;

interface I18nContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, values?: Values) => string;
  availableLocales: readonly Locale[];
}

const I18nContext = createContext<I18nContextType | null>(null);

const STORAGE_KEY = "valencia_locale";

function resolveMessage(dict: any, key: string): string | undefined {
  const parts = key.split(".");
  let current: any = dict;
  for (const part of parts) {
    if (current && typeof current === "object" && part in current) {
      current = current[part];
    } else {
      return undefined;
    }
  }
  return typeof current === "string" ? current : undefined;
}

function formatMessage(message: string, values?: Values) {
  if (!values) return message;
  return Object.keys(values).reduce((acc, key) => {
    const value = values[key];
    return acc.replaceAll(`{${key}}`, String(value));
  }, message);
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(DEFAULT_LOCALE);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as Locale | null;
    if (stored && SUPPORTED_LOCALES.includes(stored)) {
      setLocaleState(stored);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, locale);
    document.documentElement.lang = locale;
    document.documentElement.dataset.locale = locale;
  }, [locale]);

  const setLocale = (next: Locale) => {
    if (SUPPORTED_LOCALES.includes(next)) {
      setLocaleState(next);
    }
  };

  const t = (key: string, values?: Values) => {
    const bundle = messages[locale] || messages[DEFAULT_LOCALE];
    const fallback = messages[DEFAULT_LOCALE];
    const fromBundle = resolveMessage(bundle, key);
    const fromFallback = resolveMessage(fallback, key);
    const message = fromBundle ?? fromFallback ?? key;
    return formatMessage(message, values);
  };

  const value = useMemo(
    () => ({ locale, setLocale, t, availableLocales: SUPPORTED_LOCALES }),
    [locale]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}
