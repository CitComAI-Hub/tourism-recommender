"use client";

import type React from "react";
import { Button } from "@/components/ui/button";
import { Check } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

interface LanguageToggleProps {
  className?: string;
  variant?: "ghost" | "outline";
  size?: "icon" | "sm" | "default";
}

export function LanguageToggle({
  className,
  variant = "outline",
  size = "icon",
}: LanguageToggleProps) {
  const { locale, setLocale, availableLocales } = useI18n();

  const handleToggle = () => {
    if (availableLocales.length < 2) return;
    const currentIndex = availableLocales.indexOf(locale);
    const next = availableLocales[(currentIndex + 1) % availableLocales.length];
    setLocale(next);
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleToggle}
      className={cn("rounded-full font-semibold", className)}
      aria-label="Toggle language"
    >
      {locale.toUpperCase()}
    </Button>
  );
}

function SpainFlag({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 640 480"
      className={className}
      role="img"
      aria-label="Bandera de Espana"
    >
      <rect width="640" height="480" fill="#AA151B" />
      <rect width="640" height="240" y="120" fill="#F1BF00" />
    </svg>
  );
}

function USFlag({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 640 480"
      className={className}
      role="img"
      aria-label="US Flag"
    >
      <rect width="640" height="480" fill="#BD3D44" />
      <rect y="37" width="640" height="37" fill="#fff" />
      <rect y="111" width="640" height="37" fill="#fff" />
      <rect y="185" width="640" height="37" fill="#fff" />
      <rect y="259" width="640" height="37" fill="#fff" />
      <rect y="333" width="640" height="37" fill="#fff" />
      <rect y="407" width="640" height="37" fill="#fff" />
      <rect width="260" height="259" fill="#192F5D" />
    </svg>
  );
}

const FLAGS: Record<string, (props: { className?: string }) => React.ReactNode> = {
  es: SpainFlag,
  en: USFlag,
};

const LOCALE_LABELS: Record<string, { label: string; nativeName: string }> = {
  es: { label: "Spanish", nativeName: "Espanol" },
  en: { label: "English", nativeName: "English" },
};

export function LanguageToggleInline({ className }: { className?: string }) {
  const { locale, setLocale, availableLocales } = useI18n();

  return (
    <div
      className={cn("flex flex-col gap-1 px-1 py-1", className)}
      role="radiogroup"
      aria-label="Seleccionar idioma"
    >
      {availableLocales.map((loc) => {
        const isActive = loc === locale;
        const info = LOCALE_LABELS[loc];
        const FlagIcon = FLAGS[loc];
        return (
          <button
            key={loc}
            type="button"
            role="radio"
            aria-checked={isActive}
            onClick={() => setLocale(loc)}
            className={cn(
              "flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-sm transition-colors",
              isActive
                ? "text-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            {FlagIcon && (
              <span className="inline-flex h-3.5 w-5 shrink-0 overflow-hidden rounded-[2px]">
                <FlagIcon className="h-full w-full" />
              </span>
            )}
            <span className="flex-1 text-left text-sm">
              {info?.nativeName ?? loc}
            </span>
            {isActive && (
              <Check className="h-3.5 w-3.5 shrink-0 text-foreground" />
            )}
          </button>
        );
      })}
    </div>
  );
}

export function LanguageToggleCompact({ className }: { className?: string }) {
  const { locale, setLocale, availableLocales } = useI18n();

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border/60 bg-muted/40 p-1",
        className
      )}
      role="radiogroup"
      aria-label="Seleccionar idioma"
    >
      {availableLocales.map((loc) => {
        const isActive = loc === locale;
        const info = LOCALE_LABELS[loc];
        const FlagIcon = FLAGS[loc];
        return (
          <button
            key={loc}
            type="button"
            role="radio"
            aria-checked={isActive}
            aria-label={info?.nativeName ?? loc}
            onClick={() => setLocale(loc)}
            className={cn(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-all",
              isActive
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {FlagIcon && (
              <span className="inline-flex h-3 w-4 shrink-0 overflow-hidden rounded-[2px]">
                <FlagIcon className="h-full w-full" />
              </span>
            )}
            <span>{loc.toUpperCase()}</span>
          </button>
        );
      })}
    </div>
  );
}
