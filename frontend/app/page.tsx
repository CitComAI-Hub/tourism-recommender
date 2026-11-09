"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/lib/user-context";
import { useI18n } from "@/lib/i18n";
import { Loader2 } from "lucide-react";

export default function HomePage() {
  const router = useRouter();
  const { onboardingComplete, isHydrated } = useUser();
  const { t } = useI18n();

  useEffect(() => {
    // Small delay to ensure state is hydrated from localStorage
    const timer = setTimeout(() => {
      if (!isHydrated) return;
      if (onboardingComplete) {
        router.replace("/explore");
      } else {
        router.replace("/onboarding");
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [onboardingComplete, isHydrated, router]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">{t("home.loading")}</p>
      </div>
    </div>
  );
}
