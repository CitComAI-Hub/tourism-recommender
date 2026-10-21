"use client";

import { Button } from "@/components/ui/button";
import { MapPin, Calendar, User, LogOut } from "lucide-react";
import { useUser } from "@/lib/user-context";
import { useRouter } from "next/navigation";
import { useI18n } from "@/lib/i18n";
import { LanguageToggleInline } from "@/components/language-toggle";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface AppHeaderProps {
  acceptedCount: number;
  onPlanClick: () => void;
}

export function AppHeader({ acceptedCount, onPlanClick }: AppHeaderProps) {
  const { profile, resetUser } = useUser();
  const router = useRouter();
  const { t } = useI18n();

  const handleLogout = () => {
    resetUser();
    router.replace("/onboarding");
  };

  return (
    <header className="sticky top-0 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border">
      <div className="flex items-center justify-between h-16 px-4 lg:px-6">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
            <MapPin className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-serif text-xl text-foreground hidden sm:block">
            {t("common.brand")}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          {/* Plan Button */}
          <Button
            onClick={onPlanClick}
            variant={acceptedCount > 0 ? "default" : "outline"}
            className="rounded-full"
          >
            <Calendar className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">{t("header.plan_button")}</span>
            {acceptedCount > 0 && (
              <span className="ml-2 bg-primary-foreground text-primary w-5 h-5 rounded-full text-xs flex items-center justify-center font-medium">
                {acceptedCount}
              </span>
            )}
          </Button>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <User className="w-5 h-5" />
                <span className="sr-only">{t("header.user_menu")}</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <div className="px-3 py-2">
                <p className="text-sm font-medium">{profile?.name || t("common.user")}</p>
                <p className="text-xs text-muted-foreground">
                  {t("header.interests", { count: profile?.preferred_categories.length || 0 })}
                </p>
              </div>
              <DropdownMenuSeparator />
              <div className="px-2 py-2">
                <p className="px-1 pb-1 text-xs font-medium text-muted-foreground">
                  {t("header.language")}
                </p>
                <LanguageToggleInline />
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout} className="text-destructive">
                <LogOut className="w-4 h-4 mr-2" />
                {t("header.logout")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
