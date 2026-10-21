"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { VALENCIA_CENTER, type Location } from "@/lib/api";

interface PlannerSettings {
  maxStopsPerDay: number;
  maxTotalMinutes: number;
  visitTimeMinutes: number;
  startLocation: Location;
}

interface PlannerSettingsProps {
  open: boolean;
  onClose: () => void;
  settings: PlannerSettings;
  onSettingsChange: (settings: PlannerSettings) => void;
}

export function PlannerSettings({
  open,
  onClose,
  settings,
  onSettingsChange,
}: PlannerSettingsProps) {
  const formatTime = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (mins === 0) return `${hours}h`;
    return `${hours}h ${mins}min`;
  };

  return (
    <Sheet open={open} onOpenChange={onClose}>
      <SheetContent side="right" className="w-full sm:max-w-md">
        <SheetHeader className="mb-6">
          <SheetTitle className="font-serif text-2xl">
            Configuración del plan
          </SheetTitle>
        </SheetHeader>

        <div className="space-y-8">
          {/* Max stops */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Paradas máximas por día</Label>
              <span className="text-sm font-medium text-primary">
                {settings.maxStopsPerDay}
              </span>
            </div>
            <Slider
              value={[settings.maxStopsPerDay]}
              onValueChange={([value]) =>
                onSettingsChange({ ...settings, maxStopsPerDay: value })
              }
              min={2}
              max={10}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Número máximo de lugares a visitar en un día
            </p>
          </div>

          {/* Total time */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Tiempo total disponible</Label>
              <span className="text-sm font-medium text-primary">
                {formatTime(settings.maxTotalMinutes)}
              </span>
            </div>
            <Slider
              value={[settings.maxTotalMinutes]}
              onValueChange={([value]) =>
                onSettingsChange({ ...settings, maxTotalMinutes: value })
              }
              min={180}
              max={720}
              step={30}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Tiempo máximo que quieres dedicar al recorrido
            </p>
          </div>

          {/* Visit time */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Tiempo por visita</Label>
              <span className="text-sm font-medium text-primary">
                {formatTime(settings.visitTimeMinutes)}
              </span>
            </div>
            <Slider
              value={[settings.visitTimeMinutes]}
              onValueChange={([value]) =>
                onSettingsChange({ ...settings, visitTimeMinutes: value })
              }
              min={30}
              max={180}
              step={15}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Tiempo promedio estimado en cada lugar
            </p>
          </div>

          {/* Start Location Info */}
          <div className="p-4 rounded-xl bg-secondary/50 border border-border">
            <Label className="mb-2 block">Punto de inicio</Label>
            <p className="text-sm text-muted-foreground">
              Centro de Valencia
              <br />
              <span className="text-xs">
                ({VALENCIA_CENTER.lat.toFixed(4)}, {VALENCIA_CENTER.lon.toFixed(4)})
              </span>
            </p>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
