"use client";

import { useSyncExternalStore } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { isMuted, setMuted, subscribeMuted } from "@/lib/sounds";

export default function SoundToggle() {
  // Server snapshot is "not muted"; the stored preference applies after hydration.
  const muted = useSyncExternalStore(subscribeMuted, isMuted, () => false);

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={muted ? "Unmute sounds" : "Mute sounds"}
      aria-pressed={muted}
      onClick={() => setMuted(!muted)}
    >
      {muted ? <VolumeX /> : <Volume2 />}
    </Button>
  );
}
