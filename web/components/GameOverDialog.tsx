"use client";

import { Handshake, RefreshCw, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { GameResult } from "@/lib/gameResult";

type GameOverDialogProps = {
  result: GameResult | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNewGame: () => void;
};

export default function GameOverDialog({ result, open, onOpenChange, onNewGame }: GameOverDialogProps) {
  if (!result) return null;
  const Icon = result.winner ? Trophy : Handshake;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader className="items-center text-center sm:text-center">
          <span className="mb-2 flex size-14 items-center justify-center rounded-full bg-primary/15 text-primary">
            <Icon className="size-7" />
          </span>
          <DialogTitle className="text-2xl">{result.title}</DialogTitle>
          <DialogDescription className="text-base">{result.detail}</DialogDescription>
        </DialogHeader>
        <DialogFooter className="grid grid-cols-2 gap-2 sm:grid-cols-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            View board
          </Button>
          <Button onClick={onNewGame}>
            <RefreshCw />
            New game
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
