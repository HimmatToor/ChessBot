"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type MoveListProps = {
  /** Moves in SAN, in order played (white, black, white, ...). */
  moves: string[];
  /** Sizing from the parent (e.g. fixed height on phones, fill the panel on desktop). */
  className?: string;
};

export default function MoveList({ moves, className }: MoveListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the latest move in view as the game grows.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [moves.length]);

  if (moves.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground",
          className,
        )}
      >
        No moves yet
      </div>
    );
  }

  const rows: [string, string | undefined][] = [];
  for (let i = 0; i < moves.length; i += 2) rows.push([moves[i], moves[i + 1]]);
  const lastIndex = moves.length - 1;

  return (
    <div
      ref={scrollRef}
      className={cn("overflow-y-auto rounded-lg border bg-background/40 font-mono text-sm", className)}
    >
      <ol>
        {rows.map(([white, black], row) => (
          <li
            key={row}
            className="grid grid-cols-[2.75rem_1fr_1fr] items-center odd:bg-muted/40"
          >
            <span className="px-2 py-1 text-right text-muted-foreground">{row + 1}.</span>
            <MoveCell san={white} latest={row * 2 === lastIndex} />
            {black && <MoveCell san={black} latest={row * 2 + 1 === lastIndex} />}
          </li>
        ))}
      </ol>
    </div>
  );
}

function MoveCell({ san, latest }: { san: string; latest: boolean }) {
  return (
    <span
      className={cn(
        "mx-1 rounded px-2 py-1",
        latest && "bg-primary/20 font-semibold text-primary",
      )}
    >
      {san}
    </span>
  );
}
