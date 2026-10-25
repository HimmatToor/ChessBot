import { cn } from "@/lib/utils";
import type { MaterialSide } from "@/lib/material";

// Solid glyphs in a single muted colour read well on both themes.
// U+FE0E forces text (not emoji) rendering, which matters for ♟ on iOS/macOS.
const GLYPHS = { q: "♛︎", r: "♜︎", b: "♝︎", n: "♞︎", p: "♟︎" };

type PlayerBarProps = {
  name: string;
  color: "w" | "b";
  isTurn: boolean;
  material: MaterialSide;
};

export default function PlayerBar({ name, color, isTurn, material }: PlayerBarProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors",
        isTurn ? "border-primary/60 bg-primary/10" : "bg-card",
      )}
    >
      <span
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-md text-xl leading-none",
          color === "w" ? "bg-white text-slate-900" : "bg-slate-900 text-white",
        )}
        aria-hidden
      >
        {color === "w" ? "♔" : "♚"}
      </span>

      <div className="flex min-w-0 flex-col">
        <span className="font-medium leading-tight">{name}</span>
        <CapturedPieces material={material} />
      </div>

      {isTurn && (
        <span className="ml-auto flex shrink-0 items-center gap-2 text-xs font-medium text-primary">
          <span className="size-2 animate-pulse rounded-full bg-primary" />
          To move
        </span>
      )}
    </div>
  );
}

function CapturedPieces({ material }: { material: MaterialSide }) {
  const { captured, advantage } = material;
  // Reserve the row's height even when empty so the bar doesn't jump.
  return (
    <div className="flex h-5 items-center text-muted-foreground" aria-label={describe(material)}>
      {captured.map((type, i) => (
        <span
          key={i}
          // Overlap identical pieces; leave a gap between different types.
          className={cn("text-base leading-none", i > 0 && (captured[i - 1] === type ? "-ml-1.5" : "ml-1"))}
          aria-hidden
        >
          {GLYPHS[type]}
        </span>
      ))}
      {advantage > 0 && (
        <span className="ml-1.5 text-xs font-semibold" aria-hidden>
          +{advantage}
        </span>
      )}
    </div>
  );
}

function describe({ captured, advantage }: MaterialSide): string {
  if (captured.length === 0) return "No material captured";
  const lead = advantage > 0 ? `, up ${advantage}` : "";
  return `Captured: ${captured.join(" ")}${lead}`;
}
