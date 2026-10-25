import { cn } from "@/lib/utils";

type PlayerBarProps = {
  name: string;
  color: "w" | "b";
  isTurn: boolean;
};

export default function PlayerBar({ name, color, isTurn }: PlayerBarProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-lg border px-3 py-2 transition-colors",
        isTurn ? "border-primary/60 bg-primary/10" : "bg-card",
      )}
    >
      <span
        className={cn(
          "flex size-8 items-center justify-center rounded-md text-xl leading-none",
          color === "w" ? "bg-white text-slate-900" : "bg-slate-900 text-white",
        )}
        aria-hidden
      >
        {color === "w" ? "♔" : "♚"}
      </span>
      <span className="font-medium">{name}</span>
      {isTurn && (
        <span className="ml-auto flex items-center gap-2 text-xs font-medium text-primary">
          <span className="size-2 animate-pulse rounded-full bg-primary" />
          To move
        </span>
      )}
    </div>
  );
}
