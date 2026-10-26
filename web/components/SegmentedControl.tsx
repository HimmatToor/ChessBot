import { cn } from "@/lib/utils";

type Option<T extends string> = { value: T; label: string; title?: string };

type SegmentedControlProps<T extends string> = {
  label: string;
  options: Option<T>[];
  value: T;
  onChange: (value: T) => void;
};

/** A labelled row of mutually exclusive pill buttons (e.g. White / Black). */
export default function SegmentedControl<T extends string>({ label, options, value, onChange }: SegmentedControlProps<T>) {
  return (
    <div className="flex flex-col gap-1.5">
      {/* Label above (not beside) so the buttons get the panel's full width. */}
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</span>
      <div
        role="group"
        aria-label={label}
        className="grid gap-1 rounded-lg bg-muted p-1"
        style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      >
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            title={option.title}
            onClick={() => onChange(option.value)}
            className={cn(
              "truncate rounded-md px-1.5 py-1.5 text-sm font-medium transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
              value === option.value
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
