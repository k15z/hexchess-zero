import { Badge } from "@/components/ui/badge";
import type { GameState } from "@/hooks/use-game";

const STATUS_LABELS: Record<string, string> = {
  checkmate_white: "Checkmate \u2014 White wins!",
  checkmate_black: "Checkmate \u2014 Black wins!",
  stalemate: "Stalemate \u2014 Draw",
  draw_repetition: "Draw by repetition",
  draw_fifty: "Draw by fifty-move rule",
  draw_material: "Draw \u2014 insufficient material",
};

export function StatusBar({ state }: { state: GameState }) {
  if (!state.engineLoaded) {
    return <StatusShell>Loading engine...</StatusShell>;
  }

  if (state.aiThinking) {
    return (
      <StatusShell className="text-warning thinking-pulse">
        AI is thinking...
      </StatusShell>
    );
  }

  const label = STATUS_LABELS[state.status];
  if (label) {
    const prefix = state.selfPlayMode ? `[Self-Play \u00b7 Move ${state.moveCount}] ` : "";
    return <StatusShell className="text-success font-semibold">{prefix}{label}</StatusShell>;
  }

  const side = state.sideToMove === "white" ? "White" : "Black";
  const prefix = state.selfPlayMode ? `[Self-Play \u00b7 Move ${state.moveCount}] ` : "";

  if (state.isInCheck) {
    return (
      <StatusShell className="text-destructive font-semibold">
        {prefix}{side} to move &mdash; CHECK!
      </StatusShell>
    );
  }

  return (
    <StatusShell>
      <span className="flex items-center gap-2">
        {prefix}{side} to move
        {state.modelName && (
          <Badge variant="secondary" className="text-xs">
            {state.modelName}
          </Badge>
        )}
      </span>
    </StatusShell>
  );
}

function StatusShell({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className="bg-card border border-border rounded-lg px-6 py-3 text-center min-w-[300px] shadow-md">
      <span className={className}>{children}</span>
    </div>
  );
}
