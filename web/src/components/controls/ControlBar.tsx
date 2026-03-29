import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { GameState } from "@/hooks/use-game";

interface Props {
  state: GameState;
  onNewGame: () => void;
  onUndo: () => void;
  onFlipSide: () => void;
  onToggleSelfPlay: () => void;
  onSetSpeed: (val: number) => void;
}

export function ControlBar({ state, onNewGame, onUndo, onFlipSide, onToggleSelfPlay, onSetSpeed }: Props) {
  const disabled = state.aiThinking;

  return (
    <div className="flex gap-3 flex-wrap justify-center items-center">
      <Button variant="outline" onClick={onNewGame} disabled={disabled}>
        New Game
      </Button>

      <Button variant="outline" onClick={onUndo} disabled={disabled || state.selfPlayMode}>
        Undo
      </Button>

      <Button variant="outline" onClick={onFlipSide} disabled={disabled || state.selfPlayMode}>
        Flip Side
      </Button>

      <Button
        variant={state.selfPlayMode ? "default" : "outline"}
        onClick={onToggleSelfPlay}
        disabled={disabled}
      >
        {state.selfPlayMode ? "Stop" : "Self-Play"}
      </Button>

      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span>Speed</span>
        <Slider
          className="w-20"
          min={0}
          max={1000}
          step={50}
          value={[state.selfPlayDelay]}
          onValueChange={(v) => {
            const val = Array.isArray(v) ? v[0] : v;
            onSetSpeed(val);
          }}
        />
      </div>
    </div>
  );
}
