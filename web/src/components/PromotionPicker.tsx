import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { PIECE_SYMBOLS } from "@/engine/board-geometry";
import type { GameState } from "@/hooks/use-game";

interface Props {
  state: GameState;
  onSelect: (piece: string) => void;
  onCancel: () => void;
}

const PROMOTION_OPTIONS = ["queen", "rook", "bishop", "knight"];

export function PromotionPicker({ state, onSelect, onCancel }: Props) {
  const open = state.pendingPromotion !== null;
  const color = state.playerColor;

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onCancel(); }}>
      <DialogContent className="max-w-xs">
        <DialogHeader>
          <DialogTitle>Promote pawn to:</DialogTitle>
        </DialogHeader>
        <div className="flex gap-3 justify-center py-2">
          {PROMOTION_OPTIONS.filter(opt =>
            state.pendingPromotion?.moves.some(m => m.promotion === opt)
          ).map(opt => (
            <Button
              key={opt}
              variant="outline"
              className="w-16 h-16 text-4xl"
              onClick={() => onSelect(opt)}
              title={opt.charAt(0).toUpperCase() + opt.slice(1)}
            >
              {PIECE_SYMBOLS[color]?.[opt]}
            </Button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
