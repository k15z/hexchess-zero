import { HexBoard } from "@/components/board/HexBoard";
import { StatusBar } from "@/components/status/StatusBar";
import { ControlBar } from "@/components/controls/ControlBar";
import { ModelLoader } from "@/components/controls/ModelLoader";
import { PromotionPicker } from "@/components/PromotionPicker";
import { useGame } from "@/hooks/use-game";

export default function App() {
  const { state, actions } = useGame();

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="flex flex-col items-center gap-4 p-5 w-full max-w-[800px]">
        <StatusBar state={state} />
        <HexBoard state={state} onCellClick={actions.selectCell} />
        <ControlBar
          state={state}
          onNewGame={actions.newGame}
          onUndo={actions.undo}
          onFlipSide={actions.flipSide}
          onToggleSelfPlay={actions.toggleSelfPlay}
          onSetSpeed={actions.setSpeed}
        />
        <ModelLoader
          modelName={state.modelName}
          onLoadModel={actions.loadModel}
          onClearModel={actions.clearModel}
        />
        <PromotionPicker
          state={state}
          onSelect={actions.applyPromotion}
          onCancel={actions.cancelPromotion}
        />
      </div>
    </div>
  );
}
