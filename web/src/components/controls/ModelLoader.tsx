import { useRef } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface Props {
  modelName: string | null;
  onLoadModel: (file: File) => void;
  onClearModel: () => void;
}

export function ModelLoader({ modelName, onLoadModel, onClearModel }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onLoadModel(file);
    e.target.value = "";
  };

  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        accept=".onnx"
        className="hidden"
        onChange={handleChange}
      />
      <Button
        variant="outline"
        size="sm"
        onClick={() => inputRef.current?.click()}
      >
        Load Model
      </Button>
      {modelName ? (
        <Badge
          variant="secondary"
          className="cursor-pointer hover:bg-destructive/20"
          onClick={onClearModel}
          title="Click to remove model"
        >
          {modelName} &times;
        </Badge>
      ) : (
        <span className="text-xs text-muted-foreground">No model (heuristic)</span>
      )}
    </div>
  );
}
