"use client";

import { useEffect, useState } from "react";
import "./mermaid.css";

let mermaidReady: Promise<typeof import("mermaid")> | null = null;

function loadMermaid() {
  if (!mermaidReady) {
    mermaidReady = import("mermaid").then((mod) => {
      mod.default.initialize({
        startOnLoad: false,
        theme: "neutral",
        fontFamily:
          'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        flowchart: { htmlLabels: true, curve: "basis" },
      });
      return mod;
    });
  }
  return mermaidReady;
}

export interface MermaidProps {
  chart: string;
  caption?: string;
}

export function Mermaid({ chart, caption }: MermaidProps) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const id = `mermaid-${Math.random().toString(36).slice(2, 9)}`;

    loadMermaid().then(async (mod) => {
      if (cancelled) return;
      try {
        const { svg: rendered } = await mod.default.render(id, chart.trim());
        if (!cancelled) setSvg(rendered);
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [chart]);

  if (error) {
    return <pre className="mermaid-error">{error}</pre>;
  }

  return (
    <figure className="diagram-figure">
      <div
        className="mermaid-container"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
      {caption && <figcaption className="diagram-caption">{caption}</figcaption>}
    </figure>
  );
}
