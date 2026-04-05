import { defineDocs, defineConfig } from "fumadocs-mdx/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export const docs = defineDocs({
  dir: "content/docs",
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    // Prepend rehype-katex before built-in plugins (Shiki) so math nodes
    // are converted to KaTeX HTML before the code highlighter sees them.
    rehypePlugins: (v) => [rehypeKatex as any, ...v],
  },
});
