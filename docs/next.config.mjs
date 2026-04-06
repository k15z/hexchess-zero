import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

const basePath = "/hexchess-zero";

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: "export",
  basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
};

export default withMDX(config);
