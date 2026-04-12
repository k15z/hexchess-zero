import { spawnSync } from "node:child_process";
import { constants } from "node:fs";
import { access, copyFile, mkdir } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = resolve(__dirname, "..");
const repoRoot = resolve(docsDir, "..");
const wasmPkgDir = resolve(repoRoot, "bindings/wasm/pkg");
const publicWasmDir = resolve(docsDir, "public/wasm");
const requiredFiles = ["hexchess_wasm.js", "hexchess_wasm_bg.wasm"];
const forceBuild = process.argv.includes("--build");

async function fileExists(path) {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function hasPreparedWasm() {
  for (const file of requiredFiles) {
    if (!(await fileExists(resolve(wasmPkgDir, file)))) {
      return false;
    }
  }
  return true;
}

function buildWasm() {
  console.log("Building docs WASM bundle with wasm-pack...");
  const result = spawnSync("wasm-pack", ["build", "--target", "web", "bindings/wasm"], {
    cwd: repoRoot,
    stdio: "inherit",
  });

  if (result.error?.code === "ENOENT") {
    console.error(
      "Docs build needs bindings/wasm/pkg. Install wasm-pack or prebuild it with `wasm-pack build --target web bindings/wasm`."
    );
    process.exit(1);
  }

  if (typeof result.status === "number" && result.status !== 0) {
    process.exit(result.status);
  }

  if (result.status === null) {
    console.error("wasm-pack did not exit cleanly.");
    process.exit(1);
  }
}

async function copyPreparedWasm() {
  await mkdir(publicWasmDir, { recursive: true });
  await Promise.all(
    requiredFiles.map((file) =>
      copyFile(resolve(wasmPkgDir, file), resolve(publicWasmDir, file))
    )
  );
  console.log(`Prepared docs WASM assets in ${relative(docsDir, publicWasmDir)}.`);
}

if (forceBuild || !(await hasPreparedWasm())) {
  buildWasm();
}

if (!(await hasPreparedWasm())) {
  console.error(
    "Docs build is missing the generated WASM assets in bindings/wasm/pkg. Run `wasm-pack build --target web bindings/wasm` first."
  );
  process.exit(1);
}

await copyPreparedWasm();
