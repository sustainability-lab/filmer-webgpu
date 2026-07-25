#!/usr/bin/env node
/**
 * ONNX Runtime Web 1.22 requests several adapter limits when creating its
 * WebGPU device, but omits maxStorageBuffersPerShaderStage. FiLMeR's exported
 * graph has a Concat kernel with nine storage buffers, while the WebGPU default
 * is eight. Request the adapter's advertised limit so supported devices do not
 * silently create an invalid compute pipeline.
 *
 * Upstream source:
 * node_modules/onnxruntime-web/lib/wasm/jsep/backend-webgpu.ts
 */

import { createRequire } from "node:module";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const webGpuEntry = require.resolve("onnxruntime-web/webgpu");
const distDirectory = dirname(webGpuEntry);
const candidates = readdirSync(distDirectory)
  .filter(
    (name) =>
      name.startsWith("ort.webgpu") &&
      (name.endsWith(".js") || name.endsWith(".mjs")),
  )
  .map((name) => join(distDirectory, name));

const workgroupLimit =
  /maxComputeWorkgroupStorageSize:\s*([$A-Z_a-z][$\w]*)\.limits\.maxComputeWorkgroupStorageSize/g;
let patchedFiles = 0;
let verifiedFiles = 0;

for (const filePath of candidates) {
  let source = readFileSync(filePath, "utf8");
  let replacements = 0;
  source = source.replace(
    workgroupLimit,
    (match, adapterName, offset, fullSource) => {
      const preceding = fullSource.slice(Math.max(0, offset - 180), offset);
      if (preceding.includes("maxStorageBuffersPerShaderStage")) {
        return match;
      }
      replacements += 1;
      return (
        `maxStorageBuffersPerShaderStage:${adapterName}.limits.` +
        `maxStorageBuffersPerShaderStage,${match}`
      );
    },
  );

  if (replacements > 0) {
    writeFileSync(filePath, source);
    patchedFiles += 1;
  }
  if (source.includes("maxStorageBuffersPerShaderStage")) {
    verifiedFiles += 1;
  }
}

if (candidates.length === 0 || verifiedFiles !== candidates.length) {
  throw new Error(
    `Unsupported onnxruntime-web layout: verified ${verifiedFiles}/${candidates.length} WebGPU bundles`,
  );
}

console.log(
  `Verified WebGPU storage-buffer device limit in ${verifiedFiles} bundles` +
    (patchedFiles ? ` (${patchedFiles} patched)` : ""),
);
