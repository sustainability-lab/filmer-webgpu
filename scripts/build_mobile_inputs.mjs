#!/usr/bin/env node
/**
 * Build the compact live-GFS tensor consumed by the native FiLMeR apps.
 *
 * The mobile bundle contains the exact 20 records and 127x137 crop used by the
 * browser. GRIB selection/decoding happens once in CI; phones download only
 * little-endian float32 tensors plus a checksum-pinned manifest.
 */

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const gribParser = require("grib-js/lib/parser");

const GRID_HEIGHT = 127;
const GRID_WIDTH = 137;
const GRID_CELLS = GRID_HEIGHT * GRID_WIDTH;
const CROP = { latStart: 214, latEnd: 341, lonStart: 264, lonEnd: 401 };
const CHANNELS = [
  ["T2", "TMP", "2 m above ground"],
  ["Q2", "SPFH", "2 m above ground"],
  ["PSFC", "PRES", "surface"],
  ["U10", "UGRD", "10 m above ground"],
  ["V10", "VGRD", "10 m above ground"],
  ["TCC", "TCDC", "entire atmosphere"],
  ["PRATE", "PRATE", "surface"],
  ["T_1000", "TMP", "1000 mb"],
  ["T_850", "TMP", "850 mb"],
  ["T_700", "TMP", "700 mb"],
  ["T_500", "TMP", "500 mb"],
  ["U_850", "UGRD", "850 mb"],
  ["U_700", "UGRD", "700 mb"],
  ["U_500", "UGRD", "500 mb"],
  ["V_850", "VGRD", "850 mb"],
  ["V_700", "VGRD", "700 mb"],
  ["V_500", "VGRD", "500 mb"],
  ["Q_850", "SPFH", "850 mb"],
  ["Q_700", "SPFH", "700 mb"],
  ["Q_500", "SPFH", "500 mb"],
];

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function pad(value, width) {
  return String(value).padStart(width, "0");
}

function latestCandidate() {
  const now = new Date();
  const hour = Math.floor(now.getUTCHours() / 6) * 6;
  const cycle = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), hour),
  );
  cycle.setUTCHours(cycle.getUTCHours() - 6);
  return cycle;
}

function objectName(cycle, forecastHour) {
  const date = `${cycle.getUTCFullYear()}${pad(cycle.getUTCMonth() + 1, 2)}${pad(cycle.getUTCDate(), 2)}`;
  const hour = pad(cycle.getUTCHours(), 2);
  return `gfs.${date}/${hour}/atmos/gfs.t${hour}z.pgrb2.0p25.f${pad(forecastHour, 3)}`;
}

function objectUrl(object) {
  return `https://storage.googleapis.com/download/storage/v1/b/global-forecast-system/o/${encodeURIComponent(object)}?alt=media`;
}

function parseIndex(text) {
  const records = text
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [record, offset, , variable, level] = line.split(":");
      return { record: Number(record), offset: Number(offset), end: -1, variable, level };
    });
  return records.map((record, index) => ({
    ...record,
    end: records[index + 1]?.offset - 1,
  }));
}

function selectedRecords(records) {
  return CHANNELS.map(([name, variable, level]) => {
    const record = records.find(
      (candidate) => candidate.variable === variable && candidate.level === level,
    );
    if (!record || !Number.isFinite(record.end)) {
      throw new Error(`GFS index is missing ${variable}:${level} (${name})`);
    }
    return { ...record, name };
  });
}

async function fetchChecked(url, options = {}) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`HTTP ${response.status}`);
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt < 3) await new Promise((resolveDelay) => setTimeout(resolveDelay, attempt * 1_000));
    }
  }
  throw new Error(`${new URL(url).hostname} request failed: ${lastError}`);
}

async function availableCycle(candidate) {
  for (let offset = 0; offset < 4; offset += 1) {
    const cycle = new Date(candidate.getTime() - offset * 6 * 3_600_000);
    const response = await fetch(objectUrl(`${objectName(cycle, 21)}.idx`));
    if (response.ok) {
      await response.body?.cancel();
      return cycle;
    }
    await response.body?.cancel();
  }
  throw new Error("No complete GFS cycle was available in the last 24 hours");
}

function crop(values) {
  if (values.length !== 721 * 1440) {
    throw new Error(`Unexpected decoded GFS field length ${values.length}`);
  }
  const output = new Float32Array(GRID_CELLS);
  let destination = 0;
  for (let row = CROP.latStart; row < CROP.latEnd; row += 1) {
    const offset = row * 1440;
    for (let column = CROP.lonStart; column < CROP.lonEnd; column += 1) {
      output[destination] = values[offset + column];
      destination += 1;
    }
  }
  return output;
}

async function frame(cycle, forecastHour) {
  const object = objectName(cycle, forecastHour);
  const indexResponse = await fetchChecked(objectUrl(`${object}.idx`));
  const records = selectedRecords(parseIndex(await indexResponse.text()));
  const output = new Float32Array(20 * GRID_CELLS);
  let selectedBytes = 0;
  for (let start = 0; start < records.length; start += 4) {
    const batch = records.slice(start, start + 4);
    const buffers = await Promise.all(
      batch.map(async (record) => {
        const response = await fetchChecked(objectUrl(object), {
          headers: { Range: `bytes=${record.offset}-${record.end}` },
        });
        if (response.status !== 206 && response.status !== 200) {
          throw new Error(`GFS range was not honored for ${record.name}`);
        }
        return response.arrayBuffer();
      }),
    );
    for (let index = 0; index < buffers.length; index += 1) {
      const buffer = buffers[index];
      selectedBytes += buffer.byteLength;
      const messages = gribParser.parseDataView(buffer);
      const field = messages[0]?.fields?.[0];
      if (!field) throw new Error(`GRIB decoder returned no ${batch[index].name} field`);
      if (field.grid.definition.ni !== 1440 || field.grid.definition.nj !== 721) {
        throw new Error(`Unexpected GFS grid for ${batch[index].name}`);
      }
      output.set(crop(field.data), (start + index) * GRID_CELLS);
    }
  }
  return { output, selectedBytes };
}

const requestedCycle = argument("--cycle", null);
const candidate = requestedCycle ? new Date(requestedCycle) : latestCandidate();
if (Number.isNaN(candidate.getTime())) throw new Error(`Invalid --cycle ${requestedCycle}`);
const cycle = requestedCycle ? candidate : await availableCycle(candidate);
const outputDirectory = resolve(argument("--output", "mobile-inputs"));
const baseUrl = argument(
  "--base-url",
  "https://github.com/sustainability-lab/filmer-webgpu/releases/download/mobile-inputs",
);

const previousCycle = new Date(cycle.getTime() - 6 * 3_600_000);
const requests = [
  { cycle: previousCycle, forecastHour: 3 },
  ...Array.from({ length: 8 }, (_, index) => ({
    cycle,
    forecastHour: index * 3,
  })),
];
const tensor = new Float32Array(requests.length * 20 * GRID_CELLS);
let selectedBytes = 0;
for (let index = 0; index < requests.length; index += 1) {
  const request = requests[index];
  console.log(
    `[${index + 1}/${requests.length}] ${request.cycle.toISOString()} f${pad(request.forecastHour, 3)}`,
  );
  const decoded = await frame(request.cycle, request.forecastHour);
  tensor.set(decoded.output, index * 20 * GRID_CELLS);
  selectedBytes += decoded.selectedBytes;
}

const bytes = Buffer.from(tensor.buffer, tensor.byteOffset, tensor.byteLength);
const sha256 = createHash("sha256").update(bytes).digest("hex");
const frameTimes = [
  new Date(cycle.getTime() - 3 * 3_600_000).toISOString(),
  ...Array.from({ length: 8 }, (_, index) =>
    new Date(cycle.getTime() + index * 3 * 3_600_000).toISOString(),
  ),
];
const outputTimes = Array.from({ length: 8 }, (_, index) =>
  new Date(cycle.getTime() + (index + 1) * 3 * 3_600_000).toISOString(),
);
const manifest = {
  schemaVersion: 1,
  kind: "filmer-mobile-gfs",
  generatedAt: new Date().toISOString(),
  provider: "NOAA NCEP GFS public Google Cloud mirror",
  product: "GFS 0.25-degree pgrb2",
  initialization: cycle.toISOString(),
  cadenceHours: 3,
  frameTimes,
  outputTimes,
  selectedGribBytes: selectedBytes,
  gfs: {
    file: "gfs.f32",
    url: `${baseUrl}/gfs.f32`,
    dtype: "little-endian float32",
    shape: [requests.length, 20, GRID_HEIGHT, GRID_WIDTH],
    normalized: false,
    bytes: bytes.byteLength,
    sha256,
  },
  semantics:
    "Each FiLMeR output at t+3h is conditioned on GFS(t-3h,t); outputs are not fed back. The package contains eight three-hour output steps.",
  scope:
    "Predictors only. The installed native app supplies the pinned FP32 model and normalized static geography.",
};

await mkdir(outputDirectory, { recursive: true });
await Promise.all([
  writeFile(resolve(outputDirectory, "gfs.f32"), bytes),
  writeFile(resolve(outputDirectory, "latest.json"), `${JSON.stringify(manifest, null, 2)}\n`),
]);
console.log(
  `Wrote ${(bytes.byteLength / 1024 / 1024).toFixed(2)} MiB (${sha256}); selected ${(selectedBytes / 1024 / 1024).toFixed(1)} MiB of GRIB records.`,
);
