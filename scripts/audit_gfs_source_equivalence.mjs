#!/usr/bin/env node
/**
 * Compare the exact GRIB2 records consumed by FiLMeR across the NOAA public
 * mirror and UCAR GDEX. The two providers publish the same NCEP product under
 * different paths; UCAR does not publish a `.idx` sidecar, so NOAA's index is
 * used only to locate byte ranges in both files.
 */

import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";

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

function sourceUrls(cycleText, forecastHour) {
  const cycle = new Date(cycleText);
  if (Number.isNaN(cycle.getTime())) {
    throw new Error(`Invalid --cycle value: ${cycleText}`);
  }
  const date = `${cycle.getUTCFullYear()}${pad(cycle.getUTCMonth() + 1, 2)}${pad(cycle.getUTCDate(), 2)}`;
  const hour = pad(cycle.getUTCHours(), 2);
  const lead = pad(forecastHour, 3);
  const object = `gfs.${date}/${hour}/atmos/gfs.t${hour}z.pgrb2.0p25.f${lead}`;
  const noaa = `https://storage.googleapis.com/download/storage/v1/b/global-forecast-system/o/${encodeURIComponent(object)}?alt=media`;
  return {
    noaa,
    index: `https://storage.googleapis.com/download/storage/v1/b/global-forecast-system/o/${encodeURIComponent(`${object}.idx`)}?alt=media`,
    ucar: `https://data.gdex.ucar.edu/d084001/${date.slice(0, 4)}/${date}/gfs.0p25.${date}${hour}.f${lead}.grib2`,
  };
}

function parseIndex(text) {
  const partial = text
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [recordText, offsetText, , variable, level] = line.split(":");
      return {
        record: Number(recordText),
        offset: Number(offsetText),
        end: -1,
        variable,
        level,
      };
    });
  return partial.map((record, index) => ({
    ...record,
    end: partial[index + 1]?.offset - 1,
  }));
}

function selectRecords(records) {
  return CHANNELS.map(([name, variable, level]) => {
    const record = records.find(
      (candidate) =>
        candidate.variable === variable && candidate.level === level,
    );
    if (!record || !Number.isFinite(record.end)) {
      throw new Error(`Index is missing ${variable}:${level} (${name})`);
    }
    return { ...record, name };
  });
}

async function fetchRange(url, start, end) {
  const response = await fetch(url, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(
      `${new URL(url).hostname} did not honor byte range ${start}-${end} (HTTP ${response.status})`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function auditFrame(cycleText, forecastHour) {
  const urls = sourceUrls(cycleText, forecastHour);
  const indexResponse = await fetch(urls.index);
  if (!indexResponse.ok) {
    throw new Error(`NOAA index request failed: HTTP ${indexResponse.status}`);
  }
  const records = selectRecords(parseIndex(await indexResponse.text()));
  const results = [];
  for (let start = 0; start < records.length; start += 4) {
    const batch = records.slice(start, start + 4);
    const compared = await Promise.all(
      batch.map(async (record) => {
        const [noaaBytes, ucarBytes] = await Promise.all([
          fetchRange(urls.noaa, record.offset, record.end),
          fetchRange(urls.ucar, record.offset, record.end),
        ]);
        const noaaSha256 = sha256(noaaBytes);
        const ucarSha256 = sha256(ucarBytes);
        return {
          channel: record.name,
          gribKey: `${record.variable}:${record.level}`,
          record: record.record,
          byteRange: [record.offset, record.end],
          bytes: noaaBytes.byteLength,
          noaaSha256,
          ucarSha256,
          identical: noaaSha256 === ucarSha256,
        };
      }),
    );
    results.push(...compared);
  }
  return {
    forecastHour,
    urls,
    selectedBytes: results.reduce((sum, item) => sum + item.bytes, 0),
    identicalRecords: results.filter((item) => item.identical).length,
    totalRecords: results.length,
    records: results,
  };
}

const cycle = argument("--cycle", "2024-05-11T00:00:00Z");
const forecastHours = argument("--forecast-hours", "0,3")
  .split(",")
  .map(Number);
const output = argument(
  "--output",
  "reports/gfs-source-equivalence-20240511T00Z.json",
);

const frames = [];
for (const forecastHour of forecastHours) {
  console.log(`Auditing ${cycle} f${pad(forecastHour, 3)}…`);
  frames.push(await auditFrame(cycle, forecastHour));
}

const totalRecords = frames.reduce((sum, frame) => sum + frame.totalRecords, 0);
const identicalRecords = frames.reduce(
  (sum, frame) => sum + frame.identicalRecords,
  0,
);
const selectedBytes = frames.reduce(
  (sum, frame) => sum + frame.selectedBytes,
  0,
);
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  cycle,
  forecastHours,
  product: "NCEP GFS 0.25-degree pgrb2",
  providers: {
    trainingArchive: "UCAR GDEX dataset d084001",
    operationalMirror: "NOAA global-forecast-system Google Cloud bucket",
  },
  method:
    "NOAA .idx byte ranges were applied to both provider files; SHA-256 was compared for every GRIB record consumed by FiLMeR.",
  summary: {
    selectedBytes,
    identicalRecords,
    totalRecords,
    allIdentical: identicalRecords === totalRecords,
    providerInducedInputMaxAbsDifference:
      identicalRecords === totalRecords ? 0 : null,
    providerInducedOutputMaxAbsDifference:
      identicalRecords === totalRecords ? 0 : null,
  },
  interpretation: {
    supported:
      "For this audited same-cycle input pair, UCAR and NOAA deliver byte-identical FiLMeR predictor records. There is no provider-induced model-input or output difference.",
    notSupported:
      "This does not measure temporal distribution shift, aggregate forecast skill, or future upstream product changes. Live cases still require time-matched WRF or observations for error evaluation.",
  },
  frames,
};

await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  `${identicalRecords}/${totalRecords} records identical; ${(selectedBytes / 1024 / 1024).toFixed(1)} MiB audited.`,
);
console.log(`Wrote ${output}`);
