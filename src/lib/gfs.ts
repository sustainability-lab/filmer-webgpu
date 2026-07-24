import gribParser, { type GribField } from "grib-js/lib/parser";
import { GRID_CELLS, GRID_HEIGHT, GRID_WIDTH } from "./preprocess";

// grib-js checks `instanceof Buffer` even on its documented browser path.
// A sentinel keeps that guard defined; browser inputs are always ArrayBuffers.
if (typeof globalThis.Buffer === "undefined") {
  Object.defineProperty(globalThis, "Buffer", {
    configurable: true,
    value: class BrowserBufferSentinel {},
  });
}

const GCS_BUCKET = "global-forecast-system";
const CROP = {
  latStart: 214,
  latEnd: 341,
  lonStart: 264,
  lonEnd: 401,
} as const;

export const GFS_CHANNELS = [
  { name: "T2", variable: "TMP", level: "2 m above ground" },
  { name: "Q2", variable: "SPFH", level: "2 m above ground" },
  { name: "PSFC", variable: "PRES", level: "surface" },
  { name: "U10", variable: "UGRD", level: "10 m above ground" },
  { name: "V10", variable: "VGRD", level: "10 m above ground" },
  { name: "TCC", variable: "TCDC", level: "entire atmosphere" },
  { name: "PRATE", variable: "PRATE", level: "surface" },
  { name: "T_1000", variable: "TMP", level: "1000 mb" },
  { name: "T_850", variable: "TMP", level: "850 mb" },
  { name: "T_700", variable: "TMP", level: "700 mb" },
  { name: "T_500", variable: "TMP", level: "500 mb" },
  { name: "U_850", variable: "UGRD", level: "850 mb" },
  { name: "U_700", variable: "UGRD", level: "700 mb" },
  { name: "U_500", variable: "UGRD", level: "500 mb" },
  { name: "V_850", variable: "VGRD", level: "850 mb" },
  { name: "V_700", variable: "VGRD", level: "700 mb" },
  { name: "V_500", variable: "VGRD", level: "500 mb" },
  { name: "Q_850", variable: "SPFH", level: "850 mb" },
  { name: "Q_700", variable: "SPFH", level: "700 mb" },
  { name: "Q_500", variable: "SPFH", level: "500 mb" },
] as const;

type IndexRecord = {
  record: number;
  offset: number;
  end: number;
  variable: string;
  level: string;
  description: string;
};

export type GfsProgress = {
  stage: "index" | "download" | "decode";
  frame: string;
  completedFields: number;
  totalFields: number;
  loadedBytes: number;
  totalBytes: number;
};

function pad(value: number, width: number) {
  return String(value).padStart(width, "0");
}

function cycleParts(cycle: Date) {
  return {
    date: `${cycle.getUTCFullYear()}${pad(cycle.getUTCMonth() + 1, 2)}${pad(
      cycle.getUTCDate(),
      2,
    )}`,
    hour: pad(cycle.getUTCHours(), 2),
  };
}

export function gfsObjectName(cycle: Date, forecastHour: number) {
  if (forecastHour < 0 || forecastHour > 384 || forecastHour % 3 !== 0) {
    throw new Error("GFS forecast hour must be a 3-hour lead from 0 to 384");
  }
  const { date, hour } = cycleParts(cycle);
  return `gfs.${date}/${hour}/atmos/gfs.t${hour}z.pgrb2.0p25.f${pad(
    forecastHour,
    3,
  )}`;
}

function publicObjectUrl(object: string) {
  return `https://storage.googleapis.com/download/storage/v1/b/${GCS_BUCKET}/o/${encodeURIComponent(
    object,
  )}?alt=media`;
}

export function parseGfsIndex(text: string): IndexRecord[] {
  const partial = text
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [recordText, offsetText, , variable, level, ...rest] =
        line.split(":");
      const record = Number(recordText);
      const offset = Number(offsetText);
      if (!Number.isFinite(record) || !Number.isFinite(offset)) {
        throw new Error(`Malformed GFS index line: ${line}`);
      }
      return {
        record,
        offset,
        end: -1,
        variable,
        level,
        description: rest.join(":"),
      };
    });
  return partial.map((item, index) => ({
    ...item,
    end:
      index + 1 < partial.length
        ? partial[index + 1].offset - 1
        : Number.POSITIVE_INFINITY,
  }));
}

export function selectGfsChannels(records: IndexRecord[]) {
  return GFS_CHANNELS.map((channel) => {
    const selected = records.find(
      (record) =>
        record.variable === channel.variable && record.level === channel.level,
    );
    if (!selected) {
      throw new Error(
        `GFS index is missing ${channel.variable}:${channel.level} (${channel.name})`,
      );
    }
    if (!Number.isFinite(selected.end)) {
      throw new Error(`GFS record ${selected.record} has no ending byte offset`);
    }
    return selected;
  });
}

async function decodeGrib(buffer: ArrayBuffer): Promise<GribField> {
  const messages = gribParser.parseDataView(buffer);
  if (!messages[0]) {
    throw new Error("GRIB2 decoder returned no messages");
  }
  return messages[0];
}

export function cropGlobalField(values: number[]): Float32Array {
  if (values.length !== 721 * 1440) {
    throw new Error(
      `Expected a 721x1440 GFS field; received ${values.length} values`,
    );
  }
  const cropped = new Float32Array(GRID_CELLS);
  let destination = 0;
  for (let row = CROP.latStart; row < CROP.latEnd; row += 1) {
    const rowOffset = row * 1440;
    for (let column = CROP.lonStart; column < CROP.lonEnd; column += 1) {
      cropped[destination] = values[rowOffset + column];
      destination += 1;
    }
  }
  if (destination !== GRID_HEIGHT * GRID_WIDTH) {
    throw new Error("Internal GFS crop dimensions do not match FiLMeR");
  }
  return cropped;
}

async function fetchRange(
  url: string,
  record: IndexRecord,
): Promise<ArrayBuffer> {
  const response = await fetch(url, {
    headers: { Range: `bytes=${record.offset}-${record.end}` },
    cache: "force-cache",
  });
  if (!response.ok || (response.status !== 206 && response.status !== 200)) {
    throw new Error(
      `GFS byte-range request failed (${response.status}) for ${record.variable}:${record.level}`,
    );
  }
  return response.arrayBuffer();
}

export async function fetchGfsFrame(
  cycle: Date,
  forecastHour: number,
  onProgress: (progress: GfsProgress) => void,
): Promise<Float32Array> {
  const object = gfsObjectName(cycle, forecastHour);
  const frame = `${cycle.toISOString().slice(0, 13)}Z +${forecastHour}h`;
  const indexResponse = await fetch(publicObjectUrl(`${object}.idx`), {
    cache: "no-cache",
  });
  if (!indexResponse.ok) {
    throw new Error(
      `GFS index is unavailable (${indexResponse.status}); the selected cycle may not have completed`,
    );
  }
  const records = selectGfsChannels(
    parseGfsIndex(await indexResponse.text()),
  );
  const totalBytes = records.reduce(
    (sum, record) => sum + record.end - record.offset + 1,
    0,
  );
  onProgress({
    stage: "index",
    frame,
    completedFields: 0,
    totalFields: records.length,
    loadedBytes: 0,
    totalBytes,
  });
  const output = new Float32Array(20 * GRID_CELLS);
  let completedFields = 0;
  let loadedBytes = 0;

  // Four concurrent record reads keep bandwidth busy without holding all
  // twenty global decoded fields in memory at once.
  for (let start = 0; start < records.length; start += 4) {
    const batch = records.slice(start, start + 4);
    const buffers = await Promise.all(
      batch.map((record) => fetchRange(publicObjectUrl(object), record)),
    );
    loadedBytes += buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
    onProgress({
      stage: "download",
      frame,
      completedFields,
      totalFields: records.length,
      loadedBytes,
      totalBytes,
    });
    for (let index = 0; index < buffers.length; index += 1) {
      const message = await decodeGrib(buffers[index]);
      const field = message.fields[0];
      if (
        field.grid.definition.ni !== 1440 ||
        field.grid.definition.nj !== 721
      ) {
        throw new Error(
          `Unexpected GFS grid ${field.grid.definition.ni}x${field.grid.definition.nj}`,
        );
      }
      output.set(
        cropGlobalField(field.data),
        (start + index) * GRID_CELLS,
      );
      completedFields += 1;
      onProgress({
        stage: "decode",
        frame,
        completedFields,
        totalFields: records.length,
        loadedBytes,
        totalBytes,
      });
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
    }
  }
  return output;
}

async function loadF32(relativePath: string) {
  const response = await fetch(`${import.meta.env.BASE_URL}${relativePath}`);
  if (!response.ok) {
    throw new Error(`Static artifact fetch failed (${response.status})`);
  }
  return new Float32Array(await response.arrayBuffer());
}

export async function loadStaticMonth(month: number) {
  if (month < 1 || month > 12) throw new Error("Month must be between 1 and 12");
  return loadF32(`data/static/static-month-${pad(month, 2)}.f32`);
}

export async function loadOutputGrid(domainId: number) {
  const code = `d${pad(domainId, 2)}`;
  const [latitude, longitude] = await Promise.all([
    loadF32(`data/static/grid-${code}-latitude.f32`),
    loadF32(`data/static/grid-${code}-longitude.f32`),
  ]);
  return { latitude, longitude };
}

export function previousCycleInput(cycle: Date) {
  return {
    cycle: new Date(cycle.getTime() - 6 * 60 * 60 * 1000),
    forecastHour: 3,
  };
}
