import { strFromU8, unzipSync } from "fflate";
import { GRID_CELLS, assembleInput } from "./preprocess";

export type SequenceManifest = {
  schemaVersion: 1;
  kind: "filmer-conditional-gfs-sequence";
  domainId: number;
  initialization: string;
  horizonHours: number;
  cadenceHours: 3;
  frameTimes: string[];
  outputTimes: string[];
  gfs: {
    file: string;
    shape: [number, 20, 127, 137];
    normalized: false;
  };
  static: {
    file: string;
    shape: [30, 127, 137];
    normalized: true;
    month: number;
  };
  outputGrid?: {
    latitudeFile: string;
    longitudeFile: string;
    shape: [99, 99];
  };
  semantics: string;
};

export type SequenceBundle = {
  manifest: SequenceManifest;
  gfs: Float32Array;
  normalizedStatic: Float32Array;
  latitude?: Float32Array;
  longitude?: Float32Array;
};

function f32FromU8(bytes: Uint8Array): Float32Array {
  const copied = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  return new Float32Array(copied);
}

export async function readSequenceBundle(file: File): Promise<SequenceBundle> {
  const entries = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const manifestBytes = entries["manifest.json"];
  if (!manifestBytes) throw new Error("Bundle is missing manifest.json");
  const manifest = JSON.parse(strFromU8(manifestBytes)) as SequenceManifest;
  if (
    manifest.schemaVersion !== 1 ||
    manifest.kind !== "filmer-conditional-gfs-sequence"
  ) {
    throw new Error("Unsupported FiLMeR bundle schema");
  }
  if (manifest.cadenceHours !== 3) {
    throw new Error("FiLMeR v1.0 accepts only 3-hour GFS cadence");
  }
  const expectedSteps = manifest.horizonHours / 3;
  if (
    manifest.outputTimes.length !== expectedSteps ||
    manifest.frameTimes.length !== expectedSteps + 1
  ) {
    throw new Error("Bundle timestamps do not match the requested horizon");
  }
  const gfsBytes = entries[manifest.gfs.file];
  const staticBytes = entries[manifest.static.file];
  if (!gfsBytes || !staticBytes) {
    throw new Error("Bundle is missing GFS or static tensors");
  }
  const gfs = f32FromU8(gfsBytes);
  const normalizedStatic = f32FromU8(staticBytes);
  if (gfs.length !== manifest.frameTimes.length * 20 * GRID_CELLS) {
    throw new Error("GFS tensor length does not match the bundle manifest");
  }
  if (normalizedStatic.length !== 30 * GRID_CELLS) {
    throw new Error("Static tensor must contain 30x127x137 values");
  }
  const latitudeBytes = manifest.outputGrid
    ? entries[manifest.outputGrid.latitudeFile]
    : undefined;
  const longitudeBytes = manifest.outputGrid
    ? entries[manifest.outputGrid.longitudeFile]
    : undefined;
  const latitude = latitudeBytes ? f32FromU8(latitudeBytes) : undefined;
  const longitude = longitudeBytes ? f32FromU8(longitudeBytes) : undefined;
  if (
    (latitude && latitude.length !== 99 * 99) ||
    (longitude && longitude.length !== 99 * 99)
  ) {
    throw new Error("Output-grid coordinates must contain 99x99 values");
  }
  return { manifest, gfs, normalizedStatic, latitude, longitude };
}

export function inputForSequenceStep(
  bundle: SequenceBundle,
  step: number,
): Float32Array {
  const frameSize = 20 * GRID_CELLS;
  if (step < 0 || step >= bundle.manifest.outputTimes.length) {
    throw new Error(`Sequence step ${step} is outside the bundle`);
  }
  const previous = bundle.gfs.subarray(step * frameSize, (step + 1) * frameSize);
  const current = bundle.gfs.subarray(
    (step + 1) * frameSize,
    (step + 2) * frameSize,
  );
  return assembleInput(previous, current, bundle.normalizedStatic);
}
