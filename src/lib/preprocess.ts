import metadata from "../data/model-metadata.json";

export const GRID_HEIGHT = 127;
export const GRID_WIDTH = 137;
export const OUTPUT_SIZE = 99;
export const GRID_CELLS = GRID_HEIGHT * GRID_WIDTH;
export const OUTPUT_CELLS = OUTPUT_SIZE * OUTPUT_SIZE;

export type Domain = (typeof metadata.domains)[number];

export function domainById(id: number): Domain {
  const domain = metadata.domains.find((candidate) => candidate.id === id);
  if (!domain) {
    throw new Error(`Unknown trained domain d${String(id).padStart(2, "0")}`);
  }
  return domain;
}

export function normalizeGfsFrame(raw: Float32Array): Float32Array {
  if (raw.length !== 20 * GRID_CELLS) {
    throw new Error(
      `Expected ${20 * GRID_CELLS} GFS values, received ${raw.length}`,
    );
  }
  const normalized = new Float32Array(raw.length);
  for (let channel = 0; channel < 20; channel += 1) {
    const mean = metadata.gfs.mean[channel];
    const std = metadata.gfs.std[channel];
    const offset = channel * GRID_CELLS;
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      normalized[offset + cell] = (raw[offset + cell] - mean) / std;
    }
  }
  return normalized;
}

export function normalizeStaticFields(raw: Float32Array): Float32Array {
  if (raw.length !== 30 * GRID_CELLS) {
    throw new Error(
      `Expected ${30 * GRID_CELLS} static values, received ${raw.length}`,
    );
  }
  const normalized = new Float32Array(raw);
  for (let channel = 1; channel < 9; channel += 1) {
    const offset = channel * GRID_CELLS;
    let sum = 0;
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      sum += raw[offset + cell];
    }
    const mean = sum / GRID_CELLS;
    let squared = 0;
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      const deviation = raw[offset + cell] - mean;
      squared += deviation * deviation;
    }
    // PyTorch torch.std uses Bessel's correction by default.
    const std = Math.sqrt(squared / (GRID_CELLS - 1)) || 1;
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      normalized[offset + cell] = (raw[offset + cell] - mean) / std;
    }
  }
  return normalized;
}

export function assembleInput(
  previousGfs: Float32Array,
  currentGfs: Float32Array,
  normalizedStatic: Float32Array,
): Float32Array {
  const previous = normalizeGfsFrame(previousGfs);
  const current = normalizeGfsFrame(currentGfs);
  if (normalizedStatic.length !== 30 * GRID_CELLS) {
    throw new Error("Static tensor does not have 30x127x137 values");
  }
  const input = new Float32Array(70 * GRID_CELLS);
  input.set(previous, 0);
  input.set(current, 20 * GRID_CELLS);
  input.set(normalizedStatic, 40 * GRID_CELLS);
  return input;
}

export function normalizedProjection(
  domainId: number,
  timestamp: Date,
): Float32Array {
  const domain = domainById(domainId);
  const [latMin, lonMin, latMax, lonMax] = domain.modelBounds;
  const latCenter = (latMin + latMax) / 2;
  const lonCenter = (lonMin + lonMax) / 2;
  const latSpan = latMax - latMin;
  const lonSpan = lonMax - lonMin;
  let relativeLat = 0.5;
  let relativeLon = 0.5;
  if (domainId !== 1) {
    const [pLatMin, pLonMin, pLatMax, pLonMax] =
      metadata.domains[0].modelBounds;
    relativeLat = (latCenter - pLatMin) / (pLatMax - pLatMin);
    relativeLon = (lonCenter - pLonMin) / (pLonMax - pLonMin);
  }
  const yearStart = Date.UTC(timestamp.getUTCFullYear(), 0, 0);
  const dayOfYear = Math.floor(
    (timestamp.getTime() - yearStart) / 86_400_000,
  );
  const raw = [
    latMin,
    lonMin,
    latMax,
    lonMax,
    latCenter,
    lonCenter,
    latSpan,
    lonSpan,
    latSpan * lonSpan,
    latSpan / lonSpan,
    relativeLat,
    relativeLon,
    domain.resolutionKm,
    timestamp.getUTCHours() / 24,
    dayOfYear / 365,
    (timestamp.getUTCMonth() + 1) / 12,
  ];
  return Float32Array.from(
    raw.map(
      (value, index) =>
        (value - metadata.projection.mean[index]) /
        metadata.projection.std[index],
    ),
  );
}

export function reconstructPhysical(
  state: Float32Array,
  occurrence: Float32Array,
  intensity: Float32Array,
): Float32Array {
  if (
    state.length !== 5 * OUTPUT_CELLS ||
    occurrence.length !== OUTPUT_CELLS ||
    intensity.length !== OUTPUT_CELLS
  ) {
    throw new Error("Unexpected FiLMeR output shape");
  }
  const physical = new Float32Array(6 * OUTPUT_CELLS);
  metadata.targets.stateIndices.forEach((sourceIndex, channel) => {
    const mean = metadata.targets.mean[sourceIndex];
    const std = metadata.targets.std[sourceIndex];
    const offset = channel * OUTPUT_CELLS;
    for (let cell = 0; cell < OUTPUT_CELLS; cell += 1) {
      physical[offset + cell] = state[offset + cell] * std + mean;
    }
  });
  const precipOffset = 5 * OUTPUT_CELLS;
  for (let cell = 0; cell < OUTPUT_CELLS; cell += 1) {
    const wet = occurrence[cell] > 0; // sigmoid(logit) > 0.5
    physical[precipOffset + cell] = wet
      ? Math.expm1(Math.min(intensity[cell], 10))
      : 0;
  }
  return physical;
}

export function sequenceRequirements(horizonHours: number) {
  if (horizonHours <= 0 || horizonHours % 3 !== 0) {
    throw new Error("Horizon must be a positive multiple of three hours");
  }
  const outputSteps = horizonHours / 3;
  return {
    outputSteps,
    gfsFrames: outputSteps + 1,
    firstGfsLeadHours: -3,
    lastGfsLeadHours: horizonHours - 3,
  };
}

export { metadata };
