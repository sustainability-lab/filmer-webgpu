import { OUTPUT_CELLS } from "./preprocess";

export type VerificationMetric = {
  variable: string;
  unit: string;
  bias: number;
  mae: number;
  rmse: number;
  targetMin: number;
  targetMax: number;
  predictionMin: number;
  predictionMax: number;
};

export type ValidationManifest = {
  kind: "held-out-wrf-reference-not-observations";
  split: string;
  domain: string;
  domainId: number;
  resolutionKm: number;
  previousGfsTime: string;
  currentGfsTime: string;
  validTime: string;
  gridShape: [number, number];
  variables: Array<{ name: string; unit: string }>;
  semantics: {
    state: string;
    precipitation: string;
    scope: string;
  };
  artifacts: {
    input: { file: string; bytes: number; sha256: string; shape: number[] };
    projection: { file: string; bytes: number; sha256: string; shape: number[] };
    target: { file: string; bytes: number; sha256: string; shape: number[] };
    pythonPrediction: {
      file: string;
      bytes: number;
      sha256: string;
      shape: number[];
    };
  };
  metrics: VerificationMetric[];
};

export function verificationMetrics(
  prediction: Float32Array,
  target: Float32Array,
  variables: Array<{ name: string; unit: string }>,
): VerificationMetric[] {
  if (
    prediction.length !== variables.length * OUTPUT_CELLS ||
    target.length !== prediction.length
  ) {
    throw new Error("Verification tensors have unexpected shapes");
  }
  return variables.map(({ name, unit }, channel) => {
    const offset = channel * OUTPUT_CELLS;
    let sum = 0;
    let sumAbsolute = 0;
    let sumSquared = 0;
    let targetMin = Number.POSITIVE_INFINITY;
    let targetMax = Number.NEGATIVE_INFINITY;
    let predictionMin = Number.POSITIVE_INFINITY;
    let predictionMax = Number.NEGATIVE_INFINITY;
    for (let cell = 0; cell < OUTPUT_CELLS; cell += 1) {
      const predicted = prediction[offset + cell];
      const observed = target[offset + cell];
      const error = predicted - observed;
      sum += error;
      sumAbsolute += Math.abs(error);
      sumSquared += error * error;
      targetMin = Math.min(targetMin, observed);
      targetMax = Math.max(targetMax, observed);
      predictionMin = Math.min(predictionMin, predicted);
      predictionMax = Math.max(predictionMax, predicted);
    }
    return {
      variable: name,
      unit,
      bias: sum / OUTPUT_CELLS,
      mae: sumAbsolute / OUTPUT_CELLS,
      rmse: Math.sqrt(sumSquared / OUTPUT_CELLS),
      targetMin,
      targetMax,
      predictionMin,
      predictionMax,
    };
  });
}

export function differenceField(
  prediction: Float32Array,
  target: Float32Array,
): Float32Array {
  if (prediction.length !== target.length) {
    throw new Error("Prediction and reference arrays differ in length");
  }
  return Float32Array.from(
    prediction,
    (value, index) => value - target[index],
  );
}

export async function loadValidationManifest(): Promise<ValidationManifest> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}data/validation/manifest.json`,
    { cache: "no-cache" },
  );
  if (!response.ok) {
    throw new Error(
      `Validation manifest fetch failed with HTTP ${response.status}`,
    );
  }
  return response.json() as Promise<ValidationManifest>;
}

export async function loadValidationFloat(
  file: string,
): Promise<Float32Array> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}data/validation/${file}`,
  );
  if (!response.ok) {
    throw new Error(
      `Validation artifact ${file} failed with HTTP ${response.status}`,
    );
  }
  return new Float32Array(await response.arrayBuffer());
}
