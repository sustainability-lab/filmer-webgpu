import * as ort from "onnxruntime-web/webgpu";
import {
  OUTPUT_CELLS,
  normalizedProjection,
  reconstructPhysical,
} from "./preprocess";

export type Backend = "webgpu" | "wasm";

export type Artifact = {
  file: string;
  url: string;
  bytes: number;
  sha256: string;
  precision: "fp16" | "fp32";
};

export type ModelManifest = {
  release: string;
  checkpointSha256: string;
  artifacts: {
    webgpu: Artifact;
    wasm: Artifact;
  };
};

export type ModelRun = {
  state: Float32Array;
  occurrence: Float32Array;
  intensity: Float32Array;
  physical: Float32Array;
  elapsedMilliseconds: number;
};

export type LoadProgress = {
  loaded: number;
  total: number;
  stage: "download" | "checksum" | "compile";
};

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function verifiedDownload(
  artifact: Artifact,
  onProgress: (progress: LoadProgress) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(artifact.url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Model fetch failed with HTTP ${response.status}`);
  }
  const total = Number(response.headers.get("content-length")) || artifact.bytes;
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming downloads are unavailable in this browser");
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ loaded, total, stage: "download" });
  }
  const model = new Uint8Array(loaded);
  let cursor = 0;
  chunks.forEach((chunk) => {
    model.set(chunk, cursor);
    cursor += chunk.byteLength;
  });
  onProgress({ loaded, total, stage: "checksum" });
  const actual = bytesToHex(await crypto.subtle.digest("SHA-256", model));
  if (actual !== artifact.sha256) {
    throw new Error(
      `Model checksum mismatch: expected ${artifact.sha256}, received ${actual}`,
    );
  }
  return model.buffer;
}

export function webGpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export class FilmerSession {
  readonly backend: Backend;
  readonly artifact: Artifact;
  private readonly session: ort.InferenceSession;

  private constructor(
    backend: Backend,
    artifact: Artifact,
    session: ort.InferenceSession,
  ) {
    this.backend = backend;
    this.artifact = artifact;
    this.session = session;
  }

  static async create(
    manifest: ModelManifest,
    requested: "auto" | Backend,
    onProgress: (progress: LoadProgress) => void,
  ): Promise<FilmerSession> {
    const candidates: Backend[] =
      requested === "auto"
        ? webGpuAvailable()
          ? ["webgpu", "wasm"]
          : ["wasm"]
        : [requested];
    let lastError: unknown;
    for (const backend of candidates) {
      try {
        const artifact = manifest.artifacts[backend];
        const buffer = await verifiedDownload(artifact, onProgress);
        onProgress({
          loaded: artifact.bytes,
          total: artifact.bytes,
          stage: "compile",
        });
        ort.env.wasm.numThreads = crossOriginIsolated
          ? Math.min(navigator.hardwareConcurrency || 1, 4)
          : 1;
        if (backend === "webgpu") {
          ort.env.webgpu.powerPreference = "high-performance";
        }
        const session = await ort.InferenceSession.create(buffer, {
          executionProviders: [backend],
          graphOptimizationLevel: "all",
        });
        return new FilmerSession(backend, artifact, session);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError instanceof Error
      ? lastError
      : new Error("Unable to initialize FiLMeR");
  }

  async run(
    input: Float32Array,
    domainId: number,
    forecastTimestamp: Date,
  ): Promise<ModelRun> {
    const projection = normalizedProjection(domainId, forecastTimestamp);
    const started = performance.now();
    const outputs = await this.session.run({
      input_data: new ort.Tensor("float32", input, [1, 70, 127, 137]),
      projection: new ort.Tensor("float32", projection, [1, 16]),
    });
    const elapsedMilliseconds = performance.now() - started;
    const state = outputs.state.data as Float32Array;
    const occurrence = outputs.precip_occurrence.data as Float32Array;
    const intensity = outputs.precip_intensity.data as Float32Array;
    if (
      state.length !== 5 * OUTPUT_CELLS ||
      occurrence.length !== OUTPUT_CELLS ||
      intensity.length !== OUTPUT_CELLS
    ) {
      throw new Error("Runtime returned unexpected output shapes");
    }
    return {
      state,
      occurrence,
      intensity,
      physical: reconstructPhysical(state, occurrence, intensity),
      elapsedMilliseconds,
    };
  }
}

export async function loadModelManifest(): Promise<ModelManifest> {
  const response = await fetch(`${import.meta.env.BASE_URL}models/manifest.json`);
  if (!response.ok) {
    throw new Error(`Model manifest fetch failed with HTTP ${response.status}`);
  }
  return response.json() as Promise<ModelManifest>;
}

export async function loadFloatFixture(file: string): Promise<Float32Array> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${file}`);
  if (!response.ok) {
    throw new Error(`Fixture fetch failed with HTTP ${response.status}`);
  }
  return new Float32Array(await response.arrayBuffer());
}

export function parityMetrics(actual: Float32Array, expected: Float32Array) {
  if (actual.length !== expected.length) {
    throw new Error("Parity arrays differ in length");
  }
  let maxAbs = 0;
  let sumAbs = 0;
  let sumSquared = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const difference = actual[index] - expected[index];
    const absolute = Math.abs(difference);
    maxAbs = Math.max(maxAbs, absolute);
    sumAbs += absolute;
    sumSquared += difference * difference;
  }
  return {
    maxAbs,
    meanAbs: sumAbs / actual.length,
    rmse: Math.sqrt(sumSquared / actual.length),
  };
}
