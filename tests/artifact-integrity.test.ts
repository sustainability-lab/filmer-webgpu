import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { reconstructPhysical } from "../src/lib/preprocess";
import { verificationMetrics } from "../src/lib/verification";

const dataDirectory = resolve("public/data");
const require = createRequire(import.meta.url);

function bytes(file: string) {
  return readFileSync(resolve(dataDirectory, file));
}

function sha256(file: string) {
  return createHash("sha256").update(bytes(file)).digest("hex");
}

function f32(file: string) {
  const source = bytes(file);
  const copy = source.buffer.slice(
    source.byteOffset,
    source.byteOffset + source.byteLength,
  );
  return new Float32Array(copy);
}

describe("committed parity fixture", () => {
  const fixture = JSON.parse(
    readFileSync(resolve(dataDirectory, "fixture.json"), "utf8"),
  ) as {
    inputs: Record<string, { file: string; shape: number[] }>;
    outputs: Record<string, { file: string; shape: number[] }>;
    sha256: Record<string, string>;
  };

  it("matches every published fixture checksum and shape", () => {
    const artifacts = { ...fixture.inputs, ...fixture.outputs };
    for (const [name, artifact] of Object.entries(artifacts)) {
      expect(sha256(artifact.file), name).toBe(fixture.sha256[name]);
      expect(bytes(artifact.file).byteLength, name).toBe(
        artifact.shape.reduce((product, dimension) => product * dimension, 1) *
          4,
      );
    }
  });

  it("reproduces the Python physical-unit output within one float32 rounding step", () => {
    const actual = reconstructPhysical(
      f32(fixture.outputs.state.file),
      f32(fixture.outputs.occurrence.file),
      f32(fixture.outputs.intensity.file),
    );
    const expected = f32(fixture.outputs.physical.file);
    expect(actual).toHaveLength(expected.length);
    let maximumAbsoluteError = 0;
    for (let index = 0; index < actual.length; index += 1) {
      maximumAbsoluteError = Math.max(
        maximumAbsoluteError,
        Math.abs(actual[index] - expected[index]),
      );
    }
    expect(maximumAbsoluteError).toBeLessThanOrEqual(0.0078125);
  });
});

describe("published static geography", () => {
  const staticDirectory = resolve(dataDirectory, "static");
  const manifest = JSON.parse(
    readFileSync(resolve(staticDirectory, "manifest.json"), "utf8"),
  ) as {
    static: Array<{
      file: string;
      bytes: number;
      sha256: string;
      month: number;
      shape: number[];
    }>;
    grids: Array<{
      domainId: number;
      shape: number[];
      latitude: { file: string; bytes: number; sha256: string };
      longitude: { file: string; bytes: number; sha256: string };
    }>;
  };

  it("contains all twelve monthly tensors with verified bytes and hashes", () => {
    expect(manifest.static.map((artifact) => artifact.month)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
    for (const artifact of manifest.static) {
      const source = readFileSync(resolve(staticDirectory, artifact.file));
      expect(source.byteLength, artifact.file).toBe(artifact.bytes);
      expect(
        createHash("sha256").update(source).digest("hex"),
        artifact.file,
      ).toBe(artifact.sha256);
      expect(artifact.shape).toEqual([30, 127, 137]);
    }
  });

  it("contains coordinate grids for every trained domain", () => {
    expect(manifest.grids.map((grid) => grid.domainId)).toEqual([1, 2, 3, 4]);
    for (const grid of manifest.grids) {
      expect(grid.shape).toEqual([99, 99]);
      for (const coordinate of [grid.latitude, grid.longitude]) {
        const source = readFileSync(resolve(staticDirectory, coordinate.file));
        expect(source.byteLength, coordinate.file).toBe(coordinate.bytes);
        expect(
          createHash("sha256").update(source).digest("hex"),
          coordinate.file,
        ).toBe(coordinate.sha256);
      }
    }
  });
});

describe("browser model artifacts", () => {
  const modelDirectory = resolve("public/models");
  const manifest = JSON.parse(
    readFileSync(resolve(modelDirectory, "manifest.json"), "utf8"),
  ) as {
    artifacts: Record<
      string,
      {
        bytes: number;
        sha256: string;
        parts: Array<{ file: string; bytes: number; sha256: string }>;
      }
    >;
  };

  it("reassembles each same-origin model to the release checksum", () => {
    for (const [runtime, artifact] of Object.entries(manifest.artifacts)) {
      const hash = createHash("sha256");
      let totalBytes = 0;
      for (const part of artifact.parts) {
        const source = readFileSync(resolve(modelDirectory, part.file));
        expect(source.byteLength, part.file).toBe(part.bytes);
        expect(
          createHash("sha256").update(source).digest("hex"),
          part.file,
        ).toBe(part.sha256);
        hash.update(source);
        totalBytes += source.byteLength;
      }
      expect(totalBytes, runtime).toBe(artifact.bytes);
      expect(hash.digest("hex"), runtime).toBe(artifact.sha256);
    }
  });
});

describe("browser dependency patches", () => {
  it("requests the adapter storage-buffer limit needed by FiLMeR Concat kernels", () => {
    const webGpuBundle = require.resolve("onnxruntime-web/webgpu");
    const source = readFileSync(webGpuBundle, "utf8");
    expect(source).toContain(
      "maxStorageBuffersPerShaderStage",
    );
  });
});

describe("held-out WRF verification fixture", () => {
  const validationDirectory = resolve(dataDirectory, "validation");
  const manifest = JSON.parse(
    readFileSync(resolve(validationDirectory, "manifest.json"), "utf8"),
  ) as {
    kind: string;
    domain: string;
    gridShape: number[];
    variables: Array<{ name: string; unit: string }>;
    artifacts: Record<
      string,
      { file: string; bytes: number; sha256: string; shape: number[] }
    >;
    metrics: Array<{ variable: string; bias: number; mae: number; rmse: number }>;
  };

  function validationF32(file: string) {
    const source = readFileSync(resolve(validationDirectory, file));
    const copy = source.buffer.slice(
      source.byteOffset,
      source.byteOffset + source.byteLength,
    );
    return new Float32Array(copy);
  }

  it("publishes hash-verified input, target, and Python prediction tensors", () => {
    expect(manifest.kind).toBe("held-out-wrf-reference-not-observations");
    expect(manifest.domain).toBe("d01");
    expect(manifest.gridShape).toEqual([99, 99]);
    for (const artifact of Object.values(manifest.artifacts)) {
      const source = readFileSync(resolve(validationDirectory, artifact.file));
      expect(source.byteLength, artifact.file).toBe(artifact.bytes);
      expect(
        createHash("sha256").update(source).digest("hex"),
        artifact.file,
      ).toBe(artifact.sha256);
      expect(source.byteLength, artifact.file).toBe(
        artifact.shape.reduce((product, dimension) => product * dimension, 1) *
          4,
      );
    }
  });

  it("recomputes every published Python-vs-WRF metric", () => {
    const actual = verificationMetrics(
      validationF32(manifest.artifacts.pythonPrediction.file),
      validationF32(manifest.artifacts.target.file),
      manifest.variables,
    );
    expect(actual).toHaveLength(6);
    actual.forEach((metric, index) => {
      expect(metric.variable).toBe(manifest.metrics[index].variable);
      expect(metric.bias).toBeCloseTo(manifest.metrics[index].bias, 7);
      expect(metric.mae).toBeCloseTo(manifest.metrics[index].mae, 7);
      expect(metric.rmse).toBeCloseTo(manifest.metrics[index].rmse, 7);
    });
  });
});
