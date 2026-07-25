import { describe, expect, it } from "vitest";
import { OUTPUT_CELLS, normalizedProjection } from "../src/lib/preprocess";
import {
  differenceField,
  verificationMetrics,
} from "../src/lib/verification";

describe("WRF comparison metrics", () => {
  it("computes signed bias, MAE, and RMSE per output channel", () => {
    const target = new Float32Array(2 * OUTPUT_CELLS);
    const prediction = new Float32Array(2 * OUTPUT_CELLS);
    prediction.fill(2, 0, OUTPUT_CELLS);
    prediction.fill(-3, OUTPUT_CELLS);
    const metrics = verificationMetrics(prediction, target, [
      { name: "A", unit: "u" },
      { name: "B", unit: "v" },
    ]);
    expect(metrics[0]).toMatchObject({ bias: 2, mae: 2, rmse: 2 });
    expect(metrics[1]).toMatchObject({ bias: -3, mae: 3, rmse: 3 });
  });

  it("constructs a signed FiLMeR-minus-WRF field", () => {
    expect(
      Array.from(
        differenceField(
          Float32Array.from([3, -1, 5]),
          Float32Array.from([1, 2, 8]),
        ),
      ),
    ).toEqual([2, -3, -3]);
  });
});

describe("resolution conditioning probe", () => {
  it("changes only the resolution coordinate while preserving fixed shape", () => {
    const timestamp = new Date("2025-01-01T06:00:00Z");
    const trained = normalizedProjection(1, timestamp, 27);
    const probe = normalizedProjection(1, timestamp, 1);
    expect(trained).toHaveLength(16);
    expect(probe).toHaveLength(16);
    const changed = Array.from(probe)
      .map((value, index) => (value === trained[index] ? null : index))
      .filter((index) => index !== null);
    expect(changed).toEqual([12]);
  });
});
