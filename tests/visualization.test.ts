import { describe, expect, it } from "vitest";
import { OUTPUT_CELLS } from "../src/lib/preprocess";
import {
  VARIABLE_VISUALS,
  createRasterPixels,
  deriveDisplayFields,
  displayValue,
  paperColor,
} from "../src/lib/visualization";

describe("paper-aligned meteorological rendering", () => {
  it("uses Balbir's paper palette assignment for every model output", () => {
    expect(VARIABLE_VISUALS.map((item) => item.palette)).toEqual([
      "viridis",
      "coolwarm",
      "coolwarm",
      "YlGnBu",
      "viridis",
      "Blues",
      "YlOrRd",
      "YlGnBu",
    ]);
  });

  it("displays surface pressure in the paper's hPa unit", () => {
    expect(VARIABLE_VISUALS[4].displayUnit).toBe("hPa");
    expect(displayValue(4, 101_325)).toBeCloseTo(1013.25, 6);
  });

  it("places tensor row zero at the southern canvas edge", () => {
    const values = new Float32Array(6 * OUTPUT_CELLS);
    for (let row = 0; row < 99; row += 1) {
      for (let column = 0; column < 99; column += 1) {
        values[row * 99 + column] = row;
      }
    }
    const { pixels } = createRasterPixels(values, 0, "field");
    const northWest = Array.from(pixels.slice(0, 3));
    const southWestOffset = 98 * 99 * 4;
    const southWest = Array.from(
      pixels.slice(southWestOffset, southWestOffset + 3),
    );
    expect(northWest).toEqual(paperColor("viridis", 1));
    expect(southWest).toEqual(paperColor("viridis", 0));
  });

  it("uses a zero-centred coolwarm scale for signed error", () => {
    const values = new Float32Array(6 * OUTPUT_CELLS);
    values.fill(-2, 0, Math.floor(OUTPUT_CELLS / 2));
    values.fill(2, Math.floor(OUTPUT_CELLS / 2), OUTPUT_CELLS);
    const raster = createRasterPixels(values, 0, "difference");
    expect(raster.low).toBe(-2);
    expect(raster.high).toBe(2);
  });

  it("shares the WRF target scale with the prediction comparison", () => {
    const prediction = new Float32Array(6 * OUTPUT_CELLS);
    const target = new Float32Array(6 * OUTPUT_CELLS);
    prediction.fill(500, 0, OUTPUT_CELLS);
    target.fill(270, 0, OUTPUT_CELLS);
    target[0] = 260;
    target[OUTPUT_CELLS - 1] = 280;
    const raster = createRasterPixels(prediction, 0, "field", target);
    expect(raster.low).toBe(270);
    expect(raster.high).toBe(270);
  });

  it("keeps one scale across a sequence of forecast frames", () => {
    const first = new Float32Array(8 * OUTPUT_CELLS);
    const second = new Float32Array(8 * OUTPUT_CELLS);
    first.fill(1, 0, OUTPUT_CELLS);
    second.fill(9, 0, OUTPUT_CELLS);
    const sequence = new Float32Array(first.length + second.length);
    sequence.set(first);
    sequence.set(second, first.length);

    const raster = createRasterPixels(first, 0, "field", sequence);

    expect(raster.low).toBe(1);
    expect(raster.high).toBe(9);
  });

  it("derives paper wind-speed and RH2 fields without changing model outputs", () => {
    const physical = new Float32Array(6 * OUTPUT_CELLS);
    physical.fill(300, 0, OUTPUT_CELLS);
    physical.fill(3, OUTPUT_CELLS, 2 * OUTPUT_CELLS);
    physical.fill(4, 2 * OUTPUT_CELLS, 3 * OUTPUT_CELLS);
    physical.fill(0.01, 3 * OUTPUT_CELLS, 4 * OUTPUT_CELLS);
    physical.fill(100_000, 4 * OUTPUT_CELLS, 5 * OUTPUT_CELLS);
    const display = deriveDisplayFields(physical);
    expect(display).toHaveLength(8 * OUTPUT_CELLS);
    expect(display[6 * OUTPUT_CELLS]).toBe(5);
    expect(display[7 * OUTPUT_CELLS]).toBeGreaterThan(40);
    expect(display[7 * OUTPUT_CELLS]).toBeLessThan(50);
    expect(display.slice(0, 6 * OUTPUT_CELLS)).toEqual(physical);
  });
});
