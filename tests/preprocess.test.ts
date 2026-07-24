import { describe, expect, it } from "vitest";
import BinaryDataView from "grib-js/lib/BinaryDataView";
import {
  GRID_CELLS,
  normalizeGfsFrame,
  normalizeStaticFields,
  normalizedProjection,
  reconstructPhysical,
  sequenceRequirements,
} from "../src/lib/preprocess";
import {
  cropGlobalField,
  gfsObjectName,
  parseGfsIndex,
  selectGfsChannels,
} from "../src/lib/gfs";

describe("FiLMeR preprocessing", () => {
  it("maps a mean GFS field to zero", () => {
    const raw = new Float32Array(20 * GRID_CELLS);
    const means = [
      279.14801025390625,
      0.00737017672508955,
      96662.6796875,
      -0.07939903438091278,
      0.1585005223751068,
      29.957345962524414,
      0.00002888190465455409,
      281.20037841796875,
      274.9693908691406,
      267.9512939453125,
      253.70689392089844,
      1.3378373384475708,
      3.3072314262390137,
      6.474985599517822,
      0.14968198537826538,
      0.008808541111648083,
      -0.024266289547085762,
      0.004861405119299889,
      0.0027453640941530466,
      0.0010316260159015656,
    ];
    means.forEach((mean, channel) => {
      raw.fill(mean, channel * GRID_CELLS, (channel + 1) * GRID_CELLS);
    });
    const normalized = normalizeGfsFrame(raw);
    let maximum = Number.NEGATIVE_INFINITY;
    let minimum = Number.POSITIVE_INFINITY;
    normalized.forEach((value) => {
      maximum = Math.max(maximum, value);
      minimum = Math.min(minimum, value);
    });
    expect(maximum).toBeLessThan(1e-5);
    expect(minimum).toBeGreaterThan(-1e-5);
  });

  it("preserves static categorical channels and standardizes continuous ones", () => {
    const raw = new Float32Array(30 * GRID_CELLS);
    raw.fill(1, 0, GRID_CELLS);
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      raw[GRID_CELLS + cell] = cell;
    }
    raw.fill(0.25, 9 * GRID_CELLS);
    const normalized = normalizeStaticFields(raw);
    expect(normalized[0]).toBe(1);
    expect(normalized[9 * GRID_CELLS]).toBe(0.25);
    let mean = 0;
    for (let cell = 0; cell < GRID_CELLS; cell += 1) {
      mean += normalized[GRID_CELLS + cell];
    }
    expect(mean / GRID_CELLS).toBeCloseTo(0, 5);
  });

  it("reproduces the d01 projection vector shape", () => {
    const projection = normalizedProjection(
      1,
      new Date("2025-01-01T06:00:00Z"),
    );
    expect(projection).toHaveLength(16);
    expect(Array.from(projection).every(Number.isFinite)).toBe(true);
  });

  it("requires 33 GFS frames for a conditional 96-hour product", () => {
    expect(sequenceRequirements(96)).toEqual({
      outputSteps: 32,
      gfsFrames: 33,
      firstGfsLeadHours: -3,
      lastGfsLeadHours: 93,
    });
  });

  it("rejects hourly and zero-length horizons", () => {
    expect(() => sequenceRequirements(1)).toThrow(
      "positive multiple of three",
    );
    expect(() => sequenceRequirements(0)).toThrow(
      "positive multiple of three",
    );
  });

  it("uses occurrence logits and expm1 intensity for precipitation", () => {
    const state = new Float32Array(5 * 99 * 99);
    const occurrence = new Float32Array(99 * 99);
    const intensity = new Float32Array(99 * 99);
    occurrence[0] = 1;
    occurrence[1] = -1;
    intensity[0] = Math.log(3);
    intensity[1] = Math.log(3);
    const physical = reconstructPhysical(state, occurrence, intensity);
    const offset = 5 * 99 * 99;
    expect(physical[offset]).toBeCloseTo(2, 5);
    expect(physical[offset + 1]).toBe(0);
  });
});

describe("operational GFS acquisition", () => {
  it("decodes the 24-bit signed descriptors used by NCEP complex packing", () => {
    const positive = new BinaryDataView(
      Uint8Array.from([0x00, 0x00, 0x01]).buffer,
    );
    const negative = new BinaryDataView(
      Uint8Array.from([0x80, 0x00, 0x01]).buffer,
    );
    expect(positive.read("grib24")).toBe(1);
    expect(negative.read("grib24")).toBe(-1);
  });

  it("builds the public GFS object name", () => {
    expect(gfsObjectName(new Date("2026-07-24T06:00:00Z"), 93)).toBe(
      "gfs.20260724/06/atmos/gfs.t06z.pgrb2.0p25.f093",
    );
  });

  it("rejects forecast leads outside the published 3-hour sequence", () => {
    const cycle = new Date("2026-07-24T06:00:00Z");
    expect(() => gfsObjectName(cycle, -3)).toThrow();
    expect(() => gfsObjectName(cycle, 1)).toThrow();
    expect(() => gfsObjectName(cycle, 387)).toThrow();
  });

  it("selects the exact checkpoint channel order from an index", () => {
    const variables = [
      ["TMP", "2 m above ground"],
      ["SPFH", "2 m above ground"],
      ["PRES", "surface"],
      ["UGRD", "10 m above ground"],
      ["VGRD", "10 m above ground"],
      ["TCDC", "entire atmosphere"],
      ["PRATE", "surface"],
      ["TMP", "1000 mb"],
      ["TMP", "850 mb"],
      ["TMP", "700 mb"],
      ["TMP", "500 mb"],
      ["UGRD", "850 mb"],
      ["UGRD", "700 mb"],
      ["UGRD", "500 mb"],
      ["VGRD", "850 mb"],
      ["VGRD", "700 mb"],
      ["VGRD", "500 mb"],
      ["SPFH", "850 mb"],
      ["SPFH", "700 mb"],
      ["SPFH", "500 mb"],
    ];
    const lines = variables
      .map(
        ([variable, level], index) =>
          `${index + 1}:${index * 100}:d=2026072406:${variable}:${level}:anl:`,
      )
      .concat(["21:2000:d=2026072406:HGT:surface:anl:"])
      .join("\n");
    expect(
      selectGfsChannels(parseGfsIndex(lines)).map((record) => [
        record.variable,
        record.level,
      ]),
    ).toEqual(variables);
  });

  it("crops the exact 127x137 training window", () => {
    const global = new Array<number>(721 * 1440).fill(0);
    global[214 * 1440 + 264] = 12;
    global[340 * 1440 + 400] = 34;
    const crop = cropGlobalField(global);
    expect(crop.length).toBe(127 * 137);
    expect(crop[0]).toBe(12);
    expect(crop.at(-1)).toBe(34);
  });

  it("rejects a field that is not the exact GFS 0.25-degree grid", () => {
    expect(() => cropGlobalField(new Array<number>(100).fill(0))).toThrow(
      "721x1440",
    );
  });
});
