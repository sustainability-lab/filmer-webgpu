import { OUTPUT_CELLS } from "./preprocess";

type Rgb = readonly [number, number, number];

export type PaperPalette =
  | "viridis"
  | "coolwarm"
  | "YlGnBu"
  | "Blues"
  | "YlOrRd";

export type VariableVisual = {
  shortName: string;
  label: string;
  sourceUnit: string;
  displayUnit: string;
  displayFactor: number;
  palette: PaperPalette;
};

// These are the exact variable/palette assignments in Balbir's paper plotting
// script. The sequential stops are the canonical Matplotlib/ColorBrewer
// anchors; interpolation is continuous in RGB for browser rendering.
export const VARIABLE_VISUALS: readonly VariableVisual[] = [
  {
    shortName: "T2",
    label: "2 m temperature",
    sourceUnit: "K",
    displayUnit: "K",
    displayFactor: 1,
    palette: "viridis",
  },
  {
    shortName: "U10",
    label: "10 m zonal wind",
    sourceUnit: "m s⁻¹",
    displayUnit: "m s⁻¹",
    displayFactor: 1,
    palette: "coolwarm",
  },
  {
    shortName: "V10",
    label: "10 m meridional wind",
    sourceUnit: "m s⁻¹",
    displayUnit: "m s⁻¹",
    displayFactor: 1,
    palette: "coolwarm",
  },
  {
    shortName: "Q2",
    label: "2 m specific humidity",
    sourceUnit: "kg kg⁻¹",
    displayUnit: "kg kg⁻¹",
    displayFactor: 1,
    palette: "YlGnBu",
  },
  {
    shortName: "PSFC",
    label: "Surface pressure",
    sourceUnit: "Pa",
    displayUnit: "hPa",
    displayFactor: 0.01,
    palette: "viridis",
  },
  {
    shortName: "PRECIP",
    label: "Precipitation",
    sourceUnit: "mm",
    displayUnit: "mm",
    displayFactor: 1,
    palette: "Blues",
  },
  {
    shortName: "WIND",
    label: "10 m wind speed",
    sourceUnit: "m s⁻¹",
    displayUnit: "m s⁻¹",
    displayFactor: 1,
    palette: "YlOrRd",
  },
  {
    shortName: "RH2",
    label: "2 m relative humidity",
    sourceUnit: "%",
    displayUnit: "%",
    displayFactor: 1,
    palette: "YlGnBu",
  },
] as const;

const PALETTE_STOPS: Record<PaperPalette, readonly Rgb[]> = {
  viridis: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 73, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [181, 222, 43],
    [253, 231, 37],
  ],
  coolwarm: [
    [59, 76, 192],
    [94, 125, 226],
    [135, 169, 252],
    [176, 203, 255],
    [221, 221, 221],
    [245, 196, 173],
    [238, 138, 110],
    [214, 82, 67],
    [180, 4, 38],
  ],
  YlGnBu: [
    [255, 255, 217],
    [237, 248, 177],
    [199, 233, 180],
    [127, 205, 187],
    [65, 182, 196],
    [29, 145, 192],
    [34, 94, 168],
    [37, 52, 148],
    [8, 29, 88],
  ],
  Blues: [
    [247, 251, 255],
    [222, 235, 247],
    [198, 219, 239],
    [158, 202, 225],
    [107, 174, 214],
    [66, 146, 198],
    [33, 113, 181],
    [8, 81, 156],
    [8, 48, 107],
  ],
  YlOrRd: [
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
    [128, 0, 38],
  ],
};

function interpolate(
  stops: readonly Rgb[],
  normalized: number,
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, normalized));
  const position = clamped * (stops.length - 1);
  const leftIndex = Math.min(stops.length - 2, Math.floor(position));
  const fraction = position - leftIndex;
  const left = stops[leftIndex];
  const right = stops[leftIndex + 1];
  return [
    Math.round(left[0] + (right[0] - left[0]) * fraction),
    Math.round(left[1] + (right[1] - left[1]) * fraction),
    Math.round(left[2] + (right[2] - left[2]) * fraction),
  ];
}

export function paperColor(
  palette: PaperPalette,
  normalized: number,
): [number, number, number] {
  return interpolate(PALETTE_STOPS[palette], normalized);
}

export function paletteCssGradient(palette: PaperPalette) {
  const stops = PALETTE_STOPS[palette]
    .map(
      ([red, green, blue], index) =>
        `rgb(${red} ${green} ${blue}) ${(index / (PALETTE_STOPS[palette].length - 1)) * 100}%`,
    )
    .join(", ");
  return `linear-gradient(90deg, ${stops})`;
}

function differenceColor(value: number): [number, number, number] {
  return paperColor("coolwarm", value);
}

export function displayValue(variableIndex: number, value: number) {
  return value * VARIABLE_VISUALS[variableIndex].displayFactor;
}

export function deriveDisplayFields(physical: Float32Array) {
  if (physical.length === 8 * OUTPUT_CELLS) return physical;
  if (physical.length !== 6 * OUTPUT_CELLS) {
    throw new Error("Display derivation expects six physical model outputs");
  }
  const display = new Float32Array(8 * OUTPUT_CELLS);
  display.set(physical);
  const windOffset = 6 * OUTPUT_CELLS;
  const rhOffset = 7 * OUTPUT_CELLS;
  for (let cell = 0; cell < OUTPUT_CELLS; cell += 1) {
    const temperatureKelvin = physical[cell];
    const zonalWind = physical[OUTPUT_CELLS + cell];
    const meridionalWind = physical[2 * OUTPUT_CELLS + cell];
    const specificHumidity = physical[3 * OUTPUT_CELLS + cell];
    const surfacePressure = physical[4 * OUTPUT_CELLS + cell];
    display[windOffset + cell] = Math.hypot(zonalWind, meridionalWind);

    // Same formula and constants as the paper plotting script.
    const temperatureCelsius = temperatureKelvin - 273.15;
    const vaporPressure =
      (surfacePressure * specificHumidity) /
      (0.622 + (1 - 0.622) * specificHumidity);
    const saturationPressure =
      610.94 *
      Math.exp(
        (17.625 * temperatureCelsius) /
          (temperatureCelsius + 243.04),
      );
    display[rhOffset + cell] = Math.max(
      0,
      Math.min(100, (vaporPressure / saturationPressure) * 100),
    );
  }
  return display;
}

export type RasterPixels = {
  pixels: Uint8ClampedArray;
  low: number;
  high: number;
};

export function createRasterPixels(
  values: Float32Array,
  variableIndex: number,
  mode: "field" | "difference",
  scaleValues?: Float32Array | null,
): RasterPixels {
  const visual = VARIABLE_VISUALS[variableIndex];
  const offset = variableIndex * OUTPUT_CELLS;
  const raw = Array.from(
    values.slice(offset, offset + OUTPUT_CELLS),
    (value) => value * visual.displayFactor,
  );
  const scaleRaw =
    mode === "field" && scaleValues
      ? Array.from(
          scaleValues.slice(offset, offset + OUTPUT_CELLS),
          (value) => value * visual.displayFactor,
        )
      : raw;
  const sorted = [...scaleRaw].sort((left, right) => left - right);
  let low: number;
  let high: number;
  if (mode === "difference") {
    const magnitudes = raw
      .map((value) => Math.abs(value))
      .sort((left, right) => left - right);
    const maximum =
      magnitudes[Math.floor(magnitudes.length * 0.98)] ||
      magnitudes.at(-1) ||
      1;
    low = -maximum;
    high = maximum;
  } else {
    low = sorted[Math.floor(sorted.length * 0.02)];
    high = sorted[Math.floor(sorted.length * 0.98)];
  }
  const span = high > low ? high - low : 1;
  const pixels = new Uint8ClampedArray(OUTPUT_CELLS * 4);
  for (let sourceIndex = 0; sourceIndex < OUTPUT_CELLS; sourceIndex += 1) {
    const normalized = Math.max(
      0,
      Math.min(1, (raw[sourceIndex] - low) / span),
    );
    const [red, green, blue] =
      mode === "difference"
        ? differenceColor(normalized)
        : paperColor(visual.palette, normalized);
    const sourceRow = Math.floor(sourceIndex / 99);
    const sourceColumn = sourceIndex % 99;
    // Matplotlib's `origin="lower"` places tensor row 0 at the southern edge.
    // Canvas row 0 is the top, so flip rows exactly once during rasterization.
    const destinationRow = 98 - sourceRow;
    const destination = (destinationRow * 99 + sourceColumn) * 4;
    pixels[destination] = red;
    pixels[destination + 1] = green;
    pixels[destination + 2] = blue;
    pixels[destination + 3] = 224;
  }
  return { pixels, low, high };
}
