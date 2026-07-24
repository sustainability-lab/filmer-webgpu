#!/usr/bin/env node
/**
 * Add the 24-bit sign-magnitude descriptor required by current NCEP GFS
 * Template 5.3 fields. grib-js handles one-, two-, and four-byte descriptors,
 * but otherwise falls back to one byte and misaligns the packed bitstream.
 */

import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const binaryPath = require.resolve("grib-js/lib/BinaryDataView");
const parserPath = require.resolve("grib-js/lib/parser");

let binarySource = readFileSync(binaryPath, "utf8");
if (!binarySource.includes("grib24()")) {
  const insertionPoint = "    grib32() {\n";
  if (!binarySource.includes(insertionPoint)) {
    throw new Error("Unsupported grib-js BinaryDataView layout");
  }
  binarySource = binarySource.replace(
    insertionPoint,
    `    grib24() {
        const raw = (this.uint8() << 16) | (this.uint8() << 8) | this.uint8();
        const sign = raw >>> 23;
        const value = raw & 0x7fffff;
        return 1 === sign ? -value : value
    }
${insertionPoint}`,
  );
  writeFileSync(binaryPath, binarySource);
}

let parserSource = readFileSync(parserPath, "utf8");
const originalDescriptor = `        let gribSigned = 'grib8';
        if (2 == descriptorSpatial) {
          gribSigned = 'grib16'
        }
`;
const fixedDescriptor = `        const gribSigned = 'grib' + (descriptorSpatial * 8);
        if (![1, 2, 3, 4].includes(descriptorSpatial)) {
          throw new Error('Unsupported spatial descriptor width: ' + descriptorSpatial);
        }
`;
if (parserSource.includes(originalDescriptor)) {
  parserSource = parserSource.replace(originalDescriptor, fixedDescriptor);
  writeFileSync(parserPath, parserSource);
} else if (!parserSource.includes(fixedDescriptor)) {
  throw new Error("Unsupported grib-js parser layout");
}
