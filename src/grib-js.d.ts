declare module "grib-js/lib/parser" {
  export type GribField = {
    fields: Array<{
      data: number[];
      grid: {
        definition: {
          ni: number;
          nj: number;
        };
      };
    }>;
  };

  const parser: {
    parseDataView(data: ArrayBuffer): GribField[];
  };
  export default parser;
}

declare module "grib-js/lib/BinaryDataView" {
  export default class BinaryDataView {
    constructor(data: ArrayBuffer);
    read(type: string, offset?: number): number;
  }
}
