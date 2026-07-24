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
