import { useEffect, useMemo, useState } from "react";
import { geoMercator, geoPath } from "d3-geo";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import { metadata, OUTPUT_CELLS, domainById } from "../lib/preprocess";

type Props = {
  values: Float32Array | null;
  variableIndex: number;
  domainId: number;
};

const SURVEY_OF_INDIA_SOURCE =
  "https://surveyofindia.gov.in/pages/outline-maps-of-india";

function d3CompatibleWinding(
  feature: Feature<Geometry>,
): Feature<Geometry> {
  // Survey of India GeoJSON follows RFC 7946 (counter-clockwise exterior
  // rings). d3-geo's spherical path convention uses the opposite winding for
  // polygons smaller than a hemisphere, so reverse every ring at render time.
  if (feature.geometry.type === "Polygon") {
    return {
      ...feature,
      geometry: {
        ...feature.geometry,
        coordinates: feature.geometry.coordinates.map((ring) =>
          [...ring].reverse(),
        ),
      },
    };
  }
  if (feature.geometry.type === "MultiPolygon") {
    return {
      ...feature,
      geometry: {
        ...feature.geometry,
        coordinates: feature.geometry.coordinates.map((polygon) =>
          polygon.map((ring) => [...ring].reverse()),
        ),
      },
    };
  }
  return feature;
}

function rasterDataUrl(values: Float32Array, variableIndex: number) {
  const offset = variableIndex * OUTPUT_CELLS;
  const channel = Array.from(values.slice(offset, offset + OUTPUT_CELLS)).sort(
    (left, right) => left - right,
  );
  const low = channel[Math.floor(channel.length * 0.02)];
  const high = channel[Math.floor(channel.length * 0.98)];
  const scale = high > low ? high - low : 1;
  const canvas = document.createElement("canvas");
  canvas.width = 99;
  canvas.height = 99;
  const context = canvas.getContext("2d");
  if (!context) return "";
  const image = context.createImageData(99, 99);
  for (let index = 0; index < OUTPUT_CELLS; index += 1) {
    const normalized = Math.max(
      0,
      Math.min(1, (values[offset + index] - low) / scale),
    );
    const destination = index * 4;
    image.data[destination] = 238 - Math.round(normalized * 189);
    image.data[destination + 1] = 232 - Math.round(normalized * 122);
    image.data[destination + 2] = 214 - Math.round(normalized * 42);
    image.data[destination + 3] = 224;
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

export function ForecastMap({ values, variableIndex, domainId }: Props) {
  const [geometry, setGeometry] = useState<Feature<Geometry> | null>(null);
  const [boundaryError, setBoundaryError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    fetch(`${import.meta.env.BASE_URL}data/india-outline-soi.geojson`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Boundary request failed (${response.status})`);
        }
        return (await response.json()) as FeatureCollection<Geometry>;
      })
      .then((collection) => {
        if (!collection.features[0]) {
          throw new Error("Boundary file contains no features");
        }
        if (current) {
          setGeometry(d3CompatibleWinding(collection.features[0]));
        }
      })
      .catch((loadError: unknown) => {
        if (current) {
          setBoundaryError(
            loadError instanceof Error
              ? loadError.message
              : "Unable to load the official boundary",
          );
        }
      });
    return () => {
      current = false;
    };
  }, []);

  const projection = useMemo(
    () =>
      geometry
        ? geoMercator().fitExtent(
            [
              [44, 34],
              [676, 504],
            ],
            geometry,
          )
        : null,
    [geometry],
  );
  const path = projection ? geoPath(projection) : null;
  const selected = domainById(domainId);
  const [latMin, lonMin, latMax, lonMax] = selected.paperBounds;
  const northWest = projection?.([lonMin, latMax]) ?? [0, 0];
  const southEast = projection?.([lonMax, latMin]) ?? [0, 0];
  const raster = useMemo(
    () => (values ? rasterDataUrl(values, variableIndex) : ""),
    [values, variableIndex],
  );

  return (
    <div className="map-shell">
      <svg
        aria-label={`Regional map for ${selected.name}`}
        className="h-full w-full"
        role="img"
        viewBox="0 0 720 560"
      >
        <defs>
          <clipPath id="india-clip">
            {geometry && path ? <path d={path(geometry) ?? ""} /> : null}
          </clipPath>
          <pattern
            id="grid"
            width="14"
            height="14"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 14 0 L 0 0 0 14"
              fill="none"
              stroke="#8c897e"
              strokeOpacity="0.18"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width="720" height="560" fill="#e8e4da" />
        {geometry && path ? (
          <path
            d={path(geometry) ?? ""}
            fill="#f7f5ee"
            stroke="#373a36"
            strokeWidth="1.5"
          />
        ) : (
          <text
            x="360"
            y="270"
            fill="#686a64"
            fontFamily="Geist Mono, monospace"
            fontSize="13"
            textAnchor="middle"
          >
            {boundaryError ?? "Loading Survey of India boundary…"}
          </text>
        )}
        {raster && projection ? (
          <image
            href={raster}
            x={northWest[0]}
            y={northWest[1]}
            width={southEast[0] - northWest[0]}
            height={southEast[1] - northWest[1]}
            opacity="0.9"
            preserveAspectRatio="none"
            clipPath={domainId === 1 ? "url(#india-clip)" : undefined}
          />
        ) : null}
        {projection ? (
          <>
            <rect
              x={northWest[0]}
              y={northWest[1]}
              width={southEast[0] - northWest[0]}
              height={southEast[1] - northWest[1]}
              fill={raster ? "url(#grid)" : "rgba(183, 103, 55, 0.08)"}
              stroke="#a6552c"
              strokeWidth="3"
            />
            {metadata.domains
              .filter((domain) => domain.id !== domainId)
              .map((domain) => {
                const [dLatMin, dLonMin, dLatMax, dLonMax] =
                  domain.paperBounds;
                const start = projection([dLonMin, dLatMax]) ?? [0, 0];
                const end = projection([dLonMax, dLatMin]) ?? [0, 0];
                return (
                  <rect
                    key={domain.code}
                    x={start[0]}
                    y={start[1]}
                    width={end[0] - start[0]}
                    height={end[1] - start[1]}
                    fill="none"
                    stroke="#6d6e67"
                    strokeDasharray="4 5"
                    strokeWidth="1.5"
                  />
                );
              })}
          </>
        ) : null}
        <g transform="translate(34 512)">
          <rect width="252" height="24" rx="4" fill="#f7f5ee" opacity="0.92" />
          <text
            x="10"
            y="16"
            fill="#373a36"
            fontFamily="Geist Mono, monospace"
            fontSize="11"
          >
            {selected.code.toUpperCase()} · {selected.resolutionKm} KM · 99 × 99
          </text>
        </g>
      </svg>
      <a
        className="absolute bottom-2 right-3 bg-stone-100/90 px-2 py-1 font-mono text-[9px] text-stone-600 underline decoration-stone-400 underline-offset-2"
        href={SURVEY_OF_INDIA_SOURCE}
        rel="noreferrer"
        target="_blank"
      >
        Boundary: Survey of India · 1:16M · 2026
      </a>
      {!values ? (
        <div className="absolute inset-x-6 bottom-10 border border-stone-500/30 bg-stone-100/90 p-4 font-mono text-xs text-stone-700 backdrop-blur">
          Load the parity fixture or run the model to reveal a field.
        </div>
      ) : null}
    </div>
  );
}
