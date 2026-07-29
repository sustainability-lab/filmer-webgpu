import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { geoMercator, geoPath } from "d3-geo";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import { domainById } from "../lib/preprocess";
import {
  VARIABLE_VISUALS,
  createRasterPixels,
  paletteCssGradient,
} from "../lib/visualization";

type Props = {
  values: Float32Array | null;
  variableIndex: number;
  domainId: number;
  unit: string;
  mode?: "field" | "difference";
  scaleValues?: Float32Array | null;
};

type ViewTransform = {
  scale: number;
  x: number;
  y: number;
};

const SURVEY_OF_INDIA_SOURCE =
  "https://surveyofindia.gov.in/pages/outline-maps-of-india";
const VIEW_WIDTH = 720;
const VIEW_HEIGHT = 560;

function d3CompatibleWinding(
  feature: Feature<Geometry>,
): Feature<Geometry> {
  // Survey of India GeoJSON follows RFC 7946 (counter-clockwise exterior
  // rings). d3-geo uses the opposite winding for sub-hemisphere polygons.
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

export function rasterVisualization(
  values: Float32Array,
  variableIndex: number,
  mode: "field" | "difference",
  scaleValues?: Float32Array | null,
) {
  const { pixels, low, high } = createRasterPixels(
    values,
    variableIndex,
    mode,
    scaleValues,
  );
  const canvas = document.createElement("canvas");
  canvas.width = 99;
  canvas.height = 99;
  const context = canvas.getContext("2d");
  if (!context) return { url: "", low, high };
  const image = context.createImageData(99, 99);
  image.data.set(pixels);
  context.putImageData(image, 0, 0);
  return { url: canvas.toDataURL("image/png"), low, high };
}

function formatScaleValue(value: number, unit: string) {
  if (Math.abs(value) >= 10_000) return value.toFixed(0);
  if (Math.abs(value) < 0.1 && unit !== "mm") return value.toExponential(1);
  return value.toFixed(2);
}

export function ForecastMap({
  values,
  variableIndex,
  domainId,
  unit,
  mode = "field",
  scaleValues = null,
}: Props) {
  const [geometry, setGeometry] = useState<Feature<Geometry> | null>(null);
  const [boundaryError, setBoundaryError] = useState<string | null>(null);
  const [view, setView] = useState<ViewTransform>({ scale: 1, x: 0, y: 0 });
  const drag = useRef<{
    pointerId: number;
    clientX: number;
    clientY: number;
    originX: number;
    originY: number;
  } | null>(null);

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

  useEffect(() => {
    setView({ scale: 1, x: 0, y: 0 });
  }, [domainId]);

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
  const visual = VARIABLE_VISUALS[variableIndex];
  const [latMin, lonMin, latMax, lonMax] = selected.paperBounds;
  const northWest = projection?.([lonMin, latMax]) ?? [0, 0];
  const southEast = projection?.([lonMax, latMin]) ?? [0, 0];
  const raster = useMemo(
    () =>
      values
        ? rasterVisualization(values, variableIndex, mode, scaleValues)
        : { url: "", low: 0, high: 0 },
    [values, variableIndex, mode, scaleValues],
  );

  function zoomAt(nextScale: number, pointX = 360, pointY = 280) {
    setView((current) => {
      const scale = Math.max(1, Math.min(8, nextScale));
      const worldX = (pointX - current.x) / current.scale;
      const worldY = (pointY - current.y) / current.scale;
      return {
        scale,
        x: pointX - worldX * scale,
        y: pointY - worldY * scale,
      };
    });
  }

  function onWheel(event: ReactWheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    const pointX =
      ((event.clientX - bounds.left) / bounds.width) * VIEW_WIDTH;
    const pointY =
      ((event.clientY - bounds.top) / bounds.height) * VIEW_HEIGHT;
    zoomAt(view.scale * (event.deltaY < 0 ? 1.22 : 1 / 1.22), pointX, pointY);
  }

  function onPointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      originX: view.x,
      originY: view.y,
    };
  }

  function onPointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    setView((current) => ({
      ...current,
      x:
        drag.current!.originX +
        ((event.clientX - drag.current!.clientX) / bounds.width) * VIEW_WIDTH,
      y:
        drag.current!.originY +
        ((event.clientY - drag.current!.clientY) / bounds.height) * VIEW_HEIGHT,
    }));
  }

  function onPointerUp(event: ReactPointerEvent<SVGSVGElement>) {
    if (drag.current?.pointerId === event.pointerId) {
      drag.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div className="map-shell">
      <svg
        aria-label={`Regional map for ${selected.name}. Drag to pan and use the wheel or zoom controls to magnify.`}
        className="h-full w-full cursor-grab active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        role="img"
        style={{ touchAction: "none" }}
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
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
        <rect width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="#e8e4da" />
        <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
          {geometry && path ? (
            <path
              d={path(geometry) ?? ""}
              fill="#f7f5ee"
              stroke="#373a36"
              strokeWidth={1.5 / view.scale}
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
          {raster.url && projection ? (
            <image
              href={raster.url}
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
                fill={
                  raster.url ? "url(#grid)" : "rgba(183, 103, 55, 0.08)"
                }
                stroke="#a6552c"
                strokeWidth={3 / view.scale}
              />
            </>
          ) : null}
        </g>
      </svg>

      <div className="map-zoom-controls" aria-label="Map navigation controls">
        <button
          aria-label="Zoom in"
          onClick={() => zoomAt(view.scale * 1.5)}
          type="button"
        >
          +
        </button>
        <button
          aria-label="Zoom out"
          disabled={view.scale <= 1}
          onClick={() => zoomAt(view.scale / 1.5)}
          type="button"
        >
          −
        </button>
        <button
          aria-label="Reset map view"
          className="map-reset"
          onClick={() => setView({ scale: 1, x: 0, y: 0 })}
          type="button"
        >
          reset
        </button>
      </div>

      {values ? (
        <div className="map-colorbar" aria-label={`Color scale in ${unit}`}>
          <div className="map-colorbar-heading">
            <span>
              {mode === "difference"
                ? "FiLMeR − WRF · coolwarm"
                : `${visual.shortName} · ${visual.palette}`}
            </span>
            <strong>{unit}</strong>
          </div>
          <div
            className={`map-colorbar-gradient ${
              mode === "difference" ? "map-colorbar-difference" : ""
            }`}
            style={{
              background: paletteCssGradient(
                mode === "difference" ? "coolwarm" : visual.palette,
              ),
            }}
          />
          <div className="map-colorbar-labels">
            <span>{formatScaleValue(raster.low, unit)}</span>
            {mode === "difference" ? <span>0</span> : null}
            <span>{formatScaleValue(raster.high, unit)}</span>
          </div>
        </div>
      ) : null}

      <div className="map-domain-badge">
        {selected.code.toUpperCase()} · {selected.resolutionKm} KM · 99 × 99
      </div>
      <div
        aria-label="North is up"
        className="map-orientation"
      >
        <strong>N ↑</strong>
      </div>
      <a
        className="map-attribution"
        href={SURVEY_OF_INDIA_SOURCE}
        rel="noreferrer"
        target="_blank"
      >
        Boundary: Survey of India · 1:16M · 2026
      </a>
      {!values ? (
        <div className="map-empty-state">
          Select a domain and variable, then run the forecast.
        </div>
      ) : null}
    </div>
  );
}
