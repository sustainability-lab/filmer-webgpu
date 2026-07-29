import { useEffect, useMemo, useRef, useState } from "react";
import {
  DownloadSimple,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  X,
} from "@phosphor-icons/react";
import { strToU8, zipSync } from "fflate";
import { ForecastMap } from "./components/ForecastMap";
import {
  fetchGfsFrame,
  loadOutputGrid,
  loadStaticMonth,
  previousCycleInput,
  type GfsProgress,
} from "./lib/gfs";
import {
  assembleInput,
  metadata,
  type Domain,
} from "./lib/preprocess";
import {
  FilmerSession,
  loadModelManifest,
  type LoadProgress,
  type ModelManifest,
} from "./lib/runtime";
import {
  VARIABLE_VISUALS,
  deriveDisplayFields,
} from "./lib/visualization";

const TIMESTAMP_OPTIONS = [1, 2, 4, 8] as const;

type RuntimeState =
  | "idle"
  | "loading"
  | "running"
  | "success"
  | "error";

type ForecastFrame = {
  validTime: string;
  physical: Float32Array;
  inferenceMilliseconds: number;
};

type RunTimings = {
  modelMilliseconds: number;
  modelCached: boolean;
  geographyMilliseconds: number;
  weatherMilliseconds: number;
  inferenceMilliseconds: number;
  totalMilliseconds: number;
};

function latestCompletedGfsCycle() {
  const now = new Date();
  const cycleHour = Math.floor(now.getUTCHours() / 6) * 6;
  const cycle = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      cycleHour,
    ),
  );
  cycle.setUTCHours(cycle.getUTCHours() - 6);
  return cycle;
}

function formatUtc(value: string | Date) {
  const date = typeof value === "string" ? new Date(value) : value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "UTC",
  }).format(date);
}

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDuration(milliseconds: number) {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(1)} s`;
}

function DomainButton({
  domain,
  selected,
  disabled,
  onSelect,
}: {
  domain: Domain;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      aria-pressed={selected}
      className={`domain-choice ${selected ? "domain-choice-selected" : ""}`}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <span>{domain.name}</span>
      <small>
        {domain.code.toUpperCase()} · {domain.resolutionKm} km
      </small>
    </button>
  );
}

export default function App() {
  const [domainId, setDomainId] = useState(1);
  const [variableIndex, setVariableIndex] = useState(0);
  const [outputSteps, setOutputSteps] = useState(2);
  const [manifest, setManifest] = useState<ModelManifest | null>(null);
  const [runtimeState, setRuntimeState] = useState<RuntimeState>("idle");
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState(
    "Ready to download the latest forecast",
  );
  const [error, setError] = useState<string | null>(null);
  const [frames, setFrames] = useState<ForecastFrame[]>([]);
  const [activeFrameIndex, setActiveFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cycleUsed, setCycleUsed] = useState<Date | null>(null);
  const [downloadedBytes, setDownloadedBytes] = useState(0);
  const [timings, setTimings] = useState<RunTimings | null>(null);
  const [walkthroughOpen, setWalkthroughOpen] = useState(false);
  const [outputGrid, setOutputGrid] = useState<{
    latitude: Float32Array;
    longitude: Float32Array;
  } | null>(null);
  const sessionRef = useRef<FilmerSession | null>(null);

  const selectedDomain = metadata.domains.find(
    (domain) => domain.id === domainId,
  )!;
  const selectedVariable = VARIABLE_VISUALS[variableIndex];
  const busy = runtimeState === "loading" || runtimeState === "running";
  const activeFrame = frames[activeFrameIndex] ?? null;
  const displayValues = useMemo(
    () => (activeFrame ? deriveDisplayFields(activeFrame.physical) : null),
    [activeFrame],
  );

  useEffect(() => {
    loadModelManifest()
      .then(setManifest)
      .catch((manifestError: unknown) => {
        setRuntimeState("error");
        setError(
          manifestError instanceof Error
            ? manifestError.message
            : "The model manifest could not be loaded",
        );
      });
  }, []);

  useEffect(() => {
    let current = true;
    setOutputGrid(null);
    loadOutputGrid(domainId)
      .then((grid) => {
        if (current) setOutputGrid(grid);
      })
      .catch(() => {
        // Coordinates are included in downloads when available. Their absence
        // does not prevent a forecast or the domain-aware map from running.
      });
    return () => {
      current = false;
    };
  }, [domainId]);

  useEffect(() => {
    if (!isPlaying || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setActiveFrameIndex((current) => (current + 1) % frames.length);
    }, 1100);
    return () => window.clearInterval(timer);
  }, [isPlaying, frames.length]);

  useEffect(() => {
    if (!walkthroughOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setWalkthroughOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [walkthroughOpen]);

  function selectDomain(id: number) {
    setDomainId(id);
    resetForecast();
  }

  function selectOutputSteps(steps: number) {
    setOutputSteps(steps);
    resetForecast();
  }

  function resetForecast() {
    setFrames([]);
    setActiveFrameIndex(0);
    setIsPlaying(false);
    setCycleUsed(null);
    setProgress(0);
    setRuntimeState("idle");
    setStatusText("Ready to download the latest forecast");
    setError(null);
    setTimings(null);
  }

  async function ensureSession() {
    if (sessionRef.current) {
      setProgress(30);
      setStatusText("Model ready");
      return sessionRef.current;
    }
    if (!manifest) throw new Error("The forecast model is still loading");
    setRuntimeState("loading");
    return FilmerSession.create(
      manifest,
      "wasm",
      (next: LoadProgress) => {
        if (next.stage === "download") {
          const fraction = next.total ? next.loaded / next.total : 0;
          setProgress(3 + fraction * 25);
          setStatusText(
            `Downloading model · ${formatBytes(next.loaded)} of ${formatBytes(next.total)}`,
          );
        } else if (next.stage === "checksum") {
          setProgress(29);
          setStatusText("Verifying model");
        } else {
          setProgress(30);
          setStatusText("Preparing model");
        }
      },
    ).then((session) => {
      sessionRef.current = session;
      return session;
    });
  }

  async function runLatestForecast() {
    setError(null);
    setFrames([]);
    setActiveFrameIndex(0);
    setIsPlaying(false);
    setDownloadedBytes(0);
    setTimings(null);
    setProgress(1);
    setRuntimeState("loading");

    try {
      const runStarted = performance.now();
      const runDomainId = domainId;
      const runOutputSteps = outputSteps;
      const gfsFrameCount = runOutputSteps + 1;
      const cycle = latestCompletedGfsCycle();
      const modelCached = Boolean(sessionRef.current);
      const modelStarted = performance.now();
      const session = await ensureSession();
      const modelMilliseconds = performance.now() - modelStarted;
      setCycleUsed(cycle);

      setRuntimeState("running");
      setProgress(32);
      setStatusText("Loading regional geography");
      const geographyStarted = performance.now();
      const normalizedStatic = await loadStaticMonth(
        cycle.getUTCMonth() + 1,
      );
      const geographyMilliseconds = performance.now() - geographyStarted;

      const prior = previousCycleInput(cycle);
      const requests = [
        { cycle: prior.cycle, forecastHour: prior.forecastHour },
        { cycle, forecastHour: 0 },
        ...Array.from({ length: runOutputSteps - 1 }, (_, index) => ({
          cycle,
          forecastHour: (index + 1) * 3,
        })),
      ];
      const gfs: Float32Array[] = [];
      let completedBytes = 0;
      const weatherStarted = performance.now();

      for (let frameIndex = 0; frameIndex < requests.length; frameIndex += 1) {
        let frameBytes = 0;
        const request = requests[frameIndex];
        const frame = await fetchGfsFrame(
          request.cycle,
          request.forecastHour,
          (next: GfsProgress) => {
            frameBytes = next.totalBytes;
            const fieldFraction =
              next.totalFields > 0
                ? next.completedFields / next.totalFields
                : 0;
            setProgress(
              34 + ((frameIndex + fieldFraction) / gfsFrameCount) * 52,
            );
            setDownloadedBytes(completedBytes + next.loadedBytes);
            setStatusText(
              `Downloading weather data · ${frameIndex + 1} of ${gfsFrameCount} · ${next.completedFields}/${next.totalFields} fields`,
            );
          },
          "noaa",
        );
        gfs.push(frame);
        completedBytes += frameBytes;
        setDownloadedBytes(completedBytes);
      }
      const weatherMilliseconds = performance.now() - weatherStarted;

      const nextFrames: ForecastFrame[] = [];
      let inferenceMilliseconds = 0;
      for (let step = 0; step < runOutputSteps; step += 1) {
        setProgress(88 + (step / runOutputSteps) * 12);
        setStatusText(`Running forecast · ${step + 1} of ${runOutputSteps}`);
        const validTime = new Date(
          cycle.getTime() + (step + 1) * 3 * 60 * 60 * 1000,
        );
        const result = await session.run(
          assembleInput(gfs[step], gfs[step + 1], normalizedStatic),
          runDomainId,
          validTime,
        );
        nextFrames.push({
          validTime: validTime.toISOString(),
          physical: result.physical,
          inferenceMilliseconds: result.elapsedMilliseconds,
        });
        inferenceMilliseconds += result.elapsedMilliseconds;
        setFrames([...nextFrames]);
        setActiveFrameIndex(nextFrames.length - 1);
        setProgress(88 + ((step + 1) / runOutputSteps) * 12);
      }

      const totalMilliseconds = performance.now() - runStarted;
      setTimings({
        modelMilliseconds,
        modelCached,
        geographyMilliseconds,
        weatherMilliseconds,
        inferenceMilliseconds,
        totalMilliseconds,
      });
      setProgress(100);
      setRuntimeState("success");
      setStatusText("Forecast ready");
    } catch (runError) {
      setRuntimeState("error");
      setStatusText("Forecast could not be completed");
      setError(
        runError instanceof Error
          ? runError.message
          : "An unexpected forecast error occurred",
      );
    }
  }

  function downloadResults() {
    if (!frames.length) return;
    const coordinates = outputGrid
      ? {
          latitudeFile: "grid-latitude.f32",
          longitudeFile: "grid-longitude.f32",
          shape: [99, 99],
        }
      : null;
    const outputManifest = {
      schemaVersion: 1,
      kind: "filmer-browser-forecast",
      model: "FiLMeR v1.0 Variant B",
      checkpointSha256: metadata.model.checkpointSha256,
      domain: selectedDomain.code,
      trainedResolutionKm: selectedDomain.resolutionKm,
      initialization: cycleUsed?.toISOString(),
      cadenceHours: 3,
      shapePerTimestamp: [6, 99, 99],
      dtype: "little-endian float32",
      variables: metadata.targets.outputVariables,
      units: metadata.targets.units,
      timestamps: frames.map((frame) => frame.validTime),
      files: frames.map(
        (frame, index) =>
          `forecast-${String(index + 1).padStart(2, "0")}-${frame.validTime
            .slice(0, 16)
            .replaceAll(":", "")}.f32`,
      ),
      backend: "wasm",
      semantics:
        "Conditional regional downscaling from paired 3-hourly NOAA GFS inputs.",
      coordinates,
    };
    const entries: Record<string, Uint8Array> = {
      "manifest.json": strToU8(JSON.stringify(outputManifest, null, 2)),
    };
    frames.forEach((frame, index) => {
      entries[outputManifest.files[index]] = new Uint8Array(
        frame.physical.buffer,
        frame.physical.byteOffset,
        frame.physical.byteLength,
      );
    });
    if (outputGrid) {
      entries["grid-latitude.f32"] = new Uint8Array(
        outputGrid.latitude.buffer,
        outputGrid.latitude.byteOffset,
        outputGrid.latitude.byteLength,
      );
      entries["grid-longitude.f32"] = new Uint8Array(
        outputGrid.longitude.buffer,
        outputGrid.longitude.byteOffset,
        outputGrid.longitude.byteLength,
      );
    }
    const archive = zipSync(entries, { level: 6 });
    const url = URL.createObjectURL(
      new Blob([archive], { type: "application/zip" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `filmer-${selectedDomain.code}-${cycleUsed
      ?.toISOString()
      .slice(0, 13)
      .replaceAll(":", "")}.zip`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return (
    <main className="app">
      <header className="app-header">
        <a className="brand" href={import.meta.env.BASE_URL}>
          <span className="brand-mark" aria-hidden="true">
            FM
          </span>
          <span>
            <strong>FiLMeR Forecast</strong>
            <small>Regional weather, in your browser</small>
          </span>
        </a>
        <span className={`app-status app-status-${runtimeState}`}>
          <i aria-hidden="true" />
          {runtimeState === "success"
            ? "Forecast ready"
            : busy
              ? "Working"
              : "Ready"}
        </span>
      </header>

      <div className="workspace">
        <aside className="controls">
          <div className="intro">
            <p className="eyebrow">Latest NOAA GFS</p>
            <h1>Run a regional weather forecast.</h1>
            <p>
              Choose a trained domain and variable. FiLMeR downloads what it
              needs and runs locally in this browser.
            </p>
            <button
              className="walkthrough-link"
              onClick={() => setWalkthroughOpen(true)}
              type="button"
            >
              <Play aria-hidden="true" size={14} weight="fill" />
              Watch the 40-second walkthrough
            </button>
          </div>

          <section className="step">
            <div className="step-heading">
              <span>1</span>
              <div>
                <h2>Select domain</h2>
                <p>Four regions used to train and evaluate FiLMeR.</p>
              </div>
            </div>
            <div className="domain-grid">
              {metadata.domains.map((domain) => (
                <DomainButton
                  disabled={busy}
                  domain={domain}
                  key={domain.code}
                  onSelect={() => selectDomain(domain.id)}
                  selected={domain.id === domainId}
                />
              ))}
            </div>
          </section>

          <section className="step">
            <div className="step-heading">
              <span>2</span>
              <div>
                <h2>Select variable</h2>
                <p>Choose the field to display on the map.</p>
              </div>
            </div>
            <label className="variable-select">
              <span className="sr-only">Forecast variable</span>
              <select
                disabled={busy}
                onChange={(event) =>
                  setVariableIndex(Number(event.target.value))
                }
                value={variableIndex}
              >
                {VARIABLE_VISUALS.map((variable, index) => (
                  <option key={variable.shortName} value={index}>
                    {variable.label} ({variable.displayUnit})
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="step step-run">
            <div className="step-heading">
              <span>3</span>
              <div>
                <h2>Run forecast</h2>
                <p>
                  {outputSteps * 3}-hour outlook · {outputSteps}{" "}
                  {outputSteps === 1 ? "timestamp" : "timestamps"} · 3-hourly
                </p>
              </div>
            </div>
            <label className="timestamp-select">
              <span>Forecast timestamps</span>
              <select
                disabled={busy}
                onChange={(event) =>
                  selectOutputSteps(Number(event.target.value))
                }
                value={outputSteps}
              >
                {TIMESTAMP_OPTIONS.map((steps) => (
                  <option key={steps} value={steps}>
                    {steps} {steps === 1 ? "timestamp" : "timestamps"} ·{" "}
                    {steps * 3} h
                  </option>
                ))}
              </select>
            </label>
            <button
              className="run-button"
              disabled={busy || !manifest}
              onClick={runLatestForecast}
              type="button"
            >
              {busy ? "Running forecast…" : "Download data & run"}
            </button>

            {(busy || runtimeState === "success" || error) && (
              <div className="run-progress" aria-live="polite">
                <div className="progress-copy">
                  <span>{statusText}</span>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div
                  aria-label={statusText}
                  aria-valuemax={100}
                  aria-valuemin={0}
                  aria-valuenow={Math.round(progress)}
                  className="progress-track"
                  role="progressbar"
                >
                  <span style={{ width: `${progress}%` }} />
                </div>
                {busy && downloadedBytes > 0 ? (
                  <small>{formatBytes(downloadedBytes)} weather data received</small>
                ) : null}
                {error ? <p className="error-message">{error}</p> : null}
                {runtimeState === "success" && timings ? (
                  <dl className="timing-grid" aria-label="Run timings">
                    <div>
                      <dt>Model</dt>
                      <dd>
                        {timings.modelCached
                          ? "cached"
                          : formatDuration(timings.modelMilliseconds)}
                      </dd>
                    </div>
                    <div>
                      <dt>Geography</dt>
                      <dd>{formatDuration(timings.geographyMilliseconds)}</dd>
                    </div>
                    <div>
                      <dt>Weather data</dt>
                      <dd>{formatDuration(timings.weatherMilliseconds)}</dd>
                    </div>
                    <div>
                      <dt>Inference</dt>
                      <dd>{formatDuration(timings.inferenceMilliseconds)}</dd>
                    </div>
                    <div>
                      <dt>Total</dt>
                      <dd>{formatDuration(timings.totalMilliseconds)}</dd>
                    </div>
                  </dl>
                ) : null}
              </div>
            )}
          </section>
        </aside>

        <section className="forecast">
          <div className="forecast-heading">
            <div>
              <p className="eyebrow">
                {activeFrame ? "Prediction" : "Forecast map"}
              </p>
              <h2>{selectedVariable.label}</h2>
              <p>
                {activeFrame
                  ? `${selectedDomain.name} · valid ${formatUtc(activeFrame.validTime)} UTC`
                  : `${selectedDomain.name} · ${selectedDomain.resolutionKm} km trained domain`}
              </p>
            </div>
            {frames.length ? (
              <button
                className="download-button"
                onClick={downloadResults}
                type="button"
              >
                <DownloadSimple aria-hidden="true" size={18} weight="bold" />
                Download results
              </button>
            ) : null}
          </div>

          <ForecastMap
            domainId={domainId}
            unit={selectedVariable.displayUnit}
            values={displayValues}
            variableIndex={variableIndex}
          />

          {frames.length > 1 ? (
            <div className="playback">
              <div className="playback-buttons">
                <button
                  aria-label="Previous timestamp"
                  onClick={() => {
                    setIsPlaying(false);
                    setActiveFrameIndex(
                      (current) => (current - 1 + frames.length) % frames.length,
                    );
                  }}
                  type="button"
                >
                  <SkipBack aria-hidden="true" size={17} weight="fill" />
                </button>
                <button
                  aria-label={isPlaying ? "Pause animation" : "Play animation"}
                  className="play-button"
                  onClick={() => setIsPlaying((current) => !current)}
                  type="button"
                >
                  {isPlaying ? (
                    <Pause aria-hidden="true" size={18} weight="fill" />
                  ) : (
                    <Play aria-hidden="true" size={18} weight="fill" />
                  )}
                </button>
                <button
                  aria-label="Next timestamp"
                  onClick={() => {
                    setIsPlaying(false);
                    setActiveFrameIndex(
                      (current) => (current + 1) % frames.length,
                    );
                  }}
                  type="button"
                >
                  <SkipForward aria-hidden="true" size={17} weight="fill" />
                </button>
              </div>
              <label className="timeline">
                <span>{formatUtc(frames[0].validTime)} UTC</span>
                <input
                  aria-label="Forecast timestamp"
                  max={frames.length - 1}
                  min={0}
                  onChange={(event) => {
                    setIsPlaying(false);
                    setActiveFrameIndex(Number(event.target.value));
                  }}
                  step={1}
                  type="range"
                  value={activeFrameIndex}
                />
                <span>{formatUtc(frames.at(-1)!.validTime)} UTC</span>
              </label>
              <strong className="frame-count">
                {activeFrameIndex + 1} / {frames.length}
              </strong>
            </div>
          ) : null}
        </section>
      </div>

      <footer>
        <span>FiLMeR v1.0 · conditional GFS downscaling · research prototype</span>
        <span className="footer-links">
          <a
            href={`${import.meta.env.BASE_URL}method.html`}
            rel="noreferrer"
            target="_blank"
          >
            Method &amp; validation
          </a>
          <a
            href="https://github.com/sustainability-lab/filmer-webgpu"
            rel="noreferrer"
            target="_blank"
          >
            Source
          </a>
        </span>
      </footer>

      {walkthroughOpen ? (
        <div
          aria-label="How to run FiLMeR"
          aria-modal="true"
          className="walkthrough-backdrop"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) {
              setWalkthroughOpen(false);
            }
          }}
          role="dialog"
        >
          <section className="walkthrough-dialog">
            <div className="walkthrough-heading">
              <div>
                <p className="eyebrow">First run</p>
                <h2>Forecast in three steps.</h2>
              </div>
              <button
                aria-label="Close walkthrough"
                onClick={() => setWalkthroughOpen(false)}
                type="button"
              >
                <X aria-hidden="true" size={18} weight="bold" />
              </button>
            </div>
            <video
              controls
              playsInline
              poster={`${import.meta.env.BASE_URL}walkthrough/filmer-web-walkthrough-poster.jpg`}
              preload="metadata"
            >
              <source
                src={`${import.meta.env.BASE_URL}walkthrough/filmer-web-walkthrough.mp4`}
                type="video/mp4"
              />
              Your browser cannot play the walkthrough video.
            </video>
            <ol className="walkthrough-steps">
              <li>
                <strong>Choose a trained domain.</strong>
                <span>Use 27 km India or a trained 9 km regional domain.</span>
              </li>
              <li>
                <strong>Pick a field and timestamps.</strong>
                <span>Each forecast timestamp is three hours apart.</span>
              </li>
              <li>
                <strong>Download data &amp; run.</strong>
                <span>Keep the tab open; the map appears when inference finishes.</span>
              </li>
            </ol>
            <p className="walkthrough-note">
              The model runs entirely in this browser. Most waiting time is the
              download and decoding of current GFS weather data.
            </p>
          </section>
        </div>
      ) : null}
    </main>
  );
}
