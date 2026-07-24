import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { strToU8, zipSync } from "fflate";
import {
  ArrowRight,
  CheckCircle,
  CloudArrowDown,
  Cpu,
  DownloadSimple,
  Gauge,
  ListChecks,
  MapPin,
  NavigationArrow,
  Pulse,
  Warning,
} from "@phosphor-icons/react";
import { ForecastMap } from "./components/ForecastMap";
import {
  inputForSequenceStep,
  readSequenceBundle,
  type SequenceBundle,
} from "./lib/bundle";
import {
  metadata,
  assembleInput,
  sequenceRequirements,
  type Domain,
} from "./lib/preprocess";
import {
  fetchGfsFrame,
  loadOutputGrid,
  loadStaticMonth,
  previousCycleInput,
  type GfsProgress,
} from "./lib/gfs";
import {
  FilmerSession,
  loadFloatFixture,
  loadModelManifest,
  parityMetrics,
  webGpuAvailable,
  type Backend,
  type LoadProgress,
  type ModelManifest,
} from "./lib/runtime";

const variableLabels = [
  "2 m temperature",
  "10 m zonal wind",
  "10 m meridional wind",
  "2 m humidity",
  "Surface pressure",
  "Precipitation",
];

type RuntimeState =
  | "idle"
  | "loading"
  | "ready"
  | "running"
  | "success"
  | "error";

type StepStatus = "queued" | "fetching" | "running" | "complete" | "error";

type StepDetail = {
  index: number;
  leadHours: number;
  validTime: string;
  inputPair: string;
  status: StepStatus;
  inferenceMilliseconds?: number;
};

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function latestCompletedGfsCycle() {
  const now = new Date();
  const cycleHour = Math.floor(now.getUTCHours() / 6) * 6;
  const candidate = new Date(
    Date.UTC(
      now.getUTCFullYear(),
      now.getUTCMonth(),
      now.getUTCDate(),
      cycleHour,
    ),
  );
  // Leave one cycle for the complete 0–93 h trajectory to arrive.
  candidate.setUTCHours(candidate.getUTCHours() - 6);
  return candidate.toISOString().slice(0, 16);
}

function DomainButton({
  domain,
  selected,
  onSelect,
}: {
  domain: Domain;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      className={`domain-button ${selected ? "domain-button-active" : ""}`}
      onClick={onSelect}
      type="button"
    >
      <span className="font-mono text-xs uppercase tracking-[0.18em]">
        {domain.code}
      </span>
      <span className="mt-2 block text-left text-sm font-medium">
        {domain.name}
      </span>
      <span className="mt-1 block text-left font-mono text-xs text-stone-500">
        {domain.resolutionKm} km
      </span>
    </button>
  );
}

function StageBar({
  label,
  detail,
  percent,
  state,
}: {
  label: string;
  detail: string;
  percent: number;
  state: "waiting" | "active" | "complete" | "error";
}) {
  return (
    <div className="stage-row">
      <div className="flex items-baseline justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-stone-800">{label}</p>
          <p className="mt-0.5 font-mono text-[10px] text-stone-500">
            {detail}
          </p>
        </div>
        <span className={`stage-state stage-state-${state}`}>
          {state === "complete" ? "ready" : state}
        </span>
      </div>
      <div
        aria-label={`${label} ${Math.round(percent)} percent`}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={Math.round(percent)}
        className="stage-track"
        role="progressbar"
      >
        <div
          className="stage-fill"
          style={{ transform: `translateX(${Math.max(0, percent) - 100}%)` }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [domainId, setDomainId] = useState(1);
  const [horizon, setHorizon] = useState(96);
  const [variableIndex, setVariableIndex] = useState(0);
  const [requestedBackend, setRequestedBackend] = useState<"auto" | Backend>(
    "auto",
  );
  const [runtimeState, setRuntimeState] = useState<RuntimeState>("idle");
  const [runtimeBackend, setRuntimeBackend] = useState<Backend | null>(null);
  const [manifest, setManifest] = useState<ModelManifest | null>(null);
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [field, setField] = useState<Float32Array | null>(null);
  const [stepMilliseconds, setStepMilliseconds] = useState<number | null>(null);
  const [parity, setParity] = useState<ReturnType<typeof parityMetrics> | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [sequenceBundle, setSequenceBundle] =
    useState<SequenceBundle | null>(null);
  const [sequenceProgress, setSequenceProgress] = useState(0);
  const [gfsCycle, setGfsCycle] = useState(latestCompletedGfsCycle);
  const [gfsProgress, setGfsProgress] = useState<GfsProgress | null>(null);
  const [gfsFrameIndex, setGfsFrameIndex] = useState(0);
  const [gfsDownloadedBytes, setGfsDownloadedBytes] = useState(0);
  const [staticReady, setStaticReady] = useState(false);
  const [stepDetails, setStepDetails] = useState<StepDetail[]>([]);
  const [outputGrid, setOutputGrid] = useState<{
    latitude: Float32Array;
    longitude: Float32Array;
  } | null>(null);
  const [lastForecastTime, setLastForecastTime] = useState(
    "2025-01-01T06:00:00Z",
  );
  const sessionRef = useRef<FilmerSession | null>(null);
  const requirements = sequenceRequirements(horizon);
  const selectedDomain = metadata.domains.find(
    (domain) => domain.id === domainId,
  )!;
  const activeStep = stepDetails.find(
    (step) => step.status === "fetching" || step.status === "running",
  );

  useEffect(() => {
    Promise.all([
      loadFloatFixture("fixture-physical.f32"),
      loadModelManifest(),
    ])
      .then(([physical, modelManifest]) => {
        setField(physical);
        setManifest(modelManifest);
      })
      .catch((loadError: unknown) => {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load the static fixture",
        );
      });
  }, []);

  useEffect(() => {
    loadOutputGrid(domainId)
      .then(setOutputGrid)
      .catch((gridError: unknown) => {
        setError(
          gridError instanceof Error
            ? gridError.message
            : "Unable to load output coordinates",
        );
      });
  }, [domainId]);

  const modelArtifact = useMemo(() => {
    if (!manifest) return null;
    if (requestedBackend === "wasm") return manifest.artifacts.wasm;
    if (requestedBackend === "webgpu") return manifest.artifacts.webgpu;
    return webGpuAvailable()
      ? manifest.artifacts.webgpu
      : manifest.artifacts.wasm;
  }, [manifest, requestedBackend]);

  function updateStep(index: number, patch: Partial<StepDetail>) {
    setStepDetails((current) =>
      current.map((step) => (step.index === index ? { ...step, ...patch } : step)),
    );
  }

  function failActiveSteps() {
    setStepDetails((current) =>
      current.map((step) =>
        step.status === "fetching" || step.status === "running"
          ? { ...step, status: "error" }
          : step,
      ),
    );
  }

  function operationalSteps(cycle: Date, count: number): StepDetail[] {
    return Array.from({ length: count }, (_, index) => {
      const leadHours = (index + 1) * 3;
      const leftLead = index === 0 ? "previous cycle +003" : `f${String((index - 1) * 3).padStart(3, "0")}`;
      const rightLead = `f${String(index * 3).padStart(3, "0")}`;
      return {
        index,
        leadHours,
        validTime: new Date(
          cycle.getTime() + leadHours * 60 * 60 * 1000,
        ).toISOString(),
        inputPair: `${leftLead} → ${rightLead}`,
        status: index === 0 ? "fetching" : "queued",
      };
    });
  }

  async function ensureSession() {
    if (!manifest) throw new Error("Model manifest is not ready");
    let session = sessionRef.current;
    if (
      !session ||
      (requestedBackend !== "auto" && session.backend !== requestedBackend)
    ) {
      setRuntimeState("loading");
      session = await FilmerSession.create(
        manifest,
        requestedBackend,
        setProgress,
      );
      sessionRef.current = session;
      setRuntimeBackend(session.backend);
      setRuntimeState("ready");
    }
    return session;
  }

  async function runParityStep() {
    setError(null);
    setParity(null);
    setSequenceProgress(0);
    setStepDetails([
      {
        index: 0,
        leadHours: 3,
        validTime: "2025-01-01T06:00:00Z",
        inputPair: "committed PyTorch parity fixture",
        status: "running",
      },
    ]);
    try {
      const session = await ensureSession();
      setRuntimeState("running");
      const [
        input,
        referenceState,
        referenceOccurrence,
        referenceIntensity,
      ] = await Promise.all([
        loadFloatFixture("fixture-input.f32"),
        loadFloatFixture("fixture-state.f32"),
        loadFloatFixture("fixture-occurrence.f32"),
        loadFloatFixture("fixture-intensity.f32"),
      ]);
      const result = await session.run(
        input,
        domainId,
        new Date("2025-01-01T06:00:00Z"),
      );
      setField(result.physical);
      setLastForecastTime("2025-01-01T06:00:00Z");
      setStepMilliseconds(result.elapsedMilliseconds);
      setSequenceProgress(1);
      updateStep(0, {
        status: "complete",
        inferenceMilliseconds: result.elapsedMilliseconds,
      });
      if (domainId === 1) {
        const stateParity = parityMetrics(result.state, referenceState);
        const occurrenceParity = parityMetrics(
          result.occurrence,
          referenceOccurrence,
        );
        const intensityParity = parityMetrics(
          result.intensity,
          referenceIntensity,
        );
        setParity({
          maxAbs: Math.max(
            stateParity.maxAbs,
            occurrenceParity.maxAbs,
            intensityParity.maxAbs,
          ),
          meanAbs:
            (stateParity.meanAbs +
              occurrenceParity.meanAbs +
              intensityParity.meanAbs) /
            3,
          rmse:
            (stateParity.rmse +
              occurrenceParity.rmse +
              intensityParity.rmse) /
            3,
        });
      }
      setRuntimeState("success");
    } catch (runError) {
      setRuntimeState("error");
      updateStep(0, { status: "error" });
      setError(
        runError instanceof Error ? runError.message : "Inference failed",
      );
    }
  }

  async function runSequence() {
    setError(null);
    setParity(null);
    setSequenceProgress(0);
    try {
      if (!sequenceBundle) throw new Error("Choose a prepared sequence bundle");
      setStepDetails(
        sequenceBundle.manifest.outputTimes.map((time, index) => ({
          index,
          leadHours: (index + 1) * 3,
          validTime: `${time}Z`,
          inputPair: `bundle frame ${index} → ${index + 1}`,
          status: index === 0 ? "running" : "queued",
        })),
      );
      const session = await ensureSession();
      setRuntimeState("running");
      const timings: number[] = [];
      for (
        let step = 0;
        step < sequenceBundle.manifest.outputTimes.length;
        step += 1
      ) {
        updateStep(step, { status: "running" });
        const input = inputForSequenceStep(sequenceBundle, step);
        const result = await session.run(
          input,
          sequenceBundle.manifest.domainId,
          new Date(`${sequenceBundle.manifest.outputTimes[step]}Z`),
        );
        timings.push(result.elapsedMilliseconds);
        updateStep(step, {
          status: "complete",
          inferenceMilliseconds: result.elapsedMilliseconds,
        });
        updateStep(step + 1, { status: "running" });
        setField(result.physical);
        setLastForecastTime(
          `${sequenceBundle.manifest.outputTimes[step]}Z`,
        );
        setSequenceProgress(step + 1);
      }
      setStepMilliseconds(
        timings.reduce((sum, timing) => sum + timing, 0) / timings.length,
      );
      setRuntimeState("success");
    } catch (sequenceError) {
      setRuntimeState("error");
      failActiveSteps();
      setError(
        sequenceError instanceof Error
          ? sequenceError.message
          : "Sequence inference failed",
      );
    }
  }

  async function runLiveGfs() {
    setError(null);
    setParity(null);
    setGfsProgress(null);
    setGfsFrameIndex(0);
    setGfsDownloadedBytes(0);
    setStaticReady(false);
    try {
      const cycle = new Date(`${gfsCycle}:00Z`);
      if (
        Number.isNaN(cycle.getTime()) ||
        ![0, 6, 12, 18].includes(cycle.getUTCHours()) ||
        cycle.getUTCMinutes() !== 0
      ) {
        throw new Error("Choose a GFS cycle at 00, 06, 12, or 18 UTC");
      }
      setStepDetails(operationalSteps(cycle, requirements.outputSteps));
      setSequenceProgress(0);
      const session = await ensureSession();
      setRuntimeState("running");
      const normalizedStatic = await loadStaticMonth(
        cycle.getUTCMonth() + 1,
      );
      setStaticReady(true);
      const prior = previousCycleInput(cycle);
      let frameNumber = 0;
      let accountedBytes = 0;
      const progressForFrame = (next: GfsProgress) => {
        setGfsProgress(next);
        setGfsFrameIndex(frameNumber + 1);
        setGfsDownloadedBytes(accountedBytes + next.loadedBytes);
      };
      let frameBytes = 0;
      let previous = await fetchGfsFrame(
        prior.cycle,
        prior.forecastHour,
        (next) => {
          frameBytes = next.totalBytes;
          progressForFrame(next);
        },
      );
      accountedBytes += frameBytes;
      frameNumber += 1;
      frameBytes = 0;
      let current = await fetchGfsFrame(cycle, 0, (next) => {
        frameBytes = next.totalBytes;
        progressForFrame(next);
      });
      accountedBytes += frameBytes;
      frameNumber += 1;
      const timings: number[] = [];
      for (let step = 0; step < requirements.outputSteps; step += 1) {
        updateStep(step, { status: "running" });
        const forecastTime = new Date(
          cycle.getTime() + (step + 1) * 3 * 60 * 60 * 1000,
        );
        const result = await session.run(
          assembleInput(previous, current, normalizedStatic),
          domainId,
          forecastTime,
        );
        timings.push(result.elapsedMilliseconds);
        updateStep(step, {
          status: "complete",
          inferenceMilliseconds: result.elapsedMilliseconds,
        });
        setField(result.physical);
        setLastForecastTime(forecastTime.toISOString());
        setSequenceProgress(step + 1);
        if (step + 1 < requirements.outputSteps) {
          updateStep(step + 1, { status: "fetching" });
          previous = current;
          frameBytes = 0;
          current = await fetchGfsFrame(
            cycle,
            (step + 1) * 3,
            (next) => {
              frameBytes = next.totalBytes;
              progressForFrame(next);
            },
          );
          accountedBytes += frameBytes;
          frameNumber += 1;
        }
      }
      setGfsDownloadedBytes(accountedBytes);
      setStepMilliseconds(
        timings.reduce((sum, timing) => sum + timing, 0) / timings.length,
      );
      setRuntimeState("success");
    } catch (liveError) {
      setRuntimeState("error");
      failActiveSteps();
      setError(
        liveError instanceof Error
          ? liveError.message
          : "Operational GFS inference failed",
      );
    }
  }

  const progressPercent =
    progress && progress.total
      ? Math.min(100, (progress.loaded / progress.total) * 100)
      : 0;
  const modelStagePercent = sessionRef.current
    ? 100
    : progress
      ? progress.stage === "download"
        ? progressPercent * 0.96
        : progress.stage === "checksum"
          ? 98
          : 99
      : 0;
  const gfsStagePercent = gfsProgress
    ? Math.min(
        100,
        (((Math.max(1, gfsFrameIndex) - 1) +
          gfsProgress.completedFields / gfsProgress.totalFields) /
          requirements.gfsFrames) *
          100,
      )
    : sequenceBundle
      ? 100
      : 0;
  const plannedOutputSteps = stepDetails.length || requirements.outputSteps;
  const completedOutputSteps = Math.max(
    sequenceProgress,
    stepDetails.filter((step) => step.status === "complete").length,
  );
  const inferenceStagePercent =
    (completedOutputSteps / Math.max(1, plannedOutputSteps)) * 100;
  const fieldSummary = useMemo(() => {
    if (!field) return null;
    const start = variableIndex * 99 * 99;
    const end = start + 99 * 99;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let index = start; index < end; index += 1) {
      minimum = Math.min(minimum, field[index]);
      maximum = Math.max(maximum, field[index]);
      sum += field[index];
    }
    return {
      minimum,
      maximum,
      mean: sum / (99 * 99),
      unit: metadata.targets.units[variableIndex],
    };
  }, [field, variableIndex]);

  function downloadLatestOutput() {
    if (!field) return;
    const coordinates =
      outputGrid
        ? {
            latitudeFile: "grid-latitude.f32",
            longitudeFile: "grid-longitude.f32",
            shape: [99, 99],
          }
        : {
            note: "The parity fixture has no grid-coordinate arrays; trained domain bounds are only an approximate display extent.",
          };
    const outputManifest = {
      schemaVersion: 1,
      kind: "filmer-browser-output",
      model: "FiLMeR v1.0 Variant B",
      checkpointSha256: metadata.model.checkpointSha256,
      domain: selectedDomain.code,
      resolutionKm: selectedDomain.resolutionKm,
      forecastTime: lastForecastTime,
      shape: [6, 99, 99],
      dtype: "little-endian float32",
      variables: metadata.targets.outputVariables,
      units: metadata.targets.units,
      backend: runtimeBackend ?? "reference-fixture",
      timingScope: "model compute only",
      semantics:
        "Conditional regional downscaling from paired 3-hourly GFS inputs.",
      coordinates,
    };
    const entries: Record<string, Uint8Array> = {
      "manifest.json": strToU8(JSON.stringify(outputManifest, null, 2)),
      "filmer-output.f32": new Uint8Array(
        field.buffer,
        field.byteOffset,
        field.byteLength,
      ),
    };
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
    anchor.download = `filmer-${selectedDomain.code}-${lastForecastTime.replaceAll(":", "")}.zip`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-[100dvh] bg-[#f3f1ea] text-[#252724]">
      <header className="border-b border-stone-400/40">
        <div className="mx-auto grid max-w-[1400px] grid-cols-1 gap-6 px-4 py-6 md:grid-cols-[1fr_auto] md:px-8">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center border border-stone-700 bg-[#252724] text-stone-100">
              <span className="font-mono text-xs font-semibold">FM</span>
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                FiLMeR WebGPU
              </p>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-stone-500">
                Regional weather downscaling workbench
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[11px] uppercase tracking-[0.14em] text-stone-600">
            <span>Variant B</span>
            <span>39.77 M parameters</span>
            <span className="inline-flex items-center gap-2">
              <span
                className={`size-2 rounded-full ${
                  webGpuAvailable() ? "bg-[#a6552c]" : "bg-stone-400"
                }`}
              />
              {webGpuAvailable() ? "WebGPU available" : "WASM only"}
            </span>
          </div>
        </div>
      </header>

      <section className="operational-ribbon" aria-label="Forecast context">
        <div>
          <span>domain</span>
          <strong>{selectedDomain.code.toUpperCase()}</strong>
          <small>{selectedDomain.resolutionKm} km trained grid</small>
        </div>
        <div>
          <span>GFS cycle</span>
          <strong>{gfsCycle.replace("T", " ")}</strong>
          <small>UTC</small>
        </div>
        <div>
          <span>product</span>
          <strong>{horizon} h / 3 h</strong>
          <small>{requirements.outputSteps} conditional fields</small>
        </div>
        <div>
          <span>displayed valid time</span>
          <strong>{lastForecastTime.replace("T", " ").slice(0, 16)}</strong>
          <small>UTC · {variableLabels[variableIndex]}</small>
        </div>
        <div>
          <span>execution</span>
          <strong>{runtimeBackend?.toUpperCase() ?? "not loaded"}</strong>
          <small>
            {runtimeState}
            {stepMilliseconds
              ? ` · ${stepMilliseconds.toFixed(1)} ms latest mean`
              : ""}
          </small>
        </div>
      </section>

      <section className="mx-auto grid max-w-[1400px] grid-cols-1 border-x border-stone-400/30 lg:grid-cols-[minmax(0,1.2fr)_minmax(360px,0.8fr)]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
          className="border-b border-stone-400/30 p-5 md:p-8 lg:border-b-0 lg:border-r"
        >
          <div className="mb-6 grid grid-cols-1 gap-5 md:grid-cols-[1fr_0.7fr]">
            <div>
              <p className="eyebrow">Operational scenario explorer</p>
              <h1 className="mt-3 max-w-[20ch] text-3xl font-semibold leading-[1] tracking-[-0.045em] md:text-5xl">
                GFS forcing to regional fields, in this browser.
              </h1>
            </div>
            <div className="self-end border-t border-stone-500/40 pt-4">
              <p className="max-w-[46ch] text-sm leading-relaxed text-stone-600">
                ONNX Runtime Web executes the trained FiLMeR network on the
                local GPU, with a WebAssembly CPU fallback. Forecast semantics
                remain conditional on a complete 3-hourly GFS sequence.
              </p>
            </div>
          </div>
          <div className="field-readout">
            <div>
              <span>field</span>
              <strong>{variableLabels[variableIndex]}</strong>
            </div>
            <div>
              <span>valid</span>
              <strong>{lastForecastTime.replace("T", " ").slice(0, 16)} UTC</strong>
            </div>
            <div>
              <span>range</span>
              <strong>
                {fieldSummary
                  ? `${fieldSummary.minimum.toFixed(2)}–${fieldSummary.maximum.toFixed(2)} ${fieldSummary.unit}`
                  : "—"}
              </strong>
            </div>
            <div>
              <span>spatial mean</span>
              <strong>
                {fieldSummary
                  ? `${fieldSummary.mean.toFixed(2)} ${fieldSummary.unit}`
                  : "—"}
              </strong>
            </div>
          </div>
          <ForecastMap
            values={field}
            variableIndex={variableIndex}
            domainId={domainId}
          />
          <div className="mt-5 flex gap-2 overflow-x-auto pb-2">
            {variableLabels.map((label, index) => (
              <button
                key={label}
                className={`variable-pill ${
                  variableIndex === index ? "variable-pill-active" : ""
                }`}
                onClick={() => setVariableIndex(index)}
                type="button"
              >
                {label}
              </button>
            ))}
            <button
              className="variable-pill ml-auto inline-flex items-center gap-2"
              disabled={!field}
              onClick={downloadLatestOutput}
              type="button"
            >
              <DownloadSimple size={15} weight="bold" />
              Download output
            </button>
          </div>
        </motion.div>

        <motion.aside
          initial={{ opacity: 0, x: 18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.08 }}
          className="divide-y divide-stone-400/30"
        >
          <section className="p-5 md:p-8">
            <div className="section-heading">
              <MapPin size={18} weight="bold" />
              <h2>Trained geography</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-stone-600">
              Choose one of the four domains present in training. The
              checkpoint is not validated for arbitrary coordinates.
            </p>
            <div className="mt-5 grid grid-cols-2 gap-2">
              {metadata.domains.map((domain) => (
                <DomainButton
                  key={domain.code}
                  domain={domain}
                  selected={domain.id === domainId}
                  onSelect={() => {
                    setDomainId(domain.id);
                    setParity(null);
                  }}
                />
              ))}
            </div>
            <div className="custom-domain-note">
              <NavigationArrow size={18} weight="bold" />
              <div>
                <p className="text-xs font-semibold">
                  Custom coordinates / resolution are not enabled
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-stone-600">
                  The projection MLP accepts numbers, but the checkpoint has no
                  validation beyond these four domains and its static encoder
                  is tied to the d01 WPS grid. Arbitrary values would create a
                  plausible-looking, scientifically unsupported map.
                </p>
                <p className="mt-2 font-mono text-[10px] text-[#7f3d20]">
                  Needs new geogrid → WRF supervision → retraining/transfer test
                  → held-out validation
                </p>
              </div>
            </div>
          </section>

          <section className="p-5 md:p-8">
            <div className="section-heading">
              <Gauge size={18} weight="bold" />
              <h2>Conditional horizon</h2>
            </div>
            <div className="mt-5 grid grid-cols-[1fr_auto] gap-4">
              <label className="text-sm font-medium" htmlFor="horizon">
                Requested lead time
              </label>
              <output className="font-mono text-sm">{horizon} h</output>
              <input
                id="horizon"
                className="col-span-2 accent-[#a6552c]"
                type="range"
                min="3"
                max="96"
                step="3"
                value={horizon}
                onChange={(event) => setHorizon(Number(event.target.value))}
              />
            </div>
            <div
              aria-label="Lead-time presets"
              className="mt-3 grid grid-cols-4 gap-1"
            >
              {[3, 24, 48, 96].map((hours) => (
                <button
                  className={`border px-2 py-2 font-mono text-[10px] transition-colors ${
                    horizon === hours
                      ? "border-[#a6552c] bg-[#ebe2d8] text-[#7f3d20]"
                      : "border-stone-400/40 text-stone-600 hover:bg-white/30"
                  }`}
                  key={hours}
                  onClick={() => setHorizon(hours)}
                  type="button"
                >
                  +{hours} h
                </button>
              ))}
            </div>
            <div className="mt-5 grid grid-cols-2 border-y border-stone-400/40 py-4">
              <div>
                <p className="metric">{requirements.outputSteps}</p>
                <p className="metric-label">3 h output steps</p>
              </div>
              <div className="border-l border-stone-400/40 pl-4">
                <p className="metric">{requirements.gfsFrames}</p>
                <p className="metric-label">required GFS frames</p>
              </div>
            </div>
            <div className="mt-4 flex items-start gap-3 text-xs leading-relaxed text-stone-600">
              <Warning className="mt-0.5 shrink-0 text-[#a6552c]" size={16} />
              <p>
                A {horizon}-hour request needs GFS at leads −3 through +
                {requirements.lastGfsLeadHours} h. FiLMeR outputs cannot be fed
                back as GFS state, so this is conditional downscaling, not a
                standalone autoregressive forecast.
              </p>
            </div>
            <div className="mt-5 border-t border-stone-400/40 pt-5">
              <label className="block text-sm font-medium" htmlFor="gfs-cycle">
                Operational GFS cycle
              </label>
              <p className="mt-1 text-xs leading-relaxed text-stone-500">
                Direct CORS-safe byte-range reads from NOAA’s public Google
                Cloud mirror. Only the checkpoint’s 20 GRIB records are
                downloaded, decoded, and cropped locally.
              </p>
              <input
                id="gfs-cycle"
                className="control mt-3"
                type="datetime-local"
                step="21600"
                value={gfsCycle}
                onChange={(event) => setGfsCycle(event.target.value)}
              />
              <button
                className="sequence-button mt-3"
                disabled={
                  runtimeState === "loading" || runtimeState === "running"
                }
                onClick={runLiveGfs}
                type="button"
              >
                <span>
                  {runtimeState === "running" && gfsProgress
                    ? `GFS frame ${Math.min(
                        gfsFrameIndex,
                        requirements.gfsFrames,
                      )}/${requirements.gfsFrames} · ${
                        gfsProgress.completedFields
                      }/20 fields`
                    : `Fetch NOAA GFS & run ${horizon} h`}
                </span>
                <CloudArrowDown size={18} weight="bold" />
              </button>
              {gfsProgress ? (
                <div className="mt-3 font-mono text-[10px] uppercase tracking-[0.1em] text-stone-500">
                  <div className="flex justify-between gap-3">
                    <span>
                      {gfsProgress.stage} · {gfsProgress.frame}
                    </span>
                    <span>
                      {formatBytes(gfsDownloadedBytes)}
                    </span>
                  </div>
                  <div className="mt-2 h-1 overflow-hidden bg-stone-300">
                    <div
                      className="h-full bg-[#a6552c] transition-transform"
                      style={{
                        transform: `translateX(${
                          (gfsProgress.completedFields /
                            gfsProgress.totalFields) *
                            100 -
                          100
                        }%)`,
                      }}
                    />
                  </div>
                </div>
              ) : null}
              <p className="mt-3 text-[11px] leading-relaxed text-stone-500">
                Static-only tradeoff: record ranges avoid 500 MB global files,
                but NOAA GRIB records are still global fields. A 96 h run can
                transfer hundreds of MB; a regional subset proxy is the next
                production optimization.
              </p>
            </div>
            <div className="mt-5 border-t border-stone-400/40 pt-5">
              <label
                className="block text-sm font-medium"
                htmlFor="sequence-bundle"
              >
                Prepared GFS trajectory bundle
              </label>
              <p className="mt-1 text-xs leading-relaxed text-stone-500">
                ZIP from <code>scripts/prepare_sequence.py</code>; contains
                pre-cropped GFS frames and a normalized WPS geogrid.
              </p>
              <input
                id="sequence-bundle"
                className="mt-3 block w-full text-xs file:mr-3 file:border-0 file:bg-[#ded9cf] file:px-3 file:py-2 file:text-xs file:font-medium"
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  setError(null);
                  readSequenceBundle(file)
                    .then((bundle) => {
                      setSequenceBundle(bundle);
                      setDomainId(bundle.manifest.domainId);
                      setHorizon(bundle.manifest.horizonHours);
                      setSequenceProgress(0);
                    })
                    .catch((bundleError: unknown) =>
                      setError(
                        bundleError instanceof Error
                          ? bundleError.message
                          : "Unable to read sequence bundle",
                      ),
                    );
                }}
              />
              {sequenceBundle ? (
                <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.12em] text-[#7f3d20]">
                  {sequenceBundle.manifest.outputTimes.length} steps ·{" "}
                  {sequenceBundle.manifest.horizonHours} h · d
                  {String(sequenceBundle.manifest.domainId).padStart(2, "0")}
                </p>
              ) : null}
            </div>
          </section>

          <section className="p-5 md:p-8">
            <div className="section-heading">
              <Cpu size={18} weight="bold" />
              <h2>Local execution</h2>
            </div>
            <label className="mt-5 block text-sm font-medium" htmlFor="backend">
              Runtime backend
            </label>
            <select
              id="backend"
              className="control mt-2"
              value={requestedBackend}
              onChange={(event) => {
                setRequestedBackend(
                  event.target.value as "auto" | Backend,
                );
                sessionRef.current = null;
                setRuntimeState("idle");
                setRuntimeBackend(null);
              }}
            >
              <option value="auto">Auto — WebGPU then WASM</option>
              <option value="webgpu" disabled={!webGpuAvailable()}>
                WebGPU — fp16
              </option>
              <option value="wasm">WebAssembly — fp32</option>
            </select>
            <div className="mt-3 flex justify-between font-mono text-[11px] uppercase tracking-[0.12em] text-stone-500">
              <span>{modelArtifact?.precision ?? "—"}</span>
              <span>
                {modelArtifact ? formatBytes(modelArtifact.bytes) : "—"}
              </span>
            </div>

            {runtimeState === "loading" && progress ? (
              <div className="mt-5">
                <div className="mb-2 flex justify-between font-mono text-[11px] uppercase tracking-[0.12em]">
                  <span>{progress.stage}</span>
                  <span>{progressPercent.toFixed(0)}%</span>
                </div>
                <div className="h-1 overflow-hidden bg-stone-300">
                  <div
                    className="h-full bg-[#a6552c] transition-transform"
                    style={{
                      transform: `translateX(${progressPercent - 100}%)`,
                    }}
                  />
                </div>
              </div>
            ) : null}

            <button
              className="run-button mt-5"
              disabled={
                runtimeState === "loading" || runtimeState === "running"
              }
              onClick={runParityStep}
              type="button"
            >
              <span>
                {runtimeState === "loading"
                  ? "Loading verified model"
                  : runtimeState === "running"
                    ? "Running FiLMeR"
                    : "Run one validated step"}
              </span>
              {runtimeState === "success" ? (
                <CheckCircle size={18} weight="bold" />
              ) : runtimeState === "loading" ? (
                <CloudArrowDown size={18} weight="bold" />
              ) : (
                <ArrowRight size={18} weight="bold" />
              )}
            </button>
            <button
              className="sequence-button mt-2"
              disabled={
                !sequenceBundle ||
                runtimeState === "loading" ||
                runtimeState === "running"
              }
              onClick={runSequence}
              type="button"
            >
              <span>
                {runtimeState === "running" && sequenceBundle
                  ? `Sequence ${sequenceProgress}/${sequenceBundle.manifest.outputTimes.length}`
                  : sequenceBundle
                    ? `Run ${sequenceBundle.manifest.outputTimes.length}-step bundle`
                    : "Choose a sequence bundle"}
              </span>
              <Gauge size={18} weight="bold" />
            </button>
            {error ? (
              <p className="mt-3 border-l-2 border-[#a6552c] pl-3 text-xs leading-relaxed text-[#7f3d20]">
                {error}
              </p>
            ) : null}
          </section>
        </motion.aside>
      </section>

      <section className="mx-auto max-w-[1400px] border-x border-t border-stone-400/30">
        <div className="grid grid-cols-1 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="p-5 md:p-8 lg:border-r lg:border-stone-400/30">
            <div className="section-heading">
              <Pulse size={18} weight="bold" />
              <h2>Run progress</h2>
            </div>
            <p className="mt-2 max-w-[58ch] text-xs leading-relaxed text-stone-600">
              Every bar has a distinct scope. Acquisition and model compute are
              reported separately so a fast kernel is not mistaken for a fast
              end-to-end forecast.
            </p>
            <div className="mt-5 space-y-5">
              <StageBar
                label="Verified model"
                detail={
                  progress
                    ? `${progress.stage} · ${modelArtifact ? formatBytes(modelArtifact.bytes) : "artifact"}`
                    : modelArtifact
                      ? `${modelArtifact.precision} · ${formatBytes(modelArtifact.bytes)}`
                      : "manifest pending"
                }
                percent={modelStagePercent}
                state={
                  sessionRef.current
                    ? "complete"
                    : progress
                      ? "active"
                      : runtimeState === "error"
                        ? "error"
                        : "waiting"
                }
              />
              <StageBar
                label="WPS static geography"
                detail={
                  staticReady
                    ? `month ${String(
                        new Date(`${gfsCycle}:00Z`).getUTCMonth() + 1,
                      ).padStart(2, "0")} · 30 × 127 × 137`
                    : sequenceBundle
                      ? "normalized static tensor in bundle"
                      : "cached monthly d01 artifact"
                }
                percent={staticReady || Boolean(sequenceBundle) ? 100 : 0}
                state={
                  staticReady || sequenceBundle ? "complete" : "waiting"
                }
              />
              <StageBar
                label="NOAA GFS inputs"
                detail={
                  gfsProgress
                    ? `${gfsFrameIndex}/${requirements.gfsFrames} frames · ${formatBytes(
                        gfsDownloadedBytes,
                      )}`
                    : sequenceBundle
                      ? `${requirements.gfsFrames} prepared frames`
                      : `${requirements.gfsFrames} frames required`
                }
                percent={gfsStagePercent}
                state={
                  gfsStagePercent >= 100
                    ? "complete"
                    : gfsStagePercent > 0
                      ? "active"
                      : "waiting"
                }
              />
              <StageBar
                label="FiLMeR outputs"
                detail={`${completedOutputSteps}/${plannedOutputSteps} conditional steps`}
                percent={inferenceStagePercent}
                state={
                  inferenceStagePercent >= 100
                    ? "complete"
                    : inferenceStagePercent > 0
                      ? "active"
                      : runtimeState === "error" && stepDetails.length
                        ? "error"
                        : "waiting"
                }
              />
            </div>
          </div>
          <div className="min-w-0 border-t border-stone-400/30 lg:border-t-0">
            <div className="flex items-start justify-between gap-4 border-b border-stone-400/30 p-5 md:px-8 md:py-6">
              <div>
                <div className="section-heading">
                  <ListChecks size={18} weight="bold" />
                  <h2>Step ledger</h2>
                </div>
                <p className="mt-2 text-xs text-stone-600">
                  Input pair, output lead, valid UTC time, and measured model
                  latency for each field.
                </p>
              </div>
              {activeStep ? (
                <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-[#7f3d20]">
                  active +{activeStep.leadHours} h
                </span>
              ) : null}
            </div>
            {stepDetails.length ? (
              <div className="max-h-[390px] overflow-auto">
                <table className="step-table">
                  <thead>
                    <tr>
                      <th>step</th>
                      <th>GFS input pair</th>
                      <th>valid UTC</th>
                      <th>model time</th>
                      <th>status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stepDetails.map((step) => (
                      <tr key={step.index}>
                        <td className="font-mono">
                          {String(step.index + 1).padStart(2, "0")} · +
                          {step.leadHours} h
                        </td>
                        <td className="font-mono text-[10px]">
                          {step.inputPair}
                        </td>
                        <td className="font-mono text-[10px]">
                          {step.validTime.replace("T", " ").slice(0, 16)}
                        </td>
                        <td className="font-mono text-[10px]">
                          {step.inferenceMilliseconds == null
                            ? "—"
                            : `${step.inferenceMilliseconds.toFixed(1)} ms`}
                        </td>
                        <td>
                          <span
                            className={`step-status step-status-${step.status}`}
                          >
                            {step.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="grid min-h-64 place-items-center p-8 text-center">
                <div>
                  <p className="font-mono text-xs text-stone-600">
                    No run has started
                  </p>
                  <p className="mt-2 max-w-[42ch] text-xs leading-relaxed text-stone-500">
                    Choose a trained domain, GFS cycle, horizon, and runtime;
                    then start an operational or parity run.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-[1400px] border border-stone-400/30">
        <div className="grid grid-cols-1 lg:grid-cols-3">
          <div className="p-5 md:p-8">
            <p className="eyebrow">Numerical parity</p>
            <p className="mt-4 font-mono text-2xl font-semibold">
              {parity ? parity.maxAbs.toExponential(2) : "Pending run"}
            </p>
            <p className="mt-2 text-xs leading-relaxed text-stone-600">
              Maximum absolute difference from the committed PyTorch fixture.
              Only d01 uses the identical reference projection.
            </p>
          </div>
          <div className="border-t border-stone-400/30 p-5 md:p-8 lg:border-l lg:border-t-0">
            <p className="eyebrow">Active engine</p>
            <p className="mt-4 font-mono text-2xl font-semibold uppercase">
              {runtimeBackend ?? "Not loaded"}
            </p>
            <p className="mt-2 text-xs leading-relaxed text-stone-600">
              WebGPU uses the fp16 release artifact. WASM uses the fp32 artifact
              and may be substantially slower in a non-isolated Pages tab.
            </p>
          </div>
          <div className="border-t border-stone-400/30 p-5 md:p-8 lg:border-l lg:border-t-0">
            <p className="eyebrow">Timing scope</p>
            <p className="mt-4 font-mono text-2xl font-semibold">Compute only</p>
            <p className="mt-2 text-xs leading-relaxed text-stone-600">
              Excludes model download, GFS retrieval, crop/alignment, static
              geogrid preparation, serialization, and output I/O. Compare WRF
              wall time only after measuring the same end-to-end boundary.
            </p>
          </div>
        </div>
      </section>

      <footer className="mx-auto flex max-w-[1400px] flex-col gap-3 px-4 py-8 text-xs text-stone-500 md:flex-row md:items-center md:justify-between md:px-8">
        <p>
          Research prototype · FiLMeR v1.0 · IIT Gandhinagar Sustainability Lab
        </p>
        <p className="font-mono">
          Checkpoint {metadata.model.checkpointSha256.slice(0, 12)}…
        </p>
      </footer>
    </main>
  );
}
