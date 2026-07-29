import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { motion } from "framer-motion";
import { strToU8, zipSync } from "fflate";
import {
  ArrowRight,
  CaretLeft,
  CaretRight,
  CheckCircle,
  CloudArrowDown,
  Cpu,
  DownloadSimple,
  Gauge,
  Info,
  ListChecks,
  Lightning,
  MapPin,
  NavigationArrow,
  Pause,
  Play,
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
  type GfsProvider,
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
import {
  differenceField,
  loadValidationFloat,
  loadValidationManifest,
  verificationMetrics,
  type ValidationManifest,
  type VerificationMetric,
} from "./lib/verification";
import {
  VARIABLE_VISUALS,
  deriveDisplayFields,
  displayValue,
} from "./lib/visualization";
import gfsSourceAudit from "../reports/gfs-source-equivalence-20240511T00Z.json";

const variableLabels = VARIABLE_VISUALS.map((variable) => variable.label);

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

type ForecastFrame = {
  validTime: string;
  values: Float32Array;
};

type ComparisonView = "prediction" | "reference" | "difference";
type RunKind = "none" | "parity" | "validation" | "bundle" | "live";

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
          <p
            className={`mt-0.5 font-mono text-[10px] text-stone-500 ${
              state === "active" ? "stage-detail-live" : ""
            }`}
          >
            {detail}
          </p>
        </div>
        <span className={`stage-state stage-state-${state}`}>
          {state === "complete"
            ? "ready"
            : state === "active"
              ? `${Math.round(percent)}%`
              : state}
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

function InfoTip({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <span className="info-tip">
      <button aria-label={label} type="button">
        <Info size={15} weight="bold" />
      </button>
      <span className="info-tip-content" role="tooltip">
        {children}
      </span>
    </span>
  );
}

export default function App() {
  const [domainId, setDomainId] = useState(1);
  const [horizon, setHorizon] = useState(96);
  const [variableIndex, setVariableIndex] = useState(0);
  const [requestedBackend, setRequestedBackend] = useState<"auto" | Backend>(
    "wasm",
  );
  const [runtimeState, setRuntimeState] = useState<RuntimeState>("idle");
  const [runKind, setRunKind] = useState<RunKind>("none");
  const [runtimeBackend, setRuntimeBackend] = useState<Backend | null>(null);
  const [manifest, setManifest] = useState<ModelManifest | null>(null);
  const [validationManifest, setValidationManifest] =
    useState<ValidationManifest | null>(null);
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [field, setField] = useState<Float32Array | null>(null);
  const [forecastFrames, setForecastFrames] = useState<ForecastFrame[]>([]);
  const [activeFrameIndex, setActiveFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [referenceField, setReferenceField] =
    useState<Float32Array | null>(null);
  const [comparisonView, setComparisonView] =
    useState<ComparisonView>("prediction");
  const [comparisonMetrics, setComparisonMetrics] = useState<
    VerificationMetric[] | null
  >(null);
  const [validationParity, setValidationParity] = useState<
    ReturnType<typeof parityMetrics> | null
  >(null);
  const [validationResolution, setValidationResolution] = useState(27);
  const [stepMilliseconds, setStepMilliseconds] = useState<number | null>(null);
  const [parity, setParity] = useState<ReturnType<typeof parityMetrics> | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [sequenceBundle, setSequenceBundle] =
    useState<SequenceBundle | null>(null);
  const [sequenceProgress, setSequenceProgress] = useState(0);
  const [gfsCycle, setGfsCycle] = useState(latestCompletedGfsCycle);
  const [gfsProvider, setGfsProvider] =
    useState<GfsProvider>("noaa");
  const [gfsProgress, setGfsProgress] = useState<GfsProgress | null>(null);
  const [gfsFrameIndex, setGfsFrameIndex] = useState(0);
  const [gfsDownloadedBytes, setGfsDownloadedBytes] = useState(0);
  const [staticReady, setStaticReady] = useState(false);
  const [stepDetails, setStepDetails] = useState<StepDetail[]>([]);
  const [currentAction, setCurrentAction] = useState(
    "Ready for a verification or forecast run",
  );
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [runElapsedMilliseconds, setRunElapsedMilliseconds] = useState(0);
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
      loadValidationManifest(),
    ])
      .then(([physical, modelManifest, heldOutManifest]) => {
        setField(physical);
        setManifest(modelManifest);
        setValidationManifest(heldOutManifest);
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
    if (runStartedAt === null) return;
    const update = () => {
      setRunElapsedMilliseconds(Date.now() - runStartedAt);
    };
    update();
    if (runtimeState !== "loading" && runtimeState !== "running") return;
    const timer = window.setInterval(update, 200);
    return () => window.clearInterval(timer);
  }, [runStartedAt, runtimeState]);

  useEffect(() => {
    if (!isPlaying || forecastFrames.length < 2) return;
    const timer = window.setInterval(() => {
      setActiveFrameIndex((current) => (current + 1) % forecastFrames.length);
    }, 900);
    return () => window.clearInterval(timer);
  }, [isPlaying, forecastFrames.length]);

  useEffect(() => {
    const selectedFrame = forecastFrames[activeFrameIndex];
    if (selectedFrame) setLastForecastTime(selectedFrame.validTime);
  }, [activeFrameIndex, forecastFrames]);

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

  const predictionField =
    forecastFrames[activeFrameIndex]?.values ?? field;
  const predictionDisplayField = useMemo(
    () => (predictionField ? deriveDisplayFields(predictionField) : null),
    [predictionField],
  );
  const referenceDisplayField = useMemo(
    () => (referenceField ? deriveDisplayFields(referenceField) : null),
    [referenceField],
  );
  const displayedField = useMemo(() => {
    if (!predictionDisplayField) return null;
    if (comparisonView === "reference" && referenceDisplayField) {
      return referenceDisplayField;
    }
    if (comparisonView === "difference" && referenceDisplayField) {
      return differenceField(
        predictionDisplayField,
        referenceDisplayField,
      );
    }
    return predictionDisplayField;
  }, [predictionDisplayField, referenceDisplayField, comparisonView]);

  function beginRun(
    action: string,
    kind: RunKind,
    preserveComparison = false,
  ) {
    setError(null);
    setParity(null);
    setValidationParity(null);
    setSequenceProgress(0);
    setForecastFrames([]);
    setActiveFrameIndex(0);
    setIsPlaying(false);
    setCurrentAction(action);
    setRunKind(kind);
    setRunStartedAt(Date.now());
    setRunElapsedMilliseconds(0);
    if (!preserveComparison) {
      setReferenceField(null);
      setComparisonMetrics(null);
      setComparisonView("prediction");
    }
  }

  function publishFrame(next: ForecastFrame, prior: ForecastFrame[]) {
    const frames = [...prior, next];
    setForecastFrames(frames);
    setActiveFrameIndex(frames.length - 1);
    setField(next.values);
    setLastForecastTime(next.validTime);
    return frames;
  }

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
        (next) => {
          setProgress(next);
          setCurrentAction(
            next.stage === "download"
              ? `Downloading model · ${formatBytes(next.loaded)} / ${formatBytes(next.total)}`
              : next.stage === "checksum"
                ? "Verifying model SHA-256 checksum"
                : "Compiling ONNX execution graph",
          );
        },
      );
      sessionRef.current = session;
      setRuntimeBackend(session.backend);
      setRuntimeState("ready");
      setCurrentAction(`Model ready · ${session.backend.toUpperCase()}`);
    }
    return session;
  }

  async function ensureVerifiedSession() {
    if (!manifest) throw new Error("Model manifest is not ready");
    let session = sessionRef.current;
    if (!session || session.backend !== "wasm") {
      setRequestedBackend("wasm");
      setRuntimeState("loading");
      session = await FilmerSession.create(manifest, "wasm", (next) => {
        setProgress(next);
        setCurrentAction(
          next.stage === "download"
            ? `Downloading verified fp32 model · ${formatBytes(next.loaded)} / ${formatBytes(next.total)}`
            : next.stage === "checksum"
              ? "Verifying fp32 model SHA-256 checksum"
              : "Compiling verified WASM execution graph",
        );
      });
      sessionRef.current = session;
      setRuntimeBackend("wasm");
      setRuntimeState("ready");
    }
    return session;
  }

  async function runParityStep() {
    beginRun("Preparing committed Python parity tensors", "parity");
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
      setCurrentAction("Loading parity input and raw PyTorch heads");
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
      publishFrame(
        {
          values: result.physical,
          validTime: "2025-01-01T06:00:00Z",
        },
        [],
      );
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
      setCurrentAction(
        `Parity complete · ${result.elapsedMilliseconds.toFixed(1)} ms model compute`,
      );
    } catch (runError) {
      setRuntimeState("error");
      updateStep(0, { status: "error" });
      setCurrentAction("Parity run failed");
      setError(
        runError instanceof Error ? runError.message : "Inference failed",
      );
    }
  }

  async function runHeldOutVerification(resolutionOverride?: number) {
    const effectiveResolution =
      resolutionOverride ?? validationResolution;
    beginRun(
      effectiveResolution === 27
        ? "Loading cached GFS demo and held-out WRF target"
        : `Preparing ${effectiveResolution} km conditioning probe against the 27 km WRF target`,
      "validation",
    );
    setDomainId(1);
    setStaticReady(true);
    setStepDetails([
      {
        index: 0,
        leadHours: 3,
        validTime: "2025-01-01T06:00:00Z",
        inputPair: "cached training GFS 00Z → 03Z",
        status: "running",
      },
    ]);
    try {
      if (!validationManifest) {
        throw new Error("Held-out validation manifest is not ready");
      }
      const session = await ensureVerifiedSession();
      setRuntimeState("running");
      setCurrentAction(
        `Loading held-out WRF reference · ${formatBytes(
          validationManifest.artifacts.target.bytes,
        )}`,
      );
      const [input, target, pythonPrediction] = await Promise.all([
        loadValidationFloat(validationManifest.artifacts.input.file),
        loadValidationFloat(validationManifest.artifacts.target.file),
        loadValidationFloat(
          validationManifest.artifacts.pythonPrediction.file,
        ),
      ]);
      setCurrentAction(
        `Running FiLMeR · d01 conditioning ${effectiveResolution} km`,
      );
      const result = await session.run(
        input,
        validationManifest.domainId,
        new Date(validationManifest.validTime),
        effectiveResolution,
      );
      const metrics = verificationMetrics(
        deriveDisplayFields(result.physical),
        deriveDisplayFields(target),
        VARIABLE_VISUALS.map(({ shortName, sourceUnit }) => ({
          name: shortName,
          unit: sourceUnit,
        })),
      );
      publishFrame(
        {
          values: result.physical,
          validTime: validationManifest.validTime,
        },
        [],
      );
      setReferenceField(target);
      setComparisonMetrics(metrics);
      setComparisonView("prediction");
      setValidationParity(
        effectiveResolution === 27
          ? parityMetrics(result.physical, pythonPrediction)
          : null,
      );
      setStepMilliseconds(result.elapsedMilliseconds);
      setSequenceProgress(1);
      updateStep(0, {
        status: "complete",
        inferenceMilliseconds: result.elapsedMilliseconds,
      });
      setRuntimeState("success");
      setCurrentAction(
        effectiveResolution === 27
          ? `Cached demo complete · WRF comparison ready · ${result.elapsedMilliseconds.toFixed(1)} ms`
          : `${effectiveResolution} km conditioning probe complete · fixed 99 × 99 output`,
      );
    } catch (verificationError) {
      setRuntimeState("error");
      updateStep(0, { status: "error" });
      setCurrentAction("Held-out WRF comparison failed");
      setError(
        verificationError instanceof Error
          ? verificationError.message
          : "Held-out WRF comparison failed",
      );
    }
  }

  async function runSequence() {
    beginRun("Reading prepared GFS trajectory bundle", "bundle");
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
      let producedFrames: ForecastFrame[] = [];
      for (
        let step = 0;
        step < sequenceBundle.manifest.outputTimes.length;
        step += 1
      ) {
        updateStep(step, { status: "running" });
        setCurrentAction(
          `Running bundle step ${step + 1}/${sequenceBundle.manifest.outputTimes.length}`,
        );
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
        producedFrames = publishFrame(
          {
            values: result.physical,
            validTime: `${sequenceBundle.manifest.outputTimes[step]}Z`,
          },
          producedFrames,
        );
        setSequenceProgress(step + 1);
      }
      setStepMilliseconds(
        timings.reduce((sum, timing) => sum + timing, 0) / timings.length,
      );
      setRuntimeState("success");
      setCurrentAction(
        `Bundle complete · ${producedFrames.length} fields ready for playback`,
      );
    } catch (sequenceError) {
      setRuntimeState("error");
      failActiveSteps();
      setCurrentAction("Prepared sequence run failed");
      setError(
        sequenceError instanceof Error
          ? sequenceError.message
          : "Sequence inference failed",
      );
    }
  }

  async function runLiveGfs() {
    beginRun(
      `Validating requested ${gfsProvider.toUpperCase()} GFS cycle`,
      "live",
    );
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
      setCurrentAction(
        `Loading normalized WPS static geography · month ${cycle.getUTCMonth() + 1}`,
      );
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
        setCurrentAction(
          `GFS frame ${frameNumber + 1}/${requirements.gfsFrames} · ${
            next.completedFields
          }/${next.totalFields} fields · ${next.stage}`,
        );
      };
      let frameBytes = 0;
      let previous = await fetchGfsFrame(
        prior.cycle,
        prior.forecastHour,
        (next) => {
          frameBytes = next.totalBytes;
          progressForFrame(next);
        },
        gfsProvider,
      );
      accountedBytes += frameBytes;
      frameNumber += 1;
      frameBytes = 0;
      let current = await fetchGfsFrame(cycle, 0, (next) => {
        frameBytes = next.totalBytes;
        progressForFrame(next);
      }, gfsProvider);
      accountedBytes += frameBytes;
      frameNumber += 1;
      const timings: number[] = [];
      let producedFrames: ForecastFrame[] = [];
      for (let step = 0; step < requirements.outputSteps; step += 1) {
        updateStep(step, { status: "running" });
        const forecastTime = new Date(
          cycle.getTime() + (step + 1) * 3 * 60 * 60 * 1000,
        );
        setCurrentAction(
          `Running FiLMeR step ${step + 1}/${requirements.outputSteps} · valid ${forecastTime
            .toISOString()
            .slice(0, 16)} UTC`,
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
        producedFrames = publishFrame(
          {
            values: result.physical,
            validTime: forecastTime.toISOString(),
          },
          producedFrames,
        );
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
            gfsProvider,
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
      setCurrentAction(
        `Operational sequence complete · ${producedFrames.length} fields ready for playback`,
      );
    } catch (liveError) {
      setRuntimeState("error");
      failActiveSteps();
      setCurrentAction("Operational GFS run failed");
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
      : runKind === "validation" || runKind === "parity"
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
    if (!displayedField) return null;
    const start = variableIndex * 99 * 99;
    const end = start + 99 * 99;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let index = start; index < end; index += 1) {
      const value = displayValue(variableIndex, displayedField[index]);
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      sum += value;
    }
    return {
      minimum,
      maximum,
      mean: sum / (99 * 99),
      unit: VARIABLE_VISUALS[variableIndex].displayUnit,
    };
  }, [displayedField, variableIndex]);
  const selectedVerification = comparisonMetrics?.[variableIndex] ?? null;

  function downloadLatestOutput() {
    if (!predictionField) return;
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
        predictionField.buffer,
        predictionField.byteOffset,
        predictionField.byteLength,
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
          <small>{gfsProvider.toUpperCase()} · UTC</small>
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
                Regional weather fields from GFS.
              </h1>
            </div>
            <div className="self-end border-t border-stone-500/40 pt-4">
              <p className="max-w-[46ch] text-sm leading-relaxed text-stone-600">
                Start with a verified example, or choose NOAA/UCAR inputs.
                FiLMeR runs locally.
              </p>
            </div>
          </div>
          <div className="field-readout">
            <div>
              <span>field</span>
              <strong>
                {comparisonView === "difference" ? "error · " : ""}
                {variableLabels[variableIndex]}
              </strong>
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
          <div className="map-toolbar">
            <div
              aria-label="Displayed dataset"
              className="map-view-toggle"
              role="group"
            >
              {(["prediction", "reference", "difference"] as const).map(
                (view) => (
                  <button
                    className={
                      comparisonView === view
                        ? "map-view-toggle-active"
                        : ""
                    }
                    disabled={!referenceField && view !== "prediction"}
                    key={view}
                    onClick={() => setComparisonView(view)}
                    type="button"
                  >
                    {view === "prediction"
                      ? "FiLMeR"
                      : view === "reference"
                        ? "WRF target"
                        : "FiLMeR − WRF"}
                  </button>
                ),
              )}
            </div>
            <div
              aria-label="Forecast time controls"
              className="time-controls"
              role="group"
            >
              <button
                aria-label="Previous forecast time"
                disabled={forecastFrames.length < 2}
                onClick={() => {
                  setIsPlaying(false);
                  setActiveFrameIndex((current) =>
                    Math.max(0, current - 1),
                  );
                }}
                type="button"
              >
                <CaretLeft size={15} weight="bold" />
              </button>
              <button
                aria-label={isPlaying ? "Pause animation" : "Play animation"}
                disabled={forecastFrames.length < 2}
                onClick={() => setIsPlaying((current) => !current)}
                type="button"
              >
                {isPlaying ? (
                  <Pause size={15} weight="fill" />
                ) : (
                  <Play size={15} weight="fill" />
                )}
              </button>
              <button
                aria-label="Next forecast time"
                disabled={forecastFrames.length < 2}
                onClick={() => {
                  setIsPlaying(false);
                  setActiveFrameIndex((current) =>
                    Math.min(forecastFrames.length - 1, current + 1),
                  );
                }}
                type="button"
              >
                <CaretRight size={15} weight="bold" />
              </button>
              <span>
                {forecastFrames.length
                  ? `${activeFrameIndex + 1}/${forecastFrames.length}`
                  : "single field"}
              </span>
            </div>
          </div>
          {forecastFrames.length > 1 ? (
            <div className="time-scrubber">
              <input
                aria-label="Displayed forecast time"
                max={forecastFrames.length - 1}
                min={0}
                onChange={(event) => {
                  setIsPlaying(false);
                  setActiveFrameIndex(Number(event.target.value));
                }}
                step={1}
                type="range"
                value={activeFrameIndex}
              />
              <div>
                <span>
                  {forecastFrames[0].validTime.slice(0, 16).replace("T", " ")}
                </span>
                <span>
                  {forecastFrames
                    .at(-1)!
                    .validTime.slice(0, 16)
                    .replace("T", " ")}
                </span>
              </div>
            </div>
          ) : null}
          <ForecastMap
            values={displayedField}
            variableIndex={variableIndex}
            domainId={domainId}
            unit={VARIABLE_VISUALS[variableIndex].displayUnit}
            scaleValues={
              comparisonView === "difference"
                ? null
                : referenceDisplayField
            }
            mode={
              comparisonView === "difference" ? "difference" : "field"
            }
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
              disabled={!predictionField}
              onClick={downloadLatestOutput}
              type="button"
            >
              <DownloadSimple size={15} weight="bold" />
              Download output
            </button>
          </div>
          {selectedVerification && validationManifest ? (
            <section className="verification-results" aria-live="polite">
              <div className="verification-results-heading">
                <div>
                  <p className="eyebrow">Held-out WRF comparison</p>
                  <h2>
                    {variableLabels[variableIndex]} ·{" "}
                    {validationResolution === 27
                      ? "matched d01 case"
                      : `${validationResolution} km conditioning sensitivity`}
                  </h2>
                </div>
                <span>
                  {validationResolution === 27
                    ? `Browser↔Python max |Δ| ${
                        validationParity
                          ? validationParity.maxAbs.toExponential(2)
                          : "—"
                      }`
                    : "Off-training conditioning probe"}
                </span>
              </div>
              <div className="verification-grid">
                <div>
                  <span>mean bias</span>
                  <strong>
                    {displayValue(
                      variableIndex,
                      selectedVerification.bias,
                    ).toPrecision(4)}{" "}
                    {VARIABLE_VISUALS[variableIndex].displayUnit}
                  </strong>
                </div>
                <div>
                  <span>MAE</span>
                  <strong>
                    {displayValue(
                      variableIndex,
                      selectedVerification.mae,
                    ).toPrecision(4)}{" "}
                    {VARIABLE_VISUALS[variableIndex].displayUnit}
                  </strong>
                </div>
                <div>
                  <span>RMSE</span>
                  <strong>
                    {displayValue(
                      variableIndex,
                      selectedVerification.rmse,
                    ).toPrecision(4)}{" "}
                    {VARIABLE_VISUALS[variableIndex].displayUnit}
                  </strong>
                </div>
              </div>
              <div className="verification-table-wrap">
                <table className="verification-table">
                  <thead>
                    <tr>
                      <th>field</th>
                      <th>bias</th>
                      <th>MAE</th>
                      <th>RMSE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonMetrics!.map((metric, index) => (
                      <tr
                        className={
                          index === variableIndex
                            ? "verification-row-active"
                            : ""
                        }
                        key={metric.variable}
                        onClick={() => setVariableIndex(index)}
                      >
                        <td>{metric.variable}</td>
                        <td>
                          {displayValue(index, metric.bias).toPrecision(4)}
                        </td>
                        <td>
                          {displayValue(index, metric.mae).toPrecision(4)}
                        </td>
                        <td>
                          {displayValue(index, metric.rmse).toPrecision(4)}{" "}
                          {VARIABLE_VISUALS[index].displayUnit}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="verification-scope">
                {validationManifest.semantics.scope} Use the map tabs for the
                FiLMeR field, WRF target, and signed error.
              </p>
            </section>
          ) : null}
        </motion.div>

        <motion.aside
          initial={{ opacity: 0, x: 18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 20, delay: 0.08 }}
          className="order-first divide-y divide-stone-400/30 lg:order-none"
        >
          <section className="quick-start-panel p-5 md:p-8">
            <div className="quick-start-heading">
              <div>
                <p className="eyebrow">Start here</p>
                <h2>See a verified prediction now</h2>
              </div>
              <span>cached · no GFS wait</span>
            </div>
            <p>
              Runs the released checkpoint on a fixed training-pipeline GFS
              pair, then opens the matching WRF target and error maps.
            </p>
            <button
              className="demo-button"
              disabled={
                !validationManifest ||
                runtimeState === "loading" ||
                runtimeState === "running"
              }
              onClick={() => {
                setValidationResolution(27);
                void runHeldOutVerification(27);
              }}
              type="button"
            >
              <span>
                {runtimeState === "loading" && runKind === "validation"
                  ? "Loading verified model…"
                  : runtimeState === "running" && runKind === "validation"
                    ? "Running cached prediction…"
                    : "Run default prediction"}
              </span>
              <Lightning size={20} weight="fill" />
            </button>
            <div className="source-audit-strip">
              <div>
                <span>UCAR ↔ NOAA audit</span>
                <strong>
                  {gfsSourceAudit.summary.identicalRecords}/
                  {gfsSourceAudit.summary.totalRecords} exact
                </strong>
              </div>
              <div>
                <span>provider input Δ</span>
                <strong>
                  {gfsSourceAudit.summary.providerInducedInputMaxAbsDifference}
                </strong>
              </div>
              <InfoTip label="What the UCAR and NOAA comparison proves">
                The 40 FiLMeR predictor records in the audited 11 May 2024
                f000/f003 pair were byte-identical across UCAR GDEX and NOAA.
                This isolates provider effects; it does not measure temporal
                shift or forecast skill.
              </InfoTip>
            </div>
          </section>

          <section className="p-5 md:p-8">
            <div className="section-heading">
              <MapPin size={18} weight="bold" />
              <h2>Trained geography</h2>
              <InfoTip label="About supported geography">
                The released checkpoint was trained on India d01 and three 9
                km nested domains. New locations need a compatible geogrid and
                held-out validation.
              </InfoTip>
            </div>
            <p className="section-lede">Choose a domain seen during training.</p>
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
                  Multi-resolution, within evidence
                </p>
                <p className="mt-1 text-xs leading-relaxed text-stone-600">
                  One checkpoint learned 9 km and 27 km domains. Other values
                  remain experiments; the output is always 99 × 99.
                </p>
              </div>
            </div>
          </section>

          <section className="p-5 md:p-8">
            <div className="section-heading">
              <Gauge size={18} weight="bold" />
              <h2>Conditional horizon</h2>
              <InfoTip label="What conditional horizon means">
                Each 3-hour output uses two GFS states. FiLMeR does not feed
                its output back into itself, so a 96-hour product requires a
                complete GFS trajectory.
              </InfoTip>
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
              className="mt-3 grid grid-cols-5 gap-1"
            >
              {[3, 6, 24, 48, 96].map((hours) => (
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
                Needs GFS leads −3 to +{requirements.lastGfsLeadHours} h; this
                is conditional downscaling, not an autonomous forecast.
              </p>
            </div>
            <div className="mt-5 border-t border-stone-400/40 pt-5">
              <div className="control-label">
                <label htmlFor="gfs-source">GFS source</label>
                <InfoTip label="About GFS data sources">
                  NOAA is the operational default. UCAR GDEX is the archive
                  used by the training workflow; because it has no index, the
                  matching NOAA index locates byte ranges in the UCAR file.
                </InfoTip>
              </div>
              <select
                className="control mt-2"
                id="gfs-source"
                onChange={(event) =>
                  setGfsProvider(event.target.value as GfsProvider)
                }
                value={gfsProvider}
              >
                <option value="noaa">NOAA · operational mirror</option>
                <option value="ucar">UCAR GDEX · training-source archive</option>
              </select>
              <label
                className="mt-4 block text-sm font-medium"
                htmlFor="gfs-cycle"
              >
                Cycle (UTC)
              </label>
              <input
                id="gfs-cycle"
                className="control mt-3"
                type="datetime-local"
                step="21600"
                value={gfsCycle}
                onChange={(event) => setGfsCycle(event.target.value)}
              />
              <button
                className="primary-live-button mt-3"
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
                    : `Fetch ${gfsProvider.toUpperCase()} GFS & run ${horizon} h`}
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
              <p className="source-note">
                {gfsProvider === "ucar"
                  ? "Archive mode. UCAR can lag current cycles; the app stops if its byte ranges do not decode as the matching NCEP product."
                  : "Operational default. Only 20 required GRIB records are range-read, decoded, and cropped in-browser."}
              </p>
              {gfsProvider === "ucar" ? (
                <button
                  className="archive-preset-button"
                  onClick={() => {
                    setGfsCycle("2024-05-11T00:00");
                    setHorizon(3);
                  }}
                  type="button"
                >
                  Use 11 May 2024 archive cycle · 3 h
                </button>
              ) : null}
            </div>
            <details className="advanced-panel mt-5">
              <summary>Use a prepared trajectory bundle</summary>
              <div className="pt-4">
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
            </details>
          </section>

          <details className="aside-details">
            <summary>
              <Cpu size={18} weight="bold" />
              <span>Runtime & advanced tools</span>
              <small>verified WASM default</small>
            </summary>
            <div className="p-5 md:p-8">
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
              <option value="wasm">Verified — WebAssembly fp32</option>
              <option value="webgpu" disabled={!webGpuAvailable()}>
                Experimental — WebGPU fp16
              </option>
            </select>
            {requestedBackend === "webgpu" ? (
              <p className="mt-2 border-l-2 border-[#a6552c] pl-3 text-[11px] leading-relaxed text-[#7f3d20]">
                WebGPU executes successfully, but its browser parity is not
                acceptable for scientific output. Use the fp32 WASM path for
                validated fields.
              </p>
            ) : null}
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

            <details className="advanced-panel mt-5">
              <summary>Scientific verification & sensitivity tools</summary>
              <div className="pt-4">
                <button
                  className="run-button"
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
                        : "Run numerical parity fixture"}
                  </span>
                  {runtimeState === "success" ? (
                    <CheckCircle size={18} weight="bold" />
                  ) : runtimeState === "loading" ? (
                    <CloudArrowDown size={18} weight="bold" />
                  ) : (
                    <ArrowRight size={18} weight="bold" />
                  )}
                </button>
                <div className="verification-card">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold">
                    Held-out WRF verification
                  </p>
                  <p className="mt-1 text-[11px] leading-relaxed text-stone-600">
                    GFS 00Z/03Z → WRF 06Z · d01 · 1 Jan 2025. Compares all six
                    physical outputs.
                  </p>
                </div>
                <span className="verification-badge">sample_test</span>
              </div>
              <label
                className="mt-4 flex items-center justify-between gap-3 text-[11px]"
                htmlFor="resolution-probe"
              >
                <span>Resolution conditioning value</span>
                <span className="font-mono">{validationResolution} km</span>
              </label>
              <input
                aria-describedby="resolution-probe-note"
                className="mt-2 w-full accent-[#a6552c]"
                disabled={
                  runtimeState === "loading" || runtimeState === "running"
                }
                id="resolution-probe"
                max={54}
                min={1}
                onChange={(event) =>
                  setValidationResolution(Number(event.target.value))
                }
                step={1}
                type="range"
                value={validationResolution}
              />
              <div className="mt-1 flex justify-between font-mono text-[9px] text-stone-500">
                <span>1 km probe</span>
                <span>9 km trained</span>
                <span>27 km d01 truth</span>
                <span>54 km probe</span>
              </div>
              <p
                className={`mt-3 text-[10px] leading-relaxed ${
                  validationResolution === 27
                    ? "text-stone-500"
                    : "border-l-2 border-[#a6552c] pl-3 text-[#7f3d20]"
                }`}
                id="resolution-probe-note"
              >
                {validationResolution === 27
                  ? "Matched validation: fixed 99 × 99 output over the trained 27 km d01 geogrid."
                  : `Sensitivity experiment only: changing the conditioning scalar to ${validationResolution} km does not create ${validationResolution} km physics or new spatial information. The output remains 99 × 99 and is compared with a 27 km WRF target.`}
              </p>
              <button
                className="sequence-button mt-3"
                disabled={
                  !validationManifest ||
                  runtimeState === "loading" ||
                  runtimeState === "running"
                }
                onClick={() => void runHeldOutVerification()}
                type="button"
              >
                <span>
                  {validationResolution === 27
                    ? "Run FiLMeR vs held-out WRF"
                    : `Run ${validationResolution} km conditioning probe`}
                </span>
                <Pulse size={18} weight="bold" />
              </button>
            </div>
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
              </div>
            </details>
            </div>
          </details>
          {error ? (
            <p className="m-5 border-l-2 border-[#a6552c] pl-3 text-xs leading-relaxed text-[#7f3d20] md:m-8">
              {error}
            </p>
          ) : null}
        </motion.aside>
      </section>

      <details className="workspace-details mx-auto max-w-[1400px]">
        <summary>
          <div>
            <span>Evidence & run details</span>
            <strong>
              {runtimeState === "running" || runtimeState === "loading"
                ? currentAction
                : comparisonMetrics
                  ? "WRF metrics ready"
                  : `${gfsSourceAudit.summary.identicalRecords}/${gfsSourceAudit.summary.totalRecords} UCAR↔NOAA records exact`}
            </strong>
          </div>
          <span className={`workspace-status workspace-status-${runtimeState}`}>
            {runtimeState}
          </span>
        </summary>
        <div className="workspace-details-body">
      <section
        aria-label="Scientific alignment checks"
        className="science-evidence"
      >
        <div className="science-evidence-heading">
          <p className="eyebrow">Scientific alignment</p>
          <h2>What now matches the paper—and what is actually verified</h2>
        </div>
        <div>
          <span>Rendering</span>
          <strong>Paper palettes · north-up</strong>
          <p>
            Variable-specific Matplotlib palettes and{" "}
            <code>origin=&apos;lower&apos;</code>; PSFC in hPa plus derived
            wind speed and RH2.
          </p>
        </div>
        <div>
          <span>Provider equivalence</span>
          <strong>40 / 40 records exact</strong>
          <p>
            {formatBytes(gfsSourceAudit.summary.selectedBytes)} SHA-256 audited
            across UCAR and NOAA for one same-cycle pair.
          </p>
        </div>
        <div>
          <span>Error evidence</span>
          <strong>Held-out WRF case</strong>
          <p>
            Metrics for six model outputs and two derived diagnostics. WRF is
            supervision—not an observation or aggregate skill result.
          </p>
        </div>
      </section>

      <section className="verification-results mx-auto max-w-[1400px] border-x border-t border-stone-400/30">
        <div className="verification-results-heading">
          <div>
            <p className="eyebrow">Held-out WRF comparison</p>
            <h2>Prediction, supervision target, and signed error</h2>
          </div>
          <div className="verification-scope">
            <strong>
              {comparisonMetrics
                ? `${validationResolution} km conditioning · d01 · 2025-01-01 06Z`
                : "Ready to run"}
            </strong>
            <span>one sample · WRF is not an observation</span>
          </div>
        </div>
        {comparisonMetrics ? (
          <>
            <div className="verification-metric-grid">
              {comparisonMetrics.map((metric, index) => (
                <button
                  className={`verification-metric ${
                    variableIndex === index
                      ? "verification-metric-active"
                      : ""
                  }`}
                  key={metric.variable}
                  onClick={() => setVariableIndex(index)}
                  type="button"
                >
                  <span>
                    {metric.variable} ·{" "}
                    {VARIABLE_VISUALS[index].displayUnit}
                  </span>
                  <strong>
                    {displayValue(index, metric.rmse).toPrecision(4)}
                  </strong>
                  <small>RMSE</small>
                  <dl>
                    <div>
                      <dt>bias</dt>
                      <dd>
                        {displayValue(index, metric.bias).toPrecision(3)}
                      </dd>
                    </div>
                    <div>
                      <dt>MAE</dt>
                      <dd>
                        {displayValue(index, metric.mae).toPrecision(3)}
                      </dd>
                    </div>
                  </dl>
                </button>
              ))}
            </div>
            <div className="verification-footnote">
              <p>
                The target is the paper pipeline’s held-out WRF tensor after
                the exact 99 × 99 bilinear transform. State fields are compared
                in physical units; precipitation is RAINC + RAINNC.
              </p>
              <p className="font-mono">
                {validationResolution === 27
                  ? `Browser ↔ Python physical parity max |Δ| ${
                      validationParity
                        ? validationParity.maxAbs.toExponential(2)
                        : "pending"
                    }`
                  : "No Python parity reference is claimed for an off-training conditioning value."}
              </p>
            </div>
          </>
        ) : (
          <div className="verification-empty">
            <p>
              Run the default prediction above to unlock all six metrics and
              the FiLMeR / WRF / error map switcher.
            </p>
            <span>
              Live future cycles remain “verification pending” until a
              time-matched WRF analysis or observations become available.
            </span>
          </div>
        )}
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
            <div
              aria-live="polite"
              className={`execution-now execution-now-${runtimeState}`}
            >
              <div>
                <span>current operation</span>
                <strong>{currentAction}</strong>
              </div>
              <div>
                <span>end-to-end elapsed</span>
                <strong>
                  {(runElapsedMilliseconds / 1000).toFixed(1)} s
                </strong>
              </div>
            </div>
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
                label="Static geography / prepared input"
                detail={
                  runKind === "parity"
                    ? "preassembled parity tensor · mean-static placeholder"
                    : runKind === "validation"
                      ? "checksum-identical d01 WPS geogrid · January"
                  : staticReady
                    ? `month ${String(
                        new Date(`${gfsCycle}:00Z`).getUTCMonth() + 1,
                      ).padStart(2, "0")} · 30 × 127 × 137`
                    : sequenceBundle
                      ? "normalized static tensor in bundle"
                      : "cached monthly d01 artifact"
                }
                percent={
                  staticReady ||
                  Boolean(sequenceBundle) ||
                  runKind === "parity"
                    ? 100
                    : 0
                }
                state={
                  staticReady ||
                  sequenceBundle ||
                  runKind === "parity"
                    ? "complete"
                    : "waiting"
                }
              />
              <StageBar
                label="GFS boundary conditions"
                detail={
                  runKind === "validation"
                    ? "held-out 00Z + 03Z pair · exact training normalization"
                    : runKind === "parity"
                      ? "committed preassembled numerical fixture"
                  : gfsProgress
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
              <StageBar
                label="Reference comparison"
                detail={
                  comparisonMetrics
                    ? `six fields · WRF target · ${
                        validationResolution === 27
                          ? "matched 27 km case"
                          : `${validationResolution} km conditioning probe`
                      }`
                    : runKind === "live" || runKind === "bundle"
                      ? "verification pending; no future WRF target in this run"
                      : "held-out WRF target available on request"
                }
                percent={comparisonMetrics ? 100 : 0}
                state={
                  comparisonMetrics
                    ? "complete"
                    : runKind === "validation" &&
                        runtimeState === "running"
                      ? "active"
                      : runKind === "validation" &&
                          runtimeState === "error"
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
                      <tr
                        className={
                          forecastFrames[step.index] &&
                          activeFrameIndex === step.index
                            ? "step-row-active"
                            : forecastFrames[step.index]
                              ? "step-row-ready"
                              : ""
                        }
                        key={step.index}
                        onClick={() => {
                          if (!forecastFrames[step.index]) return;
                          setIsPlaying(false);
                          setActiveFrameIndex(step.index);
                        }}
                      >
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
                    Start with the cached prediction, or choose a domain, GFS
                    source, cycle, and horizon for an archive/operational run.
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
              WASM uses the parity-verified fp32 artifact and is the default.
              WebGPU uses the smaller fp16 artifact only as an explicit
              experimental execution-path test.
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
        </div>
      </details>

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
