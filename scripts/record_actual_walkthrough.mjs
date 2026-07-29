import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const url = process.env.FILMER_RECORD_URL ?? "http://127.0.0.1:4173/";
const output = resolve(
  process.env.FILMER_RECORD_OUTPUT ?? "/tmp/filmer-actual-app-capture.mp4",
);
const browser = await puppeteer.launch({
  executablePath:
    process.env.CHROME_PATH ??
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  protocolTimeout: 420_000,
  userDataDir:
    process.env.FILMER_CHROME_PROFILE ??
    "/tmp/filmer-walkthrough-chrome-profile",
  args: [
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
const pageErrors = [];
let recorder;
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("console", (message) => {
  if (
    message.type() === "error" &&
    !message.text().startsWith("Failed to load resource:")
  ) {
    pageErrors.push(message.text());
  }
});

try {
  await page.goto(url, { waitUntil: "networkidle2", timeout: 120_000 });
  await page.waitForSelector(".run-button:not([disabled])", {
    timeout: 120_000,
  });
  recorder = await page.screencast({
    path: output,
    format: "mp4",
    fps: 30,
    quality: 20,
    speed: 1.6,
  });

  await new Promise((resolveDelay) => setTimeout(resolveDelay, 2_000));
  const domains = await page.$$(".domain-choice");
  await domains[2].click();
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 850));
  await page.select(".variable-select select", "5");
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 850));
  await page.select(".timestamp-select select", "4");
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_100));
  await page.click(".run-button");

  console.log("Running the real NOAA download and local ONNX model…");
  await page.waitForSelector(".app-status-success", { timeout: 360_000 });
  console.log("Forecast ready; recording the result interactions…");
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 2_500));

  await page.$eval(".forecast", (element) =>
    element.scrollIntoView({ behavior: "smooth", block: "center" }),
  );
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_800));
  await page.click('button[aria-label="Play animation"]');
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 4_500));

  await page.click('button[aria-label="Zoom in"]');
  await page.click('button[aria-label="Zoom in"]');
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 1_000));
  const map = await page.$(".map-shell svg");
  const bounds = await map.boundingBox();
  if (bounds) {
    const centerX = bounds.x + bounds.width / 2;
    const centerY = bounds.y + bounds.height / 2;
    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX - 120, centerY + 55, { steps: 20 });
    await page.mouse.up();
  }
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 3_500));
  await page.click('button[aria-label="Reset map view"]');
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 2_000));

  await recorder.stop();
  recorder = undefined;
  if (pageErrors.length) {
    throw new Error(`Browser errors: ${pageErrors.join(" | ")}`);
  }
  console.log(`Recorded actual FiLMeR run to ${output}`);
} finally {
  if (recorder) {
    await recorder.stop().catch(() => {});
  }
  await browser.close();
}
