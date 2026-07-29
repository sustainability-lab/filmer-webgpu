#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const framesDirectory = join(root, "docs", "walkthrough", "frames");
const publicDirectory = join(root, "public", "walkthrough");
mkdirSync(framesDirectory, { recursive: true });
mkdirSync(publicDirectory, { recursive: true });

const outline = JSON.parse(
  readFileSync(join(root, "public", "data", "india-outline-soi.geojson"), "utf8"),
);
const geometry = outline.features[0].geometry;
const polygons =
  geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];

function indiaPath() {
  const left = 724;
  const top = 142;
  const width = 488;
  const height = 486;
  const west = 66;
  const east = 99;
  const south = 6;
  const north = 38;
  const project = ([longitude, latitude]) => [
    left + ((longitude - west) / (east - west)) * width,
    top + ((north - latitude) / (north - south)) * height,
  ];
  return polygons
    .flatMap((polygon) =>
      polygon.map((ring) =>
        ring
          .map((point, index) => {
            const [x, y] = project(point);
            return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
          })
          .join(" ") + " Z",
      ),
    )
    .join(" ");
}

const boundaryPath = indiaPath();
const steps = [
  {
    eyebrow: "STEP 1 OF 3",
    title: "Choose a trained domain",
    body: "Start with India at 27 km, or one of the trained 9 km regional domains.",
    active: 0,
    variable: "Temperature",
    timestamps: "8 · 24 hours",
    action: "Download data & run",
    status: "Ready",
    mapLabel: "India · 27 km",
  },
  {
    eyebrow: "STEP 2 OF 3",
    title: "Pick a field and timestamps",
    body: "Choose the surface variable, then request 1, 2, 4 or 8 timestamps at 3-hour spacing.",
    active: 1,
    variable: "Rainfall",
    timestamps: "8 · 24 hours",
    action: "Download data & run",
    status: "Ready",
    mapLabel: "Rainfall · 8 timestamps",
  },
  {
    eyebrow: "STEP 3 OF 3",
    title: "Download data and run",
    body: "The browser fetches the model and current GFS inputs, then runs FiLMeR locally on this device.",
    active: 2,
    variable: "Rainfall",
    timestamps: "8 · 24 hours",
    action: "Running in this browser…",
    status: "GFS weather data · 68%",
    mapLabel: "No server-side inference",
  },
  {
    eyebrow: "FORECAST READY",
    title: "Explore, animate, download",
    body: "Pan and zoom the map, use Play for multiple timestamps, then download the forecast data.",
    active: 3,
    variable: "Rainfall",
    timestamps: "8 · 24 hours",
    action: "Download forecast",
    status: "8 timestamps complete · 31 s",
    mapLabel: "29 Jul · 15:00 UTC",
  },
];

const escapeXml = (value) =>
  value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

function stepDots(active) {
  return [0, 1, 2]
    .map(
      (index) => `
        <circle cx="${80 + index * 34}" cy="86" r="7"
          fill="${index <= Math.min(active, 2) ? "#0b6b5b" : "#d7dfdc"}"/>
        ${index < 2 ? `<rect x="${87 + index * 34}" y="83" width="20" height="6" rx="3"
          fill="${index < active ? "#0b6b5b" : "#d7dfdc"}"/>` : ""}`,
    )
    .join("");
}

function frameSvg(step) {
  const running = step.active === 2;
  const ready = step.active === 3;
  const progressWidth = running ? 292 : ready ? 430 : 0;
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="page" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f6faf7"/>
      <stop offset="1" stop-color="#e7f0ec"/>
    </linearGradient>
    <linearGradient id="heat" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0" stop-color="#12245a"/>
      <stop offset=".22" stop-color="#1f72b7"/>
      <stop offset=".46" stop-color="#27a98b"/>
      <stop offset=".68" stop-color="#d7d74f"/>
      <stop offset=".84" stop-color="#f08b33"/>
      <stop offset="1" stop-color="#b92245"/>
    </linearGradient>
    <linearGradient id="bar" x1="0" x2="1">
      <stop offset="0" stop-color="#12245a"/>
      <stop offset=".25" stop-color="#1f72b7"/>
      <stop offset=".5" stop-color="#27a98b"/>
      <stop offset=".72" stop-color="#d7d74f"/>
      <stop offset="1" stop-color="#b92245"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="18" flood-color="#17332c" flood-opacity=".12"/>
    </filter>
    <clipPath id="india"><path d="${boundaryPath}"/></clipPath>
  </defs>
  <rect width="1280" height="720" fill="url(#page)"/>
  <text x="52" y="48" font-family="Arial, Helvetica, sans-serif" font-size="23" font-weight="700" fill="#113c34">FiLMeR</text>
  <text x="142" y="48" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#57706a">Forecast locally, from current GFS data</text>
  <rect x="52" y="67" width="1176" height="606" rx="24" fill="#fff" filter="url(#shadow)"/>
  <rect x="52" y="67" width="570" height="606" rx="24" fill="#fbfdfc"/>
  <path d="M622 67V673" stroke="#e2e9e6"/>
  ${stepDots(step.active)}
  <text x="80" y="135" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700" letter-spacing="1.6" fill="#0b6b5b">${step.eyebrow}</text>
  <text x="80" y="179" font-family="Arial, Helvetica, sans-serif" font-size="30" font-weight="700" fill="#17332c">${escapeXml(step.title)}</text>
  <foreignObject x="80" y="198" width="472" height="62">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font:16px/1.45 Arial,Helvetica,sans-serif;color:#5b6e69">${escapeXml(step.body)}</div>
  </foreignObject>
  <text x="80" y="294" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" letter-spacing="1.2" fill="#75847f">DOMAIN</text>
  <rect x="80" y="307" width="444" height="52" rx="10" fill="${step.active === 0 ? "#e2f1ec" : "#f2f5f4"}" stroke="${step.active === 0 ? "#0b6b5b" : "#d8e0dd"}"/>
  <text x="98" y="339" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#17332c">India</text>
  <text x="468" y="339" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#57706a">27 km</text>
  <text x="80" y="397" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" letter-spacing="1.2" fill="#75847f">VARIABLE</text>
  <rect x="80" y="410" width="214" height="52" rx="10" fill="${step.active === 1 ? "#e2f1ec" : "#f2f5f4"}" stroke="${step.active === 1 ? "#0b6b5b" : "#d8e0dd"}"/>
  <text x="98" y="442" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#17332c">${step.variable}</text>
  <text x="312" y="397" font-family="Arial, Helvetica, sans-serif" font-size="11" font-weight="700" letter-spacing="1.2" fill="#75847f">TIMESTAMPS</text>
  <rect x="312" y="410" width="212" height="52" rx="10" fill="${step.active === 1 ? "#e2f1ec" : "#f2f5f4"}" stroke="${step.active === 1 ? "#0b6b5b" : "#d8e0dd"}"/>
  <text x="330" y="442" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#17332c">${step.timestamps}</text>
  <rect x="80" y="493" width="444" height="58" rx="12" fill="${step.active >= 2 ? "#0b6b5b" : "#17332c"}"/>
  <text x="302" y="529" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#fff">${escapeXml(step.action)}</text>
  ${step.active >= 2 ? `
    <text x="80" y="582" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#36534c">${escapeXml(step.status)}</text>
    <rect x="80" y="597" width="444" height="8" rx="4" fill="#e4ebe8"/>
    <rect x="80" y="597" width="${progressWidth}" height="8" rx="4" fill="#0b6b5b"/>
    <text x="80" y="635" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#6c7d78">${ready ? "Model 1.2 s  ·  GFS 27.9 s  ·  Inference 1.9 s" : "The GFS download is usually the longest step."}</text>
  ` : `<text x="80" y="595" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#6c7d78">Only trained domains and 3-hour timestamps are offered.</text>`}
  <rect x="650" y="95" width="550" height="548" rx="18" fill="#eef3f1"/>
  <rect x="671" y="116" width="508" height="507" rx="14" fill="#dde8e3"/>
  <g clip-path="url(#india)">
    <rect x="680" y="126" width="520" height="510" fill="url(#heat)"/>
    <g opacity=".13" stroke="#fff">
      ${Array.from({ length: 20 }, (_, index) => `<path d="M680 ${143 + index * 24}H1200"/>`).join("")}
      ${Array.from({ length: 20 }, (_, index) => `<path d="M${700 + index * 25} 126V636"/>`).join("")}
    </g>
  </g>
  <path d="${boundaryPath}" fill="none" stroke="#fff" stroke-width="2.4" stroke-linejoin="round"/>
  <rect x="687" y="133" width="198" height="34" rx="17" fill="#fff" fill-opacity=".92"/>
  <text x="786" y="155" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#17332c">${escapeXml(step.mapLabel)}</text>
  <rect x="1140" y="184" width="12" height="266" rx="6" fill="url(#bar)"/>
  <text x="1159" y="194" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#36534c">high</text>
  <text x="1159" y="451" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#36534c">low</text>
  ${ready ? `
    <circle cx="919" cy="590" r="26" fill="#fff"/>
    <path d="M913 578L932 590L913 602Z" fill="#0b6b5b"/>
    <text x="959" y="595" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#17332c">Play 8 timestamps</text>
  ` : ""}
  <text x="1199" y="698" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#6f817b">Everything runs in your browser · current weather data downloaded on demand</text>
</svg>`;
}

for (const [index, step] of steps.entries()) {
  const svgPath = join(framesDirectory, `frame-${index + 1}.svg`);
  const pngPath = join(framesDirectory, `frame-${index + 1}.png`);
  await import("node:fs").then(({ writeFileSync }) =>
    writeFileSync(svgPath, frameSvg(step)),
  );
  execFileSync("inkscape", [
    svgPath,
    "--export-type=png",
    "--export-filename",
    pngPath,
    "--export-width=1280",
    "--export-height=720",
  ]);
}

const output = join(publicDirectory, "filmer-web-walkthrough.mp4");
execFileSync(
  "ffmpeg",
  [
    "-y",
    ...steps.flatMap((_, index) => [
      "-loop",
      "1",
      "-t",
      "11",
      "-i",
      join(framesDirectory, `frame-${index + 1}.png`),
    ]),
    "-filter_complex",
    "[0:v]fps=30,format=yuv420p[v0];[1:v]fps=30,format=yuv420p[v1];[2:v]fps=30,format=yuv420p[v2];[3:v]fps=30,format=yuv420p[v3];[v0][v1]xfade=transition=fade:duration=1.333:offset=9.667[x1];[x1][v2]xfade=transition=fade:duration=1.333:offset=19.334[x2];[x2][v3]xfade=transition=fade:duration=1.333:offset=29.001[v]",
    "-map",
    "[v]",
    "-c:v",
    "libx264",
    "-preset",
    "slow",
    "-crf",
    "22",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    "-t",
    "40",
    output,
  ],
  { stdio: "inherit" },
);

execFileSync("ffmpeg", [
  "-y",
  "-i",
  join(framesDirectory, "frame-4.png"),
  "-q:v",
  "3",
  join(publicDirectory, "filmer-web-walkthrough-poster.jpg"),
]);

console.log(`Wrote ${output}`);
