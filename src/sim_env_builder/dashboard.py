"""Rollout review dashboard. Pre-Kit safe: stdlib only, no Isaac imports.

Serves a self-contained page over an output directory: per-episode rollout
videos next to a milestone-progress timeline built from the per-episode JSON.
Clicking the timeline seeks the videos to that moment.
"""

from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# Camera stream in the video filename -> key used by the page layout.
CAMERA_KEYS = {
    "external_camera_rgb": "external",
    "external_camera_2_rgb": "base",
    "wrist_camera_rgb": "wrist",
}

_VIDEO_RE = re.compile(r"robot-cam-env(\d+)-([a-z0-9_]+)-episode-(\d+)\.mp4$")
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _rel(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()


def _load_milestones(run_dir: Path, episode: int) -> dict | None:
    path = run_dir / "milestones" / f"episode_{episode:03d}_milestones.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _scan_run(base: Path, run_dir: Path) -> dict | None:
    """Index one <task>/<timestamp> run directory; None when it holds no videos."""
    episodes: dict[int, dict] = {}
    viewport = None
    for f in sorted(run_dir.iterdir()):
        if not f.is_file() or f.suffix != ".mp4":
            continue
        if f.name.startswith("rl-video"):
            viewport = viewport or _rel(base, f)
            continue
        m = _VIDEO_RE.fullmatch(f.name)
        if m is None:
            continue
        cam = CAMERA_KEYS.get(m.group(2))
        if cam is None:
            continue
        entry = episodes.setdefault(int(m.group(3)), {"episode": int(m.group(3)), "videos": {}})
        # Lowest env id wins when several envs record the same episode index.
        entry["videos"].setdefault(cam, _rel(base, f))
    if not episodes and viewport is None:
        return None
    for episode, entry in episodes.items():
        entry["milestones"] = _load_milestones(run_dir, episode)
    return {
        "task": run_dir.parent.name,
        "timestamp": run_dir.name,
        "run_id": f"{run_dir.parent.name}/{run_dir.name}",
        "viewport": viewport,
        "episodes": [episodes[k] for k in sorted(episodes)],
    }


def build_index(base_dir: str | Path) -> dict:
    """Scan <base_dir>/<task>/<timestamp>/ runs into the JSON served at /api/runs.

    Runs are listed newest first across all tasks (the timestamp directory
    names sort lexicographically).
    """
    base = Path(base_dir)
    runs = []
    if base.is_dir():
        for task_dir in (p for p in base.iterdir() if p.is_dir()):
            for run_dir in (p for p in task_dir.iterdir() if p.is_dir()):
                run = _scan_run(base, run_dir)
                if run is not None:
                    runs.append(run)
    runs.sort(key=lambda r: (r["timestamp"], r["task"]), reverse=True)
    return {"root": str(base), "runs": runs}


def resolve_under(base_dir: Path, relpath: str) -> Path | None:
    """Resolve relpath inside base_dir; None when it escapes base_dir or is not a file."""
    base = base_dir.resolve()
    target = (base / relpath).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return None
    return target


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Single-range 'bytes=a-b' header -> (start, end) inclusive; None when unsatisfiable."""
    m = _RANGE_RE.fullmatch(header.strip())
    if m is None or (not m.group(1) and not m.group(2)):
        return None
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:
        # Suffix form 'bytes=-n': the last n bytes.
        start = max(size - int(m.group(2)), 0)
        end = size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


class DashboardServer(ThreadingHTTPServer):
    """HTTP server bound to one output directory."""

    def __init__(self, base_dir: str | Path, port: int, host: str = "0.0.0.0"):
        super().__init__((host, port), _Handler)
        self.base_dir = Path(base_dir).resolve()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DashboardServer

    # Video seeking issues hundreds of Range requests; per-request logging is noise.
    def log_message(self, format: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send_bytes(PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/runs":
            body = json.dumps(build_index(self.server.base_dir)).encode()
            self._send_bytes(body, "application/json")
        elif path.startswith("/files/"):
            target = resolve_under(self.server.base_dir, path[len("/files/"):])
            if target is None:
                self.send_error(404)
            else:
                self._send_file(target)
        else:
            self.send_error(404)

    def _send_bytes(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, target: Path) -> None:
        size = target.stat().st_size
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        header = self.headers.get("Range")
        rng = _parse_range(header, size) if header else None
        if header and rng is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = rng if rng else (0, size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(end - start + 1, 0)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with target.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers abort video streams mid-transfer; not an error.
                    return
                remaining -= len(chunk)


def serve(base_dir: str | Path, port: int = 8090, host: str = "0.0.0.0") -> None:
    """Serve the dashboard for base_dir until interrupted."""
    server = DashboardServer(base_dir, port, host)
    # Flush so the URL appears even when stdout is redirected to a log file.
    print(f"[sim-env-builder] dashboard for {server.base_dir}", flush=True)
    shown = "127.0.0.1" if host in ("", "0.0.0.0") else host
    print(f"[sim-env-builder] serving on {host or '0.0.0.0'}:{port} "
          f"(http://{shown}:{port}/)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# Self-contained page: inline CSS/JS only, no external resources. Chart colors
# are the validated reference categorical palette (light mode, fixed order).
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sim-env-builder dashboard</title>
<style>
:root {
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --good: #006300; --bad: #d03b3b;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
  padding: 12px 20px; background: var(--surface);
  border-bottom: 1px solid var(--grid);
}
header h1 { font-size: 16px; margin: 0 8px 0 0; }
button {
  font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--axis); border-radius: 6px; padding: 4px 14px;
  cursor: pointer; min-width: 76px;
}
button:hover { background: #f0efec; }
#clock { color: var(--muted); font-variant-numeric: tabular-nums; }
#layout {
  display: grid; grid-template-columns: 280px minmax(0, 1fr) 210px;
  gap: 16px; align-items: start;
  max-width: 1720px; margin: 0 auto; padding: 16px 20px 40px;
}
.sidebar {
  background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 10px;
  position: sticky; top: 12px;
  max-height: calc(100vh - 24px); overflow: auto;
}
.sidebar h2 { padding: 4px 6px 0; }
.item {
  padding: 6px 8px; border-radius: 6px; cursor: pointer;
  color: var(--ink-2); line-height: 1.3;
}
.item:hover { background: #f0efec; }
.item.active { background: #e9e8e3; color: var(--ink); }
.item .task { font-weight: 600; color: var(--ink); }
.item .ts { font-size: 12px; color: var(--muted);
            font-variant-numeric: tabular-nums; }
.dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; vertical-align: 1px;
}
.dot.ok { background: var(--good); }
.dot.bad { background: var(--bad); }
#playback { display: flex; align-items: center; gap: 10px; padding: 2px 6px 6px; }
@media (max-width: 1100px) {
  #layout { grid-template-columns: 1fr; }
  .sidebar { position: static; max-height: 260px; }
}
.panel {
  background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;
}
h2 { font-size: 13px; margin: 0 0 10px; color: var(--ink-2);
     text-transform: uppercase; letter-spacing: 0.04em; }
#videos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
#videos figure { margin: 0; min-width: 0; }
#videos video { width: 100%; background: #111; border-radius: 6px; display: block; }
#videos figcaption { color: var(--muted); font-size: 12px; margin-top: 4px; }
#chartwrap { position: relative; }
#chart { width: 100%; height: 280px; display: block; cursor: crosshair; }
#legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-bottom: 8px; }
#legend span { color: var(--ink-2); font-size: 12px; }
#legend i {
  display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 5px; vertical-align: -1px;
}
.notice { color: var(--muted); padding: 24px 0; text-align: center; }
.hidden { display: none; }
#tooltip {
  position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--grid); border-radius: 6px; padding: 6px 9px;
  font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  white-space: nowrap; font-variant-numeric: tabular-nums;
}
.badge {
  display: inline-block; font-weight: 600; font-size: 12px;
  padding: 2px 10px; border-radius: 999px; margin-right: 10px;
}
.badge.ok { color: var(--good); border: 1px solid var(--good); }
.badge.bad { color: var(--bad); border: 1px solid var(--bad); }
.instr { font-size: 15px; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 8px 24px; margin-top: 10px; }
.cols h3 { font-size: 12px; color: var(--muted); margin: 8px 0 4px;
           font-weight: 600; }
.cols ul { list-style: none; margin: 0; padding: 0; }
.cols li { padding: 1px 0; color: var(--ink-2); }
.cols li b { color: var(--ink); font-weight: 500; }
.mark { display: inline-block; width: 18px; font-weight: 700; }
.mark.ok { color: var(--good); }
.mark.bad { color: var(--bad); }
.kv { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.kv td { padding: 1px 0; color: var(--ink-2); }
.kv td:last-child { text-align: right; color: var(--ink); }
@media (max-width: 860px) { #videos { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Rollout review</h1>
</header>
<div id="layout">
<nav class="sidebar">
  <h2>Runs</h2>
  <div id="runlist"><div class="notice">loading...</div></div>
</nav>
<main>
  <section class="panel">
    <div id="videos">
      <figure><video id="v-external" controls preload="metadata"></video>
        <figcaption>external camera</figcaption></figure>
      <figure><video id="v-base" preload="metadata" muted></video>
        <figcaption>base camera</figcaption></figure>
      <figure><video id="v-wrist" preload="metadata" muted></video>
        <figcaption>wrist camera</figcaption></figure>
    </div>
  </section>
  <section class="panel" id="chartwrap">
    <h2>Milestone progress</h2>
    <div id="legend"></div>
    <canvas id="chart"></canvas>
    <div id="nochart" class="notice hidden"></div>
    <div id="tooltip" class="hidden"></div>
  </section>
  <section class="panel" id="verdict"><div class="notice">loading runs...</div></section>
</main>
<aside class="sidebar">
  <h2>Playback</h2>
  <div id="playback">
    <button id="playpause">Play</button>
    <span id="clock">0:00.0</span>
  </div>
  <h2>Episodes</h2>
  <div id="eplist"></div>
</aside>
</div>
<script>
"use strict";
const $ = (s) => document.querySelector(s);
// Validated reference categorical palette, light mode, fixed slot order.
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const CAMS = ["external", "base", "wrist"];
const vids = CAMS.map((c) => $("#v-" + c));
const master = vids[0];
const others = vids.slice(1);
const cv = $("#chart");
const ctx = cv.getContext("2d");
const MARGIN = { l: 36, r: 140, t: 12, b: 26 };
let index = null;
let runIdx = 0;
let epIdx = 0;
let chart = null; // {names, joints, thresholds, fps, duration}

const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fileURL = (p) => "/files/" + p.split("/").map(encodeURIComponent).join("/");
const fmtTime = (t) => {
  const m = Math.floor(t / 60), s = (t - m * 60).toFixed(1);
  return m + ":" + (s.length < 4 ? "0" : "") + s;
};

async function init() {
  index = await (await fetch("/api/runs")).json();
  if (!index.runs.length) {
    $("#runlist").innerHTML = '<div class="notice">no runs</div>';
    $("#verdict").innerHTML = '<div class="notice">no runs found under ' +
      esc(index.root) + "</div>";
    return;
  }
  renderRunList();
  selectRun(0);
}

// Green dot: some episode passed. Red dot: milestones exist and all failed.
// No dot: no milestone JSON in the run.
function verdictDot(episodes) {
  const graded = episodes.filter((e) => e.milestones);
  if (!graded.length) return "";
  const ok = graded.some((e) => e.milestones.success);
  return '<span class="dot ' + (ok ? "ok" : "bad") + '"></span>';
}

function renderRunList() {
  $("#runlist").innerHTML = index.runs.map((r, i) =>
    '<div class="item' + (i === runIdx ? " active" : "") + '" data-i="' + i + '">' +
    verdictDot(r.episodes) + '<span class="task">' + esc(r.task) + "</span>" +
    '<div class="ts">' + esc(r.timestamp) + "</div></div>").join("");
}

function renderEpList() {
  $("#eplist").innerHTML = episodeList().map((e, i) =>
    '<div class="item' + (i === epIdx ? " active" : "") + '" data-i="' + i + '">' +
    verdictDot([e]) + "episode " + esc(e.episode) + "</div>").join("");
}

$("#runlist").onclick = (ev) => {
  const item = ev.target.closest(".item");
  if (item) selectRun(Number(item.dataset.i));
};
$("#eplist").onclick = (ev) => {
  const item = ev.target.closest(".item");
  if (item) selectEpisode(Number(item.dataset.i));
};

function episodeList() {
  const r = index.runs[runIdx];
  if (r.episodes.length) return r.episodes;
  // Runs without per-episode camera videos still get their viewport video.
  if (r.viewport) {
    return [{ episode: "viewport", videos: { external: r.viewport }, milestones: null }];
  }
  return [];
}

function selectRun(i) {
  runIdx = i;
  renderRunList();
  if (episodeList().length) selectEpisode(0);
  else renderEpList();
}

function selectEpisode(i) {
  epIdx = i;
  renderEpList();
  const ep = episodeList()[i];
  master.pause();
  CAMS.forEach((cam, k) => {
    const v = vids[k];
    if (ep.videos[cam]) { v.src = fileURL(ep.videos[cam]); v.load(); }
    else { v.removeAttribute("src"); v.load(); }
  });
  buildChart(ep.milestones);
  renderVerdict(ep.milestones);
  redraw();
}

function targetJoint(ms) {
  const d = (ms.detail || []).find((x) => x.milestone === ms.target_milestone);
  return d ? d.joint : null;
}

function buildChart(ms) {
  chart = null;
  $("#legend").innerHTML = "";
  const note = $("#nochart");
  const show = (msg) => {
    note.textContent = msg;
    note.classList.remove("hidden");
    cv.classList.add("hidden");
  };
  if (!ms) return show("no articulation data for this episode");
  const ts = ms.timeseries;
  if (!ts || !ts.joints || !Object.keys(ts.joints).length) {
    return show("no timeseries in this run (recorded before timeline support)");
  }
  note.classList.add("hidden");
  cv.classList.remove("hidden");
  const fps = ts.video_fps || 30;
  const target = targetJoint(ms);
  const span = (v) => {
    let lo = 1, hi = 0;
    for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
    return hi - lo;
  };
  // Plot the target joint always, plus every joint that moved more than 1%
  // of its range. Cap at the 8 palette slots, biggest movers first.
  let names = Object.keys(ts.joints)
    .filter((n) => n === target || span(ts.joints[n]) > 0.01)
    .sort((a, b) => (a === target ? -1 : b === target ? 1 :
                     span(ts.joints[b]) - span(ts.joints[a])))
    .slice(0, SERIES.length);
  if (!names.length) return show("no joint movement recorded");
  const thresholds = {};
  for (const d of ms.detail || []) {
    if ("threshold_fraction" in d) thresholds[d.joint] = d.threshold_fraction;
  }
  const steps = Math.max(...names.map((n) => ts.joints[n].length));
  chart = { names, joints: ts.joints, thresholds, fps, duration: steps / fps };
  $("#legend").innerHTML = names.map((n, i) =>
    '<span><i style="background:' + SERIES[i] + '"></i>' + esc(n) + "</span>").join("");
  sizeCanvas();
  redraw();
}

function sizeCanvas() {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function plotRect() {
  const r = cv.getBoundingClientRect();
  return { x0: MARGIN.l, y0: MARGIN.t,
           x1: r.width - MARGIN.r, y1: r.height - MARGIN.b };
}

function redraw() {
  if (!chart || cv.classList.contains("hidden")) return;
  const p = plotRect();
  const r = cv.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  const X = (t) => p.x0 + (t / chart.duration) * (p.x1 - p.x0);
  const Y = (v) => p.y1 - v * (p.y1 - p.y0);
  ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
  // Grid and axes.
  ctx.strokeStyle = "#e1e0d9";
  ctx.fillStyle = "#898781";
  ctx.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    ctx.moveTo(p.x0, Y(v)); ctx.lineTo(p.x1, Y(v));
    ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(String(v), p.x0 - 6, Y(v));
  }
  const step = niceStep(chart.duration / 6);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 0; t <= chart.duration + 1e-9; t += step) {
    ctx.fillText(t.toFixed(step < 1 ? 1 : 0) + "s", X(t), p.y1 + 8);
  }
  // Threshold markers: dashed line at each plotted joint's milestone fraction.
  ctx.setLineDash([4, 3]);
  chart.names.forEach((n, i) => {
    const th = chart.thresholds[n];
    if (th === undefined) return;
    ctx.strokeStyle = SERIES[i];
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(p.x0, Y(th)); ctx.lineTo(p.x1, Y(th));
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
  ctx.setLineDash([]);
  // Series lines.
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  chart.names.forEach((n, i) => {
    const v = chart.joints[n];
    ctx.strokeStyle = SERIES[i];
    ctx.beginPath();
    for (let k = 0; k < v.length; k++) {
      const x = X(k / chart.fps), y = Y(v[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  // Direct end labels, spread vertically to avoid collisions.
  const labels = chart.names
    .map((n, i) => ({ n, i, y: Y(chart.joints[n][chart.joints[n].length - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < labels.length; k++) {
    labels[k].y = Math.max(labels[k].y, labels[k - 1].y + 14);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (const l of labels) {
    ctx.fillStyle = SERIES[l.i];
    ctx.fillRect(p.x1 + 6, l.y - 3.5, 7, 7);
    ctx.fillStyle = "#52514e";
    ctx.fillText(l.n, p.x1 + 17, l.y);
  }
  // Playhead follows the master video.
  const t = Math.min(master.currentTime || 0, chart.duration);
  ctx.strokeStyle = "#0b0b0b";
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(X(t), p.y0); ctx.lineTo(X(t), p.y1);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-6))));
  for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}

function chartTime(ev) {
  const p = plotRect();
  const x = ev.clientX - cv.getBoundingClientRect().left;
  const f = Math.min(Math.max((x - p.x0) / (p.x1 - p.x0), 0), 1);
  return f * chart.duration;
}

function seekAll(t) {
  for (const v of vids) {
    if (!isNaN(v.duration)) v.currentTime = Math.min(t, v.duration);
  }
  updateClock();
  redraw();
}

let dragging = false;
cv.addEventListener("pointerdown", (ev) => {
  if (!chart) return;
  dragging = true;
  cv.setPointerCapture(ev.pointerId);
  seekAll(chartTime(ev));
});
cv.addEventListener("pointermove", (ev) => {
  if (!chart) return;
  if (dragging) seekAll(chartTime(ev));
  showTooltip(ev);
});
cv.addEventListener("pointerup", () => { dragging = false; });
cv.addEventListener("pointerleave", () => $("#tooltip").classList.add("hidden"));

function showTooltip(ev) {
  const tip = $("#tooltip");
  const t = chartTime(ev);
  const rows = chart.names.map((n, i) => {
    const v = chart.joints[n];
    const k = Math.min(Math.round(t * chart.fps), v.length - 1);
    return '<i style="display:inline-block;width:8px;height:8px;border-radius:2px;' +
      "background:" + SERIES[i] + ';margin-right:5px"></i>' +
      esc(n) + " " + v[k].toFixed(3);
  });
  tip.innerHTML = "<b>" + fmtTime(t) + "</b><br>" + rows.join("<br>");
  tip.classList.remove("hidden");
  const wrap = $("#chartwrap").getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - wrap.left + 14, wrap.width - 190) + "px";
  tip.style.top = (ev.clientY - wrap.top + 14) + "px";
}

function renderVerdict(ms) {
  const el = $("#verdict");
  if (!ms) {
    el.innerHTML = '<div class="notice">no milestone JSON for this episode</div>';
    return;
  }
  const badge = ms.success
    ? '<span class="badge ok">PASS</span>' : '<span class="badge bad">FAIL</span>';
  const mark = (ok) => ok
    ? '<span class="mark ok">\\u2713</span>' : '<span class="mark bad">\\u2717</span>';
  const list = (obj) => "<ul>" + Object.entries(obj).map(([k, v]) =>
    "<li>" + mark(v) + "<b>" + esc(k) + "</b></li>").join("") + "</ul>";
  const progress = "<table class='kv'>" + Object.entries(ms.progress || {}).map(
    ([k, v]) => "<tr><td>" + esc(k) + "</td><td>" + v + "</td></tr>").join("") +
    "</table>";
  el.innerHTML =
    badge + '<span class="instr">"' + esc(ms.instruction) + '"</span>' +
    ' <span style="color:#898781">(target: ' + esc(ms.target_milestone || "any") +
    ", " + ms.steps + " steps)</span>" +
    '<div class="cols"><div><h3>Milestones</h3>' + list(ms.milestones || {}) +
    "</div>" +
    (Object.keys(ms.long_horizon || {}).length
      ? "<div><h3>Long horizon</h3>" + list(ms.long_horizon) + "</div>" : "") +
    "<div><h3>Progress</h3>" + progress + "</div></div>";
}

function updateClock() {
  $("#clock").textContent = fmtTime(master.currentTime || 0);
}

// The external video is the sync master: its native controls, plus the chart
// and the play/pause button, drive the other two streams.
function rafLoop() {
  updateClock();
  redraw();
  if (!master.paused && !master.ended) requestAnimationFrame(rafLoop);
}
master.addEventListener("play", () => {
  others.forEach((v) => v.src && v.play());
  $("#playpause").textContent = "Pause";
  requestAnimationFrame(rafLoop);
});
master.addEventListener("pause", () => {
  others.forEach((v) => v.pause());
  $("#playpause").textContent = "Play";
  updateClock();
  redraw();
});
master.addEventListener("seeked", () => {
  others.forEach((v) => {
    if (v.src && Math.abs(v.currentTime - master.currentTime) > 0.05) {
      v.currentTime = master.currentTime;
    }
  });
  updateClock();
  redraw();
});
$("#playpause").onclick = () => {
  if (master.paused) master.play(); else master.pause();
};
window.addEventListener("resize", () => { sizeCanvas(); redraw(); });
init();
</script>
</body>
</html>
"""
