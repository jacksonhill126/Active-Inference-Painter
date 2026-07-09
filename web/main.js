import * as THREE from "three";

const sceneEl = document.getElementById("scene");
const readout = document.getElementById("readout");
const specs = document.getElementById("specs");
const miniCanvas = document.getElementById("miniCanvas");
const miniCtx = miniCanvas.getContext("2d");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
sceneEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0f16);

const camera = new THREE.PerspectiveCamera(42, sceneEl.clientWidth / sceneEl.clientHeight, 0.1, 120);
camera.up.set(0, 0, 1);
const camTarget = new THREE.Vector3(0, 8.5, 0.8);
let camR = 39;
let camTheta = -0.56;
let camPhi = 1.08;
let pointerMode = null;
let lastPointerX = 0;
let lastPointerY = 0;
updateCamera();

scene.add(new THREE.HemisphereLight(0xf5f7ff, 0x10131d, 1.4));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(-12, -18, 22);
scene.add(key);

const armMat = new THREE.MeshStandardMaterial({ color: 0x9fb0d0, metalness: 0.45, roughness: 0.36 });
const jointMat = new THREE.MeshStandardMaterial({ color: 0x5ad1c4, metalness: 0.25, roughness: 0.35, emissive: 0x103c38 });
const tipMat = new THREE.MeshStandardMaterial({ color: 0xf0734e, roughness: 0.35, emissive: 0x331004 });

const joints = [0, 1, 2].map((_, i) => {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(i === 2 ? 0.45 : 0.65, 32, 16), i === 2 ? tipMat : jointMat);
  scene.add(mesh);
  return mesh;
});

const links = [0, 1].map(() => {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 1, 32), armMat);
  scene.add(mesh);
  return mesh;
});

let canvasTexture = new THREE.Texture();
canvasTexture.colorSpace = THREE.SRGBColorSpace;
const canvasMat = new THREE.MeshBasicMaterial({ map: canvasTexture, side: THREE.DoubleSide, transparent: false });
const canvasMesh = new THREE.Mesh(new THREE.PlaneGeometry(20, 20, 1, 1), canvasMat);
canvasMesh.rotation.x = Math.PI / 2;
canvasMesh.position.y = 17;
scene.add(canvasMesh);

const frame = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(20, 20)),
  new THREE.LineBasicMaterial({ color: 0x20242e })
);
frame.rotation.x = Math.PI / 2;
frame.position.y = 16.99;
scene.add(frame);

function v3(p) {
  return new THREE.Vector3(p[0], p[1], p[2]);
}

function placeLink(mesh, a, b) {
  const start = v3(a);
  const end = v3(b);
  const mid = start.clone().add(end).multiplyScalar(0.5);
  const dir = end.clone().sub(start);
  const len = dir.length();
  mesh.position.copy(mid);
  mesh.scale.set(1, len, 1);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

async function command(type, value = undefined) {
  await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value === undefined ? { type } : { type, value }),
  });
}

document.getElementById("btnMax").onclick = () => command("toggle_max_speed");
document.getElementById("btnPause").onclick = () => command("toggle_pause");
document.getElementById("btnReset").onclick = () => command("reset");
document.getElementById("btnClear").onclick = () => command("clear");
document.getElementById("btnPaint").onclick = () => command("toggle_paint");
document.getElementById("btnAgent").onclick = () => command("toggle_agent");
document.getElementById("btnHomeView").onclick = () => setHomeView();
document.getElementById("btnFaceCanvas").onclick = () => setFaceCanvasView();
document.getElementById("btnBlack").onclick = () => command("tone", "black");
document.getElementById("btnWhite").onclick = () => command("tone", "white");

window.addEventListener("keydown", (event) => {
  if (event.key === "f") command("toggle_max_speed");
  if (event.key === " ") command("toggle_pause");
  if (event.key === "r") command("reset");
  if (event.key === "c") command("clear");
  if (event.key === "h") setHomeView();
  if (event.key === "v") setFaceCanvasView();
});

renderer.domElement.addEventListener("contextmenu", (event) => event.preventDefault());
renderer.domElement.addEventListener("pointerdown", (event) => {
  renderer.domElement.setPointerCapture(event.pointerId);
  pointerMode = event.button === 2 || event.button === 1 || event.shiftKey ? "pan" : "orbit";
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;
});

renderer.domElement.addEventListener("pointerup", (event) => {
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
  pointerMode = null;
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (!pointerMode) return;
  const dx = event.clientX - lastPointerX;
  const dy = event.clientY - lastPointerY;
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;
  if (pointerMode === "orbit") {
    camTheta -= dx * 0.008;
    camPhi = clamp(camPhi - dy * 0.007, 0.18, Math.PI - 0.18);
  } else {
    panCamera(dx, dy);
  }
  updateCamera();
});

renderer.domElement.addEventListener("wheel", (event) => {
  event.preventDefault();
  camR = clamp(camR + event.deltaY * 0.045, 14, 82);
  updateCamera();
}, { passive: false });

function updateCamera() {
  const sinPhi = Math.sin(camPhi);
  camera.position.set(
    camTarget.x + camR * sinPhi * Math.sin(camTheta),
    camTarget.y - camR * sinPhi * Math.cos(camTheta),
    camTarget.z + camR * Math.cos(camPhi),
  );
  camera.up.set(0, 0, 1);
  camera.lookAt(camTarget);
}

function panCamera(dx, dy) {
  const forward = camTarget.clone().sub(camera.position).normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 0, 1)).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();
  const scale = camR * 0.0016;
  camTarget.addScaledVector(right, -dx * scale);
  camTarget.addScaledVector(up, dy * scale);
}

function setHomeView() {
  camTarget.set(0, 8.5, 0.8);
  camR = 39;
  camTheta = -0.56;
  camPhi = 1.08;
  updateCamera();
}

function setFaceCanvasView() {
  camTarget.set(0, canvasMesh.position.y, 0);
  camR = 34;
  camTheta = 0;
  camPhi = Math.PI / 2;
  updateCamera();
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

async function updateCanvasTexture() {
  const img = new Image();
  img.decoding = "async";
  img.src = `/api/canvas.png?t=${performance.now()}`;
  await img.decode();
  canvasTexture.image = img;
  canvasTexture.needsUpdate = true;
  miniCtx.imageSmoothingEnabled = true;
  miniCtx.drawImage(img, 0, 0, miniCanvas.width, miniCanvas.height);
}

let lastCanvasUpdate = 0;
async function pollState() {
  const state = await fetch("/api/state", { cache: "no-store" }).then((r) => r.json());
  if (state.codeVersion) {
    const versionText = `v${state.codeVersion}`;
    document.getElementById("codeVersion").textContent = versionText;
    document.title = `Active-Inference Arm Painter ${versionText}`;
  }
  const points = state.renderPoints || state.points;
  for (let i = 0; i < joints.length; i++) joints[i].position.copy(v3(points[i]));
  placeLink(links[0], points[0], points[1]);
  placeLink(links[1], points[1], points[2]);

  canvasMesh.position.y = state.canvas.distance;
  frame.position.y = state.canvas.distance - 0.01;

  document.getElementById("btnMax").textContent = `Max speed: ${state.maxSpeed ? "on" : "off"}`;
  document.getElementById("btnMax").classList.toggle("active", state.maxSpeed);
  document.getElementById("btnPause").textContent = state.paused ? "Resume" : "Pause";
  document.getElementById("btnPaint").textContent = `Paint: ${state.paintEnabled ? "on" : "off"}`;
  document.getElementById("btnAgent").textContent = `Agent: ${state.agentEnabled ? "on" : "off"}`;
  document.getElementById("btnAgent").classList.toggle("active", state.agentEnabled);
  const efe = state.agent?.efe || {};
  const executionForecast = state.agent?.executionForecast || {};
  const motorPrimitive = state.agent?.executingMotorPrimitive || {};
  const belief = state.agent?.belief || {};
  const beliefMean = belief.mean || [];
  const beliefStd = belief.std || [];
  const spatialBelief = state.agent?.spatialBelief || {};
  const materialPyramid = spatialBelief.materialPyramid || [];
  const pyramidText = materialPyramid.length
    ? materialPyramid.map((level) => `${level.name}:${level.gridSize}`).join(" -> ")
    : "-";
  const telemetryLog = state.telemetryLog || {};
  const planningProfile = state.agent?.planningProfile || {};
  const topPolicies = state.agent?.topPolicies || [];
  const policyRows = topPolicies.slice(0, 4).map((p, i) =>
    row(
      `q(policy) #${i + 1}`,
      `${pct(p.posterior)} / ${policyKind(p)} / ${p.rolloutMode || "dense_grid"} ${p.rolloutGridSize || "-"} / G ${num(p.total)} / C_T ${num(p.terminalCoverageMean)}`
    )
  );

  readout.innerHTML = [
    `tip <b>${state.tip.map((x) => x.toFixed(2)).join(", ")}</b>`,
    `coverage mean <b>${state.canvas.coverage.toFixed(4)}</b> / pressure summary <b>${state.contact.pressure.toFixed(3)}</b>`,
    `agent <b>${state.agentEnabled ? agentPhaseLabel(state.agent) : "scripted fallback"}</b> / sim <b>${state.simTime.toFixed(1)}s</b>`,
    `EFE G <b>${num(efe.total)}</b> = terminal risk <b>${num(efe.terminal_risk)}</b> + ambiguity <b>${num(efe.ambiguity)}</b> + transition risk <b>${num(efe.transition_risk)}</b> + transition ambiguity <b>${num(efe.transition_ambiguity)}</b> + motor risk <b>${num(efe.motor_risk)}</b> + motor ambiguity <b>${num(efe.motor_ambiguity)}</b>`,
  ].join("<br>");

  specs.innerHTML = [
    row("Driver", state.agentEnabled ? "active inference" : "scripted IK"),
    row("Code version", state.codeVersion ? `v${state.codeVersion}` : "-"),
    row("State representation", state.agent?.stateRepresentation || "-"),
    row("Material pyramid", pyramidText),
    row("Spatial transition mode", state.agent?.spatialTransitionMode || "-"),
    row("Transition model", state.agent?.transitionModel || "-"),
    row("Agent phase", agentPhaseLabel(state.agent)),
    row("Current planner time", `${num(state.agent?.currentPlanningSeconds)} s`),
    row("Last planner time", `${num(state.agent?.lastPlanningSeconds)} s`),
    row("Plan base EFE", `${num(planningProfile.baseEFESeconds)} s`),
    row("Plan motor forecast", `${num(planningProfile.motorForecastSeconds)} s / ${planningProfile.motorForecastCount ?? 0}`),
    row("Plan motor rescore", `${num(planningProfile.motorEFERescoreSeconds)} s`),
    row("Plan composition", `${num(planningProfile.compositionDiagnosticSeconds)} s`),
    row("Plan trailing train", `${num(planningProfile.trailingTrainingSeconds)} s`),
    row("Plan policies", String(planningProfile.policyCount ?? 0)),
    row(
      "Planner status",
      state.agent?.planning ? `running ${num(state.agent?.currentPlanningSeconds)} s` : (state.agent?.plannerError || "idle")
    ),
    row("Checkpoint", checkpointLabel(state.agent?.checkpoint)),
    row("Checkpoint save", state.agent?.checkpoint?.lastSaved || "-"),
    row("Checkpoint issue", state.agent?.checkpoint?.lastError || "-"),
    row("Telemetry samples", `${telemetryLog.sampleCount ?? 0} / ${telemetryLog.maxSamples ?? "-"}`),
    row("Telemetry window", `${num(telemetryLog.windowSeconds)} s`),
    row("Telemetry rate", `${num(telemetryLog.estimatedSampleHz)} Hz`),
    row("Telemetry retention", telemetryLog.retentionPolicy || "-"),
    row("Telemetry CSV", `<a href="${telemetryLog.csvEndpoint || "/api/telemetry.csv"}">download</a>`),
    row("Paintings completed", String(state.paintingCount ?? 0)),
    row("Last saved canvas", state.lastSavedCanvas || "-"),
    row("Strokes", String(state.agent?.strokeCount ?? 0)),
    row("Minimum stop coverage", pct(state.agent?.minimumStopCoverage)),
    row("Last stop blocked", state.agent?.lastStopBlocked ? "yes" : "no"),
    row("Motor feasibility rejects", String(state.agent?.motorRejections ?? 0)),
    row("Motor primitive candidates", String(state.agent?.motorPrimitiveCandidateCount ?? 0)),
    row("Motor posterior mass", pct(state.agent?.motorPrimitivePosteriorMass)),
    row("Executing motor primitive", motorPrimitive.kind || executionForecast.motor_primitive_kind || "-"),
    row("Exec uncertainty", num(executionForecast.execution_uncertainty)),
    row("Exec overshoot", num(executionForecast.overshoot)),
    row("Exec contact loss", pct(executionForecast.contact_loss_probability)),
    row("Exec pressure mean", num(executionForecast.pressure_mean)),
    row("Joint current rms", num(executionForecast.joint_current_rms)),
    row("Joint torque rms", num(executionForecast.joint_torque_rms)),
    row("Joint path deg", num(executionForecast.joint_path_length_deg)),
    row("Top q(policy)", pct(state.agent?.posterior)),
    row("Policy precision", num(state.agent?.policyPrecision)),
    row("Policy posterior entropy", num(state.agent?.posteriorEntropy)),
    row("Passage candidates", String(state.agent?.passageCandidateCount ?? 0)),
    row("Passage posterior mass", pct(state.agent?.passagePosteriorMass)),
    row("Passage-plan candidates", String(state.agent?.passagePlanCandidateCount ?? 0)),
    row("Passage-plan posterior mass", pct(state.agent?.passagePlanPosteriorMass)),
    row("Planning scope", state.agent?.planningScope || "-"),
    row("Hold scope", state.agent?.holdScope || "-"),
    row(
      "Active passage",
      state.agent?.activePassage
        ? `${state.agent.activePassage.kind} ${state.agent.activePassageCompletedStrokes}/${state.agent.activePassageTotalStrokes}`
        : state.agent?.activePassagePlan
          ? `${state.agent.activePassagePlan.kind} ${state.agent.activePassageCompletedStrokes}/${state.agent.activePassageTotalStrokes}`
          : "-"
    ),
    row("Queued passage marks", String(state.agent?.passageQueueLength ?? 0)),
    row("EFE total", num(efe.total)),
    row("Terminal risk", num(efe.terminal_risk)),
    row("Terminal entropy", num(efe.terminal_entropy)),
    row("Ambiguity", num(efe.ambiguity)),
    row("Transition risk", num(efe.transition_risk)),
    row("Transition ambiguity", num(efe.transition_ambiguity)),
    row("Motor risk", num(efe.motor_risk)),
    row("Motor ambiguity", num(efe.motor_ambiguity)),
    row("Motor EFE approx", efe.motor_efe_approximation || "-"),
    row("Epistemic value", num(efe.epistemic_value)),
    row("Pragmatic value", num(efe.pragmatic_value)),
    row("Rollout mode", efe.rollout_mode || "-"),
    row("Rollout grid", String(efe.rollout_grid_size ?? "-")),
    row("Active patch area", pct(efe.active_patch_area_fraction)),
    row("Local transition steps", String(efe.local_transition_steps ?? 0)),
    row("Sequential patch steps", String(efe.sequential_patch_steps ?? 0)),
    row("Identity approx", efe.identity_transition_approximation || "-"),
    row("q(coverage) mean / std", `${num(beliefMean[0])} / ${num(beliefStd[0])}`),
    row("q(mean thickness) mean / std", `${num(beliefMean[1])} / ${num(beliefStd[1])}`),
    ...policyRows,
    row("Coverage observation", state.canvas.coverage.toFixed(4)),
    row("Contact pressure summary", state.contact.pressure.toFixed(3)),
    row("Contact status", state.contact.touching ? "touching" : "clear"),
    row("Force", `${state.contact.force.toFixed(2)} N`),
    row("Brush width", `${state.contact.brushWidthPx.toFixed(2)} px`),
    row("Tone", state.brushTone),
    row("Yaw / Pitch", `${state.pose.yaw.toFixed(1)} / ${state.pose.pitch.toFixed(1)}`),
    row("Roll / Elbow", `${state.pose.roll.toFixed(1)} / ${state.pose.elbow.toFixed(1)}`),
    row("Canvas distance", `${state.canvas.distance.toFixed(1)} in`),
  ].join("");

  const now = performance.now();
  if (now - lastCanvasUpdate > 120) {
    lastCanvasUpdate = now;
    updateCanvasTexture().catch(console.error);
  }
}

function policyKind(policy) {
  const motor = policy.motorPrimitive ? ` / motor ${policy.motorPrimitive.kind}` : "";
  if (policy.passagePlan) return `passage-plan ${policy.passagePlan.kind}${motor}`;
  if (policy.passage) return `passage ${policy.passage.kind}${motor}`;
  return `mark${motor}`;
}

function row(label, value) {
  return `<div class="row"><span>${label}</span><b>${value}</b></div>`;
}

function num(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function pct(value) {
  return Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—";
}

function checkpointLabel(checkpoint) {
  if (!checkpoint || !checkpoint.path) return "disabled";
  const loaded = checkpoint.loaded ? "loaded" : "cold";
  return `${checkpoint.status || "unknown"} (${loaded}, every ${checkpoint.saveEveryTransitions ?? 1})`;
}

function agentPhaseLabel(agent) {
  if (!agent) return "unknown";
  if (agent.planning) return "planning/training";
  const labels = {
    global_planning: "global planning",
    local_passage_hold: "local passage hold",
    return_center: "returning center",
    approach: "approach",
    press: "press",
    paint: "paint",
    lift: "lift",
    stop: "stop",
  };
  return labels[agent.phase] || agent.phase || "unknown";
}

function resize() {
  renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
  camera.aspect = sceneEl.clientWidth / sceneEl.clientHeight;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", resize);

async function stateLoop() {
  try {
    await pollState();
  } catch (err) {
    readout.textContent = `connection error: ${err}`;
  } finally {
    setTimeout(stateLoop, 33);
  }
}

function render() {
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

stateLoop();
render();
