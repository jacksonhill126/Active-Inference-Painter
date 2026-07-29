import * as THREE from "three";

const sceneEl = document.getElementById("scene");
const readout = document.getElementById("readout");
const specs = document.getElementById("specs");
const miniCanvas = document.getElementById("miniCanvas");
const miniCtx = miniCanvas.getContext("2d");
const robotModel = await fetch("/api/robot-model", { cache: "no-store" }).then((response) => {
  if (!response.ok) throw new Error(`robot model HTTP ${response.status}`);
  return response.json();
});
document.getElementById("modelVersion").textContent = robotModel.version;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
sceneEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0f16);
scene.fog = new THREE.Fog(0x0c0f16, 1.5, 3.8);

const camera = new THREE.PerspectiveCamera(42, sceneEl.clientWidth / sceneEl.clientHeight, 0.01, 8);
camera.up.set(0, 0, 1);
const camTarget = new THREE.Vector3(0.075, 0.24, 0.32);
let camR = 1.18;
let camTheta = -0.62;
let camPhi = 1.02;
let topView = false;
let pointerMode = null;
let lastPointerX = 0;
let lastPointerY = 0;
updateCamera();

scene.add(new THREE.HemisphereLight(0xf5f7ff, 0x10131d, 1.15));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(-0.55, -0.75, 1.35);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -1.0;
key.shadow.camera.right = 1.0;
key.shadow.camera.top = 1.0;
key.shadow.camera.bottom = -1.0;
scene.add(key);
const fill = new THREE.DirectionalLight(0xaec8ff, 0.75);
fill.position.set(0.75, 0.1, 0.8);
scene.add(fill);

const canvasTexture = new THREE.Texture();
canvasTexture.colorSpace = THREE.SRGBColorSpace;
canvasTexture.minFilter = THREE.LinearFilter;
canvasTexture.magFilter = THREE.LinearFilter;
const canvasMat = new THREE.MeshBasicMaterial({
  map: canvasTexture,
  side: THREE.DoubleSide,
  transparent: false,
  polygonOffset: true,
  polygonOffsetFactor: -2,
  polygonOffsetUnits: -2,
});
const canvasMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(robotModel.canvas.width, robotModel.canvas.height),
  canvasMat,
);
canvasMesh.rotation.x = Math.PI / 2;
canvasMesh.position.set(
  robotModel.canvas.center[0],
  robotModel.canvas.contactY - 0.00015,
  robotModel.canvas.center[2],
);
scene.add(canvasMesh);

const materialCache = new Map();
const jointNodes = new Map();
const geometryNodes = new Map();
const robotRoot = new THREE.Group();
robotRoot.name = robotModel.name;
scene.add(robotRoot);

function checkerTexture() {
  const bitmap = document.createElement("canvas");
  bitmap.width = 256;
  bitmap.height = 256;
  const context = bitmap.getContext("2d");
  const cells = 8;
  const cell = bitmap.width / cells;
  for (let y = 0; y < cells; y++) {
    for (let x = 0; x < cells; x++) {
      context.fillStyle = (x + y) % 2 ? "#30343a" : "#25292e";
      context.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  const texture = new THREE.CanvasTexture(bitmap);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function materialFor(geom) {
  const key = `${geom.material || "default"}:${(geom.rgba || []).join(",")}`;
  if (materialCache.has(key)) return materialCache.get(key);
  const definition = robotModel.materials[geom.material] || {};
  const rgba = geom.rgba || definition.rgba || [0.45, 0.48, 0.52, 1];
  const parameters = {
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    roughness: clamp(1 - 0.82 * (definition.shininess || 0.2), 0.16, 0.92),
    metalness: clamp(0.72 * (definition.specular || 0.15), 0, 0.72),
    transparent: rgba[3] < 0.999,
    opacity: rgba[3],
  };
  if (geom.material === "ground_mat") {
    parameters.map = checkerTexture();
    parameters.roughness = 0.82;
    parameters.metalness = 0.05;
  }
  const material = new THREE.MeshStandardMaterial(parameters);
  materialCache.set(key, material);
  return material;
}

function placeObjectBetween(object, startValues, endValues) {
  const start = v3(startValues);
  const end = v3(endValues);
  const direction = end.clone().sub(start);
  object.position.copy(start).add(end).multiplyScalar(0.5);
  object.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize(),
  );
}

function meshBetween(geom, capsule) {
  const [x1, y1, z1, x2, y2, z2] = geom.fromTo;
  const start = new THREE.Vector3(x1, y1, z1);
  const end = new THREE.Vector3(x2, y2, z2);
  const length = start.distanceTo(end);
  const radius = geom.size[0];
  const group = new THREE.Group();
  const material = materialFor(geom);
  const cylinder = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, length, 32),
    material,
  );
  cylinder.castShadow = true;
  cylinder.receiveShadow = true;
  group.add(cylinder);
  if (capsule) {
    const sphereGeometry = new THREE.SphereGeometry(radius, 24, 16);
    for (const sign of [-1, 1]) {
      const cap = new THREE.Mesh(sphereGeometry, material);
      cap.position.y = sign * length / 2;
      cap.castShadow = true;
      cap.receiveShadow = true;
      group.add(cap);
    }
  }
  placeObjectBetween(group, start.toArray(), end.toArray());
  return group;
}

function createGeom(geom) {
  let object;
  if ((geom.type === "cylinder" || geom.type === "capsule") && geom.fromTo) {
    object = meshBetween(geom, geom.type === "capsule");
  } else if (geom.type === "box") {
    object = new THREE.Mesh(
      new THREE.BoxGeometry(2 * geom.size[0], 2 * geom.size[1], 2 * geom.size[2]),
      materialFor(geom),
    );
    object.position.copy(v3(geom.position));
  } else if (geom.type === "plane") {
    object = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * geom.size[0], 2 * geom.size[1]),
      materialFor(geom),
    );
    object.position.copy(v3(geom.position));
  } else {
    object = new THREE.Mesh(
      new THREE.SphereGeometry(geom.size[0] || 0.01, 24, 16),
      materialFor(geom),
    );
    object.position.copy(v3(geom.position));
  }
  object.name = geom.name;
  object.castShadow = geom.type !== "plane";
  object.receiveShadow = true;
  geometryNodes.set(geom.name, object);
  return object;
}

function buildBody(definition, parent) {
  const body = new THREE.Group();
  body.name = definition.name;
  body.position.copy(v3(definition.position));
  if (definition.xyAxes) {
    const xAxis = v3(definition.xyAxes.slice(0, 3)).normalize();
    const yAxis = v3(definition.xyAxes.slice(3, 6));
    yAxis.addScaledVector(xAxis, -yAxis.dot(xAxis)).normalize();
    const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
    body.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis),
    );
  }
  body.userData.basePosition = body.position.clone();
  body.userData.baseQuaternion = body.quaternion.clone();
  parent.add(body);
  for (const joint of definition.joints) {
    jointNodes.set(joint.name, { node: body, definition: joint });
  }
  for (const geom of definition.geoms) body.add(createGeom(geom));
  for (const child of definition.bodies) buildBody(child, body);
}

for (const geom of robotModel.world.geoms) robotRoot.add(createGeom(geom));
for (const body of robotModel.world.bodies) buildBody(body, robotRoot);

const tipMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.0065, 24, 16),
  new THREE.MeshStandardMaterial({
    color: 0xf06b46,
    emissive: 0x4b1008,
    roughness: 0.28,
  }),
);
scene.add(tipMarker);
const contactHalo = new THREE.Mesh(
  new THREE.RingGeometry(0.009, 0.013, 40),
  new THREE.MeshBasicMaterial({
    color: 0x5ad1c4,
    transparent: true,
    opacity: 0.9,
    side: THREE.DoubleSide,
  }),
);
contactHalo.rotation.x = Math.PI / 2;
contactHalo.visible = false;
scene.add(contactHalo);

function applyJoint(name, value) {
  const entry = jointNodes.get(name);
  if (!entry) return;
  const { node, definition } = entry;
  const axis = v3(definition.axis).normalize();
  if (definition.type === "slide") {
    // Compound brush joints share a body. A slide axis therefore follows any
    // preceding local bend rotations already accumulated on that body.
    node.position.addScaledVector(axis.applyQuaternion(node.quaternion), value);
  } else {
    node.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(axis, value));
  }
}

function updateRobot(robotState, contact) {
  for (const node of new Set([...jointNodes.values()].map((entry) => entry.node))) {
    node.position.copy(node.userData.basePosition);
    node.quaternion.copy(node.userData.baseQuaternion);
  }
  const q = robotState.jointPositionDeg;
  for (const name of robotModel.jointOrder) {
    applyJoint(name, THREE.MathUtils.degToRad(q[name]));
  }
  const bend = robotState.brushBendRad || {};
  applyJoint("brush_bend_x", bend.x || 0);
  applyJoint("brush_bend_z", bend.z || 0);
  applyJoint("brush_compression", -Math.max(0, robotState.brushCompressionM || 0));
  tipMarker.position.copy(v3(robotState.tipM));
  tipMarker.material.color.setHex(contact.touching ? 0x5ad1c4 : 0xf06b46);
  contactHalo.visible = contact.touching;
  if (contact.touching) {
    contactHalo.position.copy(v3(robotState.mappedCartesianTargetM));
    contactHalo.position.y = robotModel.canvas.contactY - 0.00035;
  }
}

function v3(p) {
  return new THREE.Vector3(p[0], p[1], p[2]);
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
document.getElementById("btnPaint").onclick = () => command("toggle_brush_load");
document.getElementById("btnAgent").onclick = () => command("toggle_agent");
document.getElementById("btnHomeView").onclick = () => setHomeView();
document.getElementById("btnFaceCanvas").onclick = () => setFaceCanvasView();
document.getElementById("btnTopView").onclick = () => setTopView();
document.getElementById("btnBlack").onclick = () => command("tone", "black");
document.getElementById("btnWhite").onclick = () => command("tone", "white");

window.addEventListener("keydown", (event) => {
  if (event.key === "f") command("toggle_max_speed");
  if (event.key === " ") command("toggle_pause");
  if (event.key === "r") command("reset");
  if (event.key === "c") command("clear");
  if (event.key === "h") setHomeView();
  if (event.key === "v") setFaceCanvasView();
  if (event.key === "t") setTopView();
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
    topView = false;
    camTheta -= dx * 0.008;
    camPhi = clamp(camPhi - dy * 0.007, 0.18, Math.PI - 0.18);
  } else {
    panCamera(dx, dy);
  }
  updateCamera();
});

renderer.domElement.addEventListener("wheel", (event) => {
  event.preventDefault();
  camR = clamp(camR + event.deltaY * 0.0018, 0.34, 3.0);
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
  if (topView) {
    camera.up.set(0, 1, 0);
    camera.lookAt(camTarget);
  }
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
  topView = false;
  camTarget.set(0.075, 0.24, 0.32);
  camR = 1.18;
  camTheta = -0.62;
  camPhi = 1.02;
  updateCamera();
}

function setFaceCanvasView() {
  topView = false;
  camTarget.copy(v3(robotModel.canvas.center));
  camR = 0.86;
  camTheta = 0;
  camPhi = Math.PI / 2;
  updateCamera();
}

function setTopView() {
  topView = true;
  camTarget.set(robotModel.canvas.center[0], 0.27, 0.30);
  camR = 1.08;
  camTheta = 0;
  camPhi = 0.001;
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
  updateRobot(state.robot, state.contact);

  document.getElementById("btnMax").textContent = `Max speed: ${state.maxSpeed ? "on" : "off"}`;
  document.getElementById("btnMax").classList.toggle("active", state.maxSpeed);
  document.getElementById("btnPause").textContent = state.paused ? "Resume" : "Pause";
  document.getElementById("btnPaint").textContent = `Brush: ${state.brushLoaded ? "loaded" : "unloaded"}`;
  document.getElementById("btnAgent").textContent = `Agent: ${state.agentEnabled ? "on" : "off"}`;
  document.getElementById("btnAgent").classList.toggle("active", state.agentEnabled);
  const efe = state.agent?.efe || {};
  const vfe = state.agent?.vfe || {};
  const executionForecast = state.agent?.executionForecast || {};
  const motorPrimitive = state.agent?.executingMotorPrimitive || {};
  const belief = state.agent?.belief || {};
  const beliefMean = belief.mean || [];
  const beliefStd = belief.std || [];
  const spatialBelief = state.agent?.spatialBelief || {};
  const composition = state.agent?.composition || {};
  const hierarchy = composition.hierarchy || {};
  const canvasLatent = hierarchy.canvas || {};
  const relationalLatent = hierarchy.relational || {};
  const passageTrajectory = hierarchy.passageTrajectory || {};
  const passageKindUpdates = passageTrajectory.kindUpdateCounts || {};
  const topPassageTrajectory = composition.topPolicyPassageTrajectory || {};
  const passageEvaluation = composition.passageTrajectoryEvaluation || {};
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
      `${pct(p.posterior)} / ${policyKind(p)} / ${p.rolloutMode || "dense_grid"} ${p.rolloutGridSize || "-"} / ${p.hierarchyTransitionMode || "unavailable"} ${p.passageTrajectorySteps || 0} steps / G ${num(p.total)} / C_T ${num(p.terminalCoverageMean)}`
    )
  );

  readout.innerHTML = [
    `physical tip m <b>${state.robot.tipM.map((x) => x.toFixed(3)).join(", ")}</b>`,
    `coverage mean <b>${state.canvas.coverage.toFixed(4)}</b> / pressure summary <b>${state.contact.pressure.toFixed(3)}</b>`,
    `agent <b>${state.agentEnabled ? agentPhaseLabel(state.agent) : "scripted fallback"}</b> / robot view <b>${robotModeLabel(state.robot.mode)}</b> / sim <b>${state.simTime.toFixed(1)}s</b>`,
    `VFE F <b>${num(vfe.total)}</b> = complexity <b>${num(vfe.complexity)}</b> + negative log likelihood <b>${num(vfe.negative_log_likelihood)}</b>`,
    `EFE G <b>${num(efe.total)}</b> = terminal risk <b>${num(efe.terminal_risk)}</b> + ambiguity <b>${num(efe.ambiguity)}</b> + transition risk <b>${num(efe.transition_risk)}</b> + transition ambiguity <b>${num(efe.transition_ambiguity)}</b> + canvas latent risk <b>${num(efe.canvas_transition_risk)}</b> + relational risk <b>${num(efe.relational_transition_risk)}</b> + motor risk <b>${num(efe.motor_risk)}</b> + motor ambiguity <b>${num(efe.motor_ambiguity)}</b>`,
  ].join("<br>");

  specs.innerHTML = [
    row("Driver", state.agentEnabled ? "active inference" : "scripted IK"),
    row("Realized plant", state.plantBackend || "native"),
    row("Counterfactual plant", state.counterfactualPlantBackend || "native"),
    row("Robot geometry", robotModel.version),
    row("Robot state adapter", robotModeLabel(state.robot.mode)),
    row("Robot target error", `${(1000 * state.robot.alignmentErrorM).toFixed(3)} mm`),
    row("Target roll preserved", state.robot.rollPreserved ? "yes" : "no"),
    row("Code version", state.codeVersion ? `v${state.codeVersion}` : "-"),
    row("State representation", state.agent?.stateRepresentation || "-"),
    row("Material pyramid", pyramidText),
    row("Spatial transition mode", state.agent?.spatialTransitionMode || "-"),
    row("Transition model", state.agent?.transitionModel || "-"),
    row("VFE units", vfe.units || "-"),
    row("VFE approximation", vfe.approximation || "-"),
    row("Agent phase", agentPhaseLabel(state.agent)),
    row("Current planner time", `${num(state.agent?.currentPlanningSeconds)} s`),
    row("Last planner time", `${num(state.agent?.lastPlanningSeconds)} s`),
    row("Plan base EFE", `${num(planningProfile.baseEFESeconds)} s`),
    row(
      "Plan motor forecast",
      `${num(planningProfile.motorForecastSeconds)} s / ${planningProfile.motorForecastCount ?? 0}`
    ),
    row(
      "Motor forecast batches",
      `${planningProfile.motorForecastBatchCount ?? 0} batches / ${planningProfile.motorForecastWorkers ?? 0} workers`
    ),
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
    row("Canvas latent", `${canvasLatent.dimensions ?? "-"} dims / ${canvasLatent.updateCount ?? 0} updates`),
    row("Canvas posterior std", num(canvasLatent.posteriorStdMean)),
    row("Relational latent", `${relationalLatent.dimensions ?? "-"} dims / ${relationalLatent.updateCount ?? 0} updates`),
    row("Relational posterior std", num(relationalLatent.posteriorStdMean)),
    row("Relational observation", `${hierarchy.relationalObservationDimensions ?? "-"} dims / ${hierarchy.markSlots ?? "-"} slots`),
    row("Passage transition replay", String(composition.passageReplaySize ?? 0)),
    row("Hierarchy transition loss", num(composition.lastTransitionTrainingLoss)),
    row("Passage-step replay", String(composition.passageStepReplaySize ?? 0)),
    row("Passage trajectory updates", String(passageTrajectory.transitionUpdates ?? 0)),
    row(
      "Passage kind support",
      `band ${passageKindUpdates.band ?? 0} / chain ${passageKindUpdates.chain ?? 0} / polyline ${passageKindUpdates.polyline ?? 0}`
    ),
    row("Passage trajectory loss", num(composition.lastPassageTrajectoryLoss)),
    row("Passage one-step canvas KL", num(passageEvaluation.canvasKLNatsPerDim)),
    row("Passage one-step relation KL", num(passageEvaluation.relationalKLNatsPerDim)),
    row("Passage conditioning gain", `${num(passageEvaluation.conditioningGainNatsPerDim)} nats/dim`),
    row(
      "Predicted coarse trajectory",
      topPassageTrajectory.stepCount
        ? `${topPassageTrajectory.stepCount} steps / ${(topPassageTrajectory.coarseMaterialFieldShape || []).join("x")}`
        : "-"
    ),
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
    row("Canvas transition risk", num(efe.canvas_transition_risk)),
    row("Relational transition risk", num(efe.relational_transition_risk)),
    row("Passage canvas trajectory risk", num(efe.passage_canvas_trajectory_risk)),
    row("Passage relational trajectory risk", num(efe.passage_relational_trajectory_risk)),
    row("Passage likelihood observations", num(efe.passage_trajectory_observation_count)),
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
    row("Hierarchy rollout", `${efe.hierarchy_transition_mode || "unavailable"} / ${efe.passage_trajectory_steps ?? 0} steps`),
    row("q(coverage) mean / std", `${num(beliefMean[0])} / ${num(beliefStd[0])}`),
    row("q(mean thickness) mean / std", `${num(beliefMean[1])} / ${num(beliefStd[1])}`),
    ...policyRows,
    row("Coverage observation", state.canvas.coverage.toFixed(4)),
    row("Contact pressure summary", state.contact.pressure.toFixed(3)),
    row("Contact status", state.contact.touching ? "touching" : "clear"),
    row("Force", `${state.contact.force.toFixed(2)} N`),
    row("Brush width", `${state.contact.brushWidthPx.toFixed(2)} px`),
    row("Tone", state.brushTone),
    row("Controller yaw / pitch", `${state.pose.yaw.toFixed(1)} / ${state.pose.pitch.toFixed(1)}`),
    row("Controller roll / elbow", `${state.pose.roll.toFixed(1)} / ${state.pose.elbow.toFixed(1)}`),
    row(
      "Visual yaw / pitch",
      `${state.robot.jointPositionDeg.yaw.toFixed(1)} / ${state.robot.jointPositionDeg.pitch.toFixed(1)}`
    ),
    row(
      "Visual roll / elbow",
      `${state.robot.jointPositionDeg.roll.toFixed(1)} / ${state.robot.jointPositionDeg.elbow.toFixed(1)}`
    ),
    row("Physical canvas plane", `${robotModel.canvas.contactY.toFixed(4)} m`),
    row("Logical canvas plane", `${state.canvas.distance.toFixed(1)} in`),
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

function robotModeLabel(mode) {
  const labels = {
    legacy_canvas_cartesian_retarget: "legacy Cartesian → MJCF",
    mujoco_direct: "MuJoCo direct",
  };
  return labels[mode] || mode || "unknown";
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
