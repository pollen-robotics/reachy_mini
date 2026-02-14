import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import URDFLoader from 'https://cdn.jsdelivr.net/npm/urdf-loader@0.12.3/+esm';

const HF_BASE = "https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz/resolve/main/reachy_mini_3d_web_viz/static/";
const URDF_URL = HF_BASE + "assets/reachy-mini.urdf";
const MESH_BASE_URL = HF_BASE + "assets/meshes_optimized/";

const HEAD_JOINT_NAMES = ['yaw_body', 'stewart_1', 'stewart_2', 'stewart_3', 'stewart_4', 'stewart_5', 'stewart_6'];
const PASSIVE_JOINT_NAMES = [];
for (let i = 1; i <= 7; i++) {
    PASSIVE_JOINT_NAMES.push(`passive_${i}_x`, `passive_${i}_y`, `passive_${i}_z`);
}

// ============= Stewart Platform Kinematics =============
// Ported from https://github.com/pollen-robotics/reachy-mini-desktop-app

const HEAD_Z_OFFSET = 0.177;
const MOTOR_ARM_LENGTH = 0.04;

const T_HEAD_XL_330 = [
    [0.4822, -0.7068, -0.5177, 0.0206],
    [0.1766, -0.5003, 0.8476, -0.0218],
    [-0.8581, -0.5001, -0.1164, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

const PASSIVE_ORIENTATION_OFFSET = [
    [-0.13754, -0.0882156, 2.10349],
    [-Math.PI, 5.37396e-16, -Math.PI],
    [0.373569, 0.0882156, -1.0381],
    [-0.0860846, 0.0882156, 1.0381],
    [0.123977, 0.0882156, -1.0381],
    [3.0613, 0.0882156, 1.0381],
    [Math.PI, 2.10388e-17, 4.15523e-17],
];

const STEWART_ROD_DIR_IN_PASSIVE_FRAME = [
    [1.0, 0.0, 0.0],
    [0.50606941, -0.85796418, -0.08826792],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
];

const MOTORS = [
    {
        branchPosition: [0.020648178337122566, 0.021763723638894568, 1.0345743467476964e-07],
        tWorldMotor: [
            [0.8660247915798899, 0.0000044901959360, -0.5000010603477224, 0.0269905781109381],
            [-0.5000010603626028, 0.0000031810770988, -0.8660247915770969, 0.0267489144601032],
            [-0.0000022980790772, 0.9999999999848599, 0.0000049999943606, 0.0766332540902687],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        branchPosition: [0.00852381571767217, 0.028763668526131346, 1.183437210727778e-07],
        tWorldMotor: [
            [-0.8660211183436273, -0.0000044902196459, -0.5000074225075980, 0.0096699703080478],
            [0.5000074225224782, -0.0000031810634097, -0.8660211183408341, 0.0367490037948058],
            [0.0000022980697230, -0.9999999999848597, 0.0000050000112432, 0.0766333000521544],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        branchPosition: [-0.029172011376922807, 0.0069999429399361995, 4.0290270064691214e-08],
        tWorldMotor: [
            [0.0000063267948970, -0.0000010196153098, 0.9999999999794665, -0.0366606982562266],
            [0.9999999999799865, 0.0000000000135060, -0.0000063267948965, 0.0100001160862987],
            [-0.0000000000070551, 0.9999999999994809, 0.0000010196153103, 0.0766334229944826],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        branchPosition: [-0.029172040355214434, -0.0069999960097160766, -3.1608172912367394e-08],
        tWorldMotor: [
            [-0.0000036732050704, 0.0000010196153103, 0.9999999999927344, -0.0366607717202358],
            [-0.9999999999932538, -0.0000000000036776, -0.0000036732050700, -0.0099998653384376],
            [-0.0000000000000677, -0.9999999999994809, 0.0000010196153103, 0.0766334229944823],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        branchPosition: [0.008523809101930114, -0.028763713010385224, -1.4344916837716326e-07],
        tWorldMotor: [
            [-0.8660284647694136, 0.0000044901728834, -0.4999946981608615, 0.0096697448698383],
            [-0.4999946981757425, -0.0000031811099295, 0.8660284647666202, -0.0367490491228644],
            [0.0000022980794298, 0.9999999999848597, 0.0000049999943840, 0.0766333000520353],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        branchPosition: [0.020648186722822436, -0.02176369606185343, -8.957920105689965e-08],
        tWorldMotor: [
            [0.8660247915798903, -0.0000044901962204, -0.5000010603477218, 0.0269903370664035],
            [0.5000010603626028, 0.0000031810964559, 0.8660247915770964, -0.0267491384573748],
            [-0.0000022980696448, -0.9999999999848597, 0.0000050000112666, 0.0766332540903862],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
];

function rotationFromEulerXYZ(x, y, z) {
    const cx = Math.cos(x), sx = Math.sin(x);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cz = Math.cos(z), sz = Math.sin(z);
    return [
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ];
}

function eulerFromRotationXYZ(r) {
    const sy = r[0][2];
    if (Math.abs(sy) < 0.99999) {
        return [Math.atan2(-r[1][2], r[2][2]), Math.asin(sy), Math.atan2(-r[0][1], r[0][0])];
    }
    return [Math.atan2(r[2][1], r[1][1]), sy > 0 ? Math.PI / 2 : -Math.PI / 2, 0];
}

function mat3Mul(a, b) {
    const r = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            for (let k = 0; k < 3; k++)
                r[i][j] += a[i][k] * b[k][j];
    return r;
}

function mat3T(m) {
    return [[m[0][0],m[1][0],m[2][0]],[m[0][1],m[1][1],m[2][1]],[m[0][2],m[1][2],m[2][2]]];
}

function mv3(m, v) {
    return [
        m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
        m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
        m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2],
    ];
}

function v3Add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function v3Sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function v3Len(v) { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
function v3Norm(v) { const l=v3Len(v); return l<1e-10?[0,0,0]:[v[0]/l,v[1]/l,v[2]/l]; }
function v3Dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function v3Cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }

function skew(v) { return [[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]; }
function mat3Add(a, b) { return a.map((r,i)=>r.map((v,j)=>v+b[i][j])); }
function mat3Scale(m, s) { return m.map(r=>r.map(v=>v*s)); }
function mat3Id() { return [[1,0,0],[0,1,0],[0,0,1]]; }

function alignVectors(from, to) {
    const f = v3Norm(from), t = v3Norm(to);
    const d = v3Dot(f, t);
    if (d > 0.99999) return mat3Id();
    if (d < -0.99999) {
        let p = v3Cross([1,0,0], f);
        if (v3Len(p) < 0.001) p = v3Cross([0,1,0], f);
        const ax = v3Norm(p), k = skew(ax), k2 = mat3Mul(k, k);
        return mat3Add(mat3Id(), mat3Scale(k2, 2));
    }
    const c = v3Cross(f, t), s = v3Len(c);
    const k = skew(c), k2 = mat3Mul(k, k);
    return mat3Add(mat3Add(mat3Id(), k), mat3Scale(k2, (1 - d) / (s * s)));
}

function mat4Rot(m) { return [m[0].slice(0,3), m[1].slice(0,3), m[2].slice(0,3)]; }
function mat4Trans(m) { return [m[0][3], m[1][3], m[2][3]]; }

function buildHeadPoseMatrix(hp) {
    const { x=0, y=0, z=0, roll=0, pitch=0, yaw=0 } = hp;
    const r = rotationFromEulerXYZ(roll, pitch, yaw);
    return [
        r[0][0], r[0][1], r[0][2], x,
        r[1][0], r[1][1], r[1][2], y,
        r[2][0], r[2][1], r[2][2], z,
        0, 0, 0, 1,
    ];
}

function calculatePassiveJoints(headJoints, headPose16) {
    if (!headJoints || headJoints.length < 7 || !headPose16 || headPose16.length < 16) {
        return new Array(21).fill(0);
    }

    const bodyYaw = headJoints[0];
    const pose = [
        [headPose16[0], headPose16[1], headPose16[2], headPose16[3]],
        [headPose16[4], headPose16[5], headPose16[6], headPose16[7]],
        [headPose16[8], headPose16[9], headPose16[10], headPose16[11]],
        [headPose16[12], headPose16[13], headPose16[14], headPose16[15]],
    ];
    pose[2][3] += HEAD_Z_OFFSET;

    const cosY = Math.cos(bodyYaw), sinY = Math.sin(bodyYaw);
    const rZInv = [[cosY,sinY,0],[-sinY,cosY,0],[0,0,1]];
    const pRot = mat3Mul(rZInv, mat4Rot(pose));
    const pTrans = mv3(rZInv, mat4Trans(pose));

    const passiveCorr = PASSIVE_ORIENTATION_OFFSET.map(o => rotationFromEulerXYZ(o[0], o[1], o[2]));
    const passiveJoints = new Array(21).fill(0);
    let lastRServoBranch = mat3Id(), lastRWorldServo = mat3Id();
    const tArm = [MOTOR_ARM_LENGTH, 0, 0];

    for (let i = 0; i < 6; i++) {
        const motor = MOTORS[i];
        const sj = headJoints[i + 1];
        const branchWorld = v3Add(mv3(pRot, motor.branchPosition), pTrans);

        const cs = Math.cos(sj), sn = Math.sin(sj);
        const rServo = [[cs,-sn,0],[sn,cs,0],[0,0,1]];
        const twmR = mat4Rot(motor.tWorldMotor);
        const twmT = mat4Trans(motor.tWorldMotor);

        const pServoArm = v3Add(mv3(twmR, mv3(rServo, tArm)), twmT);
        const rWorldServo = mat3Mul(mat3Mul(twmR, rServo), passiveCorr[i]);
        const vecInServo = mv3(mat3T(rWorldServo), v3Sub(branchWorld, pServoArm));
        const straight = v3Norm(vecInServo);
        const rServoBranch = alignVectors(STEWART_ROD_DIR_IN_PASSIVE_FRAME[i], straight);
        const euler = eulerFromRotationXYZ(rServoBranch);

        passiveJoints[i*3] = euler[0];
        passiveJoints[i*3+1] = euler[1];
        passiveJoints[i*3+2] = euler[2];

        if (i === 5) { lastRServoBranch = rServoBranch; lastRWorldServo = rWorldServo; }
    }

    const rHeadXl330 = mat3Mul(pRot, mat4Rot(T_HEAD_XL_330));
    const rRodCurrent = mat3Mul(mat3Mul(lastRWorldServo, lastRServoBranch), passiveCorr[6]);
    const rDof = mat3Mul(mat3T(rRodCurrent), rHeadXl330);
    const e7 = eulerFromRotationXYZ(rDof);
    passiveJoints[18] = e7[0]; passiveJoints[19] = e7[1]; passiveJoints[20] = e7[2];

    return passiveJoints;
}

function applyJoints(jointMap, headJoints, headPose) {
    for (let i = 0; i < 7; i++) {
        const joint = jointMap[HEAD_JOINT_NAMES[i]];
        if (joint) joint.setJointValue(headJoints[i]);
    }

    let headPoseMatrix = null;
    if (headPose) {
        if (Array.isArray(headPose) && headPose.length === 16) {
            headPoseMatrix = headPose;
        } else if (headPose.m) {
            headPoseMatrix = headPose.m;
        } else {
            headPoseMatrix = buildHeadPoseMatrix(headPose);
        }
    }

    if (headPoseMatrix) {
        const passive = calculatePassiveJoints(headJoints, headPoseMatrix);
        for (let i = 0; i < 21; i++) {
            const joint = jointMap[PASSIVE_JOINT_NAMES[i]];
            if (joint) joint.setJointValue(passive[i]);
        }
    }
}

// ============= UI / Scene =============

function updateStatus(message, state) {
    const dot = document.getElementById('viewer-3d-dot');
    const text = document.getElementById('viewer-3d-status-text');
    if (!dot || !text) return;
    dot.className = 'w-2 h-2 rounded-full';
    if (state === 'connected') dot.classList.add('bg-green-400');
    else if (state === 'disconnected') dot.classList.add('bg-red-400');
    else dot.classList.add('bg-yellow-400', 'animate-pulse');
    if (message) text.textContent = message;
}

function parseUrdfColors(urdfText) {
    const colors = {};
    const parser = new DOMParser();
    const doc = parser.parseFromString(urdfText, 'application/xml');
    const materialMap = {};

    doc.querySelectorAll('material').forEach(mat => {
        const name = mat.getAttribute('name');
        const colorEl = mat.querySelector('color');
        if (name && colorEl) {
            const rgba = colorEl.getAttribute('rgba')?.split(' ').map(Number) || [0.5, 0.5, 0.5, 1];
            materialMap[name] = {
                color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
                opacity: rgba[3],
                name: name
            };
        }
    });

    doc.querySelectorAll('visual').forEach(visual => {
        const meshEl = visual.querySelector('mesh');
        const matEl = visual.querySelector('material');
        if (meshEl && matEl) {
            let filename = meshEl.getAttribute('filename');
            if (filename) {
                filename = filename.split('/').pop();
                const matName = matEl.getAttribute('name');
                if (matName && materialMap[matName]) {
                    colors[filename] = materialMap[matName];
                }
            }
        }
    });
    return colors;
}

let viewer = null;

export async function init() {
    if (viewer) return viewer;

    const container = document.getElementById('viewer-3d-canvas');
    if (!container) return null;

    updateStatus('Loading robot...', 'loading');

    const scene = new THREE.Scene();
    const rect = container.getBoundingClientRect();
    const camera = new THREE.PerspectiveCamera(50, rect.width / rect.height, 0.01, 100);
    camera.position.set(0.4, 0.1, -0.4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(rect.width, rect.height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 2;
    controls.target.set(0, 0.15, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 0.2;
    controls.maxDistance = 2;
    controls.update();

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
    keyLight.position.set(2, 1, 2);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 10;
    keyLight.shadow.camera.left = -1;
    keyLight.shadow.camera.right = 1;
    keyLight.shadow.camera.top = 1;
    keyLight.shadow.camera.bottom = -1;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xFFB366, 0.6);
    fillLight.position.set(-2, 0.5, 1.5);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
    rimLight.position.set(0, 1.2, -2);
    scene.add(rimLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
    hemiLight.position.set(0, 1, 0);
    scene.add(hemiLight);

    // Ground
    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(2, 2),
        new THREE.ShadowMaterial({ opacity: 0.3 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    const gridHelper = new THREE.GridHelper(1, 20, 0xbbbbbb, 0xbbbbbb);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    // Resize handler
    const resizeObserver = new ResizeObserver(() => {
        const r = container.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        camera.aspect = r.width / r.height;
        camera.updateProjectionMatrix();
        renderer.setSize(r.width, r.height);
    });
    resizeObserver.observe(container);

    // Load URDF
    let robot = null;
    let jointMap = {};

    try {
        const response = await fetch(URDF_URL);
        const urdfText = await response.text();
        const meshColors = parseUrdfColors(urdfText);

        const blob = new Blob([urdfText], { type: 'application/xml' });
        const blobUrl = URL.createObjectURL(blob);

        const loader = new URDFLoader();
        loader.packages = { 'assets': 'assets/', 'reachy_mini_description': 'assets/' };
        loader.workingPath = 'assets/';

        const gltfLoader = new GLTFLoader();
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
        gltfLoader.setDRACOLoader(dracoLoader);

        loader.loadMeshCb = (path, manager, onComplete) => {
            const filename = path.split('/').pop();
            const matData = meshColors[filename] || {};
            const opacity = matData?.opacity ?? 1;
            const isTransparent = opacity < 0.4;

            const material = new THREE.MeshPhysicalMaterial({
                color: (filename?.includes('antenna_V2') || isTransparent || matData?.name === 'antenna_material') ? 0x202020 : matData?.color || 0x808080,
                metalness: 0.0,
                roughness: (filename?.includes('antenna_V2') || isTransparent || matData?.name === 'antenna_material') ? 0.05 : 0.7,
                transparent: isTransparent,
                opacity,
                side: isTransparent ? THREE.DoubleSide : THREE.FrontSide,
            });

            if (filename?.includes('link')) {
                material.color.setHex(0xffffff);
                material.metalness = 1.0;
                material.roughness = 0.3;
            }

            if (matData.name === 'antenna_material') {
                material.clearcoat = 1.0;
                material.clearcoatRoughness = 0.0;
                material.reflectivity = 1.0;
                material.envMapIntensity = 1.5;
            }

            const meshUrl = MESH_BASE_URL + filename.replace(/\.stl$/i, '.glb');
            gltfLoader.load(meshUrl, (gltf) => {
                let geometry = null;
                gltf.scene.traverse((child) => {
                    if (child.isMesh && !geometry) geometry = child.geometry;
                });
                if (geometry) {
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    onComplete(mesh);
                } else {
                    onComplete(gltf.scene);
                }
            }, undefined, (err) => {
                console.error('Mesh load error:', filename, err);
                onComplete(null, err);
            });
        };

        robot = await new Promise((resolve, reject) => {
            loader.load(blobUrl, (loadedRobot) => {
                URL.revokeObjectURL(blobUrl);
                resolve(loadedRobot);
            }, undefined, (err) => {
                URL.revokeObjectURL(blobUrl);
                reject(err);
            });
        });

        robot.rotation.x = -Math.PI / 2;
        robot.traverse((child) => {
            if (child.isURDFJoint) jointMap[child.name] = child;
        });
        scene.add(robot);

        // Apply default pose so the head looks correct before WebSocket data arrives
        const defaultHeadJoints = [
            -0.003067961575771161,   // yaw_body
             0.615126295942142,      // stewart_1
            -0.5997864880632857,     // stewart_2
             0.5737088146692297,     // stewart_3
            -0.6197282383057989,     // stewart_4
             0.5752427954571155,     // stewart_5
            -0.5829126993965437      // stewart_6
        ];
        const defaultHeadPose = {
            x: -0.0007244991153460317, y: 0.0020444415799089123, z: -0.0013986476141853443,
            roll: 0.03186123895204955, pitch: 0.006227992850683206, yaw: -0.02666119073242053
        };
        applyJoints(jointMap, defaultHeadJoints, defaultHeadPose);

        updateStatus('Connecting...', 'loading');

    } catch (err) {
        console.error('Failed to load robot:', err);
        updateStatus('Failed to load: ' + err.message, 'disconnected');
    }

    // WebSocket
    let ws = null;

    function connectWebSocket() {
        const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProto}//${location.host}/api/state/ws/full?with_head_joints=true`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => updateStatus('Connected', 'connected');
        ws.onclose = () => {
            updateStatus('Disconnected', 'disconnected');
            if (viewer?.active) setTimeout(connectWebSocket, 2000);
        };
        ws.onerror = () => updateStatus('Connection error', 'disconnected');

        ws.onmessage = (event) => {
            if (!robot) return;
            try {
                const data = JSON.parse(event.data);
                const headJoints = (data.head_joints?.length === 7)
                    ? data.head_joints
                    : [data.body_yaw || 0, 0, 0, 0, 0, 0, 0];

                applyJoints(jointMap, headJoints, data.head_pose || null);

                if (data.antennas_position?.length >= 2) {
                    jointMap['right_antenna']?.setJointValue(-data.antennas_position[0]);
                    jointMap['left_antenna']?.setJointValue(-data.antennas_position[1]);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket data:', e);
            }
        };
    }

    connectWebSocket();

    // Animation loop
    let animId = null;
    function animate() {
        if (!viewer?.active) return;
        animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    viewer = {
        active: true,
        animate,
        stop() {
            this.active = false;
            if (animId) cancelAnimationFrame(animId);
            if (ws) { ws.close(); ws = null; }
        },
        start() {
            this.active = true;
            animate();
            if (!ws || ws.readyState !== WebSocket.OPEN) connectWebSocket();
        }
    };

    viewer.animate();
    return viewer;
}

export function getViewer() {
    return viewer;
}
