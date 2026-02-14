import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import URDFLoader from 'https://cdn.jsdelivr.net/npm/urdf-loader@0.12.3/+esm';

const HF_BASE = "https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz/resolve/main/reachy_mini_3d_web_viz/static/";
const URDF_URL = HF_BASE + "assets/reachy-mini.urdf";
const MESH_BASE_URL = HF_BASE + "assets/meshes_optimized/";

const HEAD_JOINT_NAMES = ['yaw_body', 'stewart_1', 'stewart_2', 'stewart_3', 'stewart_4', 'stewart_5', 'stewart_6'];

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

                for (let i = 0; i < 7; i++) {
                    const joint = jointMap[HEAD_JOINT_NAMES[i]];
                    if (joint) joint.setJointValue(headJoints[i]);
                }

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
