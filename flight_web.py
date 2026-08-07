import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from io import StringIO

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化工具")

# 标准列顺序
STANDARD_COLS = ["timestamp","latitude","longitude","altitude","heading","pitch","roll",
                 "ve","vn","vu","ae","an","au","vheading","vpitch","vroll"]

# ===================== 标准航空坐标转换 =====================
def ll2local_enu(lat0, lon0, lat, lon, alt):
    R = 6371000
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    east = R * dlon * np.cos(lat0_rad)
    north = R * dlat
    up = alt
    return east, north, up

# ===================== CSV解析 =====================
def parse_csv_data(csv_string):
    lines = csv_string.strip().splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    fixed_csv = '\n'.join(cleaned_lines)

    df = pd.read_csv(StringIO(fixed_csv))
    df.columns = df.columns.str.strip()

    # 错位修复：列数多1时删首列并重命名
    if len(df.columns) == len(STANDARD_COLS) + 1:
        df = df.iloc[:, 1:].reset_index(drop=True)
        df.columns = STANDARD_COLS[:len(df.columns)]
    
    required_cols = ["latitude", "longitude", "altitude", "heading", "pitch", "roll"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必填列：{missing_cols}，当前列名：{list(df.columns)}")
    
    optional_cols = {
        "timestamp": 0, "ve": 0.0, "vn": 0.0, "vu": 0.0,
        "vheading": 0.0, "vpitch": 0.0, "vroll": 0.0
    }
    for col, default in optional_cols.items():
        if col not in df.columns:
            df[col] = default
    
    all_cols = required_cols + list(optional_cols.keys())
    for col in all_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(optional_cols.get(col, 0))

    # 衍生量计算
    df["vg"] = np.sqrt(df["ve"]**2 + df["vn"]**2)
    df["gamma_deg"] = np.degrees(np.arctan2(df["vu"], np.sqrt(df["ve"]**2 + df["vn"]**2)))
    
    # 飞行状态判定（修复：匹配姿态倾角）
    df["flight_status"] = 0
    turn_mask = (abs(df["roll"]) >= 2.0) | (abs(df["vheading"]) >= 1.0) | (abs(df["vroll"]) >= 1.0)
    df.loc[turn_mask, "flight_status"] = 3
    climb_mask = (~turn_mask) & (df["vu"] >= 0.2)
    descend_mask = (~turn_mask) & (df["vu"] <= -0.2)
    df.loc[climb_mask, "flight_status"] = 1
    df.loc[descend_mask, "flight_status"] = 2

    if df["timestamp"].iloc[0] == 0 or abs(df["timestamp"].iloc[0]) < 1e9:
        df["timestamp"] = np.arange(len(df)) * 1000

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("有效数据行为0")
    return df

# ===================== 侧边栏数据输入 =====================
with st.sidebar:
    st.header("数据输入")
    input_mode = st.radio("选择数据输入方式", ["粘贴CSV文本", "上传CSV文件"])
    csv_text = st.text_area("粘贴CSV数据", height=280)
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    load_btn = st.button("加载数据", use_container_width=True)

frames_data = []
if load_btn:
    try:
        if input_mode == "粘贴CSV文本":
            if not csv_text.strip():
                st.error("请粘贴有效的CSV数据")
                st.stop()
            raw_text = csv_text
        else:
            if uploaded_file is None:
                st.error("请先选择要上传的CSV文件")
                st.stop()
            raw_text = uploaded_file.getvalue().decode('utf-8-sig', errors='ignore')
        
        df = parse_csv_data(raw_text)
        
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        for _, row in df.iterrows():
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            frames_data.append({
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "ve": float(row["ve"]),
                "vn": float(row["vn"]),
                "vu": float(row["vu"]),
                "vg": float(row["vg"]),
                "gamma": float(row["gamma_deg"]),
                "vheading": float(row["vheading"]),
                "vpitch": float(row["vpitch"]),
                "vroll": float(row["vroll"]),
                "status": int(row["flight_status"])
            })
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
        st.caption(f"首帧姿态：航向 {frames_data[0]['heading']:.1f}° / 俯仰 {frames_data[0]['pitch']:.1f}° / 滚转 {frames_data[0]['roll']:.1f}°")
        
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ===================== 3D渲染组件 =====================
if len(frames_data) > 0:
    data_json = json.dumps(frames_data, ensure_ascii=False)
    total = len(frames_data)
    
    html_template = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family:Arial, sans-serif; }
        html, body { 
            width:100%; 
            height:100%; 
            overflow:hidden; 
            background:#0e1117; 
            color:#fff;
            display:flex;
            flex-direction:column;
        }
        .control-bar {
            height:50px; 
            display:flex; 
            align-items:center; 
            gap:15px;
            padding:0 20px; 
            background:#1a1c23; 
            border-bottom:1px solid #333;
            flex-shrink:0;
        }
        .extra-bar {
            height:42px; 
            display:flex; 
            align-items:center; 
            gap:18px;
            padding:0 20px; 
            background:#14161b; 
            border-bottom:1px solid #333;
            flex-shrink:0;
        }
        .extra-bar label {
            display:flex;
            align-items:center;
            gap:5px;
            font-size:13px;
            cursor:pointer;
        }
        .extra-bar input[type="checkbox"] {
            cursor:pointer;
            accent-color: #2d6cdf;
        }
        button { padding:6px 16px; cursor:pointer; background:#2d6cdf; border:none; color:#fff; border-radius:4px; }
        button:hover { background:#3b7eea; }
        select, input[type=range] { padding:4px; border-radius:4px; border:1px solid #444; background:#222; color:#fff; }
        .info-panel {
            position:absolute; top:112px; right:20px; width:220px;
            background:rgba(0,0,0,0.8); padding:12px 15px; border-radius:8px;
            font-size:13px; line-height:1.8; z-index:10;
        }
        .info-panel .title { font-weight:bold; margin-bottom:4px; color:#4ea1ff; border-bottom:1px solid #333; padding-bottom:2px; }
        .info-panel .row { display:flex; justify-content:space-between; }
        #tooltip {
            position:absolute;
            background:rgba(0,0,0,0.9);
            color:#fff;
            padding:10px 12px;
            border-radius:6px;
            font-size:12px;
            line-height:1.7;
            pointer-events:none;
            z-index:20;
            display:none;
            min-width:200px;
        }
        #error-tip {
            position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
            color:#ff4444; font-size:16px; z-index:99;
        }
        #canvas-container { 
            flex:1;
            width:100%; 
            min-height:0;
            position:relative;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
</head>
<body>
    <div id="error-tip">正在加载3D渲染引擎...</div>

    <div class="control-bar">
        <button id="playBtn">▶️ 播放</button>
        <label>倍速：
            <select id="speedSelect">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
                <option value="3">3x</option>
            </select>
        </label>
        <input type="range" id="frameSlider" min="0" max="__TOTAL__" value="0" style="flex:1;">
        <span id="frameText">第 1 / __TOTAL_PLUS_ONE__ 帧</span>
    </div>

    <div class="extra-bar">
        <label>视角选择：
            <select id="viewSelect">
                <option value="free">自由视角</option>
                <option value="top">俯视图</option>
                <option value="side">侧视图</option>
                <option value="front">前视图</option>
                <option value="follow">跟随飞机视角</option>
                <option value="firstPerson">第一人称驾驶</option>
            </select>
        </label>
        <label><input type="checkbox" id="showAircraftAxis" checked> 机体坐标系</label>
        <label><input type="checkbox" id="showHorizonAxis" checked> 水平参考系</label>
        <label><input type="checkbox" id="showGrid" checked> 地面网格</label>
    </div>

    <div id="canvas-container">
        <div class="info-panel">
            <div class="title">飞行状态</div>
            <div class="row"><span>高度 Altitude:</span><span id="altVal">0</span> m</div>
            <div class="row"><span>合地速 Vg:</span><span id="vgVal">0</span> m/s</div>
            <div class="row"><span>航迹角 γ:</span><span id="gammaVal">0</span> °</div>
            <div class="row"><span>飞行状态:</span><span id="statusVal">平飞</span></div>
            
            <div class="title" style="margin-top:8px;">姿态角度</div>
            <div class="row"><span>航向 Heading:</span><span id="hdgVal">0</span> °</div>
            <div class="row"><span>俯仰 Pitch:</span><span id="pitVal">0</span> °</div>
            <div class="row"><span>滚转 Roll:</span><span id="rolVal">0</span> °</div>

            <div class="title" style="margin-top:8px;">姿态角速度</div>
            <div class="row"><span>偏航速率:</span><span id="vhdgVal">0</span> °/s</div>
            <div class="row"><span>俯仰速率:</span><span id="vpitVal">0</span> °/s</div>
            <div class="row"><span>滚转速率:</span><span id="vrolVal">0</span> °/s</div>

            <div class="title" style="margin-top:8px;">地速分量</div>
            <div class="row"><span>东向 Ve:</span><span id="veVal">0</span> m/s</div>
            <div class="row"><span>北向 Vn:</span><span id="vnVal">0</span> m/s</div>
            <div class="row"><span>垂向 Vu:</span><span id="vuVal">0</span> m/s</div>
        </div>
        <div id="tooltip"></div>
    </div>

    <script>
    window.onload = function() {
        try {
            if (typeof THREE === 'undefined') {
                document.getElementById('error-tip').innerText = '渲染引擎加载失败，请检查网络';
                return;
            }

            // ========== 轨道控制器 ==========
            THREE.OrbitControls = function ( object, domElement ) {
                this.object = object;
                this.domElement = domElement;
                this.target = new THREE.Vector3();
                this.enabled = true;
                this.enableDamping = true;
                this.dampingFactor = 0.05;
                this.rotateSpeed = 0.35;
                this.zoomSpeed = 0.6;
                this.minDistance = 10;
                this.maxDistance = 100000;

                var scope = this;
                var spherical = new THREE.Spherical();
                var sphericalDelta = new THREE.Spherical();
                var scale = 1;
                var isDragging = false;
                var previousMousePosition = { x: 0, y: 0 };

                function onMouseDown( event ) {
                    if (!scope.enabled) return;
                    isDragging = true;
                    previousMousePosition.x = event.clientX;
                    previousMousePosition.y = event.clientY;
                }
                function onMouseMove( event ) {
                    if ( !isDragging || !scope.enabled ) return;
                    var deltaX = event.clientX - previousMousePosition.x;
                    var deltaY = event.clientY - previousMousePosition.y;
                    sphericalDelta.theta -= deltaX * 0.01 * scope.rotateSpeed;
                    sphericalDelta.phi -= deltaY * 0.01 * scope.rotateSpeed;
                    previousMousePosition.x = event.clientX;
                    previousMousePosition.y = event.clientY;
                }
                function onMouseUp() { isDragging = false; }
                function onMouseWheel( event ) {
                    if (!scope.enabled) return;
                    event.preventDefault();
                    if ( event.deltaY < 0 ) {
                        scale /= Math.pow( 0.95, scope.zoomSpeed );
                    } else {
                        scale *= Math.pow( 0.95, scope.zoomSpeed );
                    }
                }

                this.update = function () {
                    var offset = new THREE.Vector3();
                    var position = scope.object.position;
                    offset.copy( position ).sub( scope.target );
                    spherical.setFromVector3( offset );
                    spherical.theta += sphericalDelta.theta;
                    spherical.phi += sphericalDelta.phi;
                    spherical.phi = Math.max( 0.1, Math.min( Math.PI - 0.1, spherical.phi ) );
                    spherical.radius *= scale;
                    spherical.radius = Math.max( scope.minDistance, Math.min( scope.maxDistance, spherical.radius ) );
                    offset.setFromSpherical( spherical );
                    position.copy( scope.target ).add( offset );
                    scope.object.lookAt( scope.target );

                    if ( scope.enableDamping ) {
                        sphericalDelta.theta *= ( 1 - scope.dampingFactor );
                        sphericalDelta.phi *= ( 1 - scope.dampingFactor );
                    } else {
                        sphericalDelta.set( 0, 0, 0 );
                    }
                    scale = 1;
                };

                domElement.addEventListener( 'mousedown', onMouseDown, false );
                document.addEventListener( 'mousemove', onMouseMove, false );
                document.addEventListener( 'mouseup', onMouseUp, false );
                domElement.addEventListener( 'wheel', onMouseWheel, false );
            };

            // ========== 主渲染逻辑 ==========
            const frames = __DATA_JSON__;
            const totalFrames = frames.length;

            const AIRCRAFT_SIZE = 180;
            const AIRCRAFT_AXIS_SIZE = 220;
            const HORIZON_AXIS_SIZE  = 250;

            const statusText = ["平飞","爬升","下降","转弯"];

            function enu2three(x, y, z) {
                return new THREE.Vector3(x, z, -y);
            }

            const container = document.getElementById('canvas-container');
            const tooltip = document.getElementById('tooltip');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1d29);

            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 1000000);
            const renderer = new THREE.WebGLRenderer({ antialias:true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const ambientLight = new THREE.AmbientLight(0xffffff, 1);
            scene.add(ambientLight);

            const gridHelper = new THREE.GridHelper(10000, 50, 0x444444, 0x222222);
            scene.add(gridHelper);

            // 完整轨迹线
            const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
            const fullLine = new THREE.Line(fullLineGeo, fullLineMat);
            scene.add(fullLine);

            // 已飞轨迹线
            const flownLineGeo = new THREE.BufferGeometry().setFromPoints([allPoints[0]]);
            const flownLineMat = new THREE.LineBasicMaterial({ color: 0xff3333, linewidth: 3 });
            const flownLine = new THREE.Line(flownLineGeo, flownLineMat);
            scene.add(flownLine);

            // ========== 飞机模型 ==========
            const aircraftGroup = new THREE.Group();
            const s = AIRCRAFT_SIZE / 150;

            const bodyGeo = new THREE.BoxGeometry(100 * s, 6 * s, 6 * s);
            const bodyMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            aircraftGroup.add(body);

            const noseGeo = new THREE.BoxGeometry(20 * s, 8 * s, 8 * s);
            const noseMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
            const nose = new THREE.Mesh(noseGeo, noseMat);
            nose.position.x = 60 * s;
            aircraftGroup.add(nose);

            const wingGeo = new THREE.BoxGeometry(20 * s, 3 * s, 130 * s);
            const wingMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const wing = new THREE.Mesh(wingGeo, wingMat);
            wing.position.x = -10 * s;
            aircraftGroup.add(wing);

            const hTailGeo = new THREE.BoxGeometry(15 * s, 2 * s, 60 * s);
            const hTail = new THREE.Mesh(hTailGeo, wingMat);
            hTail.position.x = -50 * s;
            aircraftGroup.add(hTail);

            const vTailGeo = new THREE.BoxGeometry(15 * s, 35 * s, 3 * s);
            const vTailMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
            const vTail = new THREE.Mesh(vTailGeo, vTailMat);
            vTail.position.set(-50 * s, 17.5 * s, 0);
            aircraftGroup.add(vTail);

            // 机体坐标系
            const aircraftAxis = new THREE.AxesHelper(AIRCRAFT_AXIS_SIZE);
            aircraftGroup.add(aircraftAxis);
            scene.add(aircraftGroup);

            // 水平参考坐标系
            const horizonGroup = new THREE.Group();
            const hAxis = new THREE.AxesHelper(HORIZON_AXIS_SIZE);
            hAxis.material.opacity = 0.45;
            hAxis.material.transparent = true;
            horizonGroup.add(hAxis);
            scene.add(horizonGroup);

            // 相机自动适配
            const box = new THREE.Box3().setFromPoints(allPoints);
            const center = box.getCenter(new THREE.Vector3());
            const sizeVec = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(sizeVec.x, sizeVec.y, sizeVec.z) || 2000;

            let currentView = 'free';
            let followAircraft = false;
            let isFirstPerson = false;

            function setCameraView(viewName) {
                currentView = viewName;
                followAircraft = (viewName === 'follow');
                isFirstPerson = (viewName === 'firstPerson');

                // 非第一人称：恢复轨道控制器
                if (!isFirstPerson) {
                    controls.enabled = true;
                    switch (viewName) {
                        case 'top':
                            camera.position.set(center.x, maxDim * 2.5, center.z);
                            camera.lookAt(center);
                            controls.target.copy(center);
                            break;
                        case 'side':
                            camera.position.set(center.x + maxDim * 2, center.y, center.z);
                            camera.lookAt(center);
                            controls.target.copy(center);
                            break;
                        case 'front':
                            camera.position.set(center.x, center.y, center.z + maxDim * 2);
                            camera.lookAt(center);
                            controls.target.copy(center);
                            break;
                        case 'follow':
                            camera.position.set(center.x + maxDim * 1.2, center.y + maxDim * 0.8, center.z + maxDim * 1.2);
                            camera.lookAt(center);
                            controls.target.copy(center);
                            break;
                        default:
                            camera.position.set(center.x + maxDim * 1.2, center.y + maxDim * 0.8, center.z + maxDim * 1.2);
                            camera.lookAt(center);
                            controls.target.copy(center);
                            break;
                    }
                    controls.update();
                } else {
                    // 第一人称：彻底禁用控制器
                    controls.enabled = false;
                }
            }

            // ========== 控件绑定 ==========
            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const viewSelect = document.getElementById('viewSelect');
            const frameText = document.getElementById('frameText');

            const altVal = document.getElementById('altVal');
            const vgVal = document.getElementById('vgVal');
            const gammaVal = document.getElementById('gammaVal');
            const statusVal = document.getElementById('statusVal');
            const hdgVal = document.getElementById('hdgVal');
            const pitVal = document.getElementById('pitVal');
            const rolVal = document.getElementById('rolVal');
            const vhdgVal = document.getElementById('vhdgVal');
            const vpitVal = document.getElementById('vpitVal');
            const vrolVal = document.getElementById('vrolVal');
            const veVal = document.getElementById('veVal');
            const vnVal = document.getElementById('vnVal');
            const vuVal = document.getElementById('vuVal');

            // 开关事件
            document.getElementById('showAircraftAxis').addEventListener('change', e => {
                aircraftAxis.visible = e.target.checked;
            });
            document.getElementById('showHorizonAxis').addEventListener('change', e => {
                horizonGroup.visible = e.target.checked;
            });
            document.getElementById('showGrid').addEventListener('change', e => {
                gridHelper.visible = e.target.checked;
            });

            let currentFrame = 0;
            let isPlaying = false;
            let speed = 1;
            let lastTime = null;
            const frameInterval = 100;

            // 射线检测悬浮弹窗
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            function updateFrame() {
                const frame = frames[currentFrame];
                const pos = enu2three(frame.x, frame.y, frame.z);

                aircraftGroup.position.copy(pos);
                horizonGroup.position.copy(pos);

                // 原始姿态计算 完全未修改
                const h = THREE.MathUtils.degToRad(90 - frame.heading);
                const p = THREE.MathUtils.degToRad(frame.pitch);
                const r = THREE.MathUtils.degToRad(frame.roll);

                aircraftGroup.rotation.order = 'YZX';
                aircraftGroup.rotation.y = h;
                aircraftGroup.rotation.z = p;
                aircraftGroup.rotation.x = r;

                horizonGroup.rotation.y = h;

                // 更新已飞轨迹
                const flownPts = allPoints.slice(0, currentFrame + 1);
                flownLine.geometry.dispose();
                flownLine.geometry = new THREE.BufferGeometry().setFromPoints(flownPts);

                if (followAircraft) {
                    controls.target.copy(pos);
                }

                // 第一人称：手动同步相机位置和姿态
                if (isFirstPerson) {
                    const offset = new THREE.Vector3(70, 12, 0);
                    offset.applyQuaternion(aircraftGroup.quaternion);
                    camera.position.copy(aircraftGroup.position).add(offset);
                    camera.quaternion.copy(aircraftGroup.quaternion);
                }

                // 更新面板数值
                altVal.textContent = frame.z.toFixed(1);
                vgVal.textContent = frame.vg.toFixed(2);
                gammaVal.textContent = frame.gamma.toFixed(2);
                statusVal.textContent = statusText[frame.status];
                hdgVal.textContent = frame.heading.toFixed(1);
                pitVal.textContent = frame.pitch.toFixed(1);
                rolVal.textContent = frame.roll.toFixed(1);
                vhdgVal.textContent = frame.vheading.toFixed(2);
                vpitVal.textContent = frame.vpitch.toFixed(2);
                vrolVal.textContent = frame.vroll.toFixed(2);
                veVal.textContent = frame.ve.toFixed(2);
                vnVal.textContent = frame.vn.toFixed(2);
                vuVal.textContent = frame.vu.toFixed(2);

                frameText.textContent = `第 ${currentFrame + 1} / ${totalFrames} 帧`;
                frameSlider.value = currentFrame;
            }

            // 鼠标悬浮轨迹弹窗
            function onMouseMoveTooltip(event) {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObject(fullLine);

                if(intersects.length > 0){
                    const idx = Math.min(Math.max(Math.round(intersects[0].index / 3), 0), totalFrames-1);
                    const f = frames[idx];
                    tooltip.innerHTML = `
                        <b>第 ${idx+1} 帧</b><br>
                        高度: ${f.z.toFixed(1)} m<br>
                        航向/俯仰/滚转: ${f.heading.toFixed(1)}° / ${f.pitch.toFixed(1)}° / ${f.roll.toFixed(1)}°<br>
                        合地速: ${f.vg.toFixed(2)} m/s<br>
                        航迹角: ${f.gamma.toFixed(2)}°<br>
                        偏航/俯仰/滚转速率: ${f.vheading.toFixed(2)} / ${f.vpitch.toFixed(2)} / ${f.vroll.toFixed(2)} °/s<br>
                        地速分量: Ve=${f.ve.toFixed(2)} Vn=${f.vn.toFixed(2)} Vu=${f.vu.toFixed(2)} m/s<br>
                        飞行状态: ${statusText[f.status]}
                    `;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX - rect.left + 15) + 'px';
                    tooltip.style.top = (event.clientY - rect.top + 15) + 'px';
                } else {
                    tooltip.style.display = 'none';
                }
            }
            renderer.domElement.addEventListener('mousemove', onMouseMoveTooltip);

            playBtn.addEventListener('click', function() {
                if (currentFrame >= totalFrames - 1) currentFrame = 0;
                isPlaying = !isPlaying;
                lastTime = null;
                playBtn.textContent = isPlaying ? '⏸️ 暂停' : '▶️ 播放';
            });

            speedSelect.addEventListener('change', e => {
                speed = parseFloat(e.target.value);
                lastTime = null;
            });

            frameSlider.addEventListener('input', e => {
                currentFrame = parseInt(e.target.value);
                isPlaying = false;
                playBtn.textContent = '▶️ 播放';
                updateFrame();
            });

            viewSelect.addEventListener('change', e => {
                setCameraView(e.target.value);
                updateFrame();
            });

            // 动画循环
            function animate(time) {
                requestAnimationFrame(animate);
                if (isPlaying) {
                    if (lastTime === null) {
                        lastTime = time;
                    } else {
                        const delta = time - lastTime;
                        if (delta > frameInterval / speed) {
                            lastTime = time;
                            if (currentFrame < totalFrames - 1) {
                                currentFrame++;
                                updateFrame();
                            } else {
                                isPlaying = false;
                                playBtn.textContent = '▶️ 播放';
                            }
                        }
                    }
                }
                controls.update();
                renderer.render(scene, camera);
            }

            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            // 初始化
            setCameraView('free');
            updateFrame();
            animate(0);
            document.getElementById('error-tip').innerText = '';

        } catch (e) {
            document.getElementById('error-tip').innerText = '渲染错误: ' + e.message;
            console.error(e);
        }
    };
    </script>
</body>
</html>
    """
    
    html_template = html_template.replace("__DATA_JSON__", data_json)
    html_template = html_template.replace("__TOTAL__", str(total - 1))
    html_template = html_template.replace("__TOTAL_PLUS_ONE__", str(total))

    components.html(html_template, height=780, scrolling=False)

else:
    st.info("👈 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")