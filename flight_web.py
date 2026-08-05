import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from io import StringIO, BytesIO

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化工具")

# ===================== 标准航空姿态解算 =====================
# 机体坐标系：X机头向前，Y右翼向右，Z机腹向下
# ENU世界坐标系：X东，Y北，Z天
def body_to_enu_rotation_matrix(heading_deg, pitch_deg, roll_deg):
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    # 初始零姿态：机头朝北，右翼朝东，机腹朝下
    R0 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])

    # 航向旋转（绕竖轴）
    Rz = np.array([
        [np.cos(h), -np.sin(h), 0],
        [np.sin(h), np.cos(h), 0],
        [0, 0, 1]
    ])
    # 俯仰旋转（绕机翼横轴）
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    # 滚转旋转（绕机头纵轴）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])

    # 航空标准内在旋转顺序：航向 → 俯仰 → 滚转
    return R0 @ Rz @ Ry @ Rx

# ENU → Three.js 坐标系变换矩阵
ENU_TO_THREE_T = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

# 经纬度转局部东北天
def ll2local_enu(lat0, lon0, lat, lon, alt):
    R = 6371000
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    east = R * dlon * np.cos(lat0_rad)
    north = R * dlat
    up = alt
    return east, north, up

# ===================== 侧边栏数据输入 =====================
with st.sidebar:
    st.header("数据输入")
    input_mode = st.radio("选择数据输入方式", ["粘贴CSV文本", "上传CSV文件"])
    csv_text = st.text_area("粘贴CSV数据", height=280)
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    load_btn = st.button("加载数据")

# 加载数据并预处理
frames_data = []
if load_btn:
    try:
        # 1. 读取CSV数据
        if input_mode == "粘贴CSV文本":
            if not csv_text.strip():
                st.error("请粘贴有效的CSV数据")
                st.stop()
            df = pd.read_csv(StringIO(csv_text))
        else:
            if uploaded_file is None:
                st.error("请先选择要上传的CSV文件")
                st.stop()
            # 修复：用字节流读取上传文件，解决Streamlit文件读取兼容性问题
            file_bytes = uploaded_file.getvalue()
            df = pd.read_csv(BytesIO(file_bytes))
        
        # 2. 校验必填列是否存在
        required_cols = ["latitude", "longitude", "altitude", "heading", "pitch", "roll"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV文件缺少必填列：{missing_cols}，请检查文件格式")
            st.stop()
        
        # 3. 去除空值行
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if len(df) == 0:
            st.error("CSV文件有效数据行不足，清洗后无可用数据")
            st.stop()
        
        # 4. 坐标转换与数据预处理
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        for _, row in df.iterrows():
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            R_enu = body_to_enu_rotation_matrix(row["heading"], row["pitch"], row["roll"])
            R_three = ENU_TO_THREE_T @ R_enu
            frames_data.append({
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "R": R_three.flatten().tolist()
            })
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")
        import traceback
        st.error(f"详细错误栈：{traceback.format_exc()}")

# ===================== 渲染3D动画组件 =====================
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
        html, body { width:100%; height:700px; overflow:hidden; background:#0e1117; color:#fff; }
        .control-bar {
            height:50px; display:flex; align-items:center; gap:15px;
            padding:0 20px; background:#1a1c23; border-bottom:1px solid #333;
        }
        .extra-bar {
            height:42px; display:flex; align-items:center; gap:15px;
            padding:0 20px; background:#14161b; border-bottom:1px solid #333;
        }
        button { padding:6px 16px; cursor:pointer; background:#2d6cdf; border:none; color:#fff; border-radius:4px; }
        button:hover { background:#3b7eea; }
        select, input[type=range] { padding:4px; border-radius:4px; border:1px solid #444; background:#222; color:#fff; }
        .info-panel {
            position:absolute; top:112px; right:20px; width:180px;
            background:rgba(0,0,0,0.75); padding:15px; border-radius:8px;
            font-size:14px; line-height:2; z-index:10;
        }
        #error-tip {
            position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
            color:#ff4444; font-size:16px; z-index:99;
        }
        #canvas-container { width:100%; height:608px; }
    </style>
</head>
<body>
    <div id="error-tip">正在加载渲染引擎...</div>

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
            </select>
        </label>
    </div>

    <div id="canvas-container"></div>

    <div class="info-panel">
        <div>航向 Heading: <span id="hdgVal">0</span> °</div>
        <div>俯仰 Pitch: <span id="pitVal">0</span> °</div>
        <div>滚转 Roll: <span id="rolVal">0</span> °</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script>
    // ========== 内嵌 OrbitControls ==========
    THREE.OrbitControls = function ( object, domElement ) {
        this.object = object;
        this.domElement = domElement;
        this.target = new THREE.Vector3();
        this.enableDamping = true;
        this.dampingFactor = 0.05;
        this.rotateSpeed = 0.35;
        this.zoomSpeed = 0.6;
        this.minDistance = 0;
        this.maxDistance = Infinity;

        var scope = this;
        var spherical = new THREE.Spherical();
        var sphericalDelta = new THREE.Spherical();
        var scale = 1;
        var isDragging = false;
        var previousMousePosition = { x: 0, y: 0 };

        function onMouseDown( event ) {
            isDragging = true;
            previousMousePosition.x = event.clientX;
            previousMousePosition.y = event.clientY;
        }

        function onMouseMove( event ) {
            if ( !isDragging ) return;
            var deltaX = event.clientX - previousMousePosition.x;
            var deltaY = event.clientY - previousMousePosition.y;
            sphericalDelta.theta -= deltaX * 0.01 * scope.rotateSpeed;
            sphericalDelta.phi -= deltaY * 0.01 * scope.rotateSpeed;
            previousMousePosition.x = event.clientX;
            previousMousePosition.y = event.clientY;
        }

        function onMouseUp() {
            isDragging = false;
        }

        function onMouseWheel( event ) {
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

    // ========== 主逻辑 ==========
    window.onerror = function(msg) {
        document.getElementById('error-tip').innerText = '渲染错误: ' + msg;
    };

    const frames = __DATA_JSON__;
    const totalFrames = frames.length;

    const AIRCRAFT_SIZE = 180;
    const AIRCRAFT_AXIS_SIZE = 220;
    const HORIZON_AXIS_SIZE  = 250;

    function enu2three(x, y, z) {
        return new THREE.Vector3(x, z, -y);
    }

    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1117);

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 1000000);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const gridHelper = new THREE.GridHelper(10000, 50, 0x444444, 0x222222);
    scene.add(gridHelper);

    const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
    const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
    const fullLine = new THREE.Line(fullLineGeo, new THREE.LineBasicMaterial({ color: 0x888888 }));
    scene.add(fullLine);

    const flownLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([allPoints[0]]),
        new THREE.LineBasicMaterial({ color: 0xff4444 })
    );
    scene.add(flownLine);

           // ========== 飞机模型：零姿态下机头沿+X，机翼沿Z展开，机身水平 ==========
    const aircraftGroup = new THREE.Group();
    const s = AIRCRAFT_SIZE / 150;

    // 机身：沿X轴，机头在+X，机尾在-X
    const bodyGeo = new THREE.BoxGeometry(100 * s, 6 * s, 6 * s);
    const bodyMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    aircraftGroup.add(body);

    // 机头红色标识：在+X最前端
    const noseGeo = new THREE.BoxGeometry(20 * s, 8 * s, 8 * s);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
    const nose = new THREE.Mesh(noseGeo, noseMat);
    nose.position.x = 60 * s;
    aircraftGroup.add(nose);

    // 主翼：沿Z轴左右展开，在机身中后段
    const wingGeo = new THREE.BoxGeometry(20 * s, 3 * s, 130 * s);
    const wingMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const wing = new THREE.Mesh(wingGeo, wingMat);
    wing.position.x = -10 * s;
    aircraftGroup.add(wing);

    // 水平尾翼：沿Z轴展开，在机尾
    const hTailGeo = new THREE.BoxGeometry(15 * s, 2 * s, 60 * s);
    const hTail = new THREE.Mesh(hTailGeo, wingMat);
    hTail.position.x = -50 * s;
    aircraftGroup.add(hTail);

    // 垂直尾翼：沿Y轴向上，在机尾
    const vTailGeo = new THREE.BoxGeometry(15 * s, 35 * s, 3 * s);
    const vTailMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
    const vTail = new THREE.Mesh(vTailGeo, vTailMat);
    vTail.position.set(-50 * s, 17.5 * s, 0);
    aircraftGroup.add(vTail);

    // 机体坐标轴
    aircraftGroup.add(new THREE.AxesHelper(AIRCRAFT_AXIS_SIZE));
    scene.add(aircraftGroup);

    // 水平基准坐标轴
    const horizonGroup = new THREE.Group();
    const hAxis = new THREE.AxesHelper(HORIZON_AXIS_SIZE);
    hAxis.material.opacity = 0.45;
    hAxis.material.transparent = true;
    horizonGroup.add(hAxis);
    scene.add(horizonGroup);

    // 相机适配
    const box = new THREE.Box3().setFromPoints(allPoints);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1000;

    let currentView = 'free';
    let followAircraft = false;

    function setCameraView(viewName) {
        currentView = viewName;
        followAircraft = (viewName === 'follow');
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
            case 'free':
            case 'follow':
                camera.position.set(center.x + maxDim * 1.5, center.y + maxDim * 1.2, center.z + maxDim * 1.5);
                controls.target.copy(center);
                break;
        }
        controls.update();
    }

    // 控件获取
    const playBtn = document.getElementById('playBtn');
    const speedSelect = document.getElementById('speedSelect');
    const frameSlider = document.getElementById('frameSlider');
    const viewSelect = document.getElementById('viewSelect');
    const frameText = document.getElementById('frameText');
    const hdgVal = document.getElementById('hdgVal');
    const pitVal = document.getElementById('pitVal');
    const rolVal = document.getElementById('rolVal');

    // 播放状态
    let currentFrame = 0;
    let isPlaying = false;
    let speed = 1;
    let lastTime = null;
    const frameInterval = 100;

    // 更新单帧姿态
function updateFrame() {
    const frame = frames[currentFrame];
    const pos = enu2three(frame.x, frame.y, frame.z);

    aircraftGroup.position.copy(pos);
    horizonGroup.position.copy(pos);

    const h = THREE.MathUtils.degToRad(90 - frame.heading);
    const p = THREE.MathUtils.degToRad(frame.pitch);
    const r = THREE.MathUtils.degToRad(frame.roll);

    // 航空标准内在旋转顺序：航向 → 俯仰 → 滚转
    aircraftGroup.rotation.order = 'YZX';
    // 航向：绕Y轴水平旋转，对齐正北基准（保持原公式不变）
    aircraftGroup.rotation.y = h;
    // 俯仰：去掉负号，修正抬头方向
    aircraftGroup.rotation.z = p;
    // 滚转：绕X轴（机身纵轴），右翼下沉为正（保持不变）
    aircraftGroup.rotation.x = r;

    // 水平基准：只同步航向，永远保持水平
    horizonGroup.rotation.y = h;

    // 更新已飞轨迹
    const flownPts = allPoints.slice(0, currentFrame + 1);
    flownLine.geometry.dispose();
    flownLine.geometry = new THREE.BufferGeometry().setFromPoints(flownPts);

    if (followAircraft) {
        controls.target.copy(pos);
    }

    hdgVal.textContent = frame.heading.toFixed(1);
    pitVal.textContent = frame.pitch.toFixed(1);
    rolVal.textContent = frame.roll.toFixed(1);
    frameText.textContent = `第 ${currentFrame + 1} / ${totalFrames} 帧`;
    frameSlider.value = currentFrame;
}
    // 事件绑定
    playBtn.addEventListener('click', function() {
        if (currentFrame >= totalFrames - 1) {
            currentFrame = 0;
        }
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

    // 窗口自适应
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
    </script>
</body>
</html>
    """
    
    # 注入数据
    html_template = html_template.replace("__DATA_JSON__", data_json)
    html_template = html_template.replace("__TOTAL__", str(total - 1))
    html_template = html_template.replace("__TOTAL_PLUS_ONE__", str(total))

    components.html(html_template, height=750, scrolling=False)

else:
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")