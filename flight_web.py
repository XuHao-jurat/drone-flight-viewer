import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from io import StringIO

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化工具")

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

# ===================== 鲁棒CSV解析（数值特征自动匹配列） =====================
def parse_csv_data(csv_string):
    df = pd.read_csv(StringIO(csv_string))
    df.columns = df.columns.str.strip()

    # 所有列强制转数值
    num_df = df.apply(pd.to_numeric, errors="coerce")

    # 统计每列数值特征
    col_info = []
    for col in num_df.columns:
        vals = num_df[col].dropna()
        if len(vals) == 0:
            continue
        col_info.append({
            "name": col,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std())
        })

    matched = {}

    # 1. 匹配高度：数值最大，均值>1000
    altitude_candidates = [c for c in col_info if c["mean"] > 1000]
    if altitude_candidates:
        matched["altitude"] = max(altitude_candidates, key=lambda x: x["mean"])["name"]

    # 2. 匹配经度：-180~180，绝对值>90
    lon_candidates = [
        c for c in col_info
        if -180 <= c["min"] and c["max"] <= 180 and abs(c["mean"]) > 90
    ]
    if lon_candidates:
        matched["longitude"] = lon_candidates[0]["name"]

    # 3. 匹配纬度：-90~90，绝对值在20~80之间
    lat_candidates = [
        c for c in col_info
        if -90 <= c["min"] and c["max"] <= 90 and 20 < abs(c["mean"]) < 80
    ]
    if lat_candidates:
        matched["latitude"] = lat_candidates[0]["name"]

    # 4. 匹配航向：0~360，均值在100~300之间
    heading_candidates = [
        c for c in col_info
        if 0 <= c["min"] and c["max"] <= 360 and 100 < c["mean"] < 300
    ]
    if heading_candidates:
        matched["heading"] = heading_candidates[0]["name"]

    # 5. 匹配俯仰/滚转：小角度(-10~10)，标准差>0.1
    angle_candidates = [
        c for c in col_info
        if -10 <= c["min"] and c["max"] <= 10 and c["std"] > 0.1
    ]
    if len(angle_candidates) >= 2:
        angle_candidates.sort(key=lambda x: x["mean"], reverse=True)
        matched["pitch"] = angle_candidates[0]["name"]
        matched["roll"] = angle_candidates[1]["name"]
    elif len(angle_candidates) == 1:
        matched["pitch"] = angle_candidates[0]["name"]
        matched["roll"] = None

    # 兜底：自动匹配失败时，按位置强制映射
    required = ["latitude", "longitude", "altitude", "heading", "pitch", "roll"]
    missing = [k for k in required if k not in matched or matched[k] is None]
    if missing:
        if len(num_df.columns) >= 6:
            cols = list(num_df.columns)
            # 从timestamp之后取6列映射
            start_idx = 1 if len(cols) > 6 else 0
            matched = {
                "latitude": cols[start_idx],
                "longitude": cols[start_idx+1],
                "altitude": cols[start_idx+2],
                "heading": cols[start_idx+3],
                "pitch": cols[start_idx+4],
                "roll": cols[start_idx+5]
            }
        else:
            raise ValueError(f"列数不足，无法解析。当前列：{list(num_df.columns)}")

    # 构建标准数据表
    std_df = pd.DataFrame()
    std_df["latitude"] = num_df[matched["latitude"]]
    std_df["longitude"] = num_df[matched["longitude"]]
    std_df["altitude"] = num_df[matched["altitude"]]
    std_df["heading"] = num_df[matched["heading"]]
    std_df["pitch"] = num_df[matched["pitch"]]
    std_df["roll"] = num_df[matched["roll"]] if matched["roll"] else 0.0

    # 清洗空值
    std_df = std_df.dropna().reset_index(drop=True)
    if len(std_df) == 0:
        raise ValueError("有效数据行为0")

    # 调试输出
    with st.expander("🔍 解析诊断详情", expanded=False):
        st.write("原始列名：", list(df.columns))
        st.write("自动匹配映射：", matched)
        st.dataframe(std_df.head(10), use_container_width=True)
        st.write(f"纬度范围：{std_df['latitude'].min():.4f} ~ {std_df['latitude'].max():.4f}")
        st.write(f"经度范围：{std_df['longitude'].min():.4f} ~ {std_df['longitude'].max():.4f}")
        st.write(f"高度范围：{std_df['altitude'].min():.2f} ~ {std_df['altitude'].max():.2f}")
        st.write(f"航向范围：{std_df['heading'].min():.2f} ~ {std_df['heading'].max():.2f}")
        st.write(f"俯仰范围：{std_df['pitch'].min():.2f} ~ {std_df['pitch'].max():.2f}")
        st.write(f"滚转范围：{std_df['roll'].min():.2f} ~ {std_df['roll'].max():.2f}")

    return std_df

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
            file_bytes = uploaded_file.getvalue()
            try:
                raw_text = file_bytes.decode('utf-8-sig')
            except UnicodeDecodeError:
                raw_text = file_bytes.decode('gbk')
        
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
                "roll": float(row["roll"])
            })
        st.success(f"✅ 数据加载成功，共 {len(frames_data)} 帧")
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
            gap:15px;
            padding:0 20px; 
            background:#14161b; 
            border-bottom:1px solid #333;
            flex-shrink:0;
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
        #canvas-container { 
            flex:1;
            width:100%; 
            min-height:0;
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
            </select>
        </label>
    </div>

    <div id="canvas-container"></div>

    <div class="info-panel">
        <div>航向 Heading: <span id="hdgVal">0</span> °</div>
        <div>俯仰 Pitch: <span id="pitVal">0</span> °</div>
        <div>滚转 Roll: <span id="rolVal">0</span> °</div>
    </div>

    <script>
    window.onload = function() {
        try {
            if (typeof THREE === 'undefined') {
                document.getElementById('error-tip').innerText = '渲染引擎加载失败，请检查网络';
                return;
            }

            // ========== 内置轨道控制器 ==========
            THREE.OrbitControls = function ( object, domElement ) {
                this.object = object;
                this.domElement = domElement;
                this.target = new THREE.Vector3();
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
                function onMouseUp() { isDragging = false; }
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

            // ========== 主渲染逻辑 ==========
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

            // 轨迹线
            const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLine = new THREE.Line(fullLineGeo, new THREE.LineBasicMaterial({ color: 0x888888 }));
            scene.add(fullLine);

            const flownLine = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([allPoints[0]]),
                new THREE.LineBasicMaterial({ color: 0xff4444 })
            );
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

            aircraftGroup.add(new THREE.AxesHelper(AIRCRAFT_AXIS_SIZE));
            scene.add(aircraftGroup);

            // 水平坐标系
            const horizonGroup = new THREE.Group();
            const hAxis = new THREE.AxesHelper(HORIZON_AXIS_SIZE);
            hAxis.material.opacity = 0.45;
            hAxis.material.transparent = true;
            horizonGroup.add(hAxis);
            scene.add(horizonGroup);

            // 相机自动适配
            const box = new THREE.Box3().setFromPoints(allPoints);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z) || 2000;

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
                    default:
                        camera.position.set(center.x + maxDim * 1.2, center.y + maxDim * 0.8, center.z + maxDim * 1.2);
                        controls.target.copy(center);
                        break;
                }
                controls.update();
            }

            // 控件绑定
            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const viewSelect = document.getElementById('viewSelect');
            const frameText = document.getElementById('frameText');
            const hdgVal = document.getElementById('hdgVal');
            const pitVal = document.getElementById('pitVal');
            const rolVal = document.getElementById('rolVal');

            let currentFrame = 0;
            let isPlaying = false;
            let speed = 1;
            let lastTime = null;
            const frameInterval = 100;

            function updateFrame() {
                const frame = frames[currentFrame];
                const pos = enu2three(frame.x, frame.y, frame.z);

                aircraftGroup.position.copy(pos);
                horizonGroup.position.copy(pos);

                const h = THREE.MathUtils.degToRad(90 - frame.heading);
                const p = THREE.MathUtils.degToRad(frame.pitch);
                const r = THREE.MathUtils.degToRad(frame.roll);

                aircraftGroup.rotation.order = 'YZX';
                aircraftGroup.rotation.y = h;
                aircraftGroup.rotation.z = p;
                aircraftGroup.rotation.x = r;

                horizonGroup.rotation.y = h;

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
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")