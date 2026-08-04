import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from io import StringIO

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化（零闪烁版）")

# ===================== 姿态旋转矩阵 =====================
def euler_rotation_matrix(heading_deg, pitch_deg, roll_deg):
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    Rh = np.array([
        [np.cos(h), -np.sin(h), 0],
        [np.sin(h), np.cos(h), 0],
        [0, 0, 1]
    ])
    Rp = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rr = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    return Rh @ Rp @ Rr

# 经纬度转局部东北天坐标系
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
        if input_mode == "粘贴CSV文本":
            df = pd.read_csv(StringIO(csv_text))
        else:
            df = pd.read_csv(uploaded_file)
        
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        for _, row in df.iterrows():
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            R = euler_rotation_matrix(row["heading"], row["pitch"], row["roll"])
            frames_data.append({
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "R": R.flatten().tolist()
            })
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")

# ===================== 渲染3D动画组件 =====================
if len(frames_data) > 0:
    data_json = json.dumps(frames_data, ensure_ascii=False)
    total = len(frames_data)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; font-family:Arial, sans-serif; }}
            html, body {{ width:100%; height:100%; overflow:hidden; background:#0e1117; color:#fff; }}
            .control-bar {{
                height:50px; display:flex; align-items:center; gap:15px;
                padding:0 20px; background:#1a1c23; border-bottom:1px solid #333;
            }}
            button {{ padding:6px 16px; cursor:pointer; background:#2d6cdf; border:none; color:#fff; border-radius:4px; }}
            button:hover {{ background:#3b7eea; }}
            select, input[type=range] {{ padding:4px; border-radius:4px; border:1px solid #444; background:#222; color:#fff; }}
            .info-panel {{
                position:absolute; top:70px; right:20px; width:180px;
                background:rgba(0,0,0,0.75); padding:15px; border-radius:8px;
                font-size:14px; line-height:2; z-index:10;
            }}
            #canvas-container {{ width:100%; height:calc(100% - 50px); }}
        </style>
    </head>
    <body>
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
            <input type="range" id="frameSlider" min="0" max="{total-1}" value="0" style="flex:1;">
            <span id="frameText">第 1 / {total} 帧</span>
        </div>
        <div id="canvas-container"></div>
        <div class="info-panel">
            <div>航向 Heading: <span id="hdgVal">0</span> °</div>
            <div>俯仰 Pitch: <span id="pitVal">0</span> °</div>
            <div>滚转 Roll: <span id="rolVal">0</span> °</div>
        </div>

        <script>
        window.onload = function() {{
            const frames = {data_json};
            const totalFrames = frames.length;

            // ENU 转 Three.js 坐标系：东→X  上→Y  北→-Z
            function enu2three(x, y, z) {{
                return new THREE.Vector3(x, z, -y);
            }}

            // ========== 初始化场景 ==========
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0e1117);

            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 1000000);
            const renderer = new THREE.WebGLRenderer({{ antialias:true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // 地面网格
            const gridHelper = new THREE.GridHelper(10000, 50, 0x444444, 0x222222);
            scene.add(gridHelper);

            // ========== 轨迹线 ==========
            const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLine = new THREE.Line(fullLineGeo, new THREE.LineBasicMaterial({{ color: 0x888888 }}));
            scene.add(fullLine);

            const flownLine = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([allPoints[0]]),
                new THREE.LineBasicMaterial({{ color: 0xff4444 }})
            );
            scene.add(flownLine);

            // ========== 飞机模型 ==========
            const aircraftGroup = new THREE.Group();
            const bodyPts = [
                new THREE.Vector3(80, 0, 0),
                new THREE.Vector3(-50, 0, 0),
                new THREE.Vector3(-10, 0, 60),
                new THREE.Vector3(-10, 0, -60),
                new THREE.Vector3(-35, 25, 0)
            ];
            [[0,1],[1,2],[1,3],[1,4]].forEach(([a,b]) => {{
                const geo = new THREE.BufferGeometry().setFromPoints([bodyPts[a], bodyPts[b]]);
                aircraftGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({{ color: 0xff3333 }})));
            }});
            aircraftGroup.add(new THREE.AxesHelper(100));
            scene.add(aircraftGroup);

            // 水平基准坐标系
            const horizonGroup = new THREE.Group();
            const hAxis = new THREE.AxesHelper(100);
            hAxis.material.opacity = 0.4;
            hAxis.material.transparent = true;
            horizonGroup.add(hAxis);
            scene.add(horizonGroup);

            // 相机自动适配
            const box = new THREE.Box3().setFromPoints(allPoints);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.set(center.x + maxDim*1.5, center.y + maxDim*1.2, center.z + maxDim*1.5);
            controls.target.copy(center);
            controls.update();

            // ========== 控件元素 ==========
            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const frameText = document.getElementById('frameText');
            const hdgVal = document.getElementById('hdgVal');
            const pitVal = document.getElementById('pitVal');
            const rolVal = document.getElementById('rolVal');

            // 播放状态
            let currentFrame = 0;
            let isPlaying = false;
            let speed = 1;
            let lastTime = null; // 修复首帧跳变bug
            const frameInterval = 100;

            // ========== 更新单帧 ==========
            function updateFrame() {{
                const frame = frames[currentFrame];
                const pos = enu2three(frame.x, frame.y, frame.z);

                aircraftGroup.position.copy(pos);
                horizonGroup.position.copy(pos);

                // 应用旋转矩阵
                const R = frame.R;
                const m = new THREE.Matrix4();
                m.set(
                    R[0], R[3], R[6], 0,
                    R[1], R[4], R[7], 0,
                    R[2], R[5], R[8], 0,
                    0, 0, 0, 1
                );
                aircraftGroup.setRotationFromMatrix(m);

                // 水平基准只转航向
                horizonGroup.rotation.y = THREE.MathUtils.degToRad(frame.heading);

                // 更新已飞轨迹
                const flownPts = allPoints.slice(0, currentFrame + 1);
                flownLine.geometry.dispose();
                flownLine.geometry = new THREE.BufferGeometry().setFromPoints(flownPts);

                // 更新UI
                hdgVal.textContent = frame.heading.toFixed(1);
                pitVal.textContent = frame.pitch.toFixed(1);
                rolVal.textContent = frame.roll.toFixed(1);
                frameText.textContent = `第 ${{currentFrame + 1}} / ${{totalFrames}} 帧`;
                frameSlider.value = currentFrame;
            }}

            // ========== 事件绑定 ==========
            playBtn.addEventListener('click', function() {{
                if (currentFrame >= totalFrames - 1) {{
                    currentFrame = 0; // 播完了就从头开始
                }}
                isPlaying = !isPlaying;
                lastTime = null; // 重置时间基准
                playBtn.textContent = isPlaying ? '⏸️ 暂停' : '▶️ 播放';
            }});

            speedSelect.addEventListener('change', e => {{
                speed = parseFloat(e.target.value);
                lastTime = null;
            }});

            frameSlider.addEventListener('input', e => {{
                currentFrame = parseInt(e.target.value);
                isPlaying = false;
                playBtn.textContent = '▶️ 播放';
                updateFrame();
            }});

            // ========== 动画主循环 ==========
            function animate(time) {{
                requestAnimationFrame(animate);

                if (isPlaying) {{
                    if (lastTime === null) {{
                        lastTime = time; // 第一帧只记录时间，不跳帧
                    }} else {{
                        const delta = time - lastTime;
                        if (delta > frameInterval / speed) {{
                            lastTime = time;
                            if (currentFrame < totalFrames - 1) {{
                                currentFrame++;
                                updateFrame();
                            }} else {{
                                isPlaying = false;
                                playBtn.textContent = '▶️ 播放';
                            }}
                        }}
                    }}
                }}

                controls.update();
                renderer.render(scene, camera);
            }}

            // 窗口自适应
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});

            // 初始化第一帧并启动循环
            updateFrame();
            animate(0);
        }};
        </script>
    </body>
    </html>
    """

    components.html(html_template, height=750, scrolling=False)

else:
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")