import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from io import StringIO

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化（零闪烁版）")

# ===================== 姿态旋转矩阵（和你之前的逻辑100%一致） =====================
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
df = None
frames_data = []
if load_btn:
    try:
        if input_mode == "粘贴CSV文本":
            df = pd.read_csv(StringIO(csv_text))
        else:
            df = pd.read_csv(uploaded_file)
        
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        # 一次性预处理所有帧
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
                "R": R.flatten().tolist()  # 旋转矩阵展平，保证姿态100%一致
            })
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")

# ===================== 渲染3D动画组件（零闪烁核心） =====================
if len(frames_data) > 0:
    # 把数据转成JSON字符串，注入到JS里
    data_json = json.dumps(frames_data)
    
    # HTML + JS 模板
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; font-family:Arial, sans-serif; }}
            body {{ background:#0e1117; color:#fff; }}
            .control-bar {{
                height:50px; display:flex; align-items:center; gap:15px;
                padding:0 20px; background:#1a1c23; border-bottom:1px solid #333;
            }}
            button {{ padding:6px 16px; cursor:pointer; background:#2d6cdf; border:none; color:#fff; border-radius:4px; }}
            button:hover {{ background:#3b7eea; }}
            select, input[type=range] {{ padding:4px; border-radius:4px; border:1px solid #444; background:#222; color:#fff; }}
            .info-panel {{
                position:absolute; top:70px; right:20px; width:180px;
                background:rgba(0,0,0,0.7); padding:15px; border-radius:8px;
                font-size:14px; line-height:2;
            }}
            #canvas-container {{ width:100%; height:calc(100vh - 50px); }}
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
            <input type="range" id="frameSlider" min="0" max="{len(frames_data)-1}" value="0" style="flex:1;">
            <span id="frameText">第 1 / {len(frames_data)} 帧</span>
        </div>

        <div id="canvas-container"></div>

        <div class="info-panel">
            <div>航向 Heading: <span id="hdgVal">0</span> °</div>
            <div>俯仰 Pitch: <span id="pitVal">0</span> °</div>
            <div>滚转 Roll: <span id="rolVal">0</span> °</div>
        </div>

        <script>
            // 接收Python传过来的数据
            const frames = {data_json};
            const totalFrames = frames.length;

            // ========== 初始化Three.js场景 ==========
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0e1117);

            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 100000);
            camera.position.set(2000, -2000, 1500);

            const renderer = new THREE.WebGLRenderer({{ antialias:true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // 网格辅助
            const gridHelper = new THREE.GridHelper(10000, 50, 0x444444, 0x222222);
            gridHelper.rotation.x = Math.PI / 2; // 网格放在XY水平面（东北天）
            scene.add(gridHelper);

            // ========== 绘制完整航线 ==========
            const allPoints = frames.map(f => new THREE.Vector3(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLineMat = new THREE.LineBasicMaterial({{ color: 0x888888 }});
            const fullLine = new THREE.Line(fullLineGeo, fullLineMat);
            scene.add(fullLine);

            // 已飞轨迹（动态更新）
            const flownLineGeo = new THREE.BufferGeometry().setFromPoints([allPoints[0]]);
            const flownLineMat = new THREE.LineBasicMaterial({{ color: 0xff4444, linewidth:3 }});
            const flownLine = new THREE.Line(flownLineGeo, flownLineMat);
            scene.add(flownLine);

            // ========== 创建飞机模型 ==========
            const aircraftGroup = new THREE.Group();
            // 机体顶点（机头向前X轴）
            const bodyPoints = [
                new THREE.Vector3(80, 0, 0),    // 机头
                new THREE.Vector3(-50, 0, 0),   // 机尾
                new THREE.Vector3(-10, 60, 0),  // 右翼
                new THREE.Vector3(-10, -60, 0), // 左翼
                new THREE.Vector3(-35, 0, 25)   // 垂尾
            ];
            // 连线
            const lines = [[0,1],[1,2],[1,3],[1,4]];
            lines.forEach(([a,b]) => {{
                const geo = new THREE.BufferGeometry().setFromPoints([bodyPoints[a], bodyPoints[b]]);
                const mat = new THREE.LineBasicMaterial({{ color: 0xff3333 }});
                aircraftGroup.add(new THREE.Line(geo, mat));
            }});
            scene.add(aircraftGroup);

            // 机体坐标轴（红X绿Y蓝Z）
            const axisHelper = new THREE.AxesHelper(100);
            aircraftGroup.add(axisHelper);

            // 水平基准坐标轴（灰色虚线）
            const horizonGroup = new THREE.Group();
            const horizonAxis = new THREE.AxesHelper(100);
            horizonAxis.material.transparent = true;
            horizonAxis.material.opacity = 0.4;
            horizonGroup.add(horizonAxis);
            scene.add(horizonGroup);

            // ========== 播放控制 ==========
            let currentFrame = 0;
            let isPlaying = false;
            let speed = 1;
            let lastTime = 0;
            const frameInterval = 100; // 每帧间隔ms，1倍速

            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const frameText = document.getElementById('frameText');
            const hdgVal = document.getElementById('hdgVal');
            const pitVal = document.getElementById('pitVal');
            const rolVal = document.getElementById('rolVal');

            playBtn.addEventListener('click', () => {{
                isPlaying = !isPlaying;
                playBtn.textContent = isPlaying ? '⏸️ 暂停' : '▶️ 播放';
            }});

            speedSelect.addEventListener('change', e => {{
                speed = parseFloat(e.target.value);
            }});

            frameSlider.addEventListener('input', e => {{
                currentFrame = parseInt(e.target.value);
                updateFrame();
            }});

            // 更新单帧画面
            function updateFrame() {{
                const frame = frames[currentFrame];
                
                // 更新飞机位置
                aircraftGroup.position.set(frame.x, frame.y, frame.z);
                horizonGroup.position.set(frame.x, frame.y, frame.z);

                // 应用旋转矩阵（和Python计算结果完全一致）
                const m = new THREE.Matrix4();
                m.set(
                    frame.R[0], frame.R[3], frame.R[6], 0,
                    frame.R[1], frame.R[4], frame.R[7], 0,
                    frame.R[2], frame.R[5], frame.R[8], 0,
                    0, 0, 0, 1
                );
                aircraftGroup.setRotationFromMatrix(m);

                // 更新已飞轨迹
                const flownPoints = allPoints.slice(0, currentFrame + 1);
                flownLine.geometry.dispose();
                flownLine.geometry = new THREE.BufferGeometry().setFromPoints(flownPoints);

                // 更新UI数值
                hdgVal.textContent = frame.heading.toFixed(1);
                pitVal.textContent = frame.pitch.toFixed(1);
                rolVal.textContent = frame.roll.toFixed(1);
                frameText.textContent = `第 ${{currentFrame + 1}} / ${{totalFrames}} 帧`;
                frameSlider.value = currentFrame;
            }}

            // 动画循环（浏览器原生，零闪烁）
            function animate(time) {{
                requestAnimationFrame(animate);
                
                if (isPlaying) {{
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

                controls.update();
                renderer.render(scene, camera);
            }}

            // 窗口大小自适应
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});

            // 初始化第一帧
            updateFrame();
            animate(0);
        </script>
    </body>
    </html>
    """

    # 渲染组件，高度设为800，适配页面
    components.html(html_template, height=800, scrolling=False)

else:
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")