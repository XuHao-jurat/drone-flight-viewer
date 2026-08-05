# 导入streamlit网页框架库
import streamlit as st
# 导入streamlit内嵌html组件，用来加载Three.js 3D页面
import streamlit.components.v1 as components
# pandas，用于读取解析csv表格数据
import pandas as pd
# numpy，矩阵、角度弧度转换、矩阵运算
import numpy as np
# json，把python帧数据转为js可识别json字符串
import json
# io.StringIO，把粘贴的csv文本模拟成文件对象给pandas读取
from io import StringIO

# ===================== 页面基础配置 =====================
# 设置网页页面配置：网页标题，页面布局为宽屏模式
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
# 设置网页大标题
st.title("✈️ 无人机飞行轨迹与姿态可视化工具")

# ===================== 标准航空姿态解算 =====================
# 机体坐标系：X机头向前，Y右翼向右，Z机腹向下
# ENU世界坐标系：X东，Y北，Z天（向上）
def body_to_enu_rotation_matrix(heading_deg, pitch_deg, roll_deg):
    """
    输入航向、俯仰、滚转角度(角度制)
    返回：机体坐标系转到ENU东北天坐标系的旋转矩阵
    """
    # 角度转为弧度，numpy三角函数需要弧度输入
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    # R0初始变换矩阵：零姿态条件：机头朝北，右翼朝东，机腹朝下
    R0 = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])

    # Rz：航向旋转矩阵，绕ENU竖轴Z旋转（偏航/航向）
    Rz = np.array([
        [np.cos(h), -np.sin(h), 0],
        [np.sin(h), np.cos(h), 0],
        [0, 0, 1]
    ])
    # Ry：俯仰旋转矩阵，绕机翼横轴Y旋转（俯仰）
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    # Rx：滚转旋转矩阵，绕机身纵轴X旋转（滚转）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])

    # 矩阵乘法 @：航空内在旋转顺序 R0 * Rz(航向) * Ry(俯仰) * Rx(滚转)
    return R0 @ Rz @ Ry @ Rx

# ENU → Three.js 坐标系变换矩阵，把东北天坐标映射到Three.js右手坐标系
ENU_TO_THREE_T = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

def ll2local_enu(lat0, lon0, lat, lon, alt):
    """
    经纬度转局部ENU东北天米制坐标
    lat0,lon0：原点基准经纬度（第一帧）
    lat,lon,alt：当前点经纬度、高度
    return east东，north北，up天
    """
    # 地球半径，单位米
    R = 6371000
    # 纬度差值（弧度）
    dlat = np.radians(lat - lat0)
    # 经度差值（弧度）
    dlon = np.radians(lon - lon0)
    # 原点纬度转弧度
    lat0_rad = np.radians(lat0)
    # 东向位移
    east = R * dlon * np.cos(lat0_rad)
    # 北向位移
    north = R * dlat
    # 向上高度，直接使用alt
    up = alt
    return east, north, up

# ===================== 侧边栏数据输入 =====================
# 创建侧边栏容器
with st.sidebar:
    # 侧边栏标题
    st.header("数据输入")
    # 单选框，选择输入模式：粘贴文本 / 上传文件
    input_mode = st.radio("选择数据输入方式", ["粘贴CSV文本", "上传CSV文件"])
    # 多行文本框，粘贴csv原始文本
    csv_text = st.text_area("粘贴CSV数据", height=280)
    # 文件上传控件，仅接受csv后缀
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    # 加载数据按钮
    load_btn = st.button("加载数据")

# 定义存储每一帧数据的空列表
frames_data = []
# 如果点击加载按钮
if load_btn:
    try:
        # 判断输入模式
        if input_mode == "粘贴CSV文本":
            # StringIO把文本伪装成文件对象，pandas读取
            df = pd.read_csv(StringIO(csv_text))
        else:
            # 读取上传的csv文件
            df = pd.read_csv(uploaded_file)
        
        # 取第一帧作为ENU局部坐标系原点
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        # 遍历csv每一行数据
        for _, row in df.iterrows():
            # 将当前行经纬度高度转为ENU局部坐标
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            # 计算机体到ENU的旋转矩阵
            R_enu = body_to_enu_rotation_matrix(row["heading"], row["pitch"], row["roll"])
            # 再转换为适配Three.js的旋转矩阵
            R_three = ENU_TO_THREE_T @ R_enu
            # 将一帧所有信息存入列表
            frames_data.append({
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "R": R_three.flatten().tolist()
            })
        # 页面输出成功提示，打印总帧数
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
    except Exception as e:
        # 捕获异常，打印错误信息
        st.error(f"数据解析失败：{str(e)}")

# ===================== 渲染3D动画组件 =====================
# 如果已经成功加载帧数据
if len(frames_data) > 0:
    # python帧数据转为json字符串，注入html里面的js
    data_json = json.dumps(frames_data, ensure_ascii=False)
    # 获取总帧数
    total = len(frames_data)
    
    html_template = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <!-- CSS样式定义页面布局 -->
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family:Arial, sans-serif; }
        html, body { width:100%; height:700px; overflow:hidden; background:#0e1117; color:#fff; }
        /* 上方控制栏：播放、倍速 */
        .control-bar {
            height:50px; display:flex; align-items:center; gap:15px;
            padding:0 20px; background:#1a1c23; border-bottom:1px solid #333;
        }
        /* 第二行控制栏：视角选择 */
        .extra-bar {
            height:42px; display:flex; align-items:center; gap:15px;
            padding:0 20px; background:#14161b; border-bottom:1px solid #333;
        }
        button { padding:6px 16px; cursor:pointer; background:#2d6cdf; border:none; color:#fff; border-radius:4px; }
        button:hover { background:#3b7eea; }
        select, input[type=range] { padding:4px; border-radius:4px; border:1px solid #444; background:#222; color:#fff; }
        /* 右侧姿态信息面板 */
        .info-panel {
            position:absolute; top:112px; right:20px; width:180px;
            background:rgba(0,0,0,0.75); padding:15px; border-radius:8px;
            font-size:14px; line-height:2; z-index:10;
        }
        /* 错误提示文字 */
        #error-tip {
            position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
            color:#ff4444; font-size:16px; z-index:99;
        }
        /* 3D画布容器 */
        #canvas-container { width:100%; height:608px; }
    </style>
</head>
<body>
    <!-- 渲染加载提示 -->
    <div id="error-tip">正在加载渲染引擎...</div>

    <!-- 第一行控制栏：播放按钮、倍速、进度条 -->
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

    <!-- 第二行：视角下拉框 -->
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

    <!-- 3D画布容器DOM -->
    <div id="canvas-container"></div>

    <!-- 右侧姿态信息面板 -->
    <div class="info-panel">
        <div>航向 Heading: <span id="hdgVal">0</span> °</div>
        <div>俯仰 Pitch: <span id="pitVal">0</span> °</div>
        <div>滚转 Roll: <span id="rolVal">0</span> °</div>
    </div>

    <!-- 引入Three.js CDN库 -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script>
    // ========== 内嵌 OrbitControls 轨道控制器，实现鼠标拖拽旋转缩放视角 ==========
    THREE.OrbitControls = function ( object, domElement ) {
        // camera对象
        this.object = object;
        // 绑定的dom画布
        this.domElement = domElement;
        // 相机lookAt的目标点
        this.target = new THREE.Vector3();
        // 开启阻尼平滑动画
        this.enableDamping = true;
        this.dampingFactor = 0.05;
        // 鼠标旋转灵敏度
        this.rotateSpeed = 0.35;
        // 滚轮缩放灵敏度
        this.zoomSpeed = 0.6;
        this.minDistance = 0;
        this.maxDistance = Infinity;

        var scope = this;
        var spherical = new THREE.Spherical();
        var sphericalDelta = new THREE.Spherical();
        var scale = 1;
        var isDragging = false;
        var previousMousePosition = { x: 0, y: 0 };

        // 鼠标按下事件
        function onMouseDown( event ) {
            isDragging = true;
            previousMousePosition.x = event.clientX;
            previousMousePosition.y = event.clientY;
        }

        // 鼠标移动事件
        function onMouseMove( event ) {
            if ( !isDragging ) return;
            var deltaX = event.clientX - previousMousePosition.x;
            var deltaY = event.clientY - previousMousePosition.y;
            sphericalDelta.theta -= deltaX * 0.01 * scope.rotateSpeed;
            sphericalDelta.phi -= deltaY * 0.01 * scope.rotateSpeed;
            previousMousePosition.x = event.clientX;
            previousMousePosition.y = event.clientY;
        }

        // 鼠标抬起
        function onMouseUp() {
            isDragging = false;
        }

        // 鼠标滚轮缩放
        function onMouseWheel( event ) {
            event.preventDefault();
            if ( event.deltaY < 0 ) {
                scale /= Math.pow( 0.95, scope.zoomSpeed );
            } else {
                scale *= Math.pow( 0.95, scope.zoomSpeed );
            }
        }

        // 控制器更新函数，每一帧调用
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

        // 绑定DOM事件监听
        domElement.addEventListener( 'mousedown', onMouseDown, false );
        document.addEventListener( 'mousemove', onMouseMove, false );
        document.addEventListener( 'mouseup', onMouseUp, false );
        domElement.addEventListener( 'wheel', onMouseWheel, false );
    };

    // ========== JS主逻辑 ==========
    // 全局捕获js异常，页面显示错误
    window.onerror = function(msg) {
        document.getElementById('error-tip').innerText = '渲染错误: ' + msg;
    };

    // python注入的帧数据占位符
    const frames = __DATA_JSON__;
    // 获取总帧数量
    const totalFrames = frames.length;

    // 飞机整体尺寸系数
    const AIRCRAFT_SIZE = 180;
    // 飞机机体坐标轴长度
    const AIRCRAFT_AXIS_SIZE = 220;
    // 水平基准坐标轴长度
    const HORIZON_AXIS_SIZE  = 250;

    // ENU坐标转换Three.js坐标：x不变，z=up，y=-north
    function enu2three(x, y, z) {
        return new THREE.Vector3(x, z, -y);
    }

    // 获取画布DOM容器
    const container = document.getElementById('canvas-container');
    // 创建Three.js场景对象，所有物体都加入scene
    const scene = new THREE.Scene();
    // 设置场景背景颜色
    scene.background = new THREE.Color(0x0e1117);

    // 创建透视相机，视场角60度
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 1000000);
    // 创建WebGL渲染器，开启抗锯齿
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    // 设置渲染器输出画布尺寸
    renderer.setSize(container.clientWidth, container.clientHeight);
    // 将渲染画布挂载到页面DOM
    container.appendChild(renderer.domElement);

    // 实例化轨道控制器，绑定相机和画布
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // 创建地面网格辅助线
    const gridHelper = new THREE.GridHelper(10000, 50, 0x444444, 0x222222);
    scene.add(gridHelper);

    // 将所有帧ENU位置全部转为Three.js坐标，得到完整轨迹点数组
    const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
    // 完整轨迹（灰色，全部航线）
    const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
    const fullLine = new THREE.Line(fullLineGeo, new THREE.LineBasicMaterial({ color: 0x888888 }));
    scene.add(fullLine);

    // 已经飞过的轨迹（红色，动态增长）
    const flownLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([allPoints[0]]),
        new THREE.LineBasicMaterial({ color: 0xff4444 })
    );
    scene.add(flownLine);

           // ========== 飞机模型：零姿态下机头沿+X，机翼沿Z展开，机身水平 ==========
    // 飞机组对象，整个飞机+机体坐标轴都放在group，统一做位置旋转
    const aircraftGroup = new THREE.Group();
    // 缩放系数
    const s = AIRCRAFT_SIZE / 150;

    // 机身几何体，沿X轴方向，机头+X
    const bodyGeo = new THREE.BoxGeometry(100 * s, 6 * s, 6 * s);
    const bodyMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    aircraftGroup.add(body);

    // 机头红色标记方块
    const noseGeo = new THREE.BoxGeometry(20 * s, 8 * s, 8 * s);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
    const nose = new THREE.Mesh(noseGeo, noseMat);
    nose.position.x = 60 * s;
    aircraftGroup.add(nose);

    // 主机翼，沿Z轴左右伸展
    const wingGeo = new THREE.BoxGeometry(20 * s, 3 * s, 130 * s);
    const wingMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const wing = new THREE.Mesh(wingGeo, wingMat);
    wing.position.x = -10 * s;
    aircraftGroup.add(wing);

    // 水平尾翼
    const hTailGeo = new THREE.BoxGeometry(15 * s, 2 * s, 60 * s);
    const hTail = new THREE.Mesh(hTailGeo, wingMat);
    hTail.position.x = -50 * s;
    aircraftGroup.add(hTail);

    // 垂直尾翼（红色）
    const vTailGeo = new THREE.BoxGeometry(15 * s, 35 * s, 3 * s);
    const vTailMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
    const vTail = new THREE.Mesh(vTailGeo, vTailMat);
    vTail.position.set(-50 * s, 17.5 * s, 0);
    aircraftGroup.add(vTail);

    // 添加机体坐标系辅助坐标轴（RGB：X红 Y绿 Z蓝）
    aircraftGroup.add(new THREE.AxesHelper(AIRCRAFT_AXIS_SIZE));
    // 将飞机整体加入场景
    scene.add(aircraftGroup);

    // horizonGroup：水平基准坐标轴，只跟随位置、航向，永远保持水平，不做俯仰滚转
    const horizonGroup = new THREE.Group();
    const hAxis = new THREE.AxesHelper(HORIZON_AXIS_SIZE);
    hAxis.material.opacity = 0.45;
    hAxis.material.transparent = true;
    horizonGroup.add(hAxis);
    scene.add(horizonGroup);

    // 计算全部轨迹包围盒，用来初始化相机位置
    const box = new THREE.Box3().setFromPoints(allPoints);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1000;

    // 当前视角模式标记
    let currentView = 'free';
    // 是否开启跟随飞机视角标记
    let followAircraft = false;

    // 设置相机预设视角函数
    function setCameraView(viewName) {
        currentView = viewName;
        followAircraft = (viewName === 'follow');
        switch (viewName) {
            case 'top':
                // 俯视图
                camera.position.set(center.x, maxDim * 2.5, center.z);
                camera.lookAt(center);
                controls.target.copy(center);
                break;
            case 'side':
                // 侧视图
                camera.position.set(center.x + maxDim * 2, center.y, center.z);
                camera.lookAt(center);
                controls.target.copy(center);
                break;
            case 'front':
                // 前视图
                camera.position.set(center.x, center.y, center.z + maxDim * 2);
                camera.lookAt(center);
                controls.target.copy(center);
                break;
            case 'free':
            case 'follow':
                // 默认自由视角初始位置
                camera.position.set(center.x + maxDim * 1.5, center.y + maxDim * 1.2, center.z + maxDim * 1.5);
                controls.target.copy(center);
                break;
        }
        controls.update();
    }

    // 获取页面上各个DOM控件对象
    const playBtn = document.getElementById('playBtn');
    const speedSelect = document.getElementById('speedSelect');
    const frameSlider = document.getElementById('frameSlider');
    const viewSelect = document.getElementById('viewSelect');
    const frameText = document.getElementById('frameText');
    const hdgVal = document.getElementById('hdgVal');
    const pitVal = document.getElementById('pitVal');
    const rolVal = document.getElementById('rolVal');

    // 播放状态变量
    let currentFrame = 0; // 当前帧索引
    let isPlaying = false; // 是否正在播放标记
    let speed = 1; // 播放倍速
    let lastTime = null; // 记录上一帧时间戳
    const frameInterval = 100; // 原始帧间隔(ms)

    // 更新单帧姿态：每一帧都调用，更新飞机位置、旋转、轨迹、UI文本
function updateFrame() {
    // 获取当前帧数据
    const frame = frames[currentFrame];
    // 将ENU坐标转为Three坐标
    const pos = enu2three(frame.x, frame.y, frame.z);

    // 设置飞机组、水平基准组的空间位置
    aircraftGroup.position.copy(pos);
    horizonGroup.position.copy(pos);

    // 角度转弧度，航向做90度偏移修正航空与图形学基准差
    const h = THREE.MathUtils.degToRad(90 - frame.heading);
    const p = THREE.MathUtils.degToRad(frame.pitch);
    const r = THREE.MathUtils.degToRad(frame.roll);

    // 设置欧拉角旋转顺序YZX，对应航空：航向→俯仰→滚转内在旋转
    aircraftGroup.rotation.order = 'YZX';
    // 航向（绕Y轴）
    aircraftGroup.rotation.y = h;
    // 俯仰（绕Z轴）
    aircraftGroup.rotation.z = p;
    // 滚转（绕X轴）
    aircraftGroup.rotation.x = r;

    // 水平基准坐标轴：只同步航向，不俯仰滚转，永远保持水平
    horizonGroup.rotation.y = h;

    // 截取从0到当前帧的点，作为已经飞过的轨迹
    const flownPts = allPoints.slice(0, currentFrame + 1);
    // 销毁旧几何体显存，防止内存泄漏
    flownLine.geometry.dispose();
    // 生成新轨迹几何体
    flownLine.geometry = new THREE.BufferGeometry().setFromPoints(flownPts);

    // 如果开启跟随视角，控制器目标点跟随飞机位置
    if (followAircraft) {
        controls.target.copy(pos);
    }

    // 更新右侧姿态面板显示文本
    hdgVal.textContent = frame.heading.toFixed(1);
    pitVal.textContent = frame.pitch.toFixed(1);
    rolVal.textContent = frame.roll.toFixed(1);
    // 更新底部帧号文字
    frameText.textContent = `第 ${currentFrame + 1} / ${totalFrames} 帧`;
    // 更新滑块位置
    frameSlider.value = currentFrame;
}
    // 播放暂停按钮点击事件
    playBtn.addEventListener('click', function() {
        // 如果播放到末尾，重置回到第0帧
        if (currentFrame >= totalFrames - 1) {
            currentFrame = 0;
        }
        // 切换播放状态布尔值
        isPlaying = !isPlaying;
        lastTime = null;
        // 修改按钮文字
        playBtn.textContent = isPlaying ? '⏸️ 暂停' : '▶️ 播放';
    });

    // 倍速下拉框变更事件
    speedSelect.addEventListener('change', e => {
        speed = parseFloat(e.target.value);
        lastTime = null;
    });

    // 拖动进度条滑块事件
    frameSlider.addEventListener('input', e => {
        currentFrame = parseInt(e.target.value);
        isPlaying = false;
        playBtn.textContent = '▶️ 播放';
        updateFrame();
    });

    // 视角下拉框切换事件
    viewSelect.addEventListener('change', e => {
        setCameraView(e.target.value);
        updateFrame();
    });

    // requestAnimationFrame动画主循环，浏览器每刷新一次执行一次
    function animate(time) {
        requestAnimationFrame(animate);
        if (isPlaying) {
            if (lastTime === null) {
                lastTime = time;
            } else {
                // 计算两次循环时间差
                const delta = time - lastTime;
                // 根据倍速判断是否需要切换下一帧
                if (delta > frameInterval / speed) {
                    lastTime = time;
                    if (currentFrame < totalFrames - 1) {
                        currentFrame++;
                        updateFrame();
                    } else {
                        // 播放结束，停止播放
                        isPlaying = false;
                        playBtn.textContent = '▶️ 播放';
                    }
                }
            }
        }
        // 更新轨道控制器
        controls.update();
        // 渲染场景相机输出画面
        renderer.render(scene, camera);
    }

    // 浏览器窗口大小变化监听，自适应画布尺寸
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    // 程序初始化：设置默认自由视角，渲染第一帧，启动动画循环，清空错误提示
    setCameraView('free');
    updateFrame();
    animate(0);
    document.getElementById('error-tip').innerText = '';
    </script>
</body>
</html>
    """
    
    # 字符串替换，把占位符替换成真实数据
    html_template = html_template.replace("__DATA_JSON__", data_json)
    html_template = html_template.replace("__TOTAL__", str(total - 1))
    html_template = html_template.replace("__TOTAL_PLUS_ONE__", str(total))

    # streamlit内嵌html组件，高度750，关闭内部滚动条
    components.html(html_template, height=750, scrolling=False)

else:
    # 没有加载数据时页面提示文字
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")