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

# ===================== CSV解析（完全未改动） =====================
def parse_csv_data(csv_string):
    lines = csv_string.strip().splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    fixed_csv = '\n'.join(cleaned_lines)

    df = pd.read_csv(StringIO(fixed_csv))
    df.columns = df.columns.str.strip()

    required_cols = ["latitude", "longitude", "altitude", "heading", "pitch", "roll"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必填列：{missing_cols}，当前列名：{list(df.columns)}")
    
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 自动检测并修复列错位
    first_lat = df["latitude"].dropna().iloc[0]
    if abs(first_lat) > 90:
        new_cols = list(df.columns[1:]) + ['_drop_col']
        df.columns = new_cols
        df = df.drop(columns=['_drop_col'])
        df.columns = df.columns.str.strip()
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("有效数据行为0")
    return df

# ===================== 【扩展】飞行指标校验配置（原有8项+新增5项） =====================
METRIC_CONFIG = {
    # ---------- 原有指标 ----------
    "当前高度": {
        "key": "altitude",
        "required_fields": ["altitude"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m"
    },
    "当前航向": {
        "key": "heading",
        "required_fields": ["heading"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "°"
    },
    "俯仰角": {
        "key": "pitch",
        "required_fields": ["pitch"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "°"
    },
    "滚转角": {
        "key": "roll",
        "required_fields": ["roll"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "°"
    },
    "地速(水平速度)": {
        "key": "ground_speed",
        "required_fields": ["latitude", "longitude"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m/s"
    },
    "升降速度": {
        "key": "vertical_speed",
        "required_fields": ["altitude"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m/s"
    },
    "累计飞行航程": {
        "key": "distance",
        "required_fields": ["latitude", "longitude"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m"
    },
    "当前经纬度": {
        "key": "lat_lon",
        "required_fields": ["latitude", "longitude"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": ""
    },
    # ---------- 【新增5项指标】 ----------
    "合地速 Vg": {
        "key": "vg_total",
        "required_fields": ["ve", "vn", "vu"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m/s"
    },
    "航迹角 γ": {
        "key": "flight_path_angle",
        "required_fields": ["ve", "vn", "vu"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "°"
    },
    "地速分量面板": {
        "key": "velocity_components",
        "required_fields": ["ve", "vn", "vu"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "m/s"
    },
    "姿态角速度面板": {
        "key": "attitude_rates",
        "required_fields": ["vheading", "vpitch", "vroll"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": "°/s"
    },
    "飞行状态判断": {
        "key": "flight_status",
        "required_fields": ["vu", "vheading", "vroll"],
        "rules": {"allow_null": False, "allow_zero": True, "required_type": "number"},
        "unit": ""
    }
}

# ===================== 指标可计算性校验函数（完全未改动） =====================
def validate_calculable_metrics(df: pd.DataFrame, config: dict, min_valid_ratio: float = 0.1) -> dict:
    result = {"calculable": {}, "incalculable": {}}
    total_rows = len(df)
    if total_rows == 0:
        for metric in config:
            result["incalculable"][metric] = ["数据表为空，无有效数据"]
        return result

    for metric_name, cfg in config.items():
        required = cfg["required_fields"]
        rules = cfg["rules"]
        errors = []

        for field in required:
            if field not in df.columns:
                errors.append(f"缺失字段【{field}】")
                continue
            col = df[field]
            if rules.get("required_type") == "number" and not pd.api.types.is_numeric_dtype(col):
                errors.append(f"字段【{field}】为非数值类型")
                continue
            valid_count = col.dropna().shape[0]
            if valid_count / total_rows < min_valid_ratio:
                errors.append(f"字段【{field}】有效数据占比不足{min_valid_ratio*100:.0f}%")
                continue
            if not rules.get("allow_zero", True):
                non_zero = col.dropna()[col.dropna() != 0].shape[0]
                if non_zero / total_rows < min_valid_ratio:
                    errors.append(f"字段【{field}】非零有效数据不足")
        if errors:
            result["incalculable"][metric_name] = errors
        else:
            result["calculable"][metric_name] = {
                "unit": cfg["unit"],
                "key": cfg["key"]
            }
    return result

# ===================== 【扩展】预计算衍生运动数据（原有逻辑保留+新增5项计算） =====================
def calc_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    R = 6371000
    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])
    
    # ---------- 原有计算逻辑 完全保留 ----------
    dlat = np.diff(lat_rad, prepend=lat_rad.iloc[0])
    dlon = np.diff(lon_rad, prepend=lon_rad.iloc[0])
    a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(lat_rad.shift(1).fillna(lat_rad.iloc[0])) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    df["_dist_step"] = R * c
    df["distance"] = np.cumsum(df["_dist_step"])
    
    if "time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]):
        dt = np.diff(df["time"], prepend=df["time"].iloc[0])
        dt[dt == 0] = 0.1
    else:
        dt = 0.1
    
    df["ground_speed"] = df["_dist_step"] / dt
    df.loc[0, "ground_speed"] = 0
    df["vertical_speed"] = np.diff(df["altitude"], prepend=df["altitude"].iloc[0]) / dt
    df.loc[0, "vertical_speed"] = 0

    # ---------- 【新增】地速相关衍生计算 ----------
    has_vel = all(col in df.columns for col in ['ve', 'vn', 'vu'])
    if has_vel:
        # 合地速 Vg = 三轴速度矢量模
        df["vg_total"] = np.sqrt(df["ve"]**2 + df["vn"]**2 + df["vu"]**2)
        # 水平面速度
        v_horizontal = np.sqrt(df["ve"]**2 + df["vn"]**2)
        # 航迹角 γ = arctan(垂直速度/水平速度)，转角度
        df["flight_path_angle"] = np.degrees(np.arctan2(df["vu"], v_horizontal))
    else:
        df["vg_total"] = 0
        df["flight_path_angle"] = 0

    # ---------- 【新增】飞行状态判断 ----------
    has_status_fields = all(col in df.columns for col in ['vu', 'vheading', 'vroll'])
    if has_status_fields:
        # 阈值定义：可根据实际数据调整
        VERTICAL_THRESHOLD = 0.5    # 垂直速度阈值 m/s
        RATE_THRESHOLD = 1.5        # 姿态角速度阈值 °/s
        
        status_list = []
        for _, row in df.iterrows():
            vu = row['vu']
            vh = row['vheading']
            vr = row['vroll']
            
            states = []
            # 垂直状态
            if abs(vu) < VERTICAL_THRESHOLD:
                states.append("平飞")
            elif vu > 0:
                states.append("上升")
            else:
                states.append("下降")
            # 转弯状态
            if abs(vh) > RATE_THRESHOLD or abs(vr) > RATE_THRESHOLD:
                states.append("机动")
            # 组合状态
            status_list.append("·".join(states))
        df["flight_status"] = status_list
    else:
        df["flight_status"] = "--"
    
    return df

# ===================== 侧边栏数据输入 =====================
with st.sidebar:
    st.header("数据输入")
    input_mode = st.radio("选择数据输入方式", ["粘贴CSV文本", "上传CSV文件"])
    csv_text = st.text_area("粘贴CSV数据", height=280)
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    load_btn = st.button("加载数据", use_container_width=True)

frames_data = []
metrics_check = {"calculable": {}, "incalculable": {}}

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
        
        # 执行指标校验（自动覆盖新增5项）
        metrics_check = validate_calculable_metrics(df, METRIC_CONFIG)
        
        # 预计算衍生数据
        df = calc_derived_metrics(df)
        
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        for idx, row in df.iterrows():
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            frame_item = {
                # 原有核心字段
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "ground_speed": float(row.get("ground_speed", 0)),
                "vertical_speed": float(row.get("vertical_speed", 0)),
                "distance": float(row.get("distance", 0)),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                # 【新增】原始速度分量字段
                "ve": float(row.get("ve", 0)),
                "vn": float(row.get("vn", 0)),
                "vu": float(row.get("vu", 0)),
                # 【新增】原始姿态角速度字段
                "vheading": float(row.get("vheading", 0)),
                "vpitch": float(row.get("vpitch", 0)),
                "vroll": float(row.get("vroll", 0)),
                # 【新增】计算衍生字段
                "vg_total": float(row.get("vg_total", 0)),
                "flight_path_angle": float(row.get("flight_path_angle", 0)),
                "flight_status": str(row.get("flight_status", "--"))
            }
            frames_data.append(frame_item)
            
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
        
        # 侧边栏展示校验结果
        st.subheader("📊 数据可用性校验")
        if metrics_check["calculable"]:
            st.write("✅ 可正常显示的数据：")
            for m in metrics_check["calculable"]:
                st.text(f"  · {m}")
        if metrics_check["incalculable"]:
            st.write("❌ 无法显示的数据：")
            for m, reasons in metrics_check["incalculable"].items():
                st.text(f"  · {m}")
                for r in reasons:
                    st.text(f"      → {r}")
        
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ===================== 3D渲染 + 底部仪表条 =====================
if len(frames_data) > 0:
    data_json = json.dumps(frames_data, ensure_ascii=False)
    metrics_json = json.dumps(metrics_check, ensure_ascii=False)
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
            gap:20px;
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
        
        #canvas-container { 
            flex:1;
            width:100%; 
            min-height:0;
            position:relative;
        }
        
        /* 底部横向仪表条 */
        .bottom-instrument-bar {
            height: 260px;
            flex-shrink: 0;
            background: #14161b;
            border-top: 1px solid #333;
            display: flex;
            padding: 10px 20px;
            gap: 20px;
        }
        
        .attitude-wrap {
            width: 240px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .attitude-wrap .title {
            font-size: 13px;
            color: #aaa;
            margin-bottom: 6px;
            align-self: flex-start;
        }
        .attitude-wrap canvas {
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: #000;
            border: 2px solid #444;
        }
        
        .data-panel {
            flex: 1;
            display: flex;
            gap: 20px;
            overflow-y: auto;
        }
        .data-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .data-col h4 {
            font-size: 14px;
            color: #ccc;
            border-bottom: 1px solid #333;
            padding-bottom: 4px;
            margin-bottom: 4px;
        }
        .data-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            padding: 4px 8px;
            background: #1a1c23;
            border-radius: 4px;
        }
        .data-item .label { color: #aaa; }
        .data-item .value { color: #00ff00; font-weight: bold; }
        .data-item.disabled { opacity: 0.5; }
        .data-item.disabled .value { color: #888; font-size: 11px; font-weight: normal; }
        /* 新增：多行数值面板样式 */
        .data-item .value.multi-line {
            text-align: right;
            line-height: 1.6;
            font-size: 12px;
        }

        #error-tip {
            position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
            color:#ff4444; font-size:16px; z-index:99;
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
        <label><input type="checkbox" id="showAircraftAxis" checked> 机体坐标系</label>
        <label><input type="checkbox" id="showHorizonAxis" checked> 水平参考系</label>
    </div>

    <div id="canvas-container"></div>

    <div class="bottom-instrument-bar">
        <div class="attitude-wrap">
            <div class="title">姿态指示器 ADI</div>
            <canvas id="attitudeCanvas" width="440" height="440"></canvas>
        </div>
        <div class="data-panel">
            <div class="data-col" id="calculableCol">
                <h4>✅ 可同步显示数据</h4>
            </div>
            <div class="data-col" id="incalculableCol">
                <h4>❌ 无法显示数据</h4>
            </div>
        </div>
    </div>

    <script>
    window.onload = function() {
        try {
            if (typeof THREE === 'undefined') {
                document.getElementById('error-tip').innerText = '渲染引擎加载失败，请检查网络';
                return;
            }

            // ========== 轨道控制器（完全未改动） ==========
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

            // ========== 姿态地平仪绘制（完全未改动） ==========
            const attitudeCanvas = document.getElementById('attitudeCanvas');
            const actx = attitudeCanvas.getContext('2d');
            const acx = attitudeCanvas.width / 2;
            const acy = attitudeCanvas.height / 2;
            const aradius = acx - 10;

            function drawAttitudeIndicator(pitchDeg, rollDeg, headingDeg) {
                actx.clearRect(0, 0, attitudeCanvas.width, attitudeCanvas.height);
                actx.save();
                actx.beginPath();
                actx.arc(acx, acy, aradius, 0, Math.PI * 2);
                actx.clip();

                actx.save();
                actx.translate(acx, acy);
                actx.rotate(rollDeg * Math.PI / 180);

                const pitchPx = pitchDeg * 6;
                actx.fillStyle = '#1e90ff';
                actx.fillRect(-aradius*2, -aradius*2, aradius*4, aradius*2 + pitchPx);
                actx.fillStyle = '#8b4513';
                actx.fillRect(-aradius*2, pitchPx, aradius*4, aradius*4 - pitchPx);

                actx.strokeStyle = '#ffffff';
                actx.lineWidth = 3;
                actx.beginPath();
                actx.moveTo(-aradius*2, pitchPx);
                actx.lineTo(aradius*2, pitchPx);
                actx.stroke();

                actx.strokeStyle = '#ffffff';
                actx.lineWidth = 2;
                actx.font = 'bold 22px Arial';
                actx.textAlign = 'center';
                actx.fillStyle = '#ffffff';
                for (let i = -30; i <= 30; i += 5) {
                    const y = pitchPx - i * 6;
                    if (y < -aradius + 20 || y > aradius - 20) continue;
                    const len = i % 10 === 0 ? 50 : 25;
                    actx.beginPath();
                    actx.moveTo(-len, y);
                    actx.lineTo(len, y);
                    actx.stroke();
                    if (i % 10 === 0 && i !== 0) {
                        actx.fillText(Math.abs(i).toString(), -70, y + 7);
                        actx.fillText(Math.abs(i).toString(), 70, y + 7);
                    }
                }
                actx.restore();
                actx.restore();

                actx.save();
                actx.translate(acx, acy);
                actx.strokeStyle = '#ffff00';
                actx.fillStyle = '#ffff00';
                actx.lineWidth = 4;
                actx.beginPath();
                actx.moveTo(-90, 0);
                actx.lineTo(-20, 0);
                actx.moveTo(20, 0);
                actx.lineTo(90, 0);
                actx.stroke();
                actx.beginPath();
                actx.arc(0, 0, 7, 0, Math.PI * 2);
                actx.fill();
                actx.beginPath();
                actx.moveTo(0, -aradius + 18);
                actx.lineTo(-14, -aradius + 40);
                actx.lineTo(14, -aradius + 40);
                actx.closePath();
                actx.fill();
                actx.restore();

                actx.fillStyle = '#00ff00';
                actx.font = 'bold 20px Arial';
                actx.textAlign = 'center';
                actx.fillText(`HDG ${headingDeg.toFixed(0)}°`, acx, acy + aradius - 20);
            }

            // ========== 数据面板动态渲染（逻辑未变，自动兼容新增项） ==========
            const metricsConfig = __METRICS_JSON__;
            const calculableCol = document.getElementById('calculableCol');
            const incalculableCol = document.getElementById('incalculableCol');
            const valueElements = {};

            function initDataPanel() {
                for (const name in metricsConfig.calculable) {
                    const cfg = metricsConfig.calculable[name];
                    const key = cfg.key;
                    const item = document.createElement('div');
                    item.className = 'data-item';
                    item.innerHTML = `
                        <span class="label">${name}</span>
                        <span class="value" id="val_${key}">--</span>
                    `;
                    calculableCol.appendChild(item);
                    valueElements[key] = document.getElementById(`val_${key}`);
                }
                if (Object.keys(metricsConfig.calculable).length === 0) {
                    calculableCol.innerHTML += '<div style="color:#666; font-size:12px;">暂无可显示数据</div>';
                }

                for (const name in metricsConfig.incalculable) {
                    const reasons = metricsConfig.incalculable[name];
                    const item = document.createElement('div');
                    item.className = 'data-item disabled';
                    item.innerHTML = `
                        <span class="label">${name}</span>
                        <span class="value">${reasons[0]}</span>
                    `;
                    incalculableCol.appendChild(item);
                }
                if (Object.keys(metricsConfig.incalculable).length === 0) {
                    incalculableCol.innerHTML += '<div style="color:#666; font-size:12px;">全部数据可正常显示</div>';
                }
            }

            // ========== 【扩展】数据更新（原有8项+新增5项） ==========
            function updateDataValues(frame) {
                // ---------- 原有数据项 ----------
                if (valueElements.altitude) valueElements.altitude.textContent = frame.z.toFixed(1) + ' m';
                if (valueElements.heading) valueElements.heading.textContent = frame.heading.toFixed(1) + ' °';
                if (valueElements.pitch) valueElements.pitch.textContent = frame.pitch.toFixed(1) + ' °';
                if (valueElements.roll) valueElements.roll.textContent = frame.roll.toFixed(1) + ' °';
                if (valueElements.ground_speed) valueElements.ground_speed.textContent = frame.ground_speed.toFixed(2) + ' m/s';
                if (valueElements.vertical_speed) valueElements.vertical_speed.textContent = frame.vertical_speed.toFixed(2) + ' m/s';
                if (valueElements.distance) valueElements.distance.textContent = frame.distance.toFixed(1) + ' m';
                if (valueElements.lat_lon) valueElements.lat_lon.textContent = frame.lat.toFixed(6) + ', ' + frame.lon.toFixed(6);

                // ---------- 新增数据项 ----------
                if (valueElements.vg_total) valueElements.vg_total.textContent = frame.vg_total.toFixed(2) + ' m/s';
                if (valueElements.flight_path_angle) valueElements.flight_path_angle.textContent = frame.flight_path_angle.toFixed(2) + ' °';
                
                // 地速分量面板：三行显示
                if (valueElements.velocity_components) {
                    valueElements.velocity_components.innerHTML = 
                        `E: ${frame.ve.toFixed(2)}<br>N: ${frame.vn.toFixed(2)}<br>U: ${frame.vu.toFixed(2)}`;
                    valueElements.velocity_components.classList.add('multi-line');
                }

                // 姿态角速度面板：三行显示
                if (valueElements.attitude_rates) {
                    valueElements.attitude_rates.innerHTML = 
                        `航向: ${frame.vheading.toFixed(2)}<br>俯仰: ${frame.vpitch.toFixed(2)}<br>滚转: ${frame.vroll.toFixed(2)}`;
                    valueElements.attitude_rates.classList.add('multi-line');
                }

                // 飞行状态判断
                if (valueElements.flight_status) {
                    valueElements.flight_status.textContent = frame.flight_status;
                }
            }

            // ========== 主渲染逻辑（完全未改动） ==========
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

            const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
            const fullLine = new THREE.Line(fullLineGeo, fullLineMat);
            scene.add(fullLine);

            const flownLineGeo = new THREE.BufferGeometry().setFromPoints([allPoints[0]]);
            const flownLineMat = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 3 });
            const flownLine = new THREE.Line(flownLineGeo, flownLineMat);
            scene.add(flownLine);

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

            const aircraftAxis = new THREE.AxesHelper(AIRCRAFT_AXIS_SIZE);
            aircraftGroup.add(aircraftAxis);
            scene.add(aircraftGroup);

            const horizonGroup = new THREE.Group();
            const hAxis = new THREE.AxesHelper(HORIZON_AXIS_SIZE);
            hAxis.material.opacity = 0.45;
            hAxis.material.transparent = true;
            horizonGroup.add(hAxis);
            scene.add(horizonGroup);

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

            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const viewSelect = document.getElementById('viewSelect');
            const frameText = document.getElementById('frameText');

            const showAircraftAxis = document.getElementById('showAircraftAxis');
            const showHorizonAxis = document.getElementById('showHorizonAxis');
            showAircraftAxis.addEventListener('change', e => {
                aircraftAxis.visible = e.target.checked;
            });
            showHorizonAxis.addEventListener('change', e => {
                horizonGroup.visible = e.target.checked;
            });

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

                frameText.textContent = `第 ${currentFrame + 1} / ${totalFrames} 帧`;
                frameSlider.value = currentFrame;

                drawAttitudeIndicator(frame.pitch, frame.roll, frame.heading);
                updateDataValues(frame);
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

            initDataPanel();
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
    html_template = html_template.replace("__METRICS_JSON__", metrics_json)
    html_template = html_template.replace("__TOTAL__", str(total - 1))
    html_template = html_template.replace("__TOTAL_PLUS_ONE__", str(total))

    components.html(html_template, height=950, scrolling=False)

else:
    st.info("👈 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")