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

# ===================== CSV解析 =====================
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

# ===================== 全量指标配置 =====================
METRIC_CONFIG = {
    "当前高度": {
        "key": "altitude",
        "paths": [
            {"fields": ["altitude"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": False}
        ],
        "unit": "m"
    },
    "当前航向": {
        "key": "heading",
        "paths": [
            {"fields": ["heading"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": False}
        ],
        "unit": "°"
    },
    "俯仰角": {
        "key": "pitch",
        "paths": [
            {"fields": ["pitch"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": False}
        ],
        "unit": "°"
    },
    "滚转角": {
        "key": "roll",
        "paths": [
            {"fields": ["roll"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": False}
        ],
        "unit": "°"
    },
    "地速(水平速度)": {
        "key": "ground_speed",
        "paths": [
            {"fields": ["ve", "vn"], "diff_order": 0, "source": "原始分量合成", "check_all_zero": True},
            {"fields": ["latitude", "longitude"], "diff_order": 1, "source": "经纬度一阶差分推算", "check_all_zero": False}
        ],
        "unit": "m/s"
    },
    "升降速度": {
        "key": "vertical_speed",
        "paths": [
            {"fields": ["vu"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": True},
            {"fields": ["altitude"], "diff_order": 1, "source": "高度一阶差分推算", "check_all_zero": False}
        ],
        "unit": "m/s"
    },
    "累计飞行航程": {
        "key": "distance",
        "paths": [
            {"fields": ["latitude", "longitude"], "diff_order": 1, "source": "经纬度积分推算", "check_all_zero": False}
        ],
        "unit": "m"
    },
    "当前经纬度": {
        "key": "lat_lon",
        "paths": [
            {"fields": ["latitude", "longitude"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": False}
        ],
        "unit": ""
    },
    "合地速 Vg": {
        "key": "vg_total",
        "paths": [
            {"fields": ["ve", "vn", "vu"], "diff_order": 0, "source": "原始分量合成", "check_all_zero": True},
            {"fields": ["latitude", "longitude", "altitude"], "diff_order": 1, "source": "位置一阶差分推算", "check_all_zero": False}
        ],
        "unit": "m/s"
    },
    "航迹角 γ": {
        "key": "flight_path_angle",
        "paths": [
            {"fields": ["ve", "vn", "vu"], "diff_order": 0, "source": "原始分量计算", "check_all_zero": True},
            {"fields": ["latitude", "longitude", "altitude"], "diff_order": 1, "source": "位置一阶差分推算", "check_all_zero": False}
        ],
        "unit": "°"
    },
    "地速分量面板": {
        "key": "velocity_components",
        "paths": [
            {"fields": ["ve", "vn", "vu"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": True},
            {"fields": ["latitude", "longitude", "altitude"], "diff_order": 1, "source": "位置一阶差分推算", "check_all_zero": False}
        ],
        "unit": "m/s"
    },
    "姿态角速度面板": {
        "key": "attitude_rates",
        "paths": [
            {"fields": ["vheading", "vpitch", "vroll"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": True},
            {"fields": ["heading", "pitch", "roll"], "diff_order": 1, "source": "姿态角一阶差分推算", "check_all_zero": False}
        ],
        "unit": "°/s"
    },
    "三轴加速度面板": {
        "key": "acceleration_components",
        "paths": [
            {"fields": ["ae", "an", "au"], "diff_order": 0, "source": "原始传感器输入", "check_all_zero": True},
            {"fields": ["ve", "vn", "vu"], "diff_order": 1, "source": "速度一阶差分推算", "check_all_zero": True},
            {"fields": ["latitude", "longitude", "altitude"], "diff_order": 2, "source": "位置二阶差分推算", "check_all_zero": False}
        ],
        "unit": "m/s²"
    },
    "飞行状态判断": {
        "key": "flight_status",
        "paths": [
            {"fields": ["vu", "vheading", "vroll"], "diff_order": 0, "source": "原始参数判断", "check_all_zero": True},
            {"fields": ["altitude", "heading", "roll"], "diff_order": 1, "source": "差分参数判断", "check_all_zero": False}
        ],
        "unit": ""
    }
}

# ===================== 指标可计算性校验 =====================
def validate_calculable_metrics(df: pd.DataFrame, config: dict, min_valid_ratio: float = 0.1) -> dict:
    result = {"calculable": {}, "incalculable": {}}
    total_rows = len(df)
    if total_rows == 0:
        for metric in config:
            result["incalculable"][metric] = ["数据表为空，无有效数据"]
        return result

    def check_path_valid(path):
        errors = []
        for field in path["fields"]:
            if field not in df.columns:
                errors.append(f"缺失字段【{field}】")
                continue
            col = df[field]
            if not pd.api.types.is_numeric_dtype(col):
                errors.append(f"字段【{field}】为非数值类型")
                continue
            valid_data = col.dropna()
            if len(valid_data) / total_rows < min_valid_ratio:
                errors.append(f"字段【{field}】有效数据占比不足{min_valid_ratio*100:.0f}%")
                continue
            if path.get("check_all_zero", False):
                non_zero = valid_data[valid_data != 0]
                if len(non_zero) == 0:
                    errors.append(f"字段【{field}】全部为0，无有效输入")
        return len(errors) == 0, errors

    for metric_name, cfg in config.items():
        all_errors = []
        matched_path = None

        for path in cfg["paths"]:
            is_valid, errs = check_path_valid(path)
            if is_valid:
                matched_path = path
                break
            else:
                all_errors.extend(errs)

        if matched_path:
            result["calculable"][metric_name] = {
                "unit": cfg["unit"],
                "key": cfg["key"],
                "diff_order": matched_path["diff_order"],
                "source": matched_path["source"]
            }
        else:
            unique_errors = list(dict.fromkeys(all_errors))
            result["incalculable"][metric_name] = unique_errors

    return result

# ===================== 全量衍生数据计算 =====================
def calc_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    R = 6371000

    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        dt = np.diff(df["timestamp"], prepend=df["timestamp"].iloc[0]) / 1000.0
        dt[dt <= 0] = 0.1
    elif "time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]):
        dt = np.diff(df["time"], prepend=df["time"].iloc[0])
        dt[dt <= 0] = 0.1
    else:
        dt = 0.1

    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])
    
    dlat = np.diff(lat_rad, prepend=lat_rad.iloc[0])
    dlon = np.diff(lon_rad, prepend=lon_rad.iloc[0])
    
    df["_ve_diff"] = R * np.cos(lat_rad) * dlon / dt
    df["_vn_diff"] = R * dlat / dt
    df["_vu_diff"] = np.diff(df["altitude"], prepend=df["altitude"].iloc[0]) / dt
    df.loc[0, ["_ve_diff", "_vn_diff", "_vu_diff"]] = 0

    has_raw_vel = all(col in df.columns for col in ['ve', 'vn', 'vu'])
    raw_vel_valid = False
    if has_raw_vel:
        raw_vel_valid = (df[['ve','vn','vu']].dropna() != 0).any(axis=1).sum() > 0

    if raw_vel_valid:
        df["_ve_use"] = df["ve"]
        df["_vn_use"] = df["vn"]
        df["_vu_use"] = df["vu"]
    else:
        df["_ve_use"] = df["_ve_diff"]
        df["_vn_use"] = df["_vn_diff"]
        df["_vu_use"] = df["_vu_diff"]

    df["_dist_step"] = R * 2 * np.arcsin(np.sqrt(
        np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(lat_rad.shift(1).fillna(lat_rad.iloc[0])) * np.sin(dlon/2)**2
    ))
    df["distance"] = np.cumsum(df["_dist_step"])
    df["ground_speed"] = df["_dist_step"] / dt
    df.loc[0, "ground_speed"] = 0
    df["vertical_speed"] = df["_vu_use"]
    df["vg_total"] = np.sqrt(df["_ve_use"]**2 + df["_vn_use"]**2 + df["_vu_use"]**2)
    v_horiz = np.sqrt(df["_ve_use"]**2 + df["_vn_use"]**2)
    df["flight_path_angle"] = np.degrees(np.arctan2(df["_vu_use"], v_horiz))
    df.loc[0, "flight_path_angle"] = 0

    has_raw_rate = all(col in df.columns for col in ['vheading', 'vpitch', 'vroll'])
    raw_rate_valid = False
    if has_raw_rate:
        raw_rate_valid = (df[['vheading','vpitch','vroll']].dropna() != 0).any(axis=1).sum() > 0

    if raw_rate_valid:
        df["_vheading_use"] = df["vheading"]
        df["_vpitch_use"] = df["vpitch"]
        df["_vroll_use"] = df["vroll"]
    else:
        heading_diff = np.diff(df["heading"], prepend=df["heading"].iloc[0])
        heading_diff = (heading_diff + 180) % 360 - 180
        df["_vheading_use"] = heading_diff / dt
        df["_vpitch_use"] = np.diff(df["pitch"], prepend=df["pitch"].iloc[0]) / dt
        df["_vroll_use"] = np.diff(df["roll"], prepend=df["roll"].iloc[0]) / dt
        df.loc[0, ["_vheading_use", "_vpitch_use", "_vroll_use"]] = 0

    has_raw_acc = all(col in df.columns for col in ['ae', 'an', 'au'])
    raw_acc_valid = False
    if has_raw_acc:
        raw_acc_valid = (df[['ae','an','au']].dropna() != 0).any(axis=1).sum() > 0

    if raw_acc_valid:
        df["_ae_use"] = df["ae"]
        df["_an_use"] = df["an"]
        df["_au_use"] = df["au"]
    else:
        df["_ae_use"] = np.diff(df["_ve_use"], prepend=df["_ve_use"].iloc[0]) / dt
        df["_an_use"] = np.diff(df["_vn_use"], prepend=df["_vn_use"].iloc[0]) / dt
        df["_au_use"] = np.diff(df["_vu_use"], prepend=df["_vu_use"].iloc[0]) / dt
        df.loc[0, ["_ae_use", "_an_use", "_au_use"]] = 0

    VERTICAL_THRESHOLD = 0.5
    RATE_THRESHOLD = 1.5
    status_list = []
    for idx, row in df.iterrows():
        vu_val = row["_vu_use"]
        vh_val = abs(row["_vheading_use"])
        vr_val = abs(row["_vroll_use"])
        
        states = []
        if abs(vu_val) < VERTICAL_THRESHOLD:
            states.append("平飞")
        elif vu_val > 0:
            states.append("上升")
        else:
            states.append("下降")
        if vh_val > RATE_THRESHOLD or vr_val > RATE_THRESHOLD:
            states.append("机动")
        status_list.append("·".join(states))
    df["flight_status"] = status_list
    
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
        metrics_check = validate_calculable_metrics(df, METRIC_CONFIG)
        df = calc_derived_metrics(df)
        
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        
        for idx, row in df.iterrows():
            e, n, u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            frame_item = {
                "x": float(e),
                "y": float(n),
                "z": float(u),
                "heading": float(row["heading"]),
                "pitch": float(row["pitch"]),
                "roll": float(row["roll"]),
                "ground_speed": float(row["ground_speed"]),
                "vertical_speed": float(row["vertical_speed"]),
                "distance": float(row["distance"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "ve": float(row["_ve_use"]),
                "vn": float(row["_vn_use"]),
                "vu": float(row["_vu_use"]),
                "vheading": float(row["_vheading_use"]),
                "vpitch": float(row["_vpitch_use"]),
                "vroll": float(row["_vroll_use"]),
                "ae": float(row["_ae_use"]),
                "an": float(row["_an_use"]),
                "au": float(row["_au_use"]),
                "vg_total": float(row["vg_total"]),
                "flight_path_angle": float(row["flight_path_angle"]),
                "flight_status": str(row["flight_status"])
            }
            frames_data.append(frame_item)
            
        st.success(f"数据加载成功，共 {len(frames_data)} 帧")
        
        st.subheader("📊 数据可用性校验")
        if metrics_check["calculable"]:
            st.write("✅ 可正常显示的数据：")
            for m, info in metrics_check["calculable"].items():
                st.text(f"  · {m}  [{info['source']}]")
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
        
        .bottom-instrument-bar {
            height: 300px;
            flex-shrink: 0;
            background: #14161b;
            border-top: 1px solid #333;
            display: flex;
            padding: 10px 20px;
            gap: 20px;
        }
        
        .attitude-wrap {
            width: 260px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .attitude-wrap .title {
            font-size: 18px;
            color: #ccc;
            margin-bottom: 8px;
            align-self: flex-start;
        }
        .attitude-wrap canvas {
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: #000;
            border: 6px solid #333;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
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
            padding: 6px 10px;
            background: #1a1c23;
            border-radius: 4px;
        }
        .data-item .label-wrap {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .data-item .label { color: #aaa; font-size: 16px; }
        .data-item .source { color: #666; font-size: 11px; }
        .data-item .value { color: #00ff00; font-weight: bold; font-size: 16px; }
        .data-item .value.high-order { color: #ff4444; }
        .data-item.disabled { opacity: 0.5; }
        .data-item.disabled .value { color: #888; font-size: 11px; font-weight: normal; }
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
            <canvas id="attitudeCanvas" width="480" height="480"></canvas>
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
                document.getElementById('error-tip').innerText = '3D引擎加载失败，请检查网络';
                return;
            }

            // ========== 轨道控制器 ==========
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

            // ========== 姿态地平仪绘制（完全参照正确版本） ==========
            const attitudeCanvas = document.getElementById('attitudeCanvas');
            const actx = attitudeCanvas.getContext('2d');
            const acx = attitudeCanvas.width / 2;
            const acy = attitudeCanvas.height / 2;
            const aradius = acx - 10;

            function drawAttitudeIndicator(pitchDeg, rollDeg, headingDeg) {
                actx.clearRect(0, 0, attitudeCanvas.width, attitudeCanvas.height);

                // 圆形裁剪
                actx.save();
                actx.beginPath();
                actx.arc(acx, acy, aradius, 0, Math.PI * 2);
                actx.clip();

                // 滚转旋转
                actx.save();
                actx.translate(acx, acy);
                actx.rotate(rollDeg * Math.PI / 180);

                // 俯仰偏移：每度对应6像素
                const pitchPx = pitchDeg * 6;

                // 天空
                actx.fillStyle = '#1e90ff';
                actx.fillRect(-aradius*2, -aradius*2, aradius*4, aradius*2 + pitchPx);

                // 地面
                actx.fillStyle = '#8b4513';
                actx.fillRect(-aradius*2, pitchPx, aradius*4, aradius*4 - pitchPx);

                // 地平线
                actx.strokeStyle = '#ffffff';
                actx.lineWidth = 3;
                actx.beginPath();
                actx.moveTo(-aradius*2, pitchPx);
                actx.lineTo(aradius*2, pitchPx);
                actx.stroke();

                // 俯仰刻度线 + 数值
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

                // 固定飞机基准符号（不随姿态旋转，黄色）
                actx.save();
                actx.translate(acx, acy);
                actx.strokeStyle = '#ffff00';
                actx.fillStyle = '#ffff00';
                actx.lineWidth = 4;
                // 左右机翼
                actx.beginPath();
                actx.moveTo(-90, 0);
                actx.lineTo(-20, 0);
                actx.moveTo(20, 0);
                actx.lineTo(90, 0);
                actx.stroke();
                // 中心圆点
                actx.beginPath();
                actx.arc(0, 0, 7, 0, Math.PI * 2);
                actx.fill();
                // 顶部航向三角标记
                actx.beginPath();
                actx.moveTo(0, -aradius + 18);
                actx.lineTo(-14, -aradius + 40);
                actx.lineTo(14, -aradius + 40);
                actx.closePath();
                actx.fill();
                actx.restore();

                // 底部航向数值
                actx.fillStyle = '#00ff00';
                actx.font = 'bold 22px Arial';
                actx.textAlign = 'center';
                actx.fillText(`HDG ${headingDeg.toFixed(0)}°`, acx, acy + aradius - 25);
            }

            // ========== 主渲染逻辑 ==========
            const frames = __DATA_JSON__;
            const metricsCheck = __METRICS_JSON__;
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

            // 完整轨迹线
            const allPoints = frames.map(f => enu2three(f.x, f.y, f.z));
            const fullLineGeo = new THREE.BufferGeometry().setFromPoints(allPoints);
            const fullLineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
            const fullLine = new THREE.Line(fullLineGeo, fullLineMat);
            scene.add(fullLine);

            // 已飞轨迹线
            const flownLineGeo = new THREE.BufferGeometry().setFromPoints([allPoints[0]]);
            const flownLineMat = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 3 });
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

            // ========== 数据面板初始化 ==========
            const calculableCol = document.getElementById('calculableCol');
            const incalculableCol = document.getElementById('incalculableCol');

            function buildDataPanels() {
                // 可计算指标
                const calcItems = metricsCheck.calculable;
                for (const name in calcItems) {
                    const info = calcItems[name];
                    const div = document.createElement('div');
                    div.className = 'data-item';
                    div.id = 'metric-' + info.key;

                    const labelWrap = document.createElement('div');
                    labelWrap.className = 'label-wrap';
                    const label = document.createElement('div');
                    label.className = 'label';
                    label.textContent = name;
                    const source = document.createElement('div');
                    source.className = 'source';
                    source.textContent = info.source;
                    labelWrap.appendChild(label);
                    labelWrap.appendChild(source);

                    const value = document.createElement('div');
                    value.className = 'value' + (info.diff_order > 0 ? ' high-order' : '');
                    value.dataset.key = info.key;
                    value.dataset.unit = info.unit;
                    value.textContent = '--';

                    div.appendChild(labelWrap);
                    div.appendChild(value);
                    calculableCol.appendChild(div);
                }

                // 不可计算指标
                const incalcItems = metricsCheck.incalculable;
                for (const name in incalcItems) {
                    const reasons = incalcItems[name];
                    const div = document.createElement('div');
                    div.className = 'data-item disabled';

                    const labelWrap = document.createElement('div');
                    labelWrap.className = 'label-wrap';
                    const label = document.createElement('div');
                    label.className = 'label';
                    label.textContent = name;
                    labelWrap.appendChild(label);

                    const value = document.createElement('div');
                    value.className = 'value';
                    value.textContent = reasons.join('；');
                    value.style.fontSize = '11px';
                    value.style.textAlign = 'right';

                    div.appendChild(labelWrap);
                    div.appendChild(value);
                    incalculableCol.appendChild(div);
                }
            }
            buildDataPanels();

            // 控件绑定
            const playBtn = document.getElementById('playBtn');
            const speedSelect = document.getElementById('speedSelect');
            const frameSlider = document.getElementById('frameSlider');
            const viewSelect = document.getElementById('viewSelect');
            const frameText = document.getElementById('frameText');

            // 坐标系开关
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

                // 原始姿态计算
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

                // 更新文本数值
                frameText.textContent = `第 ${currentFrame + 1} / ${totalFrames} 帧`;
                frameSlider.value = currentFrame;

                // 更新数据面板数值
                const valueEls = calculableCol.querySelectorAll('.value');
                valueEls.forEach(el => {
                    const key = el.dataset.key;
                    const unit = el.dataset.unit;
                    let val = frame[key];

                    if (key === 'lat_lon') {
                        val = `${frame.lat.toFixed(4)}°N\n${frame.lon.toFixed(4)}°E`;
                        el.classList.add('multi-line');
                    } else if (key === 'velocity_components') {
                        val = `东向 ${frame.ve.toFixed(2)}\n北向 ${frame.vn.toFixed(2)}\n垂向 ${frame.vu.toFixed(2)}`;
                        el.classList.add('multi-line');
                    } else if (key === 'attitude_rates') {
                        val = `航向率 ${frame.vheading.toFixed(2)}\n俯仰率 ${frame.vpitch.toFixed(2)}\n滚转率 ${frame.vroll.toFixed(2)}`;
                        el.classList.add('multi-line');
                    } else if (key === 'acceleration_components') {
                        val = `东向 ${frame.ae.toFixed(2)}\n北向 ${frame.an.toFixed(2)}\n垂向 ${frame.au.toFixed(2)}`;
                        el.classList.add('multi-line');
                    } else if (key === 'flight_status') {
                        val = frame.flight_status;
                    } else {
                        val = Number(val).toFixed(2) + unit;
                    }
                    el.textContent = val;
                });

                // 更新姿态仪表
                drawAttitudeIndicator(frame.pitch, frame.roll, frame.heading);
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
    html_template = html_template.replace("__METRICS_JSON__", metrics_json)
    html_template = html_template.replace("__TOTAL__", str(total - 1))
    html_template = html_template.replace("__TOTAL_PLUS_ONE__", str(total))

    components.html(html_template, height=1080, scrolling=False)

else:
    st.info("👈 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")