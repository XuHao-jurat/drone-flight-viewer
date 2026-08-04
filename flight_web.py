import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import time

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="无人机飞行轨迹与姿态可视化工具", layout="wide")
st.title("✈️ 无人机飞行轨迹与姿态可视化工具")

# 初始化会话状态，防止点击按钮丢失数据
if "df" not in st.session_state:
    st.session_state.df = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# ===================== 侧边栏 数据输入 =====================
with st.sidebar:
    st.header("数据输入")
    input_mode = st.radio("选择数据输入方式", ["粘贴CSV文本", "上传CSV文件"])
    csv_text = st.text_area("粘贴CSV数据", height=280)
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
    load_btn = st.button("加载数据")

# ===================== 姿态旋转矩阵函数（固定翼机体坐标系） =====================
def euler_rotation_matrix(heading_deg, pitch_deg, roll_deg):
    # heading:航向(正北0°顺时针) pitch俯仰 roll滚转 单位角度
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
    R = Rh @ Rp @ Rr
    return R

# 局部坐标：固定翼飞机模型（机头X向前，右翼Y，向上Z）
def get_aircraft_body_points(scale=80):
    pts = np.array([
        [1.0, 0.0, 0.0],   #机头
        [-0.6, 0.0, 0.0],  #机尾
        [-0.1, 0.7, 0.0],  #右翼尖
        [-0.1, -0.7, 0.0], #左翼尖
        [-0.4, 0.0, 0.25], #垂尾上
    ]) * scale
    return pts

# 经纬度转局部东北坐标系（以首点为原点）
def ll2local_enu(lat0, lon0, lat, lon, alt):
    R = 6371000
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)
    east = R * dlon * np.cos(lat0_rad)
    north = R * dlat
    up = alt
    return east, north, up

# ===================== 加载数据逻辑 =====================
if load_btn:
    try:
        if input_mode == "粘贴CSV文本":
            df = pd.read_csv(StringIO(csv_text))
        else:
            df = pd.read_csv(uploaded_file)
        # 坐标系转换：原点为第一条轨迹点
        lat0 = df["latitude"].iloc[0]
        lon0 = df["longitude"].iloc[0]
        enu_list = []
        for _, row in df.iterrows():
            e,n,u = ll2local_enu(lat0, lon0, row["latitude"], row["longitude"], row["altitude"])
            enu_list.append([e,n,u])
        enu_arr = np.array(enu_list)
        df["x_east"] = enu_arr[:,0]
        df["y_north"] = enu_arr[:,1]
        df["z_up"] = enu_arr[:,2]
        st.session_state.df = df
        st.session_state.current_frame = 0
        st.session_state.is_playing = False
        st.success(f"数据加载成功，共 {len(df)} 帧")
    except Exception as e:
        st.error(f"数据解析失败：{str(e)}")

# ===================== 绘图渲染函数 =====================
def build_3d_figure(df, frame_idx):
    fig = go.Figure()
    #完整航线
    fig.add_trace(go.Scatter3d(
        x=df["x_east"], y=df["y_north"], z=df["z_up"],
        mode="lines", line=dict(color="#aaaaaa", width=1.5), name="完整航线"
    ))
    #已飞行轨迹
    curr_df = df.iloc[:frame_idx+1]
    fig.add_trace(go.Scatter3d(
        x=curr_df["x_east"], y=curr_df["y_north"], z=curr_df["z_up"],
        mode="lines", line=dict(color="#ff4444", width=3), name="已飞航迹"
    ))

    #当前位置姿态
    row = df.iloc[frame_idx]
    cx, cy, cz = row["x_east"], row["y_north"], row["z_up"]
    head = row["heading"]
    pit = row["pitch"]
    rol = row["roll"]
    R = euler_rotation_matrix(head, pit, rol)
    body_pts = get_aircraft_body_points()
    body_rot = (R @ body_pts.T).T
    body_rot[:,0] += cx
    body_rot[:,1] += cy
    body_rot[:,2] += cz

    #绘制机体连线
    connect_idx = [[0,1],[1,2],[1,3],[1,4]]
    for seg in connect_idx:
        p1 = body_rot[seg[0]]
        p2 = body_rot[seg[1]]
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines", line=dict(width=4, color="crimson"), showlegend=False
        ))
    #重心标记
    fig.add_trace(go.Scatter3d(x=[cx],y=[cy],z=[cz],mode="markers",marker=dict(size=6,color="orange"),name="重心"))

    fig.update_layout(
        width=900, height=700,
        scene=dict(aspectmode="data",xaxis_title="East(m)",yaxis_title="North(m)",zaxis_title="Up(m)"),
        legend=dict(orientation="h")
    )
    return fig

# ===================== 主界面渲染 =====================
if st.session_state.df is not None:
    df = st.session_state.df
    total_frame = len(df)-1
    curr_frame = st.session_state.current_frame

    #顶部控件
    col1,col2,col3 = st.columns([1,4,1])
    play_btn = col1.button("▶️ 自动播放")
    pause_btn = col1.button("⏸️ 暂停")
    new_frame = col2.slider("选择帧", min_value=0, max_value=total_frame, value=curr_frame)
    col3.info(f"第 {curr_frame+1} / {total_frame+1} 帧")

    #滑动条手动控制帧
    if new_frame != curr_frame:
        st.session_state.current_frame = new_frame
        st.rerun()

    #播放暂停控制
    if play_btn:
        st.session_state.is_playing = True
    if pause_btn:
        st.session_state.is_playing = False

    #布局：3D视图 + 姿态信息 + 时序曲线
    c_left, c_right = st.columns([3,1])
    with c_left:
        fig3d = build_3d_figure(df, st.session_state.current_frame)
        st.plotly_chart(fig3d, use_container_width=True)
    with c_right:
        cur = df.iloc[st.session_state.current_frame]
        st.subheader("当前姿态")
        st.metric("航向 Heading(°)", round(cur["heading"],1))
        st.metric("俯仰 Pitch(°)", round(cur["pitch"],1))
        st.metric("滚转 Roll(°)", round(cur["roll"],1))

        st.subheader("姿态时序曲线")
        fig_angle = go.Figure()
        fig_angle.add_trace(go.Scatter(y=df["heading"], name="Heading"))
        fig_angle.add_trace(go.Scatter(y=df["pitch"], name="Pitch"))
        fig_angle.add_trace(go.Scatter(y=df["roll"], name="Roll"))
        st.plotly_chart(fig_angle, use_container_width=True)

    #自动播放循环逻辑
    if st.session_state.is_playing:
        if st.session_state.current_frame < total_frame:
            st.session_state.current_frame += 1
            time.sleep(0.08)
            st.rerun()
        else:
            st.session_state.is_playing = False
else:
    st.info("👉 请在左侧侧边栏输入CSV数据，点击【加载数据】开始可视化")