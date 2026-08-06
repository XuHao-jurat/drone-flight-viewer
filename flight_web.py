def parse_csv_data(csv_string):
    import pandas as pd
    from io import StringIO

    # --------------- 1. 先清洗每行前导逗号 ---------------
    lines = csv_string.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(','):
            line = line[1:]
        cleaned_lines.append(line)

    # --------------- 2. 不设表头，先原始读取 ---------------
    df_raw = pd.read_csv(StringIO('\n'.join(cleaned_lines)), header=None)

    # --------------- 3. 把所有列转成数值，非数值变成NaN ---------------
    num_df = df_raw.apply(pd.to_numeric, errors='coerce')

    # --------------- 4. 诊断每一列的数值范围 ---------------
    col_info = []
    for col in num_df.columns:
        vals = num_df[col].dropna()
        if len(vals) == 0:
            continue
        col_info.append({
            "col_index": col,
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
            "mean": round(float(vals.mean()), 6)
        })

    with st.expander("🔍 文件列诊断", expanded=True):
        st.write("原始列数：", len(df_raw.columns))
        st.write("每列数值统计：")
        st.json(col_info)

    # --------------- 5. 根据数值范围强制识别字段 ---------------
    def pick_col(condition):
        for item in col_info:
            if condition(item):
                return item["col_index"]
        return None

    # 纬度：-90 ~ 90，且绝对值大概几十
    lat_col = pick_col(
        lambda x: -90 <= x["min"] <= 90
        and -90 <= x["max"] <= 90
        and 20 < abs(x["mean"]) < 80
    )

    # 经度：-180 ~ 180，且绝对值大于纬度
    lon_col = pick_col(
        lambda x: -180 <= x["min"] <= 180
        and -180 <= x["max"] <= 180
        and abs(x["mean"]) > 90
    )

    # 高度：均值明显大于1000
    alt_col = pick_col(lambda x: x["mean"] > 1000)

    # 航向：0 ~ 360，且均值在100 ~ 300
    hdg_col = pick_col(
        lambda x: 0 <= x["min"] <= 360
        and 0 <= x["max"] <= 360
        and 100 < x["mean"] < 300
    )

    # 俯仰：小角度，-10 ~ 10，标准差不能太小
    pitch_col = pick_col(
        lambda x: -10 <= x["min"] <= 10
        and -10 <= x["max"] <= 10
        and x["max"] - x["min"] > 0.5
    )

    # 滚转：另一个小角度列
    used_cols = [lat_col, lon_col, alt_col, hdg_col, pitch_col]

    roll_col = pick_col(
        lambda x: -10 <= x["min"] <= 10
        and -10 <= x["max"] <= 10
        and x["col_index"] not in used_cols
        and x["max"] - x["min"] > 0.5
    )

    # --------------- 6. 兜底：如果自动识别失败，取最后6列 ---------------
    required = ["latitude", "longitude", "altitude", "heading", "pitch", "roll"]
    matched_cols = [lat_col, lon_col, alt_col, hdg_col, pitch_col, roll_col]

    if None in matched_cols:
        st.warning("自动识别字段失败，将按最后6列强制映射。")
        if len(num_df.columns) >= 6:
            last_six = list(num_df.columns[-6:])
            lat_col, lon_col, alt_col, hdg_col, pitch_col, roll_col = last_six
        else:
            raise ValueError("CSV列数不足，无法解析。")

    # --------------- 7. 构建标准DataFrame ---------------
    df = pd.DataFrame()
    df["latitude"] = num_df[lat_col]
    df["longitude"] = num_df[lon_col]
    df["altitude"] = num_df[alt_col]
    df["heading"] = num_df[hdg_col]
    df["pitch"] = num_df[pitch_col]
    df["roll"] = num_df[roll_col]

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("有效数据行为0。")

    # --------------- 8. 最终校验 ---------------
    with st.expander("✅ 解析结果校验", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.write("纬度范围：", df["latitude"].min(), "~", df["latitude"].max())
        st.write("经度范围：", df["longitude"].min(), "~", df["longitude"].max())
        st.write("高度范围：", df["altitude"].min(), "~", df["altitude"].max())
        st.write("航向范围：", df["heading"].min(), "~", df["heading"].max())
        st.write("俯仰范围：", df["pitch"].min(), "~", df["pitch"].max())
        st.write("滚转范围：", df["roll"].min(), "~", df["roll"].max())

    return df