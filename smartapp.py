import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

# Live streaming (WebSocket)
import asyncio
import json
try:
    import websockets  # pip install websockets
except ImportError:
    websockets = None

st.set_page_config(page_title="Smart Traffic Monitoring & Prediction", layout="wide")

# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\teste\Downloads\smart_traffic_management_dataset.csv")  # replace with your dataset
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

st.title("🚦 Smart Traffic Monitoring & Prediction Dashboard")
st.sidebar.markdown("## Navigation")
menu = st.sidebar.radio(
    "Select a Module",
    [
        "EDA", "KPIs & Correlations", "Outlier Detection", "Traffic Prediction",
        "Model Insights", "Forecasting", "Real-Time Simulation", "Route Optimization",
        "Accident Analysis", "Heatmap & Incidents", "Upload/Download Data", "Map Visualization",
        "A/B Model Compare", "Live Feed"
    ]
)

# Helpers
def hour_from_ts(series):
    return series.dt.hour

def day_from_ts(series):
    return series.dt.day_name()

# ------------------ EDA ------------------ #
if menu == "EDA":
    st.header("Exploratory Data Analysis")
    st.markdown("""
    - Displays dataset structure and samples for quick inspection. [1]
    - Descriptive statistics highlight traffic patterns and spread. [1]
    - Animated histogram by hour shows cyclical volume dynamics. [2]
    """)

    st.write(df.head())  # [1]
    st.write(df.describe())  # [1]

    # Original animated histogram
    fig = px.histogram(
        df,
        x="traffic_volume",
        nbins=30,
        title="Traffic Volume Distribution",
        marginal="box",
        color_discrete_sequence=['indianred'],
        animation_frame=hour_from_ts(df['timestamp'])
    )
    st.plotly_chart(fig, use_container_width=True)  # [2]

    # New: density contour animation
    if st.button("Show Density Contours by Hour (New)"):
        if "avg_vehicle_speed" in df.columns:
            fig2 = px.density_contour(
                df,
                x="avg_vehicle_speed",
                y="traffic_volume",
                animation_frame=hour_from_ts(df['timestamp']),
                title="Speed vs Volume Density Contours by Hour"
            )
            st.plotly_chart(fig2, use_container_width=True)  # [2]
        else:
            st.info("avg_vehicle_speed column not found.")  # [1]

    # New: Animated Heatmap 2D density by hour
    if st.button("Animated Heatmap: Volume vs Speed by Hour (New)"):
        if {"avg_vehicle_speed","traffic_volume"}.issubset(df.columns):
            dfa = df.copy()
            dfa["hour"] = hour_from_ts(dfa["timestamp"])
            bins_x = np.linspace(dfa["avg_vehicle_speed"].min(), dfa["avg_vehicle_speed"].max(), 40)
            bins_y = np.linspace(dfa["traffic_volume"].min(), dfa["traffic_volume"].max(), 40)
            frames = []
            for h in sorted(dfa["hour"].unique()):
                sub = dfa[dfa["hour"] == h]
                H, xe, ye = np.histogram2d(sub["avg_vehicle_speed"], sub["traffic_volume"], bins=[bins_x, bins_y])
                frames.append({"hour": h, "H": H.T})
            base = frames["H"]
            fig_h = go.Figure(
                data=[go.Heatmap(z=base, colorscale="Turbo")],
                layout=go.Layout(
                    title="Speed vs Volume Density by Hour",
                    xaxis_title="Speed", yaxis_title="Volume",
                    updatemenus=[{"type": "buttons", "buttons":[{"label":"Play","method":"animate","args":[None]}]}]
                )
            )
            fig_h.frames = [go.Frame(data=[go.Heatmap(z=f["H"], colorscale="Turbo")], name=str(f["hour"])) for f in frames]
            st.plotly_chart(fig_h, use_container_width=True)  # [2]

    # New: Small multiples by weekday
    if st.button("Facet: Volume vs Speed by Weekday (New)"):
        dfx = df.copy()
        dfx["day"] = day_from_ts(dfx["timestamp"])
        if {"avg_vehicle_speed","traffic_volume"}.issubset(dfx.columns):
            fig_fac = px.scatter(
                dfx.sample(min(5000, len(dfx))), x="avg_vehicle_speed", y="traffic_volume",
                color="day", facet_col="day", facet_col_wrap=3, opacity=0.5, title="Small Multiples: Speed vs Volume by Day"
            )
            st.plotly_chart(fig_fac, use_container_width=True)  # [4]

# ------------------ KPIs & Correlations ------------------ #
elif menu == "KPIs & Correlations":
    st.header("KPIs & Correlations")
    st.markdown("""
    - KPI cards summarize mean volume, speed, and incidents. [1]
    - Correlation heatmap toggle to inspect feature relationships. [1]
    - Hourly bar animation reveals cyclic demand. [2]
    """)

    c1, c2, c3, c4 = st.columns(4)
    mean_vol = float(df["traffic_volume"].mean()) if "traffic_volume" in df.columns else 0.0
    mean_speed = float(df["avg_vehicle_speed"].mean()) if "avg_vehicle_speed" in df.columns else 0.0

    # Safe accident count (pandas 2.x-safe)
    if "accident_reported" in df.columns:
        s = df["accident_reported"]
        if s.dtype == bool:
            inc_count = int(s.astype(int).sum())
        elif np.issubdtype(s.dtype, np.number):
            inc_count = int(pd.to_numeric(s, errors="coerce").fillna(0).sum())
        else:
            inc_count = int(s.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int).sum())
    else:
        inc_count = int(pd.Series(0, index=df.index).sum())

    records = len(df)

    c1.metric("Avg Volume", f"{mean_vol:.1f}")  # [1]
    c2.metric("Avg Speed", f"{mean_speed:.1f} km/h")  # [1]
    c3.metric("Accidents", f"{inc_count}")  # [1]
    c4.metric("Records", f"{records}")  # [1]

    if st.toggle("Show Correlation Heatmap"):
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        fig_hm = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)  # [2]

    if st.button("Animate Hourly Volume Bars"):
        df_hour = df.copy()
        df_hour["hour"] = hour_from_ts(df_hour["timestamp"])
        hourly = df_hour.groupby("hour", as_index=False)["traffic_volume"].mean()
        hourly["frame"] = hourly["hour"].rank().astype(int)
        fig_bar = px.bar(
            hourly,
            x="hour",
            y="traffic_volume",
            color="traffic_volume",
            animation_frame="frame",
            title="Hourly Avg Traffic Volume (Animated Bars)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)  # [2]

    # New: Rolling window animation
    if st.button("Animate Rolling Hourly Mean (New)"):
        dfx = df.copy()
        dfx["hour"] = hour_from_ts(dfx["timestamp"])
        hourly = dfx.groupby("hour", as_index=False)["traffic_volume"].mean().sort_values("hour")
        hourly["roll3"] = hourly["traffic_volume"].rolling(3, min_periods=1).mean()
        hourly["frame"] = hourly.index
        fig_roll = px.line(hourly, x="hour", y=["traffic_volume","roll3"],
                           animation_frame="frame", title="Rolling Hourly Mean (3-hr) Animation")
        st.plotly_chart(fig_roll, use_container_width=True)  # [2]

# ------------------ Outlier Detection ------------------ #
elif menu == "Outlier Detection":
    st.header("Outlier Detection in Traffic Volume")
    st.markdown("""
    - Z-score flags unusual spikes for proactive monitoring. [1]
    - Scatter animation over time contrasts normal vs outliers. [2]
    - Alternate: blinking alert banner when many outliers detected. [5]
    """)

    if st.button("Show Animated Outliers"):
        df_det = df.copy()
        df_det['z_score'] = (df_det['traffic_volume'] - df_det['traffic_volume'].mean()) / df_det['traffic_volume'].std()
        df_det['is_outlier'] = df_det['z_score'].abs() > 3
        fig = px.scatter(
            df_det,
            x='timestamp',
            y='traffic_volume',
            color='is_outlier',
            animation_frame=hour_from_ts(df_det['timestamp']),
            title='Outlier Traffic Visualization'
        )
        st.plotly_chart(fig, use_container_width=True)  # [2]

        share = df_det['is_outlier'].mean()
        if share > 0.1:
            st.markdown(
                """
                <div style="padding:10px;border-radius:6px;background:#ffdddd;animation: blink 1s step-start infinite;">
                ⚠️ High Outlier Share Detected
                </div>
                <style>
                @keyframes blink { 50% { opacity: 0.3; } }
                </style>
                """,
                unsafe_allow_html=True
            )  # [5]

    # New: Threshold sweep animation
    if st.button("Animate Threshold Sweep (New)"):
        dfo = df.copy()
        mu = dfo["traffic_volume"].mean()
        sd = dfo["traffic_volume"].std()
        thresholds = [2.0, 2.5, 3.0, 3.5]
        frames = []
        for t in thresholds:
            flag = (np.abs((dfo["traffic_volume"] - mu)/sd) > t)
            frames.append(go.Frame(data=[go.Scattergl(x=dfo["timestamp"], y=dfo["traffic_volume"],
                                                      mode="markers",
                                                      marker=dict(color=np.where(flag, "red", "green"), size=6))],
                                   name=f"z>{t}"))
        base_flag = (np.abs((dfo["traffic_volume"] - mu)/sd) > thresholds)
        fig_thr = go.Figure(
            data=[go.Scattergl(x=dfo["timestamp"], y=dfo["traffic_volume"], mode="markers",
                               marker=dict(color=np.where(base_flag, "red", "green"), size=6))],
            layout=go.Layout(title="Outlier Threshold Sweep", updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}])
        )
        fig_thr.frames = frames
        st.plotly_chart(fig_thr, use_container_width=True)  # [2]

# ------------------ Traffic Prediction ------------------ #
elif menu == "Traffic Prediction":
    st.header("Traffic Signal Status Prediction")
    st.markdown("""
    - Random Forest predicts signal_status using volume, speed, weather. [1]
    - Confusion matrix animation frames simulate evolving batches. [2]
    - Classification metrics printed for evaluation. [1]
    """)

    if st.button("Run Animated Prediction"):
        X = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]]
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', title='Confusion Matrix Animation')
        st.plotly_chart(fig, use_container_width=True)  # [2]

        frames = []
        for scale in [0.2, 0.5, 1.0]:
            cm_scaled = (cm * scale).astype(int)
            frames.append(go.Frame(data=[go.Heatmap(z=cm_scaled, colorscale='Viridis')]))
        fig_anim = go.Figure(
            data=[go.Heatmap(z=cm, colorscale='Viridis')],
            layout=go.Layout(title="Confusion Matrix (Staged Animation)", updatemenus=[{
                "type": "buttons",
                "buttons": [{"label": "Play", "method": "animate", "args": [None]}]
            }]),
            frames=frames
        )
        st.plotly_chart(fig_anim, use_container_width=True)  # [2]

        st.text(classification_report(y_test, preds))  # [1]

    # New: Class probability distributions
    if st.button("Animate Class Score Distributions (New)"):
        X = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]]
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        if hasattr(rf, "predict_proba"):
            proba = rf.predict_proba(X_test)
            dd = []
            for cls in range(proba.shape[6]):
                dd.append(pd.DataFrame({"score": proba[:, cls], "class": f"Class {cls}"}))
            scores = pd.concat(dd)
            scores["frame"] = scores["class"]
            fig_sc = px.histogram(scores, x="score", color="class", nbins=20,
                                  animation_frame="frame", barmode="overlay",
                                  opacity=0.6, title="Predicted Class Probability Distributions")
            st.plotly_chart(fig_sc, use_container_width=True)  # [2]

# ------------------ Model Insights ------------------ #
elif menu == "Model Insights":
    st.header("Model Insights")
    st.markdown("""
    - Feature importance visualization with animated emphasis. [2]
    - Pseudo SHAP-like perturbation animation for intuition. [5]
    - Assists explainability of signal predictions. [1]
    """)

    if st.button("Animate Feature Importance"):
        X = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]]
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        imp = rf.feature_importances_
        fdf = pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False)
        fdf["frame"] = range(1, len(fdf) + 1)
        fig_imp = px.bar(
            fdf,
            x="feature",
            y="importance",
            animation_frame="frame",
            color="importance",
            title="Feature Importance Animation"
        )
        st.plotly_chart(fig_imp, use_container_width=True)  # [2]

    if st.button("Run SHAP-like Perturbation Animation"):
        X_base = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]].copy()
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        frames_all = []
        features = X_base.columns.tolist()
        for i, feat in enumerate(features):
            grid = np.linspace(X_base[feat].quantile(0.05), X_base[feat].quantile(0.95), 12)
            preds_mode = []
            for val in grid:
                X_tmp = X_test.copy()
                X_tmp[feat] = val
                preds_mode.append(pd.Series(model.predict(X_tmp)).value_counts(normalize=True).reindex([0,1,2]).fillna(0).values)
            pm = np.array(preds_mode)
            dfm = pd.DataFrame({
                "value": np.repeat(grid, 3),
                "class": np.tile(["Green(0)", "Yellow(1)", "Red(2)"], len(grid)),
                "prob": pm.flatten(),
                "feature": feat
            })
            frames_all.append(dfm)

        viz = pd.concat(frames_all, ignore_index=True)
        fig_prob = px.line(
            viz, x="value", y="prob", color="class",
            animation_frame="feature", title="Class Share vs Feature (Perturbation Animation)"
        )
        st.plotly_chart(fig_prob, use_container_width=True)  # [2]

    # New: Polar (radar) animated reveal using go.Scatterpolar (pandas 2.x-safe)
    if st.button("Animate Polar Feature Importance (New)"):
        X = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]]
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        features = imp.index.tolist()
        values = imp.values.tolist()

        fig_pol = go.Figure()
        # Start with first point closed
        base_r = values[:1] + values[:1]
        base_theta = features[:1] + features[:1]
        fig_pol.add_trace(go.Scatterpolar(r=base_r, theta=base_theta, fill="toself", name="Importance"))
        frames = []
        for k in range(1, len(features)+1):
            rk = values[:k] + values[:1]
            thetak = features[:k] + features[:1]
            frames.append(go.Frame(data=[go.Scatterpolar(r=rk, theta=thetak, fill="toself", name="Importance")], name=str(k)))
        fig_pol.update_layout(
            title="Polar Feature Importance (Animated Reveal)",
            updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}],
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.2]))
        )
        fig_pol.frames = frames
        st.plotly_chart(fig_pol, use_container_width=True)  # [3]

# ------------------ Forecasting ------------------ #
elif menu == "Forecasting":
    st.header("Traffic Volume Forecasting")
    st.markdown("""
    - Hourly resample with rolling mean smoothing. [1]
    - Animated trend line with frame index evolution. [2]
    - Optional seasonal decomposition preview chart. [1]
    """)

    if st.button("Start Forecast Animation"):
        df_ts = df.set_index('timestamp').resample('H').mean(numeric_only=True)
        df_ts['rolling_mean'] = df_ts['traffic_volume'].rolling(window=3).mean()
        df_ts_reset = df_ts.reset_index()
        fig = px.line(
            df_ts_reset, x='timestamp', y=['traffic_volume','rolling_mean'],
            title='Animated Traffic Forecast', animation_frame=df_ts_reset.index
        )
        st.plotly_chart(fig, use_container_width=True)  # [2]

    if st.button("Show Seasonal Preview (New)"):
        df_ts = df.set_index('timestamp').resample('H').mean(numeric_only=True)
        df_ts['day'] = df_ts.index.dayofweek
        daily = df_ts.groupby('day')['traffic_volume'].mean()
        fig_day = px.bar(
            daily, title="Weekday Seasonal Preview (Avg Hourly Volume by Weekday)"
        )
        st.plotly_chart(fig_day, use_container_width=True)  # [2]

    # New: Forecast cone animation
    if st.button("Animate Forecast Cone (New)"):
        ts = df.set_index("timestamp").resample("H").mean(numeric_only=True)["traffic_volume"].dropna()
        roll = ts.rolling(6, min_periods=2)
        dfc = pd.DataFrame({
            "t": ts.index, "y": ts.values,
            "mean": roll.mean().values,
            "upper": (roll.mean() + 1.5*roll.std()).values,
            "lower": (roll.mean() - 1.5*roll.std()).values
        }).dropna()
        dfc["frame"] = np.arange(len(dfc))
        fig_cone = go.Figure()
        fig_cone.add_trace(go.Scatter(x=dfc["t"], y=dfc["upper"], line=dict(color="orange"), name="Upper"))
        fig_cone.add_trace(go.Scatter(x=dfc["t"], y=dfc["lower"], fill="tonexty", line=dict(color="orange"), name="Lower"))
        fig_cone.add_trace(go.Scatter(x=dfc["t"], y=dfc["mean"], line=dict(color="blue"), name="Rolling Mean"))
        frames = []
        for i in range(5, len(dfc), max(1, len(dfc)//30)):
            frames.append(go.Frame(data=[
                go.Scatter(x=dfc["t"][:i], y=dfc["upper"][:i]),
                go.Scatter(x=dfc["t"][:i], y=dfc["lower"][:i]),
                go.Scatter(x=dfc["t"][:i], y=dfc["mean"][:i]),
            ], name=str(i)))
        fig_cone.update_layout(title="Forecast Cone Animation",
                               updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}])
        fig_cone.frames = frames
        st.plotly_chart(fig_cone, use_container_width=True)  # [2]

# ------------------ Real-Time Simulation ------------------ #
elif menu == "Real-Time Simulation":
    st.header("Live Traffic Simulation")
    st.markdown("""
    - Live loop mutates traffic and speed, updating signal. [5]
    - Line plot refreshed in a placeholder for real-time feel. [1]
    - New: Real-time anomaly banner blinking on threshold. [5]
    """)

    if st.button("Start Real-Time Simulation"):
        placeholder = st.empty()
        alert_box = st.empty()
        df_sim = df.copy()
        for i in range(10):
            df_sim['traffic_volume'] = df_sim['traffic_volume'].apply(lambda x: max(0, x + random.randint(-20, 20)))
            df_sim['avg_vehicle_speed'] = df_sim['avg_vehicle_speed'].apply(lambda x: max(0, min(120, x + random.randint(-5, 5))))
            df_sim['signal_status'] = df_sim['traffic_volume'].apply(lambda x: 2 if x>500 else (1 if x>250 else 0))
            fig = px.line(df_sim.head(50), y=['traffic_volume','avg_vehicle_speed'], title='Real-Time Traffic Simulation')
            placeholder.plotly_chart(fig, use_container_width=True)  # [2]
            red_share = (df_sim['signal_status'] == 2).mean()
            if red_share > 0.4:
                alert_box.markdown(
                    """
                    <div style="padding:8px;border-radius:6px;background:#ffe7e7;animation: blink 0.8s step-start infinite;">
                    🔴 High Congestion Detected: Red Signals Rising
                    </div>
                    <style>@keyframes blink { 50% { opacity: 0.35; } }</style>
                    """,
                    unsafe_allow_html=True
                )  # [5]
            else:
                alert_box.empty()
            time.sleep(1)

    # New: dual axes animated subplot
    if st.button("Animate Dual Axes (New)"):
        dfr = df.head(200).copy()
        dfr["idx"] = np.arange(len(dfr))
        frames = []
        for i in range(10, len(dfr), max(1, len(dfr)//25)):
            frames.append(go.Frame(data=[
                go.Scatter(x=dfr["idx"][:i], y=dfr["traffic_volume"][:i], name="Volume", yaxis="y1"),
                go.Scatter(x=dfr["idx"][:i], y=dfr["avg_vehicle_speed"][:i], name="Speed", yaxis="y2")
            ], name=str(i)))
        base_i = 15
        fig_dual = go.Figure(
            data=[
                go.Scatter(x=dfr["idx"][:base_i], y=dfr["traffic_volume"][:base_i], name="Volume", yaxis="y1"),
                go.Scatter(x=dfr["idx"][:base_i], y=dfr["avg_vehicle_speed"][:base_i], name="Speed", yaxis="y2")
            ],
            layout=go.Layout(
                title="Dual Axes Animation",
                yaxis=dict(title="Volume"),
                yaxis2=dict(title="Speed", overlaying="y", side="right"),
                updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}]
            ),
            frames=frames
        )
        st.plotly_chart(fig_dual, use_container_width=True)  # [2]

# ------------------ Route Optimization ------------------ #
elif menu == "Route Optimization":
    st.header("Route Optimization")
    st.markdown("""
    - Simulated congestion scores with animated bars. [2]
    - Best route highlighted with success message. [1]
    - New: ETA simulation radar chart for multi-metric view. (pandas 2.x-safe) [3]
    """)

    if st.button("Simulate Route Congestion Animation"):
        routes = {f"Route {i}": random.randint(10,100) for i in range(1,6)}
        best_route = min(routes, key=routes.get)
        fig = px.bar(
            x=list(routes.keys()), y=list(routes.values()),
            color=list(routes.values()), title='Route Congestion Animation',
            animation_frame=list(range(1,6))
        )
        st.plotly_chart(fig, use_container_width=True)  # [2]
        st.success(f"Best Route: {best_route} with congestion level {routes[best_route]}")  # [1]

    # New: Radar with go.Scatterpolar to avoid DataFrame.append in px
    if st.button("Simulate ETA Radar (New)"):
        labels = [f"Route {i}" for i in range(1,6)]
        eta = np.random.uniform(12, 35, 5)
        reliability = np.random.uniform(60, 95, 5)
        safety = np.random.uniform(70, 98, 5)
        df_radar = pd.DataFrame({
            "Route": labels,
            "ETA(min)": eta,
            "Reliability(%)": reliability,
            "Safety(%)": safety
        })
        categories = ["ETA(min)", "Reliability(%)", "Safety(%)"]
        fig_radar = go.Figure()
        for _, row in df_radar.iterrows():
            r_vals = [row[c] for c in categories] + [row[categories]]  # close loop
            theta_vals = categories + [categories]
            fig_radar.add_trace(go.Scatterpolar(r=r_vals, theta=theta_vals, name=row["Route"], line=dict(shape="linear")))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Route Multi-Metric Radar (ETA, Reliability, Safety)")
        st.plotly_chart(fig_radar, use_container_width=True)  # [3]

    # New: Bar race timesteps
    if st.button("Bar Race: Congestion Timesteps (New)"):
        steps = 12
        rows = []
        for t in range(steps):
            for r in range(1,6):
                rows.append({"t": t, "route": f"Route {r}", "cong": random.randint(10,100)})
        br = pd.DataFrame(rows)
        fig_br = px.bar(br, x="cong", y="route", orientation="h", color="route",
                        animation_frame="t", animation_group="route",
                        range_x=[0, 110], title="Congestion Bar Race")
        st.plotly_chart(fig_br, use_container_width=True)  # [2]

# ------------------ Accident Analysis ------------------ #
elif menu == "Accident Analysis":
    st.header("Accident Impact Analysis")
    st.markdown("""
    - Boxplot contrasts traffic distributions by accident flag. [1]
    - Animated hourly frames for accident impact. [2]
    - New: bar race of accident counts by hour. [2]
    """)

    if st.button("Animate Accident Impact"):
        fig = px.box(
            df, x='accident_reported', y='traffic_volume', points='all',
            animation_frame=hour_from_ts(df['timestamp']),
            title='Animated Accident Impact on Traffic'
        )
        st.plotly_chart(fig, use_container_width=True)  # [2]

    if st.button("Run Accident Count Bar Race (New)"):
        dfx = df.copy()
        dfx["hour"] = hour_from_ts(dfx["timestamp"])
        race = dfx.groupby(["hour", "accident_reported"]).size().reset_index(name="count")
        race["frame"] = race["hour"].rank().astype(int)
        fig_race = px.bar(
            race, x="accident_reported", y="count", color="accident_reported",
            animation_frame="frame", title="Accident Counts by Hour (Bar Race)"
        )
        st.plotly_chart(fig_race, use_container_width=True)  # [2]

    # New: Stacked area by hour
    if st.button("Animate Stacked Area by Hour (New)"):
        dfx = df.copy()
        dfx["hour"] = hour_from_ts(dfx["timestamp"])
        agg = dfx.groupby(["hour","accident_reported"]).size().reset_index(name="count")
        wide = agg.pivot(index="hour", columns="accident_reported", values="count").fillna(0)
        wide = wide.sort_index()
        wide["frame"] = np.arange(len(wide))
        fig_area = px.area(wide.reset_index(), x="hour", y=wide.columns[:-1], animation_frame="frame",
                           title="Accident Reported Stacked Area by Hour")
        st.plotly_chart(fig_area, use_container_width=True)  # [4]

# ------------------ Heatmap & Incidents ------------------ #
elif menu == "Heatmap & Incidents":
    st.header("Traffic Heatmap & Incidents")
    st.markdown("""
    - Hour vs weekday heatmap for volume patterns. [2]
    - Incident intensity heatmap for spatial-temporal signals. [2]
    - Helps spot recurring congestion windows. [1]
    """)

    if st.button("Show Hour vs Day Heatmap"):
        dfx = df.copy()
        dfx["hour"] = hour_from_ts(dfx["timestamp"])
        dfx["day"] = day_from_ts(dfx["timestamp"])
        hm = dfx.pivot_table(index="day", columns="hour", values="traffic_volume", aggfunc="mean").reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )
        fig_hm = px.imshow(hm, aspect="auto", title="Avg Traffic Volume: Day vs Hour")
        st.plotly_chart(fig_hm, use_container_width=True)  # [2]

    if st.button("Show Incident Heatmap"):
        if "accident_reported" in df.columns:
            dfx = df.copy()
            dfx["hour"] = hour_from_ts(dfx["timestamp"])
            dfx["day"] = day_from_ts(dfx["timestamp"])
            hm2 = dfx.pivot_table(index="day", columns="hour", values="accident_reported", aggfunc="sum").reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            )
            fig_hm2 = px.imshow(hm2, aspect="auto", color_continuous_scale="Reds", title="Incidents Heatmap: Day vs Hour")
            st.plotly_chart(fig_hm2, use_container_width=True)  # [2]
        else:
            st.info("accident_reported column not found.")  # [1]

    # New: 2D density heatmap
    if st.button("2D Histogram: Speed vs Volume (New)"):
        if {"avg_vehicle_speed","traffic_volume"}.issubset(df.columns):
            fig_hex = px.density_heatmap(df, x="avg_vehicle_speed", y="traffic_volume",
                                         nbinsx=40, nbinsy=40, color_continuous_scale="Viridis",
                                         title="Density Heatmap: Speed vs Volume")
            st.plotly_chart(fig_hex, use_container_width=True)  # [2]

# ------------------ Upload/Download ------------------ #
elif menu == "Upload/Download Data":
    st.header("Upload / Download Data")
    st.markdown("""
    - Upload CSV integrates data on the fly. [1]
    - Download current working set for export. [1]
    - Compatible with Streamlit caching semantics. [1]
    """)

    uploaded_file = st.file_uploader("Upload CSV", type='csv')
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.write(df_new.head())  # [1]
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Current Data", csv, "traffic_results.csv", "text/csv")  # [1]

# ------------------ Map Visualization ------------------ #
elif menu == "Map Visualization":
    st.header("Traffic Map Visualization")
    st.markdown("""
    - Folium map shows hotspots with signal-coded markers. [7]
    - Random sample lat/lon for demo; integrate GPS if available. [7]
    - New: Hourly playhead button rerenders by hour. [7]
    """)

    if st.button("Animate Map Hotspots"):
        df_map = df.copy()
        df_map['lat'] = np.random.uniform(12.9, 13.1, len(df_map))
        df_map['lon'] = np.random.uniform(77.5, 77.7, len(df_map))
        m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=12)
        for _, row in df_map.iterrows():
            color = 'red' if row['signal_status']==2 else ('orange' if row['signal_status']==1 else 'green')
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5, color=color, fill=True, fill_opacity=0.7,
                popup=f"Traffic: {row['traffic_volume']}"
            ).add_to(m)
        st_folium(m, width=700, height=500)  # [7]

    if st.button("Play Hourly Map (New)"):
        df_map = df.copy()
        df_map["hour"] = hour_from_ts(df_map["timestamp"])
        df_map['lat'] = np.random.uniform(12.9, 13.1, len(df_map))
        df_map['lon'] = np.random.uniform(77.5, 77.7, len(df_map))
        placeholder = st.empty()
        for h in sorted(df_map["hour"].unique()):
            subset = df_map[df_map["hour"] == h]
            m = folium.Map(location=[subset['lat'].mean(), subset['lon'].mean()], zoom_start=12)
            for _, row in subset.iterrows():
                color = 'red' if row['signal_status']==2 else ('orange' if row['signal_status']==1 else 'green')
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5, color=color, fill=True, fill_opacity=0.7,
                    popup=f"Hour {h} | Traffic: {row['traffic_volume']}"
                ).add_to(m)
            with placeholder.container():
                st.markdown(f"#### Hour: {h}")  # [1]
                st_folium(m, width=700, height=500)  # [7]
            time.sleep(0.6)

    # New: HeatMap overlay
    if st.button("Show HeatMap Overlay (New)"):
        df_map = df.copy()
        df_map['lat'] = np.random.uniform(12.9, 13.1, len(df_map))
        df_map['lon'] = np.random.uniform(77.5, 77.7, len(df_map))
        m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=12)
        heat_data = df_map[["lat","lon","traffic_volume"]].dropna().values.tolist()
        HeatMap(heat_data, radius=12, blur=18, max_zoom=13).add_to(m)
        st_folium(m, width=700, height=500)  # [7]

    # New: MarkerCluster
    if st.button("Show MarkerCluster (New)"):
        df_map = df.copy()
        df_map['lat'] = np.random.uniform(12.9, 13.1, len(df_map))
        df_map['lon'] = np.random.uniform(77.5, 77.7, len(df_map))
        m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=12)
        mc = MarkerCluster().add_to(m)
        for _, row in df_map.iterrows():
            color = 'red' if row['signal_status']==2 else ('orange' if row['signal_status']==1 else 'green')
            folium.CircleMarker(location=[row['lat'], row['lon']], radius=5, color=color, fill=True,
                                fill_opacity=0.8, popup=f"Vol: {row['traffic_volume']}").add_to(mc)
        st_folium(m, width=700, height=500)  # [7]

# ------------------ A/B Model Compare ------------------ #
elif menu == "A/B Model Compare":
    st.header("A/B Model Comparison")
    st.markdown("""
    - Compare RandomForest vs GradientBoosting on accuracy/F1. [1]
    - Bar chart comparison with animated emphasis. [2]
    - Helps select robust model for deployment. [1]
    """)

    if st.button("Run A/B Comparison"):
        X = df[["traffic_volume", "avg_vehicle_speed", "temperature", "humidity"]]
        y = df["signal_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        rf.fit(X_train, y_train); gb.fit(X_train, y_train)

        prf = rf.predict(X_test)
        pgb = gb.predict(X_test)

        metrics = pd.DataFrame({
            "Model": ["RandomForest","GradientBoosting","RandomForest","GradientBoosting"],
            "Metric": ["Accuracy","Accuracy","F1-macro","F1-macro"],
            "Score": [
                accuracy_score(y_test, prf),
                accuracy_score(y_test, pgb),
                f1_score(y_test, prf, average="macro"),
                f1_score(y_test, pgb, average="macro")
            ],
            "frame": [1,1,2,2]
        })
        fig_cmp = px.bar(
            metrics, x="Model", y="Score", color="Model", facet_col="Metric",
            animation_frame="frame", title="A/B Model Comparison (Accuracy vs F1)"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)  # [2]

# ------------------ Live Feed (WebSocket) ------------------ #
elif menu == "Live Feed":
    st.header("Real-Time Live Feed")
    st.markdown("""
    - Connect to a WebSocket endpoint to receive live traffic updates. [8]
    - Updates KPIs and charts in near real-time via session state. [9]
    - Requires a running server that streams JSON messages. [8]
    """)

    # Defaults/state
    if "live_data" not in st.session_state:
        st.session_state.live_data = []  # list of dicts
    if "ws_connected" not in st.session_state:
        st.session_state.ws_connected = False
    if "ws_task" not in st.session_state:
        st.session_state.ws_task = None
    if "ws_uri" not in st.session_state:
        st.session_state.ws_uri = "ws://localhost:8000/traffic"

    uri = st.text_input("WebSocket URI", value=st.session_state.ws_uri)  # [8]
    st.session_state.ws_uri = uri

    placeholder_kpi = st.empty()
    placeholder_chart = st.empty()

    async def ws_listener(uri_str):
        if websockets is None:
            st.error("websockets package not installed. Run: pip install websockets")  # [10]
            return
        try:
            async with websockets.connect(uri_str) as ws:
                st.session_state.ws_connected = True
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    st.session_state.live_data.append(msg)
                    if len(st.session_state.live_data) > 2000:
                        st.session_state.live_data = st.session_state.live_data[-2000:]
                    await asyncio.sleep(0)
        except Exception:
            st.session_state.ws_connected = False
            st.session_state.ws_task = None

    def start_ws():
        if st.session_state.ws_task is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            st.session_state.ws_task = loop.create_task(ws_listener(st.session_state.ws_uri))
            loop.call_soon(lambda: None)

    def stop_ws():
        if st.session_state.ws_task is not None:
            st.session_state.ws_task.cancel()
            st.session_state.ws_task = None
        st.session_state.ws_connected = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect Live Feed"):
            start_ws()  # [10]
    with c2:
        if st.button("Disconnect"):
            stop_ws()  # [9]

    tail_n = st.slider("Tail length for live line", 50, 1000, 300)  # [11]

    def render_live():
        data = st.session_state.live_data
        if not data:
            st.info("Waiting for live messages...")  # [1]
            return
        df_live = pd.DataFrame(data)
        # KPI calc
        vol = float(df_live["traffic_volume"].tail(50).mean()) if "traffic_volume" in df_live else 0.0
        spd = float(df_live["avg_vehicle_speed"].tail(50).mean()) if "avg_vehicle_speed" in df_live else 0.0
        red_share = float((df_live.get("signal_status", pd.Series(0, index=df_live.index)) == 2).tail(200).mean())

        with placeholder_kpi.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Live Avg Volume (last 50)", f"{vol:.1f}")  # [1]
            k2.metric("Live Avg Speed (last 50)", f"{spd:.1f} km/h")  # [1]
            k3.metric("Red Signal Share (last 200)", f"{red_share*100:.1f}%")  # [1]

        # Primary live line
        last = df_live.tail(tail_n)
        if "timestamp" in last:
            try:
                last["timestamp"] = pd.to_datetime(last["timestamp"])
            except Exception:
                pass
        ycols = [c for c in ["traffic_volume", "avg_vehicle_speed"] if c in last.columns]
        if ycols:
            fig_live = px.line(last, x="timestamp" if "timestamp" in last else last.index, y=ycols, title="Live Volume & Speed")
            placeholder_chart.plotly_chart(fig_live, use_container_width=True)  # [1]

        # Extra: Tail animation
        if ycols:
            dfl = last.copy()
            dfl["frame"] = np.arange(len(dfl))
            fig_tail = px.line(dfl, x="timestamp" if "timestamp" in dfl else dfl.index,
                               y=ycols, animation_frame="frame", title="Live Tail Animation")
            st.plotly_chart(fig_tail, use_container_width=True)  # [2]

        # Extra: Pseudo gauge for red share
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=red_share*100,
            title={'text': "Red Share %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "crimson"}}
        ))
        st.plotly_chart(fig_g, use_container_width=True)  # [4]

    render_live()  # [1]
