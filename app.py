!pip install gradio plotly pandas numpy scikit-learn openpyxl -q

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# === LOAD DATA ===
df = pd.read_excel("Hasil_Pengolahan_DigitalTwin.xlsx")

# === CEK KOMPONEN DATA ===
for col in ["Depth","WOB","SURF_RPM","PHIF","VSH","SW","ROP_AVG","Energy_Intensity","Sustainability_Index","CO2_Emission"]:
    if col not in df.columns:
        raise ValueError(f"Kolom {col} tidak ditemukan, harap periksa dataset Excel.")

# === AI MODEL ===
X = df[["WOB","SURF_RPM","PHIF","VSH","SW"]]
y = df["ROP_AVG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

ann = MLPRegressor(hidden_layer_sizes=(16,8), activation='relu', solver='adam', max_iter=500, random_state=42)
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test)

rf_r2, rf_mae = r2_score(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_rf)
ann_r2, ann_mae = r2_score(y_test, y_pred_ann), mean_absolute_error(y_test, y_pred_ann)

# === DASHBOARD FUNCTION ===
def digital_twin_dashboard(depth_min, depth_max, show_wob, show_rpm, show_energy, show_sustain):
    subset = df[(df["Depth"] >= depth_min) & (df["Depth"] <= depth_max)]

    # === 1️⃣ Grafik utama ===
    fig = go.Figure()
    fig.update_layout(template="plotly_dark",
                      title="Kinerja Pemboran (Digital Twin Simulation)",
                      xaxis_title="Depth (m)", yaxis_title="Normalized Value",
                      title_x=0.5)
    if show_wob:
        fig.add_trace(go.Scatter(x=subset["Depth"], y=subset["WOB"], mode="lines", name="WOB", line=dict(color="orange")))
    if show_rpm:
        fig.add_trace(go.Scatter(x=subset["Depth"], y=subset["SURF_RPM"], mode="lines", name="RPM", line=dict(color="cyan")))
    if show_energy:
        fig.add_trace(go.Scatter(x=subset["Depth"], y=subset["Energy_Intensity"], mode="lines", name="Energy", line=dict(color="red")))
    if show_sustain:
        fig.add_trace(go.Scatter(x=subset["Depth"], y=subset["Sustainability_Index"], mode="lines", name="Sustainability", line=dict(color="lime")))

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))

    # === 2️⃣ AI Prediction Comparison ===
    ai_fig = go.Figure()
    ai_fig.add_trace(go.Scatter(y=y_test, x=y_pred_rf, mode="markers", name="Random Forest", marker=dict(color="blue", size=6)))
    ai_fig.add_trace(go.Scatter(y=y_test, x=y_pred_ann, mode="markers", name="ANN", marker=dict(color="orange", size=6)))
    ai_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="gray", dash='dash'), name="Ideal Fit"))
    ai_fig.update_layout(template="plotly_dark", title="AI Prediction Model (ROP)", xaxis_title="Predicted", yaxis_title="Actual")

    # === 3️⃣ 3D DIGITAL TWIN ANIMASI ===
    frames = []
    depths = np.linspace(0, 2000, 20)
    for d in depths:
        frames.append(go.Frame(data=[
            go.Scatter3d(
                x=[0, 0.5, 1, 1.5, 2], 
                y=[0, 0, 0, 0, 0], 
                z=[0, -d*0.25, -d*0.5, -d*0.75, -d],
                mode="lines+markers", 
                line=dict(color="orange", width=6),
                marker=dict(size=5, color="red")
            )
        ], name=str(int(d))))

    twin_fig = go.Figure(
        data=[go.Scatter3d(
            x=[0, 0.5, 1, 1.5, 2],
            y=[0, 0, 0, 0, 0],
            z=[0, -10, -20, -30, -40],
            mode="lines+markers",
            line=dict(color="orange", width=6),
            marker=dict(size=5, color="red"),
            name="Drill Path"
        )],
        frames=frames
    )
    twin_fig.update_layout(
        template="plotly_dark",
        title="🧱 3D Digital Twin Layout (Animated Drilling Path)",
        scene=dict(zaxis=dict(autorange="reversed"),
                   xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Depth (m)"),
        updatemenus=[dict(type="buttons",
            buttons=[dict(label="▶️ Start Animation", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}])])]
    )

    # === 4️⃣ GEO MAP (Volve Oil Field) ===
    map_fig = go.Figure(go.Scattermapbox(
        lat=[58.357], lon=[1.909],
        mode='markers+text',
        marker=dict(size=18, color='red'),
        text=['Volve Field (Well 15/9-F-15)'],
        textposition="bottom right"
    ))
    map_fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=58.357, lon=1.909), zoom=5),
        title="🌍 GeoDigital Twin – Volve Oil Field (North Sea)"
    )

    # === 5️⃣ ESG + AI Summary ===
    avg_energy = subset["Energy_Intensity"].mean()
    avg_emission = subset["CO2_Emission"].mean()
    esg_text = f"💡 **Energy Intensity:** {avg_energy:.3f} kWh/m  |  🌿 **CO₂ Emission:** {avg_emission:.3f} kgCO₂e/m"
    ai_text = f"🤖 **AI Model Performance**  \nRandom Forest → R² = {rf_r2:.3f}, MAE = {rf_mae:.3f}  \nANN → R² = {ann_r2:.3f}, MAE = {ann_mae:.3f}"

    return fig, twin_fig, map_fig, ai_fig, esg_text, ai_text

# === GRADIO UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style='display:flex; align-items:center; gap:10px;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/7/7e/Pertamina_Logo.svg' width='70'>
        <h1 style='color:white;'>Digital Twin Project Performance Dashboard</h1>
    </div>
    <p><b>Dataset:</b> Real-time drilling data + Computed Petrophysical Output (CPO) log from Volve Field, North Sea (Well 15/9-F-15)</p>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            depth_min = gr.Slider(minimum=float(df["Depth"].min()), maximum=float(df["Depth"].max()), label="Min Depth")
            depth_max = gr.Slider(minimum=float(df["Depth"].min()), maximum=float(df["Depth"].max()), label="Max Depth")
            show_wob = gr.Checkbox(label="Tampilkan WOB", value=True)
            show_rpm = gr.Checkbox(label="Tampilkan RPM", value=True)
            show_energy = gr.Checkbox(label="Tampilkan Energy Intensity", value=True)
            show_sustain = gr.Checkbox(label="Tampilkan Sustainability Index", value=True)
            submit_btn = gr.Button("🚀 Generate Digital Twin", variant="primary")

        with gr.Column(scale=3):
            main_plot = gr.Plot(label="Grafik Kinerja Pemboran")
            twin_plot = gr.Plot(label="3D Digital Twin Layout (Animated)")
            map_plot = gr.Plot(label="GeoDigital Twin – Volve Field")
            ai_plot = gr.Plot(label="AI Prediction Model")
            esg_panel = gr.Markdown(label="ESG Summary")
            ai_panel = gr.Markdown(label="AI Model Summary")

    submit_btn.click(
        fn=digital_twin_dashboard,
        inputs=[depth_min, depth_max, show_wob, show_rpm, show_energy, show_sustain],
        outputs=[main_plot, twin_plot, map_plot, ai_plot, esg_panel, ai_panel]
    )

demo.launch()
