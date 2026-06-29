import os

file_path = "/Users/adithkb/MSAI/PersonalProjects/CustomerFootfallPrediction/dashboard/app.py"

with open(file_path, "r") as f:
    content = f.read()

# Change 1: page config
content = content.replace('initial_sidebar_state="collapsed"', 'initial_sidebar_state="expanded"')

# Change 2: CSS for sidebar
old_css = """[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"],
[data-testid="stHeader"] { display: none !important; }"""
new_css = """[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarNav"], [data-testid="stHeader"] { display: none !important; }"""
content = content.replace(old_css, new_css)

# Change 3: render_hourly_chart
old_chart = """def render_hourly_chart(forecast: pd.DataFrame):
    st.markdown('<div class="panel-title" style="margin-top:2rem; margin-bottom:0.5rem; border:none;">Hourly Breakdown</div>', unsafe_allow_html=True)
    hours = forecast["hour"].tolist()
    fig = go.Figure()

    shade = [
        (12, 15, "rgba(255,255,255,0.015)",  "LUNCH"),
        (17, 21, "rgba(255,255,255,0.025)",  "EVENING"),
        (21, 25, "rgba(255,255,255,0.015)",  "LATE BAR"),
    ]
    for x0, x1, colour, label in shade:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=colour, layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=8, annotation_font_color="#3F3F46",
            annotation_font_family="Inter, sans-serif",
        )

    fig.add_trace(go.Bar(
        x=hours, y=forecast["orders_count_xgb"],
        name="Orders", marker_color="#3B82F6", opacity=0.9,
        marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=forecast["food_tickets_count_xgb"],
        name="Food Tickets", mode="lines+markers",
        line=dict(color="#D97706", width=2), marker=dict(size=5, color="#D97706", line=dict(width=1, color="#0A0A0A")),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=forecast["orders_count_baseline"],
        name="Baseline", mode="lines",
        line=dict(color="#3F3F46", width=1.5, dash="dash"), opacity=0.7,
    ))
    fig.add_vline(
        x=21, line_width=1, line_dash="dash", line_color="#3F3F46",
        annotation_text="Kitchen close",
        annotation_position="top right",
        annotation_font_size=8, annotation_font_color="#71717A",
        annotation_font_family="Inter, sans-serif",
    )

    tick_vals = sorted(set(hours))
    # Simplify x-axis labels by showing fewer of them if it's crowded, but for 24 hours every 2 hours is good
    tick_text = [f"{h:02d}:00" if h % 2 == 0 else "" for h in tick_vals]

    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, x=0,
            font=dict(size=10, color="#A1A1AA", family="Inter, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text, title=None,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
            gridcolor="#141414", showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=None, gridcolor="#141414", zeroline=False,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)"""

new_chart = """def render_hourly_chart(forecast: pd.DataFrame):
    st.markdown('<div class="panel-title" style="margin-top:2rem; margin-bottom:0.5rem; border:none;">Hourly Breakdown</div>', unsafe_allow_html=True)
    
    hours_str = [f"{int(h):02d}:00" for h in forecast["hour"]]
    
    fig = go.Figure()

    shade = [
        ("12:00", "15:00", "rgba(255,255,255,0.015)",  "LUNCH"),
        ("17:00", "21:00", "rgba(255,255,255,0.025)",  "EVENING"),
        ("21:00", "01:00", "rgba(255,255,255,0.015)",  "LATE BAR"),
    ]
    for x0, x1, colour, label in shade:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=colour, layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=8, annotation_font_color="#3F3F46",
            annotation_font_family="Inter, sans-serif",
        )

    fig.add_trace(go.Bar(
        x=hours_str, y=forecast["orders_count_xgb"],
        name="Orders", marker_color="#3B82F6", opacity=0.9,
        marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=hours_str, y=forecast["food_tickets_count_xgb"],
        name="Food Tickets", mode="lines+markers",
        line=dict(color="#D97706", width=2), marker=dict(size=5, color="#D97706", line=dict(width=1, color="#0A0A0A")),
    ))
    
    baseline_vals = forecast["orders_count_baseline"].tolist()
    
    fig.add_trace(go.Scatter(
        x=hours_str, y=baseline_vals,
        name="Same Slot Last Week", mode="lines",
        line=dict(color="#3F3F46", width=1, dash="dash"), opacity=0.5,
        connectgaps=False
    ))
    
    fig.add_vline(
        x="21:00", line_width=1, line_dash="dash", line_color="#3F3F46",
        annotation_text="Kitchen close",
        annotation_position="top right",
        annotation_font_size=8, annotation_font_color="#71717A",
        annotation_font_family="Inter, sans-serif",
    )

    tick_vals = hours_str
    tick_text = [h if int(h.split(":")[0]) % 2 == 0 else "" for h in tick_vals]

    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, x=0,
            font=dict(size=10, color="#A1A1AA", family="Inter, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=hours_str,
            tickvals=tick_vals, ticktext=tick_text, title=None,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
            gridcolor="#141414", showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=None, gridcolor="#141414", zeroline=False,
            tickfont=dict(size=9, color="#71717A", family="JetBrains Mono, monospace"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)"""
content = content.replace(old_chart, new_chart)

# Change 4: Sidebar controls
old_controls = """    # ── Controls ─────────────────────────────────────────────────────────────
    with st.expander("Overrides & Settings", expanded=False):
        c_date, c_flags, c_wx, c_data = st.columns([1.5, 3, 2, 1.5])
        with c_date:
            sel_date = st.date_input("Forecast Date", value=default_date, min_value=available[0], max_value=available[-1])
            forecast_date = pd.Timestamp(sel_date)
        with c_flags:
            st.write("Overrides")
            f1, f2 = st.columns(2)
            with f1:
                live_music    = st.toggle("Live music",      value=True)
                special_event = st.toggle("Special event",   value=False)
                major_sports  = st.toggle("Sports event",    value=False)
            with f2:
                cruise  = st.toggle("Cruise ship",       value=False)
                st_pats = st.toggle("St. Patrick's week", value=False)
        with c_wx:
            st.write("Weather")
            rain = st.slider("Rain (mm)",        0.0, 20.0, 1.0, 0.5)
            temp = st.slider("Temperature (°C)", 0.0, 25.0, 12.0, 0.5)
        with c_data:
            st.write("Data Sync")
            if st.button("Refresh Live Data", use_container_width=True):
                with st.spinner("Fetching..."):
                    subprocess.run([sys.executable, "src/fetch_public_data.py"], capture_output=True, cwd=Path(__file__).parent.parent)
                    subprocess.run([sys.executable, "src/fetch_footfall.py"], capture_output=True, cwd=Path(__file__).parent.parent)
                    subprocess.run([sys.executable, "-c", "from src.features import build_features; import pandas as pd; df=build_features(pd.read_csv('data/synthetic/odonoghues_hourly.csv')); df.to_parquet('data/processed/features.parquet', index=False)"], capture_output=True, cwd=Path(__file__).parent.parent)
                    st.cache_data.clear()
                    st.rerun()"""

new_controls = """    # ── Sidebar Controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h3 style='color: var(--text-pri); margin-bottom: 1rem; font-weight: 600; font-size: 1.1rem;'>Settings & Overrides</h3>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Forecast Date</div>", unsafe_allow_html=True)
        sel_date = st.date_input("Forecast Date", value=default_date, min_value=available[0], max_value=available[-1], label_visibility="collapsed")
        forecast_date = pd.Timestamp(sel_date)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Overrides</div>", unsafe_allow_html=True)
        live_music    = st.toggle("Live music",      value=True)
        special_event = st.toggle("Special event",   value=False)
        major_sports  = st.toggle("Sports event",    value=False)
        cruise        = st.toggle("Cruise ship",     value=False)
        st_pats       = st.toggle("St. Patrick's week", value=False)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Weather</div>", unsafe_allow_html=True)
        rain = st.slider("Rain (mm)",        0.0, 20.0, 1.0, 0.5)
        temp = st.slider("Temperature (°C)", 0.0, 25.0, 12.0, 0.5)
        
        st.markdown("<hr style='margin: 1.5rem 0; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.8rem; color: var(--text-sec); margin-bottom: 0.5rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Data Sync</div>", unsafe_allow_html=True)
        if st.button("Refresh Live Data", use_container_width=True):
            with st.spinner("Fetching..."):
                subprocess.run([sys.executable, "src/fetch_public_data.py"], capture_output=True, cwd=Path(__file__).parent.parent)
                subprocess.run([sys.executable, "src/fetch_footfall.py"], capture_output=True, cwd=Path(__file__).parent.parent)
                subprocess.run([sys.executable, "-c", "from src.features import build_features; import pandas as pd; df=build_features(pd.read_csv('data/synthetic/odonoghues_hourly.csv')); df.to_parquet('data/processed/features.parquet', index=False)"], capture_output=True, cwd=Path(__file__).parent.parent)
                st.cache_data.clear()
                st.rerun()"""
content = content.replace(old_controls, new_controls)

with open(file_path, "w") as f:
    f.write(content)

print("Updates applied successfully.")
