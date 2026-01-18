import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mast Cell Analytics Suite", layout="wide", page_icon="🧬")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import date
import base64

# ==========================================
#        SHARED MATH ENGINE (The Core)
# ==========================================

def four_param_logistic(x, min_val, max_val, ec50, hill_slope):
    return min_val + (max_val - min_val) / (1 + (x / ec50)**(-hill_slope))

def get_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_metrics(doses, responses):
    try:
        # 1. SCRUBBER: Handle commas and % signs
        if isinstance(doses, pd.Series) and doses.dtype == object:
             doses = doses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
        if isinstance(responses, pd.Series) and responses.dtype == object:
             responses = responses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)

        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        mask = ~np.isnan(doses) & ~np.isnan(responses)
        x_clean = doses[mask]
        y_clean = responses[mask]

        if len(y_clean) < 4: return None, None, None, None, None, "Not enough data"

        # 2. AUTO-SCALE: If max <= 1.0, convert to %
        if max(y_clean) <= 1.0: y_clean = y_clean * 100

        # 3. FIT
        p0 = [min(y_clean), max(y_clean), np.median(x_clean), 1]
        popt, _ = curve_fit(four_param_logistic, x_clean, y_clean, p0, maxfev=10000)
        
        # 4. METRICS
        min_val, max_val, ec50, hill_slope = popt
        ec90 = ec50 * ((90 / 10) ** (1 / abs(hill_slope)))
        ec25 = ec50 * ((25 / 75) ** (1 / abs(hill_slope)))
        
        y_pred = four_param_logistic(x_clean, *popt)
        r2 = get_r_squared(y_clean, y_pred)
        
        status = "OK"
        if r2 < 0.9: status = "⚠️ Poor Fit"

        return popt, ec25, ec50, ec90, r2, status
    except:
        return None, None, None, None, None, "Fit Failed"

# ==========================================
#        APP NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Mode Selector")
app_mode = st.sidebar.radio("Choose Analysis Type:", 
    ["Standardized Protocol (IgE/SP)", "Custom Experiment (Flexible)"])

st.sidebar.divider()

# ==========================================
#   MODE 1: STANDARDIZED PROTOCOL (The LIMS)
# ==========================================
if app_mode == "Standardized Protocol (IgE/SP)":
    
    st.title("🧬 Mast Cell Multi-Donor Analyzer")

    # --- METADATA ---
    with st.expander("📝 Experiment Setup", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: test_date = st.date_input("Date", date.today())
        with c2: raw_link = st.text_input("Raw Data Link", "http://...")
        with c3: num_donors = st.number_input("How many Donors?", 1, 10, 1)

        st.write("**Define Donors:**")
        donors = []
        cols = st.columns(min(num_donors, 5)) # Show max 5 columns for layout
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i in range(num_donors):
            col_idx = i % 5
            with cols[col_idx]:
                name = st.text_input(f"Donor {i+1} Name", f"Donor_{i+1}")
                donors.append({"name": name, "color": colors[i], "ige_cols": [], "sp_cols": []})

    # --- UPLOAD ---
    uploaded_file = st.file_uploader("Upload Standardized Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        col_ige_dose = st.sidebar.text_input("IgE Dose Col", "Dose_IgE")
        col_sp_dose = st.sidebar.text_input("SP Dose Col", "Dose_SP")

        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()

            # --- ASSIGN COLUMNS ---
            st.info("👇 Assign columns to each donor")
            available_cols = [c for c in df.columns if c not in [col_ige_dose, col_sp_dose]]
            
            for d in donors:
                with st.container():
                    st.markdown(f"**👤 {d['name']}**")
                    ca, cb = st.columns(2)
                    d['ige_cols'] = ca.multiselect(f"Anti-IgE Samples ({d['name']})", available_cols, key=f"ige_{d['name']}")
                    d['sp_cols'] = cb.multiselect(f"SP Samples ({d['name']})", available_cols, key=f"sp_{d['name']}")

            if st.button("🚀 Run Standard Analysis"):
                
                # Helper to plot standardize mode
                def plot_std_category(df, dose_col, donor_list, cat_name, unit):
                    fig = go.Figure()
                    res = []
                    for d in donor_list:
                        # Decide which cols to use based on category
                        target_cols = d['ige_cols'] if cat_name == "Anti-IgE" else d['sp_cols']
                        
                        if dose_col not in df.columns: continue

                        doses = df[dose_col]
                        for col in target_cols:
                            resp = df[col]
                            popt, ec25, ec50, ec90, r2, status = calculate_metrics(doses, resp)
                            
                            # --- FIX IS HERE: Check for None explicitly ---
                            if popt is not None:
                                res.append({"Donor": d['name'], "Sample": col, "EC50": ec50, "EC90": ec90, "Max": popt[1], "R²": r2})
                                
                                # Plot
                                d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                                r_plot = pd.to_numeric(resp.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                                mask = ~np.isnan(d_plot) & ~np.isnan(r_plot)
                                y_plot = r_plot[mask]
                                if max(y_plot) <= 1.0: y_plot = y_plot * 100
                                
                                # Points
                                fig.add_trace(go.Scatter(x=d_plot[mask], y=y_plot, mode='markers', marker=dict(color=d['color']), showlegend=False))
                                # Line
                                x_smooth = np.logspace(np.log10(min(d_plot[mask])+1e-9), np.log10(max(d_plot[mask])), 100)
                                y_smooth = four_param_logistic(x_smooth, *popt)
                                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f"{d['name']} {col}", line=dict(color=d['color'])))
                    
                    fig.update_layout(title=f"{cat_name}", xaxis_title=f"Dose ({unit})", yaxis_title="Degranulation %", xaxis_type="log", height=450)
                    return pd.DataFrame(res), fig

                # Generate
                st.markdown("### 📊 Results")
                r_ige, f_ige = plot_std_category(df, col_ige_dose, donors, "Anti-IgE", "µg/mL")
                r_sp, f_sp = plot_std_category(df, col_sp_dose, donors, "SP", "µM")

                c1, c2 = st.columns(2)
                with c1: 
                    st.plotly_chart(f_ige, use_container_width=True)
                    st.dataframe(r_ige)
                with c2: 
                    st.plotly_chart(f_sp, use_container_width=True)
                    st.dataframe(r_sp)
                
                # HTML Export
                st.success("Analysis Complete. Download report below.")
                html = f"<html><body><h1>Mast Cell Report ({test_date})</h1><h2>Anti-IgE</h2>{r_ige.to_html()}{f_ige.to_html(full_html=False, include_plotlyjs='cdn')}<h2>SP</h2>{r_sp.to_html()}{f_sp.to_html(full_html=False, include_plotlyjs='cdn')}</body></html>"
                b64 = base64.b64encode(html.encode()).decode()
                st.markdown(f'<a href="data:text/html;base64,{b64}" download="Report.html">📥 Download Interactive Report</a>', unsafe_allow_html=True)

        except Exception as e: st.error(f"Error: {e}")

# ==========================================
#   MODE 2: CUSTOM EXPERIMENT (Flexible)
# ==========================================
elif app_mode == "Custom Experiment (Flexible)":
    
    st.title("🧪 Custom Dose-Response Playground")
    st.markdown("Flexible mode: Upload any file, pick any columns.")

    custom_file = st.file_uploader("Upload Any Data", type=['csv', 'xlsx'], key="custom")
    
    if custom_file:
        try:
            if custom_file.name.endswith('.csv'): df_c = pd.read_csv(custom_file)
            else: df_c = pd.read_excel(custom_file)
            
            # 1. Select X-Axis
            st.sidebar.subheader("Custom Settings")
            dose_col = st.sidebar.selectbox("Which column is the Dose (X-axis)?", df_c.columns)
            unit_label = st.sidebar.text_input("X-Axis Unit Label", "ng/mL")
            
            # 2. Select Y-Axis
            st.subheader("Select Samples to Analyze")
            # Default to all columns except dose
            available = [c for c in df_c.columns if c != dose_col]
            selected_samples = st.multiselect("Select Y-columns", available, default=available[:2])

            if selected_samples:
                results_c = []
                fig_c = go.Figure()
                doses = df_c[dose_col]

                for sample in selected_samples:
                    responses = df_c[sample]
                    popt, ec25, ec50, ec90, r2, status = calculate_metrics(doses, responses)
                    
                    # --- FIX IS HERE: Check for None explicitly ---
                    if popt is not None:
                        results_c.append({
                            "Sample": sample, "EC50": ec50, "EC90": ec90, 
                            "Max Response": popt[1], "R²": r2, "Status": status
                        })
                        
                        # Plot Logic
                        d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                        r_plot = pd.to_numeric(responses.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                        mask = ~np.isnan(d_plot) & ~np.isnan(r_plot)
                        y_plot = r_plot[mask]
                        if max(y_plot) <= 1.0: y_plot = y_plot * 100

                        fig_c.add_trace(go.Scatter(x=d_plot[mask], y=y_plot, mode='markers', name=sample))
                        
                        x_smooth = np.logspace(np.log10(min(d_plot[mask])+1e-9), np.log10(max(d_plot[mask])), 100)
                        y_smooth = four_param_logistic(x_smooth, *popt)
                        fig_c.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f"{sample} Fit"))
                    else:
                        results_c.append({"Sample": sample, "Status": status})

                # Display
                c_tbl, c_plt = st.columns([1, 2])
                with c_tbl:
                    st.dataframe(pd.DataFrame(results_c))
                with c_plt:
                    fig_c.update_layout(xaxis_title=f"Dose ({unit_label})", yaxis_title="Response %", xaxis_type="log", height=500)
                    st.plotly_chart(fig_c, use_container_width=True)

        except Exception as e: st.error(f"Error reading file: {e}")