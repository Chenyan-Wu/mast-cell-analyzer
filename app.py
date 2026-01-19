import streamlit as st
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mast Cell Analytics Suite", layout="wide", page_icon="🧬")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import date
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
#        SHARED MATH ENGINE (Bio-Stabilized)
# ==========================================

def four_param_logistic(x, min_val, max_val, log_ec50, hill_slope):
    """Log-Linear 4PL Model."""
    return min_val + (max_val - min_val) / (1 + 10**((log_ec50 - x) * hill_slope))

def get_r_squared(y_true, y_pred):
    try:
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0: return 0 
        return 1 - (ss_res / ss_tot)
    except:
        return 0

def calculate_metrics(doses, responses):
    """Hybrid Fit: Fixed Bottom (0), Floating Top + QC Logic."""
    try:
        if isinstance(doses, pd.Series) and doses.dtype == object:
             doses = doses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
        if isinstance(responses, pd.Series) and responses.dtype == object:
             responses = responses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)

        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        mask = (doses > 0) & ~np.isnan(doses) & ~np.isnan(responses)
        x_raw = doses[mask]
        y_clean = responses[mask]
        
        if len(y_clean) < 4: return None, None, None, None, None, "Not enough data"
        if max(y_clean) <= 1.0: y_clean = y_clean * 100
        x_log = np.log10(x_raw)
        
        min_log = min(x_log)
        max_log = max(x_log)
        
        bounds = (
            [-0.001,        max(y_clean),  min_log - 1.0,  0.1], 
            [ 0.001,        150,           max_log + 1.0,  10.0] 
        )
        
        p0 = [0, max(y_clean), np.median(x_log), 1.0]

        popt, _ = curve_fit(four_param_logistic, x_log, y_clean, p0, bounds=bounds, maxfev=10000)
        
        min_val, max_val, log_ec50, hill_slope = popt
        
        ec50 = 10**log_ec50
        ec90 = 10**(log_ec50 + (1/hill_slope)*np.log10(90/10))
        ec25 = 10**(log_ec50 + (1/hill_slope)*np.log10(25/75))
        
        y_pred = four_param_logistic(x_log, *popt)
        r2 = get_r_squared(y_clean, y_pred)
        
        # --- STATUS / QC LOGIC ---
        if r2 < 0.9:
            status = "⚠️ Poor Fit"
        elif max_val < 25.0:
            status = "⚠️ Low (<25%)"
        else:
            status = "✅ Pass"

        return popt, ec25, ec50, ec90, r2, status
    except Exception as e:
        return None, None, None, None, None, f"Fit Failed"

# ==========================================
#        GOOGLE DRIVE CONNECTOR
# ==========================================
def save_to_google_sheet(df, sheet_name="MastCell_DB"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        if "service_account_info" in st.secrets:
            json_text = st.secrets["service_account_info"]
            creds_dict = json.loads(json_text)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
            client = gspread.authorize(creds)

        try:
            sheet = client.open(sheet_name).sheet1
        except:
            return False, f"Could not find Sheet '{sheet_name}'. Did you share it with the bot?"

        existing_data = sheet.get_all_values()
        if not existing_data:
            sheet.append_row(df.columns.tolist())
        
        data_to_upload = df.values.tolist()
        sheet.append_rows(data_to_upload)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# ==========================================
#        APP NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Mode Selector")
app_mode = st.sidebar.radio("Choose Analysis Type:", 
    ["Standardized Protocol (IgE/SP)", "Custom Experiment (Flexible)"])

st.sidebar.divider()

# ==========================================
#   MODE 1: STANDARDIZED PROTOCOL
# ==========================================
if app_mode == "Standardized Protocol (IgE/SP)":
    st.title("🧬 Mast Cell Multi-Donor Analyzer")

    with st.expander("📝 Experiment Setup", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: test_date = st.date_input("Date", date.today())
        with c2: raw_link = st.text_input("Raw Data Link", "http://...")
        with c3: num_donors = st.number_input("How many Donors?", 1, 10, 1)

        donors = []
        cols = st.columns(min(num_donors, 5)) 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i in range(num_donors):
            col_idx = i % 5
            with cols[col_idx]:
                name = st.text_input(f"Donor {i+1} Name", f"Donor_{i+1}")
                donors.append({"name": name, "color": colors[i], "ige_cols": [], "sp_cols": []})

    # --- TEMPLATE & UPLOAD ---
    st.write("---")
    
    # 1. Template Download
    template_csv = "Dose_IgE,Dose_SP,IgE_Sample_1,IgE_Sample_2,SP_Sample_1,SP_Sample_2\n1.0,10.0,45.5,50.1,15.2,18.5\n0.5,5.0,38.2,42.5,12.0,14.5\n0.1,1.0,15.0,18.2,5.5,6.0\n0.01,0.1,2.0,5.0,0.5,1.0"
    st.download_button(
        label="📥 Download Template CSV",
        data=template_csv,
        file_name="mast_cell_template.csv",
        mime="text/csv",
        help="Click to download a sample file structure."
    )

    # 2. File Upload
    uploaded_file = st.file_uploader("Upload Standardized Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        col_ige_dose = st.sidebar.text_input("IgE Dose Col", "Dose_IgE")
        col_sp_dose = st.sidebar.text_input("SP Dose Col", "Dose_SP")

        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()

            st.info("👇 Assign columns to each donor")
            available_cols = [c for c in df.columns if c not in [col_ige_dose, col_sp_dose]]
            
            for d in donors:
                with st.container():
                    st.markdown(f"**👤 {d['name']}**")
                    ca, cb = st.columns(2)
                    d['ige_cols'] = ca.multiselect(f"Anti-IgE Samples ({d['name']})", available_cols, key=f"ige_{d['name']}")
                    d['sp_cols'] = cb.multiselect(f"SP Samples ({d['name']})", available_cols, key=f"sp_{d['name']}")

            # --- PLOTTING FUNCTION (Dual Plot) ---
            def plot_std_category(df, dose_col, donor_list, cat_name, unit):
                fig_log = go.Figure()
                fig_lin = go.Figure()
                res = []
                
                for d in donor_list:
                    target_cols = d['ige_cols'] if cat_name == "Anti-IgE" else d['sp_cols']
                    if dose_col not in df.columns: continue

                    doses = df[dose_col]
                    show_legend_for_donor = True
                    
                    for col in target_cols:
                        resp = df[col]
                        popt, ec25, ec50, ec90, r2, status = calculate_metrics(doses, resp)
                        
                        if popt is not None:
                            res.append({
                                "Date": str(test_date), "Donor": d['name'], 
                                "Stimulant": cat_name, "Sample": col, 
                                "EC25": ec25, "EC50": ec50, "EC90": ec90, 
                                "Max": popt[1], "R²": r2, "Status": status
                            })
                            
                            d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                            r_plot = pd.to_numeric(resp.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                            mask = ~np.isnan(d_plot) & ~np.isnan(r_plot) & (d_plot > 0)
                            
                            x_plot_raw = d_plot[mask]
                            y_plot = r_plot[mask]
                            if max(y_plot) <= 1.0: y_plot = y_plot * 100
                            
                            # TRACES FOR LOG PLOT
                            fig_log.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', marker=dict(color=d['color']), showlegend=False))
                            
                            # TRACES FOR LINEAR PLOT
                            fig_lin.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', marker=dict(color=d['color']), showlegend=False))
                            
                            # FIT LINES
                            # Log-space smooth for Log Plot
                            x_smooth_log = np.logspace(np.log10(min(x_plot_raw)), np.log10(max(x_plot_raw)), 100)
                            y_smooth_log = four_param_logistic(np.log10(x_smooth_log), *popt)
                            
                            # Lin-space smooth for Linear Plot (visual looks better with linspace)
                            x_smooth_lin = np.linspace(min(x_plot_raw), max(x_plot_raw), 100)
                            y_smooth_lin = four_param_logistic(np.log10(x_smooth_lin), *popt)
                            
                            legend_name = f"{d['name']} {cat_name}"
                            
                            # Add Lines to both figures
                            fig_log.add_trace(go.Scatter(
                                x=x_smooth_log, y=y_smooth_log, mode='lines', name=legend_name, 
                                line=dict(color=d['color']), showlegend=show_legend_for_donor, legendgroup=d['name']
                            ))
                            
                            fig_lin.add_trace(go.Scatter(
                                x=x_smooth_lin, y=y_smooth_lin, mode='lines', name=legend_name, 
                                line=dict(color=d['color']), showlegend=False, legendgroup=d['name'] # No legend on bottom plot to save space
                            ))
                            
                            show_legend_for_donor = False
                
                # Update Layouts
                fig_log.update_layout(title=f"{cat_name} (Log Scale)", xaxis_title=f"Dose ({unit})", yaxis_title="Degranulation %", xaxis_type="log", height=450)
                fig_lin.update_layout(title=f"{cat_name} (Linear Scale)", xaxis_title=f"Dose ({unit})", yaxis_title="Degranulation %", xaxis_type="linear", height=450)
                
                return pd.DataFrame(res), fig_log, fig_lin

            # --- RUN BUTTON ---
            if st.button("🚀 Run Standard Analysis"):
                r_ige, f_ige_log, f_ige_lin = plot_std_category(df, col_ige_dose, donors, "Anti-IgE", "µg/mL")
                r_sp, f_sp_log, f_sp_lin = plot_std_category(df, col_sp_dose, donors, "SP", "µM")
                
                st.session_state['std_results'] = {
                    'r_ige': r_ige, 'f_ige_log': f_ige_log, 'f_ige_lin': f_ige_lin,
                    'r_sp': r_sp, 'f_sp_log': f_sp_log, 'f_sp_lin': f_sp_lin,
                    'raw_link': raw_link, 'test_date': test_date
                }

            # --- DISPLAY & EXPORT ---
            if 'std_results' in st.session_state:
                res = st.session_state['std_results']
                
                st.markdown("### 📊 Results")
                
                # --- IGE SECTION ---
                st.subheader("Anti-IgE Results")
                c1, c2 = st.columns(2)
                with c1: 
                    st.plotly_chart(res['f_ige_log'], use_container_width=True) # Top (Log)
                    st.plotly_chart(res['f_ige_lin'], use_container_width=True) # Bottom (Linear)
                with c2: 
                    st.dataframe(res['r_ige'], height=600)

                st.divider()

                # --- SP SECTION ---
                st.subheader("SP Results")
                c3, c4 = st.columns(2)
                with c3: 
                    st.plotly_chart(res['f_sp_log'], use_container_width=True) # Top (Log)
                    st.plotly_chart(res['f_sp_lin'], use_container_width=True) # Bottom (Linear)
                with c4: 
                    st.dataframe(res['r_sp'], height=600)

                # HTML Report (Side-by-Side Flexbox)
                html = f"""
                <html>
                <head>
                    <title>Mast Cell Report</title>
                    <style>
                        .plot-row {{ display: flex; flex-direction: row; width: 100%; }}
                        .plot-col {{ width: 50%; padding: 5px; }}
                    </style>
                </head>
                <body>
                    <h1>Mast Cell Report ({res['test_date']})</h1>
                    
                    <h2>Anti-IgE</h2>
                    {res['r_ige'].to_html()}
                    <div class="plot-row">
                        <div class="plot-col">{res['f_ige_log'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        <div class="plot-col">{res['f_ige_lin'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    </div>
                    
                    <hr>
                    
                    <h2>SP</h2>
                    {res['r_sp'].to_html()}
                    <div class="plot-row">
                        <div class="plot-col">{res['f_sp_log'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        <div class="plot-col">{res['f_sp_lin'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    </div>
                </body>
                </html>
                """
                b64 = base64.b64encode(html.encode()).decode()
                st.markdown(f'<a href="data:text/html;base64,{b64}" download="Mast_Cell_Report_{res["test_date"]}.html" style="text-decoration:none;">📥 <b>Download Interactive HTML Report</b></a>', unsafe_allow_html=True)

                st.divider()
                st.subheader("☁️ Database")
                if st.button("💾 Save Results to Google Drive"):
                    full_db = pd.concat([res['r_ige'], res['r_sp']], ignore_index=True)
                    full_db['Raw_Link'] = res['raw_link']
                    
                    success, msg = save_to_google_sheet(full_db, "MastCell_DB")
                    if success:
                        st.success(f"✅ Saved {len(full_db)} rows to Google Drive!")
                    else:
                        st.error(f"❌ Error: {msg}")

        except Exception as e: st.error(f"Error: {e}")

# ==========================================
#   MODE 2: CUSTOM EXPERIMENT
# ==========================================
elif app_mode == "Custom Experiment (Flexible)":
    st.title("🧪 Custom Dose-Response Playground")
    custom_file = st.file_uploader("Upload Any Data", type=['csv', 'xlsx'], key="custom")
    
    if custom_file:
        try:
            if custom_file.name.endswith('.csv'): df_c = pd.read_csv(custom_file)
            else: df_c = pd.read_excel(custom_file)
            
            st.sidebar.subheader("Custom Settings")
            dose_col = st.sidebar.selectbox("Which column is the Dose (X-axis)?", df_c.columns)
            unit_label = st.sidebar.text_input("X-Axis Unit Label", "ng/mL")
            
            st.subheader("Select Samples to Analyze")
            available = [c for c in df_c.columns if c != dose_col]
            selected_samples = st.multiselect("Select Y-columns", available, default=available[:2])

            if selected_samples:
                if st.button("Run Custom Analysis") or ('custom_results' in st.session_state):
                    
                    results_c = []
                    fig_log = go.Figure()
                    fig_lin = go.Figure()
                    doses = df_c[dose_col]

                    for sample in selected_samples:
                        responses = df_c[sample]
                        popt, ec25, ec50, ec90, r2, status = calculate_metrics(doses, responses)
                        
                        if popt is not None:
                            results_c.append({
                                "Date": str(date.today()), "Sample": sample, 
                                "EC25": ec25, "EC50": ec50, "EC90": ec90, 
                                "Max Response": popt[1], "R²": r2, "Status": status
                            })
                            
                            d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                            r_plot = pd.to_numeric(responses.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                            mask = ~np.isnan(d_plot) & ~np.isnan(r_plot) & (d_plot > 0)
                            
                            x_plot_raw = d_plot[mask]
                            y_plot = r_plot[mask]
                            if max(y_plot) <= 1.0: y_plot = y_plot * 100

                            # Add to both figures
                            fig_log.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', name=sample))
                            fig_lin.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', name=sample))
                            
                            # Lines
                            x_smooth_log = np.logspace(np.log10(min(x_plot_raw)), np.log10(max(x_plot_raw)), 100)
                            y_smooth_log = four_param_logistic(np.log10(x_smooth_log), *popt)
                            fig_log.add_trace(go.Scatter(x=x_smooth_log, y=y_smooth_log, mode='lines', name=f"{sample} Fit"))
                            
                            x_smooth_lin = np.linspace(min(x_plot_raw), max(x_plot_raw), 100)
                            y_smooth_lin = four_param_logistic(np.log10(x_smooth_lin), *popt)
                            fig_lin.add_trace(go.Scatter(x=x_smooth_lin, y=y_smooth_lin, mode='lines', name=f"{sample} Fit"))

                        else:
                            results_c.append({"Sample": sample, "Status": status})
                    
                    res_df_c = pd.DataFrame(results_c)
                    
                    # Store
                    st.session_state['custom_results'] = {'df': res_df_c, 'fig_log': fig_log, 'fig_lin': fig_lin}

                    # Display
                    c_tbl, c_plt = st.columns([1, 2])
                    with c_tbl: st.dataframe(res_df_c)
                    with c_plt:
                        fig_log.update_layout(title="Log Scale", xaxis_title=f"Dose ({unit_label})", yaxis_title="Response %", xaxis_type="log", height=400)
                        st.plotly_chart(fig_log, use_container_width=True)
                        
                        fig_lin.update_layout(title="Linear Scale", xaxis_title=f"Dose ({unit_label})", yaxis_title="Response %", xaxis_type="linear", height=400)
                        st.plotly_chart(fig_lin, use_container_width=True)

                    # Custom HTML Report (Side-by-Side)
                    html_c = f"""
                    <html>
                    <head><style>.row {{ display: flex; }} .col {{ width: 50%; }}</style></head>
                    <body>
                        <h1>Custom Analysis Report</h1>
                        {res_df_c.to_html()}
                        <div class="row">
                            <div class="col">{fig_log.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                            <div class="col">{fig_lin.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        </div>
                    </body>
                    </html>
                    """
                    b64_c = base64.b64encode(html_c.encode()).decode()
                    st.markdown(f'<a href="data:text/html;base64,{b64_c}" download="Custom_Report.html">📥 <b>Download Custom HTML Report</b></a>', unsafe_allow_html=True)

                    st.divider()
                    if st.button("💾 Save Custom Results to Drive"):
                        success, msg = save_to_google_sheet(res_df_c, "MastCell_DB")
                        if success: st.success(f"✅ Saved to Google Drive!")
                        else: st.error(f"❌ Error: {msg}")

        except Exception as e: st.error(f"Error reading file: {e}")