import streamlit as st

# --- PAGE SETUP ---
st.set_page_config(page_title="Mast Cell Dual-Assay Analyzer", layout="wide")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import date

# --- 1. ROBUST MATH ENGINE ---
def four_param_logistic(x, min_val, max_val, ec50, hill_slope):
    # Hill equation
    return min_val + (max_val - min_val) / (1 + (x / ec50)**(-hill_slope))

def get_r_squared(y_true, y_pred):
    # Calculate R2
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_metrics(doses, responses):
    try:
        # DATA CLEANING: Handle European commas (e.g. "0,5" -> 0.5)
        if isinstance(doses, pd.Series) and doses.dtype == object:
             doses = doses.astype(str).str.replace(',', '.', regex=False)
        if isinstance(responses, pd.Series) and responses.dtype == object:
             responses = responses.astype(str).str.replace(',', '.', regex=False)

        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        # Remove NaNs
        mask = ~np.isnan(doses) & ~np.isnan(responses)
        x_clean = doses[mask]
        y_clean = responses[mask]

        if len(y_clean) < 4: return None, None, None, None, None, "Not enough data"

        # FIT THE CURVE
        # Initial guesses [min, max, ec50, slope]
        p0 = [min(y_clean), max(y_clean), np.median(x_clean), 1]
        
        popt, _ = curve_fit(four_param_logistic, x_clean, y_clean, p0, maxfev=10000)
        min_val, max_val, ec50, hill_slope = popt
        
        # CALCULATE METRICS
        # EC90 = EC50 * ((90/10)^(1/slope))
        ec90 = ec50 * ((90 / 10) ** (1 / abs(hill_slope)))
        # EC25 = EC50 * ((25/75)^(1/slope))
        ec25 = ec50 * ((25 / 75) ** (1 / abs(hill_slope)))
        
        # R-SQUARED
        y_pred = four_param_logistic(x_clean, *popt)
        r2 = get_r_squared(y_clean, y_pred)
        
        status = "OK"
        if r2 < 0.9: status = "⚠️ Poor Fit"

        return popt, ec25, ec50, ec90, r2, status
        
    except Exception as e:
        return None, None, None, None, None, "Fit Failed"

def generate_panel(df, dose_col, sample_cols, color_hex, title, unit):
    """Reusable function to generate Left or Right panel"""
    st.markdown(f"### {title}")
    
    if dose_col not in df.columns:
        st.error(f"Missing column: {dose_col}")
        return
    
    results = []
    fig = go.Figure()
    doses = df[dose_col]

    for col in sample_cols:
        responses = df[col]
        popt, ec25, ec50, ec90, r2, status = calculate_metrics(doses, responses)
        
        if popt is not None:
            # Append Results
            results.append({
                "Sample": col,
                "EC25": ec25,
                "EC50": ec50,
                "EC90": ec90,
                "R²": r2,
                "Max": popt[1],
                "Status": status
            })
            
            # Add to Plot
            # Clean data for plotting
            d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
            r_plot = pd.to_numeric(responses.astype(str).str.replace(',', '.'), errors='coerce')
            mask = ~np.isnan(d_plot) & ~np.isnan(r_plot)
            
            # Raw dots
            fig.add_trace(go.Scatter(
                x=d_plot[mask], y=r_plot[mask], 
                mode='markers', name=f'{col}',
                marker=dict(color=color_hex)
            ))
            
            # Fitted Line
            x_min, x_max = min(d_plot[mask]), max(d_plot[mask])
            if x_min <= 0: x_min = 1e-9
            x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            y_smooth = four_param_logistic(x_smooth, *popt)
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth, 
                mode='lines', name=f'{col} Fit',
                line=dict(color=color_hex, width=1)
            ))
        else:
             results.append({"Sample": col, "Status": status})

    # 1. Show Table
    if results:
        res_df = pd.DataFrame(results)
        # Apply conditional formatting for Bad R2
        st.dataframe(
            res_df.style.applymap(lambda v: 'color: red; font-weight: bold;' if v == "⚠️ Poor Fit" else '', subset=['Status'])
                  .format({"EC25": "{:.4f}", "EC50": "{:.4f}", "EC90": "{:.4f}", "R²": "{:.3f}", "Max": "{:.1f}"})
        )
    
    # 2. Show Plot
    fig.update_layout(
        title=f"{title} (Log Scale)",
        xaxis_title=f"Dose ({unit})", yaxis_title="% Degranulation",
        xaxis_type="log", height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    return pd.DataFrame(results)

# --- 2. THE APP UI ---

# HEADER
st.title("🧬 Mast Cell Dual-Assay Analyzer")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    test_date = st.date_input("Date of Test", date.today())
with c2:
    donor_id = st.text_input("Donor / Experiment ID", "Donor_001")

st.divider()

# SIDEBAR CONFIG
with st.sidebar:
    st.header("1. Template")
    st.info("The template now includes TWO dose columns: one for IgE and one for SP.")
    
    # Generate Standardized Template
    # IgE: 9 points, SP: 10 points. We pad IgE with None.
    ige_doses = [1.0, 0.5, 0.1, 0.05, 0.01, 0.0075, 0.005, 0.0025, 0.001, None]
    sp_doses  = [3.5, 2.5, 1.5, 1.0, 0.75, 0.5, 0.3, 0.15, 0.075, 0.05]
    
    template_data = {
        "Dose_IgE": ige_doses,
        "Dose_SP": sp_doses,
        "IgE_Sample_1": [None]*10,
        "IgE_Sample_2": [None]*10,
        "SP_Sample_1": [None]*10,
        "SP_Sample_2": [None]*10
    }
    df_temp = pd.DataFrame(template_data)
    csv_temp = df_temp.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Standard Template", csv_temp, "mast_cell_template.csv", "text/csv")

    st.header("2. Settings")
    st.write("Confirm standardized columns:")
    col_ige_dose = st.text_input("IgE Dose Column Name", "Dose_IgE")
    col_sp_dose = st.text_input("SP Dose Column Name", "Dose_SP")

# MAIN UPLOAD
uploaded_file = st.file_uploader("Upload Completed Data (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Strip whitespace from headers
        df.columns = df.columns.str.strip()
        
        # --- COLUMN SELECTION ---
        st.subheader("Select Samples")
        col_select_1, col_select_2 = st.columns(2)
        
        all_cols = [c for c in df.columns if c not in [col_ige_dose, col_sp_dose]]
        
        with col_select_1:
            # Try to auto-detect columns containing "IgE"
            default_ige = [c for c in all_cols if "ige" in c.lower()]
            ige_samples = st.multiselect("Select Anti-IgE Samples", all_cols, default=default_ige)
            
        with col_select_2:
            # Try to auto-detect columns containing "SP"
            default_sp = [c for c in all_cols if "sp" in c.lower()]
            sp_samples = st.multiselect("Select SP Samples", all_cols, default=default_sp)

        st.divider()

        # --- DUAL PANELS ---
        left_panel, right_panel = st.columns(2)

        with left_panel:
            res_ige = generate_panel(df, col_ige_dose, ige_samples, "#1f77b4", "Anti-IgE (µg/mL)", "µg/mL")

        with right_panel:
            res_sp = generate_panel(df, col_sp_dose, sp_samples, "#d62728", "SP (µM)", "µM")

        # --- EXPORT REPORT ---
        st.divider()
        st.subheader("Export Report")
        
        if st.button("Generate Combined Report"):
            # Create a simple Excel-compatible CSV structure
            # We tag the results with the metadata
            
            report_lines = []
            report_lines.append(f"Experiment Date,{test_date}")
            report_lines.append(f"Donor ID,{donor_id}")
            report_lines.append("") # Empty line
            
            report_lines.append("--- Anti-IgE Results ---")
            report_lines.append(res_ige.to_csv(index=False))
            report_lines.append("")
            report_lines.append("--- SP Results ---")
            report_lines.append(res_sp.to_csv(index=False))
            
            final_csv = "\n".join(report_lines)
            
            filename = f"Results_{donor_id}_{test_date}.csv"
            st.download_button(
                label="📥 Download Final Report",
                data=final_csv,
                file_name=filename,
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")