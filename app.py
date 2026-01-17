import streamlit as st

# --- CRITICAL CONFIG ---
st.set_page_config(page_title="Mast Cell Dose-Response", layout="wide")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- 1. The Math Engine (Now with Data Cleaning) ---
def four_param_logistic(x, min_val, max_val, ec50, hill_slope):
    return min_val + (max_val - min_val) / (1 + (x / ec50)**(-hill_slope))

def calculate_metrics(doses, responses):
    try:
        # SAFEGUARD 1: Force data to numeric (coercing errors to NaN)
        # This fixes issues like "1,000" (string) vs 1000 (number)
        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        # Drop NaNs (empty cells or bad data)
        mask = ~np.isnan(doses) & ~np.isnan(responses)
        doses_clean = doses[mask]
        res_clean = responses[mask]

        if len(res_clean) < 4: # Need at least 4 points for 4PL
            return None, None, None, "Not enough data"

        # Initial guesses [min, max, ec50, slope]
        p0 = [min(res_clean), max(res_clean), np.median(doses_clean), 1]
        
        # Fit the curve
        popt, _ = curve_fit(four_param_logistic, doses_clean, res_clean, p0, maxfev=5000)
        min_val, max_val, ec50, hill_slope = popt
        
        # Calculate EC90
        ec90 = ec50 * ((90 / 10) ** (1 / abs(hill_slope)))
        
        return popt, ec50, ec90, "Success"
        
    except Exception as e:
        return None, None, None, f"Fit Failed"

# --- 2. The App Interface ---
st.title("🧬 Dose-Response Calculator (4PL)")
st.write("Status: Ready to process.")

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload CSV/Excel.")
    st.write("2. Ensure 1st column is 'Dose'.")
    st.write("3. Ensure data is numeric (no units like 'ng/ml' in cells).")

# Main
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("File uploaded. Checking structure...")
        
        # Clean column names (remove hidden spaces like "Dose ")
        df.columns = df.columns.str.strip()
        
        if 'Dose' not in df.columns:
            st.error(f"❌ Error: Could not find 'Dose' column. Found these instead: {list(df.columns)}")
            st.info("Tip: Check for typos or capitalization in your header.")
        else:
            results = []
            doses = df['Dose'].values
            fig = go.Figure()
            
            # Progress bar
            progress_bar = st.progress(0)
            cols_to_process = [c for c in df.columns if c != 'Dose']
            
            for i, col in enumerate(cols_to_process):
                responses = df[col].values
                popt, ec50, ec90, status = calculate_metrics(doses, responses)
                
                if status == "Success":
                    results.append({
                        "Sample": col, 
                        "EC50": ec50, 
                        "EC90": ec90, 
                        "Top Plateau": popt[1],
                        "Status": "OK"
                    })
                    
                    # Plotting
                    # Convert to numeric for plotting safe-guard
                    d_plot = pd.to_numeric(doses, errors='coerce')
                    r_plot = pd.to_numeric(responses, errors='coerce')
                    mask = ~np.isnan(d_plot) & ~np.isnan(r_plot)
                    
                    if np.any(mask):
                        fig.add_trace(go.Scatter(x=d_plot[mask], y=r_plot[mask], mode='markers', name=f'{col} (Raw)'))
                        
                        # Smooth line
                        x_min, x_max = min(d_plot[mask]), max(d_plot[mask])
                        if x_min <= 0: x_min = 1e-9 # Prevent log(0) error
                        x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 100)
                        y_smooth = four_param_logistic(x_smooth, *popt)
                        fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f'{col} (Fit)'))
                else:
                    results.append({"Sample": col, "Status": status})
                
                # Update progress
                progress_bar.progress((i + 1) / len(cols_to_process))

            progress_bar.empty() # Clear bar when done

            # Display Results
            st.subheader("1. Calculated Metrics")
            if len(results) > 0:
                res_df = pd.DataFrame(results)
                st.dataframe(res_df)
                
                # Download Button
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
            
            # Display Graph
            st.subheader("2. Curves")
            fig.update_layout(xaxis_type="log", xaxis_title="Dose", yaxis_title="Response", height=600)
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Critical App Error: {e}")