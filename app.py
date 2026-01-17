import streamlit as st

# --- CRITICAL: This must be the very first Streamlit command ---
st.set_page_config(page_title="Mast Cell Dose-Response", layout="wide")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- 1. The Math Engine (4PL Model) ---
def four_param_logistic(x, min_val, max_val, ec50, hill_slope):
    return min_val + (max_val - min_val) / (1 + (x / ec50)**(-hill_slope))

def calculate_metrics(doses, responses):
    # Initial guesses [min, max, ec50, slope]
    # We add error handling for empty data
    if len(responses) == 0: return None, None, None, "No Data"
    
    p0 = [min(responses), max(responses), np.median(doses), 1]
    
    try:
        popt, _ = curve_fit(four_param_logistic, doses, responses, p0, maxfev=5000)
        min_val, max_val, ec50, hill_slope = popt
        
        # Calculate EC90
        ec90 = ec50 * ((90 / 10) ** (1 / abs(hill_slope)))
        return popt, ec50, ec90, "Success"
    except Exception as e:
        return None, None, None, "Fit Failed"

# --- 2. The App Interface ---
st.title("🧬 Dose-Response Calculator (4PL)")
st.markdown("If you can see this text, the app is working.")

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("Upload a CSV file where the first column is 'Dose'.")
    
    # Create Download Template
    dummy_data = {
        'Dose': [1000, 142.8, 20.4, 2.9, 0.41, 0.06, 0.008, 0.001],
        'Sample_1': [74.98, 78.66, 79.55, 82.72, 78.25, 70.05, 43.43, 2.65],
        'Sample_2': [10, 15, 40, 75, 80, 82, 81, 80]
    }
    df_template = pd.DataFrame(dummy_data)
    csv = df_template.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Template CSV", csv, "template.csv", "text/csv")

# Main
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("File uploaded successfully. Processing...")
        
        # Check columns
        if 'Dose' not in df.columns:
            st.error("❌ Error: The first column must be named 'Dose' (case sensitive).")
        else:
            results = []
            doses = df['Dose'].values
            fig = go.Figure()

            for col in df.columns:
                if col == 'Dose': continue
                
                responses = df[col].values
                popt, ec50, ec90, status = calculate_metrics(doses, responses)
                
                if status == "Success":
                    results.append({
                        "Sample": col, 
                        "EC50": ec50, 
                        "EC90": ec90, 
                        "Max Response": popt[1],
                        "Status": "OK"
                    })
                    # Plotting
                    fig.add_trace(go.Scatter(x=doses, y=responses, mode='markers', name=f'{col} (Raw)'))
                    x_smooth = np.logspace(np.log10(min(doses) + 1e-9), np.log10(max(doses)), 100)
                    y_smooth = four_param_logistic(x_smooth, *popt)
                    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=f'{col} (Fit)'))
                else:
                    results.append({"Sample": col, "Status": "Fit Failed"})

            # Display Results
            st.subheader("Results")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            
            # Display Graph
            st.subheader("Curves")
            fig.update_layout(xaxis_type="log", height=600)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")