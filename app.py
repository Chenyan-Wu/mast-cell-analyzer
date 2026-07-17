import streamlit as st
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mast Cell Analytics Suite", layout="wide", page_icon="🧬")

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import date, datetime, timezone
import base64
import hashlib
import sqlite3
import tempfile
from pathlib import Path
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    gspread = None
    ServiceAccountCredentials = None

# ==========================================
#        SHARED MATH ENGINE
# ==========================================

def four_param_logistic(x, min_val, max_val, log_ec50, hill_slope):
    """Log-Linear 4PL Model."""
    return min_val + (max_val - min_val) / (1 + 10**((log_ec50 - x) * hill_slope))

def calculate_aic(n, rss, k):
    """Calculate Akaike Information Criterion (AIC)."""
    if rss <= 0: return -np.inf 
    return n * np.log(rss / n) + 2 * k

def get_r_squared(y_true, y_pred):
    try:
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0: return 0 
        return 1 - (ss_res / ss_tot)
    except:
        return 0

def calculate_metrics(doses, responses, multiply_by_100=False):
    """
    Refined Logic:
    - Strict Failure Threshold lowered to 20% (Clamps ceiling).
    - 20-25% generates curve but gives warning.
    """
    try:
        # 1. DATA PREP
        if isinstance(doses, pd.Series) and doses.dtype == object:
             doses = doses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
        if isinstance(responses, pd.Series) and responses.dtype == object:
             responses = responses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)

        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        mask = (doses > 0) & ~np.isnan(doses) & ~np.isnan(responses)
        x_raw = doses[mask].reset_index(drop=True)
        y_clean = responses[mask].reset_index(drop=True)
        
        if len(y_clean) < 4: return None, "NA", "NA", "NA", "NA", 0, "Not enough data"
        
        # APPLY TOGGLE
        if multiply_by_100:
            y_clean = y_clean * 100.0

        x_log = np.log10(x_raw)
        absolute_max = max(y_clean)
        
        # --- DYNAMIC CEILING (Updated to 20%) ---
        if absolute_max < 20.0:
            top_ceiling = 35.0
        else:
            top_ceiling = 200.0

        # --- MODEL 1: 4PL (Sigmoidal) ---
        min_log = min(x_log)
        max_log = max(x_log)
        
        bounds = (
            [-0.001,        absolute_max,  min_log - 1.0,  0.1], 
            [ 0.001,        top_ceiling,   max_log + 1.0,  10.0] 
        )
        
        # PRISM-MATCHED INITIAL GUESS
        half_max_y = absolute_max / 2.0
        idx_closest_to_half = (np.abs(y_clean - half_max_y)).argmin()
        guess_log_ec50 = x_log[idx_closest_to_half]
        
        p0 = [0, absolute_max, guess_log_ec50, 1.0]
        
        try:
            popt, _ = curve_fit(
                four_param_logistic, x_log, y_clean, p0, 
                bounds=bounds, maxfev=10000, ftol=1e-8, xtol=1e-8
            )
            y_pred_4pl = four_param_logistic(x_log, *popt)
            rss_4pl = np.sum((y_clean - y_pred_4pl)**2)
            aic_4pl = calculate_aic(len(y_clean), rss_4pl, 4)
            r2_4pl = get_r_squared(y_clean, y_pred_4pl)
            calc_max = popt[1] 
        except:
            popt = None
            aic_4pl = np.inf
            calc_max = 999

        # --- MODEL 2: Linear Regression ---
        slope, intercept, r_value, _, _ = linregress(x_log, y_clean)
        y_pred_lin = slope * x_log + intercept
        rss_lin = np.sum((y_clean - y_pred_lin)**2)
        aic_lin = calculate_aic(len(y_clean), rss_lin, 2)

        # --- DECISION TREE ---
        hit_ceiling = calc_max > (top_ceiling - 1.0)
        better_linear = aic_lin < (aic_4pl + 2.0)
        
        is_linear = hit_ceiling or better_linear or popt is None
        
        if is_linear:
            if absolute_max < 20.0:
                status = "⚠️ Low (<20%) + Linear"
            elif absolute_max < 25.0:
                status = "⚠️ Linear (Max < 25%)"
            else:
                status = "⚠️ Linear Trend"
            return None, "NA", "NA", "NA", "NA", absolute_max, status

        else:
            min_val, max_val, log_ec50, hill_slope = popt
            
            ec50 = 10**log_ec50
            ec90 = 10**(log_ec50 + (1/hill_slope)*np.log10(90/10))
            ec25 = 10**(log_ec50 + (1/hill_slope)*np.log10(25/75))
            
            # --- STATUS UPDATED ---
            if absolute_max < 20.0:
                status = "⚠️ Low (<20%)" 
            elif absolute_max < 25.0:
                if r2_4pl < 0.9:
                    status = "⚠️ Poor Fit (<25%)"
                else:
                    status = "⚠️ Max < 25%"
            elif r2_4pl < 0.9:
                status = "⚠️ Poor Fit"
            else:
                status = "✅ Pass"

            return popt, ec25, ec50, ec90, r2_4pl, absolute_max, status

    except Exception as e:
        return None, "NA", "NA", "NA", "NA", 0, f"Fit Failed"

# ==========================================
#        GOOGLE DRIVE CONNECTOR
# ==========================================
def save_to_google_sheet(df, sheet_name="MastCell_DB"):
    if gspread is None or ServiceAccountCredentials is None:
        return False, "Google Sheets support is unavailable. Install the packages in requirements.txt."
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

ARCHIVE_DB_PATH = "mastcell_results.db"


def _as_float(value):
    """Convert numeric result values to SQLite-safe floats."""
    try:
        if pd.isna(value) or value == "NA":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def create_run_id(results_df, raw_points_df, test_date, raw_link, source_file):
    """Create a stable ID so saving the same analyzed run twice is idempotent."""
    payload = {
        "test_date": str(test_date),
        "raw_link": str(raw_link).strip(),
        "source_file": str(source_file),
        "results": results_df.fillna("").to_dict(orient="records"),
        "raw_points": raw_points_df.fillna("").to_dict(orient="records"),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def initialize_archive(conn):
    """Create the versioned archive schema when it does not yet exist."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id TEXT PRIMARY KEY,
            test_date TEXT NOT NULL,
            assay_type TEXT NOT NULL,
            raw_link TEXT,
            source_file TEXT,
            created_at TEXT NOT NULL,
            result_count INTEGER NOT NULL,
            point_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analysis_results (
            run_id TEXT NOT NULL,
            result_index INTEGER NOT NULL,
            result_key TEXT NOT NULL,
            donor TEXT,
            stimulant TEXT,
            sample TEXT,
            ec25 REAL,
            ec50 REAL,
            ec90 REAL,
            observed_max REAL,
            r_squared REAL,
            status TEXT,
            model TEXT NOT NULL,
            min_val REAL,
            fitted_max REAL,
            log_ec50 REAL,
            hill_slope REAL,
            dose_unit TEXT,
            PRIMARY KEY (run_id, result_index),
            UNIQUE (run_id, result_key),
            FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS raw_points (
            run_id TEXT NOT NULL,
            result_index INTEGER NOT NULL,
            point_index INTEGER NOT NULL,
            dose REAL NOT NULL,
            response REAL NOT NULL,
            PRIMARY KEY (run_id, result_index, point_index),
            FOREIGN KEY (run_id, result_index)
                REFERENCES analysis_results(run_id, result_index) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_results_donor
            ON analysis_results(donor);
        CREATE INDEX IF NOT EXISTS idx_results_stimulant
            ON analysis_results(stimulant);
        CREATE INDEX IF NOT EXISTS idx_runs_test_date
            ON analysis_runs(test_date);
        """
    )


def save_analysis_to_db(
    results_df,
    raw_points_df,
    test_date,
    raw_link,
    source_file,
    assay_type="Standardized Protocol",
    db_path=ARCHIVE_DB_PATH,
):
    """Atomically archive one complete analysis without creating duplicates."""
    run_id = create_run_id(results_df, raw_points_df, test_date, raw_link, source_file)
    try:
        with sqlite3.connect(db_path) as conn:
            initialize_archive(conn)
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO analysis_runs
                    (run_id, test_date, assay_type, raw_link, source_file,
                     created_at, result_count, point_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(test_date),
                    assay_type,
                    str(raw_link).strip(),
                    str(source_file),
                    datetime.now(timezone.utc).isoformat(),
                    len(results_df),
                    len(raw_points_df),
                ),
            ).rowcount

            if not inserted:
                return True, {"run_id": run_id, "duplicate": True, "db_path": db_path}

            key_to_index = {}
            for result_index, (_, row) in enumerate(results_df.reset_index(drop=True).iterrows()):
                result_key = str(row["Result_Key"])
                key_to_index[result_key] = result_index
                conn.execute(
                    """
                    INSERT INTO analysis_results
                        (run_id, result_index, result_key, donor, stimulant, sample,
                         ec25, ec50, ec90, observed_max, r_squared, status, model,
                         min_val, fitted_max, log_ec50, hill_slope, dose_unit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result_index,
                        result_key,
                        str(row.get("Donor", "")),
                        str(row.get("Stimulant", "")),
                        str(row.get("Sample", "")),
                        _as_float(row.get("EC25")),
                        _as_float(row.get("EC50")),
                        _as_float(row.get("EC90")),
                        _as_float(row.get("Max")),
                        _as_float(row.get("R²")),
                        str(row.get("Status", "")),
                        str(row.get("Model", "Linear")),
                        _as_float(row.get("Fit_Min")),
                        _as_float(row.get("Fit_Max")),
                        _as_float(row.get("Log_EC50")),
                        _as_float(row.get("Hill_Slope")),
                        str(row.get("Dose_Unit", "")),
                    ),
                )

            point_counters = {}
            for _, point in raw_points_df.iterrows():
                result_key = str(point["Result_Key"])
                result_index = key_to_index[result_key]
                point_index = point_counters.get(result_index, 0)
                conn.execute(
                    """
                    INSERT INTO raw_points
                        (run_id, result_index, point_index, dose, response)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result_index,
                        point_index,
                        float(point["Dose"]),
                        float(point["Response"]),
                    ),
                )
                point_counters[result_index] = point_index + 1

        return True, {"run_id": run_id, "duplicate": False, "db_path": db_path}
    except Exception as e:
        return False, str(e)


def load_archive(db_path=ARCHIVE_DB_PATH):
    """Load archive runs and fitted results into query-friendly dataframes."""
    if not Path(db_path).exists():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        initialize_archive(conn)
        runs = pd.read_sql_query(
            "SELECT * FROM analysis_runs ORDER BY test_date DESC, created_at DESC", conn
        )
        results = pd.read_sql_query(
            """
            SELECT r.test_date, r.assay_type, r.raw_link, r.source_file,
                   a.*
            FROM analysis_results a
            JOIN analysis_runs r ON r.run_id = a.run_id
            ORDER BY r.test_date DESC, a.donor, a.stimulant, a.sample
            """,
            conn,
        )
    return runs, results


def load_raw_points(run_id, result_index, db_path=ARCHIVE_DB_PATH):
    """Load the archived observations for one fitted result."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT dose, response FROM raw_points
            WHERE run_id = ? AND result_index = ? ORDER BY point_index
            """,
            conn,
            params=(run_id, int(result_index)),
        )


def merge_archive(source_db_path, target_db_path=ARCHIVE_DB_PATH):
    """Validate and merge a portable archive into the working database."""
    required_tables = {"analysis_runs", "analysis_results", "raw_points"}
    try:
        with sqlite3.connect(source_db_path) as source:
            available_tables = {
                row[0]
                for row in source.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if not required_tables.issubset(available_tables):
                missing = ", ".join(sorted(required_tables - available_tables))
                return False, f"Not a compatible mast-cell archive; missing: {missing}"
            source_run_count = source.execute(
                "SELECT COUNT(*) FROM analysis_runs"
            ).fetchone()[0]

        with sqlite3.connect(target_db_path) as target:
            initialize_archive(target)
            before_count = target.execute(
                "SELECT COUNT(*) FROM analysis_runs"
            ).fetchone()[0]
            target.execute("ATTACH DATABASE ? AS imported", (str(source_db_path),))
            target.execute(
                """
                INSERT OR IGNORE INTO analysis_runs
                    (run_id, test_date, assay_type, raw_link, source_file,
                     created_at, result_count, point_count)
                SELECT run_id, test_date, assay_type, raw_link, source_file,
                       created_at, result_count, point_count
                FROM imported.analysis_runs
                """
            )
            target.execute(
                """
                INSERT OR IGNORE INTO analysis_results
                    (run_id, result_index, result_key, donor, stimulant, sample,
                     ec25, ec50, ec90, observed_max, r_squared, status, model,
                     min_val, fitted_max, log_ec50, hill_slope, dose_unit)
                SELECT run_id, result_index, result_key, donor, stimulant, sample,
                       ec25, ec50, ec90, observed_max, r_squared, status, model,
                       min_val, fitted_max, log_ec50, hill_slope, dose_unit
                FROM imported.analysis_results
                """
            )
            target.execute(
                """
                INSERT OR IGNORE INTO raw_points
                    (run_id, result_index, point_index, dose, response)
                SELECT run_id, result_index, point_index, dose, response
                FROM imported.raw_points
                """
            )
            after_count = target.execute(
                "SELECT COUNT(*) FROM analysis_runs"
            ).fetchone()[0]
        return True, {
            "source_runs": source_run_count,
            "imported_runs": after_count - before_count,
            "total_runs": after_count,
        }
    except Exception as e:
        return False, str(e)


def build_archived_curve(result_row, points_df, log_scale=True):
    """Rebuild a curve from stored raw points and the original fitted parameters."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points_df["dose"],
            y=points_df["response"],
            mode="markers",
            name="Observed",
        )
    )
    if result_row.get("model") == "4PL" and pd.notna(result_row.get("log_ec50")):
        min_dose = points_df["dose"].min()
        max_dose = points_df["dose"].max()
        if log_scale:
            x_smooth = np.logspace(np.log10(min_dose), np.log10(max_dose), 200)
        else:
            x_smooth = np.linspace(min_dose, max_dose, 200)
        y_smooth = four_param_logistic(
            np.log10(x_smooth),
            result_row["min_val"],
            result_row["fitted_max"],
            result_row["log_ec50"],
            result_row["hill_slope"],
        )
        fig.add_trace(
            go.Scatter(x=x_smooth, y=y_smooth, mode="lines", name="Archived 4PL fit")
        )
    fig.update_layout(
        title=f"{result_row.get('donor', '')} – {result_row.get('sample', '')}",
        xaxis_title=f"Dose ({result_row.get('dose_unit', '')})",
        yaxis_title="Degranulation %",
        xaxis_type="log" if log_scale else "linear",
        height=480,
    )
    return fig

def build_qc_export(results_df, raw_link, test_date, stimulant_label):
    """Create a QC-ready table with core fields used downstream."""
    qc = results_df.copy()
    qc["QC_Date"] = str(test_date)
    qc["Raw_Data_Link"] = raw_link
    qc["Assay_Type"] = stimulant_label
    ordered_cols = ["QC_Date", "Donor", "Assay_Type", "Stimulant", "Sample", "EC25", "EC50", "EC90", "Max", "R²", "Status", "Raw_Data_Link"]
    existing = [c for c in ordered_cols if c in qc.columns]
    return qc[existing]


def build_archive_qc_export(results_df):
    """Format database query results as the same downstream QC table."""
    rename_map = {
        "test_date": "QC_Date",
        "donor": "Donor",
        "assay_type": "Assay_Type",
        "stimulant": "Stimulant",
        "sample": "Sample",
        "ec25": "EC25",
        "ec50": "EC50",
        "ec90": "EC90",
        "observed_max": "Max",
        "r_squared": "R²",
        "status": "Status",
        "raw_link": "Raw_Data_Link",
        "run_id": "Run_ID",
    }
    qc = results_df.rename(columns=rename_map)
    ordered = [
        "Run_ID", "QC_Date", "Donor", "Assay_Type", "Stimulant", "Sample",
        "EC25", "EC50", "EC90", "Max", "R²", "Status", "Raw_Data_Link",
    ]
    return qc[[column for column in ordered if column in qc.columns]]


def get_default_donor_samples(available_columns, donor_index):
    """Match donor N to IgE_Sample_N and SP_Sample_N when those columns exist."""
    donor_number = donor_index + 1
    expected_ige = f"IgE_Sample_{donor_number}"
    expected_sp = f"SP_Sample_{donor_number}"
    return {
        "ige": [expected_ige] if expected_ige in available_columns else [],
        "sp": [expected_sp] if expected_sp in available_columns else [],
    }

# ==========================================
#        APP NAVIGATION
# ==========================================

st.sidebar.title("⚙️ Mode Selector")
app_mode = st.sidebar.radio("Choose Analysis Type:", 
    ["Standardized Protocol (IgE/SP)", "Archive & Retrospective", "Custom Experiment (Flexible)"])

st.sidebar.divider()

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
                donors.append({"id": i, "name": name, "color": colors[i], "ige_cols": [], "sp_cols": []})

    st.write("---")
    
    template_data = """Dose_IgE,Dose_SP,IgE_Sample_1,IgE_Sample_2,SP_Sample_1,SP_Sample_2
1,10,45.0,42.0,55.0,50.0
0.5,5,40.0,38.0,50.0,45.0
0.1,2.5,35.0,30.0,45.0,40.0
0.05,1.25,25.0,22.0,35.0,30.0
0.01,0.625,15.0,12.0,25.0,20.0
0.0075,0.3125,10.0,8.0,15.0,12.0
0.005,0.15625,5.0,4.0,10.0,8.0
0.0025,0.078125,2.0,1.0,5.0,4.0
0.001,0.0390625,1.0,0.5,2.0,1.0
,0.01953125,,,1.0,0.5"""
    
    st.download_button(
        label="📥 Download Template CSV",
        data=template_data,
        file_name="mast_cell_template.csv",
        mime="text/csv",
        help="Click to download the standard dose template."
    )

    uploaded_file = st.file_uploader("Upload Standardized Data", type=['csv', 'xlsx'])
    
    if uploaded_file:
        col_ige_dose = st.sidebar.text_input("IgE Dose Col", "Dose_IgE")
        col_sp_dose = st.sidebar.text_input("SP Dose Col", "Dose_SP")

        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()

            available_cols = [c for c in df.columns if c not in [col_ige_dose, col_sp_dose]]
            column_signature = hashlib.sha256(
                "\x1f".join(map(str, available_cols)).encode("utf-8")
            ).hexdigest()[:8]

            # --- AUTO-DETECT DATA SCALE ---
            global_max = 0
            for c in available_cols:
                m = pd.to_numeric(df[c].astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce').max()
                if pd.notna(m) and m > global_max:
                    global_max = m
            
            is_fraction = (0 < global_max <= 2.0)
            
            st.write("---")
            st.subheader("⚙️ Data Processing Options")
            multiply_toggle = st.checkbox("🔄 Convert Fractions to Percentages (x100)", value=is_fraction, help="Automatically converts decimals (0.58) to percentages (58.0)")
            
            if is_fraction:
                st.info(f"💡 **Auto-Detected Decimal Format:** The highest value in your file is **{global_max:.4f}**. We automatically checked the box to convert your data to percentages. If your max response truly is {global_max:.4f}%, please uncheck the box.")

            st.write("---")
            st.info(
                "👇 Sample columns are matched automatically by donor number "
                "(IgE_Sample_N and SP_Sample_N). You can still change every selection."
            )
            
            for d in donors:
                with st.container():
                    st.markdown(f"**👤 {d['name']}**")
                    ca, cb = st.columns(2)
                    defaults = get_default_donor_samples(available_cols, d["id"])
                    d['ige_cols'] = ca.multiselect(
                        f"Anti-IgE Samples ({d['name']})",
                        available_cols,
                        default=defaults["ige"],
                        key=f"ige_{column_signature}_{d['id']}",
                    )
                    d['sp_cols'] = cb.multiselect(
                        f"SP Samples ({d['name']})",
                        available_cols,
                        default=defaults["sp"],
                        key=f"sp_{column_signature}_{d['id']}",
                    )

            def plot_std_category(df, dose_col, donor_list, cat_name, unit, mult_100):
                fig_log = go.Figure()
                fig_lin = go.Figure()
                res = []
                raw_points = []
                
                for d in donor_list:
                    target_cols = d['ige_cols'] if cat_name == "Anti-IgE" else d['sp_cols']
                    if dose_col not in df.columns: continue

                    doses = df[dose_col]
                    show_legend_for_donor = True
                    
                    for col in target_cols:
                        resp = df[col]
                        popt, ec25, ec50, ec90, r2, absolute_max_val, status = calculate_metrics(doses, resp, mult_100)
                        
                        if status != "Not enough data" and status != "Fit Failed":
                            result_key = f"{d['id']}|{d['name']}|{cat_name}|{col}"
                            fit_values = popt if popt is not None else [None, None, None, None]
                            res.append({
                                "Date": str(test_date), "Donor": d['name'], 
                                "Stimulant": cat_name, "Sample": col, 
                                "EC25": ec25, "EC50": ec50, "EC90": ec90, 
                                "Max": absolute_max_val, "R²": r2, "Status": status,
                                "Result_Key": result_key,
                                "Model": "4PL" if popt is not None else "Linear",
                                "Fit_Min": fit_values[0], "Fit_Max": fit_values[1],
                                "Log_EC50": fit_values[2], "Hill_Slope": fit_values[3],
                                "Dose_Unit": unit,
                            })
                            
                            d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                            r_plot = pd.to_numeric(resp.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                            mask = ~np.isnan(d_plot) & ~np.isnan(r_plot) & (d_plot > 0)
                            x_plot_raw = d_plot[mask]
                            y_plot = r_plot[mask]
                            
                            if mult_100: y_plot = y_plot * 100.0

                            for dose_value, response_value in zip(x_plot_raw, y_plot):
                                raw_points.append({
                                    "Result_Key": result_key,
                                    "Dose": float(dose_value),
                                    "Response": float(response_value),
                                })
                            
                            fig_log.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', marker=dict(color=d['color']), showlegend=False))
                            fig_lin.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', marker=dict(color=d['color']), showlegend=False))
                            
                            if popt is not None:
                                x_smooth_log = np.logspace(np.log10(min(x_plot_raw)), np.log10(max(x_plot_raw)), 100)
                                y_smooth_log = four_param_logistic(np.log10(x_smooth_log), *popt)
                                
                                x_smooth_lin = np.linspace(min(x_plot_raw), max(x_plot_raw), 100)
                                y_smooth_lin = four_param_logistic(np.log10(x_smooth_lin), *popt)
                                
                                legend_name = f"{d['name']} {cat_name}"
                                fig_log.add_trace(go.Scatter(x=x_smooth_log, y=y_smooth_log, mode='lines', name=legend_name, line=dict(color=d['color']), showlegend=show_legend_for_donor, legendgroup=d['name']))
                                fig_lin.add_trace(go.Scatter(x=x_smooth_lin, y=y_smooth_lin, mode='lines', name=legend_name, line=dict(color=d['color']), showlegend=False, legendgroup=d['name']))
                                show_legend_for_donor = False
                
                fig_log.update_layout(title=f"{cat_name} (Log Scale)", xaxis_title=f"Dose ({unit})", yaxis_title="Degranulation %", xaxis_type="log", height=450)
                fig_lin.update_layout(title=f"{cat_name} (Linear Scale)", xaxis_title=f"Dose ({unit})", yaxis_title="Degranulation %", xaxis_type="linear", height=450)
                return pd.DataFrame(res), fig_log, fig_lin, pd.DataFrame(raw_points)

            if st.button("🚀 Run Standard Analysis"):
                r_ige, f_ige_log, f_ige_lin, raw_ige = plot_std_category(df, col_ige_dose, donors, "Anti-IgE", "µg/mL", multiply_toggle)
                r_sp, f_sp_log, f_sp_lin, raw_sp = plot_std_category(df, col_sp_dose, donors, "SP", "µM", multiply_toggle)
                raw_points = pd.concat([raw_ige, raw_sp], ignore_index=True)
                st.session_state['std_results'] = {
                    'r_ige': r_ige, 'f_ige_log': f_ige_log, 'f_ige_lin': f_ige_lin,
                    'r_sp': r_sp, 'f_sp_log': f_sp_log, 'f_sp_lin': f_sp_lin,
                    'raw_points': raw_points, 'raw_link': raw_link,
                    'test_date': test_date, 'source_file': uploaded_file.name,
                }

            if 'std_results' in st.session_state:
                res = st.session_state['std_results']
                display_columns = ["Date", "Donor", "Stimulant", "Sample", "EC25", "EC50", "EC90", "Max", "R²", "Status"]
                st.markdown("### 📊 Results")
                
                st.subheader("Anti-IgE Results")
                c1, c2 = st.columns(2)
                with c1: 
                    st.plotly_chart(res['f_ige_log'], use_container_width=True)
                    st.plotly_chart(res['f_ige_lin'], use_container_width=True)
                with c2: st.dataframe(res['r_ige'].reindex(columns=display_columns), height=600)

                st.divider()

                st.subheader("SP Results")
                c3, c4 = st.columns(2)
                with c3: 
                    st.plotly_chart(res['f_sp_log'], use_container_width=True)
                    st.plotly_chart(res['f_sp_lin'], use_container_width=True)
                with c4: st.dataframe(res['r_sp'].reindex(columns=display_columns), height=600)

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Mast Cell Report</title>
                    <style>
                        @page {{
                            size: landscape;
                            margin: 1cm;
                        }}
                        body {{ font-family: sans-serif; -webkit-print-color-adjust: exact; }}
                        .container {{ width: 100%; display: flex; flex-wrap: wrap; page-break-inside: avoid; }}
                        .plot-col {{ width: 49%; padding: 5px; }}
                        .table-container {{ width: 100%; margin-top: 20px; page-break-inside: avoid; }}
                        h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
                        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
                        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        @media print {{
                            .no-print {{ display: none; }}
                        }}
                    </style>
                </head>
                <body>
                    <h1>Mast Cell Report ({res['test_date']})</h1>
                    <p class="no-print" style="color:red; font-weight:bold;">👉 Press Ctrl+P (or Cmd+P) and choose "Save as PDF". Ensure Layout is set to "Landscape".</p>
                    
                    <h2>Anti-IgE Analysis</h2>
                    <div class="table-container">
                        {res['r_ige'].reindex(columns=display_columns).to_html(index=False)}
                    </div>
                    <div class="container">
                        <div class="plot-col">{res['f_ige_log'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        <div class="plot-col">{res['f_ige_lin'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    </div>
                    
                    <div style="page-break-before: always;"></div>
                    
                    <h2>SP Analysis</h2>
                    <div class="table-container">
                        {res['r_sp'].reindex(columns=display_columns).to_html(index=False)}
                    </div>
                    <div class="container">
                        <div class="plot-col">{res['f_sp_log'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        <div class="plot-col">{res['f_sp_lin'].to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    </div>
                </body>
                </html>
                """
                b64 = base64.b64encode(html.encode()).decode()
                st.markdown(f'<a href="data:text/html;base64,{b64}" download="Mast_Cell_Report_Landscape.html" style="background-color:#FF4B4B;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;font-weight:bold;">📥 Download Report (Print to PDF)</a>', unsafe_allow_html=True)
                st.caption("ℹ️ To get a PDF: Download this file, open it in your browser, and select 'Print' -> 'Save as PDF'. The layout is pre-set to Horizontal/Landscape.")

                st.divider()
                st.subheader("☁️ Database")
                if st.button("💾 Save Results to Google Drive"):
                    full_db = pd.concat([res['r_ige'], res['r_sp']], ignore_index=True)
                    google_db = full_db.reindex(columns=display_columns).copy()
                    google_db['Raw_Link'] = res['raw_link']
                    success, msg = save_to_google_sheet(google_db, "MastCell_DB")
                    if success: st.success(f"✅ Saved {len(google_db)} rows to Google Drive!")
                    else: st.error(f"❌ Error: {msg}")

                st.subheader("🧩 One-Run QC + Archive")
                full_db = pd.concat([res['r_ige'], res['r_sp']], ignore_index=True)
                full_db["Raw_Link"] = res["raw_link"]

                c_qc1, c_qc2 = st.columns(2)
                with c_qc1:
                    qc_df = build_qc_export(full_db, res["raw_link"], res["test_date"], "Standardized Protocol")
                    csv_bytes = qc_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download QC CSV (Donor + ECxx + Max + Status)",
                        data=csv_bytes,
                        file_name=f"QC_{res['test_date']}.csv",
                        mime="text/csv"
                    )
                with c_qc2:
                    archive_ready = "raw_points" in res and "Result_Key" in full_db.columns
                    if not archive_ready:
                        st.warning("Run the analysis once more to create the complete curve archive.")
                    if st.button("🗄️ Archive Complete Run", disabled=not archive_ready):
                        ok, out = save_analysis_to_db(
                            full_db,
                            res["raw_points"],
                            res["test_date"],
                            res["raw_link"],
                            res.get("source_file", "uploaded_file"),
                        )
                        if ok:
                            if out["duplicate"]:
                                st.info(f"Already archived as run {out['run_id']}; no duplicate was added.")
                            else:
                                st.success(f"✅ Complete run archived as {out['run_id']}.")
                        else:
                            st.error(f"❌ Could not save to local DB: {out}")

                if Path(ARCHIVE_DB_PATH).exists():
                    st.download_button(
                        "📦 Download Complete SQLite Database",
                        data=Path(ARCHIVE_DB_PATH).read_bytes(),
                        file_name="mastcell_results.db",
                        mime="application/vnd.sqlite3",
                        help="Keep this file as a portable backup, especially when the app is hosted.",
                    )

        except Exception as e: st.error(f"Error: {e}")

elif app_mode == "Archive & Retrospective":
    st.title("🗂️ Archive & Retrospective Analysis")
    st.caption(
        "Search prior donors and ECxx results, reconstruct the original fitted curve, "
        "and generate a new QC export from one database."
    )

    with st.expander("📤 Restore or merge a database backup"):
        archive_upload = st.file_uploader(
            "Select a mastcell_results.db backup", type=["db", "sqlite", "sqlite3"]
        )
        if archive_upload is not None and st.button("Merge Backup into Working Archive"):
            with tempfile.NamedTemporaryFile(suffix=".db") as temporary_archive:
                temporary_archive.write(archive_upload.getvalue())
                temporary_archive.flush()
                merged, merge_info = merge_archive(temporary_archive.name)
            if merged:
                st.success(
                    f"Imported {merge_info['imported_runs']} new run(s); "
                    f"the archive now contains {merge_info['total_runs']} run(s)."
                )
            else:
                st.error(f"Could not import this backup: {merge_info}")

    if not Path(ARCHIVE_DB_PATH).exists():
        st.info("No archive exists yet. Analyze and archive a standardized run first.")
    else:
        try:
            runs_df, archive_df = load_archive()
            m1, m2, m3 = st.columns(3)
            m1.metric("Archived runs", len(runs_df))
            m2.metric("Result rows", len(archive_df))
            m3.metric("Donors", archive_df["donor"].nunique() if not archive_df.empty else 0)

            if archive_df.empty:
                st.info("The database is valid but does not contain archived result rows yet.")
            else:
                st.subheader("🔎 Find archived results")
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    selected_dates = st.multiselect(
                        "Date", sorted(archive_df["test_date"].dropna().unique(), reverse=True)
                    )
                with f2:
                    selected_donors = st.multiselect(
                        "Donor", sorted(archive_df["donor"].dropna().unique())
                    )
                with f3:
                    selected_stimulants = st.multiselect(
                        "Stimulant", sorted(archive_df["stimulant"].dropna().unique())
                    )
                with f4:
                    selected_statuses = st.multiselect(
                        "Status", sorted(archive_df["status"].dropna().unique())
                    )

                filtered = archive_df.copy()
                if selected_dates:
                    filtered = filtered[filtered["test_date"].isin(selected_dates)]
                if selected_donors:
                    filtered = filtered[filtered["donor"].isin(selected_donors)]
                if selected_stimulants:
                    filtered = filtered[filtered["stimulant"].isin(selected_stimulants)]
                if selected_statuses:
                    filtered = filtered[filtered["status"].isin(selected_statuses)]

                qc_history = build_archive_qc_export(filtered)
                st.dataframe(qc_history, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Filtered QC CSV",
                    data=qc_history.to_csv(index=False).encode("utf-8"),
                    file_name=f"QC_archive_{date.today()}.csv",
                    mime="text/csv",
                    disabled=filtered.empty,
                )

                if not filtered.empty:
                    st.subheader("📈 Reconstruct an archived curve")
                    curve_rows = filtered.reset_index(drop=True)
                    selected_curve_index = st.selectbox(
                        "Select result",
                        options=range(len(curve_rows)),
                        format_func=lambda index: (
                            f"{curve_rows.iloc[index]['test_date']} | "
                            f"{curve_rows.iloc[index]['donor']} | "
                            f"{curve_rows.iloc[index]['stimulant']} | "
                            f"{curve_rows.iloc[index]['sample']}"
                        ),
                    )
                    selected_result = curve_rows.iloc[selected_curve_index]
                    archived_points = load_raw_points(
                        selected_result["run_id"], selected_result["result_index"]
                    )
                    log_scale = st.toggle("Logarithmic dose axis", value=True)
                    archived_figure = build_archived_curve(
                        selected_result, archived_points, log_scale=log_scale
                    )
                    chart_col, detail_col = st.columns([2, 1])
                    with chart_col:
                        st.plotly_chart(archived_figure, use_container_width=True)
                    with detail_col:
                        st.markdown(f"**Run ID:** `{selected_result['run_id']}`")
                        st.markdown(f"**Model:** {selected_result['model']}")
                        st.markdown(f"**EC25:** {selected_result['ec25']}")
                        st.markdown(f"**EC50:** {selected_result['ec50']}")
                        st.markdown(f"**EC90:** {selected_result['ec90']}")
                        st.markdown(f"**Observed max:** {selected_result['observed_max']}")
                        st.markdown(f"**R²:** {selected_result['r_squared']}")
                        st.markdown(f"**Status:** {selected_result['status']}")
                        if selected_result["raw_link"]:
                            st.markdown(f"[Open original raw data]({selected_result['raw_link']})")

            st.divider()
            st.download_button(
                "📦 Download Complete SQLite Database",
                data=Path(ARCHIVE_DB_PATH).read_bytes(),
                file_name="mastcell_results.db",
                mime="application/vnd.sqlite3",
                help="Keep this portable backup outside the app, especially when hosted.",
            )
        except Exception as e:
            st.error(f"Could not read the archive: {e}")

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
                
                # Auto-Detect for Custom Mode
                global_max_c = 0
                for c in selected_samples:
                    m = pd.to_numeric(df_c[c].astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce').max()
                    if pd.notna(m) and m > global_max_c:
                        global_max_c = m
                is_fraction_c = (0 < global_max_c <= 2.0)
                
                multiply_toggle_c = st.checkbox("🔄 Convert Fractions to Percentages (x100)", value=is_fraction_c)
                if is_fraction_c:
                    st.info(f"💡 Auto-Detected Decimal Format. Global max is {global_max_c:.4f}. Box checked automatically.")
                
                if st.button("Run Custom Analysis") or ('custom_results' in st.session_state):
                    results_c = []
                    fig_log = go.Figure()
                    fig_lin = go.Figure()
                    doses = df_c[dose_col]

                    for sample in selected_samples:
                        responses = df_c[sample]
                        popt, ec25, ec50, ec90, r2, absolute_max_val, status = calculate_metrics(doses, responses, multiply_toggle_c)
                        
                        if status != "Not enough data" and status != "Fit Failed":
                            results_c.append({
                                "Date": str(date.today()), "Sample": sample, 
                                "EC25": ec25, "EC50": ec50, "EC90": ec90, 
                                "Max Response": absolute_max_val, "R²": r2, "Status": status
                            })
                            d_plot = pd.to_numeric(doses.astype(str).str.replace(',', '.'), errors='coerce')
                            r_plot = pd.to_numeric(responses.astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
                            mask = ~np.isnan(d_plot) & ~np.isnan(r_plot) & (d_plot > 0)
                            x_plot_raw = d_plot[mask]
                            y_plot = r_plot[mask]
                            
                            if multiply_toggle_c: y_plot = y_plot * 100.0

                            fig_log.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', name=sample))
                            fig_lin.add_trace(go.Scatter(x=x_plot_raw, y=y_plot, mode='markers', name=sample))
                            
                            if popt is not None:
                                x_smooth_log = np.logspace(np.log10(min(x_plot_raw)), np.log10(max(x_plot_raw)), 100)
                                y_smooth_log = four_param_logistic(np.log10(x_smooth_log), *popt)
                                fig_log.add_trace(go.Scatter(x=x_smooth_log, y=y_smooth_log, mode='lines', name=f"{sample} Fit"))
                                
                                x_smooth_lin = np.linspace(min(x_plot_raw), max(x_plot_raw), 100)
                                y_smooth_lin = four_param_logistic(np.log10(x_smooth_lin), *popt)
                                fig_lin.add_trace(go.Scatter(x=x_smooth_lin, y=y_smooth_lin, mode='lines', name=f"{sample} Fit"))
                        else:
                            results_c.append({"Sample": sample, "Status": status})
                    
                    res_df_c = pd.DataFrame(results_c)
                    st.session_state['custom_results'] = {'df': res_df_c, 'fig_log': fig_log, 'fig_lin': fig_lin}

                    c_tbl, c_plt = st.columns([1, 2])
                    with c_tbl: st.dataframe(res_df_c)
                    with c_plt:
                        fig_log.update_layout(title="Log Scale", xaxis_title=f"Dose ({unit_label})", yaxis_title="Response %", xaxis_type="log", height=400)
                        st.plotly_chart(fig_log, use_container_width=True)
                        fig_lin.update_layout(title="Linear Scale", xaxis_title=f"Dose ({unit_label})", yaxis_title="Response %", xaxis_type="linear", height=400)
                        st.plotly_chart(fig_lin, use_container_width=True)

                    html_c = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <style>
                            @page {{ size: landscape; margin: 1cm; }}
                            body {{ font-family: sans-serif; }}
                            .container {{ width: 100%; display: flex; flex-wrap: wrap; }}
                            .plot-col {{ width: 49%; padding: 5px; }}
                            .table-container {{ width: 100%; margin-top: 20px; }}
                            table {{ width: 100%; border-collapse: collapse; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        </style>
                    </head>
                    <body>
                        <h1>Custom Analysis Report</h1>
                        <p class="no-print" style="color:red;">👉 Press Ctrl+P -> Save as PDF (Landscape).</p>
                        <div class="table-container">{res_df_c.to_html(index=False)}</div>
                        <div class="container">
                            <div class="plot-col">{fig_log.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                            <div class="plot-col">{fig_lin.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                        </div>
                    </body>
                    </html>
                    """
                    b64_c = base64.b64encode(html_c.encode()).decode()
                    st.markdown(f'<a href="data:text/html;base64,{b64_c}" download="Custom_Report_Landscape.html" style="background-color:#FF4B4B;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;font-weight:bold;">📥 Download Report (Print to PDF)</a>', unsafe_allow_html=True)

                    st.divider()
                    if st.button("💾 Save Custom Results to Drive"):
                        success, msg = save_to_google_sheet(res_df_c, "MastCell_DB")
                        if success: st.success(f"✅ Saved to Google Drive!")
                        else: st.error(f"❌ Error: {msg}")

        except Exception as e: st.error(f"Error reading file: {e}")
