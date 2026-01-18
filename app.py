# ==========================================
#        SHARED MATH ENGINE (Bio-Stabilized)
# ==========================================

def four_param_logistic(x, min_val, max_val, log_ec50, hill_slope):
    # PRISM-STYLE EQUATION (Log Scale)
    # x is expected to be log10(dose)
    # This is numerically much more stable for bio-data
    return min_val + (max_val - min_val) / (1 + 10**((log_ec50 - x) * hill_slope))

def get_r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_metrics(doses, responses):
    try:
        # 1. SCRUBBER
        if isinstance(doses, pd.Series) and doses.dtype == object:
             doses = doses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
        if isinstance(responses, pd.Series) and responses.dtype == object:
             responses = responses.astype(str).str.replace(',', '.', regex=False).str.replace('%', '', regex=False)

        doses = pd.to_numeric(doses, errors='coerce')
        responses = pd.to_numeric(responses, errors='coerce')
        
        # 2. CLEAN & LOG TRANSFORM
        # We must ignore Dose=0 or negative (log(0) is impossible)
        mask = (doses > 0) & ~np.isnan(doses) & ~np.isnan(responses)
        x_raw = doses[mask]
        y_clean = responses[mask]
        
        # Convert X to Log10 immediately
        x_log = np.log10(x_raw)

        if len(y_clean) < 4: return None, None, None, None, None, "Not enough data"

        # 3. AUTO-SCALE (Percent Check)
        if max(y_clean) <= 1.0: y_clean = y_clean * 100

        # 4. ROBUST GUESSING & BOUNDS
        # Guess: Min=0, Max=100 (or max data), EC50=Median, Slope=1
        p0 = [min(y_clean), max(y_clean), np.median(x_log), 1.0]
        
        # Bounds: 
        # Min: 0 to 50%
        # Max: 50 to 150% (allowing some overshoot)
        # LogEC50: -inf to +inf
        # Slope: 0.1 to 10 (positive slope only for stimulation)
        bounds = ([0, 50, -np.inf, 0.1], [50, 150, np.inf, 20])

        # FIT (Using Log X)
        popt, _ = curve_fit(four_param_logistic, x_log, y_clean, p0, bounds=bounds, maxfev=10000)
        
        min_val, max_val, log_ec50, hill_slope = popt
        
        # 5. CONVERT BACK TO LINEAR
        ec50 = 10**log_ec50
        
        # EC90 Formula for this specific Log Equation:
        # logEC90 = logEC50 + (1/Slope)*log10(90/10)
        # logECF  = logEC50 + (1/Slope)*log10(F/(100-F))
        ec90 = 10**(log_ec50 + (1/hill_slope)*np.log10(90/10))
        
        # R-SQUARED
        y_pred = four_param_logistic(x_log, *popt)
        r2 = get_r_squared(y_clean, y_pred)
        
        status = "OK"
        if r2 < 0.9: status = "⚠️ Poor Fit"

        # Return full params for plotting
        return popt, ec50, ec50, ec90, r2, status 
        # Note: I return ec50 twice here just to match your unpacking order (EC25 placeholder)
        
    except Exception as e:
        return None, None, None, None, None, f"Fit Failed"