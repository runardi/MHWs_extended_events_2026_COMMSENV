"""
Marine Heatwave (MHW) Extended Event Detection
===============================================
Extends the standard MHW detection framework (Hobday et al. 2016) with
custom pre- and post-event period identification and extended event metrics.

Dependencies
------------
marineHeatWaves https://github.com/ecjoliver/marineHeatWaves

Usage
-----
# As a script (runs the built-in example at the bottom of this file):
    python mhw_extended_events.py

# As an imported module:
    import pandas as pd
    from mhw_extended_events import detect_mhw_extended
    _, df_events = detect_mhw_extended(t, sst)
        

References
----------
Hobday, A.J. et al. (2016). A hierarchical approach to defining marine
    heatwaves. Progress in Oceanography, 141, 227-238.
Nardi, R. U. et al., Persistent warm water anomalies before and after 
    marine heatwaves amplify heat exposure and associated risks. Nature 
    Communications Earth & Environment
"""

import warnings
from datetime import date
import numpy as np
import pandas as pd
from scipy.signal import detrend

try:
    import marineHeatWaves as mhw
except ImportError:
    raise ImportError(
        "'marineHeatWaves' package not found. "
        "Install it from: https://github.com/ecjoliver/marineHeatWaves"
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)

def detect_mhw_extended(
    t,
    sst,
    climatology_period=None,
    extension_window: int = 730,
    consec_days: int = 3,
    detrend_sst: bool = True,
    label: str = "series",
    verbose: bool = True,
):
    """
    Detect marine heatwaves and extend standard MHW boundaries to capture pre- and post-event periods.

    Parameters
    ----------
    t : array-like of int
        Ordinal dates aligned with *sst*.
        If you have a pandas DatetimeIndex or date strings, convert with: t = pd.to_datetime(dates).map(pd.Timestamp.toordinal).to_numpy()
    sst : array-like of float
        Daily sea surface temperature in °C. NaNs are supported.
    climatology_period : list of [int, int], optional [start_year, end_year] of the baseline climatology. Defaults to the full span of *t*.
    extension_window : int
        Maximum days to search on each side of an event (default 730).
    consec_days : int
        Consecutive below-climatology days required to delimit a phase boundary (default 3).   
    detrend_sst : bool
        If True (default), linearly detrend the SST. Set to False to use the raw SST values,which preserves the long-term warming trend in the record.       
    label : str
        Identifier stored in the "label" column (default 'series').
    verbose : bool
        Print a detection summary and exclusion details (default True).

    Returns
    -------
    mhw_detected : dict
        The raw output of "marineHeatWaves.detect" 
        
    df_extended : pd.DataFrame
        One row per *retained* MHW event with the pre/post extension statistics calculated by this module:
                  
            label                       : str   -- series identifier
            event_number                : int   -- sequential event index
            event_start_idx             : int   -- array index of event start
            event_end_idx               : int   -- array index of event end
            start_date                  : date  -- calendar date of event start
            category                    : str   -- MHW category (Hobday et al.)
            event_onset                 : °C day⁻¹ -- rate of onset
            event_decline               : °C day⁻¹ -- rate of decline
            event_duration              : days      -- number of valid event days
            event_mean_intensity        : °C        -- mean SST anomaly
            event_max_intensity         : °C        -- peak SST anomaly
            event_cumulative_intensity  : °C·days   -- cumulative SST anomaly
            recovery_time               : days -- days between previous event end and this event start
            pre_has_mhw                 : bool -- another MHW overlaps the pre-event window
            post_has_mhw                : bool -- another MHW overlaps the post-event window
            pre_start_idx               : int      -- array index of pre-event start
            pre_end_idx                 : int      -- array index of pre-event end
            pre_onset_rate              : °C day⁻¹ -- rate of SST anomaly change across pre phase
            pre_duration                : days     -- number of valid pre-event days
            pre_mean_intensity          : °C       -- mean SST anomaly
            pre_max_intensity           : °C       -- peak SST anomaly
            pre_cumulative_intensity    : °C·days  -- cumulative SST anomaly
            post_start_idx              : int      -- array index of post-event start
            post_end_idx                : int      -- array index of post-event end
            post_offset_rate            : °C day⁻¹ -- absolute rate of SST anomaly change across post phase
            post_duration               : days     -- number of valid post-event days
            post_mean_intensity         : °C       -- mean SST anomaly
            post_max_intensity          : °C       -- peak SST anomaly
            post_cumulative_intensity   : °C·days  -- cumulative SST anomaly
            cum_int_sum                 : °C·days  -- pre_cumulative_intensity + post_cumulative_intensity
            ratio_pre_event_mean_int    : dimensionless -- pre_mean_intensity / event_mean_intensity
            ratio_post_event_mean_int   : dimensionless -- post_mean_intensity / event_mean_intensity
            ratio_pre_post_mean_int     : dimensionless -- pre_mean_intensity / post_mean_intensity
            ratio_pre_event_cum_int     : dimensionless -- pre_cumulative_intensity / event_cumulative_intensity
            ratio_post_event_cum_int    : dimensionless -- post_cumulative_intensity / event_cumulative_intensity
            ratio_pre_post_cum_int      : dimensionless -- pre_cumulative_intensity / post_cumulative_intensity
            ratio_sum_event_cum_int     : dimensionless -- cum_int_sum / event_cumulative_intensity
            ratio_rate                  : dimensionless -- pre_onset_rate / post_offset_rate
    Notes
    -----
    Events are excluded when:
    - No *consec_days*-consecutive below-climatology run is found within
      *extension_window* days for either the pre- or post-event phase.
    - Either phase contains any NaN day.
    """
    t   = np.asarray(t,   dtype=int)
    sst = np.asarray(sst, dtype=float)

    # Default climatology: full span of the time series
    if climatology_period is None:
        years = [date.fromordinal(int(d)).year for d in t]
        climatology_period = [min(years), max(years)]

    
    # Optionally detrend, preserving the original NaN mask             
    nan_mask = np.isnan(sst)

    if detrend_sst:
        sst_interp = np.interp(
            np.arange(len(sst)), np.where(~nan_mask)[0], sst[~nan_mask]
        )
        sst_proc           = detrend(sst_interp)
        sst_proc[nan_mask] = np.nan
        if verbose:
            print("  Detrending: ON (linear trend removed)")
    else:
        sst_proc = sst.copy()
        if verbose:
            print("  Detrending: OFF (raw SST used)")

    # MHW detection                             
    mhw_90, clim_90 = mhw.detect(t, sst_proc.copy(), climatologyPeriod=climatology_period, pctile=90)

    sst_anomaly = sst_proc - clim_90["seas"]

    
    # Extend event boundaries                                           
    extended, start_inds, end_inds, rules = _detect_extended_events(sst_proc, sst_anomaly, t, mhw_90, clim_90,window=extension_window, consec_days=consec_days)
        
    all_starts = np.array(mhw_90["index_start"])
    all_ends   = np.array(mhw_90["index_end"])

    def _overlaps_other_mhw(win_s, win_e, skip_s):
        return any(
            s != skip_s and s <= win_e and e >= win_s
            for s, e in zip(all_starts, all_ends)
        )

    # Exclusion checks and per-event statistics                    
    rows = []
    n_no_rule = n_nan_gap = 0

    for i, ((pre_s_ext, post_e_ext), (pre_rule, post_rule)) in enumerate(
        zip(extended, rules)
    ):
        orig_s    = start_inds[i]
        orig_e    = end_inds[i]
        ev_date   = date.fromordinal(int(t[orig_s]))

        if pre_s_ext is None or post_e_ext is None:
            n_no_rule += 1
            if verbose:
                print(
                    f"  EXCLUDED (no rule met)  event {i+1} starting {ev_date} "
                    f"| pre='{pre_rule}'  post='{post_rule}'"
                )
            continue

        pre_s, pre_e   = pre_s_ext, orig_s - 1
        post_s, post_e = orig_e + 1, post_e_ext

        if _has_nan_gap(sst_proc, pre_s,  pre_e) or \
           _has_nan_gap(sst_proc, post_s, post_e):
            n_nan_gap += 1
            if verbose:
                print(
                    f"  EXCLUDED  "
                    f"event {i+1} starting {ev_date}"
                )
            continue

        recovery_time = (
            int(orig_s - end_inds[i - 1] - 1) if i > 0 else np.nan
        )

        pre_stats   = _period_stats(sst_proc, clim_90["seas"], pre_s,  pre_e)
        event_stats = _period_stats(sst_proc, clim_90["seas"], orig_s, orig_e)
        post_stats  = _period_stats(sst_proc, clim_90["seas"], post_s, post_e)

        row = {
            "label":                      label,
            "event_number":               i + 1,
            "event_start_idx":            int(orig_s),
            "event_end_idx":              int(orig_e),
            "start_date":                 ev_date,
            "category":                   mhw_90["category"][i],
            "event_onset":                mhw_90["rate_onset"][i],
            "event_decline":              mhw_90["rate_decline"][i],
            "event_duration":             event_stats["duration"],
            "event_mean_intensity":       event_stats["mean_intensity"],
            "event_max_intensity":        event_stats["max_intensity"],
            "event_cumulative_intensity": event_stats["cumulative_intensity"],
            "recovery_time":              recovery_time,
            "pre_has_mhw":                _overlaps_other_mhw(pre_s,  pre_e,  orig_s),
            "pre_start_idx":              int(pre_s),
            "pre_end_idx":                int(pre_e),
            "pre_onset_rate":         _two_point_rate(sst_anomaly, pre_s,  pre_e),            
            "pre_duration":               pre_stats["duration"],
            "pre_mean_intensity":         pre_stats["mean_intensity"],
            "pre_max_intensity":          pre_stats["max_intensity"],
            "pre_cumulative_intensity":   pre_stats["cumulative_intensity"],
            "post_has_mhw":               _overlaps_other_mhw(post_s, post_e, orig_s),
            "post_start_idx":             int(post_s),
            "post_end_idx":               int(post_e),
            "post_offset_rate":       abs(_two_point_rate(sst_anomaly, post_s, post_e)),
            "post_duration":              post_stats["duration"],
            "post_mean_intensity":        post_stats["mean_intensity"],
            "post_max_intensity":         post_stats["max_intensity"],
            "post_cumulative_intensity":  post_stats["cumulative_intensity"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if verbose:
        print(
            f"  Detected {len(start_inds)} events | "
            f"Retained {len(df)}"
        )

    if not df.empty:
        df = _add_ratio_columns(df)

    return mhw_90, df

# ---------------------------------------------------------------------------

def _has_nan_gap(sst: np.ndarray, start: int, end: int):
    "True if sst[start:end+1] contains any NaN value."
    if end < start:
        return False
    return bool(np.isnan(sst[start:end + 1]).any())

def _period_stats(sst: np.ndarray, clim_mean: np.ndarray,
                  start: int, end: int):
    "Duration and intensity metrics for a sub-period."
    empty = dict(duration=0, mean_intensity=np.nan,
                 max_intensity=np.nan, cumulative_intensity=np.nan)
    if end < start:
        return empty
    seg  = sst[start:end + 1]
    clim = clim_mean[start:end + 1]
    valid = ~np.isnan(seg)
    if not valid.any():
        return empty
    intens = seg[valid] - clim[valid]
    return dict(
        duration             = int(valid.sum()),
        mean_intensity       = float(np.mean(intens)),
        max_intensity        = float(np.max(intens)),
        cumulative_intensity = float(np.sum(intens)),
    )

def _two_point_rate(sst_anomaly: np.ndarray, start: int, end: int):
    "Rate of change (°C day⁻¹) between boundary-averaged endpoints."
    T = len(sst_anomaly)
    if not (0 <= start < T and 0 <= end < T):
        return np.nan
    sv = (0.5 * (sst_anomaly[start] + sst_anomaly[start - 1])
          if start > 0 else sst_anomaly[start])
    so = 0.5 if start > 0 else 0.0
    ev = (0.5 * (sst_anomaly[end] + sst_anomaly[end + 1])
          if end < T - 1 else sst_anomaly[end])
    eo = 0.5 if end < T - 1 else 0.0
    if np.isnan(sv) or np.isnan(ev):
        return np.nan
    dur = (end - start) + so + eo
    return np.nan if dur == 0 else float((ev - sv) / dur)


def _detect_extended_events(sst_proc, sst_anomaly, t, mhw_stats, clim_90,
                             window, consec_days):
    "Extend MHW boundaries with pre/post phase detection."
    start_inds = np.array(mhw_stats["index_start"])
    end_inds   = np.array(mhw_stats["index_end"])
    T, N = len(sst_proc), consec_days

    def below(j): return not np.isnan(sst_anomaly[j]) and sst_anomaly[j] < 0.0
    def above(j): return not np.isnan(sst_anomaly[j]) and sst_anomaly[j] >= 0.0

    extended, rules = [], []

    for start, end in zip(start_inds, end_inds):

        # --- PRE: search backwards ---
        pre_ext, pre_rule = None, "No rule met"
        for i in range(start - 1, max(0, start - window - 1), -1):
            rs = i - N + 1
            if rs < 0:
                break
            if all(below(j) for j in range(rs, i + 1)):
                ts = next((j for j in range(i + 1, start + 1) if above(j)), None)
                if ts is not None:
                    pre_ext  = ts
                    pre_rule = f"{N}-days below climatology"
                break

        # --- POST: search forwards ---
        post_ext, post_rule = None, "No rule met"
        for i in range(end + 1, min(T - N + 1, end + window + 1)):
            if all(below(j) for j in range(i, i + N)):
                te = next((j for j in range(i - 1, end - 1, -1) if above(j)), None)
                if te is not None:
                    post_ext  = te
                    post_rule = f"{N}-days below climatology"
                break

        extended.append((pre_ext, post_ext))
        rules.append((pre_rule, post_rule))

    return extended, start_inds, end_inds, rules


def _add_ratio_columns(df: pd.DataFrame):
    "Append derived ratio columns to the events dataframe."
    for col, short in [
        ("mean_intensity",       "mean_int"),
        ("cumulative_intensity", "cum_int"),
    ]:
        pre, post, ev = f"pre_{col}", f"post_{col}", f"event_{col}"
        ev_d   = df[ev].replace(0, np.nan)
        post_d = df[post].replace(0, np.nan)
        df[f"ratio_pre_event_{short}"]  = df[pre].replace(0, np.nan) / ev_d
        df[f"ratio_post_event_{short}"] = post_d / ev_d
        df[f"ratio_pre_post_{short}"]   = df[pre] / post_d

    df["cum_int_sum"] = (
        df["pre_cumulative_intensity"] + df["post_cumulative_intensity"]
    )
    df["ratio_sum_event_cum_int"] = (
        df["cum_int_sum"] / df["event_cumulative_intensity"].replace(0, np.nan)
    )
    df["ratio_rate"] = (
        df["pre_onset_rate"]
        / df["post_offset_rate"].replace(0, np.nan)
    )
    final_cols = [
        "label", "event_number", "event_start_idx", "event_end_idx",
        "start_date", "category",
        "event_onset", "event_decline",
        "event_duration", "event_mean_intensity", "event_max_intensity", "event_cumulative_intensity",
        "recovery_time", "pre_has_mhw", "post_has_mhw","pre_start_idx", "pre_end_idx",
        "pre_onset_rate",
        "pre_duration", "pre_mean_intensity", "pre_max_intensity", "pre_cumulative_intensity","post_start_idx", "post_end_idx",
        "post_offset_rate",
        "post_duration", "post_mean_intensity", "post_max_intensity", "post_cumulative_intensity",
        "cum_int_sum",
        "ratio_pre_event_mean_int", "ratio_post_event_mean_int", "ratio_pre_post_mean_int",
        "ratio_pre_event_cum_int",  "ratio_post_event_cum_int",  "ratio_pre_post_cum_int",
         "ratio_sum_event_cum_int", "ratio_rate",
    ]
    df = df[[c for c in final_cols if c in df.columns]]
    return df.replace([np.inf, -np.inf], np.nan)


###############################################################################
# EXAMPLE USAGE
# Runs when the script is executed directly (python mhw_extended_events.py or %run in iPython)
if __name__ == '__main__':

    import pandas as pd
    
    df = pd.read_csv("test_sst_data.csv")   
    
    sst = df["sst"].to_numpy(dtype=float)
    t = pd.to_datetime(df["time"]).map(pd.Timestamp.toordinal).to_numpy()
    
    _, df_extended_events = detect_mhw_extended(t, sst,climatology_period = [2003, 2023],detrend_sst= True,label= "test_sst")
    
    print(df_extended_events[[
        "start_date", "category",
        "pre_duration", "event_duration", "post_duration",
        "pre_mean_intensity", "event_mean_intensity", "post_mean_intensity",
    ]].to_string(index=False))

    df_extended_events.to_csv("teste_output_extended_events.csv", index=False)
