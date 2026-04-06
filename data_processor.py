"""
Data Processor — Loads CSV data, processes and aggregates conflict, piracy,
and sanctions data per route and per quarter for GRV calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from route_engine import ROUTES, is_point_in_route_corridor, get_route_country_codes, get_all_route_keys

DATA_DIR = Path(__file__).parent / "dataset"


def load_conflict_data():
    """Load and parse conflict dataset."""
    df = pd.read_csv(DATA_DIR / "sheet_1_global_conflict_1995_2025.csv")
    df["date"] = pd.to_datetime(df["date_int"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["year_quarter"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    return df


def load_piracy_data():
    """Load and parse piracy dataset."""
    df = pd.read_csv(DATA_DIR / "sheet_2_global_piracy_1995_2025.csv")
    df["date"] = pd.to_datetime(df["date_int"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["year_quarter"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    return df


def load_sanctions_data():
    """Load and parse sanctions dataset."""
    df = pd.read_csv(DATA_DIR / "sheet_3_global_sanctions_1995_2025.csv")
    df["date_imposed_dt"] = pd.to_datetime(df["date_imposed"].astype(str), format="%Y%m%d", errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────
# Route-level aggregation using VECTORIZED operations
# ─────────────────────────────────────────────────────────────

def _match_events_to_route_vectorized(lats, lons, route_key):
    """
    Vectorized matching of events to a route using bounding boxes.
    Returns a boolean mask.
    """
    route = ROUTES[route_key]
    mask = np.zeros(len(lats), dtype=bool)
    
    for (lat_min, lat_max, lon_min, lon_max) in route["bounding_boxes"]:
        box_match = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        mask |= box_match
    
    return mask


def aggregate_conflict_per_route(conflict_df, sample_size=None):
    """
    Aggregate conflict data per route.
    Uses vectorized bounding-box matching for speed.
    Returns a dict of route_key -> DataFrame with quarterly aggregated features.
    """
    if sample_size and len(conflict_df) > sample_size:
        df = conflict_df.sample(n=sample_size, random_state=42)
    else:
        df = conflict_df
    
    lats = df["latitude"].values
    lons = df["longitude"].values
    
    route_data = {}
    for route_key in get_all_route_keys():
        mask = _match_events_to_route_vectorized(lats, lons, route_key)
        matched = df[mask]
        
        if len(matched) > 0:
            agg = matched.groupby("year_quarter").agg(
                conflict_count=("conflict_id", "count"),
                conflict_fatalities=("fatalities", "sum"),
                conflict_avg_fatalities=("fatalities", "mean"),
                conflict_max_fatalities=("fatalities", "max"),
            ).reset_index()
        else:
            agg = pd.DataFrame(columns=["year_quarter", "conflict_count", "conflict_fatalities",
                                         "conflict_avg_fatalities", "conflict_max_fatalities"])
        
        route_data[route_key] = agg
    
    return route_data


def aggregate_piracy_per_route(piracy_df):
    """
    Aggregate piracy data per route.
    Returns a dict of route_key -> DataFrame with quarterly aggregated features.
    """
    lats = piracy_df["latitude"].values
    lons = piracy_df["longitude"].values
    
    route_data = {}
    for route_key in get_all_route_keys():
        mask = _match_events_to_route_vectorized(lats, lons, route_key)
        matched = piracy_df[mask]
        
        if len(matched) > 0:
            agg = matched.groupby("year_quarter").agg(
                piracy_count=("incident_id", "count"),
                piracy_casualties=("severity_casualties", "sum"),
                piracy_avg_casualties=("severity_casualties", "mean"),
                piracy_economic_loss=("severity_economic_usd", "sum"),
            ).reset_index()
        else:
            agg = pd.DataFrame(columns=["year_quarter", "piracy_count", "piracy_casualties",
                                         "piracy_avg_casualties", "piracy_economic_loss"])
        
        route_data[route_key] = agg
    
    return route_data


def aggregate_sanctions_per_route(sanctions_df):
    """
    Aggregate sanctions data per route based on country codes.
    Returns a dict of route_key -> sanctions severity score.
    """
    route_sanctions = {}
    for route_key in get_all_route_keys():
        country_codes = get_route_country_codes(route_key)
        matched = sanctions_df[
            sanctions_df["country_code"].isin(country_codes) &
            (sanctions_df["active_flag"] == 1)
        ]
        
        if len(matched) > 0:
            route_sanctions[route_key] = {
                "sanctions_count": len(matched),
                "sanctions_severity_sum": matched["severity_score"].sum(),
                "sanctions_severity_mean": matched["severity_score"].mean(),
                "sanctions_economic": matched["sanction_type_economic"].sum(),
                "sanctions_complete": matched["sanction_type_complete"].sum(),
            }
        else:
            route_sanctions[route_key] = {
                "sanctions_count": 0,
                "sanctions_severity_sum": 0,
                "sanctions_severity_mean": 0,
                "sanctions_economic": 0,
                "sanctions_complete": 0,
            }
    
    return route_sanctions


# ─────────────────────────────────────────────────────────────
# Full data processing pipeline
# ─────────────────────────────────────────────────────────────

def compute_route_scores(conflict_df, piracy_df, sanctions_df, sample_conflict=50000):
    """
    Compute normalized risk scores for each route across all 3 dimensions.
    Returns a DataFrame with one row per route and columns for each score.
    """
    # Aggregate per route
    conflict_agg = aggregate_conflict_per_route(conflict_df, sample_size=sample_conflict)
    piracy_agg = aggregate_piracy_per_route(piracy_df)
    sanctions_agg = aggregate_sanctions_per_route(sanctions_df)
    
    route_scores = []
    for route_key in get_all_route_keys():
        route = ROUTES[route_key]
        
        # Conflict metrics
        c_agg = conflict_agg[route_key]
        if len(c_agg) > 0:
            conflict_total_events = c_agg["conflict_count"].sum()
            conflict_total_fatalities = c_agg["conflict_fatalities"].sum()
            conflict_intensity = conflict_total_fatalities / max(conflict_total_events, 1)
        else:
            conflict_total_events = 0
            conflict_total_fatalities = 0
            conflict_intensity = 0
        
        # Piracy metrics
        p_agg = piracy_agg[route_key]
        if len(p_agg) > 0:
            piracy_total_events = p_agg["piracy_count"].sum()
            piracy_total_casualties = p_agg["piracy_casualties"].sum()
            piracy_severity = piracy_total_casualties / max(piracy_total_events, 1)
        else:
            piracy_total_events = 0
            piracy_total_casualties = 0
            piracy_severity = 0
        
        # Sanctions metrics
        s_data = sanctions_agg[route_key]
        sanctions_count = s_data["sanctions_count"]
        sanctions_severity = s_data["sanctions_severity_sum"]
        
        route_scores.append({
            "route_key": route_key,
            "route_name": route["name"],
            "short_name": route["short_name"],
            "distance_nm": route.get("distance_nm", 0),
            "conflict_events": conflict_total_events,
            "conflict_fatalities": conflict_total_fatalities,
            "conflict_intensity": conflict_intensity,
            "piracy_events": piracy_total_events,
            "piracy_casualties": piracy_total_casualties,
            "piracy_severity": piracy_severity,
            "sanctions_count": sanctions_count,
            "sanctions_severity": sanctions_severity,
        })
    
    df = pd.DataFrame(route_scores)
    
    # Normalize scores to 0-1 scale using min-max normalization
    for col in ["conflict_events", "conflict_fatalities", "conflict_intensity",
                 "piracy_events", "piracy_casualties", "piracy_severity",
                 "sanctions_count", "sanctions_severity"]:
        col_max = df[col].max()
        if col_max > 0:
            df[col + "_norm"] = df[col] / col_max
        else:
            df[col + "_norm"] = 0
    
    # Composite sub-scores (0-1)
    df["conflict_score"] = (
        0.4 * df["conflict_events_norm"] +
        0.4 * df["conflict_fatalities_norm"] +
        0.2 * df["conflict_intensity_norm"]
    )
    
    df["piracy_score"] = (
        0.4 * df["piracy_events_norm"] +
        0.3 * df["piracy_casualties_norm"] +
        0.3 * df["piracy_severity_norm"]
    )
    
    df["sanctions_score"] = (
        0.5 * df["sanctions_count_norm"] +
        0.5 * df["sanctions_severity_norm"]
    )
    
    return df


def compute_grv(route_scores_df, weights, commodity_key=None):
    """
    Calculate GRV (1-10 scale) for each route.
    
    When a commodity is selected, only routes listed in the commodity's 
    'applicable_routes' are included — ensuring we only show routes 
    where that commodity is actually shipped.
    
    Args:
        route_scores_df: DataFrame from compute_route_scores()
        weights: dict with keys 'conflict', 'piracy', 'sanctions' (should sum to 1.0)
        commodity_key: optional commodity for sensitivity adjustment
    
    Returns:
        DataFrame with GRV column added, filtered to applicable routes
    """
    from route_engine import get_commodity_sensitivity, COMMODITIES
    
    df = route_scores_df.copy()
    
    # Filter to only applicable routes for this commodity
    if commodity_key and commodity_key in COMMODITIES:
        commodity = COMMODITIES[commodity_key]
        applicable = commodity.get("applicable_routes", None)
        if applicable:
            df = df[df["route_key"].isin(applicable)].copy()
    
    df["grv_raw"] = (
        weights["conflict"] * df["conflict_score"] +
        weights["piracy"] * df["piracy_score"] +
        weights["sanctions"] * df["sanctions_score"]
    )
    
    # Apply commodity sensitivity as a blended modifier
    if commodity_key:
        df["commodity_multiplier"] = df["route_key"].apply(
            lambda rk: get_commodity_sensitivity(commodity_key, rk)
        )
        # Blended: base contribution (30%) + sensitivity-driven (70%)
        df["grv_adjusted"] = df["grv_raw"] * (0.3 + 0.7 * df["commodity_multiplier"])
    else:
        df["commodity_multiplier"] = 1.0
        df["grv_adjusted"] = df["grv_raw"]
    
    # Scale to 1-10
    max_grv = df["grv_adjusted"].max()
    if max_grv > 0:
        df["grv"] = (df["grv_adjusted"] / max_grv * 9 + 1).round(1)  # maps to 1-10
    else:
        df["grv"] = 1.0
    
    # Risk level categorization (1-10 scale)
    df["risk_level"] = pd.cut(
        df["grv"],
        bins=[0, 3, 5, 7.5, 10.1],
        labels=["Low", "Moderate", "High", "Critical"]
    )
    
    # Route recommendation (lower GRV = preferred)
    df["rank"] = df["grv"].rank(method="min").astype(int)
    
    return df.sort_values("grv", ascending=True)


def build_quarterly_features(conflict_df, piracy_df, sanctions_df, sample_conflict=50000):
    """
    Build quarterly feature matrix for ML model training.
    Each row = (quarter, route) with conflict/piracy/sanctions features.
    Used by the ML model to learn weight patterns.
    """
    conflict_agg = aggregate_conflict_per_route(conflict_df, sample_size=sample_conflict)
    piracy_agg = aggregate_piracy_per_route(piracy_df)
    sanctions_agg = aggregate_sanctions_per_route(sanctions_df)
    
    # Collect all unique quarters
    all_quarters = set()
    for route_key in get_all_route_keys():
        if len(conflict_agg[route_key]) > 0:
            all_quarters.update(conflict_agg[route_key]["year_quarter"].tolist())
        if len(piracy_agg[route_key]) > 0:
            all_quarters.update(piracy_agg[route_key]["year_quarter"].tolist())
    
    all_quarters = sorted(all_quarters)
    
    rows = []
    for quarter in all_quarters:
        for route_key in get_all_route_keys():
            row = {"year_quarter": quarter, "route_key": route_key}
            
            # Conflict features
            c_df = conflict_agg[route_key]
            c_q = c_df[c_df["year_quarter"] == quarter]
            if len(c_q) > 0:
                row["conflict_count"] = c_q["conflict_count"].values[0]
                row["conflict_fatalities"] = c_q["conflict_fatalities"].values[0]
                row["conflict_avg_fatalities"] = c_q["conflict_avg_fatalities"].values[0]
            else:
                row["conflict_count"] = 0
                row["conflict_fatalities"] = 0
                row["conflict_avg_fatalities"] = 0
            
            # Piracy features
            p_df = piracy_agg[route_key]
            p_q = p_df[p_df["year_quarter"] == quarter]
            if len(p_q) > 0:
                row["piracy_count"] = p_q["piracy_count"].values[0]
                row["piracy_casualties"] = p_q["piracy_casualties"].values[0]
                row["piracy_economic_loss"] = p_q.get("piracy_economic_loss", pd.Series([0])).values[0]
            else:
                row["piracy_count"] = 0
                row["piracy_casualties"] = 0
                row["piracy_economic_loss"] = 0
            
            # Sanctions score (static — snapshot)
            s_data = sanctions_agg[route_key]
            row["sanctions_count"] = s_data["sanctions_count"]
            row["sanctions_severity"] = s_data["sanctions_severity_sum"]
            
            rows.append(row)
    
    return pd.DataFrame(rows)
