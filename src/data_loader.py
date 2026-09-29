"""
src/data_loader.py
------------------
Module for loading, standardizing, and cleaning IPL match and ball-by-ball delivery data.

Author: Candidate for JioStar Data Analytics Internship
Project: IPL Match Tension & Win-Probability Swing
"""

import os
import pandas as pd
from typing import Tuple, Dict

# Franchise name changes and spelling normalizations across IPL seasons
TEAM_NAME_MAPPING: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",              # Rebranded in 2019
    "Kings XI Punjab": "Punjab Kings",                  # Rebranded in 2021
    "Rising Pune Supergiants": "Rising Pune Supergiant",# Spelling variation in 2016 vs 2017
    "Deccan Chargers": "Sunrisers Hyderabad",           # Hyderabad franchise continuation (2013 onwards)
}


def load_raw_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw matches.csv and deliveries.csv from the specified data directory.
    
    Args:
        data_dir: Path to directory containing CSV files.
        
    Returns:
        Tuple of (matches_df, deliveries_df)
    """
    matches_path = os.path.join(data_dir, "matches.csv")
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    
    if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
        raise FileNotFoundError(
            f"Missing dataset files in '{data_dir}'. "
            "Please run 'python scripts/download_data.py' first."
        )
        
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    return matches, deliveries


def standardize_team_names(
    matches: pd.DataFrame, 
    deliveries: pd.DataFrame, 
    mapping: Dict[str, str] = TEAM_NAME_MAPPING
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes franchise names across all relevant columns in matches and deliveries.
    
    Interview Rationale:
    Franchise rebranding (e.g., Delhi Daredevils -> Delhi Capitals) creates duplicate
    categorical levels for the same underlying team. Standardizing avoids sparse categories
    and ensures team-level features carry consistent historical context.
    """
    matches_clean = matches.copy()
    deliveries_clean = deliveries.copy()
    
    # Standardize columns in matches
    match_team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in match_team_cols:
        if col in matches_clean.columns:
            matches_clean[col] = matches_clean[col].replace(mapping)
            
    # Standardize columns in deliveries
    delivery_team_cols = ["batting_team", "bowling_team"]
    for col in delivery_team_cols:
        if col in deliveries_clean.columns:
            deliveries_clean[col] = deliveries_clean[col].replace(mapping)
            
    return matches_clean, deliveries_clean


def filter_valid_matches(
    matches: pd.DataFrame, 
    deliveries: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters out 'no result' and Duckworth-Lewis-Stern (DLS) affected matches.
    
    Interview Rationale:
    1. 'no result' matches have no winner, making binary win-probability modeling undefined.
    2. DLS-affected matches have reduced overs and revised targets calculated by an external table.
       Including them distorts standard 20-over required run rates and ball countdown metrics,
       introducing severe synthetic noise into the win-probability model.
    """
    # Filter matches: keep only normal/tie results and non-DLS games
    valid_matches = matches[
        (matches["result"] != "no result") & 
        (matches["dl_applied"] == 0)
    ].copy()
    
    # Synchronize deliveries: retain only deliveries belonging to valid matches
    valid_deliveries = deliveries[
        deliveries["match_id"].isin(valid_matches["id"])
    ].copy()
    
    return valid_matches, valid_deliveries


def load_and_clean_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline wrapper: loads raw data, standardizes team names, and filters invalid games.
    """
    matches_raw, deliveries_raw = load_raw_data(data_dir)
    matches_std, deliveries_std = standardize_team_names(matches_raw, deliveries_raw)
    matches_clean, deliveries_clean = filter_valid_matches(matches_std, deliveries_std)
    return matches_clean, deliveries_clean


if __name__ == "__main__":
    print("Testing data_loader pipeline...")
    matches, deliveries = load_and_clean_data()
    print(f"Cleaned matches shape: {matches.shape}")
    print(f"Cleaned deliveries shape: {deliveries.shape}")
    print(f"Unique teams: {sorted(matches['team1'].unique())}")
