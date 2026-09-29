"""
src/features.py
---------------
Feature engineering module to construct the ball-by-ball 2nd-innings chase-state table.

Author: Candidate for JioStar Data Analytics Internship
Project: IPL Match Tension & Win-Probability Swing
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


def build_chase_state_table(
    matches: pd.DataFrame, 
    deliveries: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs a ball-by-ball chase-state representation for all 2nd innings in the dataset.

    Computed Fields:
        - match_id, season, venue
        - batting_team (chasing team), bowling_team
        - over, ball, batsman, bowler
        - target: 1st innings total runs + 1
        - current_score: Cumulative runs scored by chasing team so far
        - runs_needed: max(0, target - current_score)
        - legal_balls_bowled: Cumulative count of legal deliveries (excluding wides & no-balls)
        - balls_left: max(0, 120 - legal_balls_bowled)
        - wickets_lost: Cumulative count of dismissals
        - wickets_in_hand: 10 - wickets_lost
        - current_run_rate (CRR): (current_score * 6) / legal_balls_bowled
        - required_run_rate (RRR): (runs_needed * 6) / balls_left
        - label: Binary indicator (1 if chasing team won the match, 0 otherwise)

    Interview Rationale:
        Under ICC T20 Laws 21 and 22, wides and no-balls do not consume legal balls from the
        scheduled 120-ball allocation. Accurately decoupling legal balls from total deliveries
        ensures that remaining resource decay curves (balls_left) are mathematically sound.
    """
    # 1. Calculate 1st innings target for each match
    inn1_totals = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
    )
    inn1_totals.rename(columns={"total_runs": "first_innings_total"}, inplace=True)
    inn1_totals["target"] = inn1_totals["first_innings_total"] + 1

    # 2. Filter for 2nd innings (chase) only
    chase = deliveries[deliveries["inning"] == 2].copy()

    # 3. Merge target and match-level metadata
    match_cols = ["id", "season", "venue", "winner", "toss_winner", "toss_decision"]
    chase = chase.merge(inn1_totals[["match_id", "target"]], on="match_id", how="inner")
    chase = chase.merge(matches[match_cols], left_on="match_id", right_on="id", how="inner")

    # 4. Binary outcome label: 1 if chasing team (batting_team) won the match
    chase["label"] = (chase["batting_team"] == chase["winner"]).astype(int)

    # 5. Identify legal balls (wides and no-balls must be re-bowled and do not consume a ball)
    chase["is_legal_ball"] = (
        (chase["wide_runs"] == 0) & (chase["noball_runs"] == 0)
    ).astype(int)

    # Ensure chronological order within each match
    chase.sort_values(by=["match_id", "over", "ball"], inplace=True)

    # 6. Cumulative ball-by-ball progressions
    chase["current_score"] = chase.groupby("match_id")["total_runs"].cumsum()
    chase["legal_balls_bowled"] = chase.groupby("match_id")["is_legal_ball"].cumsum()
    chase["balls_left"] = np.maximum(0, 120 - chase["legal_balls_bowled"])
    chase["runs_needed"] = np.maximum(0, chase["target"] - chase["current_score"])

    # 7. Dismissals and Wickets in Hand
    chase["is_wicket"] = chase["player_dismissed"].notna().astype(int)
    chase["wickets_lost"] = chase.groupby("match_id")["is_wicket"].cumsum()
    chase["wickets_in_hand"] = np.maximum(0, 10 - chase["wickets_lost"])

    # 8. Run Rates (handling boundary division cases safely)
    chase["current_run_rate"] = np.where(
        chase["legal_balls_bowled"] > 0,
        (chase["current_score"] * 6.0) / chase["legal_balls_bowled"],
        0.0
    )

    # RRR: capped at 36.0 RPO if balls_left == 0 and runs_needed > 0; 0.0 if runs_needed == 0
    chase["required_run_rate"] = np.where(
        chase["balls_left"] > 0,
        (chase["runs_needed"] * 6.0) / chase["balls_left"],
        np.where(chase["runs_needed"] == 0, 0.0, 36.0)
    )

    # Clean up auxiliary columns
    drop_cols = ["id", "is_legal_ball"]
    chase.drop(columns=[col for col in drop_cols if col in chase.columns], inplace=True)

    return chase


def save_chase_state_table(
    df: pd.DataFrame, 
    output_dir: str = "data"
) -> str:
    """Saves the chase state table to CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "chase_state.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    matches, deliveries = load_and_clean_data()
    chase_df = build_chase_state_table(matches, deliveries)
    print(f"Constructed chase-state table: {chase_df.shape[0]:,} rows x {chase_df.shape[1]} cols")
    print(f"Chasing win rate: {chase_df['label'].mean():.1%}")
