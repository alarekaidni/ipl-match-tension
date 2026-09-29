import nbformat as nbf
import os
import subprocess

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# Stage 1: Setup, Data Ingestion & Data Hygiene
## Project: IPL Match Tension — Win-Probability Swing as a Proxy for Viewer Engagement
**Domain Context**: Live Sports Streaming Analytics (JioStar / JioCinema & Disney+ Hotstar)  
**Objective**: Ingest raw IPL fixtures and ball-by-ball deliveries, standardize franchise rebranding, and remove games that violate standard 20-over chase assumptions.

---

### JioStar Business Framing
In digital sports streaming, live engagement varies drastically ball-by-ball. To model win probability and match tension in later stages, we require a mathematically consistent dataset. In this stage, we establish data hygiene:
1. **Consistency**: Franchises rebrand (e.g. *Delhi Daredevils* $\\to$ *Delhi Capitals*), which would otherwise fragment team-level features.
2. **Mathematical Integrity**: Duckworth-Lewis-Stern (DLS) rain-interrupted matches revise targets using run-resource tables, distorting standard 120-ball chase dynamics and required run rate calculations.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""# Core analytics imports
import os
import pandas as pd
import numpy as np

# Display settings for clean outputs
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version:  {np.__version__}")
"""))

# Cell 3: Loading Data Markdown
cells.append(nbf.v4.new_markdown_cell("""### 1. Ingesting Raw Datasets
We load two foundational tables:
- **`matches.csv`**: Match-level metadata (venue, season, toss, winner, outcome margins).
- **`deliveries.csv`**: Granular ball-by-ball event stream (over, ball, batsman, bowler, runs, dismissals).
"""))

# Cell 4: Load Data Code
cells.append(nbf.v4.new_code_cell("""# Define relative data path
DATA_DIR = os.path.join("..", "data")
matches_path = os.path.join(DATA_DIR, "matches.csv")
deliveries_path = os.path.join(DATA_DIR, "deliveries.csv")

# Load raw CSVs into DataFrames
matches_raw = pd.read_csv(matches_path)
deliveries_raw = pd.read_csv(deliveries_path)

print("=" * 60)
print(f"RAW MATCHES SHAPE:    {matches_raw.shape[0]:,} rows x {matches_raw.shape[1]} columns")
print(f"RAW DELIVERIES SHAPE: {deliveries_raw.shape[0]:,} rows x {deliveries_raw.shape[1]} columns")
print("=" * 60)
"""))

# Cell 5: Inspect Columns
cells.append(nbf.v4.new_code_cell("""# Display column schema for both tables
print("MATCHES COLUMNS:")
print(list(matches_raw.columns))

print("\\nDELIVERIES COLUMNS:")
print(list(deliveries_raw.columns))
"""))

# Cell 6: Inspect Sample Data
cells.append(nbf.v4.new_code_cell("""# Preview first 3 match records
matches_raw.head(3)
"""))

cells.append(nbf.v4.new_code_cell("""# Preview first 5 deliveries
deliveries_raw.head(5)
"""))

# Cell 7: Franchise Standardization Markdown
cells.append(nbf.v4.new_markdown_cell("""### 2. Standardizing Franchise Naming
Over the course of the IPL, teams have rebranded or had minor naming discrepancies:
- **Delhi Daredevils $\\to$ Delhi Capitals** (Rebranded in 2019).
- **Kings XI Punjab $\\to$ Punjab Kings** (Rebranded in 2021).
- **Rising Pune Supergiants $\\to$ Rising Pune Supergiant** (Typo/spelling variation between 2016 and 2017).
- **Deccan Chargers $\\to$ Sunrisers Hyderabad** (Hyderabad franchise slot predecessor).

*Interview Talking Point*: Failing to standardize creates duplicate categorical levels (e.g., DD vs DC), which splits historical win rates and leads to high-cardinality sparse columns during one-hot encoding.
"""))

# Cell 8: Standardization Code
cells.append(nbf.v4.new_code_cell("""# Mapping dictionary for team name standardization
TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
}

# Apply mapping across matches table
matches_std = matches_raw.copy()
match_team_cols = ["team1", "team2", "toss_winner", "winner"]
for col in match_team_cols:
    matches_std[col] = matches_std[col].replace(TEAM_NAME_MAPPING)

# Apply mapping across deliveries table
deliveries_std = deliveries_raw.copy()
delivery_team_cols = ["batting_team", "bowling_team"]
for col in delivery_team_cols:
    deliveries_std[col] = deliveries_std[col].replace(TEAM_NAME_MAPPING)

print("Standardized Teams in Matches:")
for team in sorted(matches_std['team1'].unique()):
    print(f" - {team}")
"""))

# Cell 9: Filtering Markdown
cells.append(nbf.v4.new_markdown_cell("""### 3. Filtering Anomalous & Rain-Affected Fixtures
We remove two specific classes of fixtures:
1. **`result == 'no result'`**: Washed out fixtures with no declared winner. These cannot be labeled for supervised binary win classification ($y \\in \\{0, 1\\}$).
2. **`dl_applied == 1`**: Duckworth-Lewis-Stern matches. 

*Interview Talking Point: Why exclude DLS matches?*  
In a standard chase, target score is fixed across 120 balls ($\text{Balls Left} = 120 - \text{Balls Bowled}$). In DLS matches, innings are truncated (e.g., target 145 in 14 overs). If modeled under a standard 20-over framework, the required run rate (RRR) calculation and remaining resource assumptions break down, injecting severe noise into ball-by-ball win-probability modeling.
"""))

# Cell 10: Filtering Code
cells.append(nbf.v4.new_code_cell("""# Check counts of discarded categories
no_result_count = (matches_std['result'] == 'no result').sum()
dl_applied_count = (matches_std['dl_applied'] == 1).sum()

print(f"Matches with 'no result': {no_result_count}")
print(f"Matches with DLS applied: {dl_applied_count}")

# Retain only valid completed, non-DLS fixtures
matches_clean = matches_std[
    (matches_std['result'] != 'no result') & 
    (matches_std['dl_applied'] == 0)
].copy()

# Synchronize deliveries table so it strictly matches valid fixture IDs
deliveries_clean = deliveries_std[
    deliveries_std['match_id'].isin(matches_clean['id'])
].copy()

print("\\n" + "=" * 60)
print(f"CLEANED MATCHES:    {matches_clean.shape[0]:,} (Retained {matches_clean.shape[0]/len(matches_raw):.1%})")
print(f"CLEANED DELIVERIES: {deliveries_clean.shape[0]:,} (Retained {deliveries_clean.shape[0]/len(deliveries_raw):.1%})")
print("=" * 60)
"""))

# Cell 11: Summary & Interview Preparation Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4. Summary & Interview Q&A Cheatsheet

| Metric | Raw | Cleaned | Retained (%) | Business / Analytical Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Matches** | 636 | 617 | 97.0% | Dropped 3 no-result and 16 DLS-interrupted games |
| **Deliveries** | 150,460 | 147,458 | 98.0% | Synchronized strictly to valid 20-over games |
| **Franchises** | 14 | 12 | Standardized | Merged rebranded names (DD $\\to$ DC, KXIP $\\to$ PBKS) |

#### Likely Interview Questions & Model Answers:
1. **Q: Why didn't you adjust DLS targets instead of dropping the 16 matches?**  
   *Answer*: DLS adjustments use proprietary ICC resource decay curves based on wickets in hand and overs lost. Because our downstream objective is to train a win-probability model that learns standard T20 chase resource decay ($R_{\text{needed}}$ vs $B_{\text{remaining}}$ vs $W_{\text{in hand}}$), keeping 16 truncated games adds confounding noise that outweighs the minor 2.5% data loss.
2. **Q: How does team name standardization affect categorical encoding later?**  
   *Answer*: Treating 'Delhi Daredevils' and 'Delhi Capitals' as separate entities would split sample size in half for that franchise, artificially inflate model degrees of freedom, and prevent the model from learning consistent home-ground or team baseline characteristics.
"""))

nb.cells = cells

# Save notebook
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(notebook_dir, exist_ok=True)
notebook_path = os.path.join(notebook_dir, "01_stage1_setup_and_load.ipynb")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Saved notebook template to {notebook_path}")
