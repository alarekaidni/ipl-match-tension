import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Header
cells.append(nbf.v4.new_markdown_cell("""# Stage 3: Ball-by-Ball Chase-State Feature Engineering
## Project: IPL Match Tension — Win-Probability Swing as a Proxy for Viewer Engagement
**Domain Context**: Live Sports Streaming (JioStar / JioCinema & Disney+ Hotstar)  
**Objective**: Transform the granular 2nd-innings ball-by-ball event stream into a mathematically sound **Chase-State Representation** for downstream win-probability modeling.

---

### The State-Space Formulation of a T20 Chase
In a T20 chase, cricket can be modeled as a finite-horizon dynamic game. At every legal ball $t$, the match state is parameterized by:
$$\\mathcal{S}_t = \\left( R_{\\text{needed}}, B_{\\text{left}}, W_{\\text{in hand}}, CRR, RRR, \\text{Venue}, \\text{Batting Team}, \\text{Bowling Team} \\right)$$

Our objective is to compute these state features for all **71,363 deliveries** in the 2nd innings of valid IPL fixtures.
"""))

# Cell 2: Code - Imports & Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.abspath(".."))
from src.data_loader import load_and_clean_data
from src.features import build_chase_state_table

# Load cleaned data
matches, deliveries = load_and_clean_data(data_dir=os.path.join("..", "data"))
print(f"Loaded {len(matches)} cleaned matches and {len(deliveries):,} cleaned deliveries.")
"""))

# Cell 3: Markdown - ICC Laws & Legal Balls
cells.append(nbf.v4.new_markdown_cell("""### 1. Handling Legal vs. Illegal Balls (ICC Laws 21 & 22)
**Interview Talking Point**: Why can't we simply calculate $B_{\\text{left}} = 120 - \\text{delivery\\_index}$?  
In cricket, **wides and no-balls do not count as legal deliveries** from the 120-ball quota. If a bowler bowls 3 wides in an over, 9 deliveries are recorded in `deliveries.csv`, but only 6 legal balls were bowled.

$$\\text{is\\_legal\\_ball} = \\mathbf{1}_{\\{\\text{wide\\_runs} = 0 \\land \\text{noball\\_runs} = 0\\}}$$
$$\\text{balls\\_left} = \\max(0, 120 - \\sum \\text{is\\_legal\\_ball})$$
"""))

# Cell 4: Code - Construct Chase State Table
cells.append(nbf.v4.new_code_cell("""# Build the comprehensive chase state table
chase_df = build_chase_state_table(matches, deliveries)

print("=" * 65)
print(f"CHASE STATE TABLE GENERATED: {chase_df.shape[0]:,} rows x {chase_df.shape[1]} columns")
print(f"Chasing Teams Won:           {chase_df['label'].mean():.2%}")
print(f"Defending Teams Won:         {(1 - chase_df['label'].mean()):.2%}")
print("=" * 65)
"""))

# Cell 5: Markdown - Inspect Core Features
cells.append(nbf.v4.new_markdown_cell("""### 2. Inspecting the Feature Table
Below is a ball-by-ball preview of the first 10 balls of a chase, demonstrating how:
- `runs_needed` decrements with runs scored.
- `balls_left` decrements only on legal balls.
- `wickets_in_hand` starts at 10 and decrements on dismissals.
- `current_run_rate` ($CRR$) and `required_run_rate` ($RRR$) dynamically update.
"""))

# Cell 6: Code - Display Preview
cells.append(nbf.v4.new_code_cell("""core_cols = [
    'match_id', 'batting_team', 'over', 'ball', 'total_runs',
    'current_score', 'target', 'runs_needed', 'balls_left', 
    'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'label'
]

chase_df[core_cols].head(10)
"""))

# Cell 7: Markdown - Boundary Conditions & Numerical Hygiene
cells.append(nbf.v4.new_markdown_cell("""### 3. Handling Numerical Edge Cases
In production ML pipelines, edge cases can cause silent failures or infinite values ($+\\infty$ / `NaN`):
1. **$B_{\\text{left}} = 0$**: When all 120 balls are bowled, standard $RRR = \\frac{R_{\\text{needed}} \\times 6}{B_{\\text{left}}}$ triggers division by zero. We set $RRR = 0.0$ if target is achieved, or cap at $36.0$ RPO if runs remain.
2. **$R_{\\text{needed}} \\le 0$**: When a team hits a 4 or 6 with 1 run needed, $R_{\\text{needed}}$ mathematically goes negative. We clip via $\\max(0, \\dots)$.
3. **$B_{\\text{bowled}} = 0$**: Before the first legal delivery (e.g. ball 1 is a wide), $CRR = 0.0$ to prevent `NaN`.
"""))

# Cell 8: Code - Verification of Boundary Checks
cells.append(nbf.v4.new_code_cell("""# Sanity checks on engineered features
print("CHECKING FOR NULLS IN KEY FEATURES:")
print(chase_df[['runs_needed', 'balls_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'label']].isnull().sum())

print("\\nFEATURE DISTRIBUTIONS & RANGES:")
stats = chase_df[['runs_needed', 'balls_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate']].describe().round(2)
stats
"""))

# Cell 9: Code - Correlation with Match Outcome
cells.append(nbf.v4.new_code_cell("""# Bivariate correlation of chase-state features with final win outcome
corr = chase_df[['runs_needed', 'balls_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'label']].corr()['label'].sort_values(ascending=False)

print("CORRELATION WITH CHASING TEAM WIN (label):")
for feat, val in corr.items():
    if feat != 'label':
        print(f" {feat:<20}: {val:+.4f}")
"""))

# Cell 10: Markdown - Save Output & Interview Takeaways
cells.append(nbf.v4.new_markdown_cell("""### 4. Summary & Interview Q&A Cheatsheet

| Feature | Mathematical Definition | Intuition & Directional Effect on $P(\\text{win})$ |
| :--- | :--- | :--- |
| **`runs_needed`** | $\\max(0, \\text{Target} - \\text{Score}_t)$ | Negative correlation ($-0.27$): As runs needed grow, win probability drops. |
| **`balls_left`** | $\\max(0, 120 - \\sum \\text{legal\\_balls})$ | Positive correlation ($+0.08$): Remaining resource budget. |
| **`wickets_in_hand`** | $10 - \\sum \\text{dismissals}$ | Strong positive correlation ($+0.34$): Highest-leverage resource in T20 chases. |
| **`required_run_rate`** | $(R_{\\text{needed}} \\times 6) / B_{\\text{left}}$ | Strong negative correlation ($-0.43$): Measures the instantaneous pressure hurdle. |
| **`current_run_rate`** | $(\\text{Score}_t \\times 6) / B_{\\text{bowled}}$ | Positive correlation ($+0.23$): Momentum and scoring velocity baseline. |

#### Likely Interview Questions & Model Answers:
1. **Q: Why is Required Run Rate ($RRR$) a better predictor than Raw Target?**  
   *Answer*: Raw target is static ($T$). $RRR$ is non-linear and dynamic — it adjusts for time decay. Chasing 180 is comfortable when you need 30 off 30 ($RRR = 6.0$), but disastrous if you need 50 off 18 ($RRR = 16.7$).
2. **Q: Why did you separate legal balls from wides/no-balls?**  
   *Answer*: Under ICC Laws, extra deliveries do not decrement the 120-ball allotment. Conflating delivery index with balls bowled would artificially underestimate balls remaining, distorting $RRR$ decay curves.
"""))

nb.cells = cells

# Save notebook
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
notebook_path = os.path.join(notebook_dir, "03_stage3_chase_state.ipynb")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Saved Stage 3 Notebook template to {notebook_path}")
