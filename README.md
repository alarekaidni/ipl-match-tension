# IPL Match Tension: Win-Probability Swing as a Proxy for Viewer Engagement Moments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Sports OTT Analytics](https://img.shields.io/badge/Domain-Sports%20OTT%20Analytics-red.svg)]()

> **Target Application**: Data Analytics Internship / PS-II at **JioStar** (Streaming + Sports / Cricket Analytics).

---

## 📌 Executive Summary & Business Rationale

In high-concurrency sports streaming (such as **JioCinema** and **Disney+ Hotstar** streaming the Indian Premier League), platform engineering and monetization teams face a core operational challenge: **How do we identify peak match moments in real time to maximize live concurrency and revenue without degrading viewer experience?**

This project constructs a ball-by-ball **Win-Probability Model** for second-innings (chasing) IPL matches and calculates the **Win-Probability Swing ($\Delta P_{\text{win}}$)** produced by each delivery and over. 

We use this swing metric as a **quantitative proxy for match tension** to solve three streaming business problems:
1. **Smart Push Notification Triggers**: Dispatch push notifications to dormant/casual app users at moments of maximum volatility ($\Delta P_{\text{win}}$ spikes) rather than fixed-time intervals, driving app re-opens when climax is highest.
2. **Dynamic Ad-Break Placement**: Identify low-tension, predictable game states (near $P_{\text{win}} \approx 0$ or $1$) for serving high-CPM mid-roll/unskippable video ads without inducing viewer churn.
3. **Automated Highlight Generation**: Automatically flag deliveries with the highest absolute swings ($|\Delta P_{\text{win}}|$) to compile instant "Game Changer" clip packages for social and in-app feeds.

---

## ⚠️ Important Methodological Limitation & Boundary

> **No Claim of Direct Viewership Measurement**:  
> We do **not** possess proprietary streaming telemetry (e.g., live concurrent viewers, bitrate switches, or churn logs). Therefore, we make **no claim** of having measured real viewer engagement directly. Instead, **win-probability swing is evaluated strictly as a structural proxy for match tension** derived from on-field game dynamics. Viewership correlation is framed as an operational hypothesis for an internal OTT analytics pipeline.

---

## 🏗️ Repository Architecture

```text
ipl-match-tension/
├── .gitignore                         # Standard exclusions (.venv, caches, checkpoints)
├── requirements.txt                   # Reproducible package dependencies
├── README.md                          # Project documentation and interview defence
├── data/                              # Data directory
│   ├── matches.csv                    # Match-level records (636 fixtures)
│   └── deliveries.csv                 # Granular ball-by-ball event stream (150,460 deliveries)
├── notebooks/
│   ├── 01_stage1_setup_and_load.ipynb # Stage 1: Data ingestion, franchise standardization & hygiene
│   └── ...                            # Subsequent stages (EDA, Feature Table, Modeling, Swing)
├── scripts/
│   ├── download_data.py               # Automated dataset downloader
│   └── build_stage1_notebook.py       # Notebook generator & execution pipeline
└── src/
    ├── __init__.py
    ├── data_loader.py                 # Modular, production-ready data pipeline
    └── features.py                    # Chase-state feature engineering (Stage 3)
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ installed (or [`uv`](https://github.com/astral-sh/uv) package manager).

### Quickstart
```bash
# 1. Clone repository
git clone https://github.com/your-username/ipl-match-tension.git
cd ipl-match-tension

# 2. Set up virtual environment and install dependencies
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

uv pip install -r requirements.txt

# 3. Download data (if not already present in data/)
python scripts/download_data.py

# 4. Launch JupyterLab / Notebook
jupyter lab
```

---

## 🧭 Project Roadmap & Stages

- [x] **Stage 1: Setup, Load & Data Hygiene**
  - Downloaded canonical ball-by-ball IPL dataset.
  - Standardized franchise rebrands (*Delhi Daredevils $\to$ Delhi Capitals*, *Kings XI Punjab $\to$ Punjab Kings*, *Rising Pune Supergiants $\to$ Rising Pune Supergiant*, *Deccan Chargers $\to$ Sunrisers Hyderabad*).
  - Filtered 3 'no result' washouts and 16 Duckworth-Lewis-Stern (DLS) truncated matches to preserve standard 120-ball chase dynamics.
- [x] **Stage 2: Exploratory Data Analysis & Strategic Business Takeaways**
  - Generated 4 publication-quality charts saved to `reports/figures/`.
  - Discovered venue asymmetry: Chepauk favors batting first (64.6% bat-first wins), while Chinnaswamy/Sawai Mansingh heavily favor chasing (56–70% chase wins).
  - Uncovered the historical T20 chasing paradigm shift, peaking at 67.9% chasing wins in 2016.
  - Quantified phase scoring: Powerplay (7.50 RPO, 17.8% boundary rate), Middle overs (7.54 RPO, 12.7% boundary rate, lowest variance $\sigma=4.11$), Death overs (9.42 RPO, 18.6% boundary rate, 2x wicket hazard at 0.50/over).
  - Defined 3 JioStar business applications: Event-driven push notifications (Overs 15–18), dynamic ad breaks (Overs 7–14), and automated short-form highlight clipping.
- [x] **Stage 3: Chase-State Table Construction**
  - Built production feature module [`src/features.py`](src/features.py) generating 71,363 ball-by-ball chase records.
  - Decoupled legal balls from illegal extras (wides/no-balls) per ICC Laws 21 & 22 to ensure exact ball budget accounting.
  - Computed dynamic state metrics: `runs_needed`, `balls_left`, `wickets_in_hand`, `current_run_rate`, `required_run_rate`, `target`, `venue`, and binary `label`.
  - Implemented boundary protections (preventing division-by-zero at $B_{\text{left}}=0$ and negative runs needed).
  - Exported optimized dataset to `data/chase_state.parquet`.
- [x] **Stage 4: Win-Probability Modeling & Calibration**
  - Evaluated on a strict chronological season split (Train: 2008–2015 [58,366 balls], Test: 2016–2017 [12,997 balls]) preventing intra-match leakage.
  - Implemented Logistic Regression baseline (Accuracy: **76.6%**, Log-Loss: **0.4594**, Brier Score: **0.1545**, ROC-AUC: **0.8415**).
  - Implemented Regularized XGBoost (Accuracy: **74.6%**, Log-Loss: **0.4649**, Brier Score: **0.1564**, ROC-AUC: **0.8407**).
  - Evaluated Brier score and generated Reliability Diagrams ([`reports/figures/05_calibration_curves.png`](reports/figures/05_calibration_curves.png)), proving both models produce well-calibrated probabilities.
  - Persisted trained model artifacts in `models/` directory for downstream swing inference.
- [ ] **Stage 5: Match Tension Swing Analysis**
  - Calculate $\Delta P_{\text{win}}$ per ball; aggregate tension volatility per over.
  - Top 20 highest-tension matches and overs in IPL history.
- [ ] **Stage 6: Strategic Visuals & Production Recommendations**
  - Case studies of iconic chases (e.g. MI vs RR 2014, CSK finals).
  - Implementation blueprint for push notification triggers and ad-break placement.

---

## 🎯 Interview Cheatsheet: Stage 1 Key Takeaways

1. **Why drop DLS matches instead of adjusting targets?**  
   *DLS adjusts targets via proprietary non-linear resource curves. In a standard chase model, $R_{\text{needed}}$ and $B_{\text{left}}$ are calibrated against a 120-ball budget. Keeping 16 truncated games (e.g. 130 in 12 overs) injects synthetic noise into run-rate decay curves for only a 2.5% data gain.*

2. **Why standardize franchise names?**  
   *Franchises like Delhi Daredevils and Delhi Capitals represent the same geographical market and home pitch (Feroz Shah Kotla / Arun Jaitley Stadium). Keeping them separate artificially fractures sample size, introduces high-cardinality noise, and prevents the model from learning stadium-team baselines.*
