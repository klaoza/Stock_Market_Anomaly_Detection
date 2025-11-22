# Multi-Agent Stock Market Anomaly Detection System

**Project #3 - Data Mining Course**  
**Created by:** Islem Nasri  
**Instructor:** Dr-ing Rym Besrour  
**Academic Year:** 2025/2026

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Detection Methods](#detection-methods)
- [Understanding the Results](#understanding-the-results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements a sophisticated multi-agent anomaly detection system specifically designed for stock market data analysis. The system uses **unsupervised machine learning** to identify unusual patterns, extreme price movements, and abnormal trading behavior without requiring labeled training data.

### Key Objectives

- Detect anomalies in stock market data using multiple algorithms
- Provide explainable AI insights for detected anomalies
- Enable interactive exploration and visualization
- Compare different detection methods and create consensus results
- Generate comprehensive reports for further analysis

### Why Multi-Agent Architecture?

The system uses a **modular agent-based design** where each agent has a specific responsibility:
- **Separation of concerns**: Each agent handles one task
- **Modularity**: Easy to add/remove detection methods
- **Scalability**: Agents can run independently or in parallel
- **Maintainability**: Changes to one agent don't affect others

---

## System Architecture

The system consists of **14 specialized agents** orchestrated by a central coordinator:

### Core Agents

1. **CoordinatorAgent**: Orchestrates the entire pipeline and manages agent communication
2. **DataIngestionAgent**: Loads and validates CSV data
3. **FeatureEngineeringAgent**: Creates 75+ engineered features
4. **ScalerAgent**: Normalizes data using Standard or Robust scaling
5. **EmbeddingAgent**: Reduces dimensionality (PCA, t-SNE, UMAP)

### Detection Agents

6. **IsolationForestAgent**: Tree-based isolation detection
7. **LOFAgent**: Local density deviation detection
8. **DBSCANAgent**: Density-based clustering with auto-epsilon
9. **HDBSCANAgent**: Hierarchical density clustering with fallbacks

### Analysis Agents

10. **ScoreFusionAgent**: Combines results using voting strategies
11. **EvaluationAgent**: Compares methods and computes metrics
12. **EventSearchAgent**: Links anomalies to potential market events
13. **ReportingAgent**: Generates comprehensive CSV reports
14. **VisualizationAgent**: Creates interactive plots and charts

### Data Flow

```
Upload CSV → Ingestion → Feature Engineering → Scaling → Dimensionality Reduction
                                                               ↓
    Report ← Evaluation ← Score Fusion ← [Parallel Detection Methods]
                                         (IF, LOF, DBSCAN, HDBSCAN)
```

---

## Features

### Data Processing
- Automatic feature engineering (returns, volatility, momentum, trends)
- Multiple scaling methods (Standard, Robust, MinMax)
- Missing value handling and outlier management
- Date/time parsing and validation

### Anomaly Detection
- **4 Detection Methods**: Isolation Forest, LOF, DBSCAN, HDBSCAN
- Train/test split for generalization evaluation
- Automatic hyperparameter estimation (e.g., epsilon for DBSCAN)
- Multiple fallback strategies for robustness

### Visualization
- 2D and 3D scatter plots with train/test distinction
- Consensus heatmaps showing vote counts
- Method comparison bar charts
- Overlap matrices between methods
- Radar charts for feature profiles
- Interactive Plotly visualizations

### Explainability
- Method-specific descriptions
- Feature importance analysis
- Consensus voting mechanism
- Event context search for anomaly dates

### Reporting
- Comprehensive CSV export
- Original + engineered features
- Scores and labels from all methods
- Consensus voting results
- Filter and download options

---

## Installation

### Requirements

- Python 3.8+
- pip package manager

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd anomaly-detection-system

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly umap-learn shap
```

### Optional (for HDBSCAN):

```bash
pip install hdbscan
```

**Note**: HDBSCAN requires compilation. If installation fails, the system will work without it.

### Verify Installation

```bash
python -c "import streamlit, pandas, sklearn, plotly, umap; print('All dependencies installed!')"
```

---

## Usage

### Starting the Application

```bash
streamlit run app.py
```

This will open the application in your default web browser (typically at `http://localhost:8501`).

### Step-by-Step Guide

#### 1. Upload Data

- Click **"Upload Stock Data (CSV)"** in the sidebar
- Select your CSV file

**Expected CSV Format:**
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.5,102.3,99.8,101.2,1000000
2024-01-02,101.5,103.0,101.0,102.5,1200000
...
```

**Required columns**: Open, High, Low, Close, Volume  
**Optional columns**: Date/Timestamp

#### 2. Configure Preprocessing

- **Scaling Method**: Choose between Standard or Robust scaling
  - Standard: Good for normally distributed data
  - Robust: Better for data with outliers

#### 3. Select Dimensionality Reduction

- **Method**: PCA (fast), UMAP (preserves structure), or t-SNE (local structure)
- **Dimensions**: 2D for simple visualization, 3D for detailed exploration

#### 4. Choose Detection Methods

Select one or more methods:
- **Isolation Forest**: Fast, works well for global anomalies
- **LOF**: Good for local anomalies in varying density regions
- **DBSCAN**: Identifies outliers as points not in clusters
- **HDBSCAN**: Hierarchical version, more robust

#### 5. Adjust Hyperparameters

- **Contamination** (IF, LOF): Expected proportion of anomalies (0.01-0.30)
- **Min Samples** (DBSCAN, HDBSCAN): Minimum points to form a dense region
- **Test Size**: Percentage of data for testing (20-40%)

#### 6. Run Analysis

Click **"Run Analysis"** and wait for processing (typically 10-60 seconds depending on data size).

#### 7. Explore Results

Navigate through the tabs:
- **Overview**: System explanation and consensus heatmap
- **Individual Methods**: Detailed results per method
- **Consensus**: Strong anomalies detected by multiple methods
- **Comparison**: Method performance comparison
- **Events**: Potential market events linked to anomalies
- **Report**: Download comprehensive CSV report

---

## Detection Methods

### Isolation Forest

**How it works:**
1. Builds random decision trees
2. Anomalies are isolated faster (shorter paths)
3. Average path length indicates anomaly score

**Best for:** Global anomalies, fast detection

**Parameters:**
- `contamination`: Expected anomaly rate
- `n_estimators`: Number of trees (more = more accurate but slower)

### Local Outlier Factor (LOF)

**How it works:**
1. Calculates local density around each point
2. Compares point density to neighbor densities
3. Lower relative density = anomaly

**Best for:** Local anomalies in varying density regions

**Parameters:**
- `n_neighbors`: Number of neighbors for density calculation
- `contamination`: Expected anomaly rate

### DBSCAN

**How it works:**
1. Groups points into dense clusters
2. Points not in any cluster are anomalies
3. Auto-estimates epsilon using k-distance plot

**Best for:** Cluster-based detection, spatial anomalies

**Parameters:**
- `eps`: Auto-estimated (90th percentile of k-distances)
- `min_samples`: Minimum points to form a cluster

### HDBSCAN

**How it works:**
1. Hierarchical version of DBSCAN
2. Finds clusters of varying density
3. Multiple fallback strategies for robustness

**Best for:** Complex density structures, hierarchical patterns

**Parameters:**
- `min_cluster_size`: Minimum size of clusters (adaptive)
- `min_samples`: Core point threshold

---

## Understanding the Results

### Metrics Explained

**Anomaly Count**: Number of data points flagged as anomalies

**Anomaly Percentage**: Proportion of dataset that is anomalous
- **< 5%**: Conservative detection
- **5-15%**: Typical range
- **> 20%**: May need parameter adjustment

**Vote Count**: How many methods detected each point as anomaly
- **0 votes**: Normal point (all methods agree)
- **1 vote**: Weak anomaly (only one method)
- **2+ votes**: Moderate anomaly (consensus emerging)
- **All votes**: Strong anomaly (all methods agree)

### Consensus Strategies

1. **2+ Methods**: Detected by at least 2 methods (balanced)
2. **Majority**: Detected by more than half of methods (moderate)
3. **All Methods**: Detected by every method (very conservative)

### Interpreting Visualizations

**2D/3D Scatter Plots:**
- Blue points = Normal samples
- Red/Orange points = Anomalies
- Light colors = Training set
- Dark colors = Test set

**Consensus Heatmap:**
- White/Light = 0-1 votes (normal)
- Pink/Red = 2-3 votes (anomalies)
- Dark Red = All votes (strong anomalies)

**Overlap Matrix:**
- Diagonal = Total anomalies per method
- Off-diagonal = Shared anomalies between methods
- High overlap = Methods agree

---

## Project Structure

```
anomaly-detection-system/
│
├── app.py                      # Main application file
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── data/                       # Sample datasets (optional)
│   ├── sample_stock_data.csv
│   └── test_data.csv
│
├── outputs/                    # Generated reports (created automatically)
│   └── anomaly_report_*.csv
│
└── docs/                       # Additional documentation (optional)
    ├── architecture.md
    ├── algorithms.md
    └── user_guide.md
```

---

## Technical Details

### Feature Engineering

The system creates 75+ features including:

**Price Features:**
- Daily returns, log returns, absolute returns
- Intraday range, high/low ratio
- Price-to-moving-average ratios (5, 10, 20 periods)

**Volume Features:**
- Log-transformed volume
- Rolling volume statistics (mean, std)
- Volume z-scores across multiple windows

**Momentum Features:**
- Momentum over 5, 10, 20 periods
- Moving averages
- Trend change and acceleration

**Volatility Features:**
- Intraday range
- True range
- Rolling volatility

### Train/Test Split Methodology

**Important**: This is **NOT supervised learning**. The split evaluates:
1. **Generalization**: Can the model detect anomalies in unseen data?
2. **Consistency**: Do train and test anomalies have similar characteristics?
3. **Robustness**: Does the model overfit to training data?

**Process:**
1. Split data 70/30 (train/test)
2. Train model on 70% (unsupervised)
3. Predict on full dataset (100%)
4. Visualize which anomalies are from train vs test

### Computational Complexity

| Method | Training | Prediction | Memory |
|--------|----------|------------|--------|
| Isolation Forest | O(n log n) | O(log n) | Low |
| LOF | O(n²) | O(n) | Medium |
| DBSCAN | O(n log n) | O(n) | Low |
| HDBSCAN | O(n²) | O(n) | Medium |

**Recommended Dataset Sizes:**
- < 10,000 rows: All methods work well
- 10,000-50,000 rows: Use Isolation Forest or DBSCAN
- > 50,000 rows: Use Isolation Forest only

---

## Troubleshooting

### Common Issues

**1. "HDBSCAN not available"**
```bash
# Solution: Install HDBSCAN or proceed without it
pip install hdbscan

# Or use other methods
```

**2. "Memory Error"**
```
# Solution: Reduce dataset size or use Isolation Forest only
# Sample your data:
df_sample = df.sample(n=10000, random_state=42)
```

**3. "All points detected as anomalies"**
```
# Solution: Decrease contamination parameter
# Try: contamination = 0.05 instead of 0.1
```

**4. "No anomalies detected"**
```
# Solution: Increase contamination or check data quality
# Verify data has actual variance:
print(df.describe())
```

**5. "ValueError: perplexity must be less than n_samples"**
```
# Solution: Use PCA instead of t-SNE for small datasets
# Or reduce perplexity in the code
```

### Performance Optimization

**For Large Datasets (> 50,000 rows):**
1. Use only Isolation Forest
2. Reduce n_estimators to 50
3. Use PCA instead of UMAP/t-SNE
4. Increase contamination slightly

**For Slow Processing:**
1. Close other applications
2. Reduce number of detection methods
3. Use 2D instead of 3D visualization
4. Sample data before processing

---

## Requirements File

Create `requirements.txt`:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
umap-learn>=0.5.4
shap>=0.42.0
hdbscan>=0.8.33
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## References

### Academic Papers

1. **Isolation Forest**  
   Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. ICDM 2008.

2. **LOF**  
   Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers. SIGMOD 2000.

3. **DBSCAN**  
   Ester, M., et al. (1996). A density-based algorithm for discovering clusters. KDD 1996.

4. **HDBSCAN**  
   Campello, R. J., et al. (2013). Density-based clustering based on hierarchical density estimates. PAKDD 2013.

### Documentation

- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

---

## Course Information

**Module:** Data Mining  
**Instructor:** Dr-ing Rym Besrour  
**Academic Year:** 2025/2026  
**Institution:** [Your University Name]

### Project Objectives Met

- Data preprocessing and feature engineering
- Implementation of multiple anomaly detection algorithms
- Dimensionality reduction and visualization
- Interactive dashboard development
- Explainable AI and result interpretation
- Comprehensive reporting and documentation

---

## License

This project is created for educational purposes as part of the Data Mining course.

---

**Last Updated:** November 2025