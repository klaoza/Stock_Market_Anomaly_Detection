"""
Stock Market Anomaly Detection System
Multi-Agent Architecture for Intelligent Anomaly Detection
Project #3 - Created by Islem Nasri
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import umap
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
import shap
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# AGENT 1: CoordinatorAgent (Orchestrator)
# =============================================================================
class CoordinatorAgent:
    """
    Orchestrates the entire anomaly detection pipeline.
    Manages agent communication and workflow.
    """
    
    def __init__(self):
        self.agents = {}
        self.state = {}
        self.logs = []
        
    def register_agent(self, name, agent):
        """Register an agent in the system."""
        self.agents[name] = agent
        self.log(f"Registered agent: {name}")
    
    def log(self, message):
        """Log messages for transparency."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def get_state(self, key):
        """Retrieve shared state."""
        return self.state.get(key)
    
    def set_state(self, key, value):
        """Update shared state."""
        self.state[key] = value
    
    def execute_pipeline(self, uploaded_file, config):
        """
        Execute the full anomaly detection pipeline.
        This is the main orchestration method.
        """
        self.log("Starting anomaly detection pipeline...")
        
        # Step 1: Data Ingestion
        self.log("Step 1: Data Ingestion")
        df_raw = self.agents['ingestion'].load_data(uploaded_file)
        self.set_state('df_raw', df_raw)
        self.log(f"   Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
        
        # Step 2: Feature Engineering
        self.log("Step 2: Feature Engineering")
        df_features = self.agents['feature_engineering'].engineer_features(df_raw)
        self.set_state('df_features', df_features)
        self.log(f"   Created {len(df_features.columns)} total features")
        
        # Step 3: Scaling
        self.log("Step 3: Data Scaling")
        df_scaled, feature_cols = self.agents['scaler'].scale_data(
            df_features, method=config.get('scaling_method', 'robust')
        )
        self.set_state('df_scaled', df_scaled)
        self.set_state('feature_cols', feature_cols)
        self.log(f"   Scaled {len(feature_cols)} numeric features")
        
        # Step 4: Dimensionality Reduction
        self.log("Step 4: Dimensionality Reduction")
        X = df_scaled[feature_cols].values
        embeddings = self.agents['embedding'].reduce_dimensions(
            X, 
            method=config.get('reduction_method', 'PCA'),
            n_components=config.get('n_components', 2)
        )
        self.set_state('embeddings', embeddings)
        self.log(f"   Reduced to {embeddings.shape[1]} dimensions using {config.get('reduction_method', 'PCA')}")
        
        # Step 5: Train/Test Split
        self.log("Step 5: Train/Test Split")
        test_size = config.get('test_size', 0.3)
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, shuffle=True
        )
        X_train, X_test = X[train_idx], X[test_idx]
        self.set_state('train_idx', train_idx)
        self.set_state('test_idx', test_idx)
        self.set_state('X_train', X_train)
        self.set_state('X_test', X_test)
        self.log(f"   Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
        
        # Step 6: Anomaly Detection (Multiple Methods)
        self.log("Step 6: Anomaly Detection")
        detection_results = {}
        
        methods = config.get('detection_methods', ['isolation_forest', 'lof', 'dbscan', 'hdbscan'])
        
        for method in methods:
            if method == 'isolation_forest':
                self.log(f"   Running Isolation Forest...")
                result = self.agents['isolation_forest'].detect(X_train, X_test, train_idx, test_idx, config)
                detection_results['isolation_forest'] = result
                
            elif method == 'lof':
                self.log(f"   Running LOF...")
                result = self.agents['lof'].detect(X_train, X_test, train_idx, test_idx, config)
                detection_results['lof'] = result
                
            elif method == 'dbscan':
                self.log(f"   Running DBSCAN...")
                result = self.agents['dbscan'].detect(X_train, X_test, train_idx, test_idx, config)
                detection_results['dbscan'] = result
                
            elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
                self.log(f"   Running HDBSCAN...")
                result = self.agents['hdbscan'].detect(X_train, X_test, train_idx, test_idx, config)
                detection_results['hdbscan'] = result
        
        self.set_state('detection_results', detection_results)
        
        # Step 7: Score Fusion
        self.log("Step 7: Score Fusion & Consensus")
        consensus = self.agents['fusion'].fuse_scores(detection_results, len(X))
        self.set_state('consensus', consensus)
        
        # Step 8: Evaluation
        self.log("Step 8: Evaluation")
        evaluation = self.agents['evaluation'].evaluate(detection_results, consensus)
        self.set_state('evaluation', evaluation)
        
        # Step 9: Event Search (for stock market context)
        self.log("Step 9: Event Context Search")
        if config.get('enable_event_search', False) and 'Date' in df_raw.columns:
            events = self.agents['event_search'].search_events(
                df_raw, consensus['consensus_anomalies']
            )
            self.set_state('events', events)
        
        # Step 10: Generate Report
        self.log("Step 10: Generating Report")
        report_df = self.agents['reporting'].generate_report(
            df_raw, df_features, detection_results, consensus
        )
        self.set_state('report_df', report_df)
        
        self.log("Pipeline completed successfully!")
        return self.state


# =============================================================================
# AGENT 2: DataIngestionAgent
# =============================================================================
class DataIngestionAgent:
    """Handles data loading and initial validation."""
    
    def load_data(self, uploaded_file):
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            if len(df) == 0:
                raise ValueError("Empty dataset")
            
            # Convert date columns if present
            for col in ['Date', 'Timestamp', 'date', 'timestamp']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            return df
        except Exception as e:
            st.error(f"❌ Data loading failed: {str(e)}")
            return None


# =============================================================================
# AGENT 3: FeatureEngineeringAgent
# =============================================================================
class FeatureEngineeringAgent:
    """Creates advanced features for anomaly detection."""
    
    def engineer_features(self, df):
        """Engineer comprehensive feature set."""
        df = df.copy()
        
        # Price-based features
        if 'Close' in df.columns and 'Open' in df.columns:
            df['daily_return'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-8)
            df['abs_return'] = np.abs(df['daily_return'])
            df['log_return'] = np.log((df['Close'] + 1) / (df['Open'] + 1))
        
        # Volatility features
        if 'High' in df.columns and 'Low' in df.columns:
            df['intraday_range'] = (df['High'] - df['Low']) / (df['Open'] + 1e-8)
            df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-8)
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_log'] = np.log1p(df['Volume'])
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
                df[f'volume_std_{window}'] = df['Volume'].rolling(window=window, min_periods=1).std()
                df[f'volume_zscore_{window}'] = (df['Volume'] - df[f'volume_ma_{window}']) / (df[f'volume_std_{window}'] + 1e-8)
        
        # Price momentum
        if 'Close' in df.columns:
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'].pct_change(periods=period)
                df[f'ma_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
                df[f'price_to_ma_{period}'] = df['Close'] / (df[f'ma_{period}'] + 1e-8)
        
        # Trend features
        if 'Close' in df.columns:
            df['trend_change'] = df['Close'].diff().diff()
            df['trend_acceleration'] = df['trend_change'].diff()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df = df.replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df


# =============================================================================
# AGENT 4: ScalerAgent
# =============================================================================
class ScalerAgent:
    """Handles data normalization."""
    
    def __init__(self):
        self.scaler = None
        
    def scale_data(self, df, method='robust'):
        """Scale numerical features."""
        # Select numeric columns, exclude date columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['Date', 'Timestamp', 'date', 'timestamp']]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        return df_scaled, feature_cols


# =============================================================================
# AGENT 5: EmbeddingAgent
# =============================================================================
class EmbeddingAgent:
    """Handles dimensionality reduction."""
    
    def reduce_dimensions(self, X, method='PCA', n_components=2):
        """Apply dimensionality reduction."""
        if method == 'PCA':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 't-SNE':
            perplexity = min(30, len(X) - 1)
            reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                          random_state=42, n_iter=1000)
        elif method == 'UMAP':
            n_neighbors = min(15, len(X) - 1)
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                               min_dist=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings = reducer.fit_transform(X)
        return embeddings


# =============================================================================
# AGENT 6: IsolationForestAgent
# =============================================================================
class IsolationForestAgent:
    """Isolation Forest detector with proper training."""
    
    def detect(self, X_train, X_test, train_idx, test_idx, config):
        """
        Train Isolation Forest on training set, predict on all data.
        
        EXPLANATION:
        1. Train model only on X_train (unsupervised learning)
        2. Predict on full dataset (X_train + X_test concatenated)
        3. Score is the average path length (anomalies have shorter paths)
        4. Threshold determined by contamination parameter
        """
        contamination = config.get('contamination', 0.1)
        n_estimators = config.get('n_estimators', 100)
        
        # Train on training set
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train)
        
        # Predict on full dataset
        X_full = np.vstack([X_train, X_test])
        predictions = model.predict(X_full)
        scores = model.decision_function(X_full)
        
        # Create full-length labels (aligned with original data order)
        labels_full = np.zeros(len(X_train) + len(X_test), dtype=int)
        labels_full[predictions == -1] = 1  # 1 = anomaly
        
        # Reorder to match original indices
        original_order = np.argsort(np.concatenate([train_idx, test_idx]))
        labels_full = labels_full[original_order]
        scores_full = scores[original_order]
        
        result = {
            'model': model,
            'labels': labels_full,
            'scores': scores_full,
            'anomaly_count': np.sum(labels_full),
            'anomaly_percentage': (np.sum(labels_full) / len(labels_full)) * 100,
            'method': 'Isolation Forest',
            'description': 'Anomalies have shorter average path lengths in isolation trees'
        }
        
        return result


# =============================================================================
# AGENT 7: LOFAgent
# =============================================================================
class LOFAgent:
    """Local Outlier Factor detector."""
    
    def detect(self, X_train, X_test, train_idx, test_idx, config):
        """
        Train LOF on training set, predict on test set.
        
        EXPLANATION:
        1. LOF measures local density deviation
        2. Points with much lower density than neighbors = anomalies
        3. novelty=True allows prediction on new data
        """
        n_neighbors = config.get('lof_n_neighbors', min(20, len(X_train) - 1))
        contamination = config.get('contamination', 0.1)
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        model.fit(X_train)
        
        # Predict on full dataset
        X_full = np.vstack([X_train, X_test])
        predictions = model.predict(X_full)
        scores = model.decision_function(X_full)
        
        # Create labels
        labels_full = np.zeros(len(X_full), dtype=int)
        labels_full[predictions == -1] = 1
        
        # Reorder
        original_order = np.argsort(np.concatenate([train_idx, test_idx]))
        labels_full = labels_full[original_order]
        scores_full = scores[original_order]
        
        result = {
            'model': model,
            'labels': labels_full,
            'scores': scores_full,
            'anomaly_count': np.sum(labels_full),
            'anomaly_percentage': (np.sum(labels_full) / len(labels_full)) * 100,
            'method': 'LOF',
            'description': 'Anomalies have lower local density compared to neighbors'
        }
        
        return result


# =============================================================================
# AGENT 8: DBSCANAgent (with k-distance strategy)
# =============================================================================
class DBSCANAgent:
    """DBSCAN detector with automatic eps estimation."""
    
    def detect(self, X_train, X_test, train_idx, test_idx, config):
        """
        DBSCAN clustering with automatic eps estimation.
        
        EXPLANATION:
        1. Estimate eps using k-distance plot (elbow method)
        2. Points not in any cluster = anomalies (label -1)
        3. For test set, use nearest neighbor distance from train set
        """
        min_samples = config.get('dbscan_min_samples', 5)
        
        # Estimate eps using k-distance plot
        eps = self._estimate_eps(X_train, min_samples)
        
        # Train DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels_train = model.fit_predict(X_train)
        
        # For test set, use nearest neighbor approach
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_train)
        distances, indices = nbrs.kneighbors(X_test)
        
        # If average distance to nearest neighbors > eps, mark as outlier
        avg_distances = distances.mean(axis=1)
        labels_test = np.where(avg_distances > eps, -1, 0)
        
        # Combine labels
        labels_combined = np.concatenate([labels_train, labels_test])
        
        # Create binary anomaly labels
        labels_full = np.zeros(len(labels_combined), dtype=int)
        labels_full[labels_combined == -1] = 1
        
        # Reorder
        original_order = np.argsort(np.concatenate([train_idx, test_idx]))
        labels_full = labels_full[original_order]
        
        # Cap anomaly percentage at 30%
        if (np.sum(labels_full) / len(labels_full)) > 0.3:
            # Take only most outlying points
            scores = np.concatenate([distances.mean(axis=1) for _ in range(2)])  # Proxy scores
            threshold = np.percentile(scores, 70)
            labels_full = (scores > threshold).astype(int)
        
        result = {
            'model': model,
            'labels': labels_full,
            'scores': labels_full.astype(float),  # Binary scores
            'anomaly_count': np.sum(labels_full),
            'anomaly_percentage': (np.sum(labels_full) / len(labels_full)) * 100,
            'method': 'DBSCAN',
            'description': f'Density-based clustering (eps={eps:.3f}, min_samples={min_samples})',
            'eps': eps
        }
        
        return result
    
    def _estimate_eps(self, X, k):
        """Estimate eps using k-distance plot."""
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = np.sort(distances[:, -1])  # k-th nearest neighbor
        
        # Use 90th percentile as eps (conservative approach)
        eps = np.percentile(distances, 90)
        return eps


# =============================================================================
# AGENT 9: HDBSCANAgent (with safety checks)
# =============================================================================
class HDBSCANAgent:
    """HDBSCAN detector with robust defaults and better fallback."""
    
    def detect(self, X_train, X_test, train_idx, test_idx, config):
        """
        HDBSCAN clustering with safety checks to avoid 0% or 100% anomalies.
        
        EXPLANATION:
        1. Hierarchical DBSCAN - finds clusters of varying density
        2. Points with label -1 are noise/anomalies
        3. Uses conservative parameters with multiple fallback strategies
        """
        if not HDBSCAN_AVAILABLE:
            return None
        
        # Strategy 1: Very conservative initial parameters
        min_cluster_size = max(15, int(len(X_train) * 0.08))  # At least 8% of data
        min_samples = config.get('hdbscan_min_samples', 10)
        
        # Train HDBSCAN
        try:
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
                core_dist_n_jobs=-1
            )
            labels_train = model.fit_predict(X_train)
            
            # Check if clustering worked
            n_clusters = len(set(labels_train)) - (1 if -1 in labels_train else 0)
            anomaly_rate_train = np.sum(labels_train == -1) / len(labels_train)
            
            # Strategy 2: If all or none are anomalies, adjust
            if anomaly_rate_train > 0.4 or anomaly_rate_train < 0.01:
                # Try with larger clusters
                min_cluster_size = max(20, int(len(X_train) * 0.15))
                model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.0,
                    core_dist_n_jobs=-1
                )
                labels_train = model.fit_predict(X_train)
                anomaly_rate_train = np.sum(labels_train == -1) / len(labels_train)
            
            # Strategy 3: If still bad, use distance-based fallback
            if anomaly_rate_train > 0.3 or anomaly_rate_train < 0.01:
                # Fallback: use outlier scores instead
                nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_train)
                distances, _ = nbrs.kneighbors(X_train)
                outlier_scores = distances.mean(axis=1)
                
                # Take top 10% as anomalies
                threshold = np.percentile(outlier_scores, 90)
                labels_train = np.where(outlier_scores > threshold, -1, 0)
            
            # For test set, use nearest neighbor approach
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_train)
            distances, indices = nbrs.kneighbors(X_test)
            
            # If test point is far from train points, mark as anomaly
            avg_distances = distances.mean(axis=1)
            distance_threshold = np.percentile(avg_distances, 85)
            labels_test = np.where(avg_distances > distance_threshold, -1, 0)
            
            # Combine labels
            labels_combined = np.concatenate([labels_train, labels_test])
            
            # Create binary anomaly labels
            labels_full = np.zeros(len(labels_combined), dtype=int)
            labels_full[labels_combined == -1] = 1
            
            # Final safety: ensure 3-15% anomaly rate
            anomaly_rate_full = np.sum(labels_full) / len(labels_full)
            
            if anomaly_rate_full > 0.15:
                # Keep only top 15%
                scores = np.concatenate([
                    nbrs.kneighbors(X_train)[0].mean(axis=1),
                    nbrs.kneighbors(X_test)[0].mean(axis=1)
                ])
                threshold = np.percentile(scores, 85)
                labels_full = (scores > threshold).astype(int)
            elif anomaly_rate_full < 0.03:
                # More lenient: take top 5%
                scores = np.concatenate([
                    nbrs.kneighbors(X_train)[0].mean(axis=1),
                    nbrs.kneighbors(X_test)[0].mean(axis=1)
                ])
                threshold = np.percentile(scores, 95)
                labels_full = (scores > threshold).astype(int)
            
            # Reorder to match original indices
            original_order = np.argsort(np.concatenate([train_idx, test_idx]))
            labels_full = labels_full[original_order]
            
            result = {
                'model': model,
                'labels': labels_full,
                'scores': labels_full.astype(float),
                'anomaly_count': np.sum(labels_full),
                'anomaly_percentage': (np.sum(labels_full) / len(labels_full)) * 100,
                'method': 'HDBSCAN',
                'description': f'Hierarchical density clustering (min_cluster_size={min_cluster_size}, adaptive thresholding)'
            }
            
            return result
            
        except Exception as e:
            st.warning(f"HDBSCAN failed: {e}. Using fallback method.")
            # Complete fallback: distance-based outlier detection
            nbrs = NearestNeighbors(n_neighbors=10).fit(X_train)
            distances_train, _ = nbrs.kneighbors(X_train)
            distances_test, _ = nbrs.kneighbors(X_test)
            
            all_distances = np.concatenate([
                distances_train.mean(axis=1),
                distances_test.mean(axis=1)
            ])
            
            threshold = np.percentile(all_distances, 90)
            labels_combined = (all_distances > threshold).astype(int)
            
            original_order = np.argsort(np.concatenate([train_idx, test_idx]))
            labels_full = labels_combined[original_order]
            
            return {
                'model': None,
                'labels': labels_full,
                'scores': labels_full.astype(float),
                'anomaly_count': np.sum(labels_full),
                'anomaly_percentage': (np.sum(labels_full) / len(labels_full)) * 100,
                'method': 'HDBSCAN (fallback)',
                'description': 'Distance-based fallback method'
            }


# =============================================================================
# AGENT 10: ScoreFusionAgent
# =============================================================================
class ScoreFusionAgent:
    """Fuses scores from multiple detectors."""
    
    def fuse_scores(self, detection_results, n_samples):
        """
        Create consensus anomaly labels.
        
        EXPLANATION:
        1. Voting: anomaly if detected by >= threshold methods
        2. Weighted voting: weight by method reliability
        3. Soft voting: use normalized scores
        """
        # Collect all labels
        all_labels = []
        method_names = []
        
        for method, result in detection_results.items():
            if result is not None:
                all_labels.append(result['labels'])
                method_names.append(method)
        
        if len(all_labels) == 0:
            return None
        
        all_labels = np.array(all_labels)
        
        # Voting strategy: anomaly if detected by >= 2 methods
        vote_counts = np.sum(all_labels, axis=0)
        
        # Consensus thresholds
        consensus_2plus = vote_counts >= 2
        consensus_majority = vote_counts >= (len(all_labels) / 2)
        consensus_all = vote_counts == len(all_labels)
        
        return {
            'vote_counts': vote_counts,
            'consensus_2plus': consensus_2plus,
            'consensus_majority': consensus_majority,
            'consensus_all': consensus_all,
            'consensus_anomalies': np.where(consensus_2plus)[0],
            'strong_anomalies': np.where(consensus_all)[0],
            'method_names': method_names
        }


# =============================================================================
# AGENT 11: EvaluationAgent
# =============================================================================
class EvaluationAgent:
    """Evaluates detection results."""
    
    def evaluate(self, detection_results, consensus):
        """
        Evaluate and compare detection methods.
        
        METRICS:
        1. Anomaly percentage per method
        2. Overlap between methods
        3. Consensus strength
        """
        evaluation = {
            'method_summary': [],
            'overlap_matrix': None,
            'consensus_stats': {}
        }
        
        # Method summary
        for method, result in detection_results.items():
            if result is not None:
                evaluation['method_summary'].append({
                    'method': result['method'],
                    'anomaly_count': result['anomaly_count'],
                    'anomaly_percentage': result['anomaly_percentage'],
                    'description': result['description']
                })
        
        # Overlap matrix
        methods = [r['method'] for r in detection_results.values() if r is not None]
        n_methods = len(methods)
        overlap = np.zeros((n_methods, n_methods))
        
        labels_list = [r['labels'] for r in detection_results.values() if r is not None]
        
        for i in range(n_methods):
            for j in range(n_methods):
                overlap[i, j] = np.sum(labels_list[i] & labels_list[j])
        
        evaluation['overlap_matrix'] = overlap
        evaluation['method_names'] = methods
        
        # Consensus stats
        if consensus is not None:
            evaluation['consensus_stats'] = {
                'detected_by_2plus': np.sum(consensus['consensus_2plus']),
                'detected_by_majority': np.sum(consensus['consensus_majority']),
                'detected_by_all': np.sum(consensus['consensus_all'])
            }
        
        return evaluation


# =============================================================================
# AGENT 12: EventSearchAgent
# =============================================================================
class EventSearchAgent:
    """Searches for market events related to anomalies."""
    
    def search_events(self, df, anomaly_indices):
        """
        Search for potential market events on anomaly dates.
        
        EXPLANATION:
        For anomalies with extreme price movements:
        1. Extract the date
        2. Create a search query context
        3. Return context for manual/API search
        """
        events = []
        
        if 'Date' not in df.columns:
            return events
        
        for idx in anomaly_indices[:10]:  # Limit to first 10
            row = df.iloc[idx]
            
            # Calculate severity
            severity = 0
            context = []
            
            if 'daily_return' in df.columns:
                ret = row.get('daily_return', 0)
                if abs(ret) > 0.05:  # >5% move
                    severity += 1
                    context.append(f"{ret*100:.2f}% return")
            
            if 'Volume' in df.columns:
                vol_mean = df['Volume'].mean()
                if row.get('Volume', 0) > vol_mean * 2:
                    severity += 1
                    context.append("High volume")
            
            if severity > 0:
                date = row['Date']
                events.append({
                    'index': idx,
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'severity': severity,
                    'context': ', '.join(context),
                    'search_query': f"stock market news {date.strftime('%Y-%m-%d')}" if hasattr(date, 'strftime') else ""
                })
        
        return events


# =============================================================================
# AGENT 13: ReportingAgent
# =============================================================================
class ReportingAgent:
    """Generates comprehensive reports."""
    
    def generate_report(self, df_raw, df_features, detection_results, consensus):
        """
        Generate a comprehensive CSV report.
        
        COLUMNS:
        - Original data columns
        - Engineered features
        - Anomaly scores from each method
        - Anomaly labels from each method
        - Consensus vote count
        - Consensus labels
        """
        report_df = df_raw.copy()
        
        # Add key features
        feature_cols_to_add = ['daily_return', 'intraday_range', 'volume_zscore_5']
        for col in feature_cols_to_add:
            if col in df_features.columns:
                report_df[f'feature_{col}'] = df_features[col]
        
        # Add detection results
        for method, result in detection_results.items():
            if result is not None:
                report_df[f'{method}_label'] = result['labels']
                report_df[f'{method}_score'] = result['scores']
        
        # Add consensus
        if consensus is not None:
            report_df['consensus_vote_count'] = consensus['vote_counts']
            report_df['consensus_2plus'] = consensus['consensus_2plus'].astype(int)
            report_df['consensus_all'] = consensus['consensus_all'].astype(int)
        
        return report_df


# =============================================================================
# AGENT 14: VisualizationAgent
# =============================================================================
class VisualizationAgent:
    """Handles all visualization tasks."""
    
    def plot_embeddings_2d(self, embeddings, labels, train_idx, test_idx, title="2D Projection"):
        """Create 2D scatter plot."""
        fig = go.Figure()
        
        is_train = np.isin(range(len(embeddings)), train_idx)
        anomaly_mask = labels.astype(bool)
        
        # Normal - Train
        mask = (~anomaly_mask) & is_train
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0], y=embeddings[mask, 1],
                mode='markers', name='Normal (Train)',
                marker=dict(size=5, color='lightblue', opacity=0.6)
            ))
        
        # Normal - Test
        mask = (~anomaly_mask) & (~is_train)
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0], y=embeddings[mask, 1],
                mode='markers', name='Normal (Test)',
                marker=dict(size=5, color='blue', opacity=0.4)
            ))
        
        # Anomaly - Train
        mask = anomaly_mask & is_train
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0], y=embeddings[mask, 1],
                mode='markers', name='Anomaly (Train)',
                marker=dict(size=10, color='orange', symbol='x')
            ))
        
        # Anomaly - Test
        mask = anomaly_mask & (~is_train)
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0], y=embeddings[mask, 1],
                mode='markers', name='Anomaly (Test)',
                marker=dict(size=10, color='red', symbol='x')
            ))
        
        fig.update_layout(title=title, height=600)
        return fig
    
    def plot_consensus(self, embeddings, vote_counts, method_names):
        """Visualize consensus strength."""
        fig = go.Figure()
        
        n_methods = len(method_names)
        
        # Color by vote count
        fig.add_trace(go.Scatter(
            x=embeddings[:, 0], y=embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=vote_counts,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Vote Count"),
                cmin=0,
                cmax=n_methods
            ),
            text=[f"Votes: {v}/{n_methods}" for v in vote_counts],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f"Consensus Heatmap (0 to {n_methods} votes)",
            height=600
        )
        return fig
    
    def plot_comparison_bar(self, evaluation):
        """Bar chart comparing methods."""
        summary = evaluation['method_summary']
        
        fig = go.Figure(data=[
            go.Bar(
                x=[s['method'] for s in summary],
                y=[s['anomaly_count'] for s in summary],
                text=[f"{s['anomaly_percentage']:.1f}%" for s in summary],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(summary)]
            )
        ])
        
        fig.update_layout(
            title="Anomalies Detected by Each Method",
            xaxis_title="Method",
            yaxis_title="Count",
            height=400
        )
        return fig
    
    def plot_overlap_heatmap(self, overlap_matrix, method_names):
        """Heatmap showing overlap between methods."""
        fig = go.Figure(data=go.Heatmap(
            z=overlap_matrix,
            x=method_names,
            y=method_names,
            colorscale='Blues',
            text=overlap_matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Method Overlap Matrix (Number of Shared Anomalies)",
            height=500
        )
        return fig
    
    def plot_feature_importance(self, df, anomaly_mask, feature_cols):
        """Compare feature distributions."""
        if len(feature_cols) == 0:
            return None
        
        # Select top features
        top_features = feature_cols[:6]
        top_features = [f for f in top_features if f in df.columns]
        
        if len(top_features) < 3:
            return None
        
        # Create radar chart
        normal_means = df.loc[~anomaly_mask, top_features].mean()
        anomaly_means = df.loc[anomaly_mask, top_features].mean()
        
        # Normalize
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        all_values = pd.concat([normal_means, anomaly_means], axis=1).T
        normalized = scaler.fit_transform(all_values)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized[0],
            theta=top_features,
            fill='toself',
            name='Normal'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized[1],
            theta=top_features,
            fill='toself',
            name='Anomalies'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Feature Profile: Normal vs Anomalies",
            height=500
        )
        return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
class DashboardAgent:
    """Main dashboard orchestrator."""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.viz = VisualizationAgent()
        self._register_agents()
    
    def _register_agents(self):
        """Register all agents."""
        self.coordinator.register_agent('ingestion', DataIngestionAgent())
        self.coordinator.register_agent('feature_engineering', FeatureEngineeringAgent())
        self.coordinator.register_agent('scaler', ScalerAgent())
        self.coordinator.register_agent('embedding', EmbeddingAgent())
        self.coordinator.register_agent('isolation_forest', IsolationForestAgent())
        self.coordinator.register_agent('lof', LOFAgent())
        self.coordinator.register_agent('dbscan', DBSCANAgent())
        self.coordinator.register_agent('hdbscan', HDBSCANAgent())
        self.coordinator.register_agent('fusion', ScoreFusionAgent())
        self.coordinator.register_agent('evaluation', EvaluationAgent())
        self.coordinator.register_agent('event_search', EventSearchAgent())
        self.coordinator.register_agent('reporting', ReportingAgent())
    
    def render(self):
        """Render the dashboard."""
        st.set_page_config(page_title="Multi-Agent Anomaly Detection", layout="wide")
        
        st.title("Multi-Agent Stock Market Anomaly Detection System")
        st.markdown("**Project #3 - Created by Islem Nasri**")
        st.markdown("*Implementation with Proper Agent Architecture*")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=['csv'])
            
            if uploaded_file is None:
                st.info("Upload a CSV file to begin")
                st.markdown("---")
                st.markdown("### System Architecture")
                st.markdown("**Agents:**")
                st.markdown("1. Coordinator")
                st.markdown("2. Data Ingestion")
                st.markdown("3. Feature Engineering")
                st.markdown("4. Scaler")
                st.markdown("5. Embedding")
                st.markdown("6. Isolation Forest")
                st.markdown("7. LOF")
                st.markdown("8. DBSCAN")
                st.markdown("9. HDBSCAN")
                st.markdown("10. Score Fusion")
                st.markdown("11. Evaluation")
                st.markdown("12. Event Search")
                st.markdown("13. Reporting")
                return
            
            st.success("File uploaded!")
            
            # Configuration
            st.subheader("Preprocessing")
            scaling_method = st.selectbox("Scaling", ['robust', 'standard'])
            
            st.subheader("Dimensionality Reduction")
            reduction_method = st.selectbox("Method", ['PCA', 'UMAP', 't-SNE'])
            n_components = st.radio("Dimensions", [2, 3], index=0)
            
            st.subheader("Detection Methods")
            method_options = {
                'Isolation Forest': 'isolation_forest',
                'LOF': 'lof',
                'DBSCAN': 'dbscan',
                'HDBSCAN': 'hdbscan'
            }
            
            selected_methods = st.multiselect(
                "Select Methods",
                list(method_options.keys()),
                default=list(method_options.keys())
            )
            
            detection_methods = [method_options[m] for m in selected_methods]
            
            st.subheader("Hyperparameters")
            
            if 'Isolation Forest' in selected_methods or 'LOF' in selected_methods:
                contamination = st.slider("Contamination", 0.01, 0.3, 0.1, 0.01)
            else:
                contamination = 0.1
            
            if 'Isolation Forest' in selected_methods:
                n_estimators = st.slider("IF: N Estimators", 50, 200, 100, 10)
            else:
                n_estimators = 100
            
            if 'LOF' in selected_methods:
                lof_n_neighbors = st.slider("LOF: Neighbors", 5, 50, 20)
            else:
                lof_n_neighbors = 20
            
            if 'DBSCAN' in selected_methods:
                dbscan_min_samples = st.slider("DBSCAN: Min Samples", 2, 20, 5)
            else:
                dbscan_min_samples = 5
            
            if 'HDBSCAN' in selected_methods:
                hdbscan_min_samples = st.slider("HDBSCAN: Min Samples", 5, 20, 10)
            else:
                hdbscan_min_samples = 10
            
            st.subheader("Train/Test Split")
            test_size = st.slider("Test Size (%)", 10, 50, 30) / 100
            
            st.subheader("Features")
            enable_event_search = st.checkbox("Enable Event Search", value=True)
            
            run_button = st.button("Run Analysis", type="primary")
        
        # Main content
        if run_button:
            config = {
                'scaling_method': scaling_method,
                'reduction_method': reduction_method,
                'n_components': n_components,
                'detection_methods': detection_methods,
                'contamination': contamination,
                'n_estimators': n_estimators,
                'lof_n_neighbors': lof_n_neighbors,
                'dbscan_min_samples': dbscan_min_samples,
                'hdbscan_min_samples': hdbscan_min_samples,
                'test_size': test_size,
                'enable_event_search': enable_event_search
            }
            
            with st.spinner("🔄 Running pipeline..."):
                state = self.coordinator.execute_pipeline(uploaded_file, config)
            
            if state is None:
                st.error("❌ Pipeline failed")
                return
            
            # Display logs
            with st.expander("Pipeline Logs", expanded=False):
                for log in self.coordinator.logs:
                    st.text(log)
            
            # Results
            detection_results = state['detection_results']
            evaluation = state['evaluation']
            consensus = state['consensus']
            embeddings = state['embeddings']
            train_idx = state['train_idx']
            test_idx = state['test_idx']
            df_features = state['df_features']
            df_raw = state['df_raw']
            
            # Metrics
            st.subheader("Detection Summary")
            
            cols = st.columns(len(detection_results))
            for idx, (method, result) in enumerate(detection_results.items()):
                if result is not None:
                    with cols[idx]:
                        st.metric(
                            result['method'],
                            f"{result['anomaly_count']}",
                            f"{result['anomaly_percentage']:.1f}%"
                        )
            
            # Consensus metrics
            if consensus is not None:
                st.markdown("### Consensus Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Detected by 2+ Methods", 
                             np.sum(consensus['consensus_2plus']))
                with col2:
                    st.metric("Detected by Majority", 
                             np.sum(consensus['consensus_majority']))
                with col3:
                    st.metric("Detected by ALL Methods", 
                             np.sum(consensus['consensus_all']))
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Overview", "Individual Methods", "Consensus", 
                "Comparison", "Events", "Report"
            ])
            
            with tab1:
                st.subheader("How the System Works")
                
                st.markdown("""
                ### Training Process Explained
                
                **IMPORTANT:** All methods are **UNSUPERVISED** - they don't need labeled data!
                
                #### 1. Data Split
                - Data is split into **Train (70%)** and **Test (30%)**
                - This evaluates generalization, NOT for supervised learning
                
                #### 2. Each Method Trains Differently:
                
                **Isolation Forest:**
                - Builds random trees on training data
                - Anomalies are isolated faster (shorter paths)
                - Predicts on all data using learned trees
                
                **Local Outlier Factor (LOF):**
                - Calculates local density on training data
                - Points with lower density than neighbors = anomalies
                - Uses `novelty=True` to predict on test data
                
                **DBSCAN:**
                - Groups dense regions into clusters
                - Points not in any cluster = anomalies
                - Auto-estimates epsilon using k-distance plot
                - Test points checked against train clusters
                
                **HDBSCAN:**
                - Hierarchical version of DBSCAN
                - Finds clusters of varying density
                - Uses conservative parameters to avoid over-flagging
                - Multiple fallback strategies for robustness
                
                #### 3. Score Fusion:
                - Combines results from all methods
                - **Consensus voting:** anomaly if 2+ methods agree
                - Stronger anomalies: detected by ALL methods
                """)
                
                # Show split info
                st.info(f"Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
                
                # Visualization
                if consensus is not None:
                    st.subheader("Consensus Heatmap")
                    fig = self.viz.plot_consensus(
                        embeddings, 
                        consensus['vote_counts'],
                        consensus['method_names']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Individual Detection Methods")
                
                for method, result in detection_results.items():
                    if result is not None:
                        st.markdown(f"### {result['method']}")
                        st.caption(result['description'])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = self.viz.plot_embeddings_2d(
                                embeddings,
                                result['labels'],
                                train_idx,
                                test_idx,
                                title=f"{result['method']} Detection"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Anomalies", result['anomaly_count'])
                            st.metric("Percentage", f"{result['anomaly_percentage']:.2f}%")
                            
                            # Show some anomalies
                            anomaly_indices = np.where(result['labels'] == 1)[0][:5]
                            if len(anomaly_indices) > 0:
                                st.markdown("**Sample Anomalies:**")
                                for idx in anomaly_indices:
                                    st.text(f"Index: {idx}")
                        
                        st.markdown("---")
            
            with tab3:
                st.subheader("Consensus Analysis")
                
                if consensus is not None:
                    # Visualize consensus strength
                    fig = self.viz.plot_consensus(
                        embeddings,
                        consensus['vote_counts'],
                        consensus['method_names']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strong anomalies
                    st.markdown("### 🌟 Strong Anomalies (ALL methods agree)")
                    strong_indices = consensus['strong_anomalies']
                    
                    if len(strong_indices) > 0:
                        st.write(f"Found {len(strong_indices)} strong anomalies")
                        
                        # Show details
                        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        display_cols = [c for c in display_cols if c in df_raw.columns]
                        
                        st.dataframe(
                            df_raw.iloc[strong_indices][display_cols].head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("No anomalies detected by all methods")
            
            with tab4:
                st.subheader("Method Comparison")
                
                # Bar chart
                fig_bar = self.viz.plot_comparison_bar(evaluation)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Overlap heatmap
                st.markdown("### Method Overlap")
                fig_overlap = self.viz.plot_overlap_heatmap(
                    evaluation['overlap_matrix'],
                    evaluation['method_names']
                )
                st.plotly_chart(fig_overlap, use_container_width=True)
                
                # Feature importance
                st.markdown("### Feature Profile")
                if consensus is not None:
                    fig_features = self.viz.plot_feature_importance(
                        df_features,
                        consensus['consensus_2plus'],
                        state['feature_cols']
                    )
                    if fig_features:
                        st.plotly_chart(fig_features, use_container_width=True)
            
            with tab5:
                st.subheader("Event Context Search")
                
                if enable_event_search and 'events' in state:
                    events = state['events']
                    
                    if len(events) > 0:
                        st.markdown("""
                        **Potential market events related to anomalies:**
                        
                        These dates had significant anomalies. You can search for news/events:
                        """)
                        
                        for event in events:
                            with st.expander(f"{event['date']} - Severity: {event['severity']}/2"):
                                st.write(f"**Context:** {event['context']}")
                                st.write(f"**Index:** {event['index']}")
                                st.code(event['search_query'], language=None)
                                st.caption("Copy the search query above to find related news")
                    else:
                        st.info("No significant events found in top anomalies")
                else:
                    st.info("Event search disabled or no date column found")
            
            with tab6:
                st.subheader("Comprehensive Report")
                
                report_df = state['report_df']
                
                st.markdown(f"**Report contains {len(report_df)} rows with:**")
                st.markdown("- Original data")
                st.markdown("- Engineered features")
                st.markdown("- Scores from each method")
                st.markdown("- Labels from each method")
                st.markdown("- Consensus voting results")
                
                # Preview
                st.dataframe(report_df.head(20), use_container_width=True)
                
                # Download
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Report (CSV)",
                    data=csv,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


# =============================================================================
# MAIN
# =============================================================================
def main():
    dashboard = DashboardAgent()
    dashboard.render()

if __name__ == "__main__":
    main()