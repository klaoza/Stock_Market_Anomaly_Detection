"""
Validation and Testing Script for Anomaly Detection System
Tests all agents and provides performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class AnomalyDetectionValidator:
    """Validates anomaly detection system performance."""
    
    def __init__(self):
        self.results = {}
        self.comparison = []
        
    def load_data_with_ground_truth(self, data_file, ground_truth_file):
        """Load data and corresponding ground truth labels."""
        df = pd.read_csv(data_file)
        gt = pd.read_csv(ground_truth_file)
        
        # Merge ground truth
        df = df.merge(gt[['Date', 'is_anomaly', 'anomaly_type']], on='Date', how='left')
        
        return df
    
    def calculate_metrics(self, y_true, y_pred, method_name):
        """Calculate standard classification metrics."""
        metrics = {
            'method': method_name,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': np.mean(y_true == y_pred),
            'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
            'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
            'true_negatives': np.sum((y_true == 0) & (y_pred == 0)),
            'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def calculate_detection_rate_by_type(self, df, y_pred, method_name):
        """Calculate detection rates for each anomaly type."""
        df_eval = df.copy()
        df_eval['predicted'] = y_pred
        
        anomaly_types = df_eval[df_eval['is_anomaly'] == 1]['anomaly_type'].unique()
        
        type_metrics = {}
        for atype in anomaly_types:
            if atype != 'normal':
                mask = df_eval['anomaly_type'] == atype
                if mask.sum() > 0:
                    y_true_type = df_eval[mask]['is_anomaly'].values
                    y_pred_type = df_eval[mask]['predicted'].values
                    
                    detection_rate = recall_score(y_true_type, y_pred_type, zero_division=0)
                    precision = precision_score(y_true_type, y_pred_type, zero_division=0)
                    
                    type_metrics[atype] = {
                        'detection_rate': detection_rate,
                        'precision': precision,
                        'count': mask.sum()
                    }
        
        return type_metrics
    
    def test_method(self, X, y_true, method_name, model, df=None):
        """Test a single anomaly detection method."""
        print(f"\n{'='*60}")
        print(f"Testing: {method_name.upper()}")
        print('='*60)
        
        # Get predictions
        if hasattr(model, 'fit_predict'):
            y_pred = model.fit_predict(X)
            y_pred = (y_pred == -1).astype(int)
        else:
            model.fit(X)
            y_pred = model.predict(X)
            y_pred = (y_pred == -1).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, method_name)
        
        # Print metrics
        print(f"\nOverall Performance:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['true_positives']:<6} FP: {metrics['false_positives']}")
        print(f"  FN: {metrics['false_negatives']:<6} TN: {metrics['true_negatives']}")
        
        # Detection by type
        if df is not None:
            type_metrics = self.calculate_detection_rate_by_type(df, y_pred, method_name)
            print(f"\nDetection Rate by Anomaly Type:")
            for atype, tm in type_metrics.items():
                print(f"  {atype:20s}: {tm['detection_rate']:.2%} "
                      f"(Precision: {tm['precision']:.2%}, Count: {tm['count']})")
            
            metrics['type_metrics'] = type_metrics
        
        # Store results
        self.results[method_name] = metrics
        
        return metrics
    
    def compare_methods(self):
        """Generate comparison table of all methods."""
        if not self.results:
            print("No results to compare. Run test_method() first.")
            return None
        
        comparison_df = pd.DataFrame([
            {
                'Method': name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Accuracy': metrics['accuracy'],
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives']
            }
            for name, metrics in self.results.items()
        ])
        
        # Sort by F1-score
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*80)
        print("METHOD COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print()
        
        return comparison_df
    
    def plot_comparison(self, save_path='method_comparison.png'):
        """Plot comparison of methods."""
        if not self.results:
            print("No results to plot.")
            return
        
        comparison_df = pd.DataFrame([
            {
                'Method': name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }
            for name, metrics in self.results.items()
        ])
        
        # Melt for plotting
        df_melted = comparison_df.melt(id_vars='Method', 
                                       value_vars=['Precision', 'Recall', 'F1-Score'],
                                       var_name='Metric', value_name='Score')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        ax.bar(x - width, comparison_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, comparison_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, comparison_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Anomaly Detection Method Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Method'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, save_path='confusion_matrices.png'):
        """Plot confusion matrices for all methods."""
        if not self.results:
            print("No results to plot.")
            return
        
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            axes[idx].set_title(f'{name}\n(F1={metrics["f1_score"]:.3f})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to {save_path}")
        plt.close()
    
    def generate_report(self, output_file='validation_report.json'):
        """Generate comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'methods_tested': list(self.results.keys()),
            'results': self.results,
            'summary': {
                'best_precision': max(self.results.items(), 
                                     key=lambda x: x[1]['precision'])[0],
                'best_recall': max(self.results.items(), 
                                  key=lambda x: x[1]['recall'])[0],
                'best_f1': max(self.results.items(), 
                              key=lambda x: x[1]['f1_score'])[0]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved to {output_file}")
        
        return report


def run_full_validation():
    """Run complete validation pipeline."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import DBSCAN
    
    print("="*80)
    print("ANOMALY DETECTION SYSTEM VALIDATION")
    print("="*80)
    
    # Load data
    print("\n📁 Loading data...")
    try:
        validator = AnomalyDetectionValidator()
        df = validator.load_data_with_ground_truth(
            'synthetic_stock_data.csv',
            'synthetic_stock_data_ground_truth.csv'
        )
        print(f"✅ Loaded {len(df)} samples with ground truth")
    except FileNotFoundError:
        print("❌ Data files not found. Please generate synthetic data first:")
        print("   python generate_data.py")
        return
    
    # Prepare features
    print("\n⚙️ Preparing features...")
    from app import PreprocessingAgent
    
    agent = PreprocessingAgent()
    df_scaled, feature_cols, df_original = agent.process(df.drop(['is_anomaly', 'anomaly_type'], axis=1))
    X = df_scaled[feature_cols].values
    y_true = df['is_anomaly'].values
    
    print(f"✅ Prepared {X.shape[1]} features from {X.shape[0]} samples")
    print(f"   Anomaly rate in data: {y_true.mean()*100:.2f}%")
    
    # Test methods
    methods = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    for name, model in methods.items():
        validator.test_method(X, y_true, name, model, df)
    
    # Generate comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    validator.compare_methods()
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    validator.plot_comparison('validation_comparison.png')
    validator.plot_confusion_matrices('validation_confusion_matrices.png')
    
    # Generate report
    print("\n📄 Generating report...")
    report = validator.generate_report('validation_report.json')
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Best Precision: {report['summary']['best_precision']}")
    print(f"Best Recall:    {report['summary']['best_recall']}")
    print(f"Best F1-Score:  {report['summary']['best_f1']}")
    print("\n✅ All results saved to disk")


if __name__ == '__main__':
    run_full_validation()