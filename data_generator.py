"""
Synthetic Stock Data Generator with Anomalies
Generates realistic stock data with various types of anomalies for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

class StockDataGenerator:
    """Generate synthetic stock market data with controllable anomalies."""
    
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        
    def generate_base_data(self, start_price=100.0, volatility=0.02):
        """Generate base time series using geometric Brownian motion."""
        dates = [datetime.now() - timedelta(days=self.n_samples-i) 
                for i in range(self.n_samples)]
        
        # Generate returns using GBM
        mu = 0.0005  # Daily drift
        sigma = volatility  # Daily volatility
        
        returns = np.random.normal(mu, sigma, self.n_samples)
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        # High and Low around Close
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, self.n_samples)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, self.n_samples)))
        open_price = np.roll(prices, 1)
        open_price[0] = start_price
        close = prices
        
        # Generate volume
        base_volume = 1_000_000
        volume = np.random.lognormal(np.log(base_volume), 0.5, self.n_samples)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume.astype(int)
        })
        
        return df
    
    def inject_price_shock(self, df, n_shocks=5, magnitude_range=(0.05, 0.15)):
        """Inject extreme price movements."""
        indices = np.random.choice(range(100, len(df)-100), n_shocks, replace=False)
        
        for idx in indices:
            # Random direction
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(*magnitude_range)
            shock = direction * magnitude
            
            # Apply shock
            df.loc[idx, 'Close'] *= (1 + shock)
            df.loc[idx, 'High'] = max(df.loc[idx, 'High'], df.loc[idx, 'Close'])
            df.loc[idx, 'Low'] = min(df.loc[idx, 'Low'], df.loc[idx, 'Close'])
            
            # Affect next day's open
            if idx + 1 < len(df):
                df.loc[idx+1, 'Open'] = df.loc[idx, 'Close']
        
        return df, indices
    
    def inject_volatility_spike(self, df, n_spikes=5, magnitude_range=(2.0, 5.0)):
        """Inject abnormal intraday volatility."""
        indices = np.random.choice(range(100, len(df)-100), n_spikes, replace=False)
        
        for idx in indices:
            magnitude = np.random.uniform(*magnitude_range)
            normal_range = df.loc[idx, 'High'] - df.loc[idx, 'Low']
            
            # Expand range
            df.loc[idx, 'High'] = df.loc[idx, 'Close'] + normal_range * magnitude / 2
            df.loc[idx, 'Low'] = df.loc[idx, 'Close'] - normal_range * magnitude / 2
        
        return df, indices
    
    def inject_volume_anomaly(self, df, n_anomalies=5, magnitude_range=(3.0, 8.0)):
        """Inject unusual trading volume."""
        indices = np.random.choice(range(100, len(df)-100), n_anomalies, replace=False)
        
        for idx in indices:
            magnitude = np.random.uniform(*magnitude_range)
            df.loc[idx, 'Volume'] = int(df.loc[idx, 'Volume'] * magnitude)
        
        return df, indices
    
    def inject_trend_reversal(self, df, n_reversals=3):
        """Inject sharp trend reversals."""
        indices = np.random.choice(range(100, len(df)-100), n_reversals, replace=False)
        
        for idx in indices:
            # Calculate recent trend
            window = 20
            recent_trend = (df.loc[idx-1, 'Close'] - df.loc[idx-window, 'Close']) / df.loc[idx-window, 'Close']
            
            # Reverse it sharply
            reversal = -recent_trend * np.random.uniform(1.5, 2.5)
            df.loc[idx:idx+5, 'Close'] *= (1 + reversal / 5)
        
        return df, indices
    
    def inject_composite_anomaly(self, df, n_composite=3):
        """Inject multi-factor anomalies (price + volume + volatility)."""
        indices = np.random.choice(range(100, len(df)-100), n_composite, replace=False)
        
        for idx in indices:
            # Price shock
            shock = np.random.choice([-1, 1]) * np.random.uniform(0.08, 0.12)
            df.loc[idx, 'Close'] *= (1 + shock)
            
            # Volume spike
            df.loc[idx, 'Volume'] = int(df.loc[idx, 'Volume'] * np.random.uniform(4, 7))
            
            # Volatility spike
            normal_range = df.loc[idx, 'High'] - df.loc[idx, 'Low']
            df.loc[idx, 'High'] = df.loc[idx, 'Close'] + normal_range * 2
            df.loc[idx, 'Low'] = df.loc[idx, 'Close'] - normal_range * 2
        
        return df, indices
    
    def inject_quiet_period(self, df, n_periods=2, duration=5):
        """Inject unusually quiet trading periods (low volume)."""
        indices = np.random.choice(range(100, len(df)-100-duration), n_periods, replace=False)
        
        all_indices = []
        for idx in indices:
            df.loc[idx:idx+duration, 'Volume'] = (df.loc[idx:idx+duration, 'Volume'] * 0.1).astype(int)
            all_indices.extend(range(idx, idx+duration))
        
        return df, all_indices
    
    def generate(self, 
                 price_shocks=5,
                 volatility_spikes=5,
                 volume_anomalies=5,
                 trend_reversals=3,
                 composite_anomalies=3,
                 quiet_periods=2):
        """Generate complete dataset with all anomaly types."""
        
        print(f"Generating {self.n_samples} data points...")
        df = self.generate_base_data()
        
        anomaly_log = []
        
        if price_shocks > 0:
            print(f"Injecting {price_shocks} price shocks...")
            df, indices = self.inject_price_shock(df, price_shocks)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'price_shock'})
        
        if volatility_spikes > 0:
            print(f"Injecting {volatility_spikes} volatility spikes...")
            df, indices = self.inject_volatility_spike(df, volatility_spikes)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'volatility_spike'})
        
        if volume_anomalies > 0:
            print(f"Injecting {volume_anomalies} volume anomalies...")
            df, indices = self.inject_volume_anomaly(df, volume_anomalies)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'volume_anomaly'})
        
        if trend_reversals > 0:
            print(f"Injecting {trend_reversals} trend reversals...")
            df, indices = self.inject_trend_reversal(df, trend_reversals)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'trend_reversal'})
        
        if composite_anomalies > 0:
            print(f"Injecting {composite_anomalies} composite anomalies...")
            df, indices = self.inject_composite_anomaly(df, composite_anomalies)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'composite_anomaly'})
        
        if quiet_periods > 0:
            print(f"Injecting {quiet_periods} quiet periods...")
            df, indices = self.inject_quiet_period(df, quiet_periods)
            for idx in indices:
                anomaly_log.append({'index': idx, 'type': 'quiet_period'})
        
        # Create ground truth labels
        anomaly_indices = set([a['index'] for a in anomaly_log])
        df['is_anomaly'] = [1 if i in anomaly_indices else 0 for i in range(len(df))]
        
        # Add anomaly type column
        anomaly_type_map = {a['index']: a['type'] for a in anomaly_log}
        df['anomaly_type'] = [anomaly_type_map.get(i, 'normal') for i in range(len(df))]
        
        print(f"\nGenerated {len(df)} data points with {len(anomaly_indices)} anomalies")
        print(f"Anomaly rate: {len(anomaly_indices)/len(df)*100:.2f}%")
        
        return df, pd.DataFrame(anomaly_log)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic stock data with anomalies')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of data points')
    parser.add_argument('--output', type=str, default='synthetic_stock_data.csv', help='Output filename')
    parser.add_argument('--price_shocks', type=int, default=5, help='Number of price shocks')
    parser.add_argument('--volatility_spikes', type=int, default=5, help='Number of volatility spikes')
    parser.add_argument('--volume_anomalies', type=int, default=5, help='Number of volume anomalies')
    parser.add_argument('--trend_reversals', type=int, default=3, help='Number of trend reversals')
    parser.add_argument('--composite', type=int, default=3, help='Number of composite anomalies')
    parser.add_argument('--quiet_periods', type=int, default=2, help='Number of quiet periods')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generator = StockDataGenerator(n_samples=args.n_samples, seed=args.seed)
    
    df, anomaly_log = generator.generate(
        price_shocks=args.price_shocks,
        volatility_spikes=args.volatility_spikes,
        volume_anomalies=args.volume_anomalies,
        trend_reversals=args.trend_reversals,
        composite_anomalies=args.composite,
        quiet_periods=args.quiet_periods
    )
    
    # Save main data (without ground truth labels for testing)
    df_export = df.drop(['is_anomaly', 'anomaly_type'], axis=1)
    df_export.to_csv(args.output, index=False)
    print(f"\n✅ Data saved to {args.output}")
    
    # Save ground truth separately for validation
    ground_truth_file = args.output.replace('.csv', '_ground_truth.csv')
    df[['Date', 'is_anomaly', 'anomaly_type']].to_csv(ground_truth_file, index=False)
    print(f"✅ Ground truth saved to {ground_truth_file}")
    
    # Save anomaly log
    log_file = args.output.replace('.csv', '_anomaly_log.csv')
    anomaly_log.to_csv(log_file, index=False)
    print(f"✅ Anomaly log saved to {log_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"Total anomalies: {df['is_anomaly'].sum()}")
    print(f"Anomaly rate: {df['is_anomaly'].mean()*100:.2f}%")
    print("\nAnomalies by type:")
    print(df[df['is_anomaly']==1]['anomaly_type'].value_counts())
    print("\nPrice statistics:")
    print(f"  Start: ${df['Close'].iloc[0]:.2f}")
    print(f"  End: ${df['Close'].iloc[-1]:.2f}")
    print(f"  Min: ${df['Close'].min():.2f}")
    print(f"  Max: ${df['Close'].max():.2f}")
    print(f"  Mean: ${df['Close'].mean():.2f}")
    print("\nVolume statistics:")
    print(f"  Mean: {df['Volume'].mean():,.0f}")
    print(f"  Min: {df['Volume'].min():,}")
    print(f"  Max: {df['Volume'].max():,}")


if __name__ == '__main__':
    # If run directly without args, generate with defaults
    import sys
    if len(sys.argv) == 1:
        print("Running with default parameters...")
        print("Use --help to see available options\n")
    
    main()