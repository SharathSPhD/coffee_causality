"""Test script for causal analysis components."""

import os
import sys
import json
import numpy as np
import pandas as pd

from data_generator import DataGenerator
from causal_analysis import CausalAnalyzer
from visualization import CausalVisualizer

def ensure_directories():
    """Ensure output directories exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dirs = ['plots', 'results']
    for d in dirs:
        path = os.path.join(base_dir, d)
        if not os.path.exists(path):
            os.makedirs(path)
    return base_dir

def save_results(results: dict, filename: str, results_dir: str):
    """Save analysis results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj
    
    results = convert_numpy(results)
    
    filepath = os.path.join(results_dir, f"{filename}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Setup directories
    base_dir = ensure_directories()
    plots_dir = os.path.join(base_dir, 'plots')
    results_dir = os.path.join(base_dir, 'results')
    
    # Initialize components
    generator = DataGenerator(seed=42)
    analyzer = CausalAnalyzer()
    visualizer = CausalVisualizer()
    
    # Generate data
    data = generator.generate_data(n_days=200, include_hidden=True)
    
    # Part A: Initial Data Exploration
    story_plots = visualizer.plot_synthetic_story(data)
    for i, fig in enumerate(story_plots):
        visualizer.save_figure(fig, f'story_plot_{i}', plots_dir)
    
    # Part B: Correlation Analysis
    corr_matrix = analyzer.correlation_analysis(data)
    fig = visualizer.plot_correlation_matrix(corr_matrix)
    visualizer.save_figure(fig, 'correlation_matrix', plots_dir)
    save_results({'correlations': corr_matrix.to_dict()}, 'correlation_analysis', results_dir)
    
    # Part C: IV Analysis
    # Weather as instrument for Foot Traffic
    iv_results_traffic = analyzer.instrumental_variables(
        df=data,
        outcome='Sales',
        treatment='Foot_Traffic',
        instrument='Weather'
    )
    
    # Create IV plots
    fig_iv_rel, diffs = visualizer.plot_weather_iv_relationships(
        data=data,
        instrument='Weather',
        treatment='Foot_Traffic',
        outcome='Sales',
        title_prefix='Weather as Instrument for Foot Traffic'
    )
    visualizer.save_figure(fig_iv_rel, 'iv_weather_foottraffic_rel', plots_dir)
    
    fig_iv = visualizer.plot_iv_results(iv_results_traffic)
    visualizer.save_figure(fig_iv, 'iv_weather_foottraffic_effects', plots_dir)
    
    # Save IV results
    save_results(iv_results_traffic, 'iv_weather_foottraffic', results_dir)
    
    # Part D: Double ML Analysis
    dml_combinations = [
        {
            'treatment': 'Weather',
            'features': ['Social_Media']
        },
        {
            'treatment': 'Social_Media',
            'features': ['Weather', 'Competitor']
        },
        {
            'treatment': 'Competitor',
            'features': ['Weather', 'Social_Media']
        }
    ]
    
    dml_results = {}
    for combo in dml_combinations:
        results = analyzer.double_ml_analysis(
            df=data,
            treatment=combo['treatment'],
            outcome='Sales',
            features=combo['features']
        )
        dml_results[combo['treatment']] = results
    
    # Save DML results and plots
    save_results(dml_results, 'double_ml_analysis', results_dir)
    
    # Create and save DML effects plot
    fig_dml = visualizer.plot_dml_effects(dml_results)
    visualizer.save_figure(fig_dml, 'double_ml_effects', plots_dir)

    # Transfer Entropy Analysis
    te_results = analyzer.transfer_entropy_analysis(
        df=data,
        variables=data.columns.tolist()
    )
    
    fig_te = visualizer.plot_transfer_entropy_network(te_results)
    visualizer.save_figure(fig_te, 'transfer_entropy_network', plots_dir)
    
    save_results({'transfer_entropy': te_results}, 'transfer_entropy_analysis', results_dir)

if __name__ == '__main__':
    main()