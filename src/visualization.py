"""
Visualization module for causal analysis results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple

class CausalVisualizer:
    """Creates visualizations for causal analysis results."""
    
    def __init__(self, style: str = 'default'):
        """Initialize the visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        if style == 'default':
            sns.set_theme()
        sns.set_palette('viridis')
    
    def save_figure(self, fig: plt.Figure, filename: str, plots_dir: str = 'plots'):
        """Save figure to plots directory.

        Args:
            fig (plt.Figure): Figure to save
            filename (str): Filename without extension
            plots_dir (str): Directory to save plots in
        """
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig(os.path.join(plots_dir, f"{filename}.png"), 
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def plot_synthetic_story(self, df: pd.DataFrame) -> List[plt.Figure]:
        """Create a series of story-driven visualizations for synthetic data.
        
        Args:
            df (pd.DataFrame): Generated data including all variables
            
        Returns:
            List[plt.Figure]: List of figures telling the story
        """
        figures = []
        
        # 1. The Initial Puzzle: Sales Variation
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Sales'], label='Daily Sales', color='#1f77b4')
        plt.title('The Coffee Shop Mystery: Unpredictable Sales', fontsize=14, pad=20)
        plt.xlabel('Days')
        plt.ylabel('Sales')
        plt.legend()
        figures.append(fig1)
        
        # 2. The First Suspect: Weather
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        # Time alignment
        ax1.plot(df.index, df['Sales'], label='Sales', alpha=0.6)
        ax1.fill_between(df.index, 0, df['Sales'].max(),
                        where=df['Weather'] == 1,
                        alpha=0.2, color='blue', label='Cold Days')
        ax1.set_title('Sales and Weather Patterns', fontsize=12)
        ax1.legend()
        
        # Relationship
        sns.boxplot(data=df, x='Weather', y='Sales', ax=ax2)
        ax2.set_title('Sales Distribution by Weather', fontsize=12)
        figures.append(fig2)
        
        # 3. The Hidden Player: Competitor
        if 'Competitor' in df.columns:
            fig3 = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 2, figure=fig3)
            
            # Weather-Competitor relationship
            ax1 = fig3.add_subplot(gs[0, :])
            ax1.plot(df.index, df['Weather'].rolling(7).mean(),
                    label='Cold Weather Trend', alpha=0.6)
            ax1.plot(df.index, df['Competitor'].rolling(7).mean(),
                    label='Competitor Presence Trend', alpha=0.6)
            ax1.set_title('Weather and Competitor Patterns', fontsize=12)
            ax1.legend()
            
            # Sales with competitor
            ax2 = fig3.add_subplot(gs[1, 0])
            sns.boxplot(data=df, x='Competitor', y='Sales', ax=ax2)
            ax2.set_title('Sales vs Competitor Presence', fontsize=12)
            
            # Weather-Sales by Competitor
            ax3 = fig3.add_subplot(gs[1, 1])
            df_cold = df[df['Weather'] == 1]
            df_warm = df[df['Weather'] == 0]
            positions = [0, 1, 3, 4]  # Positions for grouped boxplot
            boxes = [
                df_warm[df_warm['Competitor'] == 0]['Sales'],
                df_warm[df_warm['Competitor'] == 1]['Sales'],
                df_cold[df_cold['Competitor'] == 0]['Sales'],
                df_cold[df_cold['Competitor'] == 1]['Sales']
            ]
            labels = ['Warm,No Comp', 'Warm,Comp', 'Cold,No Comp', 'Cold,Comp']
            ax3.boxplot(boxes, positions=positions)
            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels, rotation=45)
            ax3.set_title('Sales by Weather and Competitor', fontsize=12)
            figures.append(fig3)
        
        # 4. The Complete Picture
        fig4 = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig4)
        
        # Combined time series
        ax1 = fig4.add_subplot(gs[0, :])
        vars_to_plot = ['Sales', 'Foot_Traffic', 'Social_Media']
        if 'Competitor' in df.columns:
            vars_to_plot.append('Competitor')
        for var in vars_to_plot:
            # Normalize for comparison
            values = (df[var] - df[var].mean()) / df[var].std()
            ax1.plot(df.index, values, label=var, alpha=0.7)
        ax1.set_title('All Variables Over Time (Normalized)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Create subplot for correlation scatter plots
        ax2 = fig4.add_subplot(gs[1, :])
        # Use a different visualization instead of scatter matrix
        vars_for_corr = ['Sales', 'Foot_Traffic', 'Social_Media']
        if 'Competitor' in df.columns:
            vars_for_corr.append('Competitor')
        
        corr = df[vars_for_corr].corr()
        sns.heatmap(corr, 
                   annot=True, fmt='.2f',
                   cmap='coolwarm', center=0,
                   square=True, ax=ax2)
        ax2.set_title('Correlation Matrix', fontsize=12)
        
        plt.tight_layout()
        figures.append(fig4)
        
        return figures
    
    def plot_correlation_matrix(self,
                              corr_matrix: pd.DataFrame,
                              figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot correlation matrix heatmap.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Correlation Analysis', fontsize=12, pad=15)
        
        plt.tight_layout()
        return fig
    
    def plot_dml_effects(self, dml_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot DML treatment effects with confidence intervals.
        
        Args:
            dml_results (dict): Dictionary of DML results by treatment
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        # Extract effects and confidence intervals
        treatments = []
        effects = []
        errors = []
        
        for treatment, result in dml_results.items():
            if result.get('success', False):
                treatments.append(treatment)
                effects.append(result['ate'])
                errors.append(result['ate_std'] * 1.96)  # 95% CI
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(treatments))
        
        # Plot error bars
        ax.errorbar(effects, y_pos, xerr=errors, fmt='o', capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(treatments)
        
        # Add reference line at zero
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add effect size annotations
        for i, (effect, error) in enumerate(zip(effects, errors)):
            ax.annotate(f'Effect: {effect:.2f}\n(±{error:.2f})',
                        xy=(effect, i),
                        xytext=(10, 0), 
                        textcoords='offset points')
        
        ax.set_title('Treatment Effects with 95% Confidence Intervals\n(Double ML Estimates)', 
                     fontsize=12, pad=20)
        ax.set_xlabel('Estimated Effect on Sales')
        
        plt.tight_layout()
        return fig
    
    def plot_weather_iv_relationships(self,
                                  data: pd.DataFrame,
                                  instrument: str,
                                  treatment: str,
                                  outcome: str,
                                  title_prefix: str = "") -> Tuple[plt.Figure, Tuple[float, float]]:
        """Create boxplot visualization with weather labels and colors.
        
        Args:
            data (pd.DataFrame): Input data
            instrument (str): Name of instrument variable
            treatment (str): Name of treatment variable
            outcome (str): Name of outcome variable
            title_prefix (str): Prefix for plot title
            
        Returns:
            Tuple[plt.Figure, Tuple[float, float]]: Figure and (first_stage_diff, reduced_form_diff)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create weather labels mapping 
        data = data.copy()
        data['Weather_Label'] = data[instrument].map({1: 'Cold', 0: 'Warm'})
        
        # Color palette
        colors = ['#F4A460', '#4B9CD3']  # Orange for Warm(0), Blue for Cold(1)
        
        # First stage relationship
        sns.boxplot(data=data, x='Weather_Label', y=treatment, ax=ax1, 
                   order=['Warm', 'Cold'], palette=colors)
        first_stage_diff = data[data[instrument] == 1][treatment].mean() - \
                          data[data[instrument] == 0][treatment].mean()
        ax1.set_title(f'{title_prefix}\nFirst Stage: Cold-Warm Difference = {first_stage_diff:.2f}')
        
        # Reduced form relationship
        sns.boxplot(data=data, x='Weather_Label', y=outcome, ax=ax2,
                   order=['Warm', 'Cold'], palette=colors)
        reduced_form_diff = data[data[instrument] == 1][outcome].mean() - \
                           data[data[instrument] == 0][outcome].mean()
        ax2.set_title(f'Reduced Form: Cold-Warm Difference = {reduced_form_diff:.2f}')
        
        plt.tight_layout()
        return fig, (first_stage_diff, reduced_form_diff)

    def plot_iv_results(self,
                       iv_results: Dict[str, Any],
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot IV analysis results with confidence intervals.
        
        Args:
            iv_results (dict): Results from IV analysis
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot effect estimate with confidence interval
        x = [iv_results['iv_effect']]
        y = [0]
        xerr = [[iv_results['iv_effect'] - iv_results['ci_lower']],
                [iv_results['ci_upper'] - iv_results['iv_effect']]]
        
        ax.errorbar(x, y, xerr=xerr, fmt='o', capsize=5, color='blue',
                    label=f'IV Effect: {iv_results["iv_effect"]:.2f}')
        
        # Add reference line at zero
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Customize plot
        ax.set_title('IV Analysis Results with 95% CI', fontsize=12, pad=15)
        ax.set_xlabel('Effect Size')
        ax.set_yticks([])
        
        # Add annotations
        ax.annotate(f'Effect: {iv_results["iv_effect"]:.2f}\n' +
                   f'SE: {iv_results["iv_std_error"]:.2f}\n' +
                   f'95% CI: [{iv_results["ci_lower"]:.2f}, {iv_results["ci_upper"]:.2f}]',
                   xy=(iv_results['iv_effect'], 0),
                   xytext=(10, 10),
                   textcoords='offset points')
        
        plt.tight_layout()
        return fig
    
    def plot_transfer_entropy_network(self,
                                    te_values: Dict[str, float],
                                    threshold: float = 0.05,
                                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot transfer entropy as a network diagram."""
        if not te_values:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No significant transfer entropy found',
                    ha='center', va='center')
            ax.set_title('Transfer Entropy Network')
            ax.axis('off')
            return fig

        # Normalize TE values
        max_te = max(te_values.values())
        te_values = {k: v/max_te for k, v in te_values.items()}
        """Plot transfer entropy as a network diagram.
        
        Args:
            te_values (dict): Transfer entropy values
            threshold (float): Minimum TE value to show
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges for TE values above threshold
        for edge, te in te_values.items():
            if te > threshold:
                source, target = edge.split(' → ')
                G.add_edge(source, target, weight=te)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.6)
        
        # Draw edges with width proportional to TE
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos,
                             width=edge_widths,
                             edge_color='gray',
                             alpha=0.6,
                             arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add edge labels (TE values)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.3f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        ax.set_title('Transfer Entropy Network', fontsize=12, pad=15)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_method_comparison_story(self,
                                results: Dict[str, Any],
                                df: pd.DataFrame) -> List[plt.Figure]:
        """Create story-driven comparison of different causal methods.
        
        Args:
            results (dict): Results from all methods
            df (pd.DataFrame): Original data
            
        Returns:
            List[plt.Figure]: List of figures telling the story
        """
        figures = []
        
        # 1. Correlation vs Reality
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation heatmap
        sns.heatmap(results['correlation'],
                    mask=np.triu(np.ones_like(results['correlation'], dtype=bool)),
                    annot=True, fmt='.2f',
                    cmap='coolwarm', center=0,
                    square=True, ax=ax1)
        ax1.set_title('Traditional Correlation Analysis', fontsize=12)
        
        # Scatter with hidden variable
        if 'Competitor' in df.columns:
            scatter = ax2.scatter(df['Weather'], df['Sales'],
                                c=df['Competitor'],
                                cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Competitor Present')
            ax2.set_title('Weather-Sales Relationship\nColored by Hidden Competitor', fontsize=12)
            ax2.set_xlabel('Weather')
            ax2.set_ylabel('Sales')
        figures.append(fig1)
        
        # 2. IV and DML Attempts
        fig2 = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 3, figure=fig2)
        
        # IV First Stage
        ax1 = fig2.add_subplot(gs[0])
        for key, result in results['iv'].items():
            if 'Weather' in key:  # Plot Weather as instrument
                instrument, treatment = key.split('__')
                ax1.scatter(df[instrument], df[treatment], alpha=0.4)
                pred_treatment = result['first_stage'].predict(df[[instrument]])
                ax1.plot(df[instrument], pred_treatment, color='red', alpha=0.7)
                ax1.set_title(f'IV First Stage\n{instrument} → {treatment}', fontsize=12)
                ax1.set_xlabel(instrument)
                ax1.set_ylabel(treatment)
        
        # DML Effects
        ax2 = fig2.add_subplot(gs[1])
        effects = []
        labels = []
        errors = []
        cis_lower = []
        cis_upper = []
        for treatment, result in results['double_ml'].items():
            if result.get('success', False):
                effects.append(result['ate'])
                errors.append(result['ate_std'])
                cis_lower.append(result['ci_lower'])
                cis_upper.append(result['ci_upper'])
                labels.append(treatment)
        if effects:
            ax2.errorbar(effects, range(len(effects)),
                        xerr=errors, fmt='o', capsize=5)
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels)
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Double ML Effects', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'No successful DML results',
                    ha='center', va='center')
        
        # Method limitations
        ax3 = fig2.add_subplot(gs[2])
        methods = ['Correlation', 'IV', 'Double ML', 'Transfer Entropy']
        features = ['Handles Hidden Variables', 'Shows Direction', 
                   'Captures Nonlinearity', 'Time-Aware']
        scores = np.array([
            [0, 0, 0, 0],  # Correlation
            [1, 1, 0, 0],  # IV
            [1, 1, 1, 0],  # Double ML
            [1, 1, 1, 1]   # TE
        ])
        sns.heatmap(scores, annot=True, fmt='d',
                    xticklabels=features,
                    yticklabels=methods,
                    cmap='YlOrRd', ax=ax3)
        ax3.set_title('Method Capabilities', fontsize=12)
        plt.xticks(rotation=45)
        figures.append(fig2)
        
        # 3. Transfer Entropy Network
        fig3, ax = plt.subplots(figsize=(12, 8))
        G = nx.DiGraph()
        
        # Add edges from TE results
        max_te = max(results['transfer_entropy'].values())
        for edge, te in results['transfer_entropy'].items():
            if te > 0.05 * max_te:  # 5% threshold
                source, target = edge.split(' → ')
                G.add_edge(source, target, weight=te/max_te)
        
        # Layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000,
                             node_color='lightblue',
                             alpha=0.6, ax=ax)
        
        # Draw edges with varying width and color
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        # Create a mappable object for the colorbar
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        
        if weights:  # Only draw edges if there are any
            nx.draw_networkx_edges(G, pos, 
                                 width=[w*5 for w in weights],
                                 edge_color=weights,
                                 edge_cmap=plt.cm.viridis,
                                 edge_vmin=0, edge_vmax=1,
                                 arrowsize=20, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Add title
        ax.set_title('Transfer Entropy Network\nRevealing True Causal Structure',
                  fontsize=14, pad=20)
        
        # Add colorbar
        if weights:  # Only add colorbar if there are edges
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(sm, cax=cax, label='Normalized Transfer Entropy')
        
        ax.axis('off')
        plt.tight_layout()
        figures.append(fig3)
        
        return figures