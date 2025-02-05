"""
Causal Analysis module implementing various methods.

This module provides implementations of different causal inference techniques
including Instrumental Variables (IV), Double Machine Learning (DML), and
Transfer Entropy analysis for analyzing causal relationships in coffee shop data.

Example:
    >>> analyzer = CausalAnalyzer()
    >>> iv_results = analyzer.instrumental_variables(data, 'Sales', 'Foot_Traffic', 'Weather')
    >>> print(iv_results['iv_effect'])
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import LinearDML
from econml.inference import BootstrapInference, StatsModelsInference

# Add IDTxl to Python path
idtxl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'IDTxl')
sys.path.append(idtxl_path)

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from typing import Dict, Any, Tuple, Optional

class CausalAnalyzer:
    """Implements various causal inference methods.
    
    This class provides methods for analyzing causal relationships using
    different approaches including IV analysis, DML, and Transfer Entropy.
    
    Methods:
        instrumental_variables: Perform IV analysis
        double_ml_analysis: Implement Double Machine Learning method
        transfer_entropy_analysis: Calculate information flow using Transfer Entropy
    """
    
    def __init__(self, log_file=None):
        """Initialize the analyzer with proper logging.
        
        Args:
            log_file (str, optional): Path to log file. If None, creates a timestamped file.
        """
        # Setup logging
        if log_file is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'causal_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("CausalAnalyzer initialized")

    def _format_inference_results(self, effect_array):
        """Format array results for logging.
        
        Args:
            effect_array (np.ndarray): Array of effects to format
            
        Returns:
            str: Formatted string with mean and std
        """
        if effect_array.ndim == 1:
            return f"Mean: {np.mean(effect_array):.4f}, Std: {np.std(effect_array):.4f}"
        else:
            return f"Mean: {np.mean(effect_array, axis=0)}, Std: {np.std(effect_array, axis=0)}"
    
    def correlation_analysis(self, df: pd.DataFrame, 
                           variables: Optional[list] = None) -> pd.DataFrame:
        """Calculate correlations between variables.
        
        Args:
            df (pd.DataFrame): Input data
            variables (list, optional): List of variables to analyze. If None, uses all.
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if variables is None:
            return df.corr()
        return df[variables].corr()
    
    def double_ml_analysis(self, df: pd.DataFrame,
                          treatment: str,
                          outcome: str,
                          features: list) -> Dict[str, Any]:
        """Run double machine learning analysis with robust inference.
        
        Args:
            df (pd.DataFrame): Input data
            treatment (str): Treatment variable name
            outcome (str): Outcome variable name
            features (list): List of feature names
            
        Returns:
            dict: Results including ATE, std errors, and confidence intervals
        """
        try:
            # Validate inputs
            if not all(col in df.columns for col in [treatment, outcome] + (features if features else [])):
                missing = [col for col in [treatment, outcome] + (features if features else []) if col not in df.columns]
                raise ValueError(f"Missing columns in data: {missing}")

            # Prepare data
            Y = df[outcome].values
            T = df[treatment].values
            X = df[features].values if features else None
            
            # Initialize base model with limited complexity
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=10,
                random_state=42
            )
            
            # Initialize DML model
            est = LinearDML(
                model_y=model,
                model_t=model,
                random_state=42,
                cv=5  # Use 5-fold CV
            )
            
            # Fit with statsmodels inference
            self.logger.info(f"Fitting DML for {treatment} with statsmodels inference")
            est.fit(Y, T, X=X, inference=StatsModelsInference())
            
            # Get effect estimates
            ate = est.effect(X)
            
            # Get confidence intervals
            effect_interval = est.effect_interval(X, alpha=0.05)
            
            # Get nuisance scores if available
            nuisance_scores = {}
            if hasattr(est, 'nuisance_scores_y'):
                nuisance_scores['y_r2'] = float(np.mean(est.nuisance_scores_y))
            if hasattr(est, 'nuisance_scores_t'):
                nuisance_scores['t_r2'] = float(np.mean(est.nuisance_scores_t))
                
            results = {
                'ate': float(np.mean(ate)),
                'ate_std': float(np.std(ate)),
                'ci_lower': float(np.mean(effect_interval[0])),
                'ci_upper': float(np.mean(effect_interval[1])),
                'nuisance_scores': nuisance_scores,
                'success': True
            }
            
            # Log results
            self.logger.info(f"ATE: {self._format_inference_results(ate)}")
            self.logger.info(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
            for score_name, score in nuisance_scores.items():
                self.logger.info(f"{score_name}: {score:.4f}")
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f'Error in double ML analysis: {error_msg}')
            return {'error': error_msg, 'success': False}
    
    def instrumental_variables(self, df: pd.DataFrame,
                             outcome: str,
                             treatment: str,
                             instrument: str) -> Dict[str, Any]:
        """Perform instrumental variables analysis.
        
        Args:
            df (pd.DataFrame): Input data
            outcome (str): Name of outcome variable
            treatment (str): Name of treatment variable
            instrument (str): Name of instrument variable
            
        Returns:
            dict: Results including first stage and IV effects
        """
        try:
            # First stage: E[D|Z=1] - E[D|Z=0]
            first_stage = df[df[instrument] == 1][treatment].mean() - \
                         df[df[instrument] == 0][treatment].mean()
            
            # Reduced form: E[Y|Z=1] - E[Y|Z=0]
            reduced_form = df[df[instrument] == 1][outcome].mean() - \
                          df[df[instrument] == 0][outcome].mean()
            
            # Wald estimator
            iv_effect = reduced_form / first_stage
            
            # Standard errors
            n_warm = (df[instrument] == 0).sum()
            n_cold = (df[instrument] == 1).sum()
            
            var_d_warm = df[df[instrument] == 0][treatment].var()
            var_d_cold = df[df[instrument] == 1][treatment].var()
            var_y_warm = df[df[instrument] == 0][outcome].var()
            var_y_cold = df[df[instrument] == 1][outcome].var()
            
            se_first = np.sqrt(var_d_warm/n_warm + var_d_cold/n_cold)
            se_reduced = np.sqrt(var_y_warm/n_warm + var_y_cold/n_cold)
            
            se_iv = np.sqrt((se_reduced**2 / first_stage**2) + 
                            (reduced_form**2 * se_first**2 / first_stage**4))
            
            # Save detailed results
            results = {
                'iv_effect': float(iv_effect),
                'iv_std_error': float(se_iv),
                'first_stage_diff': float(first_stage),
                'reduced_form_diff': float(reduced_form),
                'ci_lower': float(iv_effect - 1.96 * se_iv),
                'ci_upper': float(iv_effect + 1.96 * se_iv),
                'n_warm': int(n_warm),
                'n_cold': int(n_cold),
                'success': True
            }
            
            # Log results
            self.logger.info(f"IV Results for {treatment} using {instrument}:")
            self.logger.info(f"IV Effect: {results['iv_effect']:.4f} (SE: {results['iv_std_error']:.4f})")
            self.logger.info(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
            self.logger.info(f"First Stage Difference: {results['first_stage_diff']:.4f}")
            self.logger.info(f"Reduced Form Difference: {results['reduced_form_diff']:.4f}")
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f'Error in IV analysis: {error_msg}')
            return {'error': error_msg, 'success': False}
    
    def transfer_entropy_analysis(self, df: pd.DataFrame,
                                variables: list,
                                max_lag: int = 5) -> Dict[str, float]:
        """Calculate transfer entropy between variables using IDTxl.
        
        Args:
            df (pd.DataFrame): Input data
            variables (list): List of variable names to analyze
            max_lag (int, optional): Maximum lag to consider. Defaults to 5.
            
        Returns:
            dict: Dictionary mapping variable pairs to their transfer entropy values
        """
        try:
            # Normalize and detrend data
            df_norm = pd.DataFrame()
            for col in variables:
                series = df[col]
                df_norm[col] = (series - series.mean()) / series.std()
            
            # Transfer entropy settings optimized for business data
            settings = {
                'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 2,      # Focus on short-term effects
                'min_lag_sources': 1,      # Start from lag 1
                'n_perm_max_stat': 200,    # Reasonable permutations
                'alpha_max_stat': 0.2,     # Generous significance level
                'tau_sources': 1,          # Single time step
                'local_values': False,     # Global analysis
                'verbose': False
            }
            
            # Prepare data for IDTxl
            data_array = np.array([df_norm[var].values for var in variables])
            data_obj = Data(data_array, dim_order='ps')
            
            # Initialize TE calculator
            te_calculator = MultivariateTE()
            te_values = {}
            
            # Process each target
            for target_idx, target in enumerate(variables):
                try:
                    # Analyze target
                    results = te_calculator.analyse_single_target(
                        settings=settings,
                        data=data_obj,
                        target=target_idx
                    )
                    
                    # Get results
                    target_results = results.get_single_target(target=target_idx, fdr=False)
                    
                    if target_results.sources_tested:
                        for src_idx in target_results.sources_tested:
                            if src_idx != target_idx:  # Skip self-loops
                                source = variables[src_idx]
                                try:
                                    te = max(target_results.omnibus_te, target_results.conditional_te[src_idx])
                                except:
                                    te = target_results.omnibus_te
                                te_values[f"{source} → {target}"] = float(te)
                                self.logger.info(f"TE from {source} to {target}: {te:.4f}")
                except Exception as e:
                    self.logger.error(f"Error analyzing {target}: {str(e)}")
            
            return te_values
            
        except Exception as e:
            self.logger.error(f'Error in transfer entropy analysis: {str(e)}')
            return {}
    
    def compare_methods(self, df: pd.DataFrame,
                       target: str = 'Sales') -> Dict[str, Any]:
        """Compare different causal inference methods.
        
        Args:
            df (pd.DataFrame): Input data
            target (str, optional): Target variable. Defaults to 'Sales'.
            
        Returns:
            dict: Results from all methods
        """
        results = {}
        
        # 1. Correlation Analysis
        self.logger.info("Running correlation analysis...")
        results['correlation'] = self.correlation_analysis(df)
        
        # 2. Double ML for each possible treatment
        self.logger.info("Running Double ML analysis...")
        results['double_ml'] = {}
        treatments = ['Weather', 'Social_Media', 'Competitor']
        features = ['Foot_Traffic']
        
        for treatment in treatments:
            results['double_ml'][treatment] = self.double_ml_analysis(
                df,
                treatment=treatment,
                outcome=target,
                features=features
            )
        
        # 3. Transfer Entropy
        self.logger.info("Running Transfer Entropy analysis...")
        variables = ['Weather', 'Foot_Traffic', 'Social_Media', target]
        if 'Competitor' in df.columns:
            variables.append('Competitor')
        
        results['transfer_entropy'] = self.transfer_entropy_analysis(
            df, variables=variables
        )
        
        return results