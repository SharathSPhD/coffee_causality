"""
Data Generator module for coffee shop synthetic data with hidden confounders.

This module provides functionality to generate synthetic data for a coffee shop
scenario, including various features like weather, foot traffic, and sales data.
The generated data includes hidden confounders and causal relationships.

Example:
    >>> generator = DataGenerator(seed=42)
    >>> data = generator.generate_data(n_days=200)
    >>> print(data.columns)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class DataGenerator:
    """Generates synthetic coffee shop data with hidden confounders.
    
    This class creates realistic synthetic data that simulates the operations of a
    coffee shop, including various factors that might affect sales and their
    interrelationships.
    
    Attributes:
        seed (int): Random seed for reproducibility
        effects (dict): Dictionary of true causal effects
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # True causal effects
        self.effects = {
            'weather_to_competitor': 0.7,  # Weather influences competitor presence
            'competitor_to_traffic': -30.0,  # Competitor reduces foot traffic
            'competitor_to_sales': -25.0,   # Competitor directly impacts sales
            'traffic_to_sales': 0.7,        # Conversion rate
            'social_to_sales': 10.0         # Social media boost
        }
    
    def generate_time_effects(self, n_days: int) -> np.ndarray:
        """Generate base time patterns (weekly, seasonal).
        
        Args:
            n_days (int): Number of days to generate
            
        Returns:
            np.ndarray: Base time effects
        """
        days = np.arange(n_days)
        weekday = days % 7
        
        # Weekly pattern
        weekly = -10 * (weekday == 0) + 5 * (weekday == 5)  # Weekend effects
        
        # Seasonal pattern (yearly)
        seasonal = 20 * np.sin(2 * np.pi * days / 365)
        
        return weekly + seasonal
    
    def generate_weather(self, n_days: int) -> np.ndarray:
        """Generate weather patterns.
        
        Args:
            n_days (int): Number of days
            
        Returns:
            np.ndarray: Weather values (cold = 1, warm = 0)
        """
        # Base seasonal pattern
        days = np.arange(n_days)
        seasonal_prob = 0.3 + 0.3 * np.sin(2 * np.pi * (days + 90) / 365)
        
        # Generate cold days
        return np.random.binomial(1, seasonal_prob, n_days)
    
    def generate_competitor(self, weather: np.ndarray) -> np.ndarray:
        """Generate competitor presence based on weather.
        
        Args:
            weather (np.ndarray): Weather values
            
        Returns:
            np.ndarray: Competitor presence (1 = present)
        """
        n_days = len(weather)
        competitor = np.zeros(n_days)
        
        for t in range(n_days):
            if weather[t] == 1:  # Cold day
                prob = self.effects['weather_to_competitor']
            else:
                prob = 0.1
            competitor[t] = np.random.binomial(1, prob)
            
        return competitor
    
    def generate_social_media(self, n_days: int) -> np.ndarray:
        """Generate social media activity.
        
        Args:
            n_days (int): Number of days
            
        Returns:
            np.ndarray: Social media engagement scores
        """
        # Base engagement
        base = np.random.poisson(5, n_days)
        
        # Random viral posts (more frequent)
        viral = np.random.binomial(1, 0.1, n_days) * np.random.poisson(50, n_days)
        
        return base + viral
    
    def generate_data(self, n_days: int = 365, include_hidden: bool = False) -> pd.DataFrame:
        """Generate complete coffee shop dataset.
        
        Args:
            n_days (int): Number of days to generate
            include_hidden (bool): Whether to include hidden variables
            
        Returns:
            pd.DataFrame: Generated data
        """
        # Generate base patterns
        time_effects = self.generate_time_effects(n_days)
        weather = self.generate_weather(n_days)
        competitor = self.generate_competitor(weather)
        social_media = self.generate_social_media(n_days)
        
        # Generate foot traffic with more deterministic relationships
        base_traffic = 100 + time_effects
        foot_traffic = np.zeros(n_days)
        
        # Add strong lagged effects
        for t in range(1, n_days):
            if weather[t-1] == 1:  # Cold weather yesterday
                foot_traffic[t] = base_traffic[t] * 0.5  # Strong reduction
            else:
                foot_traffic[t] = base_traffic[t] * (1.0 + 0.2 * (social_media[t-1] > 20))  # Boost from social media
        
        # Add competitor effect with less noise
        foot_traffic += self.effects['competitor_to_traffic'] * competitor + \
                       np.random.normal(0, 2, n_days)
        
        # Generate sales with stronger dependencies
        base_sales = self.effects['traffic_to_sales'] * foot_traffic
        sales = np.maximum(0, base_sales +
                self.effects['competitor_to_sales'] * competitor +
                self.effects['social_to_sales'] * (social_media > 20) +
                np.random.normal(0, 5, n_days))
        
        # Create DataFrame
        data = {
            'Weather': weather,
            'Foot_Traffic': foot_traffic,
            'Social_Media': social_media,
            'Sales': sales
        }
        
        if include_hidden:
            data['Competitor'] = competitor
            
        return pd.DataFrame(data)
    
    def get_true_effects(self) -> Dict[str, float]:
        """Get the ground truth causal effects.
        
        Returns:
            Dict[str, float]: Dictionary of true causal effects
        """
        return self.effects.copy()