Causal Analysis
==============

The CausalAnalyzer class implements various causal inference methods:

Instrumental Variables Analysis
-----------------------------
.. py:function:: analyze_weather_iv(data, outcome, treatment, instrument)

   Performs instrumental variables analysis using weather as an instrument.

   :param data: DataFrame containing the variables
   :param outcome: Name of the outcome variable (e.g., 'Sales')
   :param treatment: Name of the treatment variable (e.g., 'Foot_Traffic')
   :param instrument: Name of the instrument variable (e.g., 'Weather')
   :return: Dictionary containing IV analysis results

Double Machine Learning
---------------------
.. py:function:: double_ml_analysis(df, treatment, outcome, features)

   Implements double/debiased machine learning for causal inference.

   :param df: Input DataFrame
   :param treatment: Treatment variable name
   :param outcome: Outcome variable name
   :param features: List of feature names
   :return: Dictionary containing DML results

Transfer Entropy Analysis
-----------------------
.. py:function:: transfer_entropy_analysis(data, variables)

   Calculates transfer entropy between variables to analyze information flow.

   :param data: Input DataFrame
   :param variables: List of variable names
   :return: Dictionary mapping variable pairs to transfer entropy values