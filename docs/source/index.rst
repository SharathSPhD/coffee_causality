Coffee Shop Causality Analysis
===============================

Welcome to the Coffee Shop Causality Analysis documentation! This project provides tools for analyzing causal relationships in coffee shop data using various methods including Instrumental Variables (IV), Double Machine Learning (DML), and Transfer Entropy.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   contributing
   modules

Features
--------
- Causal analysis using multiple methods:
   * Instrumental Variables (IV)
   * Double Machine Learning (DML)
   * Transfer Entropy Analysis
- Comprehensive visualization tools
- Interactive data exploration
- Statistical validation techniques

Getting Started
----------------
To get started, check out the :doc:`installation` guide followed by the :doc:`tutorials`.

Quick Example
--------------
.. code-block:: python

   from causal_analysis import CausalAnalyzer
   from visualization import CausalVisualizer

   # Initialize analyzer and visualizer
   analyzer = CausalAnalyzer()
   visualizer = CausalVisualizer()

   # Analyze causal relationships
   results = analyzer.double_ml_analysis(data, "Weather", "Sales", ["Foot_Traffic"])

   # Visualize results
   visualizer.plot_dml_effects(results)