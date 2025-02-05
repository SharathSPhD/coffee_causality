Visualization
============

The CausalVisualizer class provides methods for creating various visualizations:

Synthetic Story Plots
-------------------
.. py:function:: plot_synthetic_story(data)

   Creates a series of plots to tell the data story.

   :param data: DataFrame containing coffee shop data
   :return: None (displays plots)

   Creates three main plot groups:
   1. Daily Sales Pattern
   2. Weather and Sales Relationships
   3. Competitor Analysis

Weather IV Relationships
----------------------
.. py:function:: plot_weather_iv_relationships(data, instrument, treatment, outcome)

   Visualizes instrumental variable relationships.

   :param data: Input DataFrame
   :param instrument: Name of instrument variable
   :param treatment: Name of treatment variable
   :param outcome: Name of outcome variable
   :return: Figure object and relationship differences

Information Flow Network
----------------------
.. py:function:: plot_information_network(te_results)

   Creates network visualization of information flow.

   :param te_results: Dictionary of transfer entropy results
   :return: NetworkX graph visualization