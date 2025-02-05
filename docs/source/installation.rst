Installation
============

Requirements
-------------
- Python 3.8 or higher
- pip package manager

Dependencies
-------------
The project requires the following main packages:

- numpy
- pandas
- scikit-learn
- econml
- IDTxl
- matplotlib
- seaborn
- networkx
- statsmodels

Installation Steps
-------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/coffee_causality.git
      cd coffee_causality

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install required packages:

   .. code-block:: bash

      pip install -r requirements.txt

Development Installation
-------------------------

For development, install additional dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt

This will install additional packages needed for development:

- pytest for testing
- sphinx for documentation
- black for code formatting
- flake8 for linting