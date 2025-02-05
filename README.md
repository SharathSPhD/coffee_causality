# Coffee Shop Causality Analysis

A Python package demonstrating causal inference methods through a coffee shop business case study.

## Features

- Synthetic data generation for coffee shop sales
- Multiple causal inference methods:
  - Correlation Analysis
  - Instrumental Variables (IV)
  - Double Machine Learning (DML)
  - Transfer Entropy
- Interactive visualizations
- Story-driven Jupyter notebooks
- Documentation using Sphinx
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/coffee_causality.git
cd coffee_causality
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Jupyter kernel:
```bash
python -m ipykernel install --user --name=coffee_causality_venv --display-name "Python (coffee_causality)"
```

## Project Structure

```
coffee_causality/
├── .github/                    # GitHub Actions workflows and templates
├── book/                      # Documentation and tutorials
├── docs/                      # Sphinx documentation
│   ├── build/                # Generated documentation
│   ├── source/              # Documentation source files
│   ├── Makefile            # Documentation build script
│   └── conf.py            # Sphinx configuration
├── notebooks/                 # Jupyter notebooks
│   ├── A_Problem_Definition.ipynb       # Problem setup and data generation
│   ├── B_Initial_Analysis.ipynb        # Basic statistical analysis
│   ├── C_Advanced_Analysis.ipynb       # Advanced causal methods
│   └── D_Transfer_Entropy.ipynb        # Information flow analysis
├── plots/                     # Generated visualizations
├── results/                   # Analysis results and outputs
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic data generation
│   ├── causal_analysis.py    # Causal inference implementations
│   └── visualization.py      # Visualization tools

├── .gitignore               # Git ignore rules
├── LICENSE                  # GPLv3 License
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## Core Components

### Source Code (`src/`)
- `data_generator.py`: Creates synthetic coffee shop data with hidden confounders
- `causal_analysis.py`: Implements various causal inference methods
- `visualization.py`: Provides visualization tools for analysis results

### Documentation (`docs/`)
- API Reference
- Implementation details
- Usage guides and tutorials
- Example notebooks

### Analysis Notebooks (`notebooks/`)
- Step-by-step analysis examples
- Interactive visualizations
- Results interpretation

## Dependencies

- **Data Processing**
  - numpy: Numerical computing
  - pandas: Data manipulation
  
- **Visualization**
  - matplotlib: Basic plotting
  - seaborn: Statistical visualizations
  - networkx: Network analysis
  
- **Machine Learning**
  - scikit-learn: Machine learning utilities
  - econml: Double ML implementation
  
- **Causal Analysis**
  - statsmodels: Statistical modeling
  - IDTxl: Information theory
  
- **Development**
  - jupyter: Notebook interface
  - pytest: Testing framework
  - sphinx: Documentation generation
  - sphinx-rtd-theme: Documentation theme

## Documentation

The documentation is built using Sphinx and can be found in the `docs/` directory. To build the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `docs/build/html/index.html`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing_feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing_feature`)
5. Open a Pull Request

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
