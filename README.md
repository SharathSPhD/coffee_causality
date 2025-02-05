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
- Story-driven Jupyter notebook

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

## Usage

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/causal_coffee_analysis.ipynb`

3. Select kernel: Kernel → Change kernel → Python (coffee_causality)

## Project Structure

```
coffee_causality/
├── notebooks/
│   └── causal_coffee_analysis.ipynb
├── src/
│   ├── data_generator.py
│   ├── causal_analysis.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Core Components

- `data_generator.py`: Creates synthetic coffee shop data with hidden confounders
- `causal_analysis.py`: Implements various causal inference methods
- `visualization.py`: Provides visualization tools for analysis results

## Dependencies

- numpy: Numerical computing
- pandas: Data manipulation
- matplotlib/seaborn: Visualization
- networkx: Network analysis
- scikit-learn: Machine learning
- econml: Double ML implementation
- pyitlib: Information theory
- IDTxl: Transfer entropy
- statsmodels: Statistical modeling
- jupyter: Notebook interface

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request
<<<<<<< HEAD
=======

## License

GPLv3

