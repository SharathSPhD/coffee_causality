Part A
```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_dir)

from data_generator import DataGenerator
from visualization import CausalVisualizer

# Set up plotting style
sns.set_style("whitegrid")
```

```python
# Time to generate some caffeinated data!
generator = DataGenerator(seed=42)  # Because 42 is always the answer

# Generate 200 days of café drama
data = generator.generate_data(n_days=200, include_hidden=True)

print("Generated data shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nFirst few days of mystery data:")
data.head()
```

```python
class CausalVisualizer:
    def plot_synthetic_story(self, data):
        """Create plots for telling the data story."""
        
        # Plot 1: Daily Sales Pattern
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(data)), data['Sales'], color='blue', label='Daily Sales')
        plt.title('The Coffee Shop Mystery: Unpredictable Sales')
        plt.xlabel('Days')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
        
        # Plot 2: Weather and Sales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time series with weather overlay
        ax1.plot(range(len(data)), data['Sales'], color='navy', label='Sales')
        ax1.fill_between(range(len(data)), 0, 100, 
                        where=data['Weather']==1,
                        color='lightblue', alpha=0.3,
                        label='Cold Days')
        ax1.set_title('Sales and Weather Patterns')
        ax1.set_xlabel('Days')
        ax1.legend()
        
        # Create weather labels
        data['Weather_Label'] = data['Weather'].map({0: 'Warm', 1: 'Cold'})
        
        # Box plot by weather - fixed warning by using hue
        sns.boxplot(data=data, x='Weather_Label', y='Sales', 
                   hue='Weather_Label',
                   ax=ax2,
                   palette={'Warm': 'coral', 'Cold': 'lightblue'},
                   legend=False)
        ax2.set_title('Sales Distribution by Weather')
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Competitor Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Weather and Competitor Trends
        ax1.plot(range(len(data)), data['Weather'].rolling(7).mean(), 
                label='Cold Weather Trend', color='navy')
        ax1.plot(range(len(data)), data['Competitor'].rolling(7).mean(), 
                label='Competitor Presence Trend', color='cornflowerblue')
        ax1.set_title('Weather and Competitor Patterns')
        ax1.set_xlabel('Days')
        ax1.legend()
        
        # Create competitor labels
        data['Competitor_Label'] = data['Competitor'].map(
            {0: 'No Competitor', 1: 'Competitor Present'})
        
        # Sales by Competitor - fixed warning by using hue
        sns.boxplot(data=data, x='Competitor_Label', y='Sales', 
                   hue='Competitor_Label',
                   ax=ax2,
                   palette={'No Competitor': 'navy', 'Competitor Present': 'lightgray'},
                   legend=False)
        ax2.set_title('Sales vs Competitor Presence')
        
        # Sales by Weather and Competitor combined
        g = sns.boxplot(data=data, x='Weather_Label', y='Sales', 
                       hue='Competitor_Label',
                       ax=ax4, 
                       palette=['white', 'lightgray'])
        ax4.set_title('Sales by Weather and Competitor')
        
        # Properly handling the legend
        handles, labels = ax4.get_legend_handles_labels()
        ax4.legend(handles, ['No Competitor', 'Competitor Present'], 
                  title='', loc='upper right')
        
        ax3.remove()  # Remove unused subplot
        plt.tight_layout()
        plt.show()

# Generate Plots
visualizer = CausalVisualizer()
story_plots = visualizer.plot_synthetic_story(data)
```


Part B
```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Add the src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_dir)

from data_generator import DataGenerator
from visualization import CausalVisualizer

# Set up our analytical tools
sns.set_style("whitegrid")

# Get our café data back
generator = DataGenerator(seed=42)
data = generator.generate_data(n_days=200, include_hidden=True)
```

```python
# Calculate correlations
correlation_matrix = data.corr()

# Create a heatmap that's easy on the eyes
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('The Relationship Web: Who\'s Connected to Who?')
plt.show()

# Let's look at the specific relationships with sales
sales_correlations = correlation_matrix['Sales'].sort_values(ascending=False)
print("\nRelationships with Sales (from strongest to weakest):")
for var, corr in sales_correlations.items():
    if var != 'Sales':
        print(f"{var}: {corr:.3f}")
```

```python
# Create pair plots to visualize relationships
variables = ['Sales', 'Weather', 'Foot_Traffic', 'Social_Media']
sns.pairplot(data[variables], diag_kind='kde')
plt.suptitle('Relationship Patterns between Variables', y=1.02)
plt.show()
```

```python
# Prepare features and target
features = ['Weather', 'Foot_Traffic', 'Social_Media']
X = data[features]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train our linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Print the recipe coefficients
print("Our Sales Prediction Recipe:")
print("Base sales level:", f"{model.intercept_:.2f}")
for feature, coefficient in zip(features, model.coef_):
    print(f"Impact of {feature}: {coefficient:.2f}")

# Calculate performance metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\nHow well does our recipe work?")
print(f"Training Score (R²): {train_r2:.3f}")
print(f"Testing Score (R²): {test_r2:.3f}")
print(f"Training Error (RMSE): {train_rmse:.2f}")
print(f"Testing Error (RMSE): {test_rmse:.2f}")
```

```python
# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot training predictions
ax1.scatter(y_train, y_train_pred, alpha=0.5)
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Sales')
ax1.set_ylabel('Predicted Sales')
ax1.set_title('Training Data: Reality vs. Predictions')

# Plot testing predictions
ax2.scatter(y_test, y_test_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Sales')
ax2.set_ylabel('Predicted Sales')
ax2.set_title('Testing Data: Reality vs. Predictions')

plt.tight_layout()
plt.show()

# Let's also look at the residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training residuals
residuals_train = y_train - y_train_pred
ax1.scatter(y_train_pred, residuals_train, alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Predicted Sales')
ax1.set_ylabel('Residuals')
ax1.set_title('Training Data: Residual Analysis')

# Testing residuals
residuals_test = y_test - y_test_pred
ax2.scatter(y_test_pred, residuals_test, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Sales')
ax2.set_ylabel('Residuals')
ax2.set_title('Testing Data: Residual Analysis')

plt.tight_layout()
plt.show()
```


Part C
```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_dir)

from data_generator import DataGenerator
from causal_analysis import CausalAnalyzer
from visualization import CausalVisualizer

# Set up our analytical tools
sns.set_style("whitegrid")

# Get our café data and analyzer ready
generator = DataGenerator(seed=42)
data = generator.generate_data(n_days=200, include_hidden=True)
analyzer = CausalAnalyzer()
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_weather_iv(data, outcome, treatment, instrument):
    """Perform IV analysis with weather instrument"""
    # First stage: E[D|Z=Warm] - E[D|Z=Cold]
    first_stage = data[data[instrument] == 1][treatment].mean() - \
                 data[data[instrument] == 0][treatment].mean()
    
    # Reduced form: E[Y|Z=Warm] - E[Y|Z=Cold]
    reduced_form = data[data[instrument] == 1][outcome].mean() - \
                  data[data[instrument] == 0][outcome].mean()
    
    # Wald estimator
    iv_effect = reduced_form / first_stage
    
    # Standard errors
    n_warm = (data[instrument] == 1).sum()
    n_cold = (data[instrument] == 0).sum()
    
    var_d_warm = data[data[instrument] == 1][treatment].var()
    var_d_cold = data[data[instrument] == 0][treatment].var()
    var_y_warm = data[data[instrument] == 1][outcome].var()
    var_y_cold = data[data[instrument] == 0][outcome].var()
    
    se_first = np.sqrt(var_d_warm/n_warm + var_d_cold/n_cold)
    se_reduced = np.sqrt(var_y_warm/n_warm + var_y_cold/n_cold)
    
    se_iv = np.sqrt((se_reduced**2 / first_stage**2) + 
                    (reduced_form**2 * se_first**2 / first_stage**4))
    
    results = {
        'iv_effect': iv_effect,
        'iv_std_error': se_iv,
        'first_stage_diff': first_stage,
        'reduced_form_diff': reduced_form,
        'ci_lower': iv_effect - 1.96 * se_iv,
        'ci_upper': iv_effect + 1.96 * se_iv
    }
    
    return results

def plot_weather_iv_relationships(data, instrument, treatment, outcome, title_prefix=""):
    """Create boxplot visualization with weather labels and colors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create weather labels mapping 
    data = data.copy()
    data['Weather_Label'] = data[instrument].map({1: 'Cold', 0: 'Warm'})
    
    # Color palette (now in correct order for mapping)
    colors = ['#F4A460', '#4B9CD3']  # First color (orange) for Warm(0), Second color (blue) for Cold(1)
    
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

def run_weather_iv_analysis(data):
    """Run IV analysis with weather instrument"""
    iv_combinations = [
        {
            'outcome': 'Sales',
            'treatment': 'Foot_Traffic',
            'instrument': 'Weather',
            'title': 'Weather as Instrument for Foot Traffic'
        },
        {
            'outcome': 'Sales', 
            'treatment': 'Competitor',
            'instrument': 'Weather',
            'title': 'Weather as Instrument for Competitor Effect'
        }
    ]
    
    results = {}
    for combo in iv_combinations:
        # Create visualization
        fig, diffs = plot_weather_iv_relationships(
            data,
            combo['instrument'],
            combo['treatment'],
            combo['outcome'],
            title_prefix=combo['title']
        )
        
        # Run analysis
        iv_results = analyze_weather_iv(
            data,
            combo['outcome'],
            combo['treatment'],
            combo['instrument']
        )
        
        key = f"{combo['instrument']}__{combo['treatment']}"
        results[key] = iv_results
        
        plt.show()
        print(f"\nResults for {combo['title']}:")
        print(f"IV Effect: {iv_results['iv_effect']:.4f} (SE: {iv_results['iv_std_error']:.4f})")
        print(f"95% CI: [{iv_results['ci_lower']:.4f}, {iv_results['ci_upper']:.4f}]")
        print(f"Cold-to-Warm Difference:")
        print(f"- First Stage: {iv_results['first_stage_diff']:.4f}")
        print(f"- Reduced Form: {iv_results['reduced_form_diff']:.4f}")
        print("-" * 50)
    
    return results

# Run the analysis
weather_iv_results = run_weather_iv_analysis(data)
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def run_dml_analysis(data, analyzer):
    """Run Double ML with proper feature selection and standardization"""
    # Define treatment-specific feature sets
    feature_sets = {
        'Weather': ['Social_Media'],
        'Social_Media': ['Weather', 'Competitor'],
        'Competitor': ['Weather', 'Social_Media']
    }
    
    # Initialize results storage
    results = {}
    effects = []
    errors = []
    labels = []
    
    # Standardize continuous variables
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled['Social_Media'] = scaler.fit_transform(data[['Social_Media']])
    
    for treatment in feature_sets.keys():
        print(f"\nAnalyzing {treatment}'s effect on Sales:")
        
        # Run DML with proper features
        result = analyzer.double_ml_analysis(
            df=data_scaled,
            treatment=treatment,
            outcome='Sales',
            features=feature_sets[treatment]
        )
        
        results[treatment] = result
        
        if result.get('success', False):
            print(f"- Direct Effect: {result['ate']:.4f} (SE: {result['ate_std']:.4f})")
            print(f"- 95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            if 'nuisance_scores' in result:
                print(f"- Model Quality: Y={result['nuisance_scores'].get('y_r2', 'NA'):.4f}, "
                      f"T={result['nuisance_scores'].get('t_r2', 'NA'):.4f}")
            
            effects.append(result['ate'])
            errors.append(result['ate_std'] * 1.96)
            labels.append(treatment)
            
    # Visualization
    plt.figure(figsize=(12, 6))
    y_pos = np.arange(len(labels))
    
    plt.errorbar(effects, y_pos, xerr=errors, fmt='o')
    plt.yticks(y_pos, labels)
    plt.xlabel('Estimated Direct Effect on Sales')
    plt.title('Treatment Effects with 95% Confidence Intervals\n(Controlling for Other Factors)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Add effect size annotations
    for i, (effect, error) in enumerate(zip(effects, errors)):
        plt.annotate(f'Effect: {effect:.2f}\n(±{error:.2f})',
                    xy=(effect, i),
                    xytext=(10, 0), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run the analysis
dml_results = run_dml_analysis(data, analyzer)
```
Part D
```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Add the src directory to path for imports
src_dir = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_dir)

from data_generator import DataGenerator
from causal_analysis import CausalAnalyzer
from visualization import CausalVisualizer

# Set up our analytical tools
sns.set_style("whitegrid")

# Get our café data and analyzer ready
generator = DataGenerator(seed=42)
data = generator.generate_data(n_days=200, include_hidden=True)
analyzer = CausalAnalyzer()
```

```python
# Run transfer entropy analysis
variables = data.columns.tolist()
te_results = analyzer.transfer_entropy_analysis(data, variables=variables)

# Create a DataFrame for better viewing
te_df = pd.DataFrame([
    {'Source': edge.split(' → ')[0],
     'Target': edge.split(' → ')[1],
     'TE': value}
    for edge, value in te_results.items()
])

# Sort by strength of information flow
te_df = te_df.sort_values('TE', ascending=False)

print("Information Flow Analysis:")
print(te_df)

# Create an improved network visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
G = nx.DiGraph()

# Add nodes
for var in variables:
    G.add_node(var)

# Add edges with weights
max_te = te_df['TE'].max()
edge_colors = []
edge_widths = []

for _, row in te_df.iterrows():
    if row['TE'] > max_te * 0.1:  # Only show stronger connections
        G.add_edge(row['Source'], row['Target'], weight=row['TE'])
        # Calculate edge color based on strength (blue to red)
        color_val = row['TE'] / max_te
        edge_colors.append(plt.cm.coolwarm(color_val))
        edge_widths.append(3 * row['TE'] / max_te)

# Improve layout with more space between nodes
pos = nx.spring_layout(G, k=3, iterations=50)

# Draw edges with varying thickness and colors
nx.draw_networkx_edges(G, pos, 
                      width=edge_widths,
                      edge_color=edge_colors,
                      arrows=True,
                      arrowsize=20,
                      arrowstyle='->',
                      connectionstyle='arc3, rad=0.2')  # Curved edges

# Draw nodes with improved style
nx.draw_networkx_nodes(G, pos,
                      node_color='lightblue',
                      node_size=3000,
                      alpha=0.7,
                      edgecolors='darkblue',
                      linewidths=2)

# Improve label placement
nx.draw_networkx_labels(G, pos,
                       font_size=12,
                       font_weight='bold')

plt.title('Information Flow Network in Café Chaos', 
         fontsize=14, 
         pad=20,
         fontweight='bold')

plt.axis('off')
plt.tight_layout()
plt.show()
```