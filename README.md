# CLE-SH: Comprehensive Literal Explanation Package for SHapley Values by Statistical Validity
**CLE-SH** is a Python library designed to simplify the interpretation of SHAP values through statistical validation. By integrating feature selection, univariate analysis, and interaction analysis into a unified automated pipeline, it bridges the gap between model explainability and statistical rigor.

[![Paper](https://img.shields.io/badge/Paper-10.1109/ACCESS.2026.3654890-blue)](https://ieeexplore.ieee.org/document/11355484)

## Updates
- JAN 28, 2026: 1.0.0 release
- JAN 16, 2026: CLE-SH is published on IEEE ACCESS 

## Installation

```bash
pip install clesh
```
## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- shap >= 0.40.0
- fpdf2 >= 2.7.0

## Quick Start

```python
import pandas as pd
import numpy as np
from clesh import Explainer, CLEConfig

# Load your data
X = pd.read_csv("features.csv")
shap_values = np.load("shap_values.npy")

# Initialize with default configuration
explainer = Explainer(X=X, shap_values=shap_values)

# Perform complete analysis
results = explainer.analyze()

# Save plots
explainer.save_plots("./output")

# Generate PDF report
explainer.generate_report("./report.pdf", label="My Dataset")

# Manual analysis on specific features
univariate_result = explainer.univariate("feature_name")
interaction_result = explainer.inter("target_feature", "interaction_feature")
```


## Configuration

All analysis parameters are configured using the `CLEConfig` dataclass:

```python
from clesh import CLEConfig

config = CLEConfig(
    cont_bound=10,              # Threshold for determining continuous features
    candidate_num_min=10,        # Minimum number of features to select
    candidate_num_max=20,        # Maximum number of features to select
    p_feature_selection=0.05,    # P-value threshold for feature selection
    manual_num=0,               # Manual override (0 = automatic)
    p_univariate=0.05,          # P-value threshold for univariate analysis
    p_interaction=0.05          # P-value threshold for interaction analysis
)
```

All parameters have sensible defaults.

## API Reference

### Main Classes

#### `Analyzer`
Core analysis engine that performs all statistical calculations.

```python
from clesh import Analyzer

analyzer = Analyzer(X, shap_values, config)
results = analyzer.analyze()

# Manual analysis
univariate = analyzer.univariate("feature_name")
interaction = analyzer.inter("target_feat", "interaction_feat")
```

#### `Explainer`
High-level coordinator
```python
explainer = Explainer(X=X, shap_values=shap_values, config=config)
results = explainer.analyze()
explainer.save_plots("./output")
explainer.generate_report("./report.pdf", label="Dataset")
```


#### `AnalysisResults`
Dataclass containing all analysis results:
- `feature_importance`: DataFrame with feature rankings
- `selected_features`: List of selected feature names
- `feature_types`: Dict mapping feature names to types
- `univariate_results`: Dict of univariate analysis results
- `interaction_results`: Dict of interaction analysis results
- `best_functions`: Dict of best fitting functions for continuous features

### Visualization Module

All plotting functions are in `clesh.visuals` and return figure/axis objects:

```python
from clesh.visuals import (
    plot_shap_summary,
    plot_discrete_univariate,
    plot_continuous_univariate,
    plot_discrete_interaction,
    plot_continuous_interaction,
)

fig, ax = plot_shap_summary(shap_values, X)
fig.savefig("summary.png")
```

## Advanced Usage

### Custom Feature Analysis

```python
from clesh import Analyzer, CLEConfig

analyzer = Analyzer(X, shap_values)

# Analyze specific feature
result = analyzer.univariate("age")
print(f"Feature type: {result['feature_type']}")
print(f"Statistics: {result['statistics']}")

# Interaction analysis
inter_result = analyzer.inter("target_feature", "interaction_feature")
print(f"Significant interaction: {inter_result['statistics']['significant']}")
```

### Programmatic Report Generation

```python
from clesh import Explainer
from clesh.report import generate_pdf_report

explainer = Explainer(X=X, shap_values=shap_values)
results = explainer.analyze()

# Generate PDF directly
generate_pdf_report(results, "./report.pdf", "My Dataset", config)
```

## Output Structure

Results are stored in-memory as `AnalysisResults` dataclass. Plots and reports are saved to specified directories:

```
output/
├── shap_summary_plot.jpg
├── univariate_analysis/
│   └── *.jpg
├── interaction_analysis/
│   └── *.jpg
└── clesh_report.pdf
```

## Citation

If you use CLE-SH or ideas from the package in your research, please cite our paper:

```bibtex
@ARTICLE{kim2026cle-sh,
author={Kyungjin Kim and Youngro Lee and Jongmo Seo},
journal={IEEE Access},
title={CLE-SH: Comprehensive Literal Explanation Package for SHapley Values by Statistical Validity},
year={2026},
volume={14},
number={},
pages={12514-12525},
doi={10.1109/ACCESS.2026.3654890}}
```

## Future Plans
The project is currently undergoing a refactoring process to enhance the literal explanation engine and address minor bugs.