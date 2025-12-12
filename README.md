# ECE 276C Project: RRT and RRT-CBF Motion Planning

**Authors:** Mani Amani and Pedram Aghazadeh  
**Course:** ECE 276C (Fall 2025)  
**Instructor:** Prof. Michael Yip  
**Institution:** UC San Diego, Department of Electrical and Computer Engineering

## Project Overview

This project explores motion planning algorithms, specifically comparing Rapidly-exploring Random Trees (RRT) with RRT augmented by Control Barrier Functions (RRT-CBF).

## Usage

To run the simulation, execute one of the main scripts:

```bash
python main.py       # For the single robot environment
python main_dual.py  # For the dual robot environment
```

*   **Planner Selection:** You can change the planner from RRT to RRT-CBF directly within the code.
*   **GUI:** Visualization can be enabled or disabled by modifying the GUI flag in the source code.
*   **Note:** Execution might result in collisions related to velocity constraints. These parameters may need further adjustment for optimal performance.

## Experiments & Ablation

To run multiple experiments across different random seeds and noise standard deviations:

*   Use the ablation scripts (e.g., `ablate.py` or `ablate_dual.py`) to run the experiments.
*   These scripts generate CSV files containing the raw results.

## Analysis

To extract the final results shown in the report:

```bash
python analyze_results.py
```

*   **Note:** This script uses hardcoded paths for the CSV files. Ensure these paths match your generated data location.
