# Quantum and Potential Solver Project

This project involves solving potential-related problems using quantum mechanics and numerical methods. It provides tools for various quantum simulations and solving specific types of potential profiles, including moving potentials and static potentials.

## Project Overview

- **Solver**: A solver for potential-related problems using quantum mechanics and numerical methods.
- **Profiles**: Various types of potential profiles (e.g., moving, static, hybrid) to be used in the solver.
- **Notebooks**: Jupyter Notebooks for visualizing results and running tests.
- **Testing**: Unit tests for the solver and utility functions.
- **Scripts**: Python scripts for running the solver and additional utilities.

## Directory Structure

- `config/`: Configuration files for setting parameters.
  - `params.json`: Contains configuration parameters used in the solver.

- `deepen.ipynb`: Jupyter notebook for exploring and solving problems related to the deepen potential profile.
- `diagonalization.ipynb`: Jupyter notebook for performing diagonalization of quantum states.
- `different_time_grids.ipynb`: Jupyter notebook exploring the use of different time grids in the solver.

- `notebook/`: Contains notebooks for visualization and testing.
  - `diagonalization.ipynb`: Notebook dedicated to diagonalization operations.
  
- `pytest.ini`: Configuration file for pytest settings.
- `README.md`: Project overview and documentation (this file).
- `requirements.txt`: List of required Python packages for the project.
  
- `scripts/`: Contains Python scripts for running the solver.
  - `run_solver.py`: Main script for running the quantum solver.

- `sfom_solve.ipynb`: Jupyter notebook used for solving SFOM-related problems.

- `src/`: Source code for the solver and potential profile calculations.
  - `__init__.py`: Initialization file for the `src` package.
  - `moving_potential.py`: Module for handling moving potential problems.
  - `profiles/`: Contains different potential profiles used by the solver.
    - `deepen_only.py`: Profile for solving deepen-related problems.
    - `generate.py`: Utility for generating potential profiles.
    - `hybrid_profile.py`: Hybrid potential profile.
    - `linear_profile.py`: Linear potential profile.
    - `minjerk_profile.py`: Minimum jerk potential profile.
    - `misc.py`: Miscellaneous utilities for profiles.
    - `sta_profile.py`: Static potential profile.
  - `solver.py`: Main solver for handling potential profile solutions.
  - `static_potential.py`: Module for static potential problems.
  - `utils.py`: Utility functions used throughout the project.

- `tests/`: Unit tests for validating different parts of the solver and utilities.
  - `test_fourier.py`: Tests for Fourier-related functions.
  - `test_solver.py`: Tests for the main solver functionality.
  - `test_utils.py`: Tests for utility functions.

- `test_sta.ipynb`: Notebook for testing static potential-related problems.
- `test_visualization.ipynb`: Notebook for visualizing the results of the solver and potential profiles.

## Requirements

To run the project, you need the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pytest`
- `jupyter`

Install the required packages using `pip`:

```bash
pip install -r requirements.txt

