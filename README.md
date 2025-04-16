# Manpower Optimization Project

## Overview
The Manpower Optimization project implements an optimal manpower assignment system using a weighted Hungarian algorithm. This project is designed to help production centers efficiently allocate workers to tasks based on various criteria such as skills, availability, work hours, and task complexity.

## Project Structure
```
manpower-optimization
├── data
│   └── emplyee_task_data.xslx         # Contains employee and task data
├── lab3.ipynb                          # Jupyter notebook demonstrating the optimizer               
├── optimiser
│   └── __init__.py                    # Contains the WeightedHungarianOptimizer class
├── main.py                            # Entry point for the application
└── README.md                          # Documentation for the project
```

## Requirements
- Python 3.x
- Required libraries:
  - pandas
  - scipy
  - numpy
  - matplotlib
  - Jupyter Notebook (for running the lab3 notebook)

## Setup
1. Clone the repository or download the project files.
2. Install the required libraries using pip:
   ```
   pip install pandas scipy numpy matplotlib jupyter
   ```
3. Ensure that the `data/emplyee_task_data.xslx` file is present, as it contains the necessary data for optimization.

## Usage
1. To run the optimization, execute the `main.py` file:
   ```
   python main.py
   ```
2. For a detailed demonstration of the optimizer, open and run the `notebooks/lab3.ipynb` Jupyter notebook.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.