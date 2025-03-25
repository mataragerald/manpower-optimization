import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


class DataManager:
    """Handles loading and validating employee data."""
    
    def __init__(self):
        self.data = None
        self.required_columns = [
            'Skill_Level', 'Penalty_Cost', 'Utilization_Efficiency',
            'Fatigue_Index', 'Workload', 'Productivity', 'Efficiency',
            'Quality', 'Labor_Cost'
        ]
    
    def load_data(self, excel_file):
        """Load employee data from an Excel file."""
        try:
            self.data = pd.read_excel(excel_file)
            self._validate_data()
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{excel_file}' not found.")
        except Exception as e:
            raise Exception(f"Error loading employee data: {e}")
    
    def _validate_data(self):
        """Ensure all required columns exist in the data."""
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
