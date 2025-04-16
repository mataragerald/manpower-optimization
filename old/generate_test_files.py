import os
import pandas as pd
import numpy as np


def generate_test_excel_files(output_dir='.'):
    """
    Generate two Excel files for testing the Hungarian Assignment algorithm:
    1. A tabular format file (workers as rows, tasks as columns)
    2. A list format file (columns for workers, tasks, and costs)
    
    Args:
        output_dir (str): Directory where the Excel files will be saved
    
    Returns:
        tuple: Paths to the generated files (tabular_file_path, list_file_path)
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    tabular_file_path = os.path.join(output_dir, 'hungarian_tabular_test.xlsx')
    list_file_path = os.path.join(output_dir, 'hungarian_list_test.xlsx')
    
    # Generate tabular test data
    print("Generating tabular format test file...")
    
    # Create sample worker names and task names
    workers = [f"Worker_{i}" for i in range(1, 6)]  # 5 workers
    tasks = [f"Task_{j}" for j in range(1, 7)]      # 6 tasks (rectangular matrix)
    
    # Generate random costs between 1 and 20
    np.random.seed(42)  # For reproducibility
    cost_matrix = np.random.randint(1, 21, size=(len(workers), len(tasks)))
    
    # Create a DataFrame with the tabular structure
    df_tabular = pd.DataFrame(cost_matrix, index=workers, columns=tasks)
    
    # Reset the index to make the worker names a column
    df_tabular.reset_index(inplace=True)
    df_tabular.rename(columns={'index': 'Worker'}, inplace=True)
    
    # Save to Excel
    df_tabular.to_excel(tabular_file_path, index=False)
    print(f"Tabular format file saved to: {tabular_file_path}")
    print("Sample data:")
    print(df_tabular.head())
    
    # Generate list test data
    print("\nGenerating list format test file...")
    
    # Create a list of all worker-task combinations with costs
    data = []
    for i, worker in enumerate(workers):
        for j, task in enumerate(tasks):
            # Use the same cost matrix for consistency
            cost = cost_matrix[i, j]
            data.append({
                'Employee': worker,
                'Project': task,
                'Effort': cost
            })
    
    # Create a DataFrame from the list data
    df_list = pd.DataFrame(data)
    
    # Add some random missing values to test robustness
    # Set ~5% of the values to NaN
    random_indices = np.random.choice(len(df_list), size=int(len(df_list) * 0.05), replace=False)
    for idx in random_indices:
        column = np.random.choice(['Effort'])  # Only set Effort to NaN
        df_list.loc[idx, column] = np.nan
    
    # Save to Excel
    df_list.to_excel(list_file_path, index=False)
    print(f"List format file saved to: {list_file_path}")
    print("Sample data:")
    print(df_list.head())
    
    return tabular_file_path, list_file_path