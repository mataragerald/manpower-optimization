import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def solve_hungarian_assignment(cost_matrix):
    """
    Solves the assignment problem (Hungarian Algorithm) to find the
    minimum cost matching between workers and tasks.

    Args:
        cost_matrix (np.ndarray or list of lists): A 2D array where
            cost_matrix[i, j] is the cost of assigning worker i to task j.
            - If square (n x n): Finds the perfect matching minimizing cost.
            - If rectangular (n x m): Finds the best assignment of min(n, m)
              pairs. If n > m (more workers), some workers are unassigned.
              If m > n (more tasks), some tasks are unassigned.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples representing the optimal assignments,
                    e.g., [(worker_index, task_index), ...].
            - float: The total minimum cost of the optimal assignment.
            - list: Indices of workers included in the assignment.
            - list: Indices of tasks included in the assignment.

    Raises:
        ValueError: If the cost matrix is not valid (e.g., not 2D).
    """
    cost_matrix = np.asarray(cost_matrix) # Ensure it's a NumPy array

    if cost_matrix.ndim != 2:
        raise ValueError("Cost matrix must be 2-dimensional.")

    if cost_matrix.size == 0:
        print("Warning: Empty cost matrix provided.")
        return [], 0.0, [], []

    num_workers, num_tasks = cost_matrix.shape
    print(f"Input Cost Matrix ({num_workers} workers x {num_tasks} tasks):\n{cost_matrix}\n")

    # Use scipy's linear_sum_assignment which implements the Hungarian algorithm
    # It returns optimal row indices and corresponding column indices.
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        # Catch potential issues like non-finite values if not handled prior
        print(f"Error during assignment: {e}", file=sys.stderr)
        print("Ensure cost matrix contains only finite numbers.", file=sys.stderr)
        raise # Re-raise the error after printing info


    # Extract the assignments and calculate the total cost
    assignments = list(zip(row_ind, col_ind))
    total_cost = cost_matrix[row_ind, col_ind].sum()

    print("Optimal Assignments (Worker Index -> Task Index):")
    assigned_workers = set()
    assigned_tasks = set()
    for r, c in assignments:
        print(f"  Worker {r} -> Task {c} (Cost: {cost_matrix[r, c]})")
        assigned_workers.add(r)
        assigned_tasks.add(c)

    print(f"\nTotal Minimum Cost: {total_cost}")

    # Identify unassigned if rectangular
    if num_workers > num_tasks:
        unassigned_workers = set(range(num_workers)) - assigned_workers
        if unassigned_workers:
            print(f"Unassigned Workers: {sorted(list(unassigned_workers))}")
    elif num_tasks > num_workers:
        unassigned_tasks = set(range(num_tasks)) - assigned_tasks
        if unassigned_tasks:
            print(f"Unassigned Tasks: {sorted(list(unassigned_tasks))}")

    return assignments, total_cost, sorted(list(assigned_workers)), sorted(list(assigned_tasks))


def load_cost_matrix_from_excel(file_path, sheet_name=0, worker_column=None, 
                               task_column=None, cost_column=None, 
                               default_cost=np.inf, verbose=True):
    """
    Loads a cost matrix from an Excel file.
    
    The function can handle two Excel formats:
    1. Tabular form where rows are workers, columns are tasks, and cells contain costs.
    2. List form with separate columns for workers, tasks, and costs.
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str or int): Name or index of the sheet to read (default: 0)
        worker_column (str, optional): Column name containing worker identifiers
            Only needed for list-form data
        task_column (str, optional): Column name containing task identifiers
            Only needed for list-form data
        cost_column (str, optional): Column name containing cost values
            Only needed for list-form data
        default_cost (float): Value to use for undefined worker-task pairs (default: infinity)
        verbose (bool): Whether to print information about the loaded data
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The cost matrix where cost_matrix[i, j] is the cost of
              assigning worker i to task j
            - list: List of worker identifiers
            - list: List of task identifiers
    
    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If the Excel format is not recognized or required parameters are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Try to read the file
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    if verbose:
        print(f"Successfully loaded Excel file from {file_path}")
        print(f"Sheet: {sheet_name}")
        print(f"DataFrame shape: {df.shape}")
    
    # Determine the Excel format and process accordingly
    if worker_column and task_column and cost_column:
        # List format: separate columns for workers, tasks, and costs
        if verbose:
            print("Processing in list format with specified columns")
        
        # Extract unique workers and tasks, preserving order of appearance
        workers = df[worker_column].unique().tolist()
        tasks = df[task_column].unique().tolist()
        
        n_workers = len(workers)
        n_tasks = len(tasks)
        
        # Create worker and task index mappings
        worker_indices = {w: i for i, w in enumerate(workers)}
        task_indices = {t: i for i, t in enumerate(tasks)}
        
        # Initialize cost matrix with default cost
        cost_matrix = np.full((n_workers, n_tasks), default_cost)
        
        # Fill in the cost matrix based on provided data
        for _, row in df.iterrows():
            worker = row[worker_column]
            task = row[task_column]
            cost = row[cost_column]
            
            # Skip if any value is NaN
            if pd.isna(worker) or pd.isna(task) or pd.isna(cost):
                continue
                
            worker_idx = worker_indices[worker]
            task_idx = task_indices[task]
            cost_matrix[worker_idx, task_idx] = cost
    
    else:
        # Tabular format: rows are workers, columns are tasks
        if verbose:
            print("Processing in tabular format")
        
        # First column is assumed to contain worker identifiers
        if df.shape[1] < 2:
            raise ValueError("Excel file must have at least 2 columns for tabular format")
            
        # Extract workers and tasks
        workers = df.iloc[:, 0].tolist()
        tasks = df.columns[1:].tolist()
        
        # Extract the cost matrix (all cells excluding the first column)
        cost_matrix = df.iloc[:, 1:].values
        
        # Replace any NaN values with the default cost
        cost_matrix = np.where(np.isnan(cost_matrix), default_cost, cost_matrix)
    
    if verbose:
        print(f"Created cost matrix with shape: {cost_matrix.shape}")
        print(f"Number of workers: {len(workers)}")
        print(f"Number of tasks: {len(tasks)}")
    
    return cost_matrix, workers, tasks


def load_and_solve_from_excel(file_path, sheet_name=0, worker_column=None, 
                             task_column=None, cost_column=None, 
                             default_cost=np.inf, verbose=True):
    """
    Loads data from Excel, creates a cost matrix, and solves the assignment problem.
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str or int): Name or index of the sheet to read
        worker_column (str, optional): Column name containing worker identifiers
        task_column (str, optional): Column name containing task identifiers
        cost_column (str, optional): Column name containing cost values
        default_cost (float): Value to use for undefined worker-task pairs
        verbose (bool): Whether to print information about the process
    
    Returns:
        tuple: A tuple containing:
            - list: A list of tuples representing the optimal assignments
            - float: The total minimum cost of the optimal assignment
            - list: Workers included in the assignment (original identifiers)
            - list: Tasks included in the assignment (original identifiers)
            - np.ndarray: The cost matrix that was used
    """
    # Load the data from Excel
    cost_matrix, workers, tasks = load_cost_matrix_from_excel(
        file_path, sheet_name, worker_column, task_column, cost_column, 
        default_cost, verbose
    )
    
    # Solve the assignment problem
    assignments, total_cost, assigned_worker_indices, assigned_task_indices = solve_hungarian_assignment(cost_matrix)
    
    # Map indices back to original worker and task identifiers
    assigned_workers = [workers[i] for i in assigned_worker_indices]
    assigned_tasks = [tasks[i] for i in assigned_task_indices]
    
    # Create human-readable assignments
    readable_assignments = []
    for worker_idx, task_idx in assignments:
        readable_assignments.append({
            'worker': workers[worker_idx],
            'task': tasks[task_idx],
            'cost': cost_matrix[worker_idx, task_idx]
        })
    
    if verbose:
        print("\nHuman-readable assignments:")
        for assignment in readable_assignments:
            print(f"  {assignment['worker']} -> {assignment['task']} (Cost: {assignment['cost']})")
    
    return assignments, total_cost, assigned_workers, assigned_tasks, cost_matrix
