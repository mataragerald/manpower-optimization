# Path to the Excel file
file_path = "employee_task_assignment1.xlsx"

# Load matrices from the Excel file
try:
    work_hours, cost, skill, availability = load_matrices_from_excel(file_path)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Define weights for the factors
weights = {"work_hours": 0, "cost": 0, "skill": 0}

# Reorder tasks by complexity (based on skill requirements)
sorted_task_indices = prioritize_tasks_by_skill(skill)
skill = skill[:, sorted_task_indices]
work_hours = work_hours[:, sorted_task_indices]
cost = cost[:, sorted_task_indices]
availability = availability[:, sorted_task_indices]

# Calculate the weighted aggregate matrix
try:
    weighted_matrix = calculate_weighted_matrix(work_hours, cost, skill, availability, weights)
except Exception as e:
    print(f"Error calculating weighted matrix: {e}")
    exit()

# Optimize task assignment
try:
    assignments, total_cost = optimize_task_assignment(weighted_matrix)
except Exception as e:
    print(f"Error during optimization: {e}")
    exit()

# Print the results
print("Optimal Assignments:")
for worker, task in assignments:
    print(f"Employee {worker + 1} -> Task {sorted_task_indices[task] + 1}")
print(f"Total Cost: {total_cost}")

# Visualize the results
visualize_cost_matrix(weighted_matrix, assignments, title="Weighted Cost Matrix")
visualize_bipartite_graph(weighted_matrix, assignments, title="Bipartite Graph of Assignments")






print("Loading data from Excel...")
    work_hours, cost, skill, availability = load_matrices_from_excel(file_path)
        
    # Store original indices for reference
    num_workers, num_tasks = skill.shape
    original_task_indices = np.arange(num_tasks)
        
    # Calculate initial weighted matrix
    weighted_matrix = calculate_weighted_matrix(
        work_hours, cost, skill, availability, weights
    )
        
    # Get optimal assignments
    assignments, total_cost = optimize_task_assignment(weighted_matrix)
        
    # Print original assignments
    print("\nOptimal Assignments (Original Order):")
    for worker, task in assignments:
        print(f"Employee {worker + 1} -> Task {task + 1}")
    print(f"Total Cost: {total_cost:.2f}")
        
    # Visualize results with consistent ordering
    print("\nGenerating visualizations...")
    # 1. Cost Matrix Heatmap
    visualize_cost_matrix(
        weighted_matrix,
        assignments,
        title="Weighted Cost Matrix"
    )
        
    # 2. Bipartite Graph
    worker_labels = [f"Employee {i+1}" for i in range(num_workers)]
    task_labels = [f"Task {i+1}" for i in range(num_tasks)]
        
    visualize_bipartite_graph(
        weighted_matrix,
        assignments,
        workers=worker_labels,
        tasks=task_labels,
        title="Assignment Bipartite Graph"
    )
    print("Done!")
        
except Exception as e:
    print(f"Error: {str(e)}")
    raise




def visualize_cost_matrix(cost_matrix, assignments, title="Cost Matrix Visualization"):
    """
    Visualizes the cost matrix as a heatmap with the optimal assignments highlighted.
    
    Args:
        cost_matrix (np.ndarray): The cost matrix
        assignments (list): List of (worker, task) tuples representing the assignments
        title (str): Title for the visualization
    """
    num_workers, num_tasks = cost_matrix.shape
    
    # Create a mask for highlighting the selected assignments
    mask = np.zeros_like(cost_matrix)
    for worker, task in assignments:
        mask[worker, task] = 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the heatmap
    im = ax.imshow(cost_matrix, cmap='YlOrRd')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Cost')
    
    # Highlight the assignments
    for worker, task in assignments:
        rect = plt.Rectangle((task-0.5, worker-0.5), 1, 1, 
                            fill=False, edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
        ax.text(task, worker, f"{cost_matrix[worker, task]:.2f}",
                ha='center', va='center', fontweight='bold')
    
    # Add labels
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_workers))
    ax.set_xticklabels([f"Task {i}" for i in range(num_tasks)])
    ax.set_yticklabels([f"Worker {i}" for i in range(num_workers)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add a title
    total_cost = sum(cost_matrix[w, t] for w, t in assignments)
    plt.title(f"{title}\nTotal Cost: {total_cost:.2f}")
    
    # Add text annotations for all values
    for i in range(num_workers):
        for j in range(num_tasks):
            if mask[i, j] == 0:  # Only annotate non-assignment cells
                ax.text(j, i, f"{cost_matrix[i, j]:.2f}",
                        ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.show()







    Args:
        cost_matrix (np.ndarray): The cost matrix
        assignments (list): List of (worker, task) tuples representing the assignments
        title (str): Title for the visualization
    """
    num_workers, num_tasks = cost_matrix.shape
    
    # Create figure and axis with larger size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the heatmap with enhanced colormap
    im = ax.imshow(cost_matrix, 
                   cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (Red=high, Blue=low)
                   aspect='auto',
                   interpolation='nearest')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Cost', size=10, weight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Create mask for assignments
    mask = np.zeros_like(cost_matrix)
    for worker, task in assignments:
        mask[worker, task] = 1
    
    # Highlight optimal assignments with enhanced visibility
    for worker, task in assignments:
        # Add rectangle highlight
        rect = plt.Rectangle((task-0.5, worker-0.5), 1, 1,
                           fill=False,
                           edgecolor='blue',
                           linewidth=3,
                           linestyle='-',
                           alpha=0.8)
        ax.add_patch(rect)
        
        # Add cost value for optimal assignments
        ax.text(task, worker, f"{cost_matrix[worker, task]:.1f}",
                ha='center',
                va='center',
                color='black',
                fontweight='bold',
                fontsize=11,
                bbox=dict(facecolor='white',
                         edgecolor='blue',
                         alpha=0.7,
                         pad=3))
    
    # Add cost values for non-assigned cells
    for i in range(num_workers):
        for j in range(num_tasks):
            if mask[i, j] == 0:
                ax.text(j, i, f"{cost_matrix[i, j]:.1f}",
                       ha='center',
                       va='center',
                       color='black',
                       fontsize=9,
                       alpha=0.7)
    
    # Improve grid appearance
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_workers))
    ax.set_xticks(np.arange(-.5, num_tasks, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_workers, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add labels with better formatting
    ax.set_xticklabels([f"Task {i+1}" for i in range(num_tasks)], 
                       rotation=45, 
                       ha="right",
                       fontsize=10)
    ax.set_yticklabels([f"Worker {i+1}" for i in range(num_workers)],
                       fontsize=10)
    
    # Calculate and display total cost
    total_cost = sum(cost_matrix[w, t] for w, t in assignments)
    plt.title(f"{title}\nTotal Assignment Cost: {total_cost:.2f}",
              pad=20,
              size=12,
              weight='bold')
    
    # Add annotations for optimal/suboptimal indicators
    ax.text(1.15, -0.2, "■ Optimal Assignments",
            transform=ax.transAxes,
            color='blue',
            fontsize=10)
    
    plt.tight_layout()
    plt.show()










    # Path to the Excel file
file_path = "employee_task_assignment1211.xlsx"

# Load matrices from the Excel file
try:
    work_hours, cost, skill, availability = load_matrices_from_excel(file_path)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Define weights for the factors
weights = {"work_hours": 10.0, "cost": 1.0, "skill": 1.0}

# Reorder tasks by complexity (based on skill requirements)
sorted_task_indices = prioritize_tasks_by_skill(skill)
skill = skill[:, sorted_task_indices]
work_hours = work_hours[:, sorted_task_indices]
cost = cost[:, sorted_task_indices]
availability = availability[:, sorted_task_indices]

# Calculate the weighted aggregate matrix
try:
    weighted_matrix = calculate_weighted_matrix(work_hours, cost, skill, availability, weights)
except Exception as e:
    print(f"Error calculating weighted matrix: {e}")
    exit()

# Optimize task assignment
try:
    assignments, total_cost = optimize_task_assignment(weighted_matrix)
except Exception as e:
    print(f"Error during optimization: {e}")
    exit()

# Print the results
print("Optimal Assignments:")
for worker, task in assignments:
    print(f"Employee {worker + 1} -> Task {sorted_task_indices[task] + 1}")
print(f"Total Cost: {total_cost}")

# Visualize the results
visualize_cost_matrix(weighted_matrix, assignments, title="Weighted Cost Matrix")
visualize_bipartite_graph(weighted_matrix, assignments, title="Bipartite Graph of Assignments")


