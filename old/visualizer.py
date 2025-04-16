import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        ax.text(task, worker, f"{cost_matrix[worker, task]}", 
                ha='center', va='center', fontweight='bold')
    
    # Add labels
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_workers))
    ax.set_xticklabels([f"Task {i}" for i in range(num_tasks)])
    ax.set_yticklabels([f"Worker {i}" for i in range(num_workers)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add a title
    total_cost = sum(cost_matrix[w, t] for w, t in assignments)
    plt.title(f"{title}\nTotal Cost: {total_cost}")
    
    # Add text annotations for all values
    for i in range(num_workers):
        for j in range(num_tasks):
            if mask[i, j] == 0:  # Only annotate non-assignment cells
                ax.text(j, i, f"{cost_matrix[i, j]}", 
                        ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.show()

def visualize_bipartite_graph(cost_matrix, assignments, workers=None, tasks=None, 
                              title="Assignment Bipartite Graph"):
    """
    Visualizes the assignment results as a bipartite graph.
    
    Args:
        cost_matrix (np.ndarray): Original cost matrix
        assignments (list): List of (worker, task) tuples representing the assignments
        workers (list, optional): List of worker names. If None, uses "Worker {i}"
        tasks (list, optional): List of task names. If None, uses "Task {j}"
        title (str): Title for the visualization
    """
    num_workers, num_tasks = cost_matrix.shape
    
    # Default labels if not provided
    if workers is None:
        workers = [f"Worker {i}" for i in range(num_workers)]
    if tasks is None:
        tasks = [f"Task {j}" for j in range(num_tasks)]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the plot dimensions
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, max(num_workers, num_tasks) + 1)
    ax.axis('off')
    
    # Calculate positions for nodes
    worker_x = 2
    task_x = 8
    
    worker_ys = np.linspace(1, num_workers, num_workers)
    task_ys = np.linspace(1, num_tasks, num_tasks)
    
    # Draw worker nodes (left side)
    worker_nodes = []
    for i, y in enumerate(worker_ys):
        circle = plt.Circle((worker_x, y), 0.4, color='skyblue', alpha=0.8)
        ax.add_patch(circle)
        worker_nodes.append((worker_x, y))
        
        # Add worker label
        label = workers[i] if i < len(workers) else f"Worker {i}"
        ax.text(worker_x - 1.2, y, label, ha='right', va='center', 
                fontsize=9, fontweight='bold')
    
    # Draw task nodes (right side)
    task_nodes = []
    for j, y in enumerate(task_ys):
        circle = plt.Circle((task_x, y), 0.4, color='lightgreen', alpha=0.8)
        ax.add_patch(circle)
        task_nodes.append((task_x, y))
        
        # Add task label
        label = tasks[j] if j < len(tasks) else f"Task {j}"
        ax.text(task_x + 1.2, y, label, ha='left', va='center', 
                fontsize=9, fontweight='bold')
    
    # Draw assigned edges
    total_cost = 0
    for worker_idx, task_idx in assignments:
        w_x, w_y = worker_nodes[worker_idx]
        t_x, t_y = task_nodes[task_idx]
        cost = cost_matrix[worker_idx, task_idx]
        total_cost += cost
        
        # Draw the edge
        line = plt.Line2D([w_x, t_x], [w_y, t_y], color='red', linewidth=2, alpha=0.8)
        ax.add_line(line)
        
        # Add the cost label
        mid_x = (w_x + t_x) / 2
        mid_y = (w_y + t_y) / 2
        
        # Add a slight vertical offset to prevent overlap of cost labels
        offset = 0.2 * ((worker_idx % 2) * 2 - 1)
        
        ax.text(mid_x, mid_y + offset, f"{cost}", 
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'),
                ha='center', va='center', fontsize=9)
    
    # Create a set of assigned workers and tasks for faster lookup
    assigned_workers = {w for w, _ in assignments}
    assigned_tasks = {t for _, t in assignments}
    
    # Draw unassigned nodes with different style if rectangular matrix
    if num_workers > num_tasks:
        for i in range(num_workers):
            if i not in assigned_workers:
                w_x, w_y = worker_nodes[i]
                circle = plt.Circle((w_x, w_y), 0.4, color='lightgray', alpha=0.5)
                ax.add_patch(circle)
                ax.text(w_x, w_y, 'X', color='red', ha='center', va='center', fontweight='bold')
    
    if num_tasks > num_workers:
        for j in range(num_tasks):
            if j not in assigned_tasks:
                t_x, t_y = task_nodes[j]
                circle = plt.Circle((t_x, t_y), 0.4, color='lightgray', alpha=0.5)
                ax.add_patch(circle)
                ax.text(t_x, t_y, 'X', color='red', ha='center', va='center', fontweight='bold')
    
    # Add a legend
    worker_patch = patches.Patch(color='skyblue', label='Workers')
    task_patch = patches.Patch(color='lightgreen', label='Tasks')
    assigned_edge = plt.Line2D([0], [0], color='red', linewidth=2, label='Assignment')
    unassigned_patch = patches.Patch(color='lightgray', label='Unassigned')
    
    ax.legend(handles=[worker_patch, task_patch, assigned_edge, unassigned_patch],
              loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    # Add title with total cost
    plt.title(f"{title}\nTotal Assignment Cost: {total_cost}")
    
    plt.tight_layout()
    plt.show()

def visualize_assignment_network(cost_matrix, assignments, workers=None, tasks=None, 
                                show_all_edges=False, title="Assignment Network"):
    """
    Visualizes the assignment as a network diagram with all possible connections.
    
    Args:
        cost_matrix (np.ndarray): Original cost matrix
        assignments (list): List of (worker, task) tuples representing the assignments
        workers (list, optional): List of worker names
        tasks (list, optional): List of task names
        show_all_edges (bool): Whether to show all possible edges (True) or only assignments (False)
        title (str): Title for the visualization
    """
    import networkx as nx
    
    num_workers, num_tasks = cost_matrix.shape
    
    # Default labels if not provided
    if workers is None:
        workers = [f"Worker {i}" for i in range(num_workers)]
    if tasks is None:
        tasks = [f"Task {j}" for j in range(num_tasks)]
    
    # Create a bipartite graph
    G = nx.Graph()
    
    # Add worker nodes
    for i in range(num_workers):
        G.add_node(f"W{i}", bipartite=0, label=workers[i])
    
    # Add task nodes
    for j in range(num_tasks):
        G.add_node(f"T{j}", bipartite=1, label=tasks[j])
    
    # Add edges for assignments
    assigned_edges = []
    for worker_idx, task_idx in assignments:
        edge = (f"W{worker_idx}", f"T{task_idx}")
        G.add_edge(*edge, weight=cost_matrix[worker_idx, task_idx], color='red', width=2.0)
        assigned_edges.append(edge)
    
    # Add all other edges if requested
    if show_all_edges:
        for i in range(num_workers):
            for j in range(num_tasks):
                edge = (f"W{i}", f"T{j}")
                if edge not in G.edges():
                    G.add_edge(*edge, weight=cost_matrix[i, j], color='gray', width=0.5)
    
    # Create layout
    pos = {}
    # Position worker nodes on the left
    for i in range(num_workers):
        pos[f"W{i}"] = (-5, (num_workers-1)/2 - i)
    # Position task nodes on the right
    for j in range(num_tasks):
        pos[f"T{j}"] = (5, (num_tasks-1)/2 - j)
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    
    # Collect edge properties for drawing
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_widths = [G[u][v]['width'] for u, v in G.edges()]
    
    # Draw nodes
    worker_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    task_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
    
    nx.draw_networkx_nodes(G, pos, nodelist=worker_nodes, node_color='skyblue', 
                          node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='lightgreen', 
                          node_size=500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    
    # Draw edge labels for assignments
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges() if (u, v) in assigned_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # Draw node labels
    worker_labels = {f"W{i}": workers[i] for i in range(num_workers)}
    task_labels = {f"T{j}": tasks[j] for j in range(num_tasks)}
    nx.draw_networkx_labels(G, pos, labels={**worker_labels, **task_labels}, font_size=9)
    
    # Calculate total cost
    total_cost = sum(cost_matrix[w, t] for w, t in assignments)
    
    plt.title(f"{title}\nTotal Cost: {total_cost}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to generate test Excel files
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
    import pandas as pd
    import os
    
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
