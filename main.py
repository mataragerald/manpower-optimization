from optimiser import WeightedHungarianOptimizer
import pandas as pd

def main():
    # Initialize the optimizer
    optimizer = WeightedHungarianOptimizer()

    # Load data from the Excel file
    file_path = 'data/emplyee_task_data.xslx'
    optimizer.load_data_from_excel(file_path)

    # Set custom weights for the optimization criteria
    optimizer.set_weights({
        "availability": 0.2,
        "skill": 0.4,
        "workhours": 0.2,
        "complexity": 0.2
    })

    # Optimize assignment
    assignment, total_cost = optimizer.optimize()

    # Generate and display detailed results and visualizations
    optimizer.generate_report()
    optimizer.visualize_cost_matrix()
    optimizer.visualize_bipartite_graph()
    optimizer.visualize_cost_breakdown()

if __name__ == "__main__":
    main()