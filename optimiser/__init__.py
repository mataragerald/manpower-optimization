import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional, Union, Tuple
import pandas as pd
import datetime


class WeightedHungarianOptimizer:
    """
    Implements optimal manpower assignment using a weighted Hungarian algorithm
    that considers multiple criteria as described in the paper.
    """

    def __init__(self):
        """Initialize the optimizer without any data."""
        self.workers = []
        self.tasks = []
        self.worker_skills = {}
        self.task_requirements = {}
        self.worker_availability = {}
        self.workhours_needed = {}
        self.weights = {
            "availability": 0.25,
            "skill": 0.35,
            "workhours": 0.15,
            "complexity": 0.25,
        }
        self.task_complexity = {}
        self.worker_workload = {}
        self.cost_matrix = None
        self.assignments = None
        self.total_cost = None
        self.detailed_costs = {}

    def load_data_from_dicts(
        self,
        workers: List[str],
        tasks: List[str],
        worker_skills: Dict[str, Dict[str, int]],
        task_requirements: Dict[str, Dict[str, int]],
        worker_availability: Dict[str, Dict[str, List[str]]],
        workhours_needed: Dict[str, int],
        task_complexity: Optional[Dict[str, int]] = None,
    ):
        """
        Load data directly from dictionaries.

        Args:
            workers: List of worker IDs
            tasks: List of task IDs
            worker_skills: Dictionary mapping workers to their skills and levels
            task_requirements: Dictionary mapping tasks to required skills and levels
            worker_availability: Dictionary of worker availability by day/time
            workhours_needed: Dictionary of hours needed for each task
            task_complexity: Optional dictionary of task complexity scores (1-5)
        """
        self.workers = workers
        self.tasks = tasks
        self.worker_skills = worker_skills
        self.task_requirements = task_requirements
        self.worker_availability = worker_availability
        self.workhours_needed = workhours_needed
        self.worker_workload = {worker: 0 for worker in workers}

        # If task complexity not provided, estimate from skill requirements
        if task_complexity:
            self.task_complexity = task_complexity
        else:
            self.task_complexity = {}
            for task_id, requirements in task_requirements.items():
                # Calculate complexity as the average required skill level + number of skills required
                if requirements:
                    complexity = (
                        sum(requirements.values()) / len(requirements)
                        + len(requirements) / 3
                    )
                    # Scale to 1-5 range
                    self.task_complexity[task_id] = max(1, min(5, round(complexity)))
                else:
                    self.task_complexity[task_id] = 1

    def load_data_from_excel(self, file_path: str):
        """
        Load assignment data from Excel file with specific sheets.

        The Excel file should contain the following sheets:
        - Workers: Worker IDs
        - Tasks: Task IDs
        - WorkerSkills: Matrix of worker skills
        - TaskRequirements: Matrix of task skill requirements
        - WorkerAvailability: Worker availability (binary or hours)
        - WorkHoursNeeded: Hours needed for each task
        - TaskComplexity: (Optional) Complexity score for each task

        Args:
            file_path: Path to the Excel file
        """
        try:
            excel_data = pd.ExcelFile(file_path)

            # Load workers and tasks
            workers_df = pd.read_excel(excel_data, "Workers")
            tasks_df = pd.read_excel(excel_data, "Tasks")

            self.workers = workers_df["WorkerID"].tolist()
            self.tasks = tasks_df["TaskID"].tolist()

            # Load worker skills
            skills_df = pd.read_excel(excel_data, "WorkerSkills")
            self.worker_skills = {}

            for _, row in skills_df.iterrows():
                worker_id = row["WorkerID"]
                self.worker_skills[worker_id] = {}

                for col in skills_df.columns:
                    if col != "WorkerID" and not pd.isna(row[col]):
                        self.worker_skills[worker_id][col] = int(row[col])

            # Load task requirements
            requirements_df = pd.read_excel(excel_data, "TaskRequirements")
            self.task_requirements = {}

            for _, row in requirements_df.iterrows():
                task_id = row["TaskID"]
                self.task_requirements[task_id] = {}

                for col in requirements_df.columns:
                    if col != "TaskID" and not pd.isna(row[col]):
                        self.task_requirements[task_id][col] = int(row[col])

            # Load worker availability (simplified format)
            availability_df = pd.read_excel(excel_data, "WorkerAvailability")
            self.worker_availability = {}

            for _, row in availability_df.iterrows():
                worker_id = row["WorkerID"]
                day = row["Day"]

                if worker_id not in self.worker_availability:
                    self.worker_availability[worker_id] = {}

                self.worker_availability[worker_id][day] = [
                    row["StartTime"],
                    row["EndTime"],
                ]

            # Load workhours needed
            workhours_df = pd.read_excel(excel_data, "WorkHoursNeeded")
            self.workhours_needed = dict(
                zip(workhours_df["TaskID"], workhours_df["HoursNeeded"])
            )

            # Initialize worker workload
            self.worker_workload = {worker: 0 for worker in self.workers}

            # Load task complexity if available
            if "TaskComplexity" in excel_data.sheet_names:
                complexity_df = pd.read_excel(excel_data, "TaskComplexity")
                self.task_complexity = dict(
                    zip(complexity_df["TaskID"], complexity_df["ComplexityScore"])
                )
            else:
                # Estimate complexity from task requirements
                self.task_complexity = {}
                for task_id, requirements in self.task_requirements.items():
                    if requirements:
                        complexity = (
                            sum(requirements.values()) / len(requirements)
                            + len(requirements) / 3
                        )
                        self.task_complexity[task_id] = max(
                            1, min(5, round(complexity))
                        )
                    else:
                        self.task_complexity[task_id] = 1

            print(f"Successfully loaded data from {file_path}")

        except Exception as e:
            print(f"Error loading data from Excel: {e}")
            raise

    def set_weights(self, weights: Dict[str, float]):
        """
        Set the weights for different criteria.

        Args:
            weights: Dictionary with keys 'availability', 'skill', 'workhours', 'complexity'
                    and values as weights (should sum to 1).
        """
        # Validate weights sum to approximately 1
        if abs(sum(weights.values()) - 1.0) > 0.001:
            print(
                f"Warning: Weights sum to {sum(weights.values())}, not 1.0. Normalizing..."
            )
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = weights

    def calculate_assignment_cost(
        self, worker_id: str, task_id: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the cost of assigning a worker to a task based on multiple weighted criteria.

        Args:
            worker_id: ID of the worker
            task_id: ID of the task

        Returns:
            Tuple of (total_weighted_cost, detailed_costs_dictionary)
        """
        # Initialize costs for each criterion
        costs = {"availability": 0, "skill": 0, "workhours": 0, "complexity": 0}

        # 1. Worker Availability Cost
        if worker_id not in self.worker_availability:
            costs["availability"] = 100  # High cost for unavailable worker
        # Check if the worker has enough available hours in the required timeframe
        if task_id in self.workhours_needed:
            hours_needed = self.workhours_needed[task_id]
            available_hours = 0

            if worker_id in self.worker_availability:
                for day, times in self.worker_availability[worker_id].items():
                    start_time, end_time = times
                    available_hours += calculate_hours_between(start_time, end_time)

            if available_hours < hours_needed:
                costs["availability"] = 100  # High cost for insufficient availability
            else:
                costs["availability"] = max(
                    0, 10 - available_hours
                )  # Lower cost for more availability
        else:
            costs["availability"] = 50  # Default cost if no workhours are specified

        # 2. Skill Matching Cost
        if task_id in self.task_requirements and worker_id in self.worker_skills:
            task_skills = self.task_requirements[task_id]
            worker_skills = self.worker_skills[worker_id]
            skill_costs = []

            for skill, required_level in task_skills.items():
                worker_level = worker_skills.get(skill, 0)

                # Calculate skill gap cost
                if worker_level < required_level:
                    # Exponential penalty for skill gap
                    skill_costs.append((required_level - worker_level) ** 2 * 10)
                else:
                    # Slight benefit for being overqualified
                    skill_costs.append(max(-10, -5 * (worker_level - required_level)))

            # Average skill cost across all required skills
            costs["skill"] = sum(skill_costs) / len(skill_costs) if skill_costs else 50
        else:
            costs["skill"] = 50 if task_id in self.task_requirements else 10

        # 3. Workhours and Workload Cost
        if task_id in self.workhours_needed:
            hours_needed = self.workhours_needed[task_id]
            current_workload = self.worker_workload[worker_id]

            # Base cost proportional to hours
            costs["workhours"] = hours_needed * 1

            # Additional penalty for exceeding threshold (assumed 40-hour work week)
            if current_workload + hours_needed > 40:
                costs["workhours"] += (current_workload + hours_needed - 40) ** 1.5

            # Penalty for very uneven workload distribution
            if current_workload < 10 and hours_needed > 0:
                costs[
                    "workhours"
                ] -= 5  # Slight benefit for assigning work to less busy workers

        # 4. Task Complexity Cost
        if task_id in self.task_complexity:
            complexity_score = self.task_complexity[task_id]

            # For complex tasks, prefer skilled workers in relevant areas
            if worker_id in self.worker_skills and task_id in self.task_requirements:
                # Get the primary required skill for this task
                primary_skills = list(self.task_requirements[task_id].keys())
                if primary_skills:
                    # Calculate average worker skill level for required skills
                    avg_worker_skill = sum(
                        self.worker_skills[worker_id].get(skill, 0)
                        for skill in primary_skills
                    ) / len(primary_skills)

                    # Higher complexity cost for lower-skilled workers
                    costs["complexity"] = complexity_score * (
                        6 - min(5, avg_worker_skill)
                    )
                else:
                    costs["complexity"] = complexity_score * 3
            else:
                costs["complexity"] = complexity_score * 5

        # Calculate weighted total cost
        total_cost = sum(
            self.weights[criterion] * cost for criterion, cost in costs.items()
        )

        return total_cost, costs

    def build_cost_matrix(self) -> np.ndarray:
        """
        Build the weighted cost matrix for all worker-task pairs.

        Returns:
            NumPy array of the cost matrix
        """
        num_workers = len(self.workers)
        num_tasks = len(self.tasks)

        # Initialize cost matrix with zeros
        cost_matrix = np.zeros((num_workers, num_tasks))
        detailed_costs = {}

        # Calculate costs for each worker-task pair
        for i, worker in enumerate(self.workers):
            for j, task in enumerate(self.tasks):
                cost, details = self.calculate_assignment_cost(worker, task)
                cost_matrix[i, j] = cost
                detailed_costs[(worker, task)] = details

        # Store detailed costs for later visualization and analysis
        self.detailed_costs = detailed_costs
        self.cost_matrix = cost_matrix

        return cost_matrix

    def optimize(self) -> Tuple[Dict[str, str], float]:
        """
        Perform the optimization using the Hungarian algorithm.

        Returns:
            Tuple of (assignment_dict, total_cost)
        """
        if not self.cost_matrix:
            self.build_cost_matrix()

        # Apply Hungarian algorithm
        worker_indices, task_indices = linear_sum_assignment(self.cost_matrix)

        # Create assignment dictionary and calculate total cost
        assignment = {}
        total_cost = 0

        for i, j in zip(worker_indices, task_indices):
            worker = self.workers[i]
            task = self.tasks[j]
            assignment[worker] = task

            # Update worker workload
            if task in self.workhours_needed:
                self.worker_workload[worker] += self.workhours_needed[task]

            # Add to total cost
            total_cost += self.cost_matrix[i, j]

        self.assignments = assignment
        self.total_cost = total_cost

        return assignment, total_cost

    def visualize_cost_matrix(self, title: str = "Weighted Cost Matrix"):
        """
        Visualizes the cost matrix as a heatmap with highlights for optimal assignments.

        Args:
            title: Title for the visualization
        """
        if self.cost_matrix is None:
            print("Cost matrix not available. Run build_cost_matrix() first.")
            return

        if self.assignments is None:
            print("No assignments available. Run optimize() first.")
            return

        num_workers = len(self.workers)
        num_tasks = len(self.tasks)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create the heatmap
        im = ax.imshow(
            self.cost_matrix,
            cmap="RdYlBu_r",  # Red-Yellow-Blue reversed (Red=high, Blue=low)
            aspect="auto",
            interpolation="nearest",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Weighted Cost", size=10, weight="bold")
        cbar.ax.tick_params(labelsize=9)

        # Create mask for assignments
        assigned_pairs = {
            (self.workers.index(w), self.tasks.index(t))
            for w, t in self.assignments.items()
        }

        # Highlight optimal assignments
        for worker_idx, task_idx in assigned_pairs:
            # Add rectangle highlight
            rect = plt.Rectangle(
                (task_idx - 0.5, worker_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="blue",
                linewidth=3,
                linestyle="-",
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add cost value for optimal assignments
            cost = self.cost_matrix[worker_idx, task_idx]
            ax.text(
                task_idx,
                worker_idx,
                f"{cost:.1f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="blue", alpha=0.7, pad=3),
            )

        # Add cost values for non-assigned cells
        for i in range(num_workers):
            for j in range(num_tasks):
                if (i, j) not in assigned_pairs:
                    ax.text(
                        j,
                        i,
                        f"{self.cost_matrix[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        alpha=0.7,
                    )

        # Improve grid appearance
        ax.set_xticks(np.arange(num_tasks))
        ax.set_yticks(np.arange(num_workers))
        ax.set_xticks(np.arange(-0.5, num_tasks, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_workers, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

        # Add labels
        ax.set_xticklabels(self.tasks, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(self.workers, fontsize=10)

        # Add title with total cost
        plt.title(
            f"{title}\nTotal Assignment Cost: {self.total_cost:.2f}",
            pad=20,
            size=12,
            weight="bold",
        )

        # Add legend
        ax.text(
            1.15,
            -0.2,
            "■ Optimal Assignments",
            transform=ax.transAxes,
            color="blue",
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    def visualize_bipartite_graph(self, title: str = "Assignment Bipartite Graph"):
        """
        Visualizes the assignment as a bipartite graph.

        Args:
            title: Title for the visualization
        """
        if self.assignments is None:
            print("No assignments available. Run optimize() first.")
            return

        num_workers = len(self.workers)
        num_tasks = len(self.tasks)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up the plot dimensions
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, max(num_workers, num_tasks) + 1)
        ax.axis("off")

        # Calculate positions for nodes
        worker_x = 2
        task_x = 8

        worker_ys = np.linspace(1, num_workers, num_workers)
        task_ys = np.linspace(1, num_tasks, num_tasks)

        # Draw worker nodes (left side)
        worker_nodes = {}
        for i, worker in enumerate(self.workers):
            y = worker_ys[i]
            circle = plt.Circle((worker_x, y), 0.4, color="skyblue", alpha=0.8)
            ax.add_patch(circle)
            worker_nodes[worker] = (worker_x, y)

            # Add worker label
            ax.text(
                worker_x - 1.2,
                y,
                worker,
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            # Add worker skills as smaller text
            skills_text = ", ".join(
                [f"{s}:{l}" for s, l in self.worker_skills.get(worker, {}).items()]
            )
            if skills_text:
                ax.text(
                    worker_x - 1.2,
                    y - 0.3,
                    skills_text,
                    ha="right",
                    va="center",
                    fontsize=7,
                    fontstyle="italic",
                    alpha=0.7,
                )

        # Draw task nodes (right side)
        task_nodes = {}
        for j, task in enumerate(self.tasks):
            y = task_ys[j]
            circle = plt.Circle((task_x, y), 0.4, color="lightgreen", alpha=0.8)
            ax.add_patch(circle)
            task_nodes[task] = (task_x, y)

            # Add task label
            ax.text(
                task_x + 1.2,
                y,
                task,
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            # Add skill requirements as smaller text
            requirements_text = ", ".join(
                [f"{s}:{l}" for s, l in self.task_requirements.get(task, {}).items()]
            )
            if requirements_text:
                ax.text(
                    task_x + 1.2,
                    y - 0.3,
                    requirements_text,
                    ha="left",
                    va="center",
                    fontsize=7,
                    fontstyle="italic",
                    alpha=0.7,
                )

            # Add hours needed
            if task in self.workhours_needed:
                ax.text(
                    task_x + 1.2,
                    y + 0.3,
                    f"{self.workhours_needed[task]} hrs",
                    ha="left",
                    va="center",
                    fontsize=7,
                    alpha=0.7,
                )

        # Draw assigned edges
        for worker, task in self.assignments.items():
            w_x, w_y = worker_nodes[worker]
            t_x, t_y = task_nodes[task]

            # Get cost details
            worker_idx = self.workers.index(worker)
            task_idx = self.tasks.index(task)
            cost = self.cost_matrix[worker_idx, task_idx]
            detailed_costs = self.detailed_costs.get((worker, task), {})

            # Draw the edge
            line = plt.Line2D(
                [w_x, t_x], [w_y, t_y], color="red", linewidth=2, alpha=0.8
            )
            ax.add_line(line)

            # Add the cost label
            mid_x = (w_x + t_x) / 2
            mid_y = (w_y + t_y) / 2

            # Add a slight vertical offset to prevent overlap of cost labels
            offset = 0.2 * ((worker_idx % 2) * 2 - 1)

            # Add cost breakdown as a multi-line label
            cost_text = f"Total: {cost:.1f}"
            ax.text(
                mid_x,
                mid_y + offset,
                cost_text,
                bbox=dict(
                    facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"
                ),
                ha="center",
                va="center",
                fontsize=9,
            )

        # Add a legend
        worker_patch = patches.Patch(color="skyblue", label="Workers")
        task_patch = patches.Patch(color="lightgreen", label="Tasks")
        assigned_edge = plt.Line2D(
            [0], [0], color="red", linewidth=2, label="Assignment"
        )

        ax.legend(
            handles=[worker_patch, task_patch, assigned_edge],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
        )

        # Display current workload for assigned workers
        for worker, hours in self.worker_workload.items():
            if hours > 0:  # Only display for workers with assignments
                w_x, w_y = worker_nodes[worker]
                ax.text(
                    w_x,
                    w_y + 0.6,
                    f"Load: {hours}h",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(facecolor="lightyellow", alpha=0.5),
                )

        # Add title with total cost
        plt.title(f"{title}\nTotal Assignment Cost: {self.total_cost:.2f}")

        plt.tight_layout()
        plt.show()

    def visualize_cost_breakdown(self):
        """
        Visualize the breakdown of cost components for each assignment.
        """
        if not self.assignments or not self.detailed_costs:
            print("No assignments or detailed costs available.")
            return

        # Extract data for visualization
        workers = []
        tasks = []
        availability_costs = []
        skill_costs = []
        workhours_costs = []
        complexity_costs = []
        total_costs = []

        for worker, task in self.assignments.items():
            workers.append(worker)
            tasks.append(task)
            details = self.detailed_costs.get((worker, task), {})

            availability_costs.append(
                details.get("availability", 0) * self.weights["availability"]
            )
            skill_costs.append(details.get("skill", 0) * self.weights["skill"])
            workhours_costs.append(
                details.get("workhours", 0) * self.weights["workhours"]
            )
            complexity_costs.append(
                details.get("complexity", 0) * self.weights["complexity"]
            )

            worker_idx = self.workers.index(worker)
            task_idx = self.tasks.index(task)
            total_costs.append(self.cost_matrix[worker_idx, task_idx])

        # Sort by total cost
        indices = np.argsort(total_costs)
        workers = [workers[i] for i in indices]
        tasks = [tasks[i] for i in indices]
        availability_costs = [availability_costs[i] for i in indices]
        skill_costs = [skill_costs[i] for i in indices]
        workhours_costs = [workhours_costs[i] for i in indices]
        complexity_costs = [complexity_costs[i] for i in indices]
        total_costs = [total_costs[i] for i in indices]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create the bar chart
        bar_width = 0.6
        x = np.arange(len(workers))

        # Stacked bars for cost components
        p1 = ax.bar(x, availability_costs, bar_width, label="Availability")
        p2 = ax.bar(
            x, skill_costs, bar_width, bottom=availability_costs, label="Skills"
        )

        bottom = np.array(availability_costs) + np.array(skill_costs)
        p3 = ax.bar(x, workhours_costs, bar_width, bottom=bottom, label="Workhours")

        bottom += np.array(workhours_costs)
        p4 = ax.bar(x, complexity_costs, bar_width, bottom=bottom, label="Complexity")

        # Add labels and legend
        ax.set_xlabel("Worker-Task Assignments")
        ax.set_ylabel("Weighted Cost")
        ax.set_title("Cost Breakdown of Optimal Assignments")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{w} → {t}" for w, t in zip(workers, tasks)], rotation=45, ha="right"
        )
        ax.legend()

        # Add total cost labels on top of bars
        for i, v in enumerate(total_costs):
            ax.text(
                i,
                sum(
                    [
                        availability_costs[i],
                        skill_costs[i],
                        workhours_costs[i],
                        complexity_costs[i],
                    ]
                )
                + 0.3,
                f"{v:.1f}",
                ha="center",
            )

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """
        Generate a comprehensive report of the assignment solution.
        """
        if not self.assignments:
            print("No assignments available. Run optimize() first.")
            return

        print("=" * 60)
        print("OPTIMAL MANPOWER ASSIGNMENT REPORT")
        print("=" * 60)
        print(f"Total Assignment Cost: {self.total_cost:.2f}")
        print("-" * 60)
        print("OPTIMAL ASSIGNMENTS:")
        print("-" * 60)

        for worker, task in self.assignments.items():
            worker_idx = self.workers.index(worker)
            task_idx = self.tasks.index(task)
            cost = self.cost_matrix[worker_idx, task_idx]
            hours = self.workhours_needed.get(task, "N/A")

            print(f"Worker: {worker}")
            print(f"  → Task: {task}")
            print(f"  → Cost: {cost:.2f}")
            print(f"  → Hours: {hours}")

            # Print skill matching
            if task in self.task_requirements and worker in self.worker_skills:
                print("  → Skill Matching:")
                for skill, required in self.task_requirements[task].items():
                    worker_level = self.worker_skills[worker].get(skill, 0)
                    match = "MATCH" if worker_level >= required else "GAP"
                    print(
                        f"     - {skill}: Required={required}, Worker={worker_level} ({match})"
                    )

            # Print detailed cost breakdown
            if (worker, task) in self.detailed_costs:
                details = self.detailed_costs[(worker, task)]
                print("  → Cost Breakdown:")
                for criterion, weight in self.weights.items():
                    raw_cost = details.get(criterion, 0)
                    weighted_cost = raw_cost * weight
                    print(
                        f"     - {criterion.capitalize()}: {raw_cost:.2f} × {weight:.2f} = {weighted_cost:.2f}"
                    )
            print("-" * 40)

        # Worker workload summary
        print("\nWORKLOAD SUMMARY:")
        print("-" * 60)
        for worker, hours in self.worker_workload.items():
            if worker in self.assignments:  # Only show assigned workers
                print(f"{worker}: {hours} hours")

        # Unassigned workers
        unassigned_workers = set(self.workers) - set(self.assignments.keys())
        if unassigned_workers:
            print("\nUNASSIGNED WORKERS:")
            print("-" * 60)
            for worker in unassigned_workers:
                print(f"- {worker}")

        print("=" * 60)


def parse_time_value(time_value: str) -> datetime.datetime:
    """
    Parse a time value into a datetime object.

    Args:
        time_value: str

    Returns:
        A datetime object representing the time
    """
    base_date = datetime.datetime.today().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    try:
        hours, minutes = time_value.split(":")
        hours = int(hours)
        minutes = int(minutes)
        return base_date + datetime.timedelta(hours=hours, minutes=minutes)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse time value '{time_value}' as HH:MM format")

    # Default fallback
    print(f"Warning: Could not parse time value {time_value}, using 00:00")
    return base_date


def calculate_hours_between(start_time: str, end_time: str) -> float:
    """
    Calculate the number of hours between two time values.

    Args:
        start_time: Start time value (string, float, or int)
        end_time: End time value (string, float, or int)

    Returns:
        Number of hours as a float
    """
    start_dt = parse_time_value(start_time)
    end_dt = parse_time_value(end_time)

    # Handle when end_time is earlier than start_time (e.g., overnight shift)
    if end_dt < start_dt:
        end_dt += datetime.timedelta(days=1)

    time_diff = end_dt - start_dt
    # Convert to hours as float
    hours = time_diff.total_seconds() / 3600
    return max(0, hours)
