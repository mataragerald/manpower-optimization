from ortools.linear_solver import pywraplp

def calculate_assignment_cost(worker_id, task_id, worker_availability, worker_skills, task_requirements, workhours_needed, worker_workload):
    """
    Calculates the cost of assigning a worker to a task based on various factors.

    Args:
        worker_id: The ID of the worker.
        task_id: The ID of the task.
        worker_availability: A dictionary indicating worker availability (e.g., {worker_id: {day: [start_time, end_time]}}).
        worker_skills: A dictionary of worker skills (e.g., {worker_id: {'skill1': level, 'skill2': level}}).
        task_requirements: A dictionary of task skill requirements (e.g., {task_id: {'skill1': required_level, 'skill2': required_level}}).
        workhours_needed: A dictionary of workhours needed for each task (e.g., {task_id: hours}).
        worker_workload: A dictionary of current workhours assigned to each worker (e.g., {worker_id: total_hours}).

    Returns:
        The cost of the assignment. Higher cost indicates a less desirable assignment.
    """
    cost = 0

    # Worker Availability (Simplified: Assume a worker is generally available)
    if worker_id not in worker_availability:
        cost += 100  # High cost for unavailable worker

    # Skill Level Matching
    if task_id in task_requirements and worker_id in worker_skills:
        task_skills = task_requirements[task_id]
        worker_skills_for_worker = worker_skills[worker_id]
        for skill, required_level in task_skills.items():
            worker_level = worker_skills_for_worker.get(skill, 0)
            if worker_level < required_level:
                cost += (required_level - worker_level) * 20  # Cost increases with skill gap
            elif worker_level > required_level:
                cost -= (worker_level - required_level) * 5 # Slight benefit for overqualified

    elif task_id in task_requirements:
        cost += 50 # Cost if worker skills are not available

    # Workhours and Workload (Simplified: Penalize exceeding a threshold)
    if task_id in workhours_needed and worker_id in worker_workload:
        hours_needed = workhours_needed[task_id]
        current_workload = worker_workload[worker_id]
        if current_workload + hours_needed > 40: # Assuming a max 40-hour work week
            cost += (current_workload + hours_needed - 40) * 15 # Penalty for exceeding work hours

    elif task_id in workhours_needed:
        cost += 10 # Base cost if workload info is not available

    # Task Complexity (Simplified: Assume tasks have a complexity score)
    task_complexity = {'Task_0': 3, 'Task_1': 1, 'Task_2': 2, 'Task_3': 4} # Example complexity
    if task_id in task_complexity and worker_id in worker_skills:
        complexity_score = task_complexity[task_id]
        # You might want to adjust cost based on worker skill level vs task complexity
        # For example, higher complexity for lower skilled workers increases cost
        relevant_skill = list(task_requirements.get(task_id, {}).keys()) if task_requirements.get(task_id) else None
        if relevant_skill:
            worker_level = worker_skills[worker_id].get(relevant_skill, 0)
            cost += complexity_score * (5 - worker_level) # Higher complexity, lower skill = higher cost
        else:
            cost += complexity_score * 5

    elif task_id in task_complexity:
        cost += task_complexity[task_id] * 10


    return cost

def optimal_manpower_assignment_with_factors(workers, tasks, worker_availability, worker_skills, task_requirements, workhours_needed):
    """
    Solves the optimal manpower assignment problem using the OR-Tools library,
    considering worker availability, skill levels, workhours, and task complexity.

    Args:
        workers: A list of worker IDs.
        tasks: A list of task IDs.
        worker_availability: A dictionary indicating worker availability.
        worker_skills: A dictionary of worker skills.
        task_requirements: A dictionary of task skill requirements.
        workhours_needed: A dictionary of workhours needed for each task.

    Returns:
        A tuple containing:
            - assignment: A dictionary mapping worker_id to task_id in the optimal assignment.
            - min_cost: The minimum total cost of the assignment.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver unavailable.")
        return None, None

    num_workers = len(workers)
    num_tasks = len(tasks)

    # Create assignment variables
    x = {}
    for i in workers:
        for j in tasks:
            x[i, j] = solver.IntVar(0, 1, f'x[{i},{j}]')

    # Constraint: Each worker is assigned to at most one task
    for i in workers:
        solver.Add(sum(x[i, j] for j in tasks) <= 1)

    # Constraint: Each task is assigned to exactly one worker
    for j in tasks:
        solver.Add(sum(x[i, j] for i in workers) == 1)

    # Objective: Minimize the total cost
    objective_terms =
    worker_workload = {worker: 0 for worker in workers} # Keep track of workload

    for i in workers:
        for j in tasks:
            cost = calculate_assignment_cost(i, j, worker_availability, worker_skills, task_requirements, workhours_needed, worker_workload)
            if (i, j) in x:
                objective_terms.append(cost * x[i, j])

    solver.Minimize(sum(objective_terms))

    status = solver.Solve()

    assignment = {}
    min_cost = 0

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f'Minimum cost = {solver.Objective().Value()}\n')
        min_cost = solver.Objective().Value()
        for i in workers:
            for j in tasks:
                if x[i, j].SolutionValue() > 0.5:
                    cost = calculate_assignment_cost(i, j, worker_availability, worker_skills, task_requirements, workhours_needed, worker_workload)
                    print(f'Worker {i} assigned to task {j}. Cost: {cost}')
                    assignment[i] = j
                    if j in workhours_needed:
                        worker_workload[i] += workhours_needed[j]
    else:
        print('No solution found.')

    return assignment, min_cost

if __name__ == '__main__':
    # Define workers and tasks
    workers = ['Worker_0', 'Worker_1', 'Worker_2', 'Worker_3', 'Worker_4']
    tasks =

    # Define worker availability (example)
    worker_availability = {
        'Worker_0': {'Monday': ['9:00', '17:00']},
        'Worker_1': {'Tuesday': ['10:00', '18:00']},
        'Worker_2': {'Wednesday': ['8:00', '16:00']},
        'Worker_3': {'Thursday': ['9:00', '17:00']},
        'Worker_4': {'Friday': ['10:00', '16:00']},
    }

    # Define worker skills (example: skill level 1-5)
    worker_skills = {
        'Worker_0': {'Python': 4, 'Data Analysis': 3},
        'Worker_1': {'Java': 5, 'Project Management': 4},
        'Worker_2': {'Python': 3, 'Web Development': 5},
        'Worker_3': {'Data Analysis': 4, 'SQL': 4},
        'Worker_4': {'Project Management': 3, 'Communication': 5},
    }

    # Define task skill requirements (example: skill level needed)
    task_requirements = {
        'Task_0': {'Python': 3},
        'Task_1': {'Java': 4},
        'Task_2': {'Web Development': 4},
        'Task_3': {'Data Analysis': 3},
    }

    # Define workhours needed for each task (example)
    workhours_needed = {
        'Task_0': 10,
        'Task_1': 15,
        'Task_2': 12,
        'Task_3': 8,
    }

    print("Optimal Manpower Assignment with Factors using Hungarian Algorithm:")
    optimal_assignment, min_total_cost = optimal_manpower_assignment_with_factors(
        workers, tasks, worker_availability, worker_skills, task_requirements, workhours_needed
    )

    if optimal_assignment:
        print("\nOptimal Assignment:")
        for worker, task in optimal_assignment.items():
            print(f"Worker {worker} -> Task {task}")
        print(f"\nMinimum Total Cost: {min_total_cost}")