import pandas as pd
import random

# Define real people names for 12 workers
real_names = [
    "Alice", "Bob", "Charlie", "David", "Emma", 
    "Fiona", "George", "Hannah", "Ian", "Julia", 
    "Kevin", "Laura"
]

# Generate WorkerIDs based on real names
worker_ids = [f"Worker_{i}" for i in range(len(real_names))]
workers_df = pd.DataFrame({
    "WorkerID": worker_ids,
    "Name": real_names
})

# Define 20 tasks (using generic task IDs)
task_ids = [f"Task_{i}" for i in range(20)]
tasks_df = pd.DataFrame({"TaskID": task_ids})

# Define WorkerSkills for each worker.
# Skills: Assembly, Quality Control, Maintenance, Packaging, Sorting
worker_skills_data = [
    {"WorkerID": "Worker_0", "Assembly": 4, "Quality Control": 3, "Maintenance": 2, "Packaging": 4, "Sorting": 3},
    {"WorkerID": "Worker_1", "Assembly": 3, "Quality Control": 4, "Maintenance": 3, "Packaging": 3, "Sorting": 4},
    {"WorkerID": "Worker_2", "Assembly": 5, "Quality Control": 3, "Maintenance": 4, "Packaging": 2, "Sorting": 3},
    {"WorkerID": "Worker_3", "Assembly": 3, "Quality Control": 5, "Maintenance": 2, "Packaging": 4, "Sorting": 4},
    {"WorkerID": "Worker_4", "Assembly": 4, "Quality Control": 4, "Maintenance": 3, "Packaging": 5, "Sorting": 2},
    {"WorkerID": "Worker_5", "Assembly": 2, "Quality Control": 3, "Maintenance": 5, "Packaging": 3, "Sorting": 4},
    {"WorkerID": "Worker_6", "Assembly": 3, "Quality Control": 2, "Maintenance": 4, "Packaging": 3, "Sorting": 5},
    {"WorkerID": "Worker_7", "Assembly": 4, "Quality Control": 4, "Maintenance": 3, "Packaging": 3, "Sorting": 3},
    {"WorkerID": "Worker_8", "Assembly": 5, "Quality Control": 2, "Maintenance": 4, "Packaging": 4, "Sorting": 3},
    {"WorkerID": "Worker_9", "Assembly": 3, "Quality Control": 5, "Maintenance": 3, "Packaging": 2, "Sorting": 4},
    {"WorkerID": "Worker_10", "Assembly": 4, "Quality Control": 3, "Maintenance": 5, "Packaging": 3, "Sorting": 2},
    {"WorkerID": "Worker_11", "Assembly": 3, "Quality Control": 4, "Maintenance": 4, "Packaging": 5, "Sorting": 3},
]
worker_skills_df = pd.DataFrame(worker_skills_data)

# Define TaskRequirements for each of the 20 tasks (each task might require one or two key skills)
task_requirements_data = [
    {"TaskID": "Task_0", "Assembly": 3},
    {"TaskID": "Task_1", "Quality Control": 4},
    {"TaskID": "Task_2", "Maintenance": 3},
    {"TaskID": "Task_3", "Packaging": 4},
    {"TaskID": "Task_4", "Sorting": 3},
    {"TaskID": "Task_5", "Assembly": 4, "Packaging": 3},
    {"TaskID": "Task_6", "Maintenance": 4},
    {"TaskID": "Task_7", "Quality Control": 3, "Sorting": 4},
    {"TaskID": "Task_8", "Assembly": 3},
    {"TaskID": "Task_9", "Packaging": 5},
    {"TaskID": "Task_10", "Quality Control": 4},
    {"TaskID": "Task_11", "Maintenance": 3, "Assembly": 3},
    {"TaskID": "Task_12", "Sorting": 4},
    {"TaskID": "Task_13", "Packaging": 3, "Quality Control": 3},
    {"TaskID": "Task_14", "Assembly": 4},
    {"TaskID": "Task_15", "Maintenance": 4},
    {"TaskID": "Task_16", "Sorting": 3, "Packaging": 3},
    {"TaskID": "Task_17", "Quality Control": 5},
    {"TaskID": "Task_18", "Assembly": 3, "Maintenance": 3},
    {"TaskID": "Task_19", "Sorting": 4},
]
task_requirements_df = pd.DataFrame(task_requirements_data)

# Define WorkerAvailability.
# Each worker is available one day/week with shift times from 8:00 to 16:00.
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
availability_data = []
for i, wid in enumerate(worker_ids):
    day = days[i % len(days)]
    availability_data.append({"WorkerID": wid, "Day": day, "StartTime": "8:00", "EndTime": "16:00"})
worker_availability_df = pd.DataFrame(availability_data)

# Define WorkHoursNeeded for each task (typical shift durations between 7 and 10 hours)
random.seed(42)
workhours_data = [{"TaskID": tid, "HoursNeeded": random.choice([7, 8, 9, 10])} for tid in task_ids]
workhours_df = pd.DataFrame(workhours_data)

# Define TaskComplexity for each task (scores between 3 and 5)
task_complexity_data = [{"TaskID": tid, "ComplexityScore": random.choice([3, 4, 5])} for tid in task_ids]
task_complexity_df = pd.DataFrame(task_complexity_data)

# Write all DataFrames to an Excel workbook
excel_file_path = "production_center_data.xlsx"
with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
    workers_df.to_excel(writer, sheet_name="Workers", index=False)
    tasks_df.to_excel(writer, sheet_name="Tasks", index=False)
    worker_skills_df.to_excel(writer, sheet_name="WorkerSkills", index=False)
    task_requirements_df.to_excel(writer, sheet_name="TaskRequirements", index=False)
    worker_availability_df.to_excel(writer, sheet_name="WorkerAvailability", index=False)
    workhours_df.to_excel(writer, sheet_name="WorkHoursNeeded", index=False)
    task_complexity_df.to_excel(writer, sheet_name="TaskComplexity", index=False)

print(f"Excel workbook '{excel_file_path}' generated successfully.")