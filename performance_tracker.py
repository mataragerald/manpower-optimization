import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

class PerformanceTracker:
    def __init__(self):
        self.metrics_df = pd.DataFrame(columns=[
            'date', 'early_completion_hours', 'defects_count', 
            'extra_outputs', 'overtime_hours'
        ])
        
    def record_daily_metrics(self):
        """Record daily performance metrics"""
        date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Get daily metrics from user input
            early_hours = float(input("Hours saved (early completion) today: "))
            defects = int(input("Number of defects today: "))
            extra_output = int(input("Additional outputs beyond target today: "))
            overtime = float(input("Overtime hours today: "))
            
            # Add to dataframe
            new_record = pd.DataFrame([{
                'date': date,
                'early_completion_hours': early_hours,
                'defects_count': defects,
                'extra_outputs': extra_output,
                'overtime_hours': overtime
            }])
            
            self.metrics_df = pd.concat([self.metrics_df, new_record], ignore_index=True)
            print("Daily metrics recorded successfully!")
            
        except ValueError as e:
            print(f"Error: Please enter valid numbers. {str(e)}")
    
    def save_to_excel(self, filename='performance_metrics.xlsx'):
        """Save metrics to Excel file"""
        self.metrics_df.to_excel(filename, index=False)
        print(f"Metrics saved to {filename}")
    
    def load_from_excel(self, filename='performance_metrics.xlsx'):
        """Load metrics from Excel file"""
        try:
            self.metrics_df = pd.read_excel(filename)
            print("Metrics loaded successfully!")
        except FileNotFoundError:
            print("No existing metrics file found. Starting fresh.")
    
    def visualize_trends(self):
        """Generate visualization of performance trends"""
        if len(self.metrics_df) < 2:
            print("Not enough data to show trends. Please record more daily metrics.")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Over Time', fontsize=16)
        
        # 1. Overtime Reduction
        sns.lineplot(data=self.metrics_df, x='date', y='overtime_hours', ax=ax1)
        ax1.set_title('Overtime Hours Trend')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # 2. Early Completion
        sns.lineplot(data=self.metrics_df, x='date', y='early_completion_hours', ax=ax2)
        ax2.set_title('Early Completion Hours Trend')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. Defects Reduction
        sns.lineplot(data=self.metrics_df, x='date', y='defects_count', ax=ax3)
        ax3.set_title('Daily Defects Trend')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # 4. Extra Outputs
        sns.lineplot(data=self.metrics_df, x='date', y='extra_outputs', ax=ax4)
        ax4.set_title('Additional Outputs Trend')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display improvement metrics
        first_week = self.metrics_df.head(7)
        last_week = self.metrics_df.tail(7)
        
        print("\nPerformance Improvements:")
        print(f"Overtime Reduction: {(first_week['overtime_hours'].mean() - last_week['overtime_hours'].mean()):.2f} hours")
        print(f"Defects Reduction: {(first_week['defects_count'].mean() - last_week['defects_count'].mean()):.2f}%")
        print(f"Output Increase: {(last_week['extra_outputs'].mean() - first_week['extra_outputs'].mean()):.2f} units")

def main():
    tracker = PerformanceTracker()
    
    while True:
        print("\nPerformance Tracking Menu:")
        print("1. Record today's metrics")
        print("2. View performance trends")
        print("3. Save metrics to Excel")
        print("4. Load metrics from Excel")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            tracker.record_daily_metrics()
        elif choice == '2':
            tracker.visualize_trends()
        elif choice == '3':
            tracker.save_to_excel()
        elif choice == '4':
            tracker.load_from_excel()
        elif choice == '5':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

