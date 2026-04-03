import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --------------------------------------
# Configuration
# --------------------------------------
# Automatically find the latest CSV file in the results folder
list_of_files = glob.glob('*.csv') 
if not list_of_files:
    print("No CSV files found in 'results/' folder. Please run the PSO first.")
    exit()
    
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Plotting data from: {latest_file}")

# Load Data
df = pd.read_csv(latest_file)

# Setup Academic Style
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.6,
    'grid.linestyle': '--'
})

# --------------------------------------
# Plot 1: Convergence History
# --------------------------------------
plt.figure(figsize=(10, 6))

# Group by iteration to get Mean and Std Dev across all runs
agg = df.groupby("Iteration")["Best_Fitness"].agg(['mean', 'std']).reset_index()

# Plot Mean Line
plt.plot(agg["Iteration"], agg["mean"], label="Average Best Fitness", color='#0072B2', linewidth=2)

# Plot Standard Deviation Shadow
plt.fill_between(agg["Iteration"], 
                 agg["mean"] - agg["std"], 
                 agg["mean"] + agg["std"], 
                 color='#0072B2', alpha=0.2, label="Standard Deviation")

plt.xlabel("Iteration (Voting Rounds)")
plt.ylabel("Fitness Value")
plt.title("Algorithm Convergence History")
plt.legend(loc='lower right')
plt.tight_layout()

# Save
plot_filename = latest_file.replace(".csv", "_convergence.png")
plt.savefig(plot_filename, dpi=300)
print(f"Saved: {plot_filename}")

# --------------------------------------
# Plot 2: Gate Count Box Plot (Final Iteration)
# --------------------------------------
plt.figure(figsize=(6, 6))

# Get the final row for each Run ID
max_iter = df["Iteration"].max()
final_states = df[df["Iteration"] == max_iter]

# Create Box Plot
sns.boxplot(y=final_states["Active_Gates"], color='#D55E00', width=0.4)
sns.stripplot(y=final_states["Active_Gates"], color='black', size=8, jitter=0.1) # Show actual dots

plt.ylabel("Active Gates (Lower is Better)")
plt.title(f"Gate Count Distribution (n={len(final_states)} Runs)")

# Save
box_filename = latest_file.replace(".csv", "_boxplot.png")
plt.savefig(box_filename, dpi=300)
print(f"Saved: {box_filename}")

print("\nDone! Graphs are ready for your paper.")