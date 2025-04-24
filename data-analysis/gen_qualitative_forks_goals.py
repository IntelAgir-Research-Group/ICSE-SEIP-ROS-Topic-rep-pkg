import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # adjust path if needed
df = pd.read_csv(file_path)

# Drop rows with missing Goal or Forks
df_filtered = df.dropna(subset=["Goal", "Forks"]).copy()

# Split 'Goal' by 'and' and explode
df_filtered["Goal"] = df_filtered["Goal"].str.split(" and ")
df_exploded = df_filtered.explode("Goal")

# Group by Goal and sum forks
goal_forks = df_exploded.groupby("Goal")["Forks"].sum().reset_index()

# Sort by fork count (optional)
goal_forks = goal_forks.sort_values("Forks", ascending=False)

# Plot barplot
plt.figure(figsize=(5, 3))
sns.barplot(
    data=goal_forks,
    x="Goal",
    y="Forks",
    palette="Greens_r"
)

# Formatting
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.ylabel("Total Forks")
plt.xlabel("")  # No x-axis label
plt.yticks(fontsize=9)
#plt.title("Total Forks per Goal")
plt.tight_layout()

# Save as PDF
output_path = "graphs/qualitative/barplot_total_forks_per_goal.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
