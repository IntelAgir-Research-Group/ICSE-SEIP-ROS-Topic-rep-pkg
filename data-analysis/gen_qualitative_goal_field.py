import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Update path as needed
df = pd.read_csv(file_path)

# Drop rows with missing Goal or Field
df_filtered = df.dropna(subset=["Goal", "Field"]).copy()

# Split multiple goals and fields
df_filtered["Goal"] = df_filtered["Goal"].str.split(" and ")
df_filtered["Field"] = df_filtered["Field"].str.split(" and ")

# Explode both columns
df_exploded = df_filtered.explode("Goal")
df_exploded = df_exploded.explode("Field")

# Count the occurrences of each (Goal, Field) pair
goal_field_counts = df_exploded.groupby(["Goal", "Field"]).size().reset_index(name='Count')

# Pivot to create matrix format
heatmap_data = goal_field_counts.pivot(index="Goal", columns="Field", values="Count").fillna(0)

# Set up plot size and style for publication (roughly 3.5 inches wide = 9 cm)
plt.figure(figsize=(5, 3))  # inches

# Create the heatmap
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".0f",
    cmap="Greens",
    cbar_kws={"label": "Count"},
    linewidths=0.5,
    linecolor='gray'
)

# Layout adjustments for small figure
# plt.title("Goal vs Field", fontsize=10)
plt.xlabel("", fontsize=9)
plt.ylabel("", fontsize=9)
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.yticks(fontsize=9)
plt.tight_layout()

# Show or save
plt.savefig("./graphs/qualitative/heatmap_goal_field.pdf", bbox_inches="tight")  # Save for LaTeX
#plt.show()