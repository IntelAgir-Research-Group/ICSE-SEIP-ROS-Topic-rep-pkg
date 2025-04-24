import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust the path if needed
df = pd.read_csv(file_path)

# Drop rows with missing RobotType or Field
df_filtered = df.dropna(subset=["RobotType", "Field"]).copy()

# Split multiple fields and explode
df_filtered["Field"] = df_filtered["Field"].str.split(" and ")
df_exploded = df_filtered.explode("Field")

# Count occurrences of each (Field, RobotType) combination
barplot_data = df_exploded.groupby(["Field", "RobotType"]).size().reset_index(name="Count")

# Plot the barplot
plt.figure(figsize=(5, 3))
sns.barplot(
    data=barplot_data,
    x="Field",
    y="Count",
    hue="RobotType",
    palette="Greens"
)

# Formatting
# plt.title("Robot Type Distribution by Field")
plt.xlabel("Field")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/barplot_robot_type_by_field.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
