import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

file_path = "data/qualitative_analysis.csv"
df = pd.read_csv(file_path)

df_filtered = df.dropna(subset=["Msg Type", "Goal"]).copy()

def clean_msg_types(msg):
    return [re.sub(r".*[:\.]", "", m.strip()) for m in msg.split(" and ")]

df_filtered["Msg Type"] = df_filtered["Msg Type"].apply(clean_msg_types)
df_filtered = df_filtered.explode("Msg Type")

df_filtered["Goal"] = df_filtered["Goal"].str.split(" and ")
df_filtered = df_filtered.explode("Goal")

selected_msg_types = [
    "Image", "Twist", "String",
    "JointState", "PoseStamped", "Float64",
    "PointCloud2", "Vector3Stamped", "Float32",
    "Imu", "TwistStamped", "Float64MultiArray",
    "LaserScan", "PoseWithCovarianceStamped", "Int32"
]

# Filter for selected message types
df_selected = df_filtered[df_filtered["Msg Type"].isin(selected_msg_types)]

# Group and count occurrences of (Msg Type, Goal)
msg_goal_counts = df_selected.groupby(["Msg Type", "Goal"]).size().reset_index(name="Count")

# Pivot to create heatmap data
heatmap_data = msg_goal_counts.pivot(index="Msg Type", columns="Goal", values="Count").fillna(0)

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Greens", cbar=False, cbar_kws={"label": "Count"})

# Formatting
#plt.title("Heatmap of Selected Message Types vs Goal")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/heatmap_selected_msg_types_vs_goal.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
