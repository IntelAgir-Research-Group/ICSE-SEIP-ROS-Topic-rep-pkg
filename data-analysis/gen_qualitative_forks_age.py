import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # update this path if needed
df = pd.read_csv(file_path)

# Drop rows with missing 'Created' or 'Forks'
df_filtered = df.dropna(subset=["Created", "Forks"]).copy()

# Convert 'Created' to datetime with timezone-awareness
df_filtered["Created"] = pd.to_datetime(df_filtered["Created"], utc=True)

# Get current date with timezone-awareness
today = pd.to_datetime(datetime.now().astimezone())

# Calculate project age in years
df_filtered["Age (Years)"] = (today - df_filtered["Created"]).dt.total_seconds() / (365.25 * 24 * 3600)

# Optional: use size of project (in KB) to scale bubble size
df_filtered["BubbleSize"] = df_filtered["Size(KB)"].fillna(0) / 1000  # scale down

# Set minimum bubble size if too small
df_filtered["BubbleSize"] = df_filtered["BubbleSize"].clip(lower=5)

# Plot the bubble chart
plt.figure(figsize=(5, 3))
sns.scatterplot(
    data=df_filtered,
    x="Age (Years)",
    y="Forks",
    size="BubbleSize",
    sizes=(10, 300),
    alpha=0.6,
    legend=False
)

# Labels and title
#plt.title("Project Forks vs Age")
plt.xlabel("Age (years since creation)")
plt.ylabel("Number of Forks")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/bubble_forks_vs_age.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
