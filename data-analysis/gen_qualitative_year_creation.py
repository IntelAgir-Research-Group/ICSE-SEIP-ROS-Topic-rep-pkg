import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust the path if needed
df = pd.read_csv(file_path)

# Convert 'Created' to datetime
df["Created"] = pd.to_datetime(df["Created"], errors='coerce')

# Drop rows where 'Created' is missing
df_created = df.dropna(subset=["Created"]).copy()

# Extract the year of creation
df_created["Year"] = df_created["Created"].dt.year

# Count number of projects per year
year_counts = df_created["Year"].value_counts().sort_index()

# Plot line graph
plt.figure(figsize=(5, 3))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")

# Formatting
#plt.title("Number of Projects Created per Year")
plt.xlabel("Year")
plt.ylabel("Number of Projects")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(False)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/linegraph_projects_per_year.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
