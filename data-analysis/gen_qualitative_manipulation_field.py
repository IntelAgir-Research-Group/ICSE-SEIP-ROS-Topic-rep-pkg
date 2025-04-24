import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust if needed
df = pd.read_csv(file_path)

# Filter rows with Manipulation == 'YES' and valid Field
df_manip = df[(df["Manipulation"] == "YES") & (df["Field"].notna())].copy()

# Split multiple fields and explode
df_manip["Field"] = df_manip["Field"].str.split(" and ")
df_manip_exploded = df_manip.explode("Field")

# Count number of manipulation-enabled projects per field
field_counts = df_manip_exploded["Field"].value_counts().reset_index()
field_counts.columns = ["Field", "Count"]

# Plot the barplot
plt.figure(figsize=(5, 3))
sns.barplot(data=field_counts, x="Field", y="Count", palette="Greens_r")

# Formatting
#plt.title("Number of Manipulation Projects per Field")
plt.xlabel("")
plt.ylabel("Number of Projects")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/barplot_manipulation_projects_per_field.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
