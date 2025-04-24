import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data/qualitative_analysis.csv"  # Adjust the path as needed
df = pd.read_csv(file_path)

# Filter only the selected languages
selected_languages = ["Python", "C++", "C", "Elixir", "C#"]
filtered_df = df[df["Language"].isin(selected_languages)]

# Count total number of projects per selected language
language_counts = filtered_df["Language"].value_counts().reset_index()
language_counts.columns = ["Language", "Count"]

# Plot the barplot
plt.figure(figsize=(5, 3))
sns.barplot(data=language_counts, x="Language", y="Count", palette="Greens_r")

# Formatting
# plt.title("Total Number of Projects by Selected Languages")
plt.xlabel("Language")
plt.ylabel("Number of Projects")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# Save to PDF
output_path = "graphs/qualitative/barplot_selected_languages.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.show()