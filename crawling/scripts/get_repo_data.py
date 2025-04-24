import os
import subprocess
import re

PROJECTS_FILE = "./resources/repo-urls.csv"
RESULTS_FILE = "./resources/repos-pubsub.txt"
PATTERNS = [r"create_publisher", r"create_subscription"]
CLONE_DIR = "repos/"

# Create directory if it doesn't exist
os.makedirs(CLONE_DIR, exist_ok=True)

# Compile patterns for better performance in the search
compiled_patterns = [re.compile(pattern) for pattern in PATTERNS]

def clone_repo(repo_url):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(CLONE_DIR, repo_name)
    if not os.path.exists(repo_path):
        try:
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print(f"Cloned {repo_url}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_url}: {e}")
            return None
    return repo_path

def search_patterns(repo_path, patterns):
    matches = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                with open(file_path, "r", errors="ignore") as f:
                    lines = f.readlines()  # Read all lines at once
                    for i, line in enumerate(lines):
                        for pattern in patterns:
                            if pattern.search(line):
                                # Capture the current line and the next one (if it exists)
                                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                                matches.append((file_path, i + 1, line.strip(), next_line.strip()))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return matches

def process_repos():
    with open(PROJECTS_FILE, "r") as f, open(RESULTS_FILE, "w") as results_file:
        for repo_url in f:
            repo_url = repo_url.strip()
            if not repo_url:
                continue

            print(f"Processing {repo_url}...")
            repo_path = clone_repo(repo_url)
            if repo_path:
                matches = search_patterns(repo_path, compiled_patterns)
                if matches:
                    results_file.write(f"Results for {repo_url}:\n")
                    for match in matches:
                        results_file.write(f"  {match[0]}:{match[1]} - {match[2]}\n")
                        if match[3]:  # If there is a next line
                            results_file.write(f"  {match[0]}:{match[1] + 1} - {match[3]}\n")
                    results_file.write("\n")
                else:
                    results_file.write(f"  No matches found for {repo_url}\n")
            print(f"Finished processing {repo_url}")

    print(f"Search completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    process_repos()
