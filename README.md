# 🧪 Replication Package for the Paper `How ROS 2 Talks and at What Cost: Guidelines for Energy-Aware Topic Communication`

This repository contains the full replication package for the paper titled `How ROS 2 Talks and at What Cost: Guidelines for Energy-Aware Topic Communication`, submitted to the `ICSEP-SEIP 2025`.

---

## 📁 Repository Structure

The repository is organized as follows:

```
.
├── crawling/                    # Scripts to crawl GitHub repositories
│   ├── github/                  # GitHub-specific crawlers
│   └── extractors/              # Logic to extract topics
├── data-analysis/               # Graphs, statistics, and analysis scripts
│   ├── data/                    # Raw and processed data
│   ├── graphs/                  # All generated plots
│   │   └── qualitative/         # Qualitative analysis plots
├── docker/                      # Docker setup and Compose configurations
├── exp_runners/                 # Artifacts for running ROS 2 message exchange experiments
│   ├── experiments/             # Experiment output data
│   └── standalone/              # Scripts to run the experiments with shell script
├── resources/                   # Crawled metadata (e.g., repos.csv, pub/sub info)
│   ├── repo-urls.csv            # URLs of selected repositories
|   └── repos.csv                # List of crawled repositories
├── repos/                       # Cloned GitHub repositories (ignored in .gitignore)
├── README.md                    # Project instructions and overview
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignore file for logs, venv, repos, etc.
└── setup.bash                   # Environment configuration for experiments

```

---

## ⚙️ Initial Setup

Ensure your system meets the following requirements:

- **Operating System:** Ubuntu 22.04
- **ROS Version:** ROS 2 Humble
- **Tools:** Docker, Docker Compose, Python 3

### 🐍 Python Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔎 Crawling ROS 2 Repositories

Before crawling, set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN=yourtoken
```

To initiate the crawling process:

```bash
python3 crawling/scripts/github/crawler.py
```

The list of repositories will be saved in the `resources/repos.csv` file.

---

## 📊 Qualitative Analysis Data

The [qualitative analysis file](./data-analysis/data/qualitative_clean.csv) is available as the following organization:

```
.
├── data-analysis/  
│   └── data/ 
│       └── qualitative_clean.csv   # Qualitative analysis with cleaned message types (no prefix)
```

---

## 📦 Extracting Publishers and Subscribers

For this, we keep the URL of selected projects in the `./resources/repo-urls.csv` file.

```bash
python3 crawling/scripts/get_repo_data.py
```

The result of this execution is saved in the `./resources/repos-pubsub.txt` file. The script also clone all the repositories to the `repos` folder, what may also take some time to execute.

---

## 🧬 Extracting Message Types and Topics

TBA: We are still cleaning this code.

---

### 🛠️ ROS 2 Installation (if not installed)

```bash
sudo apt install ros-jazzy-ros-base
```

## Installing PowerJoular - For Energy Measurement

```bash
sudo apt install gnat gprbuild
git clone https://github.com/joular/powerjoular.git ../powerjoular
cd ../powerjoular
gprbuild
```

---

## 🧪 Running Experiments

If you wish, you can re-run all the experiments with the following command (before that, set the correct paths in the `setup.bash` file):

```bash
source setup.bash
bash run-pubsub-stdalone.sh
```

---

## 📈 Generating Data Analysis Graphs

```bash
cd data-analysis/
python3 gen_graphs-mac.py
bash run_qualitative.sh
```

All the graphs will be saved in the `graphs` folder, with qualitative analysis graphs in the `graphs/qualitative/`.

## 📉 Statistical Analysis

All the statistical tests are run at once, and their results are save in the `./data-analysis/statistics.log` file. This logs need to be improved for a better readability. 

```bash
cd data-analysis/
python3 statistical_tests.py > statistics.log
```
