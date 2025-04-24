import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Paths
prefix = "../exp_runners/experiments/message_types-stdalone-1/"
s_folder = "../exp_runners/experiments/message_types-stdalone-1/"
std_folder = "../standalone/"
d_folder = "./graphs/"

# Ensure the output folder exists
os.makedirs(d_folder, exist_ok=True)

# Load Run Table Once
def load_run_table():
    try:
        return pd.read_csv(os.path.join(std_folder, "run_table.csv"))
    except Exception as e:
        print(f"Error loading run_table.csv: {e}")
        sys.exit(1)

def load_avg_energy(component, msg_interval, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)
    df = df[df["msg_type"] == msg_interval]
    
    return df.groupby(["msg_interval", "msg_size"])

def load_avg_cpu(component, msg_interval, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Utilization"] = pd.to_numeric(energy_df["CPU Utilization"], errors="coerce")
                avg_energy_pct = energy_df["CPU Utilization"].mean() * 100
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)
    df = df[df["msg_type"] == msg_interval]
    
    return df.groupby(["msg_interval", "msg_size"])

def load_avg_energy_machine(component, msg_interval, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)
    df = df[df["msg_type"] == msg_interval]
    
    return df.groupby(["msg_interval", "msg_size"])

# Generate Boxplots
def gen_energy_boxplot(component, msg_interval, run_table):
    grouped = load_avg_energy(component, msg_interval, run_table)

    boxplot_data = []
    labels = []

    for group_name, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(group_name)

    # Define a mapping for the second value
    size_map = {1: "small", 2: "medium", 3: "large"}

    # Apply the mapping when formatting labels
    labels = [f"({float(x)}, {size_map.get(int(y), int(y))})" for x, y in labels]

    # Plot using Seaborn
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.ylabel("Power (W)")
    plt.xlabel("Interval, Size")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_{component}_energy_{msg_interval}.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_energy_boxplot_machine(component, msg_interval, run_table):
    grouped = load_avg_energy_machine(component, msg_interval, run_table)

    boxplot_data = []
    labels = []

    for group_name, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(group_name)

    # Define a mapping for the second value
    size_map = {1: "small", 2: "medium", 3: "large"}

    # Apply the mapping when formatting labels
    labels = [f"({float(x)}, {size_map.get(int(y), int(y))})" for x, y in labels]

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.title(f"Energy Consumption (Msg. Type: {msg_interval})")
    plt.ylabel("Average Power (W)")
    plt.xlabel("Interval, Size")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_machine_energy_{msg_interval}.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_energy_boxplot_by_msg_type(component, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Group by msg_type only
    grouped = df.groupby("msg_type")

    boxplot_data = []
    labels = []

    for group_name, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(group_name)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.ylabel("Power (W)")
    plt.xlabel("")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_{component}_energy_by_msg_type.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_cpu_boxplot_by_msg_type(component, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Utilization"] = pd.to_numeric(energy_df["CPU Utilization"], errors="coerce")
                avg_energy_pct = energy_df["CPU Utilization"].mean() * 100 * 8
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Group by msg_type only
    grouped = df.groupby("msg_type")

    boxplot_data = []
    labels = []

    for group_name, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(group_name)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.ylabel("CPU Usage (percentage of one core)")
    plt.xlabel("")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_{component}_cpu_by_msg_type.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_mem_boxplot_by_msg_type(component, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-{component}-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Utilization"] = pd.to_numeric(energy_df["CPU Utilization"], errors="coerce")
                avg_energy_pct = energy_df["CPU Utilization"].mean() * 100 * 8
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Group by msg_type only
    grouped = df.groupby("msg_type")

    boxplot_data = []
    labels = []

    for group_name, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(group_name)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.ylabel("CPU Usage (percentage of one core)")
    plt.xlabel("")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_{component}_cpu_by_msg_type.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_energy_boxplot_by_clients_for_msg_type(msg_type, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-server-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Filter for the specific msg_type
    df = df[df["msg_type"] == msg_type]

    # Group by number of clients (cli)
    grouped = df.groupby("cli")

    boxplot_data = []
    labels = []

    for cli_val, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(f"{cli_val} clients")

    # Plot
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    # plt.title(f"Energy Consumption (Server) - {msg_type} grouped by Clients")
    plt.ylabel("Power (W)")
    plt.xlabel("")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_server_energy_{msg_type}_by_clients.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_energy_boxplot_by_clients_for_msg_type_interval(msg_type, run_table, interval):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-server-powerjoular.csv-*.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Filter for the specific msg_type
    df = df[df["msg_type"] == msg_type]

    # interval
    df = df[df["msg_interval"] == interval]

    # Group by number of clients (cli)
    grouped = df.groupby("cli")

    boxplot_data = []
    labels = []

    for cli_val, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(f"{cli_val} clients")

    # Plot
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Power (W)")
    plt.xlabel("")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_server_energy_{msg_type}_by_clients_interval_{interval}.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_energy_boxplot_by_clients_for_msg_type_machine(msg_type, run_table):
    runs_data = {}

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        energy_files = glob.glob(os.path.join(folder_path, f"energy-server-powerjoular.csv"))

        if energy_files:
            try:
                energy_df = pd.read_csv(energy_files[0])
                energy_df["CPU Power"] = pd.to_numeric(energy_df["CPU Power"], errors="coerce")
                avg_energy_pct = energy_df["CPU Power"].mean()
                runs_data[run_id] = avg_energy_pct
            except Exception as e:
                print(f"Error processing file {energy_files[0]} for run_id {run_id}: {e}")

    avg_cpu_df = pd.DataFrame(list(runs_data.items()), columns=["__run_id", "avg_energy_pct"])
    df = run_table.merge(avg_cpu_df, on="__run_id", how="left").fillna(0)

    # Filter for the specific msg_type
    df = df[df["msg_type"] == msg_type]

    # Group by number of clients (cli)
    grouped = df.groupby("cli")

    boxplot_data = []
    labels = []

    for cli_val, group_data in grouped:
        boxplot_data.append(group_data["avg_energy_pct"].values)
        labels.append(f"{cli_val} clients")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.title(f"Energy Consumption (Server) - {msg_type} grouped by Clients")
    plt.ylabel("Average Power (W)")
    plt.xlabel("Number of Clients")
    plt.tight_layout()

    output_file = os.path.join(d_folder, f"boxplot_machine_energy_{msg_type}_by_clients.pdf")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

def gen_network_plot_by_msg_type(run_table):
    # Collect rx/tx average rates per run_id
    net_data = []

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        net_file = os.path.join(folder_path, "network.csv")

        if os.path.exists(net_file):
            try:
                net_df = pd.read_csv(net_file)
                avg_rx = net_df["rx_rate_bps"].mean()
                avg_tx = net_df["tx_rate_bps"].mean()
                net_data.append({"__run_id": run_id, "avg_rx_rate_bps": avg_rx, "avg_tx_rate_bps": avg_tx})
            except Exception as e:
                print(f"Error processing network.csv for run_id {run_id}: {e}")
        else:
            print(f"Missing network.csv for run_id {run_id}")

    # Create a DataFrame with the averages
    net_df = pd.DataFrame(net_data)

    # Merge with run_table to get msg_type
    merged_df = run_table.merge(net_df, on="__run_id", how="inner")  # Only keep runs with network data

    # Group by msg_type
    grouped = merged_df.groupby("msg_type")

    # Plot for each msg_type
    for msg_type, group in grouped:
        plt.figure(figsize=(10, 6))

        # Prepare data for boxplot (if you prefer bar plot, change it)
        data = [group["avg_rx_rate_bps"].values, group["avg_tx_rate_bps"].values]
        labels = ["RX Rate (bps)", "TX Rate (bps)"]

        sns.boxplot(data=data)
        plt.xticks(ticks=range(len(labels)), labels=labels)
        plt.title(f"Network Usage - {msg_type}")
        plt.ylabel("Average Rate (bps)")
        plt.tight_layout()

        output_file = os.path.join(d_folder, f"network_boxplot_{msg_type}.pdf")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved: {output_file}")


def gen_network_rx_tx_grouped_by_msg_type(run_table):
    # Collect rx/tx average rates per run_id
    net_data = []

    for run_id in run_table["__run_id"]:
        folder_path = os.path.join(s_folder, str(run_id))
        net_file = os.path.join(folder_path, "network.csv")

        if os.path.exists(net_file):
            try:
                net_df = pd.read_csv(net_file)
                avg_rx = net_df["rx_rate_bps"].mean()
                avg_tx = net_df["tx_rate_bps"].mean()
                net_data.append({
                    "__run_id": run_id,
                    "avg_rx_rate_bps": avg_rx,
                    "avg_tx_rate_bps": avg_tx
                })
            except Exception as e:
                print(f"Error processing network.csv for run_id {run_id}: {e}")
        else:
            print(f"Missing network.csv for run_id {run_id}")

    # Merge network data with run_table
    net_df = pd.DataFrame(net_data)
    merged_df = run_table.merge(net_df, on="__run_id", how="inner")

    # --- RX Rate Plot ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df, x="msg_type", y="avg_rx_rate_bps")
    plt.xticks(rotation=45)
    plt.title("RX Rate (bps) grouped by msg_type")
    plt.ylabel("Average RX Rate (bps)")
    plt.xlabel("Message Type")
    plt.tight_layout()
    rx_output = os.path.join(d_folder, "network_rx_rate_by_msg_type.pdf")
    plt.savefig(rx_output, dpi=300)
    plt.close()
    print(f"Saved: {rx_output}")

    # --- TX Rate Plot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged_df, x="msg_type", y="avg_tx_rate_bps")
    plt.xticks(rotation=90)
    plt.title("TX Rate (bps) grouped by msg_type")
    plt.ylabel("Average TX Rate (bps)")
    plt.xlabel("Message Type")
    plt.tight_layout()
    tx_output = os.path.join(d_folder, "network_tx_rate_by_msg_type.pdf")
    plt.savefig(tx_output, dpi=300)
    plt.close()
    print(f"Saved: {tx_output}")


def gen_network_rx_tx_for_string_by_interval(run_table, s_folder, d_folder):
    """
    Generate boxplots of RX and TX rates for 'String' msg_type grouped by msg_interval.

    :param run_table: DataFrame containing '__run_id', 'msg_type', and 'msg_interval' columns
    :param s_folder: Path to the folder where run folders are stored
    :param d_folder: Path to the folder where plots will be saved
    """
    net_data = []

    # Collect average RX/TX rates per run
    for run_id in run_table["__run_id"]:
        net_file = os.path.join(s_folder, str(run_id), "network.csv")
        if not os.path.exists(net_file):
            print(f"[WARNING] Missing network.csv for run_id {run_id}")
            continue

        try:
            net_df = pd.read_csv(net_file)
            avg_rx = net_df["rx_rate_bps"].mean()
            avg_tx = net_df["tx_rate_bps"].mean()
            net_data.append({
                "__run_id": run_id,
                "avg_rx_rate_bps": avg_rx,
                "avg_tx_rate_bps": avg_tx
            })
        except Exception as e:
            print(f"[ERROR] Processing network.csv for run_id {run_id}: {e}")

    if not net_data:
        print("[ERROR] No network data collected. Exiting plot generation.")
        return

    # Merge network data with run_table
    net_df = pd.DataFrame(net_data)
    merged_df = run_table.merge(net_df, on="__run_id", how="inner")

    # Filter only String msg_type
    string_df = merged_df[merged_df["msg_type"] == "String"]

    if string_df.empty:
        print("[ERROR] No data found for msg_type 'String'.")
        return

    # --- RX Rate Plot ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=string_df, x="msg_interval", y="avg_rx_rate_bps")
    plt.xticks(rotation=45, ha='right')
    plt.title("RX Rate (bps) for 'String' grouped by msg_interval")
    plt.ylabel("Average RX Rate (bps)")
    plt.xlabel("Message Interval (ms)")
    plt.tight_layout()
    rx_output = os.path.join(d_folder, "network_rx_rate_string_by_interval.pdf")
    plt.savefig(rx_output, dpi=300)
    plt.close()
    print(f"[SAVED] RX plot: {rx_output}")

    # --- TX Rate Plot ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=string_df, x="msg_interval", y="avg_tx_rate_bps")
    plt.xticks(rotation=45, ha='right')
    plt.title("TX Rate (bps) for 'String' grouped by msg_interval")
    plt.ylabel("Average TX Rate (bps)")
    plt.xlabel("Message Interval (ms)")
    plt.tight_layout()
    tx_output = os.path.join(d_folder, "network_tx_rate_string_by_interval.pdf")
    plt.savefig(tx_output, dpi=300)
    plt.close()
    print(f"[SAVED] TX plot: {tx_output}")

def gen_network_rx_tx_for_string_by_msg_size(run_table, s_folder, d_folder):
    """
    Generate boxplots of RX and TX rates for 'String' msg_type grouped by msg_size.

    :param run_table: DataFrame containing '__run_id', 'msg_type', and 'msg_size' columns
    :param s_folder: Path to the folder where run folders are stored
    :param d_folder: Path to the folder where plots will be saved
    """
    net_data = []

    # Collect average RX/TX rates per run
    for run_id in run_table["__run_id"]:
        net_file = os.path.join(s_folder, str(run_id), "network.csv")
        if not os.path.exists(net_file):
            print(f"[WARNING] Missing network.csv for run_id {run_id}")
            continue

        try:
            net_df = pd.read_csv(net_file)
            avg_rx = net_df["rx_rate_bps"].mean()
            avg_tx = net_df["tx_rate_bps"].mean()
            net_data.append({
                "__run_id": run_id,
                "avg_rx_rate_bps": avg_rx,
                "avg_tx_rate_bps": avg_tx
            })
        except Exception as e:
            print(f"[ERROR] Processing network.csv for run_id {run_id}: {e}")

    if not net_data:
        print("[ERROR] No network data collected. Exiting plot generation.")
        return

    # Merge network data with run_table
    net_df = pd.DataFrame(net_data)
    merged_df = run_table.merge(net_df, on="__run_id", how="inner")

    # Filter only String msg_type
    string_df = merged_df[merged_df["msg_type"] == "String"]

    if string_df.empty:
        print("[ERROR] No data found for msg_type 'String'.")
        return

    # --- RX Rate Plot ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=string_df, x="msg_size", y="avg_rx_rate_bps")
    plt.xticks(rotation=45, ha='right')
    plt.title("RX Rate (bps) for 'String' grouped by msg_size")
    plt.ylabel("Average RX Rate (bps)")
    plt.xlabel("Message Size (Bytes)")
    plt.tight_layout()
    rx_output = os.path.join(d_folder, "network_rx_rate_string_by_msg_size.pdf")
    plt.savefig(rx_output, dpi=300)
    plt.close()
    print(f"[SAVED] RX plot: {rx_output}")

    # --- TX Rate Plot ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=string_df, x="msg_size", y="avg_tx_rate_bps")
    plt.xticks(rotation=45, ha='right')
    plt.title("TX Rate (bps) for 'String' grouped by msg_size")
    plt.ylabel("Average TX Rate (bps)")
    plt.xlabel("Message Size (Bytes)")
    plt.tight_layout()
    tx_output = os.path.join(d_folder, "network_tx_rate_string_by_msg_size.pdf")
    plt.savefig(tx_output, dpi=300)
    plt.close()
    print(f"[SAVED] TX plot: {tx_output}")



# Parameters
components = {"server", "client"}
intervals = {0.1, 0.2, 0.5, 1.0}
msg_types = {
            "Image", "Pose", "JointState", "PointCloud2",
            "Imu", "String", "PoseStamped", "LaserScan",
            "Float64", "Float32", "Vector3Stamped", "Float64MultiArray",
            "TwistStamped", "PoseWithCovarianceStamped", "Int32"
            }

# Load run table once
run_table = load_run_table()

# Generate plots
for component in components:
    for interval in intervals:
        gen_energy_boxplot(component, interval, run_table)
    for m_type in msg_types:
        gen_energy_boxplot(component, m_type, run_table)
        #gen_energy_violinplot(component, m_type, run_table)
        if (component == 'server'):
            #gen_energy_boxplot_by_msg_type_machine(m_type, run_table)
            # gen_energy_boxplot_machine(component, m_type, run_table)
            #gen_energy_boxplot_by_clients_for_msg_type_machine(m_type, run_table)
            gen_energy_boxplot_by_clients_for_msg_type(m_type, run_table)
            for i in intervals:
                 gen_energy_boxplot_by_clients_for_msg_type_interval(m_type, run_table, i)
                 pass
            pass
    gen_energy_boxplot_by_msg_type(component, run_table)
    gen_cpu_boxplot_by_msg_type(component, run_table)
