import pandas as pd
import matplotlib.pyplot as plt

# Define file paths for CSV files
synced_file = "/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/recvSyncedData_runtime_vs_threads.csv"
yolo_file = "/home/shreyas/ros/src/YOLOv3-Darknet-Detection/data/runDarknet_runtime_vs_threads.csv"

# Load data from CSV files
def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

# Plotting function for individual plots
def plot_graph(data, title, xlabel, ylabel):
    if data is not None:
        threads = data.iloc[:, 0].to_numpy()  # Convert to NumPy array
        durations = data.iloc[:, 1].to_numpy()  # Convert to NumPy array
        plt.figure()
        plt.plot(threads, durations, marker='o', linestyle='-', label=title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

# Combined plot function
def combined_plot(data1, data2, label1, label2):
    if data1 is not None and data2 is not None:
        threads1 = data1.iloc[:, 0].to_numpy()  # Convert to NumPy array
        durations1 = data1.iloc[:, 1].to_numpy()  # Convert to NumPy array
        threads2 = data2.iloc[:, 0].to_numpy()  # Convert to NumPy array
        durations2 = data2.iloc[:, 1].to_numpy()  # Convert to NumPy array
        plt.figure()
        plt.plot(threads1, durations1, marker='o', linestyle='-', label=label1)
        plt.plot(threads2, durations2, marker='x', linestyle='--', label=label2)
        plt.title("Comparison: Runtime vs Threads")
        plt.xlabel("Number of Threads")
        plt.ylabel("Duration (ms)")
        plt.grid(True)
        plt.legend()

# Main script
if __name__ == "__main__":
    # Load data from the CSV files
    synced_data = load_csv_data(synced_file)
    yolo_data = load_csv_data(yolo_file)

    # Plot individual graphs
    plot_graph(synced_data, "SyncedYoloData: Runtime vs Threads", "Number of Threads", "Duration (ms)")
    plot_graph(yolo_data, "YoloClassification: Runtime vs Threads", "Number of Threads", "Duration (ms)")

    plt.show()
