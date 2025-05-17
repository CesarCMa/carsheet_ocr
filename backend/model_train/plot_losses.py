import csv
import os
import argparse
import matplotlib.pyplot as plt

def plot_loss_curves(experiment_name):
    """
    Reads the losses_log.csv file for the given experiment and plots
    the training and validation loss curves.

    Args:
        experiment_name (str): The name of the experiment.
    """
    csv_file_path = os.path.join(".", "saved_models", experiment_name, "losses_log.csv")
    plot_save_path = os.path.join(".", "saved_models", experiment_name, "loss_plot.png")

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please ensure the experiment name is correct and the training has run.")
        return

    iterations = []
    train_losses = []
    validation_losses = []

    try:
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not all(f in reader.fieldnames for f in ['iteration', 'train_loss', 'validation_loss']):
                print(f"Error: CSV file {csv_file_path} has missing headers. Expected 'iteration', 'train_loss', 'validation_loss'.")
                print(f"Found headers: {reader.fieldnames}")
                return
                
            for row in reader:
                try:
                    iterations.append(int(row['iteration']))
                    train_losses.append(float(row['train_loss']))
                    validation_losses.append(float(row['validation_loss']))
                except ValueError as e:
                    print(f"Warning: Skipping row due to data conversion error: {row} - {e}")
                    continue
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    if not iterations:
        print(f"No data found in {csv_file_path} to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(iterations, validation_losses, label='Validation Loss', marker='x', linestyle='--')

    plt.title(f'Training and Validation Loss for Experiment: {experiment_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot to {plot_save_path}: {e}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves from a CSV log.")
    parser.add_argument(
        "experiment_name",
        type=str,
        help="The name of the experiment (should match the directory in saved_models)."
    )
    args = parser.parse_args()

    plot_loss_curves(args.experiment_name) 