import csv
import os
import argparse
import matplotlib.pyplot as plt


def plot_metrics(experiment_name):
    """
    Reads the losses_log.csv file for the given experiment and plots
    the training and validation loss curves, accuracy, and normalized edit distance.

    Args:
        experiment_name (str): The name of the experiment.
    """
    csv_file_path = os.path.join(".", "saved_models", experiment_name, "losses_log.csv")
    base_save_path = os.path.join(".", "saved_models", experiment_name)

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please ensure the experiment name is correct and the training has run.")
        return

    iterations = []
    train_losses = []
    validation_losses = []
    accuracies = []
    norm_eds = []

    try:
        with open(csv_file_path, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            required_fields = [
                "iteration",
                "train_loss",
                "validation_loss",
                "accuracy",
                "norm_ED",
            ]
            if not reader.fieldnames or not all(
                f in reader.fieldnames for f in required_fields
            ):
                print(
                    f"Error: CSV file {csv_file_path} has missing headers. Expected {required_fields}."
                )
                print(f"Found headers: {reader.fieldnames}")
                return

            for row in reader:
                try:
                    iterations.append(int(row["iteration"]))
                    train_losses.append(float(row["train_loss"]))
                    validation_losses.append(float(row["validation_loss"]))
                    accuracies.append(float(row["accuracy"]))
                    norm_eds.append(float(row["norm_ED"]))
                except ValueError as e:
                    print(
                        f"Warning: Skipping row due to data conversion error: {row} - {e}"
                    )
                    continue
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    if not iterations:
        print(f"No data found in {csv_file_path} to plot.")
        return

    # Plot 1: Losses
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(
        iterations,
        validation_losses,
        label="Validation Loss",
        marker="x",
        linestyle="--",
    )
    plt.title(f"Training and Validation Loss for Experiment: {experiment_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, "loss_plot.png"))
    plt.close()

    # Plot 2: Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(
        iterations,
        accuracies,
        label="Accuracy",
        marker="o",
        linestyle="-",
        color="green",
    )
    plt.title(f"Model Accuracy for Experiment: {experiment_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, "accuracy_plot.png"))
    plt.close()

    # Plot 3: Normalized Edit Distance
    plt.figure(figsize=(12, 6))
    plt.plot(
        iterations,
        norm_eds,
        label="Normalized Edit Distance",
        marker="o",
        linestyle="-",
        color="purple",
    )
    plt.title(f"Normalized Edit Distance for Experiment: {experiment_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Edit Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path, "norm_ed_plot.png"))
    plt.close()

    print(f"Plots saved to {base_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training metrics from a CSV log."
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="The name of the experiment (should match the directory in saved_models).",
    )
    args = parser.parse_args()

    plot_metrics(args.experiment_name)
