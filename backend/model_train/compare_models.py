import argparse
import time
import torch
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
from utils import Averager
from dataset import hierarchical_dataset, AlignCollate
from app.core.recognition import CTCLabelConverter, VGGModel
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two OCR models")
    parser.add_argument("--base_model", required=True, help="Path to base model (.pth)")
    parser.add_argument(
        "--tuned_model", required=True, help="Path to tuned model (.pth)"
    )
    parser.add_argument("--val_data", required=True, help="Path to validation dataset")
    return parser.parse_args()


def load_training_options(tuned_model_path):
    """Load training options from opt.txt file in the tuned model directory"""
    opt_path = os.path.join(os.path.dirname(tuned_model_path), "opt.txt")
    if not os.path.exists(opt_path):
        raise FileNotFoundError(
            f"opt.txt not found in {os.path.dirname(tuned_model_path)}"
        )

    options = {}
    with open(opt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("---") or not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Try to evaluate the value as a Python literal
                try:
                    if value.startswith("{") or value.startswith("["):
                        value = eval(value)
                    elif value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() == "none":
                        value = None
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").isdigit():
                        value = float(value)
                except:
                    pass

                options[key] = value

    return options


def load_model(model_path, train_settings, device):
    """Load model with the same configuration as in train.py"""
    converter = CTCLabelConverter(
        character=train_settings["character"], lang_list=train_settings["lang_list"]
    )
    num_class = len(converter.character)

    model = VGGModel(
        output_channel=train_settings["network_params"]["output_channel"],
        input_channel=train_settings["network_params"]["input_channel"],
        hidden_size=train_settings["network_params"]["hidden_size"],
        num_class=num_class,
    )

    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, converter


def evaluate_model(model, evaluation_loader, converter, train_settings, device):
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    all_predictions = []  # Store predictions for comparison

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor(
            [train_settings["batch_max_length"]] * batch_size
        ).to(device)
        text_for_pred = (
            torch.LongTensor(batch_size, train_settings["batch_max_length"] + 1)
            .fill_(0)
            .to(device)
        )

        text_for_loss, length_for_loss = converter.encode(labels)

        start_time = time.time()
        preds = model(image)
        forward_time = time.time() - start_time

        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        cost = criterion(
            preds.log_softmax(2).permute(1, 0, 2),
            text_for_loss.to(device),
            preds_size.to(device),
            length_for_loss.to(device),
        )

        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode_greedy(
            preds_index.data.cpu(), preds_size.data.cpu()
        )

        # Clean up predictions by removing square brackets and quotes
        preds_str = [
            pred.replace("[", "").replace("]", "").replace("'", "")
            for pred in preds_str
        ]

        infer_time += forward_time
        valid_loss_avg.add(cost)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        # Store predictions for comparison
        for gt, pred in zip(labels, preds_str):
            all_predictions.append((gt, pred))

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            # Convert to lowercase for comparison
            gt_lower = gt.lower()
            pred_lower = pred.lower()

            if pred_lower == gt_lower:
                n_correct += 1

            if len(gt_lower) == 0 or len(pred_lower) == 0:
                norm_ED += 0
            elif len(gt_lower) > len(pred_lower):
                norm_ED += 1 - edit_distance(pred_lower, gt_lower) / len(gt_lower)
            else:
                norm_ED += 1 - edit_distance(pred_lower, gt_lower) / len(pred_lower)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)
    avg_confidence = (
        sum(confidence_score_list) / len(confidence_score_list)
        if confidence_score_list
        else 0
    )

    return {
        "accuracy": accuracy,
        "norm_ED": norm_ED,
        "avg_confidence": avg_confidence,
        "infer_time": infer_time,
        "samples": length_of_data,
        "valid_loss": valid_loss_avg.val(),
        "predictions": all_predictions,  # Return predictions for comparison
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training options from opt.txt
    train_options = load_training_options(args.tuned_model)

    # Load models
    base_model, base_converter = load_model(args.base_model, train_options, device)
    tuned_model, tuned_converter = load_model(args.tuned_model, train_options, device)

    # Prepare dataset using parameters from opt.txt
    train_settings = {
        "batch_max_length": train_options.get("batch_max_length", 25),
        "image_height": train_options.get("image_height", 32),
        "image_width": train_options.get("image_width", 100),
        "character": train_options.get(
            "character", "0123456789abcdefghijklmnopqrstuvwxyz"
        ),
        "lang_list": train_options.get("lang_list", ["en"]),
        "pad": train_options.get("pad", True),
        "contrast_adjust": train_options.get("contrast_adjust", False),
        "data_filtering_off": train_options.get("data_filtering_off", False),
        "rgb": train_options.get("rgb", False),
        "network_params": train_options.get(
            "network_params",
            {"input_channel": 1, "output_channel": 256, "hidden_size": 256},
        ),
        "transformation": train_options.get("transformation", None),
        "feature_extraction": train_options.get("feature_extraction", "VGG"),
        "sequence_modeling": train_options.get("sequence_modeling", "BiLSTM"),
        "total_data_usage_ratio": train_options.get("total_data_usage_ratio", 1.0),
    }

    AlignCollate_valid = AlignCollate(
        imgH=train_settings["image_height"],
        imgW=train_settings["image_width"],
        keep_ratio_with_pad=train_settings["pad"],
        contrast_adjust=train_settings["contrast_adjust"],
    )

    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=args.val_data, training_settings=train_settings
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=min(32, train_options.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_options.get("workers", 2)),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )

    # Evaluate models
    print("\nEvaluating Base Model...")
    base_results = evaluate_model(
        base_model, valid_loader, base_converter, train_settings, device
    )

    print("\nEvaluating Tuned Model...")
    tuned_results = evaluate_model(
        tuned_model, valid_loader, tuned_converter, train_settings, device
    )

    # Print comparison
    print("\n=== Model Comparison Results ===")
    print(f"{'Metric':<15} {'Base Model':<15} {'Tuned Model':<15} {'Improvement':<15}")
    print("-" * 60)

    metrics = ["accuracy", "norm_ED", "avg_confidence", "infer_time", "valid_loss"]
    for metric in metrics:
        base_val = base_results[metric]
        tuned_val = tuned_results[metric]
        diff = tuned_val - base_val
        print(f"{metric:<15} {base_val:<15.4f} {tuned_val:<15.4f} {diff:<15.4f}")

    print(f"\nTotal samples evaluated: {base_results['samples']}")

    # Print predictions comparison table
    print("\n=== Predictions Comparison ===")
    print(f"{'Ground Truth':<30} | {'Base Model':<30} | {'Tuned Model':<30}")
    print("-" * 95)

    for (gt, base_pred), (_, tuned_pred) in zip(
        base_results["predictions"], tuned_results["predictions"]
    ):
        print(f"{gt:<30} | {base_pred:<30} | {tuned_pred:<30}")

    # Print training parameters used
    print("\n=== Training Parameters Used ===")
    for key, value in train_settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
