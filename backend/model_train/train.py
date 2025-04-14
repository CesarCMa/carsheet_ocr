import random
import sys
import time
from test import validation

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import yaml
from dataset import AlignCollate, Batch_Balanced_Dataset, hierarchical_dataset
from loguru import logger
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from utils import Averager

from app.core.recognition import CTCLabelConverter, VGGModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_settings: dict, show_number=2, AutoMixPrecision=False):
    """
    Trains a text recognition model using the specified training settings.

    Args:
        train_settings (dict):
            A dictionary containing all the configurations and hyperparameters for training, such as dataset paths,
            model architecture, optimization settings, and training iterations.

        show_number (int, optional):
            The number of predictions to display during validation for debugging and monitoring purposes.
            Defaults to 2.

        AutoMixPrecision (bool, optional):
            If True, enables automatic mixed precision training to reduce memory usage and improve performance.
            Defaults to False.
    """
    """dataset preparation"""
    if not train_settings["data_filtering_off"]:
        logger.info(
            "Filtering the images containing characters which are not in train_settings['character']"
        )
        logger.info(
            "Filtering the images whose label is longer than train_settings['batch_max_length']"
        )

    train_dataset = Batch_Balanced_Dataset(train_settings)

    log = open(
        f"./saved_models/{train_settings["experiment_name"]}/log_dataset.txt", "a", encoding="utf8"
    )
    AlignCollate_valid = AlignCollate(
        imgH=train_settings["image_height"],
        imgW=train_settings["image_width"],
        keep_ratio_with_pad=train_settings["pad"],
        contrast_adjust=train_settings["contrast_adjust"],
    )
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=train_settings["valid_data"], training_settings=train_settings
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=min(32, train_settings["batch_size"]),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(train_settings["workers"]),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )
    log.write(valid_dataset_log)
    logger.info("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()

    """ model configuration """
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
    logger.info(
        f'model input parameters {train_settings["image_height"]} {train_settings["image_width"]} {train_settings["network_params"]["input_channel"]} {train_settings["network_params"]["output_channel"]} {train_settings["network_params"]["hidden_size"]} {num_class} {train_settings["batch_max_length"]} {train_settings["transformation"]} {train_settings["feature_extraction"]} {train_settings["sequence_modeling"]}'
    )

    if train_settings["saved_model"]:
        pretrained_dict = torch.load(train_settings["saved_model"], map_location=device)
        if train_settings["new_prediction"]:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output, len(pretrained_dict["module.Prediction.weight"])
            )

        model = torch.nn.DataParallel(model).to(device)
        logger.info(f"loading pretrained model from {train_settings['saved_model']}")
        if train_settings["finetune"]:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if train_settings["new_prediction"]:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, num_class)
            for name, param in model.module.Prediction.named_parameters():
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if "localization_fc2" in name:
                logger.info(f"Skip {name} as it is already initialized")
                continue
            try:
                if "bias" in name:
                    init.constant_(param, 0.0)
                elif "weight" in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if "weight" in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)

    model.train()
    logger.info("Model:")
    logger.info(model)
    count_parameters(model)

    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if train_settings["freeze_feature_extraction"]:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if train_settings["freeze_sequence_modeling"]:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info(f"Trainable params num : {sum(params_num)}")
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if train_settings["optim"] == "adam":
        # optimizer = optim.Adam(filtered_parameters, lr=train_settings["lr"], betas=(train_settings["beta1"], 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(
            filtered_parameters,
            lr=train_settings["lr"],
            rho=train_settings["rho"],
            eps=train_settings["eps"],
        )
    logger.info("Optimizer:")
    logger.info(optimizer)

    """ final options """
    with open(
        f"./saved_models/{train_settings['experiment_name']}/opt.txt", "a", encoding="utf8"
    ) as opt_file:
        opt_log = "------------ Options -------------\n"
        for k, v in train_settings.items():
            opt_log += f"{str(k)}: {str(v)}\n"
        opt_log += "---------------------------------------\n"
        logger.info(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if train_settings["saved_model"]:
        try:
            start_iter = int(train_settings["saved_model"].split("_")[-1].split(".")[0])
            logger.info(f"continue to train, start_iter: {start_iter}")
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler("cuda")
    t1 = time.time()

    while True:
        # train part
        optimizer.zero_grad(set_to_none=True)

        if AutoMixPrecision:
            with autocast("cuda"):
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels)
                batch_size = image.size(0)

                preds = model(image).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True

            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels)
            batch_size = image.size(0)

            preds = model(image).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True

            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings["grad_clip"])
            optimizer.step()
        loss_avg.add(cost)

        # validation part
        if (i % train_settings["val_interval"] == 0) and (i != 0):
            logger.info(f"training time: {time.time()-t1}")
            t1 = time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(
                f"./saved_models/{train_settings['experiment_name']}/log_train.txt",
                "a",
                encoding="utf8",
            ) as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_accuracy,
                        current_norm_ED,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data,
                    ) = validation(
                        model, criterion, valid_loader, converter, train_settings, device
                    )
                model.train()

                # training loss and validation loss
                loss_log = f"[{i}/{train_settings['num_iter']}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}"
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{train_settings['experiment_name']}/best_accuracy.pth",
                    )
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(
                        model.state_dict(),
                        f"./saved_models/{train_settings['experiment_name']}/best_norm_ED.pth",
                    )
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
                logger.info(loss_model_log)
                log.write(loss_model_log + "\n")

                # show some predicted results
                dashed_line = "-" * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"

                # show_number = min(show_number, len(labels))

                start = random.randint(0, len(labels) - show_number)
                for gt, pred, confidence in zip(
                    labels[start : start + show_number],
                    preds[start : start + show_number],
                    confidence_score[start : start + show_number],
                ):
                    predicted_result_log += (
                        f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                    )

                predicted_result_log += f"{dashed_line}"
                logger.info(predicted_result_log)
                log.write(predicted_result_log + "\n")
                logger.info(f"validation time: {time.time()-t1}")
                t1 = time.time()
            torch.save(
                model.state_dict(),
                f"./saved_models/{train_settings['experiment_name']}/iter_{i+1}.pth",
            )

        if i == train_settings["num_iter"]:
            logger.info("end the training")
            sys.exit()
        i += 1


def count_parameters(model):
    logger.info("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # table.add_row([name, param])
        total_params += param
        logger.info(f"{name} {param}")
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    with open("model_train/settings.yaml", "r") as f:
        train_settings = yaml.safe_load(f)
    train(train_settings)
