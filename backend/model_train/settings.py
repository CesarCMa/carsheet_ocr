from dataclasses import dataclass
from typing import Dict, List, Optional


from dataclasses import dataclass, field


@dataclass
class TrainingSettings:
    # Network parameters
    network_params: dict = field(
        default_factory=lambda: {
            "input_channel": 1,
            "output_channel": 256,
            "hidden_size": 256,
        }
    )

    # Image parameters
    image_height: int = 64
    image_width: int = 256
    rgb: bool = False
    contrast_adjust: bool = False
    pad: bool = True

    # Training parameters
    batch_size: int = 32
    num_iter: int = 10
    val_interval: int = 2
    workers: int = 2
    grad_clip: float = 5.0

    # Dataset parameters
    train_data: str = "model_train/data/en_sample"
    valid_data: str = "model_train/data/en_sample"
    select_data: List[str] = field(default_factory=lambda: ["MJ", "ST"])
    batch_ratio: List[float] = field(default_factory=lambda: [0.5, 0.5])
    total_data_usage_ratio: float = 1.0
    data_filtering_off: bool = False
    batch_max_length: int = 25

    # Model architecture
    transformation: str = "None"
    feature_extraction: str = "VGG"
    sequence_modeling: str = "BiLSTM"

    # Optimization
    optim: str = "adam"
    lr: float = 1.0
    rho: float = 0.95
    eps: float = 1e-8

    # Model loading and saving
    saved_model: str = "model_train/model_checkpoints/english_g2.pth"
    experiment_name: str = "test_finetune"
    new_prediction: bool = False
    finetune: bool = True

    # Layer freezing
    freeze_feature_extraction: bool = True
    freeze_sequence_modeling: bool = False

    character: str = (
        "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    lang_list: list[str] = ["en"]
