
# Example Inference config as python file
from ignite_conf_runner.config_file.basic_configs import BasicInferenceConfig
import torch
import torch.nn as nn

from torchvision.models.resnet import resnet34
from torchvision.transforms import Compose, ToTensor, Normalize

import mlflow
service = mlflow.tracking.get_service()

from ignite_conf_runner.data_savers import DataSaver, CsvDatasetSaver


# Local file
from dataflow import get_basic_dataloader

# Required config param
config_class = BasicInferenceConfig

# Optional config param
seed = 12345

# Optional config param
device = 'cuda'

batch_size = 128
num_workers = 8

test_data_augs = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Required config param
test_dataloader = get_basic_dataloader("test", batch_size, num_workers,
                                       device=device, data_augs=test_data_augs)

# Required config param
model = resnet34(pretrained=False, num_classes=10)
model.avgpool = nn.AdaptiveAvgPool2d(1)

# Load weights
run_uuid = "a815438826b74cf38de3dca5067632c6"
model.load_state_dict(torch.load(service.download_artifacts(run_uuid, "model_ResNet_10_val_loss=1.478798.pth")))


csv_dataset_saver = CsvDatasetSaver(predictions_header=['c_{}'.format(i) for i in range(10)],
                                    output_path="output", total=len(test_dataloader.sampler))
# Required config param
predictions_datasaver = DataSaver(csv_dataset_saver)
