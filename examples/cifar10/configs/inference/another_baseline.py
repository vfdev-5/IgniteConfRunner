
# Example Inference config as python file
from ignite_conf_runner.config_file.basic_configs import BasicInferenceConfig

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite_conf_runner.data_savers import MLFlowCsvDataSaver


# Local file
from dataflow import get_inference_dataloader

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
test_dataloader = get_inference_dataloader(batch_size, num_workers,
                                           device=device, data_augs=test_data_augs)

# Required config param
model = resnet50(pretrained=False, num_classes=10)
model.avgpool = nn.AdaptiveAvgPool2d(1)

# Required config params to load weights
# Following information can be found in the output folder or with `mlflow ui`
run_uuid = "7ffd66b7a3c34fabb1e23956053671e9"
model_weights_filename = "model_ResNet_22_val_loss=1.333357.pth"


# # Optional config param:
# def final_activation(output):
#     return torch.argmax(torch.softmax(output, dim=-1), dim=-1)


# Required config param
predictions_datasaver = MLFlowCsvDataSaver()
