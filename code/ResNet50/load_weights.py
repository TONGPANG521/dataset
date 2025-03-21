import os
import torch
import torch.nn as nn
from model import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet50-333f7ec4.pth
    model_weight_path = "./R50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    net = resnet50()
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)

    # option2
    # net = resnet50(num_classes=4)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
