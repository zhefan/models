#!/bin/bash
# resnet-32
bazel-bin/resnet/resnet_main --train_data_path=cifar100/train.bin \
                               --out_dir=resnet/output/cifar100/32_layers \
                               --log_name=resnet/log/cifar100/32_layers \
                               --dataset='cifar100' \
                               --num_res_units=5 \
                               --num_gpus=1

# resnet-50
bazel-bin/resnet/resnet_main --train_data_path=cifar100/train.bin \
                               --out_dir=resnet/output/cifar100/50_layers \
                               --log_name=resnet/log/cifar100/50_layers \
                               --dataset='cifar100' \
                               --num_res_units=8 \
                               --num_gpus=1

# resnet-110
bazel-bin/resnet/resnet_main --train_data_path=cifar100/train.bin \
                               --out_dir=resnet/output/cifar100/110_layers \
                               --log_name=resnet/log/cifar100/110_layers \
                               --dataset='cifar100' \
                               --num_res_units=18 \
                               --num_gpus=1
