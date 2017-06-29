#!/bin/bash
bazel-bin/resnet/resnet_main --train_data_path=cifar100/train.bin \
                               --out_dir=resnet/output/cifar100/32_layers \
                               --log_name=resnet/log/cifar100/32_layers \
                               --dataset='cifar100' \
                               --num_gpus=1

bazel-bin/resnet/resnet_main --train_data_path=cifar100/train.bin \
                               --out_dir=resnet/output/cifar100/110_layers \
                               --log_name=resnet/log/cifar100/110_layers \
                               --dataset='cifar100' \
                               --num_gpus=1
