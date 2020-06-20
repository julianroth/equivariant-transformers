# Equivariant Transformer Networks
This is a TensorFlow implementation of Equivariant Transformer Networks presented at ICML 2019 (https://arxiv.org/abs/1901.11399).

Equivariant Transformer (ET) layers are image-to-image mappings that incorporate prior knowledge on invariances with respect to continuous transformation groups. ET layers can be used to normalize the appearance of images prior to classification (or other operations) by a convolutional neural network.

Original Repository with Pytorch code: https://github.com/stanford-futuredata/equivariant-transformers

# Requirements:
- Python >= 3.7
- Tensorflow >= 2.2.0
- Tensorflow Addons >= 0.10.0
- Fire

# TODOs:
- Update doc strings to reflect necessary changes to the implementation
- Add option to use different padding modes for grid sample, right now only 'border' is supported

