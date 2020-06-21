# Equivariant Transformer Networks
This is a TensorFlow implementation of Equivariant Transformer Networks presented at ICML 2019 (https://arxiv.org/abs/1901.11399).

Equivariant Transformer (ET) layers are image-to-image mappings that incorporate prior knowledge on invariances with respect to continuous transformation groups. ET layers can be used to normalize the appearance of images prior to classification (or other operations) by a convolutional neural network.

Original Repository with Pytorch code: https://github.com/stanford-futuredata/equivariant-transformers

# Requirements:
- Python >= 3.7
- Tensorflow >= 2.2.0
- Tensorflow Addons >= 0.10.0
- Fire

# Usage
To download and preprocess the projective MNIST dataset, run:

```bash
python projective_mnist.py --data-dir=<PATH>
```

To train a model run:

```bash
python experiment_mnist.py train --path=<PATH> [--save_path=<SAVE_PATH>]
```

The `save_path` flag lets us specify a path to save the model that achieves the best validation accuracy during training.

To change the set of transformers used by the model, we can use the `tfs` flag to specify a list of class names from the `etn.transformers` module. For example:

```bash
python experiment_mnist.py train ... --tfs="[ShearX, HyperbolicRotation]"
```

To train a model without any transformers, we can simply set `tfs` to the empty list `[]`:

```bash
python experiment_mnist.py train ... --tfs=[]
```

We can also set the coordinate transformation that's applied immediately prior to classification by the CNN:

```bash
python experiment_mnist.py train ... --coords=logpolar
```

Therefore, to train a bare-bones model without any transformer layers or coordinate transformations, we can run:

```bash
python experiment_mnist.py train ... --tfs=[] --coords=identity
```

Feel free to play around with different combinations of transformers and coordinate systems!

To train a non-equivariant model, we can set the `equivariant` flag to `False`:

```bash
python experiment_mnist.py train ... --equivariant=False
```

To evaluate a saved model on the test set, run:

```bash
python experiment_mnist.py --load_path=<SAVE_PATH> test --test_path=<PATH>
```


# TODOs:
- Update doc strings to reflect necessary changes to the implementation
- Add option to use different padding modes for grid sample, right now only 'border' is supported

