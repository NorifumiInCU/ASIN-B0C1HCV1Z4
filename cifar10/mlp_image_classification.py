# base : https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py
"""
Title: Image classification with modern MLP models
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/05/30
Last modified: 2023/08/03
Description: Implementing the MLP-Mixer, FNet, and gMLP models for CIFAR-100 image classification.
Accelerator: GPU
"""

"""
## Introduction

This example implements three modern attention-free, multi-layer perceptron (MLP) based models for image
classification, demonstrated on the CIFAR-100 dataset:

1. The [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.
3. The [FNet](https://arxiv.org/abs/2105.03824) model, by James Lee-Thorp et al., based on unparameterized
Fourier Transform.
2. The [gMLP](https://arxiv.org/abs/2105.08050) model, by Hanxiao Liu et al., based on MLP with gating.

The purpose of the example is not to compare between these models, as they might perform differently on
different datasets with well-tuned hyperparameters. Rather, it is to show simple implementations of their
main building blocks.
"""

"""
## Setup
"""

import keras
from keras import layers

from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import tensorflow as tf
from F1ScoreCallback import F1ScoreCallback
from basic import get_basic_data
transpose = tf.transpose
dirname, basename, odir=get_basic_data(__file__)

"""
## Prepare the data
"""

num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# normalization
editx = lambda x:x.astype('float32') / 255.0
x_train = editx(x_train)
x_test = editx(x_test)

if __name__=='__main__':
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Configure the hyperparameters
"""

weight_decay = 0.0001
batch_size = 128
num_epochs = 3  # Recommended num_epochs = 50
do_mlpmixer = True
do_fnet = False
do_gmlp = False
if __name__=='__main__':
    import sys
    if len(sys.argv)==1:
        print(f'usage:{sys.argv[0]} <epochs>=3 <do mlpmixer (0:donot/1:do)>=1 <do fnet (0:donot/1:do)>=0 <do gmlp (0:donot/1:do)>=0')
        num_epochs=3
        do_mlpmixer = True
        do_fnet = False
        do_gmlp = False
    if len(sys.argv)>1:
        if sys.argv[1].isdecimal():
            num_epochs=int(sys.argv[1])
        else:
            print(f'usage:{sys.argv[0]} <epochs> <do mlpmixer (0:donot/1:do)> <do fnet (0:donot/1:do)> <do gmlp (0:donot/1:do)>')
            exit(0)
    if len(sys.argv)>2:
        do_mlpmixer=int(sys.argv[2])==1
    if len(sys.argv)>3:
        do_fnet=int(sys.argv[3])==1
    if len(sys.argv)>4:
        do_gmlp=int(sys.argv[4])==1
    print(f'epochs:{num_epochs} do_mlpmixer:{do_mlpmixer} do_fnet:{do_fnet} do_gmlp:{do_gmlp}')
    
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
embedding_dim = 256  # Number of hidden units.
num_blocks = 4  # Number of blocks.

if __name__=='__main__':
    print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
    print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
    print(f"Patches per image: {num_patches}")
    print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")

"""
## Build a classification model

We implement a method that builds a classifier given the processing blocks.
"""


def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        x = x + PositionEmbedding(sequence_length=num_patches)(x)
    # Process x using the module blocks.
    x = blocks(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)


"""
## Define an experiment

We implement a utility function to compile, train, and evaluate a given model.
"""


def run_experiment(model, f1_score_callback):
    # Create Adam optimizer with weight decay.
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # f1_score_callback = F1ScoreCallback((x_test, y_test))

    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr, f1_score_callback],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history


"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


"""
## Implement patch extraction as a layer
"""

@register_keras_serializable(name='Patches')
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        patches = keras.ops.image.extract_patches(x, self.patch_size)
        batch_size = keras.ops.shape(patches)[0]
        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
        patch_dim = keras.ops.shape(patches)[3]
        out = keras.ops.reshape(patches, (batch_size, num_patches, patch_dim))
        return out

"""
## Implement position embedding as a layer
"""


class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape

def myplot(fid:str):
    val_f1 = f1_score_callback.f1_scores
    f1_score_history = val_f1

    x_data = [i for i in range(1, len(hist.history['acc']) +1)]
    # 学習の様子をグラフへ描画
    plt.figure(figsize=(15,6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Accuracy')
    ax1.plot(x_data, hist.history['acc'], label='acc')
    ax1.plot(x_data, hist.history['val_acc'], label='val_acc')
    ax1_2=ax1.twinx()
    ax1_2.plot(x_data, f1_score_history, ':', label='f1_score')
    # プロットした後に y軸の最後の値を取得して表示
    train_acc_last = hist.history['acc'][-1]
    test_acc_last = hist.history['val_acc'][-1]
    test_f1score_last = f1_score_history[-1]
    # y軸の最後の値をテキストで表示
    common_bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
    ax1.annotate(f'last acc:{train_acc_last:.4f}', xy=(0.0, 0.3), xytext=(10, 0), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)
    ax1.annotate(f'last val_acc:{test_acc_last:.4f}', xy=(0.0, 0.4), xytext=(10, 10), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)
    ax1_2.annotate(f'last val_f1_score:{test_f1score_last:.4f}', xy=(0.6, 0.5), xytext=(10, 20), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Loss')
    ax2.plot(x_data, hist.history['loss'], label='loss')
    ax2.plot(x_data, hist.history['val_loss'], label='val_loss')
    ax2_2=ax2.twinx()
    ax2_2.plot(x_data, f1_score_history, ':', label='f1_score')
    # プロットした後に y軸の最後の値を取得して表示
    train_loss_last = hist.history['loss'][-1]
    test_loss_last = hist.history['val_loss'][-1]
    # y軸の最後の値をテキストで表示
    ax2.annotate(f'last loss:{train_loss_last:.4f}', xy=(0.0, 0.3), xytext=(10, 0), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)
    ax2.annotate(f'last val_loss:{test_loss_last:.4f}', xy=(0.0, 0.4), xytext=(10, 10), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)
    ax2_2.annotate(f'last val_f1_score:{test_f1score_last:.4f}', xy=(0.6, 0.5),xytext=(10, 20), xycoords='axes fraction', textcoords='offset points', ha='left', va='center', bbox=common_bbox)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.savefig(f'{odir}/{fsymbol}.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


"""
## The MLP-Mixer model

The MLP-Mixer is an architecture based exclusively on
multi-layer perceptrons (MLPs), that contains two types of MLP layers:

1. One applied independently to image patches, which mixes the per-location features.
2. The other applied across patches (along channels), which mixes spatial information.

This is similar to a [depthwise separable convolution based model](https://arxiv.org/abs/1610.02357)
such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization
instead of batch normalization.
"""

"""
### Implement the MLP-Mixer module
"""


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


"""
### Build, train, and evaluate the MLP-Mixer model

Note that training the model with the current settings on a V100 GPUs
takes around 8 seconds per epoch.
"""

# MLPMixer
if __name__=='__main__' and do_mlpmixer:
    mlpmixer_blocks = keras.Sequential(
        [MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
    )
    learning_rate = 0.005
    mlpmixer_classifier = build_classifier(mlpmixer_blocks)
    mlpmixer_classifier.summary()
    f1_score_callback = F1ScoreCallback((x_test, y_test))
    hist = run_experiment(mlpmixer_classifier, f1_score_callback)

    fsymbol=f'{basename}-mlpmixer-epoch{num_epochs}'
    myplot(fsymbol)

    mlpmixer_classifier.save(f'{odir}/{fsymbol}.keras')
    print(f'saved {fsymbol}.keras')

"""
The MLP-Mixer model tends to have much less number of parameters compared
to convolutional and transformer-based models, which leads to less training and
serving computational cost.

As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
when pre-trained on large datasets, or with modern regularization schemes,
the MLP-Mixer attains competitive scores to state-of-the-art models.
You can obtain better results by increasing the embedding dimensions,
increasing the number of mixer blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
"""

"""
## The FNet model

The FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer
in the Transformer block with a parameter-free 2D Fourier transformation layer:

1. One 1D Fourier Transform is applied along the patches.
2. One 1D Fourier Transform is applied along the channels.
"""

"""
### Implement the FNet module
"""


class FNetLayer(layers.Layer):
    def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply fourier transformations.
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)


"""
### Build, train, and evaluate the FNet model

Note that training the model with the current settings on a V100 GPUs
takes around 8 seconds per epoch.
"""

# FNet
if __name__=='__main__' and do_fnet:
    fnet_blocks = keras.Sequential(
        [FNetLayer(embedding_dim, dropout_rate) for _ in range(num_blocks)]
    )
    learning_rate = 0.001
    fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
    fnet_classifier.summary()
    f1_score_callback = F1ScoreCallback((x_test, y_test))
    hist = run_experiment(fnet_classifier, f1_score_callback)

    fsymbol=f'{basename}-fnet-epoch{num_epochs}'
    myplot(fsymbol)

    fnet_classifier.save(f'{odir}/{fsymbol}.keras')
    print(f'saved {fsymbol}.keras')

"""
As shown in the [FNet](https://arxiv.org/abs/2105.03824) paper,
better results can be achieved by increasing the embedding dimensions,
increasing the number of FNet blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
The FNet scales very efficiently to long inputs, runs much faster than attention-based
Transformer models, and produces competitive accuracy results.
"""

"""
## The gMLP model

The gMLP is a MLP architecture that features a Spatial Gating Unit (SGU).
The SGU enables cross-patch interactions across the spatial (channel) dimension, by:

1. Transforming the input spatially by applying linear projection across patches (along channels).
2. Applying element-wise multiplication of the input and its spatial transformation.
"""

"""
### Implement the gMLP module
"""


class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)


    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in the shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.transpose(v, perm=(0, 2, 1))
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.transpose(v_projected, perm=(0, 2, 1))
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected


"""
### Build, train, and evaluate the gMLP model

Note that training the model with the current settings on a V100 GPUs
takes around 9 seconds per epoch.
"""
# gMLP
if __name__=='__main__' and do_gmlp:
    gmlp_blocks = keras.Sequential(
        [gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
    )
    learning_rate = 0.003
    gmlp_classifier = build_classifier(gmlp_blocks)
    gmlp_classifier.summary()
    f1_score_callback = F1ScoreCallback((x_test, y_test))
    hist= run_experiment(gmlp_classifier, f1_score_callback=f1_score_callback)
    fsymbol=f'{basename}-gmlp-epoch{num_epochs}'
    myplot(fsymbol)

    gmlp_classifier.save(f'{odir}/{fsymbol}.keras')
    print(f'saved {fsymbol}.keras')
"""
As shown in the [gMLP](https://arxiv.org/abs/2105.08050) paper,
better results can be achieved by increasing the embedding dimensions,
increasing the number of gMLP blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
Note that, the paper used advanced regularization strategies, such as MixUp and CutMix,
as well as AutoAugment.
"""