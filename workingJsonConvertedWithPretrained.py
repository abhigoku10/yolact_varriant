import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
import pandas as pd
import tensorflow_addons as tfa
import os
import cv2
import random
from sklearn.preprocessing import LabelBinarizer

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}

@tf.keras.utils.register_keras_serializable()
class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x

@tf.keras.utils.register_keras_serializable()
class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output

@tf.keras.utils.register_keras_serializable()
class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # print("True")
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # else:
        #     print("False")
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def buildModel(model_name='swin_tiny_patch4_window7_224', include_top=False,
                img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=LayerNormalization, patch_norm=True,
                use_checkpoint=False, **kwargs):

    patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
    numLayers = len(depths)
    numFeatures = int(embed_dim*2 ** (numLayers - 1))
    dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]
    x = tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3))
    y = PatchEmbed(img_size = img_size, patch_size = patch_size,\
            in_chans = in_chans, embed_dim = embed_dim, norm_layer = norm_layer if patch_norm else None)(x)
    y = Dropout(drop_rate)(y)
    y = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                patches_resolution[1] // (2 ** i_layer)),
                                            depth=depths[i_layer],
                                            num_heads=num_heads[i_layer],
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                depths[:i_layer + 1])],
                                            norm_layer=norm_layer,
                                            downsample=PatchMerging if (
                                                i_layer < numLayers - 1) else None,
                                            use_checkpoint=use_checkpoint,
                                            prefix=f'layers{i_layer}') for i_layer in range(numLayers)])(y)
    y = norm_layer(epsilon=1e-5, name='norm')(y)
    y = GlobalAveragePooling1D()(y)
    y = Dense(1000, activation='softmax')(y)
    
    return tf.keras.models.Model(inputs=x, outputs=y, name=model_name)

class Augmentor:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.shift_ratio = 20
        self.brightness_range = tuple((0.5, 3))
        self.crop_value = 0.7
        self.channel_shift_range = 30
        self.rotate_angle = 5
        self.aug_list = [self.verticalFlip, self.horizontalFlip, self.horizontalShift, \
                        self.verticalShift, self.brightness, self.centerCrop, self.channelShift]
    
    def fill(self, image):
        return cv2.resize(image, self.image_shape, cv2.INTER_CUBIC)
    
    def horizontalShift(self, image):
        width = random.randint(-self.shift_ratio, self.shift_ratio)
        if width > 0:
            return image[:, :int(self.image_shape[1] - width), :]
        elif width < 0:
            return image[:, int(-1 * width):, :]
        else:
            return image
    
    def verticalShift(self, image):
        height = random.randint(-self.shift_ratio, self.shift_ratio)
        if height > 0:
            return image[:int(self.image_shape[0] - height), :, :]
        elif height < 0:
            return image[int(-1 * height):, :, :]
        else:
            return image
    
    def brightness(self, img):
        value = random.uniform(self.brightness_range[0], self.brightness_range[1])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def centerCrop(self, img):
        value = random.uniform(self.crop_value, 1)
        img = self.fill(img)
        pad_remove_ht = self.image_shape[0] - int(self.image_shape[0] * value)
        pad_remove_wd = self.image_shape[1] - int(self.image_shape[1] * value)
        return img[pad_remove_ht: -pad_remove_ht, pad_remove_wd: -pad_remove_wd, :]
        
    def channelShift(self, img):
        value = int(random.uniform(-self.channel_shift_range, self.channel_shift_range))
        img = img + value
        img[:,:,:][img[:,:,:]>255]  = 255
        img[:,:,:][img[:,:,:]<0]  = 0
        img = img.astype(np.uint8)
        return img
    
    def horizontalFlip(self, img):
        return cv2.flip(img, 1)
    
    def verticalFlip(self, img):
        return cv2.flip(img, 0)
    
    def rotateImage(self, img):
        angle = int(random.uniform(-self.rotate_angle, self.rotate_angle))
        M = cv2.getRotationMatrix2D((int(self.image_shape[1]/2), int(self.image_shape[0]/2)), angle, 1)
        img = cv2.warpAffine(img, M, (self.image_shape[1], self.image_shape[0]))
        return img


    def __call__(self, image):
        image = random.choice(self.aug_list)(image)
        return self.fill(image)
# from augment import Augmentor

def split_dataset(df, ratio, split):
    np.random.seed(0)
    classes = df["Class"].unique()
    indexes = []
    for j in classes:
        
        current_indexs = df.index[df['Class'] == j].tolist()
        if split == "train":
            np.random.shuffle(current_indexs)
            select_indexes = current_indexs[:int(len(current_indexs) * ratio)]
        if split == "valid":
            np.random.shuffle(current_indexs)
            select_indexes = current_indexs[int(len(current_indexs) * ratio):]
        indexes.extend(select_indexes)
    return indexes


class LoadDataset(tf.keras.utils.Sequence):
    def __init__(self, root, batch_size, ratio = 0.8, image_size = (224, 224),split = "train"):
        self.train_val_ratio = ratio
        self.split = split
        self.image_size = image_size
        self.root = root
        self.df = pd.read_csv(os.path.join(self.root, "Data.csv"))
        self.classes = np.array(self.df["Class"].unique())
        self.indexes = split_dataset(self.df, self.train_val_ratio, self.split)
        self.batchSize = batch_size
        self.imageSize = (224, 224)
        self.channels = 3
        self.augment = Augmentor(image_shape=(224, 224))
        self.lb = LabelBinarizer()
        self.lb.fit(self.classes)
        self.shuffle = True

        # Other initializations
        self.on_epoch_end()
    
    def on_epoch_end(self):
        '''Update indexes after each epoch.'''
        # self.indexes = np.arange(len(self.indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_image(self, idx):
        img_path = os.path.join(self.root, self.df["Path"][self.indexes[idx]])
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        if self.split == 'train':
            img = self.augment(img)
        else:
          img = cv2.resize(img, self.image_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0
        # img = np.transpose(img, (2, 0, 1))
        return img
    
    def get_class(self, idx):
        return np.where(self.classes == self.df["Class"][self.indexes[idx]])[0][0]
        # return self.df["Filename"][self.indexes[idx]]
    
    def __len__(self):
        return int(self.indexes.__len__() // self.batchSize)
    """ 
    def __getitem__(self, index):
        img = self.get_image(index)
        class_ = self.get_class(index)
        return img, class_
     """
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batchSize : (index+1) * self.batchSize]
        X = np.empty((self.batchSize, self.image_size[0], self.image_size[1], self.channels))
        y = []
        for index, k in enumerate(indexes):
            X[index,:] = self.get_image(index)
            y.append(self.get_class(index))
        y = tf.keras.utils.to_categorical(y, len(self.classes))

        return X, y

    """ def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.img_height, self.img_width, self.channels))
        Y = np.empty((self.batch_size, 5))
        Z = np.empty((self.batch_size, 2))
        lb = LabelBinarizer()
        lb.fit(self.classes)

        for i, ID in enumerate(list_IDs_temp):
            labels = []
            bbox_yaw = []


            img = tf.keras.preprocessing.image.load_img(os.path.join(self.image_path,ID[:-1] + '.png'))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img /= 255.0
            X[i,] = img
            with open(os.path.join(self.label_path, ID[:-1] + '.txt'), 'r') as f:
                for line in f:
                    label, x, y, w, h, yaw = line.split(',')
                    bbox_yaw.append([float(x), float(y), float(w), float(h), float(yaw)])
                    labels.append(label)
            labels = lb.transform(labels)
            Y[i,] = math.sqrt(float(x)), math.sqrt(float(y)), math.sqrt(float(w)), math.sqrt(float(h)), (float(yaw)+np.pi)/ (2*np.pi)
            if len(self.classes) == 2:
                labels = tf.keras.utils.to_categorical(labels, num_classes=2)
            Z[i,:] = labels


        return X, [Z, Y]
 """

if __name__ == "__main__":
    learning_rate = 1e-3
    batch_size = 64
    weight_decay = 0.0001
    label_smoothing = 0.1
    model_name = "swin_tiny_224"
    cfg = CFGS[model_name]
    model = buildModel(
        model_name=model_name, include_top=False, num_classes=1000, img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
    )
    trainDataset = LoadDataset(root = "/content/StanfordDogs", batch_size=batch_size, split='Train')
    validDataset = LoadDataset(root = "/content/StanfordDogs", batch_size=batch_size, split='Valid')
    
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit(trainDataset,
                     validation_data=validDataset,
                     use_multiprocessing=False,
                     workers=6,
                     epochs=100,
                     callbacks = [mc])

    
    