
from keras.models import load_model
from keras.models import Model
import cv2
import keras
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import spline
from PIL import Image
# 引入Tensorboard
from keras.callbacks import TensorBoard


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

# plot the filters
def plot_filters(layer, x , y , l):
    x = 8
    y = 8
    filters = layer.get_weights()[0][:,:,:,:]
    # print(filters.shape[3])
    fig = plt.figure()
    for j in range(64):
        # ax = fig.add_subplot(y,x,j+1)
        ax = fig.add_subplot(y,x,j+1)
        # print('filter', filters[:,:,0,j].shape)
        ax.matshow(filters[:,:,0,j], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    fig_filter = plt.gcf()
    path_image = path_dir+'/filter_layer'+str(l)+'.png'
    fig_filter.savefig(path_image,dpi=600)
    plt.show()
    return plt

# #读取图片
def load_data(data_path, img_rows, img_cols):
    class_names = []
    data = []
    labels = []
    label_index = 0
    data_new = []
    label_new = []
    for label_name in os.listdir(data_path):
        if os.path.splitext(label_name)[0] != 'Thumbs':
            class_names.append(label_name)
            for image_name in os.listdir(data_path + '/' + label_name):
                image = cv2.imread(data_path + '/' + label_name +'/' + image_name)
                if image is not None:
                    # print('image',image.shape)
                    image = cv2.resize(image, (img_rows, img_cols))
                    data.append(image)
                    labels.append(label_index)
            label_index = label_index + 1
    length_data = range(len(data))
    labels = keras.utils.to_categorical(labels, len(class_names))
    print('class_number:', len(class_names), 'class names:', str(class_names))
    data = np.array(data)
    data = data.astype('float32')
    return data, labels, len(class_names), class_names

img_rows = 224
img_cols = 224

# model_path = 'model/my_mode_semi_alexnet.h5'
data_path = "C:\\image_classification\\mission_3\\data_base_non_incubation\\larger_than_100000\\1001_CHROM_ORI_C00000069105_3_TOP_BLACK_24.4h.jpg"
path_image = 'C:\\image_classification\\mission_3\\image_save\\vgg16'
path_dir = 'visual-layer_vgg16'
mkdir(path_dir)
# data, labels, length, class_names = load_data(data_path, img_rows, img_cols)

# model_dir = input('model: ')
# 'model/my_mode_semi.h5'
# load the alexnet model
# model = load_model('model/my_mode_semi.h5')
# model.summary()

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16

# include_top means whether to include the fully-connected layer at the top of the network. True: add fc layer, False: no
model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

model.summary()

# Get the whole configuration
model.get_config()

# Get the configuration of the first layer
model.layers[0].get_config()
# Get the parameters
model.count_params()
# Get all the parameters of the first layer
model.layers[0].count_params()
# Get the weights of the first layer
# <tf.Variable 'conv2d_1/kernel:0' shape=(11, 11, 3, 96) dtype=float32_ref>

# plot the first convolution layer filters
l = input('N.layer: ')

layer = model.layers[int(l)]
print(layer.get_weights())
print(layer.get_weights()[0].shape)
filters = layer.get_weights()[0][:,:,0,:]

print(filters.shape[2])
x = 8
y = int(filters.shape[2]//x)
plot_filters(layer, x, y, l)

# visualizing intermediate layers
# input image
input_image = plt.imread(data_path)
input_image = cv2.resize(input_image, (img_rows, img_cols))
# input_image = x_train[0:1, :, :, :]
print('input_shape', input_image.shape)
plt.imshow(input_image[:, :, 0], cmap='gray')
fig_grey = plt.gcf()
path_grey = path_dir+'/grey' + '.png'
fig_grey.savefig(path_grey, dpi=600)
print(input_image[:, :, 0].shape)
plt.show()
plt.imshow(input_image[:, :, 0])
plt.show()


# redefine model to output right after the first hidden layer
ixs = np.arange(1,17)
print(ixs)
outputs = [model.layers[int(i)].output for i in ixs]
print('out',np.array(outputs).shape)
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = image.load_img(data_path, target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
print('img',img.shape)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block


i = 1
for fmap in feature_maps:
    print('fmap', fmap.shape)
    print(np.sqrt(fmap.shape[3]))
    square_cols = int(np.sqrt(fmap.shape[3]))
    square_rows = int(fmap.shape[3]/square_cols)
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square_rows):
        for _ in range(square_cols):
            ax = plt.subplot(square_rows, square_cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            # plt.imshow(fmap[0, :, :, ix - 1], cmap='gray') if ixs is a list
            # plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            plt.imshow(fmap[0, :, :, ix - 1])
            ix += 1
    # show the figure
    fig_output = plt.gcf()
    path_image = path_dir + '/output_layer' + str(i) + '.png'
    fig_output.savefig(path_image, dpi=600)
    plt.show()
    i += 1






