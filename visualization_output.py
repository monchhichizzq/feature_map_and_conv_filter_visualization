

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



# plot the filters
def plot_filters(layer, x, y):
    filters = layer.get_weights()[0][:,:,:,:]
    # print(filters.shape)
    # print(filters)
    # print(filters.shape[3])
    fig = plt.figure()
    for j in range(filters.shape[3]):
        # ax = fig.add_subplot(y,x,j+1)
        ax = fig.add_subplot(y,x,j+1)
        print('filter', filters[:,:,0,j].shape)
        ax.matshow(filters[:,:,0,j], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    plt.show()
    fig_filter = plt.gcf()
    # # for j in range(len(filters)):
    #     ax = fig.add_subplot(y,x,j+1)
    #     ax.matshow(filters[j][0], cmap=matplotlib.cm.binary)
    #     plt.xticks(np.array([]))
    #     plt.yticks(np.array([]))
    # plt.tight_layout()
    # fig_filter = plt.gcf()
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

img_rows = 227
img_cols = 227

model_path = 'model/my_mode_semi_alexnet.h5'
data_path = "C:\\Users\\10289005\\OneDrive - BD\\Desktop\\image_classification\\RollingBeadModel"
path_image = 'C:\\Users\\10289005\\OneDrive - BD\\Desktop\\image_classification\\image_save\\sq\\sq_alexnet50'
data, labels, length, class_names = load_data(data_path, img_rows, img_cols)

# x为数据集的feature熟悉，y为label.
X_train, X_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

# data normalization
x_train_rows = X_train.reshape(X_train.shape[0], img_rows * img_cols * 3)
x_test_rows = X_test.reshape(X_test.shape[0], img_rows * img_cols * 3)

minmax = MinMaxScaler()

x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)
# print('minmax',x_train_rows, x_test_rows)

# convert to 32 x 32 x 3
x_train = x_train_rows.reshape(x_train_rows.shape[0], img_rows, img_cols, 3)
x_test = x_test_rows.reshape(x_test_rows.shape[0], img_rows, img_cols, 3)
# print('x_train',x_train.shape, 'x_test',x_test.shape)

y_train = labels_train
y_test = labels_test
# print('y_train',y_train.shape,'y_test',y_test.shape)

# train, val, test data split
train_ratio = 0.8
x_train_, x_val, y_train_, y_val = train_test_split(x_train,y_train,train_size=train_ratio,random_state=123)

print('x_train_',x_train_.shape, 'y_train_',y_train_.shape)
print('x_val',x_val.shape, 'x_val',y_val.shape)
print('x_test',x_test.shape,'y_test',y_test.shape)



# load the alexnet model
model = load_model('model/my_mode_semi.h5')
print('test after load: ', model.predict(x_test[0:2]))
test_loss, test_acc = model.evaluate(x_test, y_test)

# Get the whole configuration
model.get_config()
print(model.get_config())
# Get the configuration of the first layer
print(model.layers[0].get_config())
# Get the parameters
print(model.count_params())
# Get all the parameters of the first layer
print(model.layers[0].count_params())
# Get the weights of the first layer
# <tf.Variable 'conv2d_1/kernel:0' shape=(11, 11, 3, 96) dtype=float32_ref>
print(model.layers[0].kernel)
# 3 channels for colorful image
# channel 0
print(model.layers[0].get_weights()[0].shape)
print(model.layers[0].get_weights()[0][:,:,0,:].shape)
filters = model.layers[0].get_weights()[0][:,:,0,:]
# Get the bias if the first layer
# model.layers[0].bias.get_value()

# plot the first convolution layer filters
layer = model.layers[0]
x = 12
y = 8
# plot_filters(layer, x, y)

# visualizing intermediate layers
# output_layer = model.layers[0].output
# print(output_layer)
# output_fn = theano.function([model.layers[0].input], output_layer)

# input image
input_image = x_train[0:1, :, :, :]
print('input_shape', input_image.shape)
plt.imshow(input_image[0, :, :, 0], cmap='gray')
plt.show()
plt.imshow(input_image[0, :, :, 0])
plt.show()

# output_image = output_fn(input_image)
# print(output_image.shape)
# # shit the collumn/ rearrange dimenson so we can plot the result as RGB images
# output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
# print(output_image.shape)


# load the model
model = load_model('model/my_mode_semi.h5')
model.summary()
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = x_train
# convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(x_train)
# plot the output from each block
square = 8


for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()



