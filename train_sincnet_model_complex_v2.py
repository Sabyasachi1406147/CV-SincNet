"""
Created on Fri Aug  5 10:28:27 2022

@authors: Sabyasachi Biswas, Cemre Ömer Ayna
"""
import numpy as np
import tensorflow as tf
import sincnet_layer_edited_ri_v3 as snc1
#import top_filter_sinc as snc1
#import sincnet_layer_v2 as snc1
import complexconv as cnn
import complexpool as pool
import complexdense as cvdense
import complexactivation as Cactivation
import matplotlib.pyplot as plt 
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay

print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

BATCH = 16
EPOCH = 125

data = h5py.File(".\\whole_data\\data_ASL_training.h5", 'r')
xr = data['dataset_real']
xi = data['dataset_imag']
y = data['dataset_label']

xr = np.squeeze(np.asarray(xr))
xi = np.squeeze(np.asarray(xi))
xr = np.expand_dims(xr, axis=2)
xi = np.expand_dims(xi, axis=2)
training_x = np.concatenate((xr, xi), axis=2)
y = np.asarray(y)
y = np.expand_dims(y, axis=0)
training_y = np.transpose(y)

data = h5py.File(".\\whole_data\\data_ASL_testing.h5", 'r')
xr = data['dataset_real']
xi = data['dataset_imag']
y = data['dataset_label']

xr = np.squeeze(np.asarray(xr))
xi = np.squeeze(np.asarray(xi))
xr = np.expand_dims(xr, axis=2)
xi = np.expand_dims(xi, axis=2)
test_x = np.concatenate((xr, xi), axis=2)
# z = test_x.view(dtype=np.complex128)
# xp = np.angle(z)
# test_x = np.concatenate((test_x, xp), axis=2)
y = np.asarray(y)
y = np.expand_dims(y, axis=0)
test_y = np.transpose(y)

# shuffle_list = list(range(xr.shape[0]))

# # val_x, test_x, val_y, test_y = train_test_split(
# #     test_x, test_y, test_size=0.90, random_state=20)

input_data = tf.keras.Input(shape=training_x.shape[1:])

def complex_conv(input, conv_filter, kernel_size, max_pool, dropout):
    x = cnn.ComplexConv1D(conv_filter, kernel_size, padding = "same")(input)
    x = cnn.ComplexConv1D(conv_filter, kernel_size, padding = "same")(x)
    x = pool.ComplexMaxPooling1D(max_pool,max_pool)(x)
    x = Cactivation.complex_bn(x)
    x = Cactivation.CReLU(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

x = snc1.SincNetLayer1D(256, 251, 3200)(input_data)
x = pool.ComplexMaxPooling1D(3,3)(x)
x = Cactivation.complex_ln(x)
x = Cactivation.CReLU(x)
x = tf.keras.layers.Dropout(0.2)(x)

x1 = complex_conv(x,128,3,3,0.2)
x2 = complex_conv(x,128,5,3,0.2)
x3 = complex_conv(x,128,7,3,0.2)

x = snc1.ComplexConcatenate()([x1, x2, x3])

x = complex_conv(x,128,5,3,0.2)
x = complex_conv(x,128,5,3,0.2)
x = complex_conv(x,128,5,3,0.2)
x = complex_conv(x,128,5,3,0.2)
x = complex_conv(x,128,5,3,0.2)
x = complex_conv(x,128,5,2,0.2)

flatten = Cactivation.complex_flatten(x)
dense = cvdense.ComplexDense(256)(flatten)
dropout = tf.keras.layers.Dropout(0.2)(dense)
relu_dense = Cactivation.CReLU(dropout) 

dense2 = cvdense.ComplexDense(100)(relu_dense)
output_layer = Cactivation.complex_softmax(dense2)

model = tf.keras.Model(inputs=input_data, outputs=output_layer)
print(model.summary())
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=["accuracy"])

# initial_learning_rate = 0.01
# def lr_exp_decay(epoch, lr):
#     k = 0.1
#     return initial_learning_rate * math.exp(-k*epoch)

# callback = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay,verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# history=model.fit(x=training_x,
#         y=training_y,
#         batch_size=BATCH,
#         epochs=EPOCH,
#         validation_data=(X_test, y_test),
#         steps_per_epoch=ceil(training_x.shape[0]/BATCH))

history=model.fit(x=training_x,
        y=training_y,
        batch_size=BATCH,
        epochs=EPOCH,
        validation_data=(test_x, test_y))

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

y_pred = model.predict(test_x)

#plotting confusion_matrix#####################################################

pred_y = []
for i in range(len(y_pred)):
    list_y = list(y_pred[i])
    a = list_y.index(max(list_y))
    pred_y.append(a)
    
cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)

import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig('.//results//examples//confusion_matrix.png')
plt.show()

cf = []
for i in range(len(cf_matrix)):
    cf.append(cf_matrix[i].astype(np.float64)/sum(cf_matrix[i].astype(np.float64)))
    
ax = sns.heatmap(cf, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix in %\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig('.//results//examples//confusion_matrix%.png')
plt.show()

##################################################

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))

plt.plot(epochs, train_acc, 'r', label='Training acc',linewidth=2)
#plt.plot(epochs, train_acc, 'r', linewidth=1)
plt.plot(epochs, val_acc, 'b', label='Validation acc',linewidth=2)
#plt.plot(epochs, val_acc, 'b', linewidth=1)
plt.title('Training and Validation Accuracy',fontsize=14)
plt.ylim([0, 1])
plt.ylabel('Accuracy',fontsize=14) 
plt.xlabel('Epoch',fontsize=14)
plt.legend()
plt.show()
plt.savefig('.//results//examples//train_vs_val.png')

plt.plot(epochs, train_loss, label='Training loss',linewidth=2)
plt.plot(epochs, val_loss, label='validation Loss',linewidth=2)
plt.title('Training and Validation Losses',fontsize=14)
plt.ylabel('Loss',fontsize=14) 
plt.xlabel('Epoch',fontsize=14)
plt.legend()
plt.show()
plt.savefig('.//results//examples//train_vas_val_loss.png')
#plt.ylim([0, 10])




model.save(".\\models\\examples\\fixedsinc_32")
model.save_weights('.\\models\\examples\\fixedsinc_32.h5')
# filters = model.layers[1].get_weights()
# filters2 = model.layers[14].get_weights()
# # model2 = tf.keras.Model(inputs=input_data, outputs=output_layer)
# # model2.load_weights('.\\models\\examples\\sincnet_weights.h5')