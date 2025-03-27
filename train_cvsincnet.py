import os
import numpy as np
import tensorflow as tf
from complex_valued_layers import sincnet_layer_ri_efficient_1D as snc1  # or rename if needed
from complex_valued_layers import complexconv as cnn
from complex_valued_layers import complexpool as pool
from complex_valued_layers import complexdense as cvdense
from complex_valued_layers import complexactivation as Cactivation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model
import seaborn as sns
import h5py

BATCH = 16
EPOCH = 150

data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/AI based Raw radar data/Sabya_Data/cascade Rad/whole_data/data_ASL_training_2.h5", 'r')
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

data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/AI based Raw radar data/Sabya_Data/cascade Rad/whole_data/data_ASL_testing_2.h5", 'r')
xr = data['dataset_real']
xi = data['dataset_imag']
y = data['dataset_label']

xr = np.squeeze(np.asarray(xr))
xi = np.squeeze(np.asarray(xi))
xr = np.expand_dims(xr, axis=2)
xi = np.expand_dims(xi, axis=2)
test_x = np.concatenate((xr, xi), axis=2)
y = np.asarray(y)
y = np.expand_dims(y, axis=0)
test_y = np.transpose(y)

input_data = tf.keras.Input(shape=training_x.shape[1:])
num_filters = 256

def complex_conv(input, conv_filter, kernel_size, max_pool, dropout):
    x = cnn.ComplexConv1D(conv_filter, kernel_size, padding="valid")(input)
    x = pool.ComplexMaxPooling1D(max_pool, max_pool)(x)
    x = Cactivation.complex_bn(x)
    x = Cactivation.CReLU(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def complex_plf(input,fn,fl,fs,max_pool):
    x = snc1.SincNetLayer1D(fn, fl, fs)(input)
    x = pool.ComplexMaxPooling1D(max_pool, max_pool)(x)
    x = Cactivation.complex_ln(x)
    x = Cactivation.CReLU(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    return x

x = complex_plf(input_data,num_filters,251,3200,4)

x = complex_conv(x, 128, 5, 3, 0.2)
x = complex_conv(x, 128, 5, 3, 0.2)
x = complex_conv(x, 128, 5, 3, 0.2)
x = complex_conv(x, 128, 5, 3, 0.2)
x = complex_conv(x, 128, 5, 3, 0.2)
x = complex_conv(x, 128, 5, 3, 0.2)

flatten = Cactivation.complex_flatten(x)
dense = cvdense.ComplexDense(256)(flatten)
dropout = tf.keras.layers.Dropout(0.2)(dense)
relu_dense = Cactivation.CReLU(dropout) 

dense2 = cvdense.ComplexDense(100)(relu_dense)
output_layer = Cactivation.complex_softmax(dense2)

model = tf.keras.Model(inputs=input_data, outputs=output_layer)
print(model.summary())
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

save_dir = "C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/AI based Raw radar data/Sabya_Data/cascade Rad/whole_data_results/100_class/models/CVGauss2D_v2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5  == 0:
            accuracy = logs['val_accuracy']
            filename = os.path.join(save_dir, f"model_epoch_{epoch+1:02d}_acc_{accuracy:.3f}")
            self.model.save(filename)
            print(f"\nModel weights saved to {filename}")

history = model.fit(
    x=training_x,
    y=training_y,
    batch_size=BATCH,
    epochs=EPOCH,
    validation_data=(test_x, test_y),
    callbacks=[CustomModelCheckpoint()]
)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(test_x)

pred_y = []
for i in range(len(y_pred)):
    list_y = list(y_pred[i])
    a = list_y.index(max(list_y))
    pred_y.append(a)
    
cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig('.//results//examples//confusion_matrix.png')
plt.show()

cf = []
for i in range(len(cf_matrix)):
    cf.append(cf_matrix[i].astype(np.float64) / sum(cf_matrix[i].astype(np.float64)))
    
ax = sns.heatmap(cf, annot=True, cmap='Blues')
plt.show()
