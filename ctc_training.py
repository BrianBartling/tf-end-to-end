import tensorflow as tf
import keras.backend as K
import ctc_model
import argparse
import matplotlib.pyplot as plt
import os
import random
import datetime
from time import time
import re

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
parser.add_argument('-use_model', dest='use_model', type=str, required=False, default=None, help='Load a model from an external file, continue training')
parser.add_argument('-save_after_ever_epoch', dest='save_after_every_epoch', action="store_true", default=False)
parser.add_argument('-reduce_lr_on_plateau', dest='reduce_lr_on_plateau', action="store_true", default=False)
args = parser.parse_args()


def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None 
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params


def populate_data(sample_filepath):
    sample_fullpath = corpus_dirpath + os.sep + sample_filepath + os.sep + sample_filepath

    img_filepath = sample_fullpath
    if distortions:
        img_filepath = img_filepath + '_distorted.jpg'
    else:
        img_filepath = img_filepath + '.png'

    sample_img = tf.io.read_file(img_filepath)
    if distortions:
        sample_img = tf.io.decode_jpeg(sample_img,channels=1)
    else:
        sample_img = tf.io.decode_png(sample_img,channels=1)
    sample_img = tf.image.convert_image_dtype(sample_img, tf.float32)
    img_width = tf.cast(img_height * len(sample_img[0]) / len(sample_img), tf.int32)
    model_input = tf.image.resize(sample_img, [img_height, img_width])

    label_file = sample_fullpath

    if semantic:
        label_file = label_file + '.semantic'
    else:
        label_file = label_file + '.agnostic'

    sample_gt_file = tf.io.read_file(label_file)
    stripped = tf.strings.strip(sample_gt_file)
    sample_gt_plain = tf.strings.split(stripped)

    target = tf.cast(word2int(sample_gt_plain), tf.int32)

    return {"model_input": model_input, "target": target}


word2int = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=args.voc, num_oov_indices=0, mask_token=None
)
int2word = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=word2int.get_vocabulary(), mask_token=None, invert=True
)

val_split = 0.1
batch_size = 32

# Parameterization
img_height = 128
params = default_model_params(img_height,word2int.vocabulary_size())
max_epochs = 64000
early_stopping_patience = 20
number_of_epochs_before_reducing_learning_rate = 8
learning_rate_reduction_factor = 0.5
minimum_learning_rate = 0.00001

semantic = args.semantic
distortions = False
corpus_dirpath = args.corpus

# Corpus
corpus_file = open(args.set,'r')
corpus_list = corpus_file.read().splitlines()
corpus_file.close()

# Train and validation split
random.shuffle(corpus_list) 
val_idx = int(len(corpus_list) * val_split) 
training_list = corpus_list[val_idx:]
validation_list = corpus_list[:val_idx]

print ('Training with ' + str(len(training_list)) + ' and validating with ' + str(len(validation_list)))

start_time = time()

train_dataset = tf.data.Dataset.from_tensor_slices(training_list)
train_dataset = (
    train_dataset.map(
        populate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .padded_batch(batch_size, padded_shapes = {
        'model_input': [img_height, None, 1],
        'target': [None]
    }, padding_values = {
        'model_input': 1.,
        'target': tf.constant(-1, dtype=tf.int32)
    })
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)
steps_per_epoch = len(training_list) // params['batch_size']

validation_dataset = tf.data.Dataset.from_tensor_slices(validation_list)
validation_dataset = (
    validation_dataset.map(
        populate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .padded_batch(batch_size, padded_shapes = {
        'model_input': [img_height, None, 1],
        'target': [None]
    }, padding_values = {
        'model_input': 1.,
        'target': tf.constant(-1, dtype=tf.int32)
    })
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)
validation_steps_per_epoch = len(validation_list) // params['batch_size']

# # # # _, ax = plt.subplots(2, 1, figsize=(15, 10))
# # # # for i,batch in enumerate(train_dataset.take(2)):
# # # #     image = batch["x"]
# # # #     label = batch["y"]
# # # #     ax[i].imshow(np.squeeze(batch['x'], axis=0), cmap="gray")
# # # #     ax[i].set_title(str(int2word(label).numpy()))
# # # #     ax[i].axis("off")
# # # # plt.show()

# Model
model = None
initial_epoch=0
if args.use_model:
    model = tf.keras.models.load_model(args.use_model, custom_objects={'CTCLayer': ctc_model.CTCLayer})
    m = re.match('.*?[\d-]+_ctc_model_v\d+-(\d+)\.h5', args.use_model)
    print(args.use_model, m)
    if m and m.groups() and len(m.groups()) == 1:
        initial_epoch = int(m.groups()[0])
    print("Model {0} loaded from checkpoint {1}. Training will resume from epoch {2}".format(
        model.name,
        args.use_model,
        initial_epoch+1
    ))
else:
    model = ctc_model.ctc_crnn(params)
    # Optimizer
    optimizer = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=optimizer)

print(model.summary())
#tf.keras.utils.plot_model(model, show_shapes=True)

start_of_training = datetime.date.today()

monitor_variable = 'val_loss'

best_model_path = "{0}_{1}".format(start_of_training, model.name)
if args.save_after_every_epoch:
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path + "-{epoch:02d}.h5", monitor=monitor_variable,
            save_best_only=True, verbose=1, save_freq='epoch')
else:
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path + "-{epoch:02d}.h5", monitor=monitor_variable,
            save_best_only=True, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor_variable,
                            patience=early_stopping_patience,
                            restore_best_weights=True,
                            verbose=1)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_variable,
                                                patience=number_of_epochs_before_reducing_learning_rate,
                                                verbose=1,
                                                factor=learning_rate_reduction_factor,
                                                min_lr=minimum_learning_rate)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="./logs/{0}_{1}/".format(start_of_training, model.name))

callbacks = [model_checkpoint, early_stop, tensorboard_callback]

if args.reduce_lr_on_plateau:
    callbacks.append(learning_rate_reduction)
else:
    print("Learning-rate reduction on Plateau disabled")

## Should we calculate class weights?

print("Training on dataset...")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=max_epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    validation_freq=1
)

print("Saving model to", args.save_model)
model.save(args.save_model)

end_time = time()
execution_time_in_seconds = round(end_time - start_time)
print("Execution time: {0:.1f}s".format(end_time - start_time))

## Testing...
# eval = model.evaluate(
#     validation_dataset,
#     metrics=['loss',CERMetric()]
# )
