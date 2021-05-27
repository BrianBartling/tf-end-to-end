import argparse
import tensorflow as tf
import cv2
import numpy as np
from ctc_model import CTCLayer
#from ctc_training import img_height

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

img_height = 128

# Read the dictionary
# dict_file = open(args.voc_file,'r')
# dict_list = dict_file.read().splitlines()
# int2word = dict()
# for word in dict_list:
#     word_idx = len(int2word)
#     int2word[word_idx] = word
# dict_file.close()
word2int = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=args.voc, num_oov_indices=0, mask_token=None
)
int2word = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=word2int.get_vocabulary(), mask_token=None, invert=True
)

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(args.model, custom_objects={'CTCLayer': CTCLayer})

# Show the model architecture
print(model.summary())

sample_img = tf.io.read_file(args.image)
sample_img = tf.io.decode_png(sample_img,channels=1)
sample_img = tf.image.convert_image_dtype(sample_img, tf.float32)
img_width = tf.cast(img_height * len(sample_img[0]) / len(sample_img), tf.int32)
model_input = tf.image.resize(sample_img, [img_height, img_width])
model_input = tf.expand_dims(model_input, axis=0)

# Get the prediction model by extracting layers till the output layer
prediction_model = tf.keras.models.Model(
    model.get_layer(name="model_input").input, model.get_layer(name="dense2").output
)
print(prediction_model.summary())

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][0]
    print("results:", results)
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(int2word(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

preds = prediction_model.predict(model_input)
pred_texts = decode_batch_predictions(preds)
print(pred_texts)
# # Restore weights
# saver = tf.train.import_meta_graph(args.model)
# saver.restore(sess,args.model[:-5])

# graph = tf.get_default_graph()

# input = graph.get_tensor_by_name("model_input:0")
# seq_len = graph.get_tensor_by_name("seq_lengths:0")
# rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
# height_tensor = graph.get_tensor_by_name("input_height:0")
# width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
# logits = tf.get_collection("logits")[0]

# # Constants that are saved inside the model itself
# WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

# decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

# image = cv2.imread(args.image,False)
# image = ctc_utils.resize(image, HEIGHT)
# image = ctc_utils.normalize(image)
# image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

# seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

# prediction = sess.run(decoded,
#                       feed_dict={
#                           input: image,
#                           seq_len: seq_lengths,
#                           rnn_keep_prob: 1.0,
#                       })

# str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
# for w in str_predictions[0]:
#     print (int2word[w]),
#     print ('\t'),
