import tensorflow as tf
import keras.backend as K

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * tf.cast(input_shape[1], 'float32')

        tf.print("y_pred:", y_pred, "input_length:", input_length)
        decode, _ = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        tf.print("decode:", decode)

        decode = K.ctc_label_dense_to_sparse(decode[0], tf.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, tf.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.cast(tf.reduce_sum(distance), "float32"))
        self.counter.assign_add(tf.cast(len(y_true), "float32"))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
        self.accuracy_fn = CERMetric()

    def call(self, y_true, y_pred, sample_weight=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int32")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int32")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int32")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        acc = self.accuracy_fn(y_true, y_pred, sample_weight)
        self.add_metric(acc, name="accuracy")

        # At test time, just return the computed predictions
        return y_pred


def ctc_crnn(params):
    input = tf.keras.layers.Input(
        shape=(params['img_height'], params['img_width'], params['img_channels']), # [batch, height, width, channels]
        name="model_input", dtype=tf.float32
    )
    targets = tf.keras.layers.Input(name="target", shape=(None,), dtype=tf.int32)

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):

        x = tf.keras.layers.Conv2D(
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv" + str(i+1),
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size = params['conv_pooling_size'][i],
            strides = params['conv_pooling_size'][i]
        )(x)

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    # Prepare output of conv block for recurrent blocks
    feature_width = params['img_width']
    if feature_width:
        feature_width = feature_width // width_reduction
    else:
        feature_width = -1
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
#    features = tf.transpose(x, perm=[2, 0, 3, 1]) # -> [width, batch, height, channels] (time_major=True)
    #features = tf.transpose(x, perm=[])
    target_shape = (feature_width, feature_dim)
    features = tf.keras.layers.Reshape(target_shape = target_shape, name="reshape")(x) # not time major

    # Recurrent block
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    features = tf.keras.layers.Dense(64, activation="relu", name="dense1")(features)
    features = tf.keras.layers.Dropout(0.2)(features)

    # forward_layer = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(rnn_hidden_units) for _ in range(rnn_hidden_layers)])
    # forward_layer = tf.keras.layers.RNN(forward_layer, return_sequences=True)
    # backward_layer = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(rnn_hidden_units) for _ in range(rnn_hidden_layers)])
    # backward_layer = tf.keras.layers.RNN(backward_layer, return_sequences = True, go_backwards=True)

    # rnn_outputs = tf.keras.layers.Bidirectional(forward_layer,
    #                                             backward_layer = backward_layer,
    #                                             dtype=tf.float32)(features)

    rnn_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(features)
    rnn_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(rnn_outputs)
    rnn_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(rnn_outputs)

    logits = tf.keras.layers.Dense(params['vocabulary_size']+2, activation="softmax", name="dense2")(rnn_outputs)

    # CTC Loss computation
    output = CTCLayer(name="ctc_loss")(targets, logits)

    model = tf.keras.models.Model(
        inputs=[input, targets], outputs=output, name="ctc_model_v1"
    )

    return model