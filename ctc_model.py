import tensorflow as tf

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

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

    forward_layer = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(rnn_hidden_units) for _ in range(rnn_hidden_layers)])
    forward_layer = tf.keras.layers.RNN(forward_layer, return_sequences=True)
    backward_layer = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(rnn_hidden_units) for _ in range(rnn_hidden_layers)])
    backward_layer = tf.keras.layers.RNN(backward_layer, return_sequences = True, go_backwards=True)

    rnn_outputs = tf.keras.layers.Bidirectional(forward_layer,
                                                backward_layer = backward_layer,
                                                dtype=tf.float32)(features)

    logits = tf.keras.layers.Dense(params['vocabulary_size']+2, activation="softmax", name="dense2")(rnn_outputs)

    # CTC Loss computation
    output = CTCLayer(name="ctc_loss")(targets, logits)

    model = tf.keras.models.Model(
        inputs=[input, targets], outputs=output, name="ctc_model_v1"
    )

    return model