import tensorflow as tf

def and_classifier_example():
    input_data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
    input_data = tf.cast(input_data, tf.float32)

    and_labels = tf.constant([0, 0, 0, 1])
    and_labels = tf.cast(and_labels, tf.float32)

    batch_size = 1
    epochs = 1000



    ######### SLP - Begin
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
    input_layer = tf.keras.Input(shape=[2, ])
    output = tf.keras.layers.Dense(units=1,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(input_layer)
    slp_classifier = tf.keras.Model(inputs=input_layer, outputs=output)
    slp_classifier.compile(optimizer=sgd, loss="mse")

    slp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs)
    ######### SLP - End


    ######### MLP - Begin
    input_layer2 = tf.keras.Input(shape=[2, ])
    hidden_layer2 = tf.keras.layers.Dense(units=4,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(input_layer2)
    output2 = tf.keras.layers.Dense(units=1,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(hidden_layer2)
    mlp_classifier = tf.keras.Model(inputs=input_layer2, outputs=output2)
    sgd2 = tf.keras.optimizers.SGD(learning_rate=0.1)
    mlp_classifier.compile(optimizer=sgd2, loss="mse")
    mlp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs)
    ######### MLP - End


    ######## SLP AND prediciton
    prediction = slp_classifier.predict(x=input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction)
    print("====== SLP AND classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d AND %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d AND %d => %.2f => 0" % (x[0], x[1], y))

    ######## MLP AND prediciton
    prediction = mlp_classifier.predict(x=input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction)
    print("====== MLP AND classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d AND %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d AND %d => %.2f => 0" % (x[0], x[1], y))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    and_classifier_example()

