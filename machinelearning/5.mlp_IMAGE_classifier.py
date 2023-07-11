#이미지를 가지고 label를 예측 분류하는 classifi 100개 중에 50개 맞추는 classifier

#data 받고 4시 45분

import tensorflow as tf

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])

    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

input_layer = tf.keras.Input(shape=[32, 32, 3, ])                                           # 3차원 이미지
hidden_layer = tf.keras.layers.Flatten()(input_layer)

hidden_layer = hidden_layer = tf.keras.layers.Dense(512, activation='relu')(hidden_layer)
hidden_layer = hidden_layer = tf.keras.layers.Dense(256, activation='relu')(hidden_layer)
hidden_layer = hidden_layer = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(564, activation='relu')(hidden_layer)
output = tf.keras.layers.Dense(10, activation='softmax')(hidden_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.summary()

model.compile(optimizer='adam',
              loss = tf.keras.losses.SqarseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images,test_labels))

plt.plot(history.histroy['accuracy'], label='accuracy')
plt.plot(history.histroy['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuarcy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

### Prediction
predicted_labels = model.predict(test_images)
predcited_labels = tf.math.argmax(input=predicted_labels, axis=1)

print("Prediction")
plt.figure(figsize=(10,10))
for i in range(25):
    plt.shbplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[predicted_labels[i]])
plt.show()
