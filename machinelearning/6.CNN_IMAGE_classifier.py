# CNN을 사용하여 이미지 처리
# MLP모델(100개 중에 50개 맞춤)의 문제점에서 시작된 CNN
# MLP문제점 1) Flatten이 있다(평탄화) 1차원으로 가정하여 학습하기 떄문에 3차원을 1차원으로 시킴(공간 정보가 사라진다), local 정보가 유실 된다. 
# MLP문제점 2) 32x32x3 image -> 3072x1(layer 2065,1024,225,128,64) layer가 너무 많아진다. parameter 줄이기

# Convolution Neural Network 
# 이미지에서 특징을 추출하기 위해 사용된다.
# 특징을 뽑아 냄과 동시에 차원을 축소하는
# 커널 필터(knerl, filter) : 차원 축소 기여 / pooling
# filter 이미지 보다 작은 32 x 32 필터 하나는 4x4 일 경우 값들이 들어가 있다. 필터를 이미지 위치에 배치하여 / 이미지 값들과 픽셀값을 곱해서 4x4 값을 만들어 냄
# 4x4 image 3x3 filter 사용 시
# 학습하는 것이 무엇이냐는 필터들의 값만 학습함 
# 필터를 여러 개 사용(하나의 특징당 필터 하나가 검출하기 때문)
# 차원을 줄이고 그거슬 1차원으로 한 후에 그것을 MLP에 넣는다

import tensorflow as tf

# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

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

input_layer = tf.keras.Input(shape=[32, 32, 3, ])
###### MLP랑 다른 점
hidden_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
hidden_layer = tf.keras.layers.MaxPooling2D((2, 2))(hidden_layer)
hidden_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.MaxPooling2D((2, 2))(hidden_layer)
hidden_layer = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(hidden_layer)
output = tf.keras.layers.Dense(10, activation='softmax')(hidden_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


####### Prediction
predicted_labels = model.predict(test_images)
predicted_labels = tf.math.argmax(input=predicted_labels, axis=1)

print("Prediction")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[predicted_labels[i]])
plt.show()

