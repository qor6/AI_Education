import tensorflow as tf

def and_classifier_example():
    input_data = tf.constant([[0,0], [0,1], [1,0], [1,1]])                                  # tf.constant(상수)는 tensor 3차원이상 배열 구조 자료 구조 함수 / 2차원 vector가 배열되어 있음 4x2
    input_data = tf.cast(input_data, tf.float32)                                            # (D, float) D를 float 형태로 변환해줌 type casting / 32 메모리 절약 vs 64 메모리 낭비, 소수점 밑 아래까지 저장
                                                                                            # input data 만들기
    and_labels = tf.constant([0,0,0,1])                                                     # binary and / 출력의 정답
    and_labels = tf.cast(and_labels, tf.float32)

    batch_size = 1  #1이면 stochast                                                         # GD에서 Stochastic: tarin data 하나를 GD 업데이트 하고, Batch: 업데이트는 안 하고 쌓아둔 후 모든 train data를 다 확인하고 평균으로 업데이트
    #Stochasgic 특징:같은 시간에 업데이트 수가 많다 100만 번=> 빠른 업데이트 / 전체적으로 loss가 줄어드나 local로 보면 약간의 들쑥날쑥이 존재함 그래서 멈춰야 하는지 계속 업데이트 시켜야 하는지 판단 어려움 
    #batch 특징:학습의 안정성이 좋다, 한번 떨어지면 올라가지 않는다 / 같은 시간에 업데이트 수가 적다 1번 => 느린 업데이트
    #절충안 :  mini-batch GD를 사용한다.

    epochs = 500                                                                            # GD진행시 max_iterations를 정하는데 for 1 to iter: for each

    ##### SLP - Begin
    # SLP 형태 정의
    
    input_layer = tf.keras.Input(shape=[2, ])                                               # 형태shape이 input (binary라서 2개) / 학습을 batch 만큼 들어와서 학습함 / 입력 레이어 노드 2개(binary)
    output = tf.keras.layers.Dense(nuits=1,                                                 # Dense(units:Dense라는 class가 층 만듬 한 층 노드 개수, 
                                  activation=tf.keras.activations.sigmoid,                  # activation:노드 각각이 사용할 function, 
                                  use_bias=True)(input_layer)                               # use_dias:dias까지 학습하는지 아니면 0으로 취급하는지 True는 dias 사용한다) => 층 하나 만들어짐
    # input_layer와 output_layer 연결 : ()는 리스트 , 함수 표현도 가능
    # => single layer network 구조

    # SLP learning(무엇으로 학습 방법:GD, 학습하기 위한 정보 : train, label 데이터 주기, 목적함수, 학습과정을 자동(tensorflow, pytorch) vs 수동)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1)                                        # 학습 방법(알고리즘) GD
    slp_classifier = tf.keras.Model(inputs=input_layer, outputs=output)                     # Model(inputs:train 시켜야 하는 네트워크, outputs:최종적인 네트워크) 
    slp_classifier.compile(optimizer=sgd,                                                   # compile(학습 방법과 목적함수)
                        loss="mse")

    slp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs)    # train == fit(batchsize:batch 학습을 진행하는데 사용되는 size로, mini batch는 기본적으로 64의 배수로 156)
    ##### SLP - End

    ##### SLP - Begin
    ######### MLP - Begin
    input_layer2 = tf.keras.Input(shape=[2, ])
    hidden_layer2 = tf.keras.layers.Dense(units=4,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(input_layer2)
    #추가 된
    output2 = tf.keras.layers.Dense(units=1,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(hidden_layer2)
    
    mlp_classifier = tf.keras.Model(inputs=input_layer2, outputs=output2)
    sgd2 = tf.keras.optimizers.SGD(learning_rate=0.1)
    mlp_classifier.compile(optimizer=sgd2, loss="mse")
    mlp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs)
    ######### MLP - End
    
    ######## SLP AND prediciton
    prediction = slp_classifier.predict(x=input_data, batch_size=batch_size)                # predict로 예측
    input_and_result = zip(input_data, prediction)
    print("====== SLP AND classifier result =====")                                         # 예측 결과 출력
    for x, y in input_and_result:
        if y > 0.5:                                                                         # sigmoid라서
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

    # 0 AND 0 => 0.02 => 0
    # 0 AND 1 => 0.20 => 0
    # 1 AND 0 => 0.20 => 0
    # 1 AND 1 => 0.02 => 1 
    # xor => label = [0,1,1,0],  epoch 1500

if __name__ == '__main__':
    and_classifier_example()