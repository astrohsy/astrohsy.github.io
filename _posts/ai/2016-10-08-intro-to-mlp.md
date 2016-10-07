---
title: 파이썬(Python) MLP(Multilayer Perceptron) 구현
categories: [AI]
tags: [mlp, ai, machinelearning, python]
description: 다층 퍼셉트론(Multi-Layer Perceptron)의 개념 및 파이썬(Python)을 활용한 구현과 MNIST 데이터 분류(Classification)
date : 2016-10-06 05:00:00 +0900
---

다층 퍼셉트론(Multilay Perceptron)을 짧게 MLP라는 용어로 사용하겠다. MLP는 하나 이상의 퍼셉트론(Perceptron)이 여러 개의 층으로 이루어져 있는 구조이다. 퍼셉트론과 마찬가지로 값을 넣고 결과값이 나오면, `퍼셉트론의 결과값`과 `실제 결과값`을 비교를 한 뒤, 오류(Error)를 전파(Propagation)을 하는 원리는 똑같다. 하지만, 다층 구조(Multilayer)이기 때문에 에러 전파에 대한 퍼셉트론 값 보정이 힘들어진다. 그리고 다층 퍼셉트론에서는 제일 마지막 층을 `Output`을 담당하는 `Output Layer`라고 정의하고, 그 아래에 있는 `Layer`들을 `은닉층(Hidden Layer)`이라고 정의한다.

그래서 이 값 전파를 위해서 멀티레이어 퍼셉트론과는 다른 식을 사용한다. 그 식은 다음과 같다.

-- 추후 작성 --

이 식을 사용해서 MNIST(손글씨) 데이터를 위한 분류기(Classifier) MLP를 파이썬으로 작성하면 다음과 같다.

## 다른 인공지능에 비해 가지는 장점

기존 싱글 퍼셉트론보다는 적용할 수 있는 범위가 넒어진다. 왜냐하면 싱글 퍼셉트론일 경우, 분류를 직선 하나로만 할 수 있기 때문에, `XOR` 같은 직선으로 나눌 수 없는 문제는 해결할 수가 없다. 하지만 MLP에서는 은닉층을 만듬으로써 선을 구부리는 것이 가능해진다. 그래서 좀 더 유연한 결과 예측이 가능해진다.

`결정트리(Decision Tree)`와 `Naive Bayes` 알고리즘에 비해 가지는 단점은 이 알고리즘들은 이산적(Discrete)한 값들에 대해서만 예상이 가능해진다. 하지만 연속적(Continuous)한 값들에 대한 예측을 할 때는 실수 값을 직접 사용해서 예측을하는 인공신경망(Neural Network)류의 알고리즘인 MLP와 퍼셉트론가 유리하다.

## 단점

은닉층을 많이 쌓고, 퍼셉트론들을 많이 생성할 수록 학습하는데, 오랜 시간이 걸린다. 그리고 최적환된 은닉층의 갯수와 퍼셉트론 갯수를 찾기위해 찾아야한다.

## 구현

`퍼셉트론 구현`
{% highlight python %}
# neural_module/neuron.py

import random
import math


class Neuron:
    def __init__(self, in_prev_layer_num_of_neuron):
        self.synaptic_weights = [random.uniform(0.01, 0.03) for i in range(in_prev_layer_num_of_neuron)]

    def activate(self, inputs):
        evaluated_value = 0.0

        for i in range(self.synaptic_weights.__len__()):
            evaluated_value += inputs[i] * self.synaptic_weights[i]
        return Neuron.activation_function(evaluated_value)

    @staticmethod
    def activation_function(evaluated_value):
        return 1 / (1 + math.exp(-evaluated_value))
{% endhighlight %}

`층(Layer) 구현`
{% highlight python %}
# neural_module/neural_layer.py

from neural_module import neuron as n

class Layer:
    def __init__(self, n_prev_neuron, n_neuron):
        self.neurons = [n.Neuron(n_prev_neuron) for i in range(n_neuron)]
        self.outputs = [0 for i in range(n_neuron)]

    def evaluate(self, data_input):
        for i, neuron in enumerate(self.neurons):
            self.outputs[i] = neuron.activate(data_input)

        return self.outputs

{% endhighlight %}

`신경망(Neural Network) 구현`
{% highlight python %}
# neural_module/neural_network.py

from neural_module import neural_layer as nl
from neural_module import neuron

class NeuralNetwork:
    def __init__(self, n_neuron_output_layer, n_neuron_hidden_layer, training_example, learning_rate=0.03):
        input_size = len(training_example[0]) - 1
        self.output_layer = nl.Layer(n_neuron_hidden_layer, n_neuron_output_layer)
        self.hidden_layer = nl.Layer(input_size, n_neuron_hidden_layer)
        self.training_example = training_example
        self.learning_rate = learning_rate

    # 전파
    def forward_propagation(self, data_input):
        output_of_hidden_units = self.hidden_layer.evaluate(data_input)
        return self.output_layer.evaluate(output_of_hidden_units)

    # 오류 역전파
    def back_propagation(self, data_input, target):
        self.forward_propagation(data_input)

        error_h = []
        out_from_k = self.output_layer.outputs
        error_k = [out_from_k[k] * (1 - out_from_k[k]) * (target[k] - out_from_k[k]) for k in range(len(out_from_k))]

        for h in range(len(self.hidden_layer.outputs)):
            o_h = self.hidden_layer.outputs[h]

            sigma = 0.0
            for k in range(len(self.output_layer.outputs)):
                sigma += self.output_layer.neurons[k].synaptic_weights[h] * error_k[k]
            error_h.append(o_h * (1 - o_h) * sigma)

        # update output layer
        for j in range(len(self.output_layer.neurons)):
            for i in range(len(self.hidden_layer.outputs)):
                self.output_layer.neurons[j].synaptic_weights[i] += self.learning_rate * error_k[j] * \
                                                                    self.hidden_layer.outputs[i]

        for j in range(len(self.hidden_layer.neurons)):
            for i in range(len(data_input)):
                self.hidden_layer.neurons[j].synaptic_weights[i] += self.learning_rate * error_h[j] * data_input[i]

    def train(self):
        for example in self.training_example:
            true_val = [0 for i in range(len(self.output_layer.outputs))]
            true_val[example[-1]] = 1

            map(neuron.Neuron.activation_function, example)
            self.back_propagation(example[:-1], true_val)

{% endhighlight %}

`신경망 사용 부분(Main)`
{% highlight python %}
# main.py

from neural_module import neural_network as nn
import time
training_example, test_example = [], []

# 학습 데이터
with open('data/optdigits.tra') as f:
    for line in f:
        training_example.append([int(x) for x in line.rstrip('\n').split(',')])

# 검증 데이터
with open('data/optdigits.tes') as f:
    for line in f:
        test_example.append([int(x) for x in line.rstrip('\n').split(',')])

network = nn.NeuralNetwork(10, 30, training_example, 0.05)

# 시간 측정 시작
start = time.time()
num_of_time = 20
for i in range(num_of_time):
    print(i, '/', num_of_time)
    network.train()

# test
true_cnt, cnt = 0, 0
for test in test_example:
    network.forward_propagation(test[:-1])
    '''
    print(test[-1])
    for i in range(len(network.output_layer.outputs)):
        print("%.3f " %network.output_layer.outputs[i], end="")
    print("")
    '''
    min_error, min_num = 1000000, -1
    for i in range(len(network.output_layer.outputs)):
        if abs(1 - network.output_layer.outputs[i]) < min_error:
            min_error = abs(network.output_layer.outputs[i] - 1)
            min_num = i

    if min_num == test[-1]:
        true_cnt += 1
    cnt += 1

print("정확도 %.10f%%" %(true_cnt/cnt*100))
print("경과시간 %.2f" %(time.time()-start))

{% endhighlight %}

## 개선방향

인공신경망는 거의 행렬(Matrix) 연산이 전부이다. 이를 GPU를 활용한 병렬로 처리하면 속도 개선을 할 수 있다. 그리고 이를 편하게 하기 위해서 파이썬의 `numpy` Library를 활용하면 될 것 같다.
