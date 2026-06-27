# [1] 데이터셋 준비
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.unit8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
# 전처리: 정규화

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape) #(784,) 1차원으로 펼쳤기 때문
img = img.reshape(28, 28)  #다시 28*28 이미지로 만듦
print(img.shape)

img_show(img)

# [2] 신경망 추론 처리

def get_data():
    (x_train, t_train), (x_test, t_test) = \
         load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) +b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# [3] accuracy 평가

x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) #가장 확률이 높은 원소의 인덱스 얻음
    if p == t[i]:  # 해당 인덱스와 정답 레이블이 맞는지 확인 후 카운트
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt / len(x))))

## 데이터 흐름
## x -> W1 -> W2 -> W3 -> y
## 784 -> 784*50 -> 50*100 -> 100*10 ->10

# [4] batch size 구현
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
