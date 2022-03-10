import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch.nn.functional as F


def softmax_with_temperature(x, temparature):
    x = x / temparature
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def plot():
    n_classes = 10
    left = np.linspace(0, n_classes-1, n_classes)
    # 適当に平均4, 分散1の正規分布をつくる．
    p = np.array([[norm.pdf(x=i, loc=4, scale=1) for i in range(n_classes)]], dtype=np.float32)
    softmax_p_t1 = softmax_with_temperature(p, 1)  # 普通のsoftmax
    softmax_p_t01 = softmax_with_temperature(p, 0.1)
    softmax_p_t05 = softmax_with_temperature(p, 0.5)
    softmax_p_t_1 = softmax_with_temperature(p, -1)  # 普通のsoftmax
    softmax_p_t_01 = softmax_with_temperature(p, -0.1)
    softmax_p_t_05 = softmax_with_temperature(p, -0.5)
    ps = ['softmax_p_t1', 'softmax_p_t01', 'softmax_p_t05', 'softmax_p_t_1', 'softmax_p_t_01', 'softmax_p_t_05']
    labels = ['T='+str(i) for i in [1, 0.1, 0.5, -1, -0.1, -0.5]]
    colors = ['red', 'red', 'red', 'blue', 'blue', 'blue']
    for i, p in enumerate(ps):
        plt.figure(figsize=(5,5))
        p_ = eval(p)
        plt.bar(left, p_.flatten(), color=colors[i], label=labels[i])
        plt.legend()
        plt.savefig('./{}.png'.format(p))


plot()