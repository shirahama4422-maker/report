# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 日本語のフォントを設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def drawScatter(x, y, xlabel, ylabel, data_name, b0, b1, R2, plot_labels=None):     
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if plot_labels is None:
        plot_labels = range(1, x.shape[0]+1)
        
    for (i, j, k) in zip(x, y, plot_labels):
        plt.plot(i, j, 'o')
        plt.annotate(k, xy=(i, j))

    # 回帰直線の追加
    x_line = np.array([np.min(x), np.max(x)])
    y_line = b0 + b1 * x_line
    plt.plot(x_line, y_line, color="red", label=f"y={b1:.2f}x+{b0:.2f}\n$R^2$={R2:.3f}")
    plt.legend()

    plt.savefig(data_name + ".png")
    plt.show()


if __name__ == "__main__":

    # データの読み込み
    data_name = "lec02_test_data"
    file_name = data_name + ".xlsx"
    df = pd.read_excel(file_name, header=0, index_col=0, engine='openpyxl')

    d = df.values
    variable_name_list = df.columns
    sample_name_list = df.index


    x = d[:, 0]
    y = d[:, 1]

    x_mean, y_mean = np.mean(x), np.mean(y)
    Sxx = np.sum((x - x_mean)**2**1)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    Syy = np.sum((y - y_mean)**2**1)

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean
    R2 = (b1 * Sxy) / Syy

  
    print("演算結果")
    print(f"回帰係数 b1: {b1:.4f}")
    print(f"切片 b0: {b0:.4f}")
    print(f"寄与率 R2: {R2:.4f}")  
    print("以上のような結果となる。")
    
    drawScatter(x, y, variable_name_list[0], variable_name_list[1],
                data_name, b0, b1, R2, plot_labels=sample_name_list)