import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# アヤメデータ読み込み
import os
if (not os.path.exists('iris.csv')):
    import urllib.request as req
    url = "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch2/iris/iris.csv"
    req.urlretrieve(url, 'iris.csv')
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 目的変数と説明変数
y=iris_data.loc[:, "Name"]
x=iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

print(f'y.unique():{y.unique()}')
label_list=[]
for name in y.unique():
    count=len(y==name)
    print(f'y label name: {name}, count:{count}')
    label_list.append(name)
print(f'label_list:{label_list}')

# 辞書変数用ラベル('を打ちたくない)
Ltest_size='test_size'
Lfid='fid'
Laccuracy='accuracy'
Ldistance='rate_distance'
LOOP_COUNT=10
save_dir=os.path.dirname(__file__)
test_list=[{Ltest_size:0.2, Lfid:'basic-train'},
           {Ltest_size:0.9, Lfid:'little-train'}]
for test in test_list:
    test_size=test[Ltest_size]
    print(f'test_size={test_size}')
    fid=test[Lfid]
    filename=f'iris-{fid}.png'
    filepath=f'{save_dir}/{filename}'
    x_data=[]
    y_data={}
    for counter in range(1, LOOP_COUNT + 1):
        print(f'counter={counter}')
        x_data.append(counter)
        i = counter - 1

        # 分離 (*3)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, shuffle=True)

        # 学習 (*4)
        clf = SVC()
        clf.fit(x_train, y_train)

        # 評価 (*5)
        y_pred = clf.predict(x_test)

        # 正解率 / counter
        if not(Laccuracy in y_data):
            y_data[Laccuracy] = []
        score = np.round(accuracy_score(y_test, y_pred), 3)
        y_data[Laccuracy].append(score)
        print(f'正解率 = {score}')
        print(f'y_train total len:{len(y_train)}') if counter == LOOP_COUNT else None
        for label in label_list:
            # 訓練データに含まれるlabelの数 / counter
            if not(label in y_data):
                y_data[label] = []
                print(f'y_data[{label}] init.')
            count_label = sum(1 for item in y_train if item == label)
            label_rate = np.round(count_label / len(y_train), 3)
            y_data[label].append(label_rate)
            print(f'y_train[{label}] len:{count_label}') if counter == LOOP_COUNT else None
        rate_max = np.max([y_data[label][i] for label in label_list])
        rate_min = np.min([y_data[label][i] for label in label_list])
        rate_distance = abs(np.round(rate_max-rate_min, 3))
        # print(f'list:{[y_data[label][i] for label in label_list]}, rate_distance:{rate_distance}')
        if not(Ldistance in y_data):
            y_data[Ldistance] = []
        y_data[Ldistance].append(rate_distance)
        if counter == LOOP_COUNT:
            print(f'y_data[{Laccuracy}]:{y_data[Laccuracy]}')
            print(f'y_data[{Ldistance}]:{y_data[Ldistance]}')
        print()
    # create plot
    import matplotlib.pyplot as plt
    ## X軸：テスト回数1〜10
    ## Y軸1:訓練データに含まれるIris-setosa?の割合
    ## Y軸2:訓練データに含まれるIris-versicolor?の割合
    ## Y軸3:訓練データに含まれるIris-virginica?の割合
    ## Y軸4:訓練データに含まれるラベルの最大割合と最小割合の差分
    ## Y軸5:正解率
    iris_line_colors=['r:', 'g:', 'b:']
    for i, label in enumerate(label_list):
        print(f'label:{label}, i:{i}, y_data[{label}]:{y_data[label]}')
        if not(label in y_data):
            print(f'{label} skip.')
            continue
        plt.plot(x_data, y_data[label], iris_line_colors[i], label=label)
    
    plt.plot(x_data, y_data[Ldistance], '-', color='purple', label=Ldistance)
    plt.bar(x_data, y_data[Laccuracy], color='orange', alpha=0.5, label=Laccuracy)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    lbl=np.round(1-test_size, 1)
    tfid=f'{fid}({lbl})'
    plt.title(tfid)
    plt.savefig(filepath, bbox_inches='tight')
    print()
    print(f'save {filepath}')
    plt.close()
