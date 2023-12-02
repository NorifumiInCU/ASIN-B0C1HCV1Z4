from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import cross_val_score
from get_extra_data import get_extra_data
import numpy as np

import os
dirname=os.path.dirname(__file__)
extra_data_dir=f'{dirname}/extra_image'

clf_fname='extra_digits.pkl'


# データを読み込む
digits = datasets.load_digits()
x = digits.images
y = digits.target
extra_data = get_extra_data(extra_data_dir)
if extra_data is None:
    print(f'extra_data failed')
    exit(1)
extra_targets = extra_data[0]
extra_images = extra_data[1]
print(f'extra_data:{len(extra_targets)}')
x = np.concatenate((x, extra_images), axis=0)
y = np.append(y, extra_targets)
x = x.reshape((-1, 64))

is_search=False
if is_search:
    # search algorithm
    allAlgorithms = all_estimators(type_filter="classifier")
    # print(f'allAlgorithms:[{allAlgorithms}]')
    warnings.simplefilter("error")
    kfold_cv = StratifiedKFold()
    best_cross_val_score = { 'name':'', 'score':0.0 }
    for name, algorithm in allAlgorithms:
        try:
            if name == "LinearSVC":
                clf = algorithm(max_iter = 10000, dual='auto')
            else:
                clf = algorithm()

            if hasattr(clf, "score"):
                # クロスバリデーション
                scores = cross_val_score(clf, x, y, cv=kfold_cv)
                # print(f'{name} 正解率 = {scores}')
            else:
                continue

            if scores.mean()*1000 > best_cross_val_score['score']*1000:
                best_cross_val_score["name"]=name
                best_cross_val_score["score"]=scores.mean()

        except Exception as e:
            pass
            # print(f'\033[31m Error:\033[0m {name},: {e.args}')
    print(f'best: {best_cross_val_score["name"]} {best_cross_val_score["score"]}')
    exit(0)

# 層別サンプリングのためのオブジェクトを作成
stratified_splitter = StratifiedShuffleSplit(n_splits=50, test_size=0.2)

max_accuracy=0.0
best_clf=None
# インデックスを取得
for train_index, test_index in stratified_splitter.split(x, y):
    # print(f'target:{y[train_index]}')
    # print(f'train_index:{train_index}')
    # print(f'test_index:{test_index}')
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # データを学習
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # 予測して精度を確認する
    y_pred = clf.predict(X_test)
    score=accuracy_score(y_test, y_pred)
    # print(f'score:{score}')

    if score*1000 > max_accuracy*1000:
        max_accuracy=score
        best_clf=clf

print(f'best accuracy:{max_accuracy}')
if best_clf is not None:
    # 学習済みモデルを保存
    import joblib
    joblib.dump(best_clf, clf_fname)
