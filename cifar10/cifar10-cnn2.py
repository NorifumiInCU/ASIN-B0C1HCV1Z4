import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import Recall, Precision
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from basic import get_basic_data

dirname, basename, odir = get_basic_data(__file__)

num_epochs=3
attempt_count=1
if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        try:
            num_epochs=int(sys.argv[1])
        except ValueError as e:
            print(f'Usage: {sys.argv[0]} <num_of_epochs=3> <attempt_count=1>')
            exit(1)
        if len(sys.argv)>2:
            attempt_count=int(sys.argv[2])
    else:
        num_epochs=3
        attempt_count=1
num_classes = 10
im_rows = 32
im_cols = 32
in_shape = (im_rows, im_cols, 3)

recall=Recall()
precision=Precision()

# データを読み込む (*1)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# データを正規化(normalize)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ラベルデータをOne-Hot形式に変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

for activation in ['relu', 'gelu']:
    keras_files = glob.glob(f'{odir}/*.keras')
    for file in keras_files:
        os.remove(f'{file}')
        print(f'removed keras file:{file}')

result_for_plot = {
    'relu':{
        'f1_score':None,
        'train_acc':None,
        'val_acc':None,
        'train_loss':None,
        'val_loss':None,
        'last_epoch':0,
        },
    'gelu':{
        'f1_score':None,
        'train_acc':None,
        'val_acc':None,
        'train_loss':None,
        'val_loss':None,
        'last_epoch':0,
        },
    }

matplotlib.rcParams.update({'font.size': 12})
Lrelu_train='relu-train'
Lrelu_val='relu-val'
Lrelu_f1_score='relu-f1_score'
Lrelu_last_epoch='relu-last_epoch'

Lgelu_train='gelu-train'
Lgelu_val='gelu-val'
Lgelu_f1_score='gelu-f1_score'
Lgelu_last_epoch='gelu-last_epoch'

Lloc='loc'
Lylimit='ylimit'
Lanno_xy='anno_xy'
Lmetrics_label='metrics_label'

for try_index in range(attempt_count):
    fsymbol=f'{try_index+1}-{basename}'
    print(f'try:{try_index} fsymbol:{fsymbol}:')

    for index, activation_method in enumerate(['relu', 'gelu']):
        print(f'index:{index} activation:{activation_method}:')
        hist_log=result_for_plot[activation_method]
        # モデルを定義 (*3)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=in_shape))
        model.add(Activation(activation_method))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation_method))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_method))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation(activation_method))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation_method))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # モデルを構築 (*4)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', Recall(), Precision()]
        )

        # Create a learning rate scheduler callback.
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        )

        # Create an early stopping callback.
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # 学習を実行 (*5)
        hist = model.fit(
            x_train, y_train,
            batch_size=32, epochs=num_epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr])

        # モデルを評価 (*6)
        score = model.evaluate(x_test, y_test, verbose=1)

        # F1スコアの計算
        y_pred_prob = model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # precision, recall, f1_scoreを計算
        _, _, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        print(f'{activation_method}: 正解率={score[1]:.4f} loss={score[0]:.4f} F1 Score:{f1_score:.4f}')
        keras_fname=f'{fsymbol}-{activation}-{f1_score:.4f}.keras'
        model.save(f'{odir}/{keras_fname}')
        print(f'dump {fsymbol}.model f1_score:{f1_score:.4f}')

        # historyオブジェクトから precision と recall を取得
        precision_key = [key for key in hist.history.keys() if 'val_precision' in key][0]
        recall_key = [key for key in hist.history.keys() if 'val_recall' in key][0]

        val_precision = hist.history[precision_key]
        val_recall = hist.history[recall_key]

        # ゼロ割を避けるために小さな値を加えておく
        epsilon = 1e-7
        val_precision = [p + epsilon for p in val_precision]
        val_recall = [r + epsilon for r in val_recall]

        # F1スコアの計算 F1スコア = 2 * (Precision * Recall) / (Precision + Recall) 
        f1_score_history = 2 * (np.array(val_precision) * np.array(val_recall)) / (np.array(val_precision) + np.array(val_recall))

        # トレーニング終了時点のエポック数
        stopped_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch!=0 else num_epochs

        # プロット用データ作成
        hist_log['train_acc']=hist.history['accuracy']
        hist_log['val_acc']=hist.history['val_accuracy']
        hist_log['f1_score']=f1_score_history
        hist_log['train_loss']=hist.history['loss']
        hist_log['val_loss']=hist.history['val_loss']
        hist_log['last_epoch']=stopped_epoch

    # plot
    for index, metrics in enumerate(
        [
            {
                Lrelu_train:result_for_plot['relu']['train_acc'],
                Lrelu_val:result_for_plot['relu']['val_acc'],
                Lrelu_f1_score:result_for_plot['relu']['f1_score'],
                Lrelu_last_epoch:result_for_plot['relu']['last_epoch'],

                Lgelu_train:result_for_plot['gelu']['train_acc'],
                Lgelu_val:result_for_plot['gelu']['val_acc'],
                Lgelu_f1_score:result_for_plot['gelu']['f1_score'],
                Lgelu_last_epoch:result_for_plot['gelu']['last_epoch'],

                Lloc:'lower right',
                Lylimit:(0.5, 1.02),
                Lanno_xy: (0, 1.0),
                Lmetrics_label:'acc'
            },
            {
                Lrelu_train:result_for_plot['relu']['train_loss'],
                Lrelu_val:result_for_plot['relu']['val_loss'],
                Lrelu_f1_score:result_for_plot['relu']['f1_score'],
                Lrelu_last_epoch:result_for_plot['relu']['last_epoch'],

                Lgelu_train:result_for_plot['gelu']['train_loss'],
                Lgelu_val:result_for_plot['gelu']['val_loss'],
                Lgelu_f1_score:result_for_plot['gelu']['f1_score'],
                Lgelu_last_epoch:result_for_plot['gelu']['last_epoch'],

                Lloc:'upper right',
                Lylimit:(0.05, 1.1),
                Lanno_xy: (0, 0.1),
                Lmetrics_label:'loss'
            }
        ]):
        plt.figure(figsize=(9, 6))
        ax1=plt.subplot2grid((1, 2), (0,0))
        ax2=plt.subplot2grid((1, 2), (0,1))

        relu_train=metrics[Lrelu_train]
        relu_val=metrics[Lrelu_val]
        relu_f1_score=metrics[Lrelu_f1_score]
        relu_stopped_epoch=metrics[Lrelu_last_epoch]

        gelu_train=metrics[Lgelu_train]
        gelu_val=metrics[Lgelu_val]
        gelu_f1_score=metrics[Lgelu_f1_score]
        gelu_stopped_epoch=metrics[Lgelu_last_epoch]

        y_min, y_max = metrics[Lylimit]
        metrics_label = metrics[Lmetrics_label]
        anno_x, anno_y = metrics[Lanno_xy]
        anno_offset_base_y = 30
        anno_offset_y = anno_offset_base_y if metrics[Lloc].find('lower') else (-1*anno_offset_base_y)
        legend_loc = metrics[Lloc]
        
        print(f'plot index:{index} metrics:{metrics_label}:')
        for a in [(ax1, relu_train, relu_val, relu_f1_score, relu_stopped_epoch, 'relu'),
                (ax2, gelu_train, gelu_val, gelu_f1_score, gelu_stopped_epoch, 'gelu')]:
            ax = a[0]
            train = a[1]
            val = a[2]
            f1_score = a[3]
            stopped_epoch = a[4]
            activation_label = a[5]
            loc = legend_loc

            print(f'acitivation:{activation_label}:')
            # 学習の様子をグラフへ描画
            ax.plot(train, label=f'train {metrics_label}')
            ax.plot(val, label=f'test {metrics_label}')
            ax.plot(f1_score, ':', label='f1_score')

            # プロットした後に y軸の最後の値を取得して表示
            train_last = train[-1]
            test_last = val[-1]
            test_f1score_last = f1_score[-1]
            # y軸の最後の値をテキストで表示
            annotation=f'test_{metrics_label}:{test_last:.4f}\n'
            annotation+=f'train_{metrics_label}:{train_last:.4f}\n'
            annotation+=f'test_f1_score:{test_f1score_last:.4f}\n'
            annotation+=f'stopped_epoch:{stopped_epoch}'

            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
            ax.annotate(annotation, xy=(anno_x, anno_y), xytext=(0, anno_offset_y), textcoords='offset points', ha='left', va='center', bbox=bbox)

            ax.set_ylim(y_min, y_max)
            ax.set_title(f'{activation_label} {metrics_label} & f1_score')
            ax.legend(loc=loc)

        fig_fpath=f'{odir}/{fsymbol}-{metrics_label}-epoch{num_epochs}.png'
        plt.savefig(f'{fig_fpath}', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f'save {fig_fpath}')
