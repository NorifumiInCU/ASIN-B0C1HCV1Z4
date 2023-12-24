from keras.callbacks import Callback
from keras.metrics import Recall, Precision
import numpy as np

class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.f1_scores = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        predict_classes = np.argmax(predict, axis=1)

        # 計算とログの表示
        recall = Recall()(self.validation_data[1], predict_classes)
        precision = Precision()(self.validation_data[1], predict_classes)

        tmp_recall = recall[0] if isinstance(recall, list) else recall
        tmp_precision = precision[0] if isinstance(precision, list) else precision
        f1 = 2 * (tmp_recall * tmp_precision) / (tmp_recall + tmp_precision + 1e-10)  # 0での除算を防ぐために小さな値を加える

        self.f1_scores.append(f1)
        self.recalls.append(recall.numpy())
        self.precisions.append(precision.numpy())

        print(f' - f1_score: {f1:.4f}, Recall: {recall.numpy():.4f}, Precision: {precision.numpy():.4f}')
