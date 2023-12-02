# 3-3 文字認識-手書き数字

## load_digits()にデータを追加して手書き数字への分類精度を上げる
### 準備
1. 下記を実行環境の同階層に配置
    - `ml_digits.py`
    - `get_extra_data.py`
    - `__init__.py`
    - `predict_compare.py`
    - `org_digits.pkl` ※ digits.pklをリネーム
2. `extra_data`ディレクトリを同階層に配置
3. `extra_data/`内に`01〜09`のディレクトリを作成
### データ準備
1. 1〜9の数字を描いた画像を`extra_data/`下の対応するディレクトリ`01〜09`に配置。画像は8x8にリサイズした時に線が残るようにする。
2. 検証用画像`1.png〜9.png`を`predict_compare.py`と同階層に配置
#### その他
- `classify_extra_data.py` `extra_data`の下に直接配置された画像を機械学習モデルにより`extra_data/xx`に振り分ける。
- `reset_classified_data.py` `extra_data/xx`に分類された画像ファイルを`extra_data/`に移動する
### 学習
1. `python3 ml_digits.py`を実行して`extra_digits.pkl`を作成
### 評価・検証
1. `python3 predict_compare.py`を実行
