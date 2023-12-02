from sklearn import datasets
import joblib
import os

if os.path.exists('extra_digits.pkl'):
    clf_fname='extra_digits.pkl'
elif os.path.exists('org_digits.pkl'):
    clf_fname='org_digits.pkl'
else:
    clf_fname='digits.pkl'
clf = joblib.load(clf_fname)
def classify_extra_data(storage_dir, is_try=True):
    '''
    storage_dir/*.pngをstorage_dir/xx/に振り分ける。xxは01〜09。
    Parameters  
        storage_dir:str  
            数字画像の格納ディレクトリパス  
        is_try:boolean  
            Trueのとき、画像ファイルの振り分けを試行する  
            defalut:True  
    '''
    
    # 指定されたディレクトリ内のファイル名をリストする関数
    try:
        with os.scandir(storage_dir) as entries:
            print(f'entries:{entries}')
            file_names = [entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith('.png')]
    except FileNotFoundError:
        print(f"'{storage_dir}' is not found.")
        return None
    print(f'storage_dir:{storage_dir}')
    print(f'file_names:{file_names}')
    from predict_myimage import predict_digit
    targets=[]
    images=[]
    counter_per_pred={}
    for file in file_names:
        file_path=f'{storage_dir}/{file}'
        pred = predict_digit(file_path, clf=clf)
        if not( pred in counter_per_pred.keys() ):
            counter_per_pred[pred]=0
            print(f'found {pred}.')
        dest_dir = f'{storage_dir}/{pred:02d}'
        destination = f'{dest_dir}/{pred:02d}_{counter_per_pred[pred]}.png'
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        print(f'mv {file_path} {destination}')
        if not is_try:
            import shutil
            shutil.move(file_path, destination)
            import cv2
            img=cv2.imread(destination)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img, (8, 8))
            img=15 - img // 16
            # img=img.reshape((-1, 64))
            images.append(img)
            targets.append(pred)
            # result['images'].append(img)
            # result['target'].append(pred)
        counter_per_pred[pred]+=1
    print(f'{counter_per_pred}')
    if not is_try:
        result=(targets, images)
        print(f'last image:')
        print(images[-1:])
    else:
        result=None
    return result

if __name__=='__main__':
    dirname=os.path.dirname(__file__)
    data_dir=f'{dirname}/extra_image'

    # データを読み込む
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target

    print(f'x.type:{type(x)}')
    print(f'y.type:{type(y)}')
    extra_data = classify_extra_data(data_dir)
    if extra_data is not None:
        import numpy as np
        extra_targets=extra_data[0]
        extra_images=extra_data[1]
        digits.images=np.concatenate((digits.images, extra_images), axis=0)
        digits.target=np.append(digits.target, extra_targets)
        print(digits.images[-1:,:])
    else:
        print(f'extra_data is None.')
