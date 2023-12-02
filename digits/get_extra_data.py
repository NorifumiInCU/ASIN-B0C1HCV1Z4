from sklearn import datasets

def get_extra_data(storage_dir):
    '''
    自前数字画像のデータを返す
    Parameters
        storage_dir: str
            自前数字画像データの格納パス $(ls storage_dir)=[01 02 ... 09]
    Returns
        None or (targets, images)
    '''
    import os
    import cv2

    try:
        with os.scandir(storage_dir) as entries:
            targets=[]
            images=[]
            dir_names = [entry.name for entry in entries if entry.is_dir()]
            for dir in dir_names:
                src_dir_name=f'{storage_dir}/{dir}'
                with os.scandir(src_dir_name) as entries2:
                    file_names = [entry.name for entry in entries2 if entry.is_file() and entry.name.lower().endswith('.png')]
                    for file_name in file_names:
                        img_path=f'{src_dir_name}/{file_name}'
                        img=cv2.imread(img_path)
                        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img=cv2.resize(img, (8,8))
                        img=15 - img // 16
                        images.append(img)
                        targets.append(int(dir))
            result=(targets, images)
    except FileNotFoundError:
        print(f"指定されたディレクトリ '{storage_dir}' は存在しません。")
        return None
    return result

if __name__=='__main__':
    import os
    dirname=os.path.dirname(__file__)
    data_dir=f'{dirname}/extra_image'

    # データを読み込む
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target

    extra_data = get_extra_data(data_dir)
    if extra_data is not None:
        import numpy as np
        extra_targets=extra_data[0]
        extra_images=extra_data[1]
        digits.images=np.concatenate((digits.images, extra_images), axis=0)
        digits.target=np.append(digits.target, extra_targets)
        print(f'digits.images.shape:{digits.images.shape}')
        print(digits.images[-1:])
        print(digits.target[-1:])
    else:
        print(f'extra_data is nothing.')
