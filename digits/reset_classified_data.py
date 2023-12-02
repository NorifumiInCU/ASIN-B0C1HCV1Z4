
def reset_classified_data(storage_dir, is_try=True):
    '''
    `extra_data/xx`に分類された画像ファイルを`extra_data/`に移動する  
    Parameters  
        storage_dir:str  
            数字画像ファイル格納ディレクトリパス  
            ls storage_dir=[01 02 ... 09]
        is_try:boolean  
            ファイルの移動を試行する場合True
            default:True
    '''
    import os
    import shutil
    
    # 指定されたディレクトリ内のファイル名をリストする関数
    try:
        with os.scandir(storage_dir) as entries:
            print(f'entries:{entries}')
            dir_names = [entry.name for entry in entries if entry.is_dir()]
            for dirname in dir_names:
                src_dir=f'{storage_dir}/{dirname}'
                with os.scandir(src_dir) as entries2:
                    for entry in entries2:
                        if not entry.is_file():
                            continue
                        file_name=entry.name
                        file_path=f'{src_dir}/{file_name}'
                        dst_file_name=file_name
                        if not file_name.lower().endswith('.png'):
                            dst_file_name+='.png'
                        destination=f'{storage_dir}/{dst_file_name}'
                        if not is_try:
                            shutil.move(file_path, destination)
                        else:
                            print(f'mv {file_path} {destination}')
    except FileNotFoundError:
        print(f"指定されたディレクトリ '{storage_dir}' は存在しません。")
        return False
    return True

if __name__=='__main__':
    import os
    dirname=os.path.dirname(__file__)
    data_dir=f'{dirname}/extra_image'

    repair_data(data_dir)