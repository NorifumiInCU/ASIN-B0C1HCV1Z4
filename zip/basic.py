import os
import pandas as pd

def get_basic_data(filepath:str):
    '''
    filepathから作成したdirname,basename,作成した出力先ディレクトリパスを返す
    Parameters:
    filepath:str
        __file__
    Return:
    (dirname, basename, odir)
    dirname  = os.path.dirname(filepath)
    basename = os.path.basename(filepath).replace('.py','')
    odir = f'{dirname}/{basename}'
    '''
    dirname=os.path.dirname(filepath)
    basename=os.path.basename(filepath).replace('.py','')
    odir=f'{dirname}/{basename}'
    if not(os.path.exists(odir)):
        os.makedirs(odir)
    return dirname, basename, odir

