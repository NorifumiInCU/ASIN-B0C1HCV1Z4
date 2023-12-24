import os

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
    abspath=os.path.abspath(filepath)
    dirname=os.path.dirname(abspath)
    basename=os.path.basename(abspath).replace('.py','')
    odir=f'{dirname}/{basename}'
    if not(os.path.exists(odir)):
        os.makedirs(odir)
    return dirname, basename, odir