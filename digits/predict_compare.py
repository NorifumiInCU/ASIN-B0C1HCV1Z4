import joblib
from sklearn.metrics import accuracy_score
from predict_myimage import predict_digit
import os

dirname=os.path.dirname(__file__)
extra_data_dir=f'{dirname}/extra_image'

clf_names=['org_digits.pkl', 'extra_digits.pkl']
for clf_name in clf_names:
    print(f'clf:{clf_name}:')
    clf = joblib.load(clf_name)
    y=[1,2,3,4,5,6,7,8,9]
    check_list=['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png',
                ]
    y_pred=[]
    for target in check_list:
        n = predict_digit(target, clf=clf)
        print(f'{target} = {n}')
        y_pred.append(n)
    score=accuracy_score(y, y_pred)
    print(f'score:{score}')
    print()
