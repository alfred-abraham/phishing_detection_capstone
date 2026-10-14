import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from evaluate_models import model_specs
from phishing_detector import load_training_data

def test_vectorizer_is_fitted_in_each_training_fold():
    X=pd.Series(['uniquealpha meeting','uniquebeta meeting','uniquegamma urgent','uniquedelta urgent',
                 'uniqueepsilon meeting','uniquezeta meeting','uniqueeta urgent','uniquetheta urgent'])
    y=pd.Series(['Safe Email','Safe Email','Phishing Email','Phishing Email']*2)
    _,pipe,_=next(model_specs(text=True))
    cv=StratifiedKFold(2,shuffle=True,random_state=42)
    result=cross_validate(pipe,X,y,cv=cv,return_estimator=True)
    for estimator,(train,valid) in zip(result['estimator'],cv.split(X,y)):
        vocabulary=estimator.named_steps['tfidf'].vocabulary_
        for index in valid:
            assert X.iloc[index].split()[0] not in vocabulary

def test_conflicting_email_labels_are_excluded(tmp_path):
    data=pd.DataFrame({'Email Text':['same','same','good meeting','bad urgent'],
        'Email Type':['Safe Email','Phishing Email','Safe Email','Phishing Email']})
    p=tmp_path/'email.csv'; data.to_csv(p,index=False)
    assert set(load_training_data(p)['Email Text'])=={'good meeting','bad urgent'}
