"""Fold-local preprocessing and F1-based model selection on repository datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score, average_precision_score, roc_auc_score
from phishing_detector import load_training_data, build_pipeline

ROOT = Path(__file__).parent


def model_specs(text=False):
    models = {
        'Logistic Regression':(LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42), {'model__C':[.1,1.]}),
        'KNN':(KNeighborsClassifier(), {'model__n_neighbors':[5,9]}),
        'Bernoulli Naive Bayes':(BernoulliNB(), {'model__alpha':[.1,1.]}),
        'Decision Tree':(DecisionTreeClassifier(class_weight='balanced',random_state=42), {'model__max_depth':[10,None]}),
        'Random Forest':(RandomForestClassifier(class_weight='balanced',n_jobs=4,random_state=42,bootstrap=False), {'model__n_estimators':[100,200]})}
    for name,(model,grid) in models.items():
        steps=[]
        if text:
            steps.append(('tfidf',TfidfVectorizer(stop_words='english',max_features=1000,sublinear_tf=True)))
            if name == 'KNN':
                steps.append(('svd',TruncatedSVD(n_components=50,random_state=42)))
        elif name in ['Logistic Regression','KNN']:
            steps.append(('scale',StandardScaler()))
        steps.append(('model',model))
        yield name, Pipeline(steps), grid


def score(y, estimator, X, positive):
    pred = estimator.predict(X)
    column = list(estimator.classes_).index(positive)
    probability = estimator.predict_proba(X)[:,column]
    binary = (y == positive).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)), 'precision':float(precision_score(y,pred,pos_label=positive,zero_division=0)),
        'recall':float(recall_score(y,pred,pos_label=positive,zero_division=0)), 'f1':float(f1_score(y,pred,pos_label=positive)),
        'average_precision':float(average_precision_score(binary,probability)), 'roc_auc':float(roc_auc_score(binary,probability))}


def run_dataset(name):
    text = name == 'email'
    if text:
        data=load_training_data(); X,y=data['Email Text'],data['Email Type']; positive='Phishing Email'
    else:
        filename='dataset_phishing.csv' if name == 'url_balanced' else 'web-page-phishing.csv'
        data=pd.read_csv(ROOT/'datasets'/filename).drop_duplicates().dropna()
        target='status' if name=='url_balanced' else 'phishing'
        if 'url' in data:
            conflicts=data.groupby('url')[target].nunique()
            data=data[~data.url.isin(conflicts[conflicts>1].index)].drop_duplicates('url')
        X,y=data.drop(columns=[target,'url'],errors='ignore'),data[target]
        positive='phishing' if name=='url_balanced' else 1
    train,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    X_train,X_test,y_train,y_test=X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
    scorer=make_scorer(f1_score,pos_label=positive)
    cv=StratifiedKFold(3,shuffle=True,random_state=42)
    rows=[]; details={}; selected=None
    for model_name,pipe,grid in model_specs(text):
        search=GridSearchCV(pipe,grid,cv=cv,scoring=scorer,n_jobs=1)
        search.fit(X_train,y_train)
        rows.append({'model':model_name,'cv_phishing_f1':float(search.best_score_), **score(y_test,search,X_test,positive)})
        details[model_name]=search.best_params_
        if selected is None or search.best_score_ > selected[0]:
            selected=(search.best_score_,model_name)
        print(name, model_name, rows[-1],flush=True)
    # The app remains a Random Forest demonstration; evaluate its exact pipeline separately.
    if text:
        app=build_pipeline(); app.fit(X_train,y_train)
        rows.append({'model':'App Random Forest (fixed parameters)','cv_phishing_f1':np.nan,**score(y_test,app,X_test,positive)})
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(results/f'{name}_metrics.csv',index=False)
    summary={'rows':len(y),'train_rows':len(train),'test_rows':len(test),'positive_label':positive,
        'selected_by_cv_f1':selected[1],'best_parameters':details,
        'protocol':'Stratified 80/20 holdout, 3-fold stratified training CV, fold-local preprocessing. Random split; no temporal generalization claim.'}
    from importlib.metadata import version
    summary['versions']={p:version(p) for p in ['pandas','numpy','scikit-learn']}
    summary['input_sha256']=__import__('hashlib').sha256((ROOT/'datasets'/('Phishing_Email_Dataset.csv' if text else filename)).read_bytes()).hexdigest()
    (results/f'{name}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for name in ['url_balanced','url_imbalanced','email']:
        run_dataset(name)
