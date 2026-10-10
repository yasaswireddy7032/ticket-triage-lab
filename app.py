import json,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix

def examples():
    groups={'billing':['invoice amount is wrong','charged twice for subscription','refund my duplicate payment','billing statement has an error','payment was declined','unexpected card charge','subscription price changed','need a receipt for payment','cancel recurring billing','tax on invoice is incorrect','refund has not arrived','update my billing address'],
    'access':['cannot log into account','password reset link expired','two factor code missing','account is locked','forgot my password','sign in keeps failing','authentication code invalid','login page gives error','reset account credentials','unable to access dashboard','session expires after login','need help signing in'],
    'shipping':['package has not arrived','delivery tracking not updated','parcel sent to wrong address','shipment is delayed','where is my order','courier lost my package','change delivery address','tracking says delivered but missing','expedite the shipment','order damaged during shipping','delivery date keeps changing','package stuck in transit']}
    return [(s,label) for label,ss in groups.items() for s in ss]

def fit(rows):
    m=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(C=4,max_iter=1000,random_state=7));m.fit([r[0] for r in rows],[r[1] for r in rows]);return m

def classify(model,text,threshold=.55):
    features=model[0].transform([text])
    if features.nnz==0:return {'route':'human_review','confidence':0.0,'reason':'no_known_terms'}
    p=model.predict_proba([text])[0];i=int(np.argmax(p));confidence=float(p[i])
    return {'route':str(model.classes_[i]) if confidence>=threshold else 'human_review','confidence':confidence,'reason':'classifier' if confidence>=threshold else 'below_threshold'}

def experiment():
    rows=examples();train,test=train_test_split(rows,test_size=.25,stratify=[r[1] for r in rows],random_state=7)
    m=fit(train);true=[r[1] for r in test];pred=m.predict([r[0] for r in test])
    return {'data':'36 hand-authored examples; illustrative only','train_n':len(train),'test_n':len(test),'macro_f1':float(f1_score(true,pred,average='macro')),'labels':m.classes_.tolist(),'confusion_matrix':confusion_matrix(true,pred,labels=m.classes_).tolist(),'example':classify(m,'my package is delayed'),'unknown':classify(m,'astronomy galaxy telescope')}
if __name__=='__main__':print(json.dumps(experiment(),indent=2))
