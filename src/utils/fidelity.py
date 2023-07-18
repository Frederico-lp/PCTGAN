import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score

def get_predictions(X_train, y_train, X_test, y_test, undersample = True):
    if undersample:
        sampling_strategy = 0.75
        rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
        X_train, y_train = rus.fit_resample(X_train, y_train) # type: ignore
        #X_test, y_test = rus.fit_resample(X_test, y_test)


        
    learners = [(AdaBoostClassifier(n_estimators=50))]
    #learners = [(RandomForestClassifier())]

    history = dict()

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X_train, y_train)

        pred = []

        for j in range (len(X_test)):
            #print(X_test.loc[[j]])
            pred.append(model.predict(X_test.iloc[[j]]))
        
    return pred

def eval_fidelity(pred1, pred2):
    same_pred = 0
    dif_pred = 0
    if len(pred1) != len(pred2):
        print("Error: different sizes")
    
    for i in range(len(pred1)):
        if pred1[i] == pred2[i]:
            same_pred += 1

        else:
            dif_pred += 1

    ratio = same_pred / (same_pred + dif_pred)

    return ratio

def get_class_ratios(real, fake, target):
    
    values = np.array(real)
    values = np.unique(values)
    class1 = values[0]
    class2 = values[1]
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    
    for i in range(len(fake)):
        if real[i] == fake[i]:  # TP or TN
            if real[i] == target:
                TP += 1
            else:             # TN
                TN += 1

        else:                # FP or FN
            if real[i] == target:
                FN += 1
            else:             # FP  
                FP += 1
        
    print(TP, TN, FP, FN)
    class1_ratio = (TP + FN) / (TP + TN + FP + FN)
    class2_ratio = (TN + FP) / (TP + TN + FP + FN)

    return class1_ratio, class2_ratio


def get_accuracy(y, pred):
    return accuracy_score(y, pred)

def get_roc_auc(y, pred):
    return roc_auc_score(y, pred)
    

