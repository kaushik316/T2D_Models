import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt


'''
Class for model evaluation - see precision, recall and accuracy
metrics along with plots summarizing performance. All model params
accept sklearn objects and all X and y params are array-like.
'''
class Evaluator(object):

    def __init__(self):
        self.name = None


    '''
    Predict class using a model that predicts probabilities
    with a user defined threshold for classification
    '''
    def predict_class(self, model, X, threshold):
        logits = model.predict_proba(X)
        predictions = [1 if float(sample[1]) > threshold else 0 for sample in logits]
        return predictions


    '''
    Summarize model performance with precision and recall statistics
    '''
    def summarize_performance(self, model, X, y, threshold=0.5, proba=True):
        if proba:
            predictions = self.predict_class(model, X, threshold)

        else:
            predictions = model.predict(X)
        
        precision = precision_score(y_true=y, y_pred=predictions)
        recall = recall_score(y_true=y, y_pred=predictions)
        accuracy = accuracy_score(y_true=y, y_pred=predictions)
        print ("Model Performance:\n Precision: {}\n Recall: {}\n Accuracy: {}".format(precision, recall, accuracy))


    '''    
    Pass model coefficents and feature names to see most important features 
    '''
    def feat_importance(self, scores, names, n=10, one_dim=True):
        imp = scores
        if not one_dim:
            imp,names = zip(*sorted(zip(imp[0],names)))
        else:
            imp,names = zip(*sorted(zip(imp,names)))
        fig = plt.figure(num=None, figsize=(10, 7), dpi=80)
        plt.barh(range(len(names[-n:])), imp[-n:], align='center')
        plt.yticks(range(len(names[-n:])), names[-n:])
        plt.title("Most Important Features \n")
        plt.xlabel("score")
        plt.ylabel("features")
        plt.show()

        return names[:-n]


    '''
    Display ROC curve and AUC for a given model
    '''
    def plot_roc_curve(self, model, X, y, proba=True):
        # calculate the fpr and tpr for all thresholds of the classification

        if proba:
            probs = model.predict_proba(X)
            preds = probs[:,1]

        else:
            probs = model.predict(X)
            preds = probs


        fpr, tpr, threshold = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)

        # method I: plt
        plt.title('Receiver Operating Characteristic\n')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    '''
    Takes a model and recursively elimiates features based on 
    importance. Step param controls number of features dropped
    at each iteration.
    '''
    def recursive_elim(self, model, X, y, step=10, cv=StratifiedKFold(2), scoring='recall'):
        rfecv = RFECV(estimator=model, step=step, cv=cv, scoring=scoring)
        rfecv.fit(X, y)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, rfecv.n_features_+ 1, step), rfecv.grid_scores_)
        plt.show()

        return rfecv
