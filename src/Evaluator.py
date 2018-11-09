import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from matplotlib import pyplot as plt


'''
Class for model evaluation - see precision, recall and accuracy
metrics along with plots summarizing performance. All model params
accept sklearn objects and all X and y params are multidimensional
numpy arrays or pandas dataframes.
'''
class Evaluator(object):

    def __init__(self):
        self.name = None

    '''
    Predict class using an sklearn model with predict_proba attr.

    Params:
    model: Sklearn model object
    X: Array-like
    threshold: Float 
    '''
    def predict_class(self, model, X, threshold):
        logits = model.predict_proba(X)
        predictions = [1 if float(sample[1]) > threshold else 0 for sample in logits]
        return predictions


    '''
    Summarize model performance with precision and recall statistics.
    Use proba to indicate if predictions are probablities.

    Params:
    model: sklearn model object
    X: Array-like 
    y: Array-like
    threshold: float
    proba: Boolean 
    '''
    def summarize_performance(self, model, X, y, threshold=0.5, proba=True, return_stats=False):
        if proba:
            predictions = self.predict_class(model, X, threshold)

        else:
            predictions = model.predict(X)
        
        precision = precision_score(y_true=y, y_pred=predictions)
        recall = recall_score(y_true=y, y_pred=predictions)
        accuracy = accuracy_score(y_true=y, y_pred=predictions)
        print ("Model Performance:\n Precision: {}\n Recall: {}\n Accuracy: {}".format(precision, recall, accuracy))

        if return_stats:
            return [precision, recall, accuracy]


    '''    
    Pass model coefficents and feature names to plot the n most 
    important features. Use one_dim to specify dimensionality of 
    scores vector. 

    Params:
    scores: Array-like
    names: Array-like
    n: Integer 
    one_dim: Boolean
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
    Display ROC curve and AUC for a given model. Use proba
    to specify whether the model is 

    Params:
    model: Sklearn or Xgboost model object
    X: Array-like
    y: Array-like
    proba: Boolean
    model_type: String ('sklearn' or 'xgboost')
    '''
    def plot_roc_curve(self, model, X, y, proba=True, model_type='sklearn'):

        if model_type == 'sklearn':
            if proba:
                probs = model.predict_proba(X)
                preds = probs[:,1]

            else:
                preds = model.predict(X)

        elif model_type =='xgboost':
            preds = model.predict(X)

        else:
            raise ValueError('Only allowed model types are sklearn and xgboost')

        fpr, tpr, threshold = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic\n')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    """
    This function prints and plots the confusion matrix. 
    Normalization can be applied by setting `normalize=True`.

    Params: 
    cm: Sklearn confusion matrix object
    classes: Array-like sequence of labels
    normalize: Boolean
    title: String 
    cmap: Color scheme for matrix
    """
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        fig = plt.figure(num=None, figsize=(10, 7), dpi=80)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    """
    Prints precision, recall and fallout for a confusion matrix

    Params:
    cm: Sklearn confusion matrix object
    """
    def summarize_cmatrix(self, cm):
        tp = cm[1,1]
        fn = cm[1,0]
        fp = cm[0,1]
        tn = cm[0,0]
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))
        return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)


    '''
    Takes a model and recursively elimiates features based on importance,
    with regard to the scoring metric evaluated at each iteration. Step
    param controls number of features dropped at each iteration.

    Params:
    model: Sklearn model object
    X: Array-like 
    y: Array-like
    cv: Sklearn cross-validation iterator
    scoring: String
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



    '''
    Function to compare performance of different models/metrics. 
    Results and labels should hold arrays of metrics and their names
    respectively.

    Params:
    results: Matrix with each row Array-like
    labels: Array-like
    xlabel: String
    ylabel: String
    titles: Array-like (String titles for each plot)
    '''
    def plot_compare(self, results, labels, xlabel, ylabel, titles):
        num_rows = int(len(results) / 2)
        num_cols = len(results) - num_rows
        
        if (num_rows != num_cols) or len(results) != len(titles):
            raise ValueError('Number of items in results should be even\
                                 and equal to number of items in titles')
        
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(13, 9), dpi=80)
        fig.subplots_adjust(hspace=0.5)
        idx=0
        
        for r_ind,row in enumerate(ax):
            for c_ind, col in enumerate(row):
                col.bar(range(len(results[idx])), [10 * stat for stat in results[idx]], align='center')
                col.set_title("\n" + titles[idx] + "\n")
                col.set_ylabel(ylabel)
                col.set_xticklabels(labels)
                col.set_xticks(range(len(labels)))
                col.set_yticklabels(round(10 * i, 3) for i in range(0, 11))
                col.set_yticks(range(11))
                
                for i, v in enumerate(results[idx]):
                    col.text(i - 0.1, v*10 +0.2 , str(round(v * 100, 1)) + " %", color='white')
                
                idx+=1
                
        plt.show()

