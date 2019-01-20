import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
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
    Use proba to indicate if predictions are probablities and binary to
    indicate if binary or multiclass prediction. If binary=False, pass 
    string to average to determine how precision and recall metrics are
    weighted across classes. See options at sklearn.metrics.recall_score.

    Params:
    model: sklearn model object
    X: Array-like 
    y: Array-like
    threshold: float
    proba: Boolean 
    binary: Boolean
    average: String
    return_stats: Boolean
    '''
    def summarize_performance(self, model, X, y, threshold=0.5, proba=True, average=None, return_stats=False, multilabel=False):
        if proba and average is None:
            predictions = self.predict_class(model, X, threshold)

        else:
            predictions = model.predict(X)
        
        # precision & recall need to be averaged across classes for multiclass problem 
        if average is not None:
            avg_metric = average
        else:
            avg_metric = 'binary'

        precision = precision_score(y_true=y, y_pred=predictions, average=avg_metric)
        recall = recall_score(y_true=y, y_pred=predictions, average=avg_metric)
        accuracy = accuracy_score(y_true=y, y_pred=predictions)

        print("Model Performance:\n Precision: {}\n Recall: {}\n Accuracy: {}".format(precision, recall, accuracy))

        if return_stats:
            return [precision, recall, accuracy]



    '''
    Function to calculate the recall scores for each class in 
    a multilabel classification problem. Returns an array 
    representing recall for each class. 

    Params:
    ytest: Array-like
    predictions: Array-like
    '''

    def multilabel_recall(self, ytest, predictions):
        # Convert to numpy arrays
        ytest = np.asarray(ytest)
        predictions = np.asarray(predictions)

        # Create a dictionary to hold 
        n_classes = len(ytest[0])
        sample_counts = {'total_true': 0 ,'total_predicted': 0}
        recall_scores = {label: sample_counts.copy() for label in range(0, n_classes)}

        # Check consistency of arrays passed in 
        assert ytest.shape[0] == predictions.shape[0]
        assert ytest.shape[1] == predictions.shape[1]

        for index, row in enumerate(ytest):

            # Get indices of positive labels and predicted positives
            true_pos = set( np.where(ytest[index])[0] )
            predicted_pos = set ( np.where(predictions[index])[0] )

            for i in range(0, n_classes):
                if i in true_pos:
                    recall_scores[i]['total_true'] += 1

                    if i in predicted_pos:
                        recall_scores[i]['total_predicted'] += 1
        
        score_list = []

        for label in recall_scores:
            try:
                score_list.append(recall_scores[label]['total_predicted'] /
                              recall_scores[label]['total_true'])

            except ZeroDivisionError: 
                score_list.append(0)

        return score_list



    '''
    Function to calculate the recall scores for each class in 
    a multilabel classification problem. Returns an array 
    representing recall for each class. 

    Params:
    ytest: Array-like
    predictions: Array-like
    '''
    def multilabel_precision(self, ytest, predictions):
        # Convert to numpy arrays
        ytest = np.asarray(ytest)
        predictions = np.asarray(predictions)

        # Create a dictionary to hold 
        n_classes = len(ytest[0])
        sample_counts = {'total_correct': 0 ,'total_predicted': 0}
        precision_scores = {label: sample_counts.copy() for label in range(0, n_classes)}

        # Check consistency of arrays passed in 
        assert ytest.shape[0] == predictions.shape[0]
        assert ytest.shape[1] == predictions.shape[1]

        for index, row in enumerate(ytest):

            # Get indices of positive labels and predicted positives
            true_pos = set( np.where(ytest[index])[0] )
            predicted_pos = set ( np.where(predictions[index])[0] )

            for i in range(0, n_classes):
                if i in predicted_pos:
                    precision_scores[i]['total_predicted'] += 1

                    if i in true_pos:
                        precision_scores[i]['total_correct'] += 1

        score_list = []

        for label in precision_scores:
            try:    
                score_list.append(precision_scores[label]['total_correct'] /
                              precision_scores[label]['total_predicted'])
            
            except ZeroDivisionError: 
                score_list.append(0)

        return score_list


    '''
    For multilabel problems, using subset accuracy can be a harsh metric since it requires an exact match
    between the label vectors. The hamming score gives credit for partially matchin label vectors, i.e if the
    model predicts some but not all of the labels correctly. 
    '''
    def hamming_score(self, y_true, y_pred, normalize=True, sample_weight=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        acc_list = []
        
        for index, value in enumerate(y_true):
            # Return indices where the label == 1
            set_true = set( np.where(y_true[index])[0] )
            
            set_pred = set( np.where(y_pred[index])[0] )
            row_acc = None
            
            # Hamming score equation for a given sample
            if len(set_true) == 0 and len(set_pred) == 0:
                row_acc = 1
            else:
                row_acc = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
            acc_list.append(row_acc)
            
        return np.mean(acc_list)


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
    Display ROC curve and AUC for a given model. Use proba to 
    specify whether the model can predict probablities or not.

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



    '''
    Display ROC curve and AUC for a multiclass classifier, using a one 
    vs rest approach. Specify the possible classes and wrap the model 
    in a OneVsRestClassifier classifier to generate individual curves 
    for each class. 

    Params:
    model: Sklearn or Xgboost model object
    X: Array-like
    y: Array-like
    classes: Array-like
    '''
    def plot_mcroc_curve(self, model, Xtrain, ytrain, Xtest, ytest, classes):
        ytrain = label_binarize(ytrain, classes=classes)
        ytest = label_binarize(ytest, classes=classes)
        n_classes=len(classes)

        clf = OneVsRestClassifier(model)
        clf.fit(Xtrain, ytrain)
        yscore = clf.predict_proba(Xtest)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ytest[:, i], yscore[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Store number of rows and cols to create subplots
        nrows = round((len(classes)/2))
        ncols = 2
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 9), dpi=80)
        fig.subplots_adjust(hspace=0.5)
        idx=0

        for r_ind,row in enumerate(ax):
            for c_ind, col in enumerate(row):
                    col.plot(fpr[idx], tpr[idx], label='ROC curve (area = %0.2f)' % roc_auc[idx])
                    col.plot([0, 1], [0, 1], 'k--')
                    col.set_xlim([0.0, 1.0])
                    col.set_ylim([0.0, 1.05])
                    col.set_xlabel('False Positive Rate')
                    col.set_ylabel('True Positive Rate')
                    col.set_title("\n" + "ROC Curve for class " + str(idx) + "\n")
                    col.legend(loc="lower right")
                    idx+=1 



    """
    This function prints and plots the confusion matrix. Normalization 
    can be applied by setting `normalize=True`.

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
    Function to compare performance of different models/metrics. Results
    and labels should hold arrays of metrics and their names respectively.

    Params:
    results: Matrix with each row Array-like
    labels: Array-like
    xlabel: String
    ylabel: String
    titles: Array-like (String titles for each plot)
    percentages: Boolean
    '''
    def plot_compare(self, results, labels, xlabel, ylabel, titles, percentages=True):
        num_rows = int(len(results) / 2)
        num_cols = 2

        if  len(results) != len(titles):
            raise ValueError('Number of items in results should be even\
                                 and equal to number of items in titles')

        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(13, 9), dpi=80)
        fig.subplots_adjust(hspace=0.5)
        idx=0

        for r_ind,row in enumerate(ax):
            for c_ind, col in enumerate(row):
                if idx > len(labels):
                    break
                col.bar(range(len(results[idx])), [10 * stat for stat in results[idx]], align='center')
                col.set_title("\n" + titles[idx] + "\n")
                col.set_ylabel(ylabel)
                col.set_xticklabels(labels)
                col.set_xticks(range(len(labels)))
                col.set_yticklabels(round(10 * i, 3) for i in range(0, 11))
                col.set_yticks(range(11))

                for i, v in enumerate(results[idx]):
                    if percentages:
                        col.text(i - 0.1, v*10 +0.2 , str(round(v * 100, 1)) + " %", color='white')
                    else:
                        col.text(i - 0.1, v*10 + 0.2 , str(round(v, 2)) , color='white')


                idx+=1

        plt.show()