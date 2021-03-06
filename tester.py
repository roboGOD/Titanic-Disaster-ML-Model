from sklearn.model_selection import StratifiedShuffleSplit
PERF_FORMAT_STRING = "\
\nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\n\
Recall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\n\
False negatives: {:4d}\nTrue negatives: {:4d}"


def test_classifier(clf, features, labels, folds=1000):

	cv = StratifiedShuffleSplit(n_splits=folds, random_state = 42)
	true_negatives = 0
	false_negatives = 0
	true_positives = 0
	false_positives = 0
	for train_idx, test_idx in cv.split(features, labels): 
	    features_train = []
	    features_test  = []
	    labels_train   = []
	    labels_test    = []
	    for ii in train_idx:
	        features_train.append( features[ii] )
	        labels_train.append( labels[ii] )
	    for jj in test_idx:
	        features_test.append( features[jj] )
	        labels_test.append( labels[jj] )
	    
	    ### fit the classifier using training set, and test on test set
	    clf.fit(features_train, labels_train)
	    predictions = clf.predict(features_test)
	    for prediction, truth in zip(predictions, labels_test):
	        if prediction == 0 and truth == 0:
	            true_negatives += 1
	        elif prediction == 0 and truth == 1:
	            false_negatives += 1
	        elif prediction == 1 and truth == 0:
	            false_positives += 1
	        elif prediction == 1 and truth == 1:
	            true_positives += 1
	        else:
	            print "Warning: Found a predicted label not == 0 or 1."
	            print "All predictions should take value 0 or 1."
	            print "Evaluating performance for processed predictions:"
	            break
	try:
	    total_predictions = true_negatives + false_negatives + false_positives + true_positives
	    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	    precision = 1.0*true_positives/(true_positives+false_positives)
	    recall = 1.0*true_positives/(true_positives+false_negatives)
	    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
	    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
	    print ""
	    print clf
	    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
	    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
	    print ""
	    return accuracy
	except:
	    print "Got a divide by zero when trying out:", clf
	    print "Precision or recall may be undefined due to a lack of true positive predicitons."
	    return 0.0
