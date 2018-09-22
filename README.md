#Part 1: gender-classification
Gender Classifier/Detection by Name

#Directories (Data Set)

Data: We have multiple csv file formats contains: first name, last name, gender, race (only name in case indian )

We have created a final_names.csv which is cleaned,structured and concatenate of above multiple csv files (You can directly use this if do not want to solve data sctructure part)

######## 1. Decision Tree Classifier ############

While implementing Dicision Tree Classifier the main problem you face is overfitting (in case of skwed data), So I try to relabel data and try to bring it on similar scale(both classes approx. have similar no. of sample or may be do not have lasrge difference ). So I have copied the same data with classes which have less no. of sample and reduced the other sample;

then we compute our feature and vectorize them ;

Randomly shuffle the data and Split the features and classes into 3 partitions, call them Trainining set, Cross Validation set, and Test Set

Why I am taking cross validation set is because you can check you overfitting on Cross validation set and then redesign your model and improve its accuracy on cross validation and then you can test on test set. 

    Below are the results I get from Dicision tree classifier:
            Accuracy on Training set: 98.24 % ;
            Accuracy on CV set: 95.56 % ;
            Accuracy on Test set: 93.76 % ;
            Pricision : 0.974  ;
            Recall : 0.623  ;       ## Recall is low because we have skewed data and its causing overfitting
            F Score: 0.76 ;

############# 2. Naive Bayes Classifier  ##################

After a little research I found a way to how to fit the data with strings matrix to fit into Naive Bayes Classifier

Splited the original Data (Skewed Data - final_names.csv) and implemented Naive Bayes to see the results and it was quit good as its completely based on probability. 

Randomly shuffled the data and split into 70% data as training set and 30% as test set:

implemented Naive Bayes Classifier (NBC) using training data set and used test data set to check the accuracy/f score of model.

    Below are the results:
            Accuracy on test set: 93.10 % ;
            Pricision : 0.967  ;
            Recall : 0.95 ;
            F Score : 0.96 ;
                    
Gender classifier build using first name of any person. 

#Part2: race-classifier

implemented using Decision Tree classifier for multiclassification 

    Below are the results:
        Accuracy on test set: 87.4 % ;
        Pricision: 0.76 ;
        Recal: 0.73 ;
        F1 Score: 0.74 ;

##Cheers##
