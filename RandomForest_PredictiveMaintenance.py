# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:59:24 2023

@author: memoh
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
import math
import random




def absolute_maximum_scale(series):
    return series / series.abs().max()

    
#calculating entropy
def entropy(labels):
    
    val, count = np.unique(labels, return_counts=True)
    prob = count/len(labels)
    entropy = np.sum([((-p)*np.log2(p)) for p in prob])
    return entropy


def entropy1(labels):
    # Count the number of occurrences for each unique label
    count = np.bincount(labels)
    
    prob = count/len(labels)
    entropy = np.sum([((-p)*np.log2(p)) for p in prob])
    
    return entropy



#calculating information gain
def infoGain(data, featCol, labels):
    
    initialEntropy = entropy(labels)
    splitEntropy = 0
    colName = featCol.name
    
    val, count = np.unique(featCol, return_counts=True)
    for v in val:
        split = data.loc[data[colName]==v, 'Failure Type']
        splitEntropy += (len(split)/len(featCol))*entropy(split)
    
    
    return initialEntropy-splitEntropy



#choose the best feature amongst the available set of features        
def chooseBestFeat(data, featNames, target):
    
    featLen = len(featNames)
    selectedFeatLen = int(math.sqrt(featLen))
    
    # Shuffle the list of features randomly
    random.shuffle(featNames)
    
    # Get the first 'selectedFeatLen' features
    selectedFeat = featNames[:selectedFeatLen]
    
    gain, bestGain, bestIdx = 0, 0, -1
    
    for i in range(len(selectedFeat)):
        gain = infoGain(data, data.loc[:, selectedFeat[i]], target)
        if gain > bestGain:
            bestGain = gain
            bestIdx = i
            
    #print(selectedFeat[bestIdx], bestGain)
    # print(bestGain)
    
    if bestGain<0.1:
        return ''
    else:
        return selectedFeat[bestIdx]


        
#build the decision tree        
def buildTree(data, featNames, target): 
    
    # print(target)
    
    #base cases
    #if there is only 1 label left in the target column
    if len(np.unique(target)) == 1:
        # print('dhakka1?')
        return target.iloc[0]
    
    elif len(target) <= 4:
        # print('dhakka2?')
        values = target.values
        unique_labels, counts = np.unique(values, return_counts=True) 
        #Find the label with the highest count
        max_count = np.max(counts)
        max_labels = unique_labels[counts == max_count]
        decision = max_labels[0]
        # print(decision)

        return decision 


    bestFeat = chooseBestFeat(data, featNames, target)
    
    if bestFeat == '':
        values = target.values
        unique_labels, counts = np.unique(values, return_counts=True) 
        #Find the label with the highest count
        max_count = np.max(counts)
        max_labels = unique_labels[counts == max_count]
        decision = max_labels[0]
        return decision
    else:
        
        #create the tree based on best feature found; using dict as the representation of tree
        tree = {bestFeat: {}}
        #start depth first traversal on children
        
        if isinstance(data.iloc[0][bestFeat], (int,float)):
            
            # calculate the average of the values of the best feature column
            avg = np.mean(data[bestFeat])
            # split the data into two groups based on the average value
            leftData = data[data[bestFeat] <= avg]
            rightData = data[data[bestFeat] > avg]
            leftTarget = data.loc[data[bestFeat] <= avg, 'Failure Type'] 
            rightTarget = data.loc[data[bestFeat] > avg, 'Failure Type'] 
            
            if len(leftTarget) == 0:
                values = target.values
                unique_labels, counts = np.unique(values, return_counts=True) 
                #Find the label with the highest count
                max_count = np.max(counts)
                max_labels = unique_labels[counts == max_count]
                majority = max_labels[0]
                tree[bestFeat][avg] = majority
            else:
                leftNode = buildTree(leftData, featNames, leftTarget)
                # attach child nodes to parent node
                tree[bestFeat][avg] = leftNode
            if len(rightTarget) == 0:
                values = target.values
                unique_labels, counts = np.unique(values, return_counts=True) 
                #Find the label with the highest count
                max_count = np.max(counts)
                max_labels = unique_labels[counts == max_count]
                majority = max_labels[0]
                tree[bestFeat]['>{}'.format(avg)] = majority
            else:
                rightNode = buildTree(rightData, featNames, rightTarget)            
                tree[bestFeat]['>{}'.format(avg)] = rightNode


        else:
            values = np.unique(data[bestFeat])
            # print('dhakka?')
            #iterate on children to create the tree further
            for v in values: 
                
                childData = data[data[bestFeat] == v]
                childData = childData.copy()
                #get target column values as per the best feature column unique values
                target_child = data.loc[data[bestFeat] == v, 'Failure Type']
                if len(target_child) == 0:
                    # if the class set is empty in the split, return majority class in the original dataset
                    majority = np.argmax(np.unique(target, return_counts=True)[1])
                    tree[bestFeat][v] = majority
                else:
                    childTree = buildTree(childData, featNames, target_child)
                    #attach childTree to parent
                    tree[bestFeat][v] = childTree
            
        
        return tree


def prediction(dt, inst):
    
    if not isinstance(dt, dict): 
    #     print('dt', dt)     
        return dt
        
    for f in dt.keys():
        
        ans = inst[f]
        if isinstance(ans, (float,int)):
            
            left_node_name = list(dt[f].keys())[0]
            right_node_name = '>{}'.format(left_node_name)
            
            if ans <= float(left_node_name):
                ans = left_node_name
            else:
                ans = right_node_name
                
        if ans in dt[f]:
            
            smallTree = dt[f][ans]
            # print(smallTree)
            
            if type(smallTree) is dict:
                
                predictPred = prediction(smallTree,inst)
                return predictPred
            
            elif not isinstance(smallTree, dict):
                # print('inside smallTree',smallTree)
                return smallTree
            
        else:
            alternative_keys = [key for key in dt[f].keys()]
            alt_key = random.choice(alternative_keys)
            
            smallTree = dt[f][alt_key]
            # print(smallTree)
            
            if type(smallTree) is dict:
                
                predictPred = prediction(smallTree,inst)
                return predictPred
            
            elif not isinstance(smallTree, dict):
                # print('inside smallTree',smallTree)
                return smallTree
            
        
    

def bootstrap(data):
    
    bootstrap_size = len(data)
    
    # Calculate the number of rows to remove from the original dataset
    removeNum = int(1/3 * len(data))
    
    # Randomly select which rows to remove from the original dataset
    removeRows = np.random.choice(data.index, size=removeNum, replace=False)
    
    # Create a copy of the original dataset
    bootstrap = data.copy()
    
    # Remove the randomly selected rows from the bootstrap
    bootstrap.drop(removeRows, inplace=True)
    
    # Create a list of indices for the rows to be duplicated in the bootstrap
    repeatRows = np.random.choice(bootstrap.index, size=removeNum, replace=True)
    
    # Duplicate the selected rows in the bootstrap
    bootstrap = bootstrap.append(bootstrap.loc[repeatRows])
    
    # Shuffle the rows in the bootstrap
    bootstrap = bootstrap.sample(frac=1).reset_index(drop=True)
    
    return bootstrap
        
        
    
def kFoldValidation(nTree):  
    
    #read the data file
    df = pd.read_csv(r"C:\Users\memoh\OneDrive\Desktop\Spring23\589\final_project\predictive_maintenance.csv")
    
    df = df.drop('Product ID', axis=1)
    df = df.drop('UDI', axis=1)
    
    df['Failure Type'] = df['Failure Type'].map({'No Failure': 1, 'Power Failure': 0, 'Tool Wear Failure': 2, 'Overstrain Failure':3, 'Random Failures':4, 'Heat Dissipation Failure':5})
    
    df['Air temperature [K]'] = absolute_maximum_scale(df['Air temperature [K]'])
    df['Process temperature [K]'] = absolute_maximum_scale(df['Process temperature [K]'])
    df['Rotational speed [rpm]'] = absolute_maximum_scale(df['Rotational speed [rpm]'])
    df['Torque [Nm]'] = absolute_maximum_scale(df['Torque [Nm]'])
    df['Tool wear [min]'] = absolute_maximum_scale(df['Tool wear [min]'])
    df['Target'] = absolute_maximum_scale(df['Target'])
    
    
    # Split the dataframe based on the class column
    dfClass1 = df[df["Failure Type"] == 0]
    dfClass2 = df[df["Failure Type"] == 1]
    dfClass3 = df[df["Failure Type"] == 2]
    dfClass4 = df[df["Failure Type"] == 3]
    dfClass5 = df[df["Failure Type"] == 4]
    dfClass6 = df[df["Failure Type"] == 5]
    
    #split each dataset into 10 parts
    class1Parts = np.array_split(dfClass1, 10)
    class2Parts = np.array_split(dfClass2, 10)
    class3Parts = np.array_split(dfClass3, 10)
    class4Parts = np.array_split(dfClass4, 10)
    class5Parts = np.array_split(dfClass5, 10)
    class6Parts = np.array_split(dfClass6, 10)

    accList = []
    f1List = []
    
    
    for i in range(10):
        
        #concatenate ith part to make the test dataset for the (i+1)th fold
        testFoldAll = pd.concat([class1Parts[i], class2Parts[i], class3Parts[i], class4Parts[i], class5Parts[i], class6Parts[i]])
        testFold = testFoldAll.iloc[:, 1:].values
        
        actualLabel = testFoldAll["Failure Type"]
        
        # concatenate rest of the parts to make the training set
        trainFold = pd.concat([pd.concat(class1Parts[:i]+class1Parts[i+1:]), 
                               pd.concat(class2Parts[:i]+class2Parts[i+1:]),
                               pd.concat(class3Parts[:i]+class3Parts[i+1:]), 
                               pd.concat(class4Parts[:i]+class4Parts[i+1:]), 
                               pd.concat(class5Parts[:i]+class5Parts[i+1:]),
                               pd.concat(class6Parts[:i]+class6Parts[i+1:])])

        # Shuffle the train_folds dataset
        trainShuffled = shuffle(trainFold)
        
        predBootstrapList = []
        
        for j in range(nTree):
            
            bootData = bootstrap(trainShuffled)
            featNames = bootData.iloc[:, 1:].columns.tolist()
            labelCol = bootData["Failure Type"]
            
            builtTree = buildTree(bootData, featNames, labelCol)
            # print('what', builtTree)

            predList = []
            
            featNames = bootData.iloc[:, 1:].columns.tolist()
            
            
            #loop to get prediction for every instance of the dataset
            for m in range(len(testFold)):
                #len(testFold)
                
                #creating a dict for passing every instance row with the column names
                entry = {}
                n = 0
                for f in featNames:
                    entry[f] = testFold[m,n]
                    n+=1
                    
                pred = prediction(builtTree, entry)
                predList.append(pred)
                
            predBootstrapList.append(predList)
            
        # print(predList)
        

        maj_vote = pd.Series(dtype=int)
        
        for items in zip(*predBootstrapList):
            counter = Counter(items)
            maj_vote = maj_vote.append(pd.Series([counter.most_common(1)[0][0]]))
            
        actualLabel = actualLabel.reset_index(drop=True)
        maj_vote = maj_vote.reset_index(drop=True)
        
        
        f1 = f1_score(actualLabel, maj_vote, average='weighted')
        f1List.append(f1)
        
        accuracy = accuracy_score(actualLabel, maj_vote)
        accList.append(accuracy)
        

        
    #     cm = {'TP':0, 'FN':0, 'FP':0, 'TN':0}
    #     cm['TP'] = ((actualLabel == 1) & (maj_vote == 1)).sum()
    #     cm['FN'] = ((actualLabel == 1) & (maj_vote == 0)).sum()
    #     cm['FP'] = ((actualLabel == 0) & (maj_vote == 1)).sum()
    #     cm['TN'] = ((actualLabel == 0) & (maj_vote == 0)).sum()
        
        
    #     accuracy = (cm['TP'] + cm['TN']) / (cm['TP'] + cm['TN'] + cm['FP'] + cm['FN'])
        
    #     precision = cm['TP'] / (cm['TP'] + cm['FP'])
        
    #     recall = cm['TP'] / (cm['TP'] + cm['FN'])
        
    #     f1 = 2*(precision*recall)/(precision+recall)
        
    #     accList.append(accuracy)
    #     precList.append(precision)
    #     recList.append(recall)
    #     f1List.append(f1)
        
    return sum(accList)/10, sum(f1List)/10
        
 
    

def main():
    
    plotAcc = []
    # plotPrec = []
    # plotRec = []
    plotF1 = []
    
    for i in range(7):
        ntreelist = [1,5,10,20,30,40,50]
        acc, f1 = kFoldValidation(ntreelist[i])
        print(acc, f1)
        print('done',ntreelist[i])
        
        plotAcc.append(acc)
        # plotPrec.append(prec)
        # plotRec.append(rec)
        plotF1.append(f1)
    
    # Plot the data using plt.plot() function multiple times
    plt.plot(ntreelist, plotAcc, color='red', marker='o', label='Accuracy')
    # plt.plot(ntreelist, plotPrec, color='green', marker='s', label='Precision')
    # plt.plot(ntreelist, plotRec, color='blue', marker='^', label='Recall')
    plt.plot(ntreelist, plotF1, color='purple', marker='*', label='F1 Score')
    
    # Set the title and axis labels
    plt.title('Metrics Plot Predictive Maintenance Dataset')
    plt.xlabel('Number of Trees')
    plt.ylabel('Metrics of Random Forest')
    
    # Show the legend
    plt.legend()
    
    # Show the plot
    plt.show()



if __name__ == "__main__":
    main()


    
    
    

    
    
    
    
    






    
    
    

    
    
    
    
    

