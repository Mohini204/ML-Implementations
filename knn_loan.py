

from random import randrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split   
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


#import the csv into data frame
df = pd.read_csv(r'/Users/venkatasamyuktamalapaka/Downloads/final_project/loan.csv')
data =pd.DataFrame(df)
print(data)

df.drop('Loan_ID', axis=1, inplace=True)

print(df)

# Select columns for one-hot encoding
columns_to_encode = ['Gender', 'Married', 'Education','Self_Employed','Dependents','Property_Area']

df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})

# Perform one-hot encoding on selected columns
df_encoded = pd.get_dummies(df, columns=columns_to_encode,dtype=int)

print(df_encoded)

def absolute_maximum_scale(series):
    return series / series.abs().max()

for col in df_encoded.columns:
    df_encoded[col] = absolute_maximum_scale(df_encoded[col])

print(df_encoded)

train_result ={}
test_result={}
train_f1={}
test_f1={}

def k_folds(data_to_load):
 folds = {} 
 fold_size = int(len(data_to_load) / 10)
 grouped = data_to_load.groupby(['Loan_Status'])
 print(grouped)

 for i in range(10):
    fold = {}
    for j in range(0,fold_size,2):
        index_0 = randrange(len(grouped.get_group(0)))
        index_1 = randrange(len(grouped.get_group(1)))
        fold [j] = ((grouped.get_group(0)).iloc[index_0].values)
        fold [j+1] = ((grouped.get_group(1)).iloc[index_1].values)


    folds[i]=fold
 return folds


def train_test_fold(folds,df):
    for i in range(10):
        test_fold = (pd.DataFrame(folds[i])).transpose()
        test_fold.columns=list(df)
        fol = []
        for j in range(10):
            if(i==j):
                continue
            else:
                fol.append((pd.DataFrame(folds[j])).transpose())
        train_fold = pd.concat(fol)
        train_fold.columns=list(df)
        print(train_fold)
        print("-----------------------------------")
        print(test_fold)
        print("-----------------------------------")
        return train_fold,test_fold
    





def training_dataset(Predict_x,training_x,Predicted_y,Prediction_y,k):
    num_of_correct = 0
    predicted =[]
    for i in range(len(Predict_x)):
        
        majority = []
        store =[]
        Predicted_disease=[]
        for j in range(len(training_x)):
            if j==i:
                continue
            # print(training_x[i],Predict_x[i])
            distance = np.sqrt(np.sum((training_x[j]-Predict_x[i])**2))
            store.append([distance,Predicted_y[j]])
            
        store.sort()
        store = store[:k]
            # print(store)
            

        for h in store:
            majority.append(h[1])

        Prediction = Counter(majority).most_common(1)[0][0]
        

        Predicted_disease.append(Prediction)
        predicted.append(Prediction)
        if Prediction_y[i] == Predicted_disease:
                num_of_correct += 1 
    
        
    accuracy_data_set=[(round(num_of_correct/len(Prediction_y),4))]
    f1 = f1_score(Prediction_y, predicted)
    return f1,accuracy_data_set





def main():

    for k in range(1,13,2):
        Accuracy_train_result=[]
        Accuracy_test_result=[]
        f1_train_result =[]
        f1_test_result=[]
        for m in range(1,11):
            df__after_shuffle = (sklearn.utils.shuffle(df_encoded, n_samples=150, random_state=40))
            
            folds_after_shuffle = k_folds(df__after_shuffle)
            train, test = train_test_fold(folds_after_shuffle,df__after_shuffle)
            y_train = train['Loan_Status'][:].values
            y_test = test['Loan_Status'][:].values
            X_train = train.drop(['Loan_Status'], axis=1)[:].values
            X_test = test.drop(['Loan_Status'], axis=1)[:].values
            # print(y_train)
            # print("-----------------------------------")
            # print(y_test)
            # print("-----------------------------------")
            # print(X_train)
            # print("-----------------------------------")
            # print(X_test)
            # print("-----------------------------------")
            
            

            
            f1_train , acc_train = training_dataset(X_train,X_train,y_train,y_train,k)
            f1_test , acc_test = training_dataset(X_test,X_train,y_train,y_test,k)
            # Accuracy_train_result.append(training_dataset(X_train,X_train,y_train,y_train,k))
            # Accuracy_test_result.append(training_dataset(X_test,X_train,y_train,y_test,k))
            Accuracy_train_result.append(acc_train)
            Accuracy_test_result.append(acc_test)
            f1_train_result.append(f1_train)
            f1_test_result.append(f1_test)

            

            
            
        train_result[k]=Accuracy_train_result
        
        test_result[k]=Accuracy_test_result

        train_f1[k] = (sum(f1_train_result) / len(f1_train_result)) 
        test_f1[k] = (sum(f1_test_result) / len(f1_test_result)) 
    # print(train_result)
    # print(test_result)
    print(train_f1,'fruit')
    print(test_f1)


    standard_deviation=[]
    x_axis=[]
    avg=[]
    
    for k_value in train_result:
        x_axis.append(k_value)
        sd_array=[]
        for i in range(0,10):
            sd_array.append(train_result.get(k_value)[i])
        sd1=np.std(sd_array, ddof=1) / np.sqrt(len(sd_array))
        standard_deviation.append(round(sd1,4))
        lst_avg = np.average(sd_array)
        avg.append(round(lst_avg,4))

    
    
   

    standard_deviation_test=[]
    x_axis_test=[]
    avg_test=[]
    
    for k_value in test_result:
        x_axis_test.append(k_value)
        sd_array_test=[]
        for i in range(0,10):
            sd_array_test.append(test_result.get(k_value)[i])
        sd1_test=np.std(sd_array_test, ddof=1) / np.sqrt(len(sd_array_test))
        standard_deviation_test.append(round(sd1_test,4))
        lst_avg_test = np.average(sd_array_test)
        avg_test.append(round(lst_avg_test,4))


    print(avg_test,x_axis_test,standard_deviation_test)
    # Calculate the F1 score
    print(x_axis_test,train_f1.values(),test_f1.values(),avg,avg_test)
  

    plt.errorbar(x_axis, avg, yerr=standard_deviation,capsize=5, color='blue',  markersize=4, linewidth=1,linestyle='-', fmt='.k')
    xi = list(range(len(x_axis)))
    plt.ylim(0.0,1.0)       
    plt.xlabel('K-value')
    plt.ylabel('Average accuracy')
    plt.title('Line plot with error bars of Accuracy of Training data on Loan dataset')
    plt.xticks(x_axis,x_axis)
    plt.show()

    
    #run the plot on training data
    plt.errorbar(x_axis_test, avg_test, yerr=standard_deviation_test,capsize=5, color='green',  markersize=4, linewidth=1,linestyle='-', fmt='.k')
    xi = list(range(len(x_axis_test)))
    plt.ylim(0.0,1.0)       
    plt.xlabel('K-value')
    plt.ylabel('Average accuracy')
    plt.title('Line plot with error bars of Accuracy of Testing data on Loan dataset')
    plt.xticks(x_axis_test,x_axis_test)

    plt.show()

    plt.errorbar(x_axis_test, train_f1.values(), color='b',  markersize=4, linewidth=1,linestyle='-', fmt='.k')
    xi = list(range(len(x_axis)))
    plt.ylim(0.0,1.0)       
    plt.xlabel('K-value')
    plt.ylabel('Average F1 score')
    plt.title('Line plot of F1 score of Training data on Loan dataset')
    
    plt.show()

    
    #run the plot on training data
    plt.errorbar(x_axis_test, test_f1.values(), color='green',  markersize=4, linewidth=1,linestyle='-', fmt='.k')
    xi = list(range(len(x_axis_test)))
    plt.ylim(0.0,1.0)       
    plt.xlabel('K-value')
    plt.ylabel('Average F1 score')
    plt.title('Line plot of F1 score of Test data on Loan dataset')
    

    plt.show()




        


if __name__ == "__main__":
    main()
            
