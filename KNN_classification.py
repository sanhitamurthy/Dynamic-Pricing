import seaborn as sns
sns.set(color_codes=True)
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

mercaritrain = pd.read_csv('C://Users//Ravi//Desktop//Mercari//train.tsv//train.tsv',delimiter='\t',encoding='utf-8')
mercaritest = pd.read_csv('C://Users//Ravi//Desktop//Mercari//test.tsv//test.tsv',delimiter='\t',encoding='utf-8')

frametrain = pd.DataFrame(mercaritrain)
frametrain=frametrain[0:1000]
frametest = pd.DataFrame(mercaritest)

frametrain=frametrain.fillna(0)



def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")


frametrain['general_cat'], frametrain['subcat_1'], frametrain['subcat_2'] = \
zip(*frametrain['category_name'].apply(lambda x: split_cat(x)))
frametrain.head()


frametrain=frametrain.drop('category_name',axis=1)
frametrain.head()


def divide_cats(data):
    if( data <=5):
        return "CAT5"
    if( 5 < data <=10):
        return "CAT10"
    if( 10 < data <=15):
        return "CAT15"
    if( 15 < data <=20):
        return "CAT20"
    if(20 < data <=25):
        return "CAT25"
    if( 25< data <=30):
        return "CAT30"
    if( 30 < data <=35):
        return "CAT35"
    if( 35 < data <=40):
        return "CAT40"
    if( 40 < data <=45):
        return "CAT45"
    if(45 < data <=50):
        return "CAT50"
    return "CATOTHER"

def Knnmethod(frametrain1,price_cats):
    frametraink1 = frametrain1.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(frametraink1, price_cats, test_size=0.20, random_state=42)
    accuracy_array = []
    k_array = []
    for k in range(1,100,3):
        knn = KNeighborsClassifier(n_neighbors=k)
        accuracy = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        accuracy_array.append(accuracy.mean())
        k_array.append(k)
    print (accuracy_array)
    print(k_array)
    class_error = 1.0 - np.array(accuracy_array)
    plt.plot(k_array, class_error)
    plt.xlabel('K')
    plt.ylabel('Classification Error')
    plt.show()
    min_ind = np.argmin(class_error)
    OptK = k_array[min_ind]
    print ("Optimal value of K is %d " %  OptK)
    knn = KNeighborsClassifier(n_neighbors=OptK)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict
    pred = knn.predict(X_test)

    # evaluate accuracy
    print("accuracy_score",accuracy_score(y_test, pred))

frametrain['price_cats'] =frametrain.price.map(lambda x : divide_cats(x))
frametrain.head()


for column in frametrain:
    if frametrain[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        frametrain[column] = le.fit_transform(frametrain[column].astype(str))

print(frametrain.info())
frametrain.head()


plt.matshow(frametrain.corr())
corr_col=frametrain.corr()
corr_col=corr_col.fillna(0)
plt.figure(figsize=(12, 10))
plt.imshow(corr_col, cmap='viridis_r', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_col)), corr_col.columns, rotation='vertical')
plt.yticks(range(len(corr_col)), corr_col.columns);
plt.suptitle(' Correlations Heat Map', fontsize=12, fontweight='bold')
plt.show()

price_cats = np.array(frametrain.pop('price_cats'))

frametrain=frametrain.drop('train_id',axis=1)

frametrain=frametrain.drop('price',axis=1)

Knnmethod(frametrain,price_cats)


frametrain1=frametrain[['brand_name','general_cat']]
Knnmethod(frametrain1,price_cats)


frametrain1=frametrain[['general_cat','subcat_1','brand_name']]
Knnmethod(frametrain1,price_cats)

frametrain1=frametrain[['general_cat','subcat_1','item_condition_id']]
Knnmethod(frametrain1,price_cats)


frametrain1=frametrain[['general_cat','subcat_1','item_condition_id','brand_name']]
Knnmethod(frametrain1,price_cats)

frametrain1=frametrain[['general_cat','subcat_1','item_condition_id','brand_name','shipping']]
Knnmethod(frametrain1,price_cats)







