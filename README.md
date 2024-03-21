### ENTER YOUR NAME : RAKESH J.S
### ENTER YOUR REGISTER NO : 212222230115
### EX. NO.1
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
``` PYTHON
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
### DATASET
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/fb93df05-0d85-46cd-8b72-f1b5dbb2f45e)
### X VALUES
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/6f82f7d9-a77a-4b07-b1c1-9e00adeff7ee)
### Y VALUES
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/3401a399-4672-416c-9735-0762b076671e)
### NULL
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/d91cc7c2-d007-4015-b247-fb31ac81acf4)
### DUPLICATE
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/a4ee7771-e2c4-4c37-89b3-975b225f734c)
### DESCRIBE
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/27bb11f8-6d9f-4f46-9ee4-4b3a4ab026d1)
### DATASET AFTER DROPPING
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/f224cb6d-38d7-43a1-9fbf-41f927583e23)
### NORMALIZE DATASET
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/ed14c22a-b340-481d-8f96-240e0e18c8e7)
### X TRAIN
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/69fa3229-5788-4c90-86e8-9614d6b00aaa)
### X TEST
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/6dbf4cca-969f-419d-82d4-f95c91a9e18d)
### LENGTH
![image](https://github.com/MukeshVelmurugan/Ex-1-NN/assets/118707363/63aa77f8-f224-4657-8f1c-a79496aa59bc)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
