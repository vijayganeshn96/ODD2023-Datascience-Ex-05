# Ex:05 Feature Generation

## AIM

To read the given data and perform  Feature Encoding & Scaling process and save the data to a file.

## ALGORITHM

1. Read the given Data.
2. Clean the Data Set using Data Cleaning Process
3. Apply Feature Generation techniques to all the feature of the data set
4. Save the data to the file

## CODE
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/12c3924e-10f3-4a92-b821-b2c5f1307ef2)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pn=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pn])
e1.fit_transform(df[['ord_2']])
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/2f939788-54f1-4bc1-95d6-37c6d9feef3a)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/d7ff6731-f363-4744-8586-7f428f22284e)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()#sparse=False
df2=df.copy()
```
```
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/b82a6148-20fc-4c49-94f6-d8e613ef50ac)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/4419eef3-0a17-445c-b658-45bbba387088)

```
pip install category_encoders
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/67ae0f42-5ff1-453b-9c92-d23da0713146)

```
from category_encoders import BinaryEncoder
be=BinaryEncoder()
dfb=df.copy()
nd=be.fit_transform(df['ord_2'])
dfb=pd.concat([dfb,nd],axis=1)
dfb
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/e01c93af-e7f7-421e-ba78-9c41fc60e917)

```
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/de9cc6ff-d4d4-4731-afde-4d6428520640)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/df74d900-a80b-46b8-af9a-0ed052212eab)

```
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/84bafc98-fa1f-4611-9a48-7a046f681246)

```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/97ebe7d7-31ae-4eb4-995e-365ae1dc4dec)

```
min_vals=np.min(np.abs(df[['Height','Weight']]))
min_vals
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/277e9a8c-2462-48c1-9431-e7cc491155c3)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=df.copy()
df1[["Height","Weight"]]=sc.fit_transform(df1[["Height","Weight"]])
df1.head(10)
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/32e467c5-56e3-4159-8624-5b227f683895)

```
max_val=np.max(np.abs(df1[['Height','Weight']]))
max_val
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/f1a2ceb5-a902-46e2-9637-31d6b714ed81)

```
min_val=np.min(np.abs(df1[['Height','Weight']]))
min_val
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/250618a6-1efa-4b9f-829d-2d934600e3ab)

```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=df.copy()
df2[["Height","Weight"]]=sc.fit_transform(df2[["Height","Weight"]])
df2
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/a230f93a-54be-4d8d-8e98-ad64204aec99)

```
from sklearn.preprocessing import Normalizer
sc=Normalizer()
df3=df.copy()
df3[["Height","Weight"]]=sc.fit_transform(df3[["Height","Weight"]])
df3
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/a13efa61-fcfb-49e5-ab5a-89628fe5e4cf)

```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df4=df.copy()
df4[["Height","Weight"]]=sc.fit_transform(df4[["Height","Weight"]])
df4
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/630f9699-8db1-470a-8983-7961cc68ddd5)

```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=df.copy()
df5[["Height","Weight"]]=sc.fit_transform(df5[["Height","Weight"]])
df5
```
![image](https://github.com/Pranav-AJ/ODD2023-Datascience-Ex-05/assets/118904526/9f90beeb-9d1d-41e2-afc8-4e347161504b)

## RESULT:
This Program has run successfully.
