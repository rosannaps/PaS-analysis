# -*- coding: utf-8 -*-
"""
@author: Alexandra Tzilivaki
"""
import pandas as pd
from sklearn.impute import KNNImputer

data=pd.read_excel('pasDFA201104.xlsx') #load data
data=data.set_index(data['CellID'])

#delete useless columns
del(data['CellID'])
del(data['Unnamed: 0'])
# del(data['location'])
data=data.loc[data.staining != 'VGAT'] #delete the 4 VGAT neurons. we are not sure of their ID.


#import cellids with NO connectivity info
no_connectivity=pd.read_csv('cellsWITHOUTconnectivity2.csv')
no_connectivity=no_connectivity.set_index('0')

no_connectivity=data[data.index.isin(no_connectivity.index)]

#which cells with no connectivity are of unknown staining?
no_connectivity=no_connectivity[no_connectivity.staining == 'Unknown']

#these are useless delete them from the dataset.
data=data[~data.index.isin(no_connectivity.index)]

##################################################################################
## imputation#####################################################################


dfWFS1=data.loc[data['staining'] == 'WFS1']
dfReelin=data.loc[data['staining'] == 'Reelin']
dfPV=data.loc[data['staining'] == 'PV']  

# for the cells with known staining, we replace nans with the mean of each label-type feature.
dfWFS1=dfWFS1.fillna(dfWFS1.mean())
dfReelin=dfReelin.fillna(dfReelin.mean())
dfPV=dfPV.fillna(dfPV.mean())

dfstainall=pd.concat([dfWFS1,dfReelin,dfPV])
dfstainall.isna().any()  #includes Cellid #check. if all false we are doing well.

dfUnknown=data.loc[data['staining']=='Unknown']


allcells=pd.concat([dfstainall,dfUnknown])

#for the cells with no staining information we replace nans by performing knn imputation
#using sklearn

imputdata=allcells.drop(['staining'], axis=1)
imputer = KNNImputer(n_neighbors=1, weights="uniform")
lala=imputer.fit_transform(imputdata)

finaldata=pd.DataFrame.from_records(lala)
finaldata.isna().any() #check. if all false we are doing well.

finaldata.columns=imputdata.columns
finaldata=finaldata.set_index(imputdata.index)
finaldata['staining']=allcells.staining

finaldata.to_csv('finaldatadecember2020.csv')


