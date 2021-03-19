import pandas as pd
import numpy as np

# The original data set obtained from Jessica via sunil had some errors as well as extra columns that are not required. 
# This code removes those and saves a new file called 'experimental_data_set.csv' that should be loaded 
# when conducting future experiments with this data. 

if __name__ == '__main__':
    dfvz = pd.read_csv("/DataFiles/FullDataSet.csv", engine='c')
    dfvz.loc[dfvz['vhse1.4'] == -0.69,'vhse1.4'] = 0.69
    dfvz.loc[dfvz['vhse2.4'] == -0.69,'vhse2.4'] = 0.69
    dfvz.loc[dfvz['vhse3.4'] == -0.69,'vhse3.4'] = 0.69
    dfvz.loc[dfvz['vhse4.4'] == -0.69,'vhse4.4'] = 0.69
    dfvz.drop(labels=['z3.linear.pred', 
                      'z3.svm.pred', 
                      'z3.tree.pred', 
                      'z3.knn.pred', 
                      'vhse.linear.pred', 
                      'vhse.svm.pred',
                      'vhse.tree.pred', 
                      'vhse.knn.pred'], axis=1, inplace=True)
    dfvz.to_csv('/home/sharyarmemon/Documents/GitHub/GESAR-V2/DataFiles/experimental_data_set.csv',
                index=True)