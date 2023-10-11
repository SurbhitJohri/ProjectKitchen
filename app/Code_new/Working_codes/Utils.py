# # PROJECT KITCHEN

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023

# # CONTENTS
#   This file utils.py has outlier and classification level split functions.
#   The function call of this is present in kitchen.py file.
#

from Libraries import *
from config import *


#To treat outliers
def outlier(data,variable,iqrfactor,replace_by): 
        SS = ((data[variable] - (data[variable].mean()))**2)
        MSE = (SS.sum())/len(data[variable])
        RMSE_Before = math.sqrt(MSE)
        Q1 = data[variable].quantile(0.25)  
        Q3 = data[variable].quantile(0.75)
        IQR = Q3 - Q1
        Min = Q1-(iqrfactor*IQR)
        Max = Q3+(iqrfactor*IQR)
        Mean = data[variable][(data[variable]>=Min) & (data[variable]<=Max)].mean()
        Median = data[variable][(data[variable]>=Min)&(data[variable]<=Max)].median()
        Mode = data[variable][(data[variable]>=Min)&(data[variable]<=Max)].mode()[0]
        #print('IQR is',IQR,'\nMin is',Min,'\nMax is',Max,'\nMean is',Mean,'\nMedian is',Median,'\nMode is',Mode)
        if replace_by == 'mean':        
            data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),Mean,data[variable])
        elif replace_by == 'median':
            data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),Median,data[variable])
        elif replace_by == 'blank':
            data[variable]=np.where((data[variable]<Min)|(data[variable]>Max),np.nan,data[variable])
        SS = ((data[variable] - (data[variable].mean()))**2)
        MSE = (SS.sum())/len(data[variable])
        RMSE_After = math.sqrt(MSE)
        #print('RMSE before is',RMSE_Before,'\nRMSE after is',RMSE_After)
        return data[variable]

#To split classification levels for each spend areas
def levelone(spend,df_txt):
    level_size = len(df_txt['Classification'].str.split('-', expand=True).columns.tolist())
    classification_col =[]
    for i in range(level_size):
        classification_col.append('Level '+str(i))
    df_txt[classification_col]=df_txt['Classification'].str.split('-', expand=True)
    for i in range(1,len(classification_col)):
        df_txt[classification_col[i]]=np.where(df_txt[classification_col[i]].isnull(),df_txt[classification_col[i-1]],df_txt[classification_col[i]])
    
    
   
    
    if  ((spend=='citric')|(spend=='carbonates')|(spend=='hydrocolloids')|(spend=='lecithin')|(spend=='PGPR')|(spend=='salt')|(spend=='malic')|(spend=='gelatin_pork')|(spend=='gelatin_beef')|(spend=='gelatin_extra')|(spend=='plant_based')|(spend=='plant_foaming')|(spend=='CARRAGEENAN')|(spend =='PECTIN')|(spend =='H&G OTHERS')|(spend =='GUM ARABIC')|(spend =='XHANTAN GUM')):
#        
        df_txt=df_txt.drop(['Classification'],axis=1)
# =============================================================================
        df_txt['Level 2'] = df_txt['Level 2'].str.replace(' ', '')
        df_txt['Level 3'] = df_txt['Level 3'].str.replace(' ', '')
        #Variance check - Dropping 0% variance     
        txtcol=pd.DataFrame(df_txt[['Level 0', 'Level 1', 'Level 2','Level 3']].nunique(axis=0))

        #Variance check - Dropping 0% variance     
        #txtcol=pd.DataFrame(df_txt[['Level 0']].nunique(axis=0))
    elif ((spend=='dairy')|(spend=='fno')|(spend=='sugar')|(spend=='polyols')|(spend=='starch')|(spend=='honey')|(spend=='colors')|(spend=='cocoa_liq')):
        
        df_txt=df_txt.drop(['Classification'],axis=1)
        df_txt['Level 2'] = df_txt['Level 2'].str.replace(' ', '')
        df_txt['Level 3'] = df_txt['Level 3'].str.replace(' ', '')
        df_txt['Level 4'] = df_txt['Level 4'].str.replace(' ', '')
        #Variance check - Dropping 0% variance     
        txtcol=pd.DataFrame(df_txt[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4']].nunique(axis=0))
    elif ((spend=='Hazelnuts_Chopped_Roasted_Nuts')|(spend == 'Almonds_Chopped_Roasted_Nuts')|(spend=='Almond_Whole_Roasted_Nuts')|(spend=='his')|(spend=='syrups')|(spend=='cocoa_powder')|(spend=='cocoa_powder_alk')|(spend=='chocolates')):

            
        df_txt=df_txt.drop(['Classification'],axis=1)
        for i in range(3,len(classification_col)):
            df_txt[classification_col[i]] = df_txt[classification_col[i]].str.replace(' ', '')
            
        txtcol = pd.DataFrame(df_txt[classification_col].nunique(axis=0)) 
         
    return df_txt,txtcol

#To split classification levels for each spend areas
def leveltwo(spend,data):
    level_size = len(data['Classification'].str.split('-', expand=True).columns.tolist())
    classification_col =[]
    for i in range(level_size):
        classification_col.append('Level '+str(i))
    data[classification_col]=data['Classification'].str.split('-', expand=True)
    for i in range(1,len(classification_col)):
        data[classification_col[i]]=np.where(data[classification_col[i]].isnull(),data[classification_col[i-1]],data[classification_col[i]])
    
    return data










# =============================================================================
# def levelone_old(spend,df_txt):
#     if  ((spend=='citric')|(spend=='carbonates')|(spend=='hydrocolloids')|(spend=='lecithin')|(spend=='PGPR')|(spend=='salt')|(spend=='malic')|(spend=='gelatin_pork')|(spend=='gelatin_beef')|(spend=='gelatin_extra')|(spend=='plant_based')):
#         df_txt[['Level 0', 'Level 1', 'Level 2','Level 3']] = df_txt['Classification'].str.split('-', expand=True)
#         df_txt['Level 1']=np.where(df_txt['Level 1'].isnull(),df_txt['Level 0'],df_txt['Level 1'])
#         df_txt['Level 2']=np.where(df_txt['Level 2'].isnull(),df_txt['Level 1'],df_txt['Level 2'])
#         df_txt['Level 3']=np.where(df_txt['Level 3'].isnull(),df_txt['Level 2'],df_txt['Level 3'])
#         df_txt=df_txt.drop(['Classification'],axis=1)
#         df_txt['Level 2'] = df_txt['Level 2'].str.replace(' ', '')
#         df_txt['Level 3'] = df_txt['Level 3'].str.replace(' ', '')
#         #Variance check - Dropping 0% variance     
#         txtcol=pd.DataFrame(df_txt[['Level 0', 'Level 1', 'Level 2','Level 3']].nunique(axis=0))
#     #elif ((spend=='lecithin')):
#         #df_txt[['Level 0']] = df_txt['Classification'].str.split('-', expand=True)
#         #df_txt['Level 0']=np.where(df_txt['Level 0'].isnull(),df_txt['Level 0'],df_txt['Level 0'])
#         #df_txt=df_txt.drop(['Classification'],axis=1)
#         #df_txt['Level 0'] = df_txt['Level 0'].str.replace(' ', '')
#         #Variance check - Dropping 0% variance     
#         #txtcol=pd.DataFrame(df_txt[['Level 0']].nunique(axis=0))
#     elif ((spend=='dairy')|(spend=='fno')|(spend=='sugar')|(spend=='polyols')|(spend=='starch')|(spend=='honey')|(spend=='colors')|(spend=='cocoa_liq')):
#         df_txt[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4']] = df_txt['Classification'].str.split('-', expand=True)
#         df_txt['Level 1']=np.where(df_txt['Level 1'].isnull(),df_txt['Level 0'],df_txt['Level 1'])
#         df_txt['Level 2']=np.where(df_txt['Level 2'].isnull(),df_txt['Level 1'],df_txt['Level 2'])
#         df_txt['Level 3']=np.where(df_txt['Level 3'].isnull(),df_txt['Level 2'],df_txt['Level 3'])
#         df_txt['Level 4']=np.where(df_txt['Level 4'].isnull(),df_txt['Level 3'],df_txt['Level 4'])
#         df_txt=df_txt.drop(['Classification'],axis=1)
#         df_txt['Level 2'] = df_txt['Level 2'].str.replace(' ', '')
#         df_txt['Level 3'] = df_txt['Level 3'].str.replace(' ', '')
#         df_txt['Level 4'] = df_txt['Level 4'].str.replace(' ', '')
#         #Variance check - Dropping 0% variance     
#         txtcol=pd.DataFrame(df_txt[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4']].nunique(axis=0))
#     elif ((spend=='nuts')|(spend=='his')|(spend=='syrups')|(spend=='cocoa_powder')|(spend=='cocoa_powder_alk')|(spend=='chocolates')):
#         df_txt[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4','Level 5']] = df_txt['Classification'].str.split('-', expand=True)
#         df_txt['Level 1']=np.where(df_txt['Level 1'].isnull(),df_txt['Level 0'],df_txt['Level 1'])
#         df_txt['Level 2']=np.where(df_txt['Level 2'].isnull(),df_txt['Level 1'],df_txt['Level 2'])
#         df_txt['Level 3']=np.where(df_txt['Level 3'].isnull(),df_txt['Level 2'],df_txt['Level 3'])
#         df_txt['Level 4']=np.where(df_txt['Level 4'].isnull(),df_txt['Level 3'],df_txt['Level 4'])
#         df_txt['Level 5']=np.where(df_txt['Level 5'].isnull(),df_txt['Level 4'],df_txt['Level 5'])
#         df_txt=df_txt.drop(['Classification'],axis=1)
#         df_txt['Level 3'] = df_txt['Level 3'].str.replace(' ', '')
#         df_txt['Level 4'] = df_txt['Level 4'].str.replace(' ', '')
#         df_txt['Level 5'] = df_txt['Level 5'].str.replace(' ', '')
#         #Variance check - Dropping 0% variance     
#         txtcol=pd.DataFrame(df_txt[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4','Level 5']].nunique(axis=0))
#     return df_txt,txtcol
#
# #To split classification levels for each spend areas
# def leveltwo_old(spend,data):
#     if ((spend=='citric')|(spend=='carbonates')|(spend=='hydrocolloids')|(spend=='lecithin')|(spend=='PGPR')|(spend=='salt')|(spend=='malic')|(spend=='gelatin_pork')|(spend=='gelatin_beef')|(spend=='gelatin_extra')|(spend=='plant_based')):
#         data[['Level 0', 'Level 1', 'Level 2','Level 3']] = data['Classification'].str.split('-', expand=True)
#         data['Level 1']=np.where(data['Level 1'].isnull(),data['Level 0'],data['Level 1'])
#         data['Level 2']=np.where(data['Level 2'].isnull(),data['Level 1'],data['Level 2'])
#         data['Level 3']=np.where(data['Level 3'].isnull(),data['Level 2'],data['Level 3'])
#     #elif ((spend=='lecithin')):
#         #df_txt[['Level 0']] = df_txt['Classification'].str.split('-', expand=True)
#         #df_txt['Level 0']=np.where(df_txt['Level 0'].isnull(),df_txt['Level 0'],df_txt['Level 0'])
#     elif ((spend=='dairy')|(spend=='fno')|(spend=='sugar')|(spend=='polyols')|(spend=='starch')|(spend=='honey')|(spend=='colors')|(spend=='cocoa_liq')):
#         data[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4']] = data['Classification'].str.split('-', expand=True)
#         data['Level 1']=np.where(data['Level 1'].isnull(),data['Level 0'],data['Level 1'])
#         data['Level 2']=np.where(data['Level 2'].isnull(),data['Level 1'],data['Level 2'])
#         data['Level 3']=np.where(data['Level 3'].isnull(),data['Level 2'],data['Level 3'])
#         data['Level 4']=np.where(data['Level 4'].isnull(),data['Level 3'],data['Level 4'])
#     elif ((spend=='nuts')|(spend=='his')|(spend=='syrups')|(spend=='cocoa_powder')|(spend=='cocoa_powder_alk')|(spend=='chocolates')):
#         data[['Level 0', 'Level 1', 'Level 2','Level 3','Level 4','Level 5']] = data['Classification'].str.split('-', expand=True)
#         data['Level 1']=np.where(data['Level 1'].isnull(),data['Level 0'],data['Level 1'])
#         data['Level 2']=np.where(data['Level 2'].isnull(),data['Level 1'],data['Level 2'])
#         data['Level 3']=np.where(data['Level 3'].isnull(),data['Level 2'],data['Level 3'])
#         data['Level 4']=np.where(data['Level 4'].isnull(),data['Level 3'],data['Level 4'])
#         data['Level 5']=np.where(data['Level 5'].isnull(),data['Level 4'],data['Level 5'])
#     return data
#
# =============================================================================

# # The End


