# # PROJECT KITCHEN

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023

# # CONTENTS
#   This file model_output_utils.py has the functions definition for merging the model outputs together.
#   These functions will be called after calling the functions from model_utils.py
#   The function call of this is present in kitchen_global.py file.





# # Addition of Region and Type column for PBI
#NEW COLUMNS - TYPE, REGION
from Libraries import *

from config import *


df_meta = pd.read_excel('selection_metadata.xlsx',sheet_name="selection_metadata")
spend = df_meta['Selected'][0]


# This gives the output in desired format with all models' results used for a particular varset along with the similarity score
def col_add(modelmeuward1,modelmeuward2,feature_importance_allmeuward,kmedmeu
            ,clus_pca_meu,clus_famd_meu):
    #MEU
    modelmeuward1['Type']= model_type
    modelmeuward2['Type']= model_type
   
    feature_importance_allmeuward['Type']= model_type
 
  
    #Model 1 df with specid*specid format
    Frames1=[modelmeuward1]
    model1=pd.concat(Frames1)
    model1 = model1.reset_index()
    kmedmeu=kmedmeu.reset_index()  
   
    kmed =[kmedmeu]
    kmed = pd.concat(kmed)
    kmed.columns = ['Specification Number','KMedoids_cluster']
    model1= pd.merge(model1,kmed,on=['Specification Number'],how='left')

    #Model 2 df with unique specid format
    Frames2=[modelmeuward2]
    model2=pd.concat(Frames2)
    model2 = model2.reset_index()
    model2= pd.merge(model2,kmed,on=['Specification Number'],how='left')


    shap=[feature_importance_allmeuward]
    featureimp=pd.concat(shap)
    model1['id']=model1['Specification Number'].astype(str)+model1['variable'].astype(str)+model1['Type'].astype(str)
    spn = [spend_meu,spend_amea,spend_na,spend_la]
    vol = [volume_meu,volume_amea,volume_na,volume_la]
    #Cleaning on few columns
    for x in spn:
        model1[x]=model1[x].fillna(0)
        model1[x].replace(to_replace=[None], value=0, inplace=True)
        model1[x]=model1[x].astype(float)
        model2[x]=model2[x].fillna(0)
        model2[x].replace(to_replace=[None], value=0, inplace=True)
        model2[x]=model2[x].astype(float)
    model1['Spend']=model1[spend_meu]+model1[spend_amea]+model1[spend_na]+model1[spend_la]
    model2['Spend']=model2[spend_meu]+model2[spend_amea]+model2[spend_na]+model2[spend_la]   
    for x in vol:
        model1[x]=model1[x].fillna(0)
        model1[x].replace(to_replace=[None], value=0, inplace=True)
        model1[x]=model1[x].astype(float)
        model2[x]=model2[x].fillna(0)
        model2[x].replace(to_replace=[None], value=0, inplace=True)
        model2[x]=model2[x].astype(float)
    model1['Spend']=model1[spend_meu]+model1[spend_amea]+model1[spend_na]+model1[spend_la]
    model2['Spend']=model2[spend_meu]+model2[spend_amea]+model2[spend_na]+model2[spend_la]   
    model1['Volume']=model1[volume_meu]+model1[volume_amea]+model1[volume_na]+model1[volume_la]
    model2['Volume']=model2[volume_meu]+model2[volume_amea]+model2[volume_na]+model2[volume_la]
    model1['cluster']=model1['cluster'].astype(str)
    model1['Specification Number']=model1['Specification Number'].astype(str)
    model1['variable']=model1['variable'].astype(str)
    model1['KMedoids_cluster']=model1['KMedoids_cluster'].astype(str)
    model2['cluster']=model2['cluster'].astype(str)
    model2['Specification Number']=model2['Specification Number'].astype(str)
    model2['variable']=model2['variable'].astype(str)
    model2['KMedoids_cluster']=model2['KMedoids_cluster'].astype(str)
    model2 = model2.groupby(['cluster']).agg({'Spend':'sum', 'Volume':'sum'})
    model2.reset_index(inplace=True)
    #to take cluster wise spend and volume to model1 data from model2
    model2.rename(columns = {'Spend':'Spend_Clusterwise','Volume':'Volume_Clusterwise'}, inplace = True)
    model1['clusid']=model1['cluster']
    model2['clusid']=model2['cluster']
    model1=pd.merge(model1,model2,on=['clusid'],how='left')
    model1=model1.drop(['cluster_y'],axis=1)
    model1.rename(columns = {'cluster_x':'cluster'}, inplace = True)

  

    frames=[clus_pca_meu]
    pca=pd.concat(frames)
    pca.reset_index(inplace=True)

    frames=[clus_famd_meu]
    famd=pd.concat(frames)
    famd.reset_index(inplace=True)

    pca['clusid']=pca['Specification Number'].astype(str)
    famd['clusid']=famd['Specification Number'].astype(str)

    pca=pca[['clusid','pca_cluster']]
    famd.rename(columns={'pca_cluster':'famd_cluster'},inplace=True)
    famd=famd[['clusid','famd_cluster']]

    model1['clusid']=model1['Specification Number'].astype(str)
    model1=pd.merge(model1,pca,on=['clusid'],how='left')
    model1=pd.merge(model1,famd,on=['clusid'],how='left')
    model1['Region']='Global'
    if(spend=='PECTIN')|(spend =='CARRAGEENAN')|(spend =='H&G OTHERS')|(spend =='GUM ARABIC')|(spend =='XHANTAN GUM'):
        model1['Region']='AMEA'
    model1.rename(columns={'cluster':'Model 1','KMedoids_cluster':'Model 2','pca_cluster':'Model 3','famd_cluster':'Model 4'},inplace=True)
    levels = [col for col in model1 if col.startswith('Level')]
    reqd = ["Specification Number","Specification description","Region","Type","Volume","Spend","Model 1","Model 2",
        "Model 3","Model 4","variable","value","Cluster name"]+levels
    model1 = model1[reqd]

    return model1,featureimp

#Feature importance of the attributes used in Herirachical clustering and k_medoids model
def feature_imp_output(kmed,Hc):#change
    #change
    kmed['Region']='Global'
    if(spend=='PECTIN')|(spend =='CARRAGEENAN')|(spend =='H&G OTHERS')|(spend =='GUM ARABIC')|(spend =='XHANTAN GUM'):
        kmed['Region']='AMEA'
    kmed = kmed[['feature_name','KMedoids_cluster',	'feature_imp_scaled_vals','Region']]
    kmed.rename(columns={'KMedoids_cluster':'Model 2'},inplace=True)
    kmed['Model']='Model 2'
    Hc['Region']='Global'
    if(spend=='PECTIN')|(spend =='CARRAGEENAN')|(spend =='H&G OTHERS')|(spend =='GUM ARABIC')|(spend =='XHANTAN GUM'):
        Hc['Region']='AMEA'
    Hc = Hc[['feature_name','cluster',	'feature_imp_scaled_vals','Region']]
    Hc.rename(columns={'cluster':'Model 1'},inplace=True)
    Hc['Model']='Model 1'
    feat_imp=[(Hc,kmed)]
    return feat_imp

#combining outputs of nc,nct,nt in one sheet
def combined_output(model1_nc,model1_nt,model1_nct):
    #renaming columns 

    model1_nc=model1_nc.rename(columns={'Model 1':'nc_Model 1','Model 2':'nc_Model 2','Model 3':'nc_Model 3','Model 4':'nc_Model 4'})
    model1_nt=model1_nt.rename(columns={'Model 1':'nt_Model 1','Model 2':'nt_Model 2','Model 3':'nt_Model 3'})
    model1_nct=model1_nct.rename(columns={'Model 1':'nct_Model 1','Model 2':'nct_Model 2','Model 3':'nct_Model 3','Model 4':'nct_Model 4'})

    #dropping variable,value as per requirement

    model1_nc.drop(['variable','value'],axis=1,inplace=True)
    model1_nt.drop(['variable','value'],axis=1,inplace=True)
    model1_nct.drop(['variable','value'],axis=1,inplace=True)

    #reordering columns
    gen_col=['Specification Number','Specification description','Region','Type','Volume','Spend','Cluster name']
    level_col=[col for col in model1_nc if col.startswith('Level')]
    model_col=['nc_Model 1','nc_Model 2','nc_Model 3','nc_Model 4']

    all_col=gen_col + level_col + model_col
    model1_nc=model1_nc[all_col]

    # =============================================================================
    # model1_nc=model1_nc[['Specification Number',
    #  'Specification description',
    #  'Region',
    #  'Type',
    #  'Volume',
    #  'Spend',
    #  'Cluster name',
    #  'Level 0',
    #  'Level 1',
    #  'Level 2',
    #  'Level 3',
    #  'Level 4',
    #  'model1_nc_Model 1',
    #  'model1_nc_Model 2',
    #  'model1_nc_Model 3',
    #  'model1_nc_Model 4']]
    # =============================================================================
     

    #dropping duplicates

    model1_nc=model1_nc.drop_duplicates().reset_index(drop=True)
    model1_nt=model1_nt.drop_duplicates().reset_index(drop=True)
    model1_nct=model1_nct.drop_duplicates().reset_index(drop=True)

    #Extracting columns to get required output

    model1_nt=model1_nt[['Specification Number','nt_Model 1','nt_Model 2','nt_Model 3']]
    model1_nct=model1_nct[['Specification Number','nct_Model 1','nct_Model 2','nct_Model 3','nct_Model 4']]


    model1=pd.merge(model1_nc,model1_nt,on='Specification Number',how='inner')

    final=pd.merge(model1,model1_nct,on='Specification Number',how='inner')
    
    return final
