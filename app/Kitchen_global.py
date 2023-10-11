# # PROJECT KITCHENdf_reqd

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023

# # CONTENTS

# This Kitchen_global.py file has all the function definitions and function calls
# The contents of the file are mentioned below:

# # Importing libraries and packages
#   This has all the libraries used in the code required to run the model & data pre-processing

import os
from config import *
from Libraries import *
from Global_DataPrep_Utils import *
from Utils import *
from Global_Model_Utils import *
from Global_Model_Output_Utils import *

# # Input data
# INPUT DATA

# df = pd.read_excel((os.getcwd()+'\Fats&Oils PLM extract for R&D.xlsx'),skiprows=[0,1,2],sheet_name='Sheet1')
# # Function call - Data pre-processing

# Data pre-processing 
# l1 = ['num_text','num_cat','num_cat_text'] #doubt why?

#Data pre-processing functions
#on call for num+text columns



def Model_Global():
    df_meta = pd.read_excel('selection_metadata.xlsx',sheet_name="selection_metadata")
    spend = df_meta['Selected'][0]
    # df, drop_var, use_var, drop_list, manual = config_model()
    # df, drop_var, use_var, drop_list = config_model()   #added 27April23
    # print(df)
    var_set = 'num_text'
    # dfmeu_nt,dfmeu_k_nt,df_reqdmeu_nt,cat_colsmeu_nt=preprocessing(df=df,var_set=var_set)
    dfmeu_nt,dfmeu_k_nt,df_reqdmeu_nt,cat_colsmeu_nt=preprocessing(var_set=var_set)           ##Aishwarya 26-05023

    #print(df_reqdmeu_nt)

    #Data pre-processing function call for num+cat+text columns
    var_set = 'num_cat_text'
    # dfmeu_nct,dfmeu_k_nct,df_reqdmeu_nct,cat_colsmeu_nct=preprocessing(df=df,var_set=var_set)
    dfmeu_nct,dfmeu_k_nct,df_reqdmeu_nct,cat_colsmeu_nct=preprocessing(var_set=var_set)       ##Aishwarya 26-05023
    #print(dfmeu_k_nct)
    type(dfmeu_k_nct)
    #print(dfmeu_k_nct.columns)

    #Data pre-processing function call for num+cat columns
    var_set = 'num_cat'
    # dfmeu_nc,dfmeu_k_nc,df_reqdmeu_nc,cat_colsmeu_nc=preprocessing(df=df,var_set=var_set)
    dfmeu_nc,dfmeu_k_nc,df_reqdmeu_nc,cat_colsmeu_nc=preprocessing(var_set=var_set)          ##Aishwarya 26-05023

    #print(dfmeu_k_nct)

    # Function call - K-Medoids

    #kmed num+cat+text version
    kmedmeuclus_nct=K_med_cluster(dfmeu_k_nct,dfmeu_k_nct.shape[0]-1) 

    kmedmeu_nct,feature_importance_med_meu_nct=modelkmedoid(dfmeu_nct,dfmeu_k_nct,kmedmeuclus_nct)


    kmed_fimp_nct=feature_importance_med_meu_nct.copy()

    #kmed num+text version
    kmedmeuclus_nt=K_med_cluster(dfmeu_k_nt,dfmeu_k_nt.shape[0]-1) 

    kmedmeu_nt,feature_importance_med_meu_nt=modelkmedoid(dfmeu_nt,dfmeu_k_nt,kmedmeuclus_nt)

    kmed_fimp_nt=feature_importance_med_meu_nt.copy()

    #kmed num+cat version
    kmedmeuclus_nc=K_med_cluster(dfmeu_k_nc,dfmeu_k_nc.shape[0]-1) 

    kmedmeu_nc,feature_importance_med_meu_nc=modelkmedoid(dfmeu_nc,dfmeu_k_nc,kmedmeuclus_nc)

    kmed_fimp_nc=feature_importance_med_meu_nc.copy()

    #print((dfmeu_nt))

    #pca num+text version
    df_pcameu_nt=pca(dfmeu_nt)

    #print(df_pcameu_nt)

    #pca num+cat+text version
    df_pcameu_nct=pca(dfmeu_nct)

    #pca num+cat version
    df_pcameu_nc=pca(dfmeu_nc)


    # PCA num+text version 

    #To get optimal cluster value across regions
    #MEU
    y = cluster_stop(df_pcameu_nt)
    x = cluster_start(df_pcameu_nt)
    meu1=sil(df_pcameu_nt,x,y,sil_affinity,sil_linkage)

    #print(x)

    #Function call - Hierarchical Clustering using PCA num+text components
    clus_pca_meu_nt=model_pca(df_pcameu_nt,meu1['n_clusters'].sum(),pca_affinity,pca_linkage)



    # PCA num+cat version 

    #To get optimal cluster value across regions
    #MEU
    y = cluster_stop(df_pcameu_nc)
    x = cluster_start(df_pcameu_nc)
    meu1=sil(df_pcameu_nc,x,y,sil_affinity,sil_linkage)

    # AMEA

    #Function call - Hierarchical Clustering using PCA num+cat components
    clus_pca_meu_nc=model_pca(df_pcameu_nc,meu1['n_clusters'].sum(),pca_affinity,pca_linkage)


    # PCA num+cat+text version 

    #To get optimal cluster value across regions
    #MEU
    y = cluster_stop(df_pcameu_nct)
    x = cluster_start(df_pcameu_nct)
    meu1=sil(df_pcameu_nct,x,y,sil_affinity,sil_linkage)


    #Function call - Hierarchical Clustering using PCA num+cat+text components
    clus_pca_meu_nct=model_pca(df_pcameu_nct,meu1['n_clusters'].sum(),pca_affinity,pca_linkage)






    #FAMD num+cat+text version
    df_famd_meu_nct=famd(dfmeu_k_nct)


    #print(df_famd_meu_nct)

    #FAMD num+cat version
    df_famd_meu_nc=famd(dfmeu_k_nc)

    # FAMD num+cat version 

    #To get optimal cluster value across regions
    #MEU
    y = cluster_stop(df_famd_meu_nc)
    x = cluster_start(df_famd_meu_nc)
    meu1=sil(df_famd_meu_nc,x,y,sil_affinity,sil_linkage)
    #AMEA


    #Function call - Hierarchical Clustering using FAMD num+cat components
    clus_famd_meu_nc=model_pca(df_famd_meu_nc,meu1['n_clusters'].sum(),pca_affinity,pca_linkage)

    # FAMD num+cat+text version 

    #To get optimal cluster value across regions
    #MEU
    y = cluster_stop(df_famd_meu_nct)
    x = cluster_start(df_famd_meu_nct)
    meu1=sil(df_famd_meu_nct,x,y,sil_affinity,sil_linkage)
    #AMEA

    #Function call - Hierarchical Clustering using FAMD num+cat+text components
    clus_famd_meu_nct=model_pca(df_famd_meu_nct,meu1['n_clusters'].sum(),pca_affinity,pca_linkage)



    # # Choosing Optimal Number of Clusters
    # Hierarchical Clustering
    # Silhoutte score value for choosing number of clusters for hierarchical clustering for all regions and distances

    #Hierarchical clustering for num+cat version 
    #MEU
    y = cluster_stop(dfmeu_nc)
    x = cluster_start(dfmeu_nc)
    meu1=sil(dfmeu_nc,x,y,sil_affinity,sil_linkage)

##-----------------------------------------------------------------MODEL NOT WORKING
    # # Function call - Model
    # Hierarchical clustering output
    # MODEL FOR ALL REGIONS AND MODEL METRICS

    #Hierarchical clustering for num+cat version 
    #MEU
    modelmeuward1_nc,modelmeuward2_nc,feature_importance_allmeuward_nc=model(dfmeu_nc,df_reqdmeu_nc,model_methodd,model_linkage,model_affinity,meu1['n_clusters'].sum())
   
    #Hierarchical clustering for num+text version 
    #MEU
    y = cluster_stop(dfmeu_nt)
    x = cluster_start(dfmeu_nt)
    meu1=sil(dfmeu_nt,x,y,sil_affinity,sil_linkage)

    #MODEL FOR ALL REGIONS AND MODEL METRICS
    #Hierarchical clustering for num+text version 
    #MEU
    modelmeuward1_nt,modelmeuward2_nt,feature_importance_allmeuward_nt=model(dfmeu_nt,df_reqdmeu_nt,model_methodd,model_linkage,model_affinity,meu1['n_clusters'].sum())

    #Hierarchical clustering for num+cat+text version 
    #MEU
    y = cluster_stop(dfmeu_nct)
    x = cluster_start(dfmeu_nct)
    meu1=sil(dfmeu_nt,x,y,sil_affinity,sil_linkage)


    #Hierarchical clustering for num+cat+text version 
    #MEU
    modelmeuward1_nct,modelmeuward2_nct,feature_importance_allmeuward_nct=model(dfmeu_nct,df_reqdmeu_nct,model_methodd,model_linkage,model_affinity,meu1['n_clusters'].sum())

    #print(feature_importance_allmeuward_nct)

    #For num+cat version
    model1_nc ,featureimp_nc= col_add(modelmeuward1_nc,modelmeuward2_nc,feature_importance_allmeuward_nc,
                kmedmeu_nc, clus_pca_meu_nc, clus_famd_meu_nc)

    #For num+cat+text version
    model1_nct ,featureimp_nct= col_add(modelmeuward1_nct,modelmeuward2_nct,feature_importance_allmeuward_nct,
                kmedmeu_nct, clus_pca_meu_nct,clus_famd_meu_nct)

    #For num+text version
    model1_nt ,featureimp_nt= col_add(modelmeuward1_nt,modelmeuward2_nt,feature_importance_allmeuward_nt,
                kmedmeu_nt, clus_pca_meu_nt,clus_famd_meu_nct)
    #model1_nt=model1_nt.drop(['famd_cluster'],axis=1)
    model1_nt=model1_nt.drop(['Model 4'],axis=1)#change



    feat_imp_nt = feature_imp_output(kmed_fimp_nt,featureimp_nt) #change
    feat_imp_nct = feature_imp_output(kmed_fimp_nct,featureimp_nct)    #change
    feat_imp_nc = feature_imp_output(kmed_fimp_nc,featureimp_nc) #change



    # # Output dataframes
    # This output will go into PBI report
    # Place to change while running the code
    # spend='his'

    #separate versions
    model1_nt.to_csv(spend+'_global_nt_results.csv',index = False)
    model1_nct.to_csv(spend+'_global_nct_results.csv',index = False)
    model1_nc.to_csv(spend+'_global_nc_results.csv',index = False)

    model1_nt

    feat_imp_nt_df= pd.DataFrame(feat_imp_nt)
    feat_imp_nct_df= pd.DataFrame(feat_imp_nct)
    feat_imp_nc_df= pd.DataFrame(feat_imp_nc)


    #combining outputs of nc,nct,nt in one sheet
    final=combined_output(model1_nc,model1_nt,model1_nct)

    #combined version
    final.to_csv(spend+'_final_global_results.csv')


    #feature importance
    feat_imp_nt_df.to_csv(spend+'_global_feature_imp_nt.csv',index = False)
    feat_imp_nct_df.to_csv(spend+'_global_feature_imp_nct.csv',index = False)
    feat_imp_nc_df.to_csv(spend+'_global_feature_imp_nc.csv',index = False)

    # # The End

# Model_Global() 