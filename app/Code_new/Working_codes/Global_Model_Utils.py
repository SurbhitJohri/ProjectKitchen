# # PROJECT KITCHEN

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023

# # CONTENTS
#   This file model_utils.py has the model function definitions for all the model which includes
#   k-medoids, hierarchical, pca, famd and also the optimal cluster - dendrogram, silhoutte, elbow 
#   It has the similarity scoring calculation and the crude logic used to define the suggested spec within each cluster 
#   It has the feature importance cluster explanability shap calculation for each cluster
#   The function call of this is present in kitchen_global.py file.
#
# # ALL REGIONS
#
# # 1. INPUT
# # 2. PRE-PROCESSING 
# #     2.1 NUMERIC
# #     2.2 CATEGORICAL
# #     2.3 TEXT
# # 3. K-MEDOIDS CLUSTERING WITH GOWER DISTANCE
# # 4. K-PROTOTYPE CLUSTERING (Not Using in the Output)
# # 5. DENDROGRAM
# # 6. OPTIMAL CLUSTERS
# # 6. CLUSTERING - WARD, SINGLE, COMPLETE, AVERAGE LINKAGES WITH ALL DISTANCE METRICS
# # 8. SIMILARITY SCORE
# # 9. HIGHEST SIMILARITY
# # 10. SIMILARITY CRUDE LOGIC
# # 11. MODEL INTERPRETABILITY
# # 12. OUTPUT 


# # K-MEDOIDS CLUSTERING

# # Function - K-Medoids
#
# Choosing Optimal clusters based on elbow method
#
# PAM k-medoids clustering with pre-computed Gower Distance Matrix
#
# Model explanability by calculating shap values on the clusters obtained


#MODEL 1
#GOWER + K-MEDOIDS
import os
from Libraries import * 
from Utils import *
from pandas import concat
from config import *


df_meta = pd.read_excel(r"C:\Users\ERY3041\Downloads\Code_new_21062023 (1)\Code_new\selection_metadata.xlsx",sheet_name="selection_metadata")
spend = df_meta['Selected'][0]


#It takes preprocessed data as input and return Optimal clusters for kmedoid
def K_med_cluster(df_region,y):
    dist_matrix = gower.gower_matrix(df_region) #Using precomputed gower distance matrix for clustering
    dict_cost_c3 = {}
    for k in range(2,y):
        clusters = (KMedoids(n_clusters=k, metric=K_cluster_metric,method=K_cluster_method, init=K_cluster_init, max_iter=K_cluster_iter, random_state=42))
        res = clusters.fit_predict(dist_matrix)
        dict_cost_c3[k] = clusters.inertia_
    print("cost (sum distance) for all k:")
    _ = [print(k,":",f'{v:.3f}') for k,v in dict_cost_c3.items()]
    # scree plot: look for elbow
    plt.figure(figsize=[8,5])
    plt.plot(dict_cost_c3.keys(), dict_cost_c3.values(), color="blue")
    plt.title("cost (sum distance) vs. number of clusters")
    plt.xticks(np.arange(2,10,1))
    plt.xlabel("number of clusters K")
    plt.ylabel("cost")
    #Elbow method to determine the optimal number of clusters
    cost_knee_c3 = KneeLocator( 
            x=list(dict_cost_c3.keys()), 
            y=list(dict_cost_c3.values()), 
            S=0.1, curve="convex", direction="decreasing", online=True)
    K_cost_c3 = cost_knee_c3.elbow   
    #print(Region)
    #print(K_cost_c3)
    #print(": elbow at k =", f'{K_cost_c3:.0f} clusters')
    return K_cost_c3

    

#It uses the optimal no. of clusters from the kmed cluster(above) function and form the clusters using Kmedoids and also returns the feature importance usinf SHAP values
def modelkmedoid(data_f,data,n_clusters): #change
    if ((data.shape[0]>=min_specid_req)&(n_clusters is not None)): # min no of spec ids to form clusters and cluster is not empty
        dist_matrix = gower.gower_matrix(data) #Using precomputed gower distance matrix for clustering
        clusters = (KMedoids(n_clusters=n_clusters, metric=K_med_metric,method=K_med_method, init=K_med_init, max_iter=K_med_max_iter, random_state=42))
        clusters.fit_predict(dist_matrix)
        data_labels = clusters.labels_
        cluster=pd.DataFrame(data_labels) #data labels as spec ids
        cluster.index=data.index
        cluster.columns=['KMedoids_cluster']
        frames = [data, cluster]
        out=pd.merge(data_f, cluster, left_index=True, right_index=True)
        y = out[['KMedoids_cluster']] #Taking only the cluster column
        X = out.drop('KMedoids_cluster',axis=1)
        #Model Explanability
        clf = RandomForestClassifier()
        print(X.shape)
        clf.fit(X,y)
        print(X.shape)
        
        shap_values = shap.TreeExplainer(clf,feature_perturbation='interventional').shap_values(X,check_additivity=False)
        feature_importance_all = pd.DataFrame()
        for i in range(0,out.KMedoids_cluster.nunique()):
            #shap.summary_plot(shap_values[i], X)
            feature_importance = pd.DataFrame(list(zip(X.columns, np.abs(shap_values[i]).mean(0))), columns=['feature_name', 'feature_importance_vals'])
            feature_importance = feature_importance.iloc[(-np.abs(feature_importance['feature_importance_vals'].values)).argsort()]
            feature_importance['KMedoids_cluster'] = i
            feature_importance_all=concat([feature_importance_all, feature_importance])
        f1 = feature_importance_all.groupby(['KMedoids_cluster']).agg({'feature_importance_vals':['min','max']}).reset_index() #scaling feature imp
        f1.columns= ['KMedoids_cluster','min_','max_'] # scaling feature imp
        feature_importance_all = pd.merge(feature_importance_all,f1,on=['KMedoids_cluster'],how='left') # scaling feature imp
        feature_importance_all['feature_imp_scaled_vals']=np.where(feature_importance_all.max_==feature_importance_all.min_,0,
                                                       (feature_importance_all.feature_importance_vals-feature_importance_all.min_)/(feature_importance_all.max_-feature_importance_all.min_))#scaling feature imp
        
        out=out[['KMedoids_cluster']]
     
    else: #change
       out = data_f
       out['KMedoids_cluster']=0
       out=out[['KMedoids_cluster']]
       feature_importance_all = pd.DataFrame()
       feature_importance_all['feature_name']=''
       feature_importance_all['KMedoids_cluster']=''
       feature_importance_all['feature_imp_scaled_vals']=''
       feature_importance_all['Region']='Global'
    return out , feature_importance_all



# ------------------------------------------------------------------------------------------------------------------

# Dimensionality Reduction - PCA and FAMD

#To get PCA components to explain max variance
def pca(data):
    pca_d = pd.DataFrame()
    comp = []
    variance = []
    if data.shape[0]<data.shape[1]:
        n=data.shape[0]
    else:
        n=data.shape[1]
    for i in range(2,n):
        pca = PCA(n_components=i)
        df_pca=pca.fit_transform(data)
        df_pca = pd.DataFrame(df_pca)
        df_pca.index=data.index
        p = pca.explained_variance_ratio_.sum()
        comp = comp+ [i]
        variance = variance + [p]
    pca_d['Components']= comp
    pca_d['Variance']=variance
    
    if pca_d['Variance'].sum()==pca_d.shape[0]: # if all the components are with variance with 1 only, then no. of components will be taken 2 only
        max_var = 2
        print(pca_d)
    else:
        pca_d=pca_d[pca_d.Variance<1] #to get components with max variance less than 100%
        max_var = pca_d[pca_d.Variance==pca_d.Variance.max()].reset_index()['Components'][0]
        
    pca_fin = PCA(n_components=max_var)
    df_pca_fin=pca_fin.fit_transform(data)
    df_pca_fin = pd.DataFrame(df_pca_fin)
    df_pca_fin.index=data.index
    return df_pca_fin

#Silhoutte score value for choosing number of clusters
def sil(data,a,b,affinity,linkage):
    if data.shape[0]>=min_specid_req and data.shape[1]>0:   # min no of spec ids to form clusters and there should be 1 column avaliable
        datashape=data.shape[0]
        l1=[]
        l2=[]
        dd=[]
        for n_clusters in range(a,b): #silhoutte for hierarchical clustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
            cluster_labels = clusterer.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            l1.append(n_clusters)
            l2.append(silhouette_avg)
            d1=pd.DataFrame(l1)
            d2=pd.DataFrame(l2)
        dd=pd.merge(d1, d2, left_index=True, right_index=True)
        dd.columns=['n_clusters','Silh score']  
        dd=dd[dd['n_clusters']<=datashape]
        dd=dd[(dd['Silh score']==dd['Silh score'].max())]
        if dd.shape[0]>1:
            dd=dd.head(1)
        dd=dd.drop_duplicates()
        print("Dataframe", dd,"Model Metric :", "a :",a,"b :",b)
    else: #change
        data = [[1,np.nan]]
        dd = pd.DataFrame(data,columns=['n_clusters','Silh_score'])
        dd=dd.drop_duplicates()
    return dd 

#To define cluster range - min
def cluster_start(df):
    if df.shape[0]<=(min_specid_req*2): #originally, it was 10
        x = begin_cluster
    else:
        x = df.shape[0]-round(df.shape[0]/2)
    return int(x)

#To define cluster range - max
def cluster_stop(df):
    if df.shape[0]<=(min_specid_req*2): #originally, it was 10
        x = df.shape[0]-begin_cluster
    else:
        x = df.shape[0]-end_cluster
    return int(x)





#Function - Hierarchical Clustering Model using PCA, FAMD components
def model_pca(data,n_clustersmodel,affinity,linkage):
    if data.shape[0]>=min_specid_req and data.shape[1]>0:   # min no of spec ids to form clusters and there should be 1 column avaliable
        cluster = AgglomerativeClustering(n_clusters=n_clustersmodel, affinity=affinity, linkage=linkage)  #HIERARCHICAL CLUSTERING
        cluster.fit_predict(data)
        data_labels = cluster.labels_
        cluster=pd.DataFrame(data_labels) #data labels as spec ids
        cluster.index=data.index
        cluster.columns=['pca_cluster'] 
        frames = [data, cluster]
        out=pd.merge(data, cluster, left_index=True, right_index=True) #merge data and cluster value
        out=out[['pca_cluster']]
    else : #change
        out = data.copy()    
        out['pca_cluster']=0
        out=out[['pca_cluster']]
    return out


# !pip show prince

# -------------------------------------------------------------------------------------------------------------

# !pip install prince==0.7.1

# FAMD 

#To get FAMD components to explain max variance
def famd(data):
    if data.shape[0]>=min_specid_req and len(data.select_dtypes(['int64','float64']).columns.tolist())>=1 and len(data.select_dtypes('object').columns.tolist())>=1: # min no of spec ids to form clusters and atleast one numerical and categorical column must present
        famd_d = pd.DataFrame()
        comp = []
        variance = []
        for i in range(2,data.shape[0]):
            famd = prince.FAMD(n_components=i,n_iter=famd_n_iter,copy=famd_copy, check_input=famd_check_input,engine='sklearn',random_state=42)
            df_famd=famd.fit_transform(data)
            df_famd = pd.DataFrame(df_famd)
            df_famd.index=data.index
            f = famd.explained_inertia_.sum()
            comp = comp+ [i]
            variance = variance + [f]
        famd_d['Components']= comp
        famd_d['Variance']=variance
        famd_d=famd_d[famd_d.Variance<1]
        max_var = famd_d[famd_d.Variance==famd_d.Variance.max()].reset_index()['Components'][0]
        famd_fin = prince.FAMD(n_components=max_var)
        df_famd_fin=famd.fit_transform(data)
        df_famd_fin = pd.DataFrame(df_famd_fin)
        df_famd_fin.index=data.index
    else:#change
        df_famd_fin=pd.DataFrame()
        df_famd_fin.index=data.index
    return df_famd_fin#change

# -------------------------------------------------------------------------------------------------------

# # HIERARCHICAL CLUSTERING

# # Dendrogram view for all regions

# # Function - Dendrogram
#
# To get the dendrogram visual for all regions and distances
#
# To visualize the optimal y value (Horizontal line)
#
# We are not using it currently because there is no requirement for dendrogram



# DENDROGRAM VIEW for all regions

#To get distance cutting line to get required number of clusters
#Not using currently as we are not giving the graph
def dend(data,methodd,labels,var,y):
    plt.figure(figsize=(50, 7))  
    plt.title("Dendrogram"+" "+var)  
    dend = shc.dendrogram(shc.linkage(data, method=methodd),labels=data.index)
    plt.axhline(y = y, color = 'r', linestyle = '-')
    link_method = linkage(data, method = methodd)
    clusters1 = fcluster(link_method, y,criterion='distance')
    c = pd.DataFrame({'cluster':clusters1})
    c.index = data.index
    #print(c.cluster.nunique())



# # Function - Model
#
# # Model -> To form Clusters
# # Similarity score within each Cluster
# # Highest Similarity based on Score
# # Highest Similarity based on Occurence
# # Model Interpretability - rf and shap values on the clusters formed

#HIERARCHICAL CLUSTERING MODEL + MODEL EXPLANABILITY + SIMILARITY SCORING
def model(data,df_reqd,methodd,linkage,affinity,n_clustersmodel):
# =============================================================================
#     data=dfmeu_nc
#     df_reqd=df_reqdmeu_nc
#     methodd=model_methodd
#     linkage=model_linkage
#     affinity=model_affinity
#     n_clustersmodel=meu1['n_clusters'].sum()
# =============================================================================
    cluster = AgglomerativeClustering(n_clusters=n_clustersmodel, affinity=affinity, linkage=linkage)  #HIERARCHICAL CLUSTERING
    cluster.fit_predict(data)
    data_labels = cluster.labels_
    cluster=pd.DataFrame(data_labels) #data labels as spec ids
    cluster.index=data.index
    cluster.columns=['cluster']
    frames = [data, cluster]
    out=pd.merge(data, cluster, left_index=True, right_index=True) #merge data and cluster value
    #------------------------------------------------------------------------------------------
    #SIMILARITY SCORE
    #df_sim = pd.DataFrame()#to get entire similarity
    #sim=pd.DataFrame(cosine_similarity(out.drop(['cluster'],axis=1)))
    #sim.index = out.index
    #sim.columns = out.index
    #df_sim = sim.copy()
    df_sim = pd.DataFrame()
    for x in list(out.cluster.unique()):
        out_x = out[out.cluster==x]
        sim=pd.DataFrame(cosine_similarity(out_x.drop(['cluster'],axis=1))) #cosine similarity
        sim.index = out_x.index
        sim.columns = out_x.index
        sim['cluster'] = x
        df_sim=concat([df_sim,sim])
    #COMBINE ORIGINAL DF+CLUSTER COLUMN+SIMILARITY COLUMN
    #To prepare df in specid*specid format
    model=pd.merge(df_reqd, cluster, left_index=True, right_index=True)
    pbi=pd.merge(model,df_sim.drop(['cluster'],axis=1), left_index=True, right_index=True)
    pbi1 = pbi.reset_index()
    no_integers = [x for x in list(pbi1.columns.unique()) if not isinstance(x, int)]
    
    # pbi1.to_excel('pbi1.xlsx')
    idvarnew = [x for x in pbi1.columns.unique() if x not in pbi1['Specification Number'].unique()]
    # print(idvarnew)
    # data=pd.melt(pbi1, id_vars =idvarnew, value_vars =pbi1['Specification Number'].unique())    #added by Aishwarya 26May23
    data=pd.melt(pbi1, id_vars =[x for x in list(pbi1.columns.unique()) if not isinstance(x, int)], value_vars =pbi1['Specification Number'].unique())
    #--------------------------------------------------------------------------------------------
    #create level columns again to merge with dataframe
    data=leveltwo(spend=spend,data=data)
    data=data.drop(['Classification'],axis=1)
    #To check number of specids in each cluster
    freqmodel=model.copy() 
    freqmodel=freqmodel.reset_index()
    freq_check = freqmodel.groupby(['cluster'])['Specification Number'].count().reset_index() 
    freq_check.columns = ['cluster','cluster_size']
    data1 = pd.merge(data,freq_check,on=['cluster'],how='left')
    #-------------------------------------------------------------------------------------------
    #HIGHEST SIMILARITY within each cluster
    data_fin= pd.DataFrame()
    for x in list(data1['Specification Number'].unique()):
        data_x = data1[data1['Specification Number']==x]
        data_x1 = data_x[data_x.variable!=x]
        data_x1['Suggested_main']=np.where(data_x1.value==data_x1.value.max(),'Yes','No')
        data_y = data1[(data1.variable==x)&(data1['Specification Number']==x)]
        data_y['Suggested_main']='No'
        data_z=concat([data_y, data_x1])
        data_fin= concat([data_fin, data_z])
    data_fin['Suggested_main']=np.where((data_fin.cluster_size==1)&(data_fin['Specification Number']==data_fin.variable),'Yes',data_fin.Suggested_main)   
    data_fin[data_fin['Suggested_main']=='Yes']['Specification Number'].nunique()
    c_alt = pd.DataFrame()
    for x in list(data1.cluster.unique()):
        c0 = data1[data1.cluster==x]
        c1 =c0.sort_values(by='value',ascending=False)
        c1['value']=100*c1.value
        c1['value']=c1['value'].round(decimals = 0)
        c2 = c1[c1['value']!=c1.value.max()]
        c2['suggested_id_main2'] = np.where(c2.value==c2.value.max(),'Yes','No')
        c3= c1[c1.value==c1.value.max()]
        c4 = concat([c2,c3])
        c_alt = concat([c_alt, c4])  
    c_alt['suggested_id_main2']=np.where((c_alt.cluster_size==1)&(c_alt.value==100),'Yes',c_alt.suggested_id_main2)
    c_alt1=c_alt[['Specification Number','variable','suggested_id_main2']]
    data_fin1=pd.merge(data_fin,c_alt[['Specification Number','variable','suggested_id_main2']],on=['Specification Number','variable'],how='left')
    #-----------------------------------------------------------------------------------------------
    #Crude Logic - Similarity based on value counts of high occurence similar spec ids(suggested)
    gr = data_fin1[data_fin1.Suggested_main=='Yes']
    gr1=gr[['cluster','variable']].value_counts().reset_index()
    gr1.columns=['cluster','variable','ind']
    gr1.sort_values(by=['cluster','ind'],ascending=False)
    gr2 = pd.merge(gr[['cluster','cluster_size']],gr1,on=['cluster'],how='left')
    gr2 = gr2.sort_values(by=['cluster','ind'],ascending=False)
    gr2 = gr2.drop_duplicates()
    gr3=pd.DataFrame()
    gr2['variable']=gr2['variable'].astype(str)
    for x in list(gr2.cluster.unique()):
        gr2_x=gr2[gr2.cluster==x]
        gr2_x['Suggested_ID_in_cluster'] = gr2_x[gr2_x.ind==gr2_x.ind.max()]['variable']
        gr3=concat([gr3, gr2_x])
    gr3 =gr3.dropna()
    gr4=pd.DataFrame()
    for x in list(gr3.cluster.unique()):
        gr3_x= gr3[gr3.cluster==x]
        reqd = list(gr3_x.Suggested_ID_in_cluster.unique())
        gr3_x['Suggested_ID_in_cluster1']= [reqd]*gr3_x.shape[0]
        gr4=concat([gr4, gr3_x])

    model=leveltwo(spend=spend,data=model)
    model=model.drop(['Classification'],axis=1)
    model=model.reset_index()
    model1 = pd.merge(model,data_fin1[data_fin1.Suggested_main=='Yes'][['Specification Number','variable','Suggested_main','value','suggested_id_main2']],on='Specification Number',how='left')
    vol=['H2 2021 - H1 2022 volume - NA, KG',
         'H2 2021 - H1 2022 volume - MEU, KG',
         'H2 2021 - H1 2022 volume - LA, KG',
         'H2 2021 - H1 2022 volume - AMEA, KG']
    spn=['H2 2021 - H1 2022 spend - NA, USD',
    'H2 2021 - H1 2022 spend - MEU, USD',
    'H2 2021 - H1 2022 spend - LA, USD',
    'H2 2021 - H1 2022 spend - AMEA, USD']
    model1['Volume']=model1[vol].sum(axis=1)  
    model1['Spend']=model1[spn].sum(axis=1)  
    model1['Average Price']=np.where(model1['Volume']!=0,model1['Spend']/model1['Volume'],0)
    #model1 - model2 output
    #Merge crude logic with model1
    gr4=gr4[['cluster', 'Suggested_ID_in_cluster1']]
    gr4['chk']=gr4.groupby('cluster').cumcount()
    gr4=gr4[gr4['chk']==0]
    gr4=gr4[['cluster','Suggested_ID_in_cluster1']]
    model11 = pd.merge(data_fin1,gr4,on=['cluster'],how='left')
    #-----------------------------------------------------------------------------------------------
    #Manual Cluster file from client

   
    df, drop_var, use_var, drop_list, manual = config_model()
        
    model11 =pd.merge(model11,manual,left_on='Specification Number',right_on='Spec number',how='left')
    model11=model11.drop('Spec number',axis=1)    
    model11['variable']=model11['variable'].astype(str)
    model12=pd.merge(model11,gr3[['cluster','variable','Suggested_ID_in_cluster']],on=['cluster','variable'],how='left')
    model12['check_var']=np.where(model12.variable==model12['Suggested_ID_in_cluster'],1,0)
    model12['Volume']=model12[vol].sum(axis=1)  
    model12['Spend']=model12[spn].sum(axis=1)  
    model12['Average Price']=np.where(model12['Volume']!=0,model12['Spend']/model12['Volume'],0)
        
        #-----------------------------------------------------------------------------------------------
    #MODEL 1, MODEL 2, FEATURE IMPORTANCE DATAFRAMES
    df_reqd=leveltwo(spend=spend,data=df_reqd)
    df_reqd=df_reqd.drop(['Classification'],axis=1)
    inputcol=df_reqd.columns.tolist() #to get input columns
    inputcol=pd.DataFrame(inputcol)
    inputcol.columns=['Input columns']

    #RF AND SHAP ON EACH CLUSTER TO GET FEATURE IMPORTANCE
    if out.shape[0]>=min_specid_req: # min no of spec ids to form clusters
        y = out[['cluster']]
        X =out.drop('cluster',axis=1)
        clf = RandomForestClassifier()
        clf.fit(X,y)
        shap_values = shap.TreeExplainer(clf,feature_perturbation='interventional').shap_values(X,check_additivity=False)
        feature_importance_all = pd.DataFrame()
        for i in range(0,out.cluster.nunique()):
            #shap.summary_plot(shap_values[i], X)
            feature_importance = pd.DataFrame(list(zip(X.columns, np.abs(shap_values[i]).mean(0))), columns=['feature_name', 'feature_importance_vals'])
            feature_importance = feature_importance.iloc[(-np.abs(feature_importance['feature_importance_vals'].values)).argsort()]
            feature_importance['cluster'] = i
            feature_importance_all=concat([feature_importance_all, feature_importance])
        f1 = feature_importance_all.groupby(['cluster']).agg({'feature_importance_vals':['min','max']}).reset_index() #scaling feature imp 
        f1.columns= ['cluster','min_','max_'] # scaling feature imp
        feature_importance_all = pd.merge(feature_importance_all,f1,on=['cluster'],how='left') # scaling feature imp
        feature_importance_all['feature_imp_scaled_vals']=np.where(feature_importance_all.max_==feature_importance_all.min_,0,
                                                       (feature_importance_all.feature_importance_vals-feature_importance_all.min_)/(feature_importance_all.max_-feature_importance_all.min_))#scaling f imp
    else : #change
        feature_importance_all = pd.DataFrame()
    return model12,model1,feature_importance_all


