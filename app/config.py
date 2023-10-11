# # PROJECT KITCHEN

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023

# # CONTENTS
#   To keep spend area parameter in the config file which helps in code automation
#   This parameter is being used in utils.py to determine the spend area classification accordingly
#   since the classification levels differ for each spend area. The function definition using this parameter
#   in utils.py have been called in kitchen.py file


from Libraries import *
import os


global model_type, min_specid_req,to_cat_col
min_specid_req=5 # min no. of specs required to make clusters regionwise
to_cat_col=2 # min no. of values that tell us to use the data as categorical
model_type='ward_euclidean' # defining model type

# defining regions
global Region_meu, Region_amea,Region_na, Region_la
Region_meu='MEU'
Region_amea='AMEA'
Region_na='NA'
Region_la='LA'

# Global variables for calling functions
#defining volumes
global volume_meu,volume_amea,volume_na,volume_la
volume_meu='H2 2021 - H1 2022 volume - MEU, KG'
volume_amea='H2 2021 - H1 2022 volume - AMEA, KG'
volume_na='H2 2021 - H1 2022 volume - NA, KG'
volume_la='H2 2021 - H1 2022 volume - LA, KG'

#defining spends
global spend_meu, spend_amea,spend_na,spend_la
spend_meu='H2 2021 - H1 2022 spend - MEU, USD'
spend_amea='H2 2021 - H1 2022 spend - AMEA, USD'
spend_na='H2 2021 - H1 2022 spend - NA, USD'
spend_la='H2 2021 - H1 2022 spend - LA, USD'

#defining menu cards
global menu_meu,menu_amea,menu_na,menu_la
menu_meu='Buy Menu Card - MEU - Menu Card Color'
menu_amea='Buy Menu Card - AMEA - Menu Card Color'
menu_na='Buy Menu Card - NA - Menu Card Color'
menu_la='Buy Menu Card - LA - Menu Card Color'

# Parameters for models
#K_med_cluster

global K_cluster_metric,K_cluster_method,K_cluster_init,K_cluster_iter
K_cluster_metric ='precomputed'
K_cluster_method='pam'
K_cluster_init='build' 
K_cluster_iter=300

#K_medoid_model
global K_med_metric,K_med_method,K_med_init,K_med_max_iter
K_med_metric='precomputed'
K_med_method='pam'
K_med_init='build' 
K_med_max_iter=300

# start_cluster and end cluster
global begin_cluster,end_cluster
begin_cluster=2
end_cluster=4

# famd model
global famd_n_iter,famd_copy,famd_check_input,famd_engine
famd_n_iter=3
famd_copy=True
famd_check_input=True
famd_engine='auto'

#Silhouette score
global sil_affinity, sil_linkage
sil_affinity ='euclidean'
sil_linkage = 'ward'

#model_pca
global pca_affinity, pca_linkage
pca_affinity ='euclidean'
pca_linkage = 'ward'

#HIERARCHICAL CLUSTERING MODEL + MODEL EXPLANABILITY + SIMILARITY SCORING
global model_affinity, model_linkage, model_methodd
model_methodd='ward'
model_linkage='ward'
model_affinity='euclidean'

def config_model():
# Global variables
    global spend
    # df_meta = pd.read_excel('Attributes_list.xlsx',sheet_name="Attributes_list")
    df_meta = pd.read_excel('selection_metadata.xlsx',sheet_name="selection_metadata")
    spend = df_meta['Selected'][0]
    print(spend)

    # Global variables
    #spend = 'citric'
    global model_type, min_specid_req,to_cat_col
    min_specid_req=5 # min no. of specs required to make clusters regionwise
    to_cat_col=2 # min no. of values that tell us to use the data as categorical
    model_type='ward_euclidean' # defining model type

    # defining regions
    global Region_meu, Region_amea,Region_na, Region_la
    Region_meu='MEU'
    Region_amea='AMEA'
    Region_na='NA'
    Region_la='LA'

    # Global variables for calling functions
    #defining volumes
    global volume_meu,volume_amea,volume_na,volume_la
    volume_meu='H2 2021 - H1 2022 volume - MEU, KG'
    volume_amea='H2 2021 - H1 2022 volume - AMEA, KG'
    volume_na='H2 2021 - H1 2022 volume - NA, KG'
    volume_la='H2 2021 - H1 2022 volume - LA, KG'

    #defining spends
    global spend_meu, spend_amea,spend_na,spend_la
    spend_meu='H2 2021 - H1 2022 spend - MEU, USD'
    spend_amea='H2 2021 - H1 2022 spend - AMEA, USD'
    spend_na='H2 2021 - H1 2022 spend - NA, USD'
    spend_la='H2 2021 - H1 2022 spend - LA, USD'

    #defining menu cards
    global menu_meu,menu_amea,menu_na,menu_la
    menu_meu='Buy Menu Card - MEU - Menu Card Color'
    menu_amea='Buy Menu Card - AMEA - Menu Card Color'
    menu_na='Buy Menu Card - NA - Menu Card Color'
    menu_la='Buy Menu Card - LA - Menu Card Color'

    # Parameters for models
    #K_med_cluster

    global K_cluster_metric,K_cluster_method,K_cluster_init,K_cluster_iter
    K_cluster_metric ='precomputed'
    K_cluster_method='pam'
    K_cluster_init='build' 
    K_cluster_iter=300

    #K_medoid_model
    global K_med_metric,K_med_method,K_med_init,K_med_max_iter
    K_med_metric='precomputed'
    K_med_method='pam'
    K_med_init='build' 
    K_med_max_iter=300

    # start_cluster and end cluster
    global begin_cluster,end_cluster
    begin_cluster=2
    end_cluster=4

    # famd model
    global famd_n_iter,famd_copy,famd_check_input,famd_engine
    famd_n_iter=3
    famd_copy=True
    famd_check_input=True
    famd_engine='auto'

    #Silhouette score
    global sil_affinity, sil_linkage
    sil_affinity ='euclidean'
    sil_linkage = 'ward'

    #model_pca
    global pca_affinity, pca_linkage
    pca_affinity ='euclidean'
    pca_linkage = 'ward'

    #HIERARCHICAL CLUSTERING MODEL + MODEL EXPLANABILITY + SIMILARITY SCORING
    global model_affinity, model_linkage, model_methodd
    model_methodd='ward'
    model_linkage='ward'
    model_affinity='euclidean'

    # prepare manual excel file with same format for attribute list automation
    global manual
    manual=pd.DataFrame(columns=['Spec number','Cluster name'])

    # Reading files to drop/use the attributes
    # global drop_var, use_var, 
    global drop_list, drop_var, use_var
    drop_list = pd.read_excel((os.getcwd()+'\Attribute_list_to_drop.xlsx'),sheet_name=spend)
    drop_var = False # Flag used for dropping attributes in the model  
    use_var= False  # Flag used for specified attributes in the model 

    # Reading input files according to the spend
# def config_model():   
    try:
        if spend =='dairy':
            df = pd.read_excel((os.getcwd()+'\Dairy PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            manual =pd.read_excel(os.getcwd()+'\\Kitchen MEU dairy spec manual clusters.xlsx',skiprows=[0],sheet_name='Spec detail')
            manual = manual[['Spec number','Cluster name']]
            drop_var=True
        
        elif spend=='fno':
            df = pd.read_excel((os.getcwd()+'\Fats&Oils PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            manual =pd.read_excel(os.getcwd()+'\\fno_manual_clusters.xlsx',sheet_name='Sheet1')
            manual = manual[['Spec number','Cluster name']]
            
        elif spend=='colors':
            df = pd.read_excel((os.getcwd()+'\Colors PLM extract for R&D (1).xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            manual =pd.read_excel(os.getcwd()+'\\clustering colours_WIP 221202.xlsx',sheet_name='all regions')
            manual.rename(columns ={'spec':'Spec number','Cluster No':'Cluster name'},inplace = True)
            manual = manual[['Spec number','Cluster name']]
        
        elif spend =='chocolates':
            global choco_cols
            df = pd.read_excel((os.getcwd()+'\Chocolate PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            choco_cols=pd.read_excel((os.getcwd()+'\chocolate_features.xlsx'),skiprows=[0,1])    
            
        elif spend=='citric':
            df = pd.read_excel((os.getcwd()+'\Citric acids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            
        elif spend=='cocoa_liq':
            df = pd.read_excel((os.getcwd()+'\Cocoa liquor PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')    
        
        elif spend =='cocoa_powder_alk':
            df = pd.read_excel((os.getcwd()+'\Cocoa powder PLM extract for R&D (1).xlsx'),skiprows = [0,1,2],sheet_name='Alkaline')
            
        elif spend =='sugar':
            df = pd.read_excel((os.getcwd()+'\Sugars PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            use_var=True
            
        elif spend =='starch':
            df = pd.read_excel((os.getcwd()+'\Starches PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend =='salt':
            df = pd.read_excel((os.getcwd()+'\Salts PLM extract for R&D (1).xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend =='carbonates':
            df = pd.read_excel((os.getcwd()+'\Carbonates PLM extract for R&D (2).xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend=='malic':
            df = pd.read_excel((os.getcwd()+'\Malic acids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend=='gelatin_pork':
            df = pd.read_excel((os.getcwd()+'\Gelatine PLM extract for R&D (2).xlsx'),skiprows = [0,1,2,3],sheet_name='Pork')
            manual =pd.read_excel(os.getcwd()+'\\Gelatin Manual Cluster.xlsx',sheet_name='Sheet1')
            use_var = True
        
        elif spend=='gelatin_beef':
            df = pd.read_excel((os.getcwd()+'\Gelatine PLM extract for R&D (2).xlsx'),skiprows = [0,1,2,3],sheet_name='Beef')
            manual =pd.read_excel(os.getcwd()+'\\Gelatin Manual Cluster.xlsx',sheet_name='Sheet1')
            use_var=True
        
        elif spend=='gelatin_extra':
            df = pd.read_excel((os.getcwd()+'\Gelatine PLM extract for R&D (2).xlsx'),skiprows = [0,1,2,3],sheet_name='Extra')
            manual =pd.read_excel(os.getcwd()+'\\Gelatin Manual Cluster.xlsx',sheet_name='Sheet1')
            use_var = True
        
        elif spend=='his':
            df = pd.read_excel((os.getcwd()+'\HIS PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend=='polyols':
            df = pd.read_excel((os.getcwd()+'\Polyols PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend=='lecithin':
            df = pd.read_excel((os.getcwd()+'\Lecithin data - PROPOSED - 10 Nov.xlsx'),skiprows = [0,1,2],sheet_name='PROPOSED data 4 autoclustering')
            use_var = True
        elif spend =='honey':
            df = pd.read_excel((os.getcwd()+'\Honey PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend=='Hazelnuts_Chopped_Roasted':
            df= pd.read_excel(('Hazelnuts_Chopped_Roasted_Nuts_PLM.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            use_var=True
        
        elif spend=='Almonds_Chopped_Roasted':
            df= pd.read_excel(('Almonds_Chopped_Roasted_Nuts_PLM.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')  
            use_var=True
        
        elif spend=='Almond_Whole_Roasted':
            df= pd.read_excel(('Almond_Whole_Roasted_PLM.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            use_var=True    
            
        elif spend=='syrups':
            df = pd.read_excel((os.getcwd()+'\Syrups PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend =='PGPR':
            df = pd.read_excel((os.getcwd()+'\PGPR PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend =='plant_foaming':
            df = pd.read_excel((os.getcwd()+'\Plant proteins and foaming ingredients PLM extract for R&D_Foaming.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
            
        elif spend =='plant_based':
            df = pd.read_excel((os.getcwd()+'\Plant proteins and foaming ingredients PLM extract for R&D_PlantProtein.xlsx'),skiprows = [0,1,2],sheet_name='Sheet1')
        
        elif spend =='CARRAGEENAN':
            df = pd.read_excel((os.getcwd()+'\All hydrocolloids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='MEU')
            df=df[df['SUB-AREA']=='CARRAGEENAN']
        
        elif spend =='PECTIN':
            df = pd.read_excel((os.getcwd()+'\All hydrocolloids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='MEU')
            df=df[df['SUB-AREA']=='PECTIN']
            
        elif spend =='H&G OTHERS':
            df = pd.read_excel((os.getcwd()+'\All hydrocolloids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='MEU')
            df=df[df['SUB-AREA']=='H&G OTHERS']
            
        elif spend =='GUM ARABIC':
            df = pd.read_excel((os.getcwd()+'\All hydrocolloids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='AMEA')
            df=df[df['SUB-AREA']=='GUM ARABIC']    
        
        elif spend =='XHANTAN GUM':
            df = pd.read_excel((os.getcwd()+'\All hydrocolloids PLM extract for R&D.xlsx'),skiprows = [0,1,2],sheet_name='AMEA')
            df=df[df['SUB-AREA']=='XHANTAN GUM']    

    except PermissionError:
        print("Your file is open. Kindly close it for the execution.")

    except FileNotFoundError:
        print("No such file is found. Kindly check the path.")

    except ValueError:
        print("No such sheet is found in excel. Kindly check the sheetname.")

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
      
    return df, drop_var, use_var, drop_list, manual

