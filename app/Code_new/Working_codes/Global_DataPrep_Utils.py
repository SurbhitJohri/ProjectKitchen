
# # PROJECT KITCHEN

# # Author : Harini & Harshita
#  Secondary Authors :Aditi & Arsh 
# ## Created Date  : September 26,2022
# ## Modified Date : February 15,2023



# # CONTENTS
# This DataPrep_Utils.py has the data preparation function definition for all the numeric, categorical and text
# features present in the data. The steps for data cleaning and processing are mentioned below:
# The function call of this is present in kitchen_global.py file.     
#

# # Data Pre-processing
# # Columns split - Numeric, Categorical, Text
# # Cleaning, Missing value, Outlier, Variance, Scaling, tf-idf
# # Input data for Model

# # Function - Data Pre-processing:
#
# Specid - Unique Identifier
# Region based split based on the main Numeric columns(Spend and Volume)
# Get all the column categories
# Form a df - region based
# Drop all blank columns and rows if any
# Drop Menu card details column
# Form three list with Numeric, Categorical and Text columns separately
# Make a copy of the df here to extract output with raw columns
#
# Pre-processing for all three datatypes done separately:
#
# Numeric:
# Missing value treatment
# Cleaning
# Correlation
# Outlier treatment
# Scaling
# Variance
#
# Columns - Quantitative-Max, Particle size-Max, 
#    
# Categorical:
# Text with one value convert to categorical column
# Missing value treatment
# Variance
# Chi-Square
# One-hot encoding
#
# Columns - Quantitative-CoA, Quantitative-Testing Frequency, Particle size-CoA, Particle size-Testing Frequency
#     
# Text:
# Missing value treatment
# classification level split
# Variance
# Combine all text columns
# Remove punctuation
# Remove numerics
# Remove single letters
# Tokenization
# Remove stopwords
# Text vectorization - TFIDF
#
# Columns - Specification description, Classification, Quantitative-Reference Method, Quantitative-Testing Details, Particle size-Reference Method, Particle size-Testing Details
#
# Combine all the features together in both all numerics and numerics+categorical without one-hot for different clustering models
#
# Remove few raw columns that have been treated so that we give the treated columns only into the model
# Get a list of all unused columns


#FUNCTION - DATA PRE-PROCESSING
import os
from Libraries import *
from Utils import *
from config import config_model
from pandas import concat


def PLM_preprocessing():
    df_meta = pd.read_excel(r"C:\Users\ERY3041\Downloads\Code_new_21062023 (1)\Code_new\selection_metadata.xlsx",sheet_name="selection_metadata")
    spend = df_meta['Selected'][0]
    df = pd.DataFrame([])
    df, drop_var, use_var, drop_list, manual = config_model()
    to_cat_col=2

    df.rename(columns = {'Spec number':'Specification Number', 'Ion Classification':'Classification',
                        'Corrected PLM classification':'Classification','last 12 months volume - NA, KG':'H2 2021 - H1 2022 volume - NA, KG', 
                        'last 12 months volume - MEU, KG':'H2 2021 - H1 2022 volume - MEU, KG',
                        'last 12 months volume - LA, KG':'H2 2021 - H1 2022 volume - LA, KG', 
                        'last 12 months volume - AMEA, KG':'H2 2021 - H1 2022 volume - AMEA, KG'
                        }, inplace = True) #rename required columns
    
    if(df['Specification Number'].dtype!=np.int64):
        df['Specification Number'] =df['Specification Number'].astype(np.int64)
    df=df.set_index('Specification Number') #set index
    #df['Classification']=df['Classification'].ffill()

    # It is used to drop the specified attributes
    if drop_var==True: 
        remove_attr = drop_list['Attributes_list'].tolist()
        df.drop(remove_attr,axis=1,inplace=True) #20230420
    
    # It is used to consider the specified attributes
    if use_var==True:
        use_attr = drop_list['Attributes_list'].tolist()
        if 'gelatin'in spend:
            use_attr = drop_list[drop_list['Attributes_list'].str.contains('PSD')]['Attributes_list'].tolist()
        use_attr = ['Classification','Specification description',volume_meu,volume_amea,volume_na,volume_la,spend_meu,spend_amea,spend_na,spend_la,menu_meu,menu_amea,menu_na,menu_la]+use_attr
        df=df[use_attr]
    
    if spend =='dairy':
        #Vincent's file column mapping
        df_map = pd.read_excel('simplified component list.xlsx',sheet_name='Sheet2')
        df_map = df_map[['Identifier 1','Replace by (component simplified for origin for auto-clustering, e.g apple puree concentrate = apple)']]
        df_map['Standard Composition'] = 'Standard Composition'
        df_map['value'] = 'Value'
        df_map =df_map.rename(columns={'Identifier 1':'Column in Data',
                                    'Replace by (component simplified for origin for auto-clustering, e.g apple puree concentrate = apple)':'Column replacement'})
        df_map['Column in Data']= df_map['Standard Composition']+" - "+df_map['Column in Data']+" - "+df_map['value']+" - %"
        df_map['Column replacement']= df_map['Standard Composition']+" - "+df_map['Column replacement']+" - "+df_map['value']+" - %"
        filter_col = [col for col in df if col.startswith('Standard Composition')]
        df_map=df_map[df_map['Column in Data'].isin(filter_col)]
        df_map=df_map.drop_duplicates()
        #Data aggregation
        filter_col = [col for col in df if col.startswith('Standard Composition')]
        #df=df.set_index('Specification Number')
        df_sub = df[filter_col]
        df_cols_nonull  = df_map
        drop = list()
        for x in list(df_cols_nonull['Column replacement'].unique()):
            df_cols_x=df_cols_nonull[df_cols_nonull['Column replacement']==x]
            df_sub[x+'_1'] = df_sub[list(df_cols_x['Column in Data'].unique())].sum(axis=1)
            drop = drop + list(df_cols_x['Column in Data'].unique())
        df_sub1=df_sub.drop(drop,axis=1)
        df_sub1.columns = df_sub1.columns.str.rstrip('_1')
        df = df.drop(filter_col,axis=1)
        df = pd.merge(df,df_sub1,left_index=True,right_index=True)
        stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))]
        for x in stdcomp:
            df[x]=df[x].fillna(0)   
        #Checking std composition sum
        df_sub1['total']=df_sub1[list(df_sub1.columns.unique())].sum(axis=1)
        df_sub1[df_sub1['total']!=1]
        #------------------------------------------------------------------------------

    mand1=df[[
        'Specification description',
                'Classification', #taking out stand alone columns and saving it in a df
    'H2 2021 - H1 2022 volume - NA, KG',
    'H2 2021 - H1 2022 volume - MEU, KG',
    'H2 2021 - H1 2022 volume - LA, KG',
    'H2 2021 - H1 2022 volume - AMEA, KG',
    'H2 2021 - H1 2022 spend - NA, USD',
    'H2 2021 - H1 2022 spend - MEU, USD',
    'H2 2021 - H1 2022 spend - LA, USD',
    'H2 2021 - H1 2022 spend - AMEA, USD',
    'Buy Menu Card - NA - Menu Card Color',
    'Buy Menu Card - MEU - Menu Card Color',
    'Buy Menu Card - LA - Menu Card Color',
    'Buy Menu Card - AMEA - Menu Card Color']]
    #-------------------------------------------------------------------------------
    #GET CATEGORIES OF COLUMNS (Numeric + Cat + Text)
    #Quantitative
    quant = [col for col in df if col.startswith('Quantitative')]
    quant1= [i for i in quant if ('Max') in i] 
    quant11= [i for i in quant if ('Target') in i] 
    quant2= [i for i in quant if ('CoA') in i]  
    quant3= [i for i in quant if ('Reference Method') in i]  
    quant4= [i for i in quant if ('Key Testing Details') in i]  
    quant5= [i for i in quant if ('Testing Frequency') in i]  
    quantwant=quant1+quant11+quant2+quant3+quant4+quant5
    #Particle Size
    part_size = [col for col in df if col.startswith('Particle')]
    part_size1= [i for i in part_size if ('Max') in i] #num
    part_size2= [i for i in part_size if ('CoA') in i]  #cat
    part_size3= [i for i in part_size if ('Reference Method') in i]  #txt
    part_size4= [i for i in part_size if ('Key Testing Details') in i]  #txt
    part_size5= [i for i in part_size if ('Testing Frequency') in i] #cat
    part_sizewant=part_size1+part_size2+part_size3+part_size4+part_size5
    #Chemical
    chemical = [col for col in df if col.startswith('Chemical Composition')]
    chemical1= [i for i in chemical if ('Max') in i] #num
    chemical2= [i for i in chemical if ('CoA') in i]  #cat
    chemical3= [i for i in chemical if ('Reference Method') in i]  #txt
    chemical4= [i for i in chemical if ('Key Testing Details') in i]  #txt
    chemical5= [i for i in chemical if ('Testing Frequency') in i] #cat
    chemicalwant=chemical1+chemical2+chemical3+chemical4+chemical5
    #Physical Tests
    phy = [col for col in df if col.startswith('Physical Tests')]
    phy1= [i for i in phy if ('Max') in i] #num
    phy2= [i for i in phy if ('CoA') in i]  #cat
    phy3= [i for i in phy if ('Reference Method') in i]  #txt
    phy4= [i for i in phy if ('Key Testing Details') in i]  #txt
    phy5= [i for i in phy if ('Testing Frequency') in i] #cat
    phywant=phy1+phy2+phy3+phy4+phy5
    #International Sieves
    intt = [col for col in df if col.startswith('International Sieves')]
    int1= [i for i in intt if ('Max') in i] #num
    int2= [i for i in intt if ('CoA') in i]  #cat
    int3= [i for i in intt if ('Reference Method') in i]  #txt
    int4= [i for i in intt if ('Key Testing Details') in i]  #txt
    int5= [i for i in intt if ('Testing Frequency') in i] #cat
    intwant=int1+int2+int3+int4+int5
    #Food Safety - 
    foodcat = [col for col in df if col.startswith('Food Safety -')]
    foodcat1= [i for i in foodcat if ('Maximum') in i] #num
    foodcat2= [i for i in foodcat if ('CoA') in i]  #cat
    foodcat3= [i for i in foodcat if ('Reference Method') in i]  #txt
    foodcat4= [i for i in foodcat if ('TXT') in i]  #txt
    foodcatwant=foodcat1+foodcat2+foodcat3+foodcat4
    #Additional Particle
    addpart_size = [col for col in df if col.startswith('Additional Particle')]
    addpart_size1= [i for i in addpart_size if ('Min') in i] #num
    #Status
    status =  [col for col in df if col.startswith('Status')]#cat
    #Standard Composition
    stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))]
    stdwant= [i for i in stdcomp if ('Value') in i]
    if spend =='chocolates':
        choco_cols['stdcomp']='Standard Composition'
        choco_cols['value']='Value, %'
        choco_cols['attr']=choco_cols['stdcomp']+' - '+choco_cols['Ingredient']+' - '+choco_cols['value']
        stdwant=list(set(stdwant).intersection(set(list(choco_cols.attr.unique()))))
    #Nutrient Composition
    nutcomp = [col for col in df if col.startswith('Nutrient Composition')]
    nutwant= [i for i in nutcomp if ('Value') in i] 
    #Shipping
    ship = [col for col in df if col.startswith('Shipping')]
    #Storage
    stor = [col for col in df if col.startswith('Storage')]
    #allergens
    allergens = [col for col in df if col.startswith('Allergens')]
    #desc
    descriptive = [col for col in df if col.startswith('Descriptive')]
    #Starch
    Starch = [col for col in df if col.startswith('Starch')]
    #State
    State = [col for col in df if col.startswith('State')]
    #HIS type
    HIS = [col for col in df if col.startswith('HIS type')]
    #Diet
    diet = [col for col in df if col.startswith('Diet')]
    #Qualitative
    qual = [col for col in df if col.startswith('Qualitative')]
    #Food
    food = [col for col in df if col.startswith('Food Safety Requirements')]
    #Total Solids
    solid = [col for col in df if col.startswith('Total Solids')]
    #Origin
    origin = [col for col in df if col.startswith('Origin')]
    #Color
    color = [col for col in df if col.startswith('Color')]
    #GM Assessment
    gm = [col for col in df if col.startswith('GM Assessment')]
    #Absence of Engineered
    abs_ = [col for col in df if col.startswith('Absence of Engineered')]
    #composition
    composition = [col for col in df if col.startswith('Composition')]
    #type
    type_ = [col for col in df if col.startswith('Type')]
    #subtype
    subtype= [col for col in df if col.startswith('Subtype')]
    # religious
    religious = [col for col in df if col.startswith('Religious')]
    # halal
    halal=[col for col in df if col.startswith('Halal')]
    #kosher
    kosher=[col for col in df if col.startswith('Kosher')]
    # microb
    microb=[col for col in df if col.startswith('Microbiological')]
    #approved supplier
    if spend=='colors':
        app_sup=[]
    else:
        app_sup=[col for col in df if col.startswith('Approved Supplier')]
    #gelatine
    g_origin = [col for col in df if col.startswith('Gelatine')]
    #source
    source = [col for col in df if col.startswith('Source')]
    #format
    format_=[col for col in df if col.startswith('Format')]
    #solid
    solids = [col for col in df if col.startswith('Solids')]
    #bloom
    bloom = [col for col in df if col.startswith('bloom')]
    #Taking required columns
    columns=bloom+solids+source+format_+g_origin+app_sup+microb+religious+halal+kosher+quantwant+part_sizewant+chemicalwant+phywant+intwant+foodcatwant+addpart_size1+stdwant+nutwant+ship+stor+allergens+descriptive+Starch+State+HIS+diet+qual+food+solid+origin+color+gm+abs_+composition+type_+subtype+status
    mand2=df[columns] #taking the categories of columns in a df

    df=pd.merge(mand1,mand2,left_index=True,right_index=True,how='left') #merge both stand alone and categories column data together
    df1=df.copy() #to get total columns list
    
    df=df.drop(['Buy Menu Card - NA - Menu Card Color',
    'Buy Menu Card - MEU - Menu Card Color',
    'Buy Menu Card - LA - Menu Card Color',
    'Buy Menu Card - AMEA - Menu Card Color'],axis=1)
    
    df = df.dropna(how='all', axis=1) #drop all blank columns
    
    Areas = ['H2 2021 - H1 2022 volume - NA, KG', #taking out region columns separately
            'H2 2021 - H1 2022 volume - MEU, KG',
            'H2 2021 - H1 2022 volume - LA, KG',
            'H2 2021 - H1 2022 volume - AMEA, KG',
            'H2 2021 - H1 2022 spend - NA, USD',
            'H2 2021 - H1 2022 spend - MEU, USD',
            'H2 2021 - H1 2022 spend - LA, USD',
            'H2 2021 - H1 2022 spend - AMEA, USD',
            'Buy Menu Card - NA - Menu Card Color',
            'Buy Menu Card - MEU - Menu Card Color',
            'Buy Menu Card - LA - Menu Card Color',
            'Buy Menu Card - AMEA - Menu Card Color']
    #MEU = [i for i in Areas if var in i]  #taking out region columns separately
    #df = df.drop(list(set(Areas).difference(set(MEU))),axis=1)  #taking out region columns separately
    #drop menu card details
    col=df.columns.tolist() #to get column list after dropping unwanted columns

    #Take out Numeric, Categorical and Text columns separately
    allergens = [col for col in df if col.startswith('Allergens')] #cat
    quant = [col for col in df if col.startswith('Quantitative')]
    quant1= [i for i in quant if ('Max') in i] #num
    quant2= [i for i in quant if ('CoA') in i]  #cat
    quant3= [i for i in quant if ('Reference Method') in i]  #txt
    quant4= [i for i in quant if ('Key Testing Details') in i]  #txt
    quant5= [i for i in quant if ('Testing Frequency') in i]  #cat
    part_size = [col for col in df if col.startswith('Particle')]
    part_size1= [i for i in part_size if ('Max') in i] #num
    part_size2= [i for i in part_size if ('CoA') in i]  #cat
    part_size3= [i for i in part_size if ('Reference Method') in i]  #txt
    part_size4= [i for i in part_size if ('Key Testing Details') in i]  #txt
    part_size5= [i for i in part_size if ('Testing Frequency') in i] #cat
    chemical = [col for col in df if col.startswith('Chemical Composition')]
    chemical1= [i for i in chemical if ('Max') in i] #num
    chemical2= [i for i in chemical if ('CoA') in i]  #cat
    chemical3= [i for i in chemical if ('Reference Method') in i]  #txt
    chemical4= [i for i in chemical if ('Key Testing Details') in i]  #txt
    chemical5= [i for i in chemical if ('Testing Frequency') in i] #cat
    phy = [col for col in df if col.startswith('Physical Tests')]
    phy1= [i for i in phy if ('Max') in i] #num
    phy2= [i for i in phy if ('CoA') in i]  #cat
    phy3= [i for i in phy if ('Reference Method') in i]  #txt
    phy4= [i for i in phy if ('Key Testing Details') in i]  #txt
    phy5= [i for i in phy if ('Testing Frequency') in i] #cat
    intt = [col for col in df if col.startswith('International Sieves')]
    int1= [i for i in intt if ('Max') in i] #num
    int2= [i for i in intt if ('CoA') in i]  #cat
    int3= [i for i in intt if ('Reference Method') in i]  #txt
    int4= [i for i in intt if ('Key Testing Details') in i]  #txt
    int5= [i for i in intt if ('Testing Frequency') in i] #cat
    foodcat = [col for col in df if col.startswith('Food Safety -')]
    foodcat1= [i for i in foodcat if ('Maximum') in i] #num
    foodcat2= [i for i in foodcat if ('CoA') in i]  #cat
    foodcat3= [i for i in foodcat if ('Reference Method') in i]  #txt
    foodcat4= [i for i in foodcat if ('TXT') in i]  #txt
    solids = [col for col in df if col.startswith('Solids')] #num
    stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))] 
    stdwant = [i for i in stdcomp if ('Value') in i] #num
    # taking selected attributes for chocolates
    if spend =='chocolates':
        choco_cols['stdcomp']='Standard Composition'
        choco_cols['value']='Value, %'
        choco_cols['attr']=choco_cols['stdcomp']+' - '+choco_cols['Ingredient']+' - '+choco_cols['value']
        stdwant=list(set(stdwant).intersection(set(list(choco_cols.attr.unique()))))
        stdwant=list(set(stdwant).intersection(set(list(cols.attributes.unique()))))    
    nutrients = [col for col in df if col.startswith('Nutrient')] 
    nut_want = [i for i in nutrients if ('Value') in i] #num
    Storage = [col for col in df if col.startswith('Storage')]
    Min_Remaining= [i for i in Storage if 'Min Remaining SL Upon Receipt' in i]  #num
    shelf_life= [i for i in Storage if 'Shelf Life' in i] #num
    store_txt = [i for i in Storage if 'TXT' in i] #txt
    storage_cndtn= [i for i in Storage if 'Storage Condition' in i] #cat
    descriptive = [col for col in df if col.startswith('Descriptive')]
    desc2= [i for i in descriptive if ('CoA') in i] #cat
    desc3= [i for i in descriptive if ('Testing Frequency') in i] #cat
    desc_txt = [i for i in descriptive if 'TXT' in i] #txt
    ship = [col for col in df if col.startswith('Shipping')] #cat
    Starch = [col for col in df if col.startswith('Starch')] #cat
    State = [col for col in df if col.startswith('State')] #cat
    HIS = [col for col in df if col.startswith('HIS type')] #cat
    diet = [col for col in df if col.startswith('Diet')]
    qual = [col for col in df if col.startswith('Qualitative')]
    addpart_size = [col for col in df if col.startswith('Additional Particle')]
    addpart_size1= [i for i in addpart_size if ('Min') in i] #num
    food = [col for col in df if col.startswith('Food Safety Requirements')] #cat
    solid = [col for col in df if col.startswith('Total Solids')] #num
    origin = [col for col in df if col.startswith('Origin')] #cat
    color = [col for col in df if col.startswith('Color')] #cat
    gm = [col for col in df if col.startswith('GM Assessment')] #cat
    abs_ = [col for col in df if col.startswith('Absence of Engineered')] #cat
    composition = [col for col in df if col.startswith('Composition')] #cat
    type_ = [col for col in df if col.startswith('Type')]#cat
    subtype= [col for col in df if col.startswith('Subtype')]#cat
    status =  [col for col in df if col.startswith('Status')]#cat
    religious = [col for col in df if col.startswith('Religious')]#cat
    # halal
    halal=[col for col in df if col.startswith('Halal')]#cat
    #kosher
    kosher=[col for col in df if col.startswith('Kosher')]#cat
    # microb
    microb=[col for col in df if col.startswith('Microbiological')]
    microb1= [i for i in microb if ('I/C') in i] #cat
    microb2= [i for i in microb if ('CoA') in i]  #cat
    microb3= [i for i in chemical if ('Reference Method') in i]  #cat
    microb4= [i for i in chemical if ('Samples') in i]  #num
    microb5= [i for i in chemical if ('Testing Frequency') in i] #cat
    microb6= [i for i in chemical if ('Marginally') in i]#num
    microb7= [i for i in chemical if ('Maximum') in i]#num
    microb8= [i for i in chemical if ('TXT') in i]#txt
    #approved supplier
    app_sup=[col for col in df if col.startswith('Approved Supplier')]#cat
    #gelatine
    g_origin = [col for col in df if col.startswith('Gelatine')]#cat
    #source
    source = [col for col in df if col.startswith('Source')]#cat
    #format
    format_=[col for col in df if col.startswith('Format')]#cat
    #bloom
    bloom=[col for col in df if col.startswith('bloom')]#num
    #NUMERIC
    num=microb4+microb6+microb7+quant1+part_size1+int1+chemical1+phy1+solids+stdwant+nut_want+Min_Remaining+shelf_life+addpart_size1+solid+foodcat1+bloom #numeric columns

    #TEXT
    txt=['Specification description','Classification']+microb8+quant3+quant4+int3+int4+phy3+phy4+chemical3+chemical4+store_txt+part_size3+part_size4+desc_txt+foodcat3+foodcat4 #text columns #CHECK WHETHER CLASSIFICATION COLUMN TEXT OR CAT
    # pd.DataFrame(txt).to_csv("txt.csv",index=False)

    #CATEGORICAL
    cat=source+format_+g_origin+app_sup+microb1+microb2+microb3+microb5+halal+kosher+religious+ship+storage_cndtn+quant2+quant5+int2+int5+phy2+phy5+chemical2+chemical5+part_size5+part_size2+desc2+desc3+allergens+Starch+State+HIS+diet+qual+food+foodcat2+origin+color+gm+abs_+composition+type_+subtype+status #categorical columns

    df_reqd=df.copy() #to keep a copy of the required data
    #---------------------------------------------------------------------------------------------

    #1. NUMERIC COLUMNS

    #txt with one val to cat
    num = list(set(num).intersection(set(list(df.columns.unique()))))
    df_num_1 = df[num] 
    num_col_one_val = [] 
    for x in list(df_num_1.columns.unique()): 
        df_num_o= df_num_1[[x]].dropna() 
        df_num_o[x]= df_num_o[x].astype(str)
        if df_num_o[x].value_counts().reset_index().shape[0]<= to_cat_col:
            num_col_one_val = num_col_one_val + [x] 
    #Fill numerics with 0      
    for x in num:
        df[x]=df[x].fillna(0) 
    #numeric column cleaning for , and text words in it
    def storage(df,x): 
        df_hr  = df[df[x].astype(str).str.contains('hrs')]
        df_hr[x] = df_hr[x].astype(str).str.strip('hrs')
        df_hr[x] = df_hr[x].astype('int64')
        df_hr[x] = df_hr[x]/24
        df_day = df[df[x].astype(str).str.contains('Days')]
        df_day[x] = df_day[x].astype(str).str.strip('Days')
        df_day[x] =df_day[x].astype(str).str.replace(',','')
        df_day[x] = df_day[x].astype('int64')
        df_null = df[df[x].isnull()]
        df_null[x]=df_null[x].fillna(0)
        df_new = concat([df_hr,df_day])
        df_new = concat([df_new, df_null])
        return df_new[x]
    strg = [col for col in df[num] if col.startswith('Storage - ')]
    for x in strg:
        df[x]= storage(df,x)
    for x in num:
        df[x]=df[x].fillna(0) 
        
    #outlier treatment for num columns
    vol=['H2 2021 - H1 2022 volume - NA, KG',
            'H2 2021 - H1 2022 volume - MEU, KG',
            'H2 2021 - H1 2022 volume - LA, KG',
            'H2 2021 - H1 2022 volume - AMEA, KG']
    spn=['H2 2021 - H1 2022 spend - NA, USD',
    'H2 2021 - H1 2022 spend - MEU, USD',
    'H2 2021 - H1 2022 spend - LA, USD',
    'H2 2021 - H1 2022 spend - AMEA, USD']
    df['Volume treated']=df[vol].sum(axis=1)  
    df['Spend treated']=df[spn].sum(axis=1)
    df['Volume treated']=outlier(df,'Volume treated',1.5,'mean') 
    df['Spend treated']=outlier(df,'Spend treated',1.5,'mean')  

    #Correlation for numerical variables - removed

    #SCALING
    # =============================================================================
    #    scal=['Volume treated','Spend treated']
    #    num=num + scal
    # =============================================================================
    #Filter only numeric columns
    df_num=df[num]
    #clean if a num column has ,
    for x in quant1:
        df_num[x]=df_num[x].astype('str').str.replace(',','.').astype('float64') 
    df_num = df_num.drop(num_col_one_val,axis=1) 

    df_num_scaled=pd.DataFrame()    
    scaler = MinMaxScaler()
    model=scaler.fit(df_num)
    scaled_data=model.transform(df_num)
    scaled_num_data= pd.DataFrame(scaled_data, index=df_num.index, columns=df_num.columns)
    df_numeric=scaled_num_data.copy() 

    #Variance check and remove 0% variance from numeric columns
    def varianceSelection(data, THRESHOLD):
        sel = VarianceThreshold(threshold=(THRESHOLD * (1 - THRESHOLD)))
        sel.fit_transform(data)
        return data[[c for (s, c) in zip(sel.get_support(),data.columns.values) if s]]
    df_num = varianceSelection(df_numeric,1)
    attr_num=df_num.columns.tolist()
    
    #-------------------------------------------------------------------------------------------

    #2. CATEGORICAL COLUMNS

    #Fill missing categorical with 'Not Applicable'
    #txt with one val to cat
    df_txt_1 = df[txt] 
    txt_col_one_val = [] 
    for x in list(df_txt_1.columns.unique()): 
        df_txt_o= df_txt_1[[x]].dropna() 
        if df_txt_o[x].value_counts().reset_index().shape[0] <= to_cat_col: 
            txt_col_one_val = txt_col_one_val + [x] 
    df_cat=pd.DataFrame() 
    cat = cat+txt_col_one_val +num_col_one_val

    for x in cat:
        df_cat[x]=df[x].fillna('Not Applicable')
        df_cat[x]= df_cat[x].astype(str)

    #Variance check
    #REMOVING 0% VARIANCE    
    #IF VALUE COUNTS=1 - REMOVE THOSE COLUMNS
    catcol=pd.DataFrame(df_cat.nunique(axis=0))
    catcol.reset_index(inplace=True)
    catcol.columns=['col','value']
    catcol1=catcol.copy()
    df_cat1=df_cat[catcol1['col']]
    catcol=catcol[catcol['value']>1]
    catcol=catcol[['col']]
    df_cat=df_cat[catcol['col']]

    #Correlation/Relationship for categorical variables usign chi square test
    #There exists a relationship between two variables if p value ≤ 0.05.
    #Has association but still we are keeping it as per business need

    #One-hot for Categorical columns
    df_cat_onehot = pd.get_dummies(df_cat, columns = df_cat.columns)
    cat_cols = list(df_cat.columns.unique())
    df_cat_onehot1 =  df_cat_onehot.copy()

    #Checking for 0% Variance
    df_cat_onehot = varianceSelection(df_cat_onehot,1)
    cat_list = list(set(list(df_cat_onehot.columns.unique().difference(set(list(df_cat_onehot1.columns.unique())))))) #check number of dropped columns in variance 
    #-----------------------------------------------------------------------------------------

    #3. TEXT COLUMNS

    #Fill text with 'Others'
    df_txt=pd.DataFrame()
    for x in txt:
        df_txt[x]=df[x].fillna('Others')
    #txt with one val to cat  
    #Classification column split - keeping here bcoz of many levels with slight value change


    df_txt,txtcol=levelone(spend=spend,df_txt=df_txt)
    #txt with one val to cat

    if 'Classification' in txt_col_one_val:
        txt_col_one_val.remove('Classification')
    df_txt = df_txt.drop(txt_col_one_val,axis=1)    
    txtcol.reset_index(inplace=True)
    txtcol.columns=['col','value']
    df_txt1=df_txt.copy()
    txtcol=txtcol[txtcol['value']==1]
    txtcol=txtcol[['col']]
    df_txt=df_txt.drop(list(txtcol['col'].unique()),axis=1)

    attr_text=df_txt.columns.to_list()
    level_col=[col for col in df_txt if col.startswith('Level')]
    for i in level_col:
        attr_text.remove(i)
    attr_text.extend(['Classification'])

    #list of attributes used in the model
    attributes=attr_num + cat_cols + attr_text
    attributes_df=pd.DataFrame({'Attributes_list':attributes})

    #converting the list of columns to csv
    attr_num_df = pd.DataFrame({'Attributes_list':attr_num, 'Type':'num'})
    attr_cat_df = pd.DataFrame({'Attributes_list':cat_cols, 'Type':'cat'})   #might have to do encoding again for selected columns
    attr_text_df = pd.DataFrame({'Attributes_list':attr_text, 'Type':'text'})
    Attributes_list = pd.concat([attr_num_df,attr_cat_df, attr_text_df],ignore_index=True)
    Attributes_list.to_csv("Attributes_list_combined_all.csv",index=False)


    # df_num.to_excel("df_num.xlsx")
    # df_txt.to_excel("df_txt.xlsx")
    # df_cat.to_excel("df_cat.xlsx")
    # df_cat_onehot.to_excel("df_cat_onehot.xlsx",index=False)
    # df_reqd.to_excel("df_reqd.xlsx")
    
    attributes_df.to_csv(spend+"_Attributes_list.csv",index=False)

    return Attributes_list



def preprocessing(var_set):

    #    df_meta = pd.read_excel('selection_metadata.xlsx',sheet_name="selection_metadata")
    #    spend = df_meta['Selected'][0]
    df, drop_var, use_var, drop_list, manual = config_model()
    to_cat_col=2
    df_meta = pd.read_excel('selection_metadata.xlsx',sheet_name="selection_metadata")
    spend = df_meta['Selected'][0]
    df.rename(columns = {'Spec number':'Specification Number', 'Ion Classification':'Classification',
                        'Corrected PLM classification':'Classification','last 12 months volume - NA, KG':'H2 2021 - H1 2022 volume - NA, KG', 
                        'last 12 months volume - MEU, KG':'H2 2021 - H1 2022 volume - MEU, KG',
                        'last 12 months volume - LA, KG':'H2 2021 - H1 2022 volume - LA, KG', 
                        'last 12 months volume - AMEA, KG':'H2 2021 - H1 2022 volume - AMEA, KG'
                        }, inplace = True) #rename required columns

    if(df['Specification Number'].dtype!=np.int64):
        df['Specification Number'] =df['Specification Number'].astype(np.int64)
    df=df.set_index('Specification Number') #set index
    #df['Classification']=df['Classification'].ffill()

    # It is used to drop the specified attributes
    if drop_var==True: 
        remove_attr = drop_list['Attributes_list'].tolist()
        df.drop(remove_attr,axis =1,inplace=True) #20230420

    # It is used to consider the specified attributes
    if use_var==True:
        use_attr = drop_list['Attributes_list'].tolist()
        if 'gelatin'in spend:
            use_attr = drop_list[drop_list['Attributes_list'].str.contains('PSD')]['Attributes_list'].tolist()
        use_attr = ['Classification','Specification description',volume_meu,volume_amea,volume_na,volume_la,spend_meu,spend_amea,spend_na,spend_la,menu_meu,menu_amea,menu_na,menu_la]+use_attr
        df=df[use_attr]

    if spend =='dairy':
    #Vincent's file column mapping
        df_map = pd.read_excel('simplified component list.xlsx',sheet_name='Sheet2')
        df_map = df_map[['Identifier 1','Replace by (component simplified for origin for auto-clustering, e.g apple puree concentrate = apple)']]
        df_map['Standard Composition'] = 'Standard Composition'
        df_map['value'] = 'Value'
        df_map =df_map.rename(columns={'Identifier 1':'Column in Data',
                                        'Replace by (component simplified for origin for auto-clustering, e.g apple puree concentrate = apple)':'Column replacement'})
        df_map['Column in Data']= df_map['Standard Composition']+" - "+df_map['Column in Data']+" - "+df_map['value']+" - %"
        df_map['Column replacement']= df_map['Standard Composition']+" - "+df_map['Column replacement']+" - "+df_map['value']+" - %"
        filter_col = [col for col in df if col.startswith('Standard Composition')]
        df_map=df_map[df_map['Column in Data'].isin(filter_col)]
        df_map=df_map.drop_duplicates()
        #Data aggregation
        filter_col = [col for col in df if col.startswith('Standard Composition')]
        #df=df.set_index('Specification Number')
        df_sub = df[filter_col]
        df_cols_nonull  = df_map
        drop = list()
        for x in list(df_cols_nonull['Column replacement'].unique()):
            df_cols_x=df_cols_nonull[df_cols_nonull['Column replacement']==x]
            df_sub[x+'_1'] = df_sub[list(df_cols_x['Column in Data'].unique())].sum(axis=1)
            drop = drop + list(df_cols_x['Column in Data'].unique())
        df_sub1=df_sub.drop(drop,axis=1)
        df_sub1.columns = df_sub1.columns.str.rstrip('_1')
        df = df.drop(filter_col,axis=1)
        df = pd.merge(df,df_sub1,left_index=True,right_index=True)
        stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))]
        for x in stdcomp:
            df[x]=df[x].fillna(0)   
        #Checking std composition sum
        df_sub1['total']=df_sub1[list(df_sub1.columns.unique())].sum(axis=1)
        df_sub1[df_sub1['total']!=1]
    #------------------------------------------------------------------------------


    mand1=df[['Specification description',
                'Classification', #taking out stand alone columns and saving it in a df
    'H2 2021 - H1 2022 volume - NA, KG',
    'H2 2021 - H1 2022 volume - MEU, KG',
    'H2 2021 - H1 2022 volume - LA, KG',
    'H2 2021 - H1 2022 volume - AMEA, KG',
    'H2 2021 - H1 2022 spend - NA, USD',
    'H2 2021 - H1 2022 spend - MEU, USD',
    'H2 2021 - H1 2022 spend - LA, USD',
    'H2 2021 - H1 2022 spend - AMEA, USD',
    'Buy Menu Card - NA - Menu Card Color',
    'Buy Menu Card - MEU - Menu Card Color',
    'Buy Menu Card - LA - Menu Card Color',
    'Buy Menu Card - AMEA - Menu Card Color']]
    #-------------------------------------------------------------------------------
    #GET CATEGORIES OF COLUMNS (Numeric + Cat + Text)
    #Quantitative
    quant = [col for col in df if col.startswith('Quantitative')]
    quant1= [i for i in quant if ('Max') in i] 
    quant11= [i for i in quant if ('Target') in i] 
    quant2= [i for i in quant if ('CoA') in i]  
    quant3= [i for i in quant if ('Reference Method') in i]  
    quant4= [i for i in quant if ('Key Testing Details') in i]  
    quant5= [i for i in quant if ('Testing Frequency') in i]  
    quantwant=quant1+quant11+quant2+quant3+quant4+quant5
    #Particle Size
    part_size = [col for col in df if col.startswith('Particle')]
    part_size1= [i for i in part_size if ('Max') in i] #num
    part_size2= [i for i in part_size if ('CoA') in i]  #cat
    part_size3= [i for i in part_size if ('Reference Method') in i]  #txt
    part_size4= [i for i in part_size if ('Key Testing Details') in i]  #txt
    part_size5= [i for i in part_size if ('Testing Frequency') in i] #cat
    part_sizewant=part_size1+part_size2+part_size3+part_size4+part_size5
    #Chemical
    chemical = [col for col in df if col.startswith('Chemical Composition')]
    chemical1= [i for i in chemical if ('Max') in i] #num
    chemical2= [i for i in chemical if ('CoA') in i]  #cat
    chemical3= [i for i in chemical if ('Reference Method') in i]  #txt
    chemical4= [i for i in chemical if ('Key Testing Details') in i]  #txt
    chemical5= [i for i in chemical if ('Testing Frequency') in i] #cat
    chemicalwant=chemical1+chemical2+chemical3+chemical4+chemical5
    #Physical Tests
    phy = [col for col in df if col.startswith('Physical Tests')]
    phy1= [i for i in phy if ('Max') in i] #num
    phy2= [i for i in phy if ('CoA') in i]  #cat
    phy3= [i for i in phy if ('Reference Method') in i]  #txt
    phy4= [i for i in phy if ('Key Testing Details') in i]  #txt
    phy5= [i for i in phy if ('Testing Frequency') in i] #cat
    phywant=phy1+phy2+phy3+phy4+phy5
    #International Sieves
    intt = [col for col in df if col.startswith('International Sieves')]
    int1= [i for i in intt if ('Max') in i] #num
    int2= [i for i in intt if ('CoA') in i]  #cat
    int3= [i for i in intt if ('Reference Method') in i]  #txt
    int4= [i for i in intt if ('Key Testing Details') in i]  #txt
    int5= [i for i in intt if ('Testing Frequency') in i] #cat
    intwant=int1+int2+int3+int4+int5
    #Food Safety - 
    foodcat = [col for col in df if col.startswith('Food Safety -')]
    foodcat1= [i for i in foodcat if ('Maximum') in i] #num
    foodcat2= [i for i in foodcat if ('CoA') in i]  #cat
    foodcat3= [i for i in foodcat if ('Reference Method') in i]  #txt
    foodcat4= [i for i in foodcat if ('TXT') in i]  #txt
    foodcatwant=foodcat1+foodcat2+foodcat3+foodcat4
    #Additional Particle
    addpart_size = [col for col in df if col.startswith('Additional Particle')]
    addpart_size1= [i for i in addpart_size if ('Min') in i] #num
    #Status
    status =  [col for col in df if col.startswith('Status')]#cat
    #Standard Composition
    stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))]
    stdwant= [i for i in stdcomp if ('Value') in i]
    if spend =='chocolates':
        choco_cols['stdcomp']='Standard Composition'
        choco_cols['value']='Value, %'
        choco_cols['attr']=choco_cols['stdcomp']+' - '+choco_cols['Ingredient']+' - '+choco_cols['value']
        stdwant=list(set(stdwant).intersection(set(list(choco_cols.attr.unique()))))
    #Nutrient Composition
    nutcomp = [col for col in df if col.startswith('Nutrient Composition')]
    nutwant= [i for i in nutcomp if ('Value') in i] 
    #Shipping
    ship = [col for col in df if col.startswith('Shipping')]
    #Storage
    stor = [col for col in df if col.startswith('Storage')]
    #allergens
    allergens = [col for col in df if col.startswith('Allergens')]
    #desc
    descriptive = [col for col in df if col.startswith('Descriptive')]
    #Starch
    Starch = [col for col in df if col.startswith('Starch')]
    #State
    State = [col for col in df if col.startswith('State')]
    #HIS type
    HIS = [col for col in df if col.startswith('HIS type')]
    #Diet
    diet = [col for col in df if col.startswith('Diet')]
    #Qualitative
    qual = [col for col in df if col.startswith('Qualitative')]
    #Food
    food = [col for col in df if col.startswith('Food Safety Requirements')]
    #Total Solids
    solid = [col for col in df if col.startswith('Total Solids')]
    #Origin
    origin = [col for col in df if col.startswith('Origin')]
    #Color
    color = [col for col in df if col.startswith('Color')]
    #GM Assessment
    gm = [col for col in df if col.startswith('GM Assessment')]
    #Absence of Engineered
    abs_ = [col for col in df if col.startswith('Absence of Engineered')]
    #composition
    composition = [col for col in df if col.startswith('Composition')]
    #type
    type_ = [col for col in df if col.startswith('Type')]
    #subtype
    subtype= [col for col in df if col.startswith('Subtype')]
    # religious
    religious = [col for col in df if col.startswith('Religious')]
    # halal
    halal=[col for col in df if col.startswith('Halal')]
    #kosher
    kosher=[col for col in df if col.startswith('Kosher')]
    # microb
    microb=[col for col in df if col.startswith('Microbiological')]
    #approved supplier
    if spend=='colors':
        app_sup=[]
    else:
        app_sup=[col for col in df if col.startswith('Approved Supplier')]
    #gelatine
    g_origin = [col for col in df if col.startswith('Gelatine')]
    #source
    source = [col for col in df if col.startswith('Source')]
    #format
    format_=[col for col in df if col.startswith('Format')]
    #solid
    solids = [col for col in df if col.startswith('Solids')]
    #bloom
    bloom = [col for col in df if col.startswith('bloom')]
    #Taking required columns
    columns=bloom+solids+source+format_+g_origin+app_sup+microb+religious+halal+kosher+quantwant+part_sizewant+chemicalwant+phywant+intwant+foodcatwant+addpart_size1+stdwant+nutwant+ship+stor+allergens+descriptive+Starch+State+HIS+diet+qual+food+solid+origin+color+gm+abs_+composition+type_+subtype+status
    mand2=df[columns] #taking the categories of columns in a df

    df=pd.merge(mand1,mand2,left_index=True,right_index=True,how='left') #merge both stand alone and categories column data together

    df1=df.copy() #to get total columns list

    df=df.drop(['Buy Menu Card - NA - Menu Card Color',
    'Buy Menu Card - MEU - Menu Card Color',
    'Buy Menu Card - LA - Menu Card Color',
    'Buy Menu Card - AMEA - Menu Card Color'],axis=1)

    df = df.dropna(how='all', axis=1) #drop all blank columns
    # df.to_csv('model_check_reqd_before_drop.csv')

    Areas = ['H2 2021 - H1 2022 volume - NA, KG', #taking out region columns separately
            'H2 2021 - H1 2022 volume - MEU, KG',
            'H2 2021 - H1 2022 volume - LA, KG',
            'H2 2021 - H1 2022 volume - AMEA, KG',
            'H2 2021 - H1 2022 spend - NA, USD',
            'H2 2021 - H1 2022 spend - MEU, USD',
            'H2 2021 - H1 2022 spend - LA, USD',
            'H2 2021 - H1 2022 spend - AMEA, USD',
            'Buy Menu Card - NA - Menu Card Color',
            'Buy Menu Card - MEU - Menu Card Color',
            'Buy Menu Card - LA - Menu Card Color',
            'Buy Menu Card - AMEA - Menu Card Color']
    #MEU = [i for i in Areas if var in i]  #taking out region columns separately
    #df = df.drop(list(set(Areas).difference(set(MEU))),axis=1)  #taking out region columns separately
    #drop menu card details
    col=df.columns.tolist() #to get column list after dropping unwanted columns

    #Take out Numeric, Categorical and Text columns separately
    allergens = [col for col in df if col.startswith('Allergens')] #cat
    quant = [col for col in df if col.startswith('Quantitative')]
    quant1= [i for i in quant if ('Max') in i] #num
    quant2= [i for i in quant if ('CoA') in i]  #cat
    quant3= [i for i in quant if ('Reference Method') in i]  #txt
    quant4= [i for i in quant if ('Key Testing Details') in i]  #txt
    quant5= [i for i in quant if ('Testing Frequency') in i]  #cat
    part_size = [col for col in df if col.startswith('Particle')]
    part_size1= [i for i in part_size if ('Max') in i] #num
    part_size2= [i for i in part_size if ('CoA') in i]  #cat
    part_size3= [i for i in part_size if ('Reference Method') in i]  #txt
    part_size4= [i for i in part_size if ('Key Testing Details') in i]  #txt
    part_size5= [i for i in part_size if ('Testing Frequency') in i] #cat
    chemical = [col for col in df if col.startswith('Chemical Composition')]
    chemical1= [i for i in chemical if ('Max') in i] #num
    chemical2= [i for i in chemical if ('CoA') in i]  #cat
    chemical3= [i for i in chemical if ('Reference Method') in i]  #txt
    chemical4= [i for i in chemical if ('Key Testing Details') in i]  #txt
    chemical5= [i for i in chemical if ('Testing Frequency') in i] #cat
    phy = [col for col in df if col.startswith('Physical Tests')]
    phy1= [i for i in phy if ('Max') in i] #num
    phy2= [i for i in phy if ('CoA') in i]  #cat
    phy3= [i for i in phy if ('Reference Method') in i]  #txt
    phy4= [i for i in phy if ('Key Testing Details') in i]  #txt
    phy5= [i for i in phy if ('Testing Frequency') in i] #cat
    intt = [col for col in df if col.startswith('International Sieves')]
    int1= [i for i in intt if ('Max') in i] #num
    int2= [i for i in intt if ('CoA') in i]  #cat
    int3= [i for i in intt if ('Reference Method') in i]  #txt
    int4= [i for i in intt if ('Key Testing Details') in i]  #txt
    int5= [i for i in intt if ('Testing Frequency') in i] #cat
    foodcat = [col for col in df if col.startswith('Food Safety -')]
    foodcat1= [i for i in foodcat if ('Maximum') in i] #num
    foodcat2= [i for i in foodcat if ('CoA') in i]  #cat
    foodcat3= [i for i in foodcat if ('Reference Method') in i]  #txt
    foodcat4= [i for i in foodcat if ('TXT') in i]  #txt
    solids = [col for col in df if col.startswith('Solids')] #num
    stdcomp = [col for col in df if (col.startswith('Standard Composition')|col.startswith('Standard composition'))] 
    stdwant = [i for i in stdcomp if ('Value') in i] #num
    # taking selected attributes for chocolates
    if spend =='chocolates':
        choco_cols['stdcomp']='Standard Composition'
        choco_cols['value']='Value, %'
        choco_cols['attr']=choco_cols['stdcomp']+' - '+choco_cols['Ingredient']+' - '+choco_cols['value']
        stdwant=list(set(stdwant).intersection(set(list(choco_cols.attr.unique()))))
        stdwant=list(set(stdwant).intersection(set(list(cols.attributes.unique()))))    
    nutrients = [col for col in df if col.startswith('Nutrient')] 
    nut_want = [i for i in nutrients if ('Value') in i] #num
    Storage = [col for col in df if col.startswith('Storage')]
    Min_Remaining= [i for i in Storage if 'Min Remaining SL Upon Receipt' in i]  #num
    shelf_life= [i for i in Storage if 'Shelf Life' in i] #num
    store_txt = [i for i in Storage if 'TXT' in i] #txt
    storage_cndtn= [i for i in Storage if 'Storage Condition' in i] #cat
    descriptive = [col for col in df if col.startswith('Descriptive')]
    desc2= [i for i in descriptive if ('CoA') in i] #cat
    desc3= [i for i in descriptive if ('Testing Frequency') in i] #cat
    desc_txt = [i for i in descriptive if 'TXT' in i] #txt
    ship = [col for col in df if col.startswith('Shipping')] #cat
    Starch = [col for col in df if col.startswith('Starch')] #cat
    State = [col for col in df if col.startswith('State')] #cat
    HIS = [col for col in df if col.startswith('HIS type')] #cat
    diet = [col for col in df if col.startswith('Diet')]
    qual = [col for col in df if col.startswith('Qualitative')]
    addpart_size = [col for col in df if col.startswith('Additional Particle')]
    addpart_size1= [i for i in addpart_size if ('Min') in i] #num
    food = [col for col in df if col.startswith('Food Safety Requirements')] #cat
    solid = [col for col in df if col.startswith('Total Solids')] #num
    origin = [col for col in df if col.startswith('Origin')] #cat
    color = [col for col in df if col.startswith('Color')] #cat
    gm = [col for col in df if col.startswith('GM Assessment')] #cat
    abs_ = [col for col in df if col.startswith('Absence of Engineered')] #cat
    composition = [col for col in df if col.startswith('Composition')] #cat
    type_ = [col for col in df if col.startswith('Type')]#cat
    subtype= [col for col in df if col.startswith('Subtype')]#cat
    status =  [col for col in df if col.startswith('Status')]#cat
    religious = [col for col in df if col.startswith('Religious')]#cat
    # halal
    halal=[col for col in df if col.startswith('Halal')]#cat
    #kosher
    kosher=[col for col in df if col.startswith('Kosher')]#cat
    # microb
    microb=[col for col in df if col.startswith('Microbiological')]
    microb1= [i for i in microb if ('I/C') in i] #cat
    microb2= [i for i in microb if ('CoA') in i]  #cat
    microb3= [i for i in chemical if ('Reference Method') in i]  #cat
    microb4= [i for i in chemical if ('Samples') in i]  #num
    microb5= [i for i in chemical if ('Testing Frequency') in i] #cat
    microb6= [i for i in chemical if ('Marginally') in i]#num
    microb7= [i for i in chemical if ('Maximum') in i]#num
    microb8= [i for i in chemical if ('TXT') in i]#txt
    #approved supplier
    app_sup=[col for col in df if col.startswith('Approved Supplier')]#cat
    #gelatine
    g_origin = [col for col in df if col.startswith('Gelatine')]#cat
    #source
    source = [col for col in df if col.startswith('Source')]#cat
    #format
    format_=[col for col in df if col.startswith('Format')]#cat
    #bloom
    bloom=[col for col in df if col.startswith('bloom')]#num
    #NUMERIC
    num=microb4+microb6+microb7+quant1+part_size1+int1+chemical1+phy1+solids+stdwant+nut_want+Min_Remaining+shelf_life+addpart_size1+solid+foodcat1+bloom #numeric columns

    #TEXT
    txt=['Specification description','Classification']+microb8+quant3+quant4+int3+int4+phy3+phy4+chemical3+chemical4+store_txt+part_size3+part_size4+desc_txt+foodcat3+foodcat4 #text columns #CHECK WHETHER CLASSIFICATION COLUMN TEXT OR CAT
    # pd.DataFrame(txt).to_csv("txt.csv",index=False)

    #CATEGORICAL
    cat=source+format_+g_origin+app_sup+microb1+microb2+microb3+microb5+halal+kosher+religious+ship+storage_cndtn+quant2+quant5+int2+int5+phy2+phy5+chemical2+chemical5+part_size5+part_size2+desc2+desc3+allergens+Starch+State+HIS+diet+qual+food+foodcat2+origin+color+gm+abs_+composition+type_+subtype+status #categorical columns

    df_reqd=df.copy() #to keep a copy of the required data
    # df_reqd.to_csv('model_check_reqd.csv')
    #---------------------------------------------------------------------------------------------

    #1. NUMERIC COLUMNS

    #txt with one val to cat
    num = list(set(num).intersection(set(list(df.columns.unique()))))
    df_num_1 = df[num] 
    num_col_one_val = [] 
    for x in list(df_num_1.columns.unique()): 
        df_num_o= df_num_1[[x]].dropna() 
        df_num_o[x]= df_num_o[x].astype(str)
        if df_num_o[x].value_counts().reset_index().shape[0]<= to_cat_col:
            num_col_one_val = num_col_one_val + [x] 
    #Fill numerics with 0      
    for x in num:
        df[x]=df[x].fillna(0) 
    #numeric column cleaning for , and text words in it
    def storage(df,x): 
        df_hr  = df[df[x].astype(str).str.contains('hrs')]
        df_hr[x] = df_hr[x].astype(str).str.strip('hrs')
        df_hr[x] = df_hr[x].astype('int64')
        df_hr[x] = df_hr[x]/24
        df_day = df[df[x].astype(str).str.contains('Days')]
        df_day[x] = df_day[x].astype(str).str.strip('Days')
        df_day[x] =df_day[x].astype(str).str.replace(',','')
        df_day[x] = df_day[x].astype('int64')
        df_null = df[df[x].isnull()]
        df_null[x]=df_null[x].fillna(0)
        df_new = concat([df_hr,df_day])
        df_new = concat([df_new, df_null])
        return df_new[x]
    strg = [col for col in df[num] if col.startswith('Storage - ')]
    for x in strg:
        df[x]= storage(df,x)
    for x in num:
        df[x]=df[x].fillna(0) 

    #outlier treatment for num columns
    vol=['H2 2021 - H1 2022 volume - NA, KG',
            'H2 2021 - H1 2022 volume - MEU, KG',
            'H2 2021 - H1 2022 volume - LA, KG',
            'H2 2021 - H1 2022 volume - AMEA, KG']
    spn=['H2 2021 - H1 2022 spend - NA, USD',
    'H2 2021 - H1 2022 spend - MEU, USD',
    'H2 2021 - H1 2022 spend - LA, USD',
    'H2 2021 - H1 2022 spend - AMEA, USD']
    df['Volume treated']=df[vol].sum(axis=1)  
    df['Spend treated']=df[spn].sum(axis=1)
    df['Volume treated']=outlier(df,'Volume treated',1.5,'mean') 
    df['Spend treated']=outlier(df,'Spend treated',1.5,'mean')  

    #Correlation for numerical variables - removed

    #SCALING
    # =============================================================================
    #    scal=['Volume treated','Spend treated']
    #    num=num + scal
    # =============================================================================
    #Filter only numeric columns
    df_num=df[num]
    #clean if a num column has ,
    for x in quant1:
        df_num[x]=df_num[x].astype('str').str.replace(',','.').astype('float64') 
    df_num = df_num.drop(num_col_one_val,axis=1) 

    df_num_scaled=pd.DataFrame()    
    scaler = MinMaxScaler()
    model=scaler.fit(df_num)
    scaled_data=model.transform(df_num)
    scaled_num_data= pd.DataFrame(scaled_data, index=df_num.index, columns=df_num.columns)
    df_numeric=scaled_num_data.copy() 

    #Variance check and remove 0% variance from numeric columns
    def varianceSelection(data, THRESHOLD):
        sel = VarianceThreshold(threshold=(THRESHOLD * (1 - THRESHOLD)))
        sel.fit_transform(data)
        return data[[c for (s, c) in zip(sel.get_support(),data.columns.values) if s]]
    df_num = varianceSelection(df_numeric,1)
    attr_num=df_num.columns.tolist()
    # df_num.to_csv('df_num_old.csv')

    #-------------------------------------------------------------------------------------------

    #2. CATEGORICAL COLUMNS

    #Fill missing categorical with 'Not Applicable'
    #txt with one val to cat
    df_txt_1 = df[txt] 
    txt_col_one_val = [] 
    for x in list(df_txt_1.columns.unique()): 
        df_txt_o= df_txt_1[[x]].dropna() 
        if df_txt_o[x].value_counts().reset_index().shape[0] <= to_cat_col: 
            txt_col_one_val = txt_col_one_val + [x] 
    df_cat=pd.DataFrame() 
    cat = cat+txt_col_one_val +num_col_one_val

    for x in cat:
        df_cat[x]=df[x].fillna('Not Applicable')
        df_cat[x]= df_cat[x].astype(str)

    #Variance check
    #REMOVING 0% VARIANCE    
    #IF VALUE COUNTS=1 - REMOVE THOSE COLUMNS
    catcol=pd.DataFrame(df_cat.nunique(axis=0))
    catcol.reset_index(inplace=True)
    catcol.columns=['col','value']
    catcol1=catcol.copy()
    df_cat1=df_cat[catcol1['col']]
    catcol=catcol[catcol['value']>1]
    catcol=catcol[['col']]
    df_cat=df_cat[catcol['col']]

    #Correlation/Relationship for categorical variables usign chi square test
    #There exists a relationship between two variables if p value ≤ 0.05.
    #Has association but still we are keeping it as per business need

    #One-hot for Categorical columns
    df_cat_onehot = pd.get_dummies(df_cat, columns = df_cat.columns)
    cat_cols = list(df_cat.columns.unique())
    df_cat_onehot1 =  df_cat_onehot.copy()

    #Checking for 0% Variance
    df_cat_onehot = varianceSelection(df_cat_onehot,1)
    cat_list = list(set(list(df_cat_onehot.columns.unique().difference(set(list(df_cat_onehot1.columns.unique())))))) #check number of dropped columns in variance 
    #-----------------------------------------------------------------------------------------

    #3. TEXT COLUMNS

    #Fill text with 'Others'
    df_txt=pd.DataFrame()
    for x in txt:
        df_txt[x]=df[x].fillna('Others')
    #txt with one val to cat  
    #Classification column split - keeping here bcoz of many levels with slight value change


    df_txt,txtcol=levelone(spend=spend,df_txt=df_txt)
    #txt with one val to cat
    if 'Classification' in txt_col_one_val:
        txt_col_one_val.remove('Classification')
    df_txt = df_txt.drop(txt_col_one_val,axis=1)    
    txtcol.reset_index(inplace=True)
    txtcol.columns=['col','value']
    df_txt1=df_txt.copy()
    txtcol=txtcol[txtcol['value']==1]
    txtcol=txtcol[['col']]
    df_txt=df_txt.drop(list(txtcol['col'].unique()),axis=1)

    # df_txt.to_csv("df_txty.csv",index='False')

    attr_text=df_txt.columns.to_list()
    level_col=[col for col in df_txt if col.startswith('Level')]
    for i in level_col:
        attr_text.remove(i)
    attr_text.extend(['Classification'])

    #list of attributes used in the model
    attributes=attr_num + cat_cols + attr_text
    #converting the list of columns to csv

    attributes_df=pd.DataFrame({'Attributes_list':attributes})

    #    attributes_df = pd.read_excel("Attributes_list.xlsx")
    #    attributes_df = attributes_df.iloc[1:,]
    #    print(attributes_df)

    attributes_df.to_csv(spend+"_Attributes_list.csv",index=False)


    ##-------------------------------------------------------------------------------------------------------Aishwarya 26-05-23
    def varianceSelection(data, THRESHOLD):
        sel = VarianceThreshold(threshold=(THRESHOLD * (1 - THRESHOLD)))
        sel.fit_transform(data)
        return data[[c for (s, c) in zip(sel.get_support(),data.columns.values) if s]]

    # index = 'Specification Number'
    # df_num_all = pd.read_excel('df_num.xlsx',index_col=index)
    # df_txt_all = pd.read_excel('df_txt.xlsx',index_col=index)
    # df_cat_all = pd.read_excel('df_cat.xlsx',index_col=index)
    # df_reqd_all = pd.read_excel('df_reqd.xlsx',index_col=index)


    att_selected = pd.read_excel('Attributes_list_selected.xlsx')
    num = att_selected['Attributes_list'][att_selected['Type']=='num'].tolist()
    cat = att_selected['Attributes_list'][att_selected['Type']=='cat'].tolist()
    txt = att_selected['Attributes_list'][att_selected['Type']=='text'].tolist()
    all_attr = num + cat + txt 

    ###Filtering input based on attributes selected by user
    df_num = df_num.filter(num, axis=1)
    df_txt = df_txt.filter(txt, axis=1)
    df_cat = df_cat.filter(cat, axis=1)
    df_cat_onehot = pd.get_dummies(df_cat, columns = df_cat.columns)
    df_cat_onehot = varianceSelection(df_cat_onehot,1)
    cat_cols = list(df_cat.columns.unique())
    constant_columns = df_reqd.columns.to_list()[:12]
    all = constant_columns + all_attr
    df_reqd = df_reqd.filter(all, axis=1)
    
    ##-----------------------------------------------------------------------------------------------------Aishwarya 26-05-23
    #Text Pre-processing (tfidf and later bert if context)
    #Combining all text columns to form a description
    df_txt['text'] = df_txt[df_txt.columns].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1)
    #Remove Punctuations
    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text
    df_txt['Description_punc'] = df_txt['text'].apply(remove_punctuations)
    #Remove Numerics
    df_txt['Description_punc'] = df_txt['Description_punc'].str.replace('\d+', '')
    #Remove Single letters
    df_txt['Description_punc'] = df_txt['Description_punc'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
    #Tokenization and case
    def tokenize(text):
        tokens = re.split('\W+', text)
        return tokens
    df_txt['Description_token'] = df_txt['Description_punc'].apply(lambda x: tokenize(x.lower()))
    #Remove Stopwords
    stopword = nltk.corpus.stopwords.words('english')
    def remove_stopwords(tokenized_list):
        text = [word for word in tokenized_list if word not in stopword]
        return text
    df_txt['Description_nostop'] = df_txt['Description_token'].apply(lambda x: remove_stopwords(x))
    x = df_txt['Description_nostop'].astype(str)
    #tf-idf vectorization
    tfidf_vect_sample = TfidfVectorizer()
    X_tfidf_sample = tfidf_vect_sample.fit_transform(x)
    X_tfidf_df = pd.DataFrame(X_tfidf_sample.toarray())
    X_tfidf_df.columns = tfidf_vect_sample.get_feature_names()
    X_tfidf_df.index= df_txt.index
    #Variance Check after doing tfidf

   
    X_tfidf_df_var = varianceSelection(X_tfidf_df,1)
    #-------------------------------------------------------------------------------------

    #Now we have scaled for numeric columns, one-hot for categorical columns and tfidf values for text columns
    
    #Combine all dataframes (num + cat one-hot + text vectors)
   #to_drop=scal
    if var_set == 'num_cat_text': #variant (num+cat+text)
        df_final=pd.merge(df_num,df_cat_onehot,left_index=True,right_index=True)
        df_final=pd.merge(df_final,X_tfidf_df_var,left_index=True,right_index=True)
        cl=df_final.columns.tolist()
        df_final_proto=pd.merge(df_num,df_cat,left_index=True,right_index=True)
        df_final_proto=pd.merge(df_final_proto,X_tfidf_df_var,left_index=True,right_index=True)
        #df_final_proto=df_final_proto.drop(to_drop,axis=1)
        #df_final=df_final.drop(to_drop,axis=1)
        # df_final_proto.to_csv("df_final_proto_nct.csv")
        
      
    elif var_set =='num_text': #variant (num+text)
        df_final=pd.merge(df_num,X_tfidf_df_var,left_index=True,right_index=True)
        cl=df_final.columns.tolist()
        df_final_proto=pd.merge(df_num,X_tfidf_df_var,left_index=True,right_index=True)
        #df_final_proto=df_final_proto.drop(to_drop,axis=1)
        #df_final=df_final.drop(to_drop,axis=1)
        # df_final_proto.to_csv("df_final_proto_nt.csv")
        
     
    else : #variant (num+cat)
        df_final=pd.merge(df_num,df_cat_onehot,left_index=True,right_index=True)
        #print(df_final.shape)
        cl=df_final.columns.tolist()
        df_final_proto=pd.merge(df_num,df_cat,left_index=True,right_index=True)
       # df_final_proto=df_final_proto.drop(to_drop,axis=1)
       # df_final=df_final.drop(to_drop,axis=1)
        # df_final_proto.to_csv("df_final_proto_nc.csv")
       
    return df_final,df_final_proto,df_reqd,cat_cols
#---------------------------------------------------------------------------------------------------------------


# # The End



