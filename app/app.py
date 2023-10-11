# Recommended to run the app at 150% screen resolution
import sys
#global path1, path2
#path1 = "C:\\Users\\ERY3041\\Downloads\\Code_new\\Working_codes\\"
#path2 = "C:\\Users\\ERY3041\Downloads\\Code_new\\"
#sys.path.insert(0,path1)
from config import config_model
from Kitchen_global import Model_Global
from Global_DataPrep_Utils import PLM_preprocessing


# importing the required Libraries
import os
import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time


global attribflg

attribflg = 0

# worksheet_name = None

attlist = pd.read_excel('Attribute_list_to_drop_or_to_use.xlsx')
xl = pd.ExcelFile('Attribute_list_to_drop_or_to_use.xlsx')
ingredientlist = xl.sheet_names
add_ingr = ['CARRAGEENAN','PECTIN','H&G OTHERS','GUM ARABIC','XHANTAN GUM']
ingredientlist.extend(add_ingr)

checklist_items = ['a','b']
active_tab = {'font-size':'16px', 'font-family': 'MDLZ BITE TYPE','color':'#FFFFFF','background-color':'#4F2170','line-height':'0.05'}
tab_style = {'font-size':'14px', 'font-family': 'MDLZ BITE TYPE','color':'#4F2170','background-color':'#E7E6E6', 'align-items': 'center','line-height':'0.05'}
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True,prevent_initial_callbacks="initial_duplicate")

server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1(className='app-header',children='project kitchen'),width=20)),# header row 
        dbc.Row([dbc.Container([
            dbc.Row([
            dbc.Col([
                    dcc.Dropdown(ingredientlist,id='demo-dropdown',className="left_dropdown_style",placeholder="Select Ingredient")            
                        ],width=6),
            dbc.Col([
                    dbc.Button(children=["Train model"],id="train_model",
                        className="button_style",n_clicks=0,disabled= True)
                        ],       
                        width=6),           
                ],               
                className = "g-0"),
        ],style={'max-width':'25%'},fluid=True),# container left
        dbc.Container([
                dbc.Row([
                     dbc.Col([html.Div("Model Output:",className="right_header_style")],width={'size':'2','offset':'0'}),
                     dbc.Col([html.Div(id ='model-run', className="right_msg")], width = 2),
                     dbc.Col([
                      dcc.Dropdown(options=['nc','nt','nct'],id="model-type",className="dropdown_style",disabled=True)
                      ],width=2),
                     dbc.Col([
                      dcc.Dropdown(options=[],id="model-number",className="dropdown_style",disabled=True )
                     ],width=2),
                     dbc.Col([
                      dcc.Dropdown(options=[],id="cluster-number",value=["all"], multi=True,className="dropdown_style",disabled=True)
                     ],width=2), 
                      dbc.Col([
                       dcc.Dropdown(options=[],id="spec-number",value=["all"],multi=True,className="dropdown_style",disabled=True)
                    ],width=2)                   
                ])
        ],style={'max-width':'75%'},fluid=True)#container right
        ]),# end of 2nd row  
        dbc.Row(
            [
            dbc.Container([  
            dbc.Row(dbc.Col(
                html.Div('Numerical', id ='hdr1', className="row-header",style ={'display': 'none'})
                )),# header row       
            dbc.Row([
                dbc.Col([html.Div([dbc.Checklist(id='checklist1')])]),
                dbc.Col([html.Div(id='output1')]),
                dbc.Col([html.Div(id='selectedoutput1')]),               
                 ] ),
            dbc.Row(dbc.Col(html.Div('Categorical', id ='hdr2', className="row-header",style ={'display': 'none'}))),# header row 
            dbc.Row([
                dbc.Col([html.Div([dbc.Checklist(id='checklist2')])]),
                dbc.Col([html.Div(id='output2')]),
                dbc.Col([html.Div(id='selectedoutput2')]),               
                 ] ),
            dbc.Row(dbc.Col(html.Div('Text', id ='hdr3', className="row-header",style ={'display': 'none'}))),# header row 
            dbc.Row([
                dbc.Col([html.Div([dbc.Checklist(id='checklist3')])]),
                dbc.Col([html.Div(id='output3')]),
                dbc.Col([html.Div(id='selectedoutput3')]),               
                 ] ),    
             ],style={'max-width':'25%'},fluid=True),# container left1    
        dbc.Container([
                dbc.Row([           
                dbc.Col([   
                        dcc.Tabs(
                        id="Tabs",value='Model Comparison', 
                                                className="tab",
                                                children=[
                                                dcc.Tab(label='Model Comparison',value='Model Comparison',style=tab_style,selected_style=active_tab),
                                                dcc.Tab(label='Distribution View',value='Distribution View',style=tab_style,selected_style=active_tab),
                                                dcc.Tab(label='Model-Cluster List',value='Model-Cluster List',style=tab_style,selected_style=active_tab)
                                                ]),       
                        ],width=12),
                ]),
                dbc.Row([html.Div(id='tab-contents')]),
        ],style={'max-width':'75%'},fluid=True)#container right
        ]),# end of 3rd row       
    ], style={'max-width':'100%','max-height': '100%'},fluid=True)
# Callback for Ingredient Dropdown
@app.callback(
           #Output('output1','children'),
            [
            Output("demo-dropdown", "value"),
            Output('output1','children'),# checklist1
            Output('output2','children'),# checklist2
            Output('output3','children'),# checklist3
            Output('hdr1','style'),
            Output('hdr2','style'),
            Output('hdr3','style'), 
            Output('train_model','disabled',allow_duplicate= True), 
           # Output('model-run','children',allow_duplicate= True),                  
            ],            
            Input("demo-dropdown", "value"),
            prevent_intital_callback= True  #prevent_intial_Call          
             )
def update_output(value):
    # global worksheet_name
    #print('selected value:',value)
    if value is None :
        raise PreventUpdate #pass
    else:    
        global worksheet_name
        df = pd.DataFrame({'Selected': [value]})
        df.to_excel('selection_metadata.xlsx', sheet_name='selection_metadata',index=False)
        worksheet_name = value
        attlist = PLM_preprocessing()             
        attlistnum=attlist[attlist['Type'].str.contains('num')]
        checklist_items1 = attlistnum.iloc[:, 0]           
        attlistnum=attlist[attlist['Type'].str.contains('cat')]
        checklist_items2 = attlistnum.iloc[:, 0]            
        attlistnum=attlist[attlist['Type'].str.contains('text')]
        checklist_items3 = attlistnum.iloc[:, 0]    
        blank_model_type = []       
        return (               
            value,
            html.Div([                        
            dbc.Checklist(
                options= checklist_items1,                    
                id='checklist1',                                       
                className="left_content_style"
            )
        ]),
        html.Div([
            dbc.Checklist(
                options= checklist_items2,                   
                id='checklist2',                   
                className="left_content_style"
            )
        ]),
        html.Div([
            dbc.Checklist(
                options= checklist_items3,                   
                id='checklist3',                   
                className="left_content_style")
        ]),
        # blank_model_type,
        html.Div('Numerical', id ='hdr1', className="row-header",style ={'display': 'block'}),
        html.Div('Categorical', id ='hdr2', className="row-header",style ={'display': 'block'}),
        html.Div('Text', id ='hdr3', className="row-header",style ={'display': 'block'}),
        True,
        #html.Div('-', id ='model-run', className="right_msg"),
        )

# Callback for selected checklist output to excel
@app.callback([Output('checklist1','value'),
               Output('checklist2','value'), 
               Output('checklist3','value'),
               Output('train_model','disabled',allow_duplicate=True),
               Output('model-type','disabled',allow_duplicate=True),  
               Output('model-number','disabled',allow_duplicate=True), 
               Output('cluster-number','disabled',allow_duplicate=True), 
               Output('spec-number','disabled',allow_duplicate=True), 
                ],
            [Input('checklist1','value'),
            Input('checklist2','value'),
            Input('checklist3','value'),
            ],
            prevent_intital_callback = True 
            )

def output_selected_checklist(value1,value2,value3):
    #print('worksheet_name in output selected checklist:',worksheet_name)
    if value1 is None or value2 is None or value3 is None:
        #return [],[],[]  
        raise PreventUpdate
    else:
        df1 = pd.DataFrame(value1, columns=['Attributes_list'])
        df2 = pd.DataFrame(value2, columns=['Attributes_list'])
        df3 = pd.DataFrame(value3, columns=['Attributes_list'])        
        dfselected = df1.append([df2,df3])       
        df_all = pd.read_csv('Attributes_list_combined_all.csv')
        df_type = dfselected.merge(df_all,on='Attributes_list',how='left')        
        df_type.to_excel("Attributes_list_selected.xlsx",sheet_name='Attributes_list', index=False)        
        attribflg=1
        return value1,value2,value3,False,False,False,False,False

# Callback for Train Model Button
@app.callback(
            [
             Output('model-run','children'),
             Output('model-type','disabled'),  
             Output('model-number','disabled'), 
             Output('cluster-number','disabled'), 
             Output('spec-number','disabled'), 
            ],
            [Input('train_model','n_clicks')],
            prevent_intital_callback = True
            )

def train_model(n_clicks):      
    if n_clicks == 0 :  
        raise PreventUpdate
    else :
        start = time.time()    
        Model_Global()        
        end = time.time()
        mresults = pd.DataFrame({'Ingredient':[worksheet_name],'Time':[round(end-start,2)]})
        mresults.to_csv('modelruntimelog.csv', mode='a', index=False,header=False)    
        return html.Div('Model Run in '+str(round(end-start,2))+' sec', id ='model-run', className="right_msg"),False,False,False,False,
              


# populate model number drop down based on model type
@app.callback(
    Output('model-number','options'),  
    Input('model-type','value'),
    prevent_intital_callback = True
)
def populate_number_dropdown(model_type):    
    if model_type is None:
        #return []
        raise PreventUpdate
    else:
        df_modeltype = pd.read_csv(worksheet_name+'_final_global_results.csv')
        #print("model type in train model:")
        #print(df_modeltype)
        number_columns = [col for col in df_modeltype if model_type +"_" in col]
        return [{'label': i, 'value': i} for i in number_columns]

# populate Cluster drop down based on model type and model number
@app.callback(
    Output('cluster-number','options'),     
    Input('model-number','value'),
    prevent_intital_callback = True)

def populate_cluster_dropdown(value):
    #print('worksheet_name in populate_cluster_dropdown:',worksheet_name)
    if value is None:
        #return []
        raise PreventUpdate
    else:
        df_cluster = pd.read_csv(worksheet_name+'_final_global_results.csv')
        #for select all option
        all = df_cluster[value].unique()
        option=[x for x in all]
        option.append("all")
        return option

#-----------------------------------------------------------------------------------------------
# populate spec no drop down based on model type and model number and cluster
@app.callback(
    Output('spec-number','options',allow_duplicate=True),     
    [Input('model-number','value'),
    Input('cluster-number','value')],
    prevent_intital_callback = True
    )
def populate_spec_dropdown(value1,value2):    
    #print('worksheet_name in populate_spec_dropdown:',worksheet_name)
    if value1 is None or value2 is None:
       # return []
       raise PreventUpdate
    else:
        #print(path2+'\\'+worksheet_name+'_final_global_results.csv')
        newdf = pd.read_csv(worksheet_name+'_final_global_results.csv')
        df_spec = newdf.filter(regex='Specification Number|'+value1)

        if value2 == ["all"]:
            option = ["all"]
            return option
            
        else:
            df_specnew = df_spec[df_spec[value1].isin(value2)]
            all = df_specnew['Specification Number'].unique()
            option = [x for x in all]
            option.append("all")
            return option
#-------------------------------------------------------------------------------------------------
#Callback for Model Output Tabs
@app.callback(
            Output('tab-contents','children'),
            Input('Tabs','value'),
            Input('model-type','value'),
            Input('model-number','value'),
            Input('cluster-number','value'),
            Input('spec-number','value'),
            prevent_intital_callback = True
            )

def showdata(tab,modeltype,modelno, clusterno,specno):
    #print('worksheet_name in showdata:',worksheet_name)
    if tab is None or modeltype is None or modelno is None or clusterno is None and specno is None:
        #return []
        raise PreventUpdate
    else:
        #Read model result file
        #print('worksheet_name inside showdata',worksheet_name)
        df_model = pd.read_csv(worksheet_name+'_final_global_results.csv')
        df_model['Spend'] = df_model['Spend'].round(decimals = 2)
        df_model['Volume'] = df_model['Volume'].round(decimals = 2)
        model = modelno
        df_model[model] = df_model[model].round(0)

        #Model Comparison Tab: Table 1 (Will not be filtered by Cluster number or Spec Number)
        df1 = df_model.filter(regex='Specification Number|'+modeltype+'_')  
        table = dash_table.DataTable(id='table-container',
                                    columns=[{"name": i, "id": i} for i in df1.columns],
                                    data=df1.to_dict('records'),                                    
                                    style_header={'font-size':'11px','font-family': 'MDLZ BITE TYPE','color': '#FFFFFF', 'background-color': '#4F2170', 'text-align': 'center','whiteSpace': 'normal'},
                                    style_table={'overflowY': 'scroll','height': "460px"},
                                    style_data={'font-size':'11px','font-family': 'MDLZ BITE TYPE','color': '#4F2170', 'border': '1px solid gray', 'text-align': 'center','whiteSpace': 'normal'},                                     
                                    fixed_rows={ 'headers': True, 'data': 0 },
                                    fill_width=True,
                                    # page_size=12,
                                    
                                    )

        #Distribution View Tab : Pie Charts
        df_pie = df_model[['Specification Number','Volume','Spend',model]]
        if clusterno == ['all']:
            df_pie_fil2 = df_pie 
        elif clusterno is not None and (specno == ["all"]):
            #print("elif")
            #print(clusterno)
            #print(specno)
            df_pie_fil2 = df_pie[df_pie[model].isin(clusterno)]
            specno = ["all"]
        else:
            #print("else")
            #print(clusterno)
            #print(specno)
            df_pie_fil = df_pie[df_pie[model].isin(clusterno)]       #filter selected cluster number        
            df_pie_fil2 = df_pie_fil[df_pie_fil['Specification Number'].isin(specno)]   #filter selected spec number
        
        df_pie_grp = df_pie_fil2.groupby(model, axis=0,as_index=False).agg({'Specification Number': 'count','Volume':'sum','Spend':'sum'})
        # df_pie_grp['Volume'] = df_pie_grp['Volume'].map("#,###,.##K".format)
        # df_pie_grp['Volume'] = (df_pie_grp['Volume']/1000).apply(lambda x: '${:,.2f}K'.format(x))


        fig={}
        fig = make_subplots(rows=1,cols=3,specs=[[{'type':'pie'},{'type':'pie'},{'type':'pie'}]], 
                            subplot_titles=("Spec Distribution", "Volume Distribution", "Spend Distribution"))

        fig.add_trace(go.Pie(labels=df_pie_grp[model], values=df_pie_grp['Specification Number'],hole=0.3,name='',
                         hovertemplate = "Cluster:%{label} <br>Spec Count: %{value}"),row=1,col=1)
        fig.add_trace(go.Pie(labels=df_pie_grp[model], values=df_pie_grp['Volume'],hole=0.3,name='Volume Distribution',
                        hovertemplate = "Cluster:%{label} <br>Volume: %{value}"),row=1,col=2)
        fig.add_trace(go.Pie(labels=df_pie_grp[model], values=df_pie_grp['Spend'],hole=0.3,name='Spend Distribution',
                        hovertemplate = "Cluster:%{label} <br>Spend: %{value}"),row=1,col=3)
        fig.update_traces(hoverinfo='label+value', textinfo='value', textposition='inside')
        fig.update_layout(legend=dict(orientation="h",entrywidth=0,yanchor="bottom",y=-1.01,xanchor="right",x=1),
                            font=dict(family="MDLZ BITE TYPE", size=12, color="#666666"),
                            hovermode="x unified")
        fig.update_annotations(font=dict(family="MDLZ BITE TYPE", size=14, color="#4F2170"),yshift=70)
                # fig.update_layout(legend=dict(yanchor="top",y=0.99))


        #Model - Cluster List Tab : Table 2
        valuevarlist = [col for col in df_model.columns if col.startswith(modelno)]
        a = ['Specification Number','Specification description','Volume', 'Spend']
        b = [col for col in df_model if col.startswith('Level')]
        id_var = a+b
        df2 = pd.melt(df_model,id_vars=id_var, value_vars=valuevarlist, value_name ='Cluster Number')
        df2[['Model Type','Model Number']] = df2['variable'].str.split('_',expand=True)
        df3 = df2.drop(columns=['variable'])
        if clusterno == ['all']:
            df4 = df3
        elif clusterno is not None and (specno == ["all"]):
            df4 = df3[df3['Cluster Number'].isin(clusterno)]
            specno = ["all"]
        else:
            df3_fil = df3[df3['Cluster Number'].isin(clusterno)]  #filter selected cluster number
            df4 = df3_fil[df3_fil['Specification Number'].isin(specno)] 

        table2 = dash_table.DataTable(id='table-container2',
                                    columns=[{"name": i, "id": i} for i in df4.columns],
                                    data=df4.to_dict('records'),
                                    style_header={'font-size':'11px','font-family': 'MDLZ BITE TYPE','color': '#FFFFFF', 'background-color': '#4F2170', 'text-align': 'center'},
                                    style_table={'overflow': 'scroll','height': "460px"}, 
                                    style_cell={'min-width':'100px'},
                                    style_data={'font-size':'11px','font-family': 'MDLZ BITE TYPE','color': '#4F2170', 'border': '1px solid gray', 'text-align': 'center'},
                                    fixed_rows={ 'headers': True },
                                     )

        if tab =="Distribution View":
            return dbc.Row([
                        dbc.Col([dcc.Graph(figure=fig, className="tab-contents"
                        # style={'width': '128vh', 'height': '80vh','left':'0px'}
                        )],),
                            ], className="g-0")
        elif tab == "Model Comparison":
            return html.Div([table],className="tab-contents")
        elif tab == "Model-Cluster List":
            return html.Div([table2], className="tab-contents")

if __name__=='__main__':
    app.run_server(debug=True, port=8000)

