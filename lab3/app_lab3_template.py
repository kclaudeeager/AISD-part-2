# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

#Include . paths
import sys
sys.path.append('./')
sys.path.append('../')

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table
import dash_bootstrap_components as dbc

SERVER_URL = 'http://localhost:4000'
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd

import requests

app = Dash(__name__)

df = pd.read_csv("experiments/iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
             dcc.Loading(
                id="loading-data",
                type="circle",
                children=[html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
                ], style=col_style)
            ])
        ], style=col_style),

          dcc.Loading(
                id="loading-upload",
                type="circle",
                children=[
        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})
                ])

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                options=[{'label': col, 'value': col} for col in df.columns[1:]],
                value=df.columns[1],  # Default value is the second feature column name
                id='hist-column'
            )
        ], style=col_style),
        dcc.Graph(id='selected_hist')
    ], style=col_style | {'height': '400px', 'width': '400px'}),

    html.Div([

        html.Div([

            html.Div([
                html.Label(['X-Axis'], style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    options=[{'label': col, 'value': col} for col in df.columns[1:]],
                    value=df.columns[1],  # Default value is the second feature column name
                    id='xaxis-column'
                )
            ]),

            html.Div([
                html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    options=[{'label': col, 'value': col} for col in df.columns[1:]],
                    value=df.columns[2],  # Default value is the third feature column name
                    id='yaxis-column'
                )
            ])
        ], style=row_style | {'margin-left': '50px', 'margin-right': '50px'}),

        dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
    dash_table.DataTable(
        id='datatable',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=15
    )
], style={'margin-top': '20px'})

    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            dcc.Loading(
                id="loading-build",
                type="circle",
                children=[
                    html.Div([
                        html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                        html.Div(id='build-response', children='Click to build new model and train')
                    ], style=col_style | {'margin-top':'20px'}),
                ]
            ),

            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            dcc.Loading(
                id="loading-train",
                type="circle",
                children=[
                    html.Div([
                        html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"}),
                        html.Div(id='container-button-train', children='')
                    ], style=col_style | {'margin-top':'20px', 'width':'90px'})
                ]
            ),

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        dcc.Loading(
            id="score_loading",
            type="circle",
            children=[
                html.Div([
                    html.Div([
                        html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                        html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
                    ], style=col_style | {'margin-top':'20px'}),
                    html.Div([
                        html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                        html.Div(dcc.Input(id='model-for-score', type='text'))
                    ], style=col_style | {'margin-top':'20px'}),            
                    html.Div([
                        html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                        html.Div(id='score-response', children='Click to score')
                    ], style=col_style | {'margin-top':'20px'})
                ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

                html.Div(id='container-button-score', children='')
            ]
        )
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        dcc.Loading(
    id="test_loading",
    type="circle",
    children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])
    ])

    ])
])

# callbacks for Explore data tab


@app.callback(
# callback annotations go here
        Output(component_id='load-response', component_property='children'),
         [Input(component_id='load-val', component_property='n_clicks'),
     Input(component_id='file-for-train', component_property='value')]
)
def update_output_load(nclicks, input_filename):
    global df, df_csv

    if nclicks != None:
        # load local data given input filename
        df = pd.read_csv(input_filename,sep=',')
        df_csv = df.to_csv(index=False)
        return 'Load done.'
    else:
        return ''
    
@app.callback(
# callback annotations go here
       Output(component_id='upload-response', component_property='children'),
       Input(component_id='upload-val', component_property='n_clicks')
        
)
def update_output_upload(nclicks):
    global df_csv

    if nclicks != None:
        # invoke the upload API endpoint
        response = requests.post(f"{SERVER_URL}/iris/datasets", data={'train': df_csv})
        if response.status_code == 201:
            dataset_index = response.json()['index']
            log_message = f"Dataset uploaded successfully! Index: {dataset_index}\n"
            print(log_message)
            return dataset_index
        else:
            return 'Error uploading dataset'
    else:
        return ''
    
@app.callback(
    Output('tablecontainer', 'children'),
    [Input('datatable', 'selected_rows')]
)
def update_datatable(_):
    global df
    return dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns],
        page_size=15,
        id='datatable'
    )

@app.callback(
    Output('build-response', 'children'),
    Input('build-val', 'n_clicks'),
    Input('dataset-for-train', 'value')
)
def update_output_build(nclicks_build, dataset_index):
    if nclicks_build and dataset_index:
        try:
            # invoke new model endpoint to build and train model given data set ID
            response = requests.post(f"{SERVER_URL}/iris/model", data={'dataset': dataset_index}, timeout=120)
        except requests.exceptions.Timeout:
            return 'Request timed out'
        
        if response.status_code == 201:
            model_index = response.json()['model index']
            log_message = f"Model built successfully! Index: {model_index}\n"
            print(log_message)
            return model_index
        else:
            return 'Error building model'
    else:
        return ''

@app.callback(
    Output('container-button-train', 'children'),
    Input('train-val', 'n_clicks'),
    Input('model-for-train', 'value'),
    Input('dataset-for-train', 'value')
)


def update_output_train(nclicks_train, model_index, dataset_index):
    if nclicks_train and model_index and dataset_index:
        try:
            # invoke retrain model endpoint
            response = requests.put(f"{SERVER_URL}/iris/model/{model_index}?dataset={dataset_index}", timeout=180)
        except requests.exceptions.Timeout:
            return 'Request timed out'
        
        if response.status_code == 200:
            log_message = "Model retrained successfully!\n"
            print(log_message)
            history = response.json()['history']
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(history)
            # Convert the DataFrame to a DataTable
            table = dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'margin': '10px', 'padding': '10px'},
                style_cell={'margin': '2px', 'padding': '1px'}
            )
            return table
        else:
            return 'Error retraining model'
    else:
        return ''
    


@app.callback(
# callback annotations go here
        Output(component_id='score-response', component_property='children'),
        [Input(component_id='score-val', component_property='n_clicks'),
         Input(component_id='model-for-score', component_property='value'),
         Input(component_id='row-for-score', component_property='value')
        ]
)
def update_output_score(nclicks, model_index, row_text):
    if nclicks is not None:
        # convert row text to list
        row_text = row_text.split(',')
        features = list(map(float, row_text))
        # invoke score model endpoint
        response = requests.get(f"{SERVER_URL}/iris/model/{model_index}?fields={','.join(map(str, features))}", timeout=120)
        print(response.content)
        if response.status_code == 200:
            score_result = response.json()['score_result']
            # Create the score response message
            score_message = f"{score_result}"
            return score_message
        else:
            return 'Error scoring model'
    else:
        return ''
    
@app.callback(
    Output('container-button-test', 'children'),
    [Input('test-val', 'n_clicks')],
    [State('model-for-test', 'value'),
     State('dataset-for-test', 'value')]
)
def update_output_test(nclicks, model_index, dataset_index):
    if nclicks:
        # invoke test model endpoint
        response = requests.get(f"{SERVER_URL}/iris/model/{model_index}/test?dataset={dataset_index}", timeout=120)
        if response.status_code == 200:
            test_result = response.json()["test_result"]
            
            # Extract accuracy, loss, model ID, and dataset ID
            accuracy = test_result['accuracy']
            loss = test_result['loss']
            model_id = test_result['model_id']
            dataset_id = test_result['dataset_id']
            
            # Create a DataFrame for accuracy, loss, model ID, and dataset ID
            summary_df = pd.DataFrame({
                'Metric': ['Model ID', 'Dataset ID', 'Accuracy', 'Loss'],
                'Value': [model_id, dataset_id, accuracy, loss]
            })

            # Create a DataFrame for precision, recall, and confusion matrix
            precision_recall_df = pd.DataFrame({
                'Class': ['Class 1', 'Class 2', 'Class 3'],
                'Precision': test_result['precision'],
                'Recall': test_result['recall']
            })

            # Create a DataFrame for confusion matrix
            confusion_matrix_df = pd.DataFrame(test_result['confusion_matrix'], index=['Class 1', 'Class 2', 'Class 3'], columns=['Class 1', 'Class 2', 'Class 3'])
            
            # Convert DataFrames to DataTables
            summary_table = dash_table.DataTable(
                data=summary_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in summary_df.columns],
                style_table={'textAlign': 'center'},
                style_cell={'textAlign': 'center'},
                id='summary-table'
            )

            precision_recall_table = dash_table.DataTable(
                data=precision_recall_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in precision_recall_df.columns],
                style_table={'textAlign': 'center'},
                style_cell={'textAlign': 'center'},
                id='precision-recall-table'
            )

            confusion_matrix_table = dash_table.DataTable(
                data=confusion_matrix_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in confusion_matrix_df.columns],
                style_table={'textAlign': 'center'},
                style_cell={'textAlign': 'center'},
                id='confusion-matrix-table'
            )

            # Return the tables and the title
            return dbc.Container([
                dbc.Row([
                    dbc.Col(html.H2("Model Evaluation Results", style={'text-align': 'center'}), width=12)
                ]),
                dbc.Row([
                    dbc.Col(html.H3(f"Model ID: {model_id}, Dataset ID: {dataset_id}", style={'text-align': 'center'}), width=12)
                ]),
                dbc.Row([
                    dbc.Col(html.H4("Summary", style={'text-align': 'center'}), width=12),
                    dbc.Col(summary_table, width=12)
                ]),
                dbc.Row([
                    dbc.Col(html.H4("Precision and Recall", style={'text-align': 'center'}), width=12),
                    dbc.Col(precision_recall_table, width=12)
                ]),
                dbc.Row([
                    dbc.Col(html.H4("Confusion Matrix", style={'text-align': 'center'}), width=12),
                    dbc.Col(confusion_matrix_table, width=12)
                ])
            ], fluid=True, style={'width': '80%', 'margin': 'auto'})
        else:
            return 'Error testing model'
    else:
        return ''


    

    
@app.callback(
    Output(component_id='indicator-graphic', component_property='figure'),
    [Input(component_id='xaxis-column', component_property='value'),
     Input(component_id='yaxis-column', component_property='value')]
)
def update_graph(xaxis_column_name, yaxis_column_name):
    global df, df_csv
    # Check if the selected column names exist in the DataFrame
    if xaxis_column_name in df.columns and yaxis_column_name in df.columns:
        # Use the selected column names to create the scatter plot
        fig = px.scatter(x=df[xaxis_column_name], y=df[yaxis_column_name])
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title=yaxis_column_name)
        return fig
    else:
        # If the selected columns do not exist, return an empty figure
        return px.scatter()


@app.callback(
    Output('selected_hist', 'figure'),
    [Input('hist-column', 'value')]
)
def update_hist(hist_column_name):
    global df, df_csv
    if hist_column_name not in df.columns:
        # Return an empty histogram figure
        empty_df = pd.DataFrame({'empty_column': []})
        return px.histogram(empty_df, x='empty_column')
        
    # Create the histogram figure using the selected column
    fig = px.histogram(df, x=df[hist_column_name])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
