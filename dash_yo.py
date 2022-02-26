from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from helpers.constants import LABELS_DIC
import pandas as pd

df_kmeans = pd.read_csv('csvs/csvs_algo/csv_kmeans.csv')
df_minibatch_kmeans = pd.read_csv('csvs/csvs_algo/csv_minibatch_kmeans.csv')
df_hierarchical = pd.read_csv('csvs/csvs_algo/csv_hierarchical.csv')
df_dbscan = pd.read_csv('csvs/csvs_algo/csv_dbscan.csv')
df_spectral = pd.read_csv('csvs/csvs_algo/csv_spectral.csv')

print(df_dbscan)

df_kmeans_group = pd.read_csv('csvs/csvs_algo/food_group_csv_kmeans.csv')
df_minibatch_kmeans_group = pd.read_csv('csvs/csvs_algo/food_group_csv_minibatch_kmeans.csv')
df_hierarchical_group = pd.read_csv('csvs/csvs_algo/food_group_csv_hierarchical.csv')
df_dbscan_group = pd.read_csv('csvs/csvs_algo/food_group_csv_dbscan.csv')
df_spectral_group = pd.read_csv('csvs/csvs_algo/food_group_csv_spectral.csv')

print(df_kmeans_group)


FOOD_GROUP_OPTIONS = [{'label': x, 'value': x} for x in range(32)]
ALGORITHMS = ["kmeans", "minibatch_kmeans", "hierarchical", "dbscan", "spectral"]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("Interactive Visualization Of Food Groups Clustering",
            className='mb-2', style={'textAlign': 'center'}),

    dbc.Row([
        dbc.Label('Choose a clustering algorithm to see a different clustering for the 32 israeli food groups:'),
        dbc.Col([
            dcc.Dropdown(
                id='cluster-algo',
                options=[{'label': x, 'value': x} for x in ALGORITHMS],
                value='kmeans')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label('3d scatter plot of the 32 food group clusters:'),
            dcc.Graph(
                id='scatter-plot',
                style={'border-width': '5', 'width': '100%',
                       'height': '750px'})
        ], width=6),

        dbc.Col([
            dbc.Label('Table of Israeli food items, their macronutrients and the food group label from the'
                      ' clustering algorithm:'),
            html.Div(id='table-placeholder')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Label('Clear food groups from the plot and table:'),
        dbc.Col([
            dcc.Dropdown(
                id='food-group',
                multi=True,
                #clearable=False,
                #value=[int(dic['value']) for dic in FOOD_GROUP_OPTIONS])
                options=FOOD_GROUP_OPTIONS,
                value=[int(dic['value']) for dic in FOOD_GROUP_OPTIONS])
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label(''),
        ])
    ]),

    dbc.Row([
        dbc.Label('Mapping table of food groups to the clustering algorithm`s labels.'),
        dbc.Label('Each food group is mapped to one label and an ideal clustering would have a one-to-one mapping'
                  ' between the food groups and the labels.'),
        dbc.Col([
            html.Div(id='table-foodgroups')
        ], width=12)
    ])

])


# Create interactivity between components and graph
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('table-placeholder', 'children'),
    Output('table-foodgroups', 'children'),
    Input('cluster-algo', 'value'),
    Input('food-group', 'value')
)
def plot_data(cluster_value, food_group_value):
    # Choose proper dataframe
    if cluster_value == "hierarchical":
        df = df_hierarchical
        df_group = df_hierarchical_group
    elif cluster_value == "kmeans":
        df = df_kmeans
        df_group = df_kmeans_group
    elif cluster_value == "minibatch_kmeans":
        df = df_minibatch_kmeans
        df_group = df_minibatch_kmeans_group
    elif cluster_value == "hierarchical":
        df = df_hierarchical
        df_group = df_hierarchical_group
    elif cluster_value == "spectral":
        df = df_spectral
        df_group = df_spectral_group
    else:
        df = df_dbscan
        df_group = df_dbscan_group
    df = df.astype({"label": 'category'})
    df_group = df_group.reset_index()

    # filter data based on user selection
    df_filtered = df[df.label.isin(food_group_value)].rename(columns={'protein': 'proteins', 'total_fat': 'fats',
                                                                      'label': 'food group'})
    df_group_filtered = df_group[["index", *[str(i) for i in food_group_value]]]

    # build scatter plot
    scatter_3d = px.scatter_3d(df_filtered, x='proteins', y='fats', z='carbohydrates', size='food_energy',
                               color='food group', hover_data=['name', 'food group'],
                               color_discrete_sequence=px.colors.qualitative.Dark2+px.colors.qualitative.Light24,
                               category_orders={'food group': [i for i in range(32)]})

    # build DataTable
    mytable = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in
                 df_filtered.loc[:, ['food group', 'proteins', 'fats', 'carbohydrates', 'name']].columns],
        data=df_filtered.to_dict('records'),
        page_size=20,
    )

    group_table = dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in [str(col) for col in df_group_filtered.columns]],
        data=df_group_filtered.to_dict('records'),
    )

    return scatter_3d, mytable, group_table


if __name__ == '__main__':
    app.run_server(debug=False)
