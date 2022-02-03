import os
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px


LABELS_DIC = {
    56208068:	'1.1 דגנים מלאים',
    51140668:	'1.2 דברי מאפה (לחמים) מדגן מלא',
    56204930:	'1.3 דגנים לא מלאים',
    51101009:	'1.4 דברי מאפה (לחמים) מדגן לא מלא',
    71001040:	'1.5 ירקות עמילניים',
    58126148:	'1.6 דברי מאפה מלוחים',
    53510019:	'1.7 דברי מאפה מתוקים',
    57208099:	'1.8 מידגנים (Cereals)',
    92530229:	'10.1 משקאות ממותקים בסוכר',
    64105419:	'10.2 משקאות ממותקים בתחליפי סוכר',
    92205000:	'10.3 משקאות אורז/ שיבולת שועל',
    93504000:	'10.4.1 משקאות אלכוהוליים לא מתוקים',
    93401010:	'10.4.2 משקאות אלכוהוליים שמכילים פחמימות',
    75100780:	'2 קבוצת הירקות',
    63149010:	'3 קבוצת הפירות',
    41101219:	'4 קבוצת קטניות',
    42116000:	'5.1 קבוצת אגוזים וגרעינים',
    43103339:	'5.2 קבוצת שמנים צמחיים',
    83107000:	'5.3 שומנים מהחי',
    90000007:	'6.1 מוצרי חלב לא ממותקים',
    11411609:	'6.2 מוצרי חלב ממותקים בסוכר',
    11511269:	'6.3 קבוצת מוצרי חלב ממותקים בתחליפי סוכר- כל הדיאט',
    41420010:	'6.4 קבוצת תחליפי חלב על בסיס קטניות- מוצרי סויה',
    24102010: '8.1 קבוצת הבשר- עוף הודו',
    26115108:	'8.2 קבוצת הבשר- דגים',
    23200109:	'8.3 קבוצת הבשר- בשר בקר וצאן',
    50030109:	'8.4 קבוצת הבשר- תחליפי בשר מהצומח',
    91703070:	'9. קבוצת הסוכרים',
    31104000:	'ביצים 7',
    14210079:	'מוצרי חלב דל שומן',
    95312600:	'משקה אנרגיה',
    41811939:	'תחליפי בשר (לייט)'
}

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))

def get_matrix_food_group(unique_groups, labels_dict):
    food_group_names = labels_dict.values()
    values = {group_name: [] for group_name in food_group_names}
    for tuple_groups in unique_groups:
        for group in food_group_names:
            if group in tuple_groups[0]:
                values[group].append('True')
            else:
                values[group].append("")
    return values


def create_table_food_group(df, labels_dict):
    data_frame = df[['labels_names', 'label']]
    marged_columns = data_frame.apply(tuple, axis=1)
    unique_groups = sorted(marged_columns.unique(), key=lambda x: x[1])
    values = get_matrix_food_group(unique_groups, labels_dict)
    return pd.DataFrame.from_dict(values).T

df_dbscan = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'df_dbscan.csv'))
df_minibatch_kmeans = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'df_minibatch_kmeans.csv'))
df_hierarchical = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'df_hierarchical.csv'))
df_kmeans = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'df_kmeans.csv'))

df_kmeans_group = create_table_food_group(df_kmeans, LABELS_DIC)
df_minibatch_kmeans_group = create_table_food_group(df_minibatch_kmeans, LABELS_DIC)
df_hierarchical_group = create_table_food_group(df_hierarchical, LABELS_DIC)
df_dbscan_group = create_table_food_group(df_dbscan, LABELS_DIC)

FOOD_GROUP_OPTIONS = [{'label': x, 'value': x} for x in range(32)]
ALGORITHMS = ["kmeans", "minibatch_kmeans", "hierarchical", "dbscan"]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Container([
    html.H1("Interactive plotly with Dash To Visulize Labling Of Food Group By Various Algorithms",
            className='mb-2', style={'textAlign':'center'}),

    dbc.Row([
        dbc.Label('Choose clustering algorithm:'),
        dbc.Col([
            dcc.Dropdown(
                id='cluster-algo',
                options=[{'label': x, 'value': x} for x in ALGORITHMS],
                value='kmeans')
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label('3d scatter plot of 32 food groups:'),
            dcc.Graph(
                id='scatter-plot',
                style={'border-width': '5', 'width': '100%',
                       'height': '750px'})
        ], width=6),

        dbc.Col([
            dbc.Label('Table of food items, food groups and their macronutrients:'),
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
        dbc.Label('Table of food groups assigned to labels by the clustering algorithm:'),
        dbc.Col([
            html.Div(id='table-foodgroups')
        ], width=12)
    ]),

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
    else:
        df = df_dbscan
        df_group = df_dbscan_group
    df = df.astype({"label": 'category'})
    df_group = df_group.reset_index()

    # filter data based on user selection
    df_filtered = df[df.label.isin(food_group_value)]
    df_group_filtered = df_group[["index"] + food_group_value]

    # build scatter plot
    scatter_3d = px.scatter_3d(df_filtered, x='protein', y='total_fat', z='carbohydrates', size='food_energy',
                               color='label', hover_data=['name', 'label'])

    # build DataTable
    mytable = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in
                 df_filtered.loc[:, ['name', 'label', 'protein', 'total_fat', 'carbohydrates']].columns],
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
    app.run_server(debug=False, host='0.0.0.0', port=8080)