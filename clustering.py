import string
from tokenize import String
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import v_measure_score
from wordcloud import WordCloud
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.decomposition import PCA
from helpers.constants import ISRAELI_DATA_PATH, MACRO_NUTRIENTS, NUMBER_OF_FOOD_GROUPS, LABELS_DIC, \
    FOOD_STOP_WORDS, ISRAELI_LABELED_PATH


def clustering_algorithm(algorithm_name, df):
    if algorithm_name == "kmeans":
        return KMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, init="k-means++", n_init=25, random_state=0).fit(df)
    elif algorithm_name == "minibatch_kmeans":
        return MiniBatchKMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, random_state=0, batch_size=16).fit(df)
    elif algorithm_name == "hierarchical":
        return AgglomerativeClustering(n_clusters=NUMBER_OF_FOOD_GROUPS, affinity="manhattan", linkage="complete").fit(df)
    elif algorithm_name == "dbscan":
        return DBSCAN(eps=2.5, min_samples=5).fit(df)
    elif algorithm_name == "mean_shift":
        return MeanShift(bandwidth=2).fit(df)
    elif algorithm_name == "spectral":
        return SpectralClustering(n_clusters=NUMBER_OF_FOOD_GROUPS, affinity="nearest_neighbors",
                                  assign_labels='discretize', random_state=0).fit(df)
    return KMeans(n_clusters=NUMBER_OF_FOOD_GROUPS, random_state=0).fit(df)


def labels_names(df):
    labels_to_names = {}
    for key in LABELS_DIC:
        food_item = (df.loc[df['smlmitzrach'] == key])
        if food_item.iloc[0]['label'] in labels_to_names:
            labels_to_names[food_item.iloc[0]['label']] += "\n " + LABELS_DIC[key]
        else:
            labels_to_names[food_item.iloc[0]['label']] = LABELS_DIC[key]
    for i in range(NUMBER_OF_FOOD_GROUPS):
        if i not in labels_to_names:
            labels_to_names[i] = 'לא סווג'
        df.loc[df['label'] == i, 'labels_names'] = labels_to_names[i]
    df['labels_names'].fillna('לא סווג', inplace=True)
    return df


def create_labeled_dataframe(algorithm_name):
    df_original = pd.read_csv(ISRAELI_DATA_PATH)
    df_original = df_original.loc[:, MACRO_NUTRIENTS + ['smlmitzrach', 'shmmitzrach', 'alcohol', 'food_energy']]
    df_original = df_original.dropna(subset=MACRO_NUTRIENTS)
    df_original['alcohol'] = df_original['alcohol'].fillna(0)
    df = df_original.loc[:, MACRO_NUTRIENTS+['alcohol']]
    clusters = clustering_algorithm(algorithm_name, df)
    df['label'] = clusters.labels_
    df['smlmitzrach'] = df_original['smlmitzrach']
    df['name'] = df_original['shmmitzrach']
    df['food_energy'] = df_original['food_energy']
    df = labels_names(df)
    df.drop(['smlmitzrach'], axis=1).to_csv(f'csvs/csvs_algo/csv_{algorithm_name}.csv')


def get_matrix_food_group(unique_groups, labels_dict):
    food_group_names = labels_dict.values()
    values = {group_name: [] for group_name in food_group_names}
    for tuple_groups in unique_groups:
        for group in food_group_names:
            if group in tuple_groups[0]:
                values[group].append('✅')
            else:
                values[group].append('🕸️')
    return values


def create_table_food_group(df: pd.DataFrame, labels_dict: Dict, algo_name: String) -> pd.DataFrame:
    data_frame = df[['labels_names', 'label']]
    marged_columns = data_frame.apply(tuple, axis=1)
    unique_groups = sorted(marged_columns.unique(), key=lambda x: x[1])
    values = get_matrix_food_group(unique_groups, labels_dict)
    pd.DataFrame.from_dict(values).T.to_csv(f'csvs/csvs_algo/food_group_csv_{algo_name}.csv')


def food_groups_wordcloud(df):
    for label in range(NUMBER_OF_FOOD_GROUPS):
        text = ""
        food_group = df.loc[df['label'] == label]
        for food_item in food_group.name:
            food_parts = food_item.split()
            for part in food_parts:
                if part not in FOOD_STOP_WORDS:
                    text += f'{part} '

        bidi_text = get_display(text)
        wordcloud = WordCloud(font_path='C:\Windows\Fonts\courbd.ttf').generate(bidi_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        food_group_name = get_display(food_group.iloc[0]['labels_names'])
        plt.title(food_group_name)
        plt.axis("off")
        plt.savefig(f'results/results_{label}.png')


def addlabels(vals, adjustment):
    for i in range(len(vals)):
        plt.text(i+adjustment, vals[i], "{:.2f}".format(vals[i]), ha='center')


def clustering_plot(ari, fm, ami, v_measure):
    bar_width = 1 / 5
    plt.subplots(figsize=(10, 8))

    br1 = np.arange(len(ari))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    br4 = [x + bar_width for x in br3]

    plt.bar(br1, ari, color='royalblue', width=bar_width, edgecolor='grey', label='Adjusted rand index')
    addlabels(ari, 0)
    plt.bar(br2, fm, color='darkorange', width=bar_width, edgecolor='grey', label='Fowlkes mallows')
    addlabels(fm, 0.2)
    plt.bar(br3, ami, color='firebrick', width=bar_width, edgecolor='grey', label='Adjusted mutal info')
    addlabels(ami, 0.4)
    plt.bar(br4, v_measure, color='springgreen', width=bar_width, edgecolor='grey', label='V-measure')
    addlabels(v_measure, 0.6)

    plt.title("Comparing Clustering algorithms labels to ground truths\n using 4 different extrinsic measures",
              fontweight='bold', fontsize=15)
    plt.xlabel('Clustering algorithms', fontweight='bold', fontsize=15)
    plt.ylabel('Extrinsic measures scores', fontweight='bold', fontsize=15)
    plt.xticks([(r + bar_width*2) - (bar_width)/2 for r in range(len(ari))],
               ['K-means clustering', 'Agglomerative clustering', 'DBSCAN', 'Spectral clustering'])
    plt.ylim((0, 0.8))
    plt.legend()
    plt.show()


def evaluate_clustering():
    df = pd.read_excel(ISRAELI_LABELED_PATH)
    df['alcohol'] = df['alcohol'].fillna(0)
    clustering_input = df[MACRO_NUTRIENTS + ['alcohol']]
    true_labels = df['SubFoodGroupLabel'].values

    ari, fm, ami, v_measure = [], [], [], []
    clustering_algorithms = ["kmeans", "hierarchical", "dbscan", "spectral"]
    for algorithm in clustering_algorithms:
        clustering = clustering_algorithm(algorithm, clustering_input)
        ari.append(adjusted_rand_score(true_labels, clustering.labels_))
        fm.append(fowlkes_mallows_score(true_labels, clustering.labels_))
        ami.append(adjusted_mutual_info_score(true_labels, clustering.labels_))
        v_measure.append(v_measure_score(true_labels, clustering.labels_))
    clustering_plot(ari, fm, ami, v_measure)


def plot_radar():
    df = pd.read_csv(ISRAELI_DATA_PATH)
    food_item1 = (df.loc[df['smlmitzrach'] == 75111030])
    food_item2 = (df.loc[df['smlmitzrach'] == 51140618])
    food_item3 = (df.loc[df['smlmitzrach'] == 11111009])

    categories = MACRO_NUTRIENTS
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[food_item1.iloc[0]['protein'], food_item1.iloc[0]['total_fat'], food_item1.iloc[0]['carbohydrates']],
        theta=categories,
        fill='toself',
        name='Cucumber'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[food_item2.iloc[0]['protein'], food_item2.iloc[0]['total_fat'], food_item2.iloc[0]['carbohydrates']],
        theta=categories,
        fill='toself',
        name='Pita'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[food_item3.iloc[0]['protein'], food_item3.iloc[0]['total_fat'], food_item3.iloc[0]['carbohydrates']],
        theta=categories,
        fill='toself',
        name='Milk'
    ))

    fig.update_layout(
        title=go.layout.Title(text='Different food groups comparison'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
    pyo.plot(fig)


if __name__ == '__main__':
    df = create_labeled_dataframe("kmeans")
    food_groups_wordcloud(df)
    plot_clusters(df, "kmeans")
    table = create_table_food_group(df, LABELS_DIC)
    for column in table.columns:
        print(table[column].sum())