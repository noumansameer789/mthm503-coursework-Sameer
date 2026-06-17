from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from course.utils import find_project_root


VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(df_scaled)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)

    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters['cluster'].astype(str)

    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _fit_dendrogram(df):
    tree = linkage(
        df,
        method='ward'
    )

    return tree


def _plot_dendrogram(df):
    fig = ff.create_dendrogram(
        df,
        linkagefun=lambda data: linkage(data, method='ward')
    )

    fig.update_layout(
        title_text='Interactive Hierarchical Clustering Dendrogram'
    )

    return fig


def _cutree(tree, height):
    clusters = fcluster(
        tree,
        t=height,
        criterion='distance'
    )

    clusters_df = pd.DataFrame({
        'cluster': clusters
    })

    return clusters_df


def _pca(df):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)

    df_pca = pd.DataFrame(
        components,
        columns=['PC1', 'PC2']
    )

    return df_pca


def _scatter_clusters(df):
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='cluster',
        title='PCA Scatter Plot Colored by Cluster Labels'
    )

    return fig
