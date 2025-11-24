import pandas as pd
import plotly.express as px
from copy import deepcopy



def plot_distribution(df, column_name, pretty_column_name=None, height=400, width=600, explode=False, top_k=None):
    assert column_name in df.columns, "Column not found in DataFrame"

    if not pretty_column_name:
        pretty_column_name = column_name.replace('_', ' ').title()

    if explode:
        # to handle semicolon-separated values (columns: _Frameworks, _Topics, Authors)
        df = deepcopy(df)
        df[column_name] = df[column_name].str.split(';')
        df[column_name] = df[column_name].apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else x)
        df = df.explode(column_name)

    if top_k:
        # show only top_k most frequent values (e.g. most frequent topics)
        top_values = df[column_name].value_counts().nlargest(top_k).index
        df = df[df[column_name].isin(top_values)]

    fig = px.histogram(df, x=column_name, nbins=50, title=f'Distribution of {pretty_column_name}')
    fig.update_layout(xaxis_title=pretty_column_name, yaxis_title='Number of documents', height=height, width=width)
    
    fig.show()

