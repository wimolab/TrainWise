"""Utilities for data visualization using Plotly."""

from copy import deepcopy

import pandas as pd
import plotly.express as px


def plot_distribution(
    df: pd.DataFrame,
    column_name: str,
    pretty_column_name: str,
    height: int = 400,
    width: int = 600,
    explode: bool = False,
    top_k: bool = False,
) -> None:
    """Plot the distribution of values in a specified DataFrame column."""
    assert column_name in df.columns, "Column not found in DataFrame"

    if not pretty_column_name:
        pretty_column_name = column_name.replace("_", " ").title()

    if explode:
        # to handle semicolon-separated values (columns: _Frameworks, _Topics, Authors)
        df = deepcopy(df)
        df[column_name] = df[column_name].str.split(";")
        df[column_name] = df[column_name].apply(
            lambda x: [i.strip() for i in x] if isinstance(x, list) else x
        )
        df = df.explode(column_name)

    if top_k:
        # show only top_k most frequent values (e.g. most frequent topics)
        top_values = df[column_name].value_counts().nlargest(top_k).index
        df = df[df[column_name].isin(top_values)]

    fig = px.histogram(df, x=column_name, nbins=50, title=f"Distribution of {pretty_column_name}")
    fig.update_layout(
        xaxis_title=pretty_column_name,
        yaxis_title="Number of documents",
        height=height,
        width=width,
    )

    fig.show()
