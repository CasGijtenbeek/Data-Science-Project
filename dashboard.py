import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Importing data
df = pd.read_excel('caffe_change.xlsx')


def all_categories():
    collectionOfCategories = []
    categories = df['Category'].unique()

    for category in categories:
        collectionOfCategories.append(
            {"label": category, "value": category}
        )

    return collectionOfCategories


app = dash.Dash(__name__)
app.title = "Dashboard"

# App layout
app.layout = html.Div([
    html.H1("Dashboard"),

    html.Div([
        html.Div([
            html.Div([
                html.Label("Category 1"),
                dcc.Dropdown(
                    id="category_dropdown",
                    options=all_categories(),
                    value="BEVERAGE"
                )
            ], style={"flex": "1", "padding": "10px"}),

            html.Div([
                html.Label("Category 2"),
                dcc.Dropdown(
                    id="category2_dropdown",
                    options=all_categories(),
                    value="FOOD"
                )
            ], style={"flex": "1", "padding": "10px"}),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center"})
    ]),
    
    dcc.Graph(id="bar_chart"),

    dcc.Graph(id="scatter_plot"),

    dcc.Graph(id="heatmap")
])

@app.callback(
    Output("bar_chart", "figure"),
    [Input("category_dropdown", "value")]
)
def update_graph(selected_category):
    filtered_df = df[df['Category'] == selected_category]

    # Compute sum per weekday, then divide by the number of unique dates for that weekday (avg per day)
    date_col = next((c for c in filtered_df.columns if 'date' in c.lower()), None)
    df_local = filtered_df.copy()
    if date_col is not None:
        df_local[date_col] = pd.to_datetime(df_local[date_col])
        days_per_weekday = df_local.groupby('Week day')[date_col].nunique()
    else:
        days_per_weekday = df_local.groupby('Week day').size()

    grouped = df_local.groupby('Week day', as_index=False)['Quantity'].sum()
    grouped['Quantity'] = grouped.apply(
        lambda r: r['Quantity'] / days_per_weekday.get(r['Week day'], 1),
        axis=1
    )

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped['Week day'] = pd.Categorical(grouped['Week day'], categories=weekday_order, ordered=True)
    grouped = grouped.sort_values('Week day')
    fig = px.bar(grouped, x='Week day', y='Quantity', title=f'Average sales for {selected_category} per week day')
    return fig


@app.callback(
    Output("scatter_plot", "figure"),
    [Input("category_dropdown", "value"),
     Input("category2_dropdown", "value")]
)
def update_scatter(selected_category, selected_category2):
    cats = [c for c in (selected_category, selected_category2) if c]
    if not cats:
        return px.scatter(title="No category selected")

    # Filter and aggregate by week and category
    filtered = df[df['Category'].isin(cats)].copy()
    grouped = filtered.groupby(['Week number', 'Category'], as_index=False)['Quantity'].sum()

    # Ensure every week × category combination exists (fill missing with 0) for continuous lines
    weeks = sorted(df['Week number'].unique())
    full_index = pd.MultiIndex.from_product([weeks, cats], names=['Week number', 'Category'])
    grouped = grouped.set_index(['Week number', 'Category']).reindex(full_index, fill_value=0).reset_index()

    # Plot both categories on the same figure
    fig = px.line(
        grouped,
        x='Week number',
        y='Quantity',
        color='Category',
        title=f'Weekly sales: {selected_category} vs {selected_category2}',
        markers=True
    )
    fig.update_xaxes(range=[1, 52], dtick=1, title='Week number')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis=dict(dtick=1))
    return fig


@app.callback(
    Output("heatmap", "figure"),
    [Input("category_dropdown", "value")]
)
def update_heatmap(selected_category):
    # Build a time-based pivot of quantities per category (prefer weekly if available)
    pivot = (
        df.groupby(['Week number', 'Category'], as_index=False)['Quantity']
        .sum()
        .pivot(index='Week number', columns='Category', values='Quantity')
        .fillna(0)
    )

    # Correlation between categories based on their time-series of sold quantities
    corr = pivot.corr()

    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        labels={'x': 'Category', 'y': 'Category', 'color': 'Correlation'},
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation between categories based on sold quantities'
    )
    fig.update_layout(height=600, margin={'l':100,'r':20,'t':50,'b':100})
    return fig



# Run server
if __name__ == "__main__":
    app.run_server(debug=True)
