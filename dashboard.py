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
        html.Label("Category"),
        dcc.Dropdown(
            id="category_dropdown",
            options=all_categories(),
            value="BEVERAGE"  # Default value
        )
    ], style={"width": "30%", "padding": "20px"}),

    dcc.Graph(id="bar_chart"),

    dcc.Graph(id="scatter_plot")
])

@app.callback(
    Output("bar_chart", "figure"),
    [Input("category_dropdown", "value")]
)
def update_graph(selected_category):
    filtered_df = df[df['Category'] == selected_category]
    grouped = filtered_df.groupby('Week day', as_index=False)['Quantity'].sum()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped['Week day'] = pd.Categorical(grouped['Week day'], categories=weekday_order, ordered=True)
    grouped = grouped.sort_values('Week day')
    fig = px.bar(grouped, x='Week day', y='Quantity', title=f'Sales for {selected_category} per week day')
    return fig


@app.callback(
    Output("scatter_plot", "figure"),
    [Input("category_dropdown", "value")]
)
def update_scatter(selected_category):
    filtered_df = df[df['Category'] == selected_category]
    grouped = filtered_df.groupby('Week number', as_index=False)['Quantity'].sum()

    fig = px.scatter(grouped, x='Week number', y='Quantity', title=f'Sales for {selected_category} per week')
    fig.update_traces(mode='lines+markers')
    return fig


# Run server
if __name__ == "__main__":
    app.run_server(debug=True)
