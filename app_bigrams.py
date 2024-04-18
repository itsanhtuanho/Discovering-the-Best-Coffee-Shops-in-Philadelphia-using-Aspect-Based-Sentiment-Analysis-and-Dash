from dash import Dash, html, dcc, callback, Output, Input
from dash.dash_table import DataTable
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Load dataframes
philly_reviews_asba = pd.read_csv('../Data/philly_reviews_asba.csv')
philly_reviews_asba = philly_reviews_asba.iloc[:, 1:]
top_10_coffee_shops = pd.read_csv('../Data/top_10_coffee_shops.csv')
top_10_coffee_shops = top_10_coffee_shops.iloc[:, 1:]
bi_grams_results = pd.read_csv('../Data/top_10_bigrams.csv')
bi_grams_results = bi_grams_results.iloc[:, 1:]

# Create Dash app
app = Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1(children='Philadelphia Coffee Shops', style={'textAlign': 'center'}),

    # Two columns layout: Left (List of Top 10) and Right (Map + Bar Chart)
    html.Div([
        # Left Column: List of Top 10 Coffee Shops
        html.Div([
            html.H2(children='Top 10 Coffee Shops'),
            DataTable(id='top-10-table', columns=[]),
        ], style={'width': '30%', 'float': 'left', 'margin-right': '5%'}),

        # Right Column: Map of all coffee shops and Bar chart of top bigrams
        html.Div([
            # Map of all coffee shops
            dcc.Graph(id='map-all-coffee-shops', style={'width': '100%'}),

            # Bar chart of top bigrams
            dcc.Graph(id='top-bigrams-chart', style={'width': '100%'}),
        ], style={'width': '65%', 'float': 'left'}),
    ]),
])

# Callback to update the map and bigrams chart based on selected coffee shop
@app.callback(
    Output('map-all-coffee-shops', 'figure'),
    Output('top-bigrams-chart', 'figure'),
    Input('top-10-table', 'active_cell'),
    prevent_initial_call=True,
)
def update_map_and_chart(active_cell):
    if active_cell is None:
        # If no cell is selected, return an empty map and bigram chart
        empty_map = px.scatter_mapbox(
            lat=[],
            lon=[],
            text=[],
            zoom=10,
            height=500,
        )

        empty_chart = px.bar(title='Top 10 Bigrams')
        return empty_map, empty_chart

    # Get the selected coffee shop
    selected_shop = top_10_coffee_shops.iloc[active_cell['row']]

    # Group by business name and calculate the average composite score
    avg_composite_scores = philly_reviews_asba.groupby('name')['composite_score'].mean().reset_index()

    # Update the map with the selected coffee shop highlighted
    fig_map = px.scatter_mapbox(
        avg_composite_scores,  # Use the grouped data
        lat="latitude",
        lon="longitude",
        text=["Name: {}<br>Avg Composite Score: {}".format(name, score) for name, score in zip(avg_composite_scores['name'], avg_composite_scores['composite_score'])],
        color_discrete_sequence=["blue"],
        zoom=10,
        height=500,
    )

    selected_lat = selected_shop['latitude']
    selected_lon = selected_shop['longitude']

    # Highlight/enlarge the marker
    fig_map.add_trace(
        go.Scattermapbox(
            lat=[selected_lat],
            lon=[selected_lon],
            text=["Name: {}<br>Avg Composite Score: {}".format(selected_shop['name'], selected_shop['composite_score'])],
            mode='markers',
            marker=dict(size=15, color='red'),
            showlegend=False,
        )
    )

    fig_map.update_layout(mapbox_style="open-street-map")

    # Update the bigrams chart with the selected coffee shop's bigrams
    fig_bigrams = px.bar(
        bi_grams_results[bi_grams_results['business_id'] == selected_shop['business_id']],
        x='bigram',
        y='frequency',
        title=f'Top 10 Bigrams for {selected_shop["name"]}',
    )
    return fig_map, fig_bigrams


# Callback to update the top 10 coffee shops table
@app.callback(
    Output('top-10-table', 'columns'),
    Output('top-10-table', 'data'),
    Input('map-all-coffee-shops', 'hoverData'),
    prevent_initial_call=True,
)
def update_top_10_table(hover_data):
    if hover_data is None or 'points' not in hover_data:
        # If no hover data or points, return an empty table
        empty_columns = [{'name': col, 'id': col} for col in ['Rank', 'Coffee Shop', 'Composite Score']]
        empty_data = []
        return empty_columns, empty_data

    # Get the name of the hovered coffee shop
    hovered_shop_name = hover_data['points'][0]['text']

    # Find the corresponding coffee shop in the top 10
    hovered_shop = top_10_coffee_shops[top_10_coffee_shops['name'] == hovered_shop_name]

    # Display top 10 coffee shops table
    top_10_table = pd.DataFrame({
        'Rank': range(1, 11),
        'Coffee Shop': top_10_coffee_shops['name'],
        'Composite Score': top_10_coffee_shops['composite_score'],
    })

    columns = [{'name': col, 'id': col} for col in top_10_table.columns]
    data = top_10_table.to_dict('records')

    return columns, data


# Run the app
if __name__ == '__main__':
    app.run_server(port=8051, debug=True)
