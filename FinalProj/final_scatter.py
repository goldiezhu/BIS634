import dash
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import plotly.express as px
import random

app = dash.Dash()

df = pd.read_csv('covid_data.csv')
df_pop = pd.read_csv('population.csv')
df.drop('new_historic_deaths', inplace=True, axis=1)
df.drop('new_historic_cases', inplace=True, axis=1)
df["date_updated"] = pd.to_datetime(df["date_updated"], format="%Y-%m-%d")
df.dropna()
df_pop.dropna()

us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

df_pop["state"] = df_pop.NAME.map(us_state_to_abbrev)
df_pop2 = df_pop[['state','POPESTIMATE2020']]
df_pop2 = df_pop2.iloc[5: , :]
df_final = df_pop2.merge(df, on='state', how='left')
df_final["new_cases_per_capita"] = df_final['new_cases'] / df_final['POPESTIMATE2020']
df_final["tot_cases_per_capita"] = df_final['tot_cases'] / df_final['POPESTIMATE2020']
df_final["new_deaths_per_capita"] = df_final['new_deaths'] / df_final['POPESTIMATE2020']
df_final["tot_deaths_per_capita"] = df_final['tot_deaths'] / df_final['POPESTIMATE2020']
#print(df_final)

app.layout = html.Div( id = 'parent',
        children=[
            html.H1(id = 'H1', children = 'COVID-19 in the U.S. per capita', style = {'textAlign':'center', 'marginTop':40,'marginBottom':40}),
            html.Div([
            html.Label(['Select a k for k-means, X-axis variable, and Y-axis variable'], style={'font-weight': 'bold', "text-align": "right","offset":1}),
            ], 
            ),  

            html.Div(
                children=[
                    dcc.Dropdown(
                        id="k_value",
                        options=[
                            {'label': 'k=1', 'value':'1'},
                            {'label': 'k=2', 'value':'2'},
                            {'label': 'k=3', 'value':'3'}, 
                            {'label': 'k=4', 'value':'4'},
                            {'label': 'k=5', 'value':'5'}, 
                            {'label': 'k=6', 'value':'6'},
                            {'label': 'k=7', 'value':'7'}, 
                            {'label': 'k=8', 'value':'8'},
                            {'label': 'k=9', 'value':'9'}, 
                        ],
                        className="dropdown",
                        value="1"),
                ], style={
                        "display": "inline-block",
                        "width": "33.3%",
                        "text-align": "center"
                },
            
            ),
            html.Div(
                children=[
                    dcc.Dropdown( id = 'dropdown_x',
                                options = [
                                    {'label': 'New Cases', 'value':'new_cases_per_capita' },
                                    {'label': 'Total Cases', 'value':'tot_cases_per_capita'},
                                    {'label': 'New Deaths', 'value':'new_deaths_per_capita'}, 
                                    {'label': 'Total Deaths', 'value':'tot_deaths_per_capita'}, 
                                    ],
                                placeholder="Select X Axis Case Type"),
                                
                        ], 
                        style={
                        "display": "inline-block",
                        "width": "33.3%",
                        "text-align": "center"
                    },
            ),
            html.Div(
                children=[
                    dcc.Dropdown( id = 'dropdown_y',
                                options = [
                                    {'label': 'New Cases', 'value':'new_cases_per_capita' },
                                    {'label': 'Total Cases', 'value':'tot_cases_per_capita'},
                                    {'label': 'New Deaths', 'value':'new_deaths_per_capita'}, 
                                    {'label': 'Total Deaths', 'value':'tot_deaths_per_capita'}, 
                                    ],
                                placeholder="Select Y Axis Case Type"),
                                
                        ], 
                        style={
                        "display": "inline-block",
                        "width": "33.3%",
                        "text-align": "center"
                    },
            ),
            dcc.Graph(id = 'bar_plot'),

        ], 
)


@app.callback(Output('bar_plot', 'figure'),
              [
                Input("k_value", "value"),
                Input("dropdown_x", "value"),
                Input("dropdown_y", "value"),
              ]
)

def graph_update(k_value, dropdown_x, dropdown_y):
    print(k_value)
    k = int(k_value)  


    df_max_scaled_x = df_final.copy()
    column = dropdown_x
    df_max_scaled_x[column] = df_max_scaled_x[column] /df_max_scaled_x[column].abs().max()

    df_max_scaled_y = df_max_scaled_x.copy()
    column = dropdown_y
    df_max_scaled_y[column] = df_max_scaled_y[column] /df_max_scaled_y[column].abs().max()

    pts = [np.array(pt) for pt in zip(df_max_scaled_y[dropdown_x], df_max_scaled_y[dropdown_y])]
    centers = random.sample(pts, k)
    old_cluster_ids, cluster_ids = None, [] 
    while cluster_ids != old_cluster_ids:
        old_cluster_ids = list(cluster_ids)
        cluster_ids = []
        for pt in pts:
            min_cluster = -1
            min_dist = float('inf')
            for i, center in enumerate(centers):
                dist = np.linalg.norm(pt - center)
                if dist < min_dist:
                    min_cluster = i
                    min_dist = dist
            cluster_ids.append(min_cluster)
        df_final['cluster'] = cluster_ids
        cluster_pts = [[pt for pt, cluster in zip(pts, cluster_ids) if cluster == match]
            for match in range(k)]
        centers = [sum(pts)/len(pts) for pts in cluster_pts]

   # df_final["new_cases_percapita"] = df_final["new_cases_percapita"].astype(str)
    #df_final["tot_cases_percapita"] = df_final["tot_cases_percapita"].astype(str) #color_discrete_sequence=px.colors.qualitative.Antique, 
    fig = px.scatter(df_final, x=dropdown_x, y=dropdown_y, color = "cluster", hover_data=["state", "POPESTIMATE2020", "new_cases", "tot_cases"])
    

    fig.update_layout(title = 'COVID-19 New vs Total Cases per Capita',
                      xaxis_title = dropdown_x,
                      yaxis_title = dropdown_y
                      )
    
    return fig



if __name__ == '__main__': 
    app.run_server(debug=True)



