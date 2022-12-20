from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import plotnine as p9
import random
import numpy as np

app = Dash(__name__)

df = pd.read_csv('covid_data.csv')
df.drop('new_historic_deaths', inplace=True, axis=1)
df.drop('new_historic_cases', inplace=True, axis=1)

#df_vacc = pd.read_csv('covid_vaccine.csv')
df_pop = pd.read_csv('population.csv')

df["date_updated"] = pd.to_datetime(df["date_updated"], format="%Y-%m-%d")

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

df_final["new_cases_percapita"] = df_final['new_cases'] / df_final['POPESTIMATE2020']
df_final["tot_cases_percapita"] = df_final['tot_cases'] / df_final['POPESTIMATE2020']
df_final["new_deaths_percapita"] = df_final['new_deaths'] / df_final['POPESTIMATE2020']
df_final["tot_deaths_percapita"] = df_final['tot_deaths'] / df_final['POPESTIMATE2020']


start_dates = df[df["state"] == "AK"]["start_date"].tolist()

app.layout = html.Div([
    html.H1("The COVID-19 Situation in the U.S.", style={'text-align': 'center'}),
    html.Div(
            children=[
                dcc.Dropdown( id = 'dropdown',
                            options = [
                                {'label': 'New Cases per capita', 'value':'new_cases_percapita' },
                                {'label': 'Total Cases per capita', 'value':'tot_cases_percapita'},
                                {'label': 'New Deaths per capita', 'value':'new_deaths_percapita'}, 
                                {'label': 'Total Deaths per capita', 'value':'tot_deaths_percapita'}, 
                                ],
                            value="new_cases"),
            ], 
            style = {
            "display": "inline-block",
            "width": "33.3%",
            "text-align": "center"
            },
        ),
        
    dcc.Graph(id="graph"),
    dcc.Slider(0, 151, marks = None, value =0, id='my_slider'),
])



@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"),
    Input("my_slider", "value")
    )
def display_choropleth(dropdown_value, slider):
    print("slider", slider)
    my_date = start_dates[slider]
    filtered_data = df_final[df_final["start_date"]==my_date]
    fig = px.choropleth(filtered_data, locations = 'state',
                    locationmode = "USA-states", color = dropdown_value, scope="usa",
                    title= "Week of " + my_date)

    return fig


app.run_server(debug=True)