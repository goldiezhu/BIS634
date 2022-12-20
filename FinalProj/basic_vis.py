import dash
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import plotly.express as px

app = dash.Dash()

df = pd.read_csv('covid_data.csv')
df_pop = pd.read_csv('population.csv')
df_vacc = pd.read_csv('covid_vaccine.csv')
df.drop('new_historic_deaths', inplace=True, axis=1)
df.drop('new_historic_cases', inplace=True, axis=1)
df["date_updated"] = pd.to_datetime(df["date_updated"], format="%Y-%m-%d")


start_dates = df[df["state"] == "AK"]["start_date"].tolist()

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


#df['state_fullname'] = df['state'].map(abbrev_to_us_state)

df_pop["state"] = df_pop.NAME.map(us_state_to_abbrev)
df_pop2 = df_pop[['state','POPESTIMATE2020']]
df_pop2 = df_pop2.iloc[5: , :]
df_final = df_pop2.merge(df, on='state', how='left')

df_vacc = df_vacc[['Date','Location','Administered_7_Day_Rolling_Average', 'Administered_Cumulative']]

df_vacc["Date"] = pd.to_datetime(df_vacc["Date"], format="%m/%d/%Y")
df_vacc["Date"] = df_vacc["Date"].dt.strftime('%Y-%m-%d')
df_vacc = df_vacc[df_vacc['Administered_7_Day_Rolling_Average'].notna()]
df_final1 = df_final.merge(df_vacc, left_on='date_updated', right_on='Date', how='left')
print(df_vacc)

app.layout = html.Div( id = 'parent',
        children=[
            html.H1(id = 'H1', children = 'COVID-19 Cases in the U.S.', style = {'textAlign':'center', 'marginTop':40,'marginBottom':40}),
            html.Div([
            html.Label(['Situation by State'], style={'font-weight': 'bold', "text-align": "right","offset":1}),
            ], 
            ),  

            html.Div(
                children=[

                    dcc.Dropdown(
                        id="state-filter",
                        options=[
                            {"label": state, "value": state}
                            for state in np.sort(df.state.unique())
                        ],
                        clearable=False,
                        className="dropdown",
                        placeholder="Select a State"),
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
                                    {'label': 'New Cases', 'value':'new_cases' },
                                    {'label': 'Total Cases', 'value':'tot_cases'},
                                    {'label': 'New Deaths', 'value':'new_deaths'}, 
                                    {'label': 'Total Deaths', 'value':'tot_deaths'}, 
                                    #{'label': 'Daily Vaccinations', 'value':'daily_vacc'}, 
                                    #{'label': 'Total number of Vaccinations', 'value':'tot_vacc'}, 
                                    ],
                                placeholder="Select a Case Type"),
                                
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
                                    {'label': 'New Cases', 'value':'new_cases' },
                                    {'label': 'Total Cases', 'value':'tot_cases'},
                                    {'label': 'New Deaths', 'value':'new_deaths'}, 
                                    {'label': 'Total Deaths', 'value':'tot_deaths'}, 
                                    #{'label': 'Daily Vaccinations', 'value':'daily_vacc'}, 
                                    #{'label': 'Total number of Vaccinations', 'value':'tot_vacc'}, 
                                    ],
                                placeholder="Select a Case Type"),
                                
                        ], 
                        style={
                        "display": "inline-block",
                        "width": "33.3%",
                        "text-align": "center"
                    },

            ),
            dcc.Slider(0, 151, marks = None, value =0, id='my_slider'),
            dcc.Graph(id = 'bar_plot'),

        ], 
)

    
@app.callback(Output('bar_plot', 'figure'),
              [
                Input("state-filter", "value"),
                Input('dropdown_x', 'value'),
                Input('dropdown_y', 'value'),
                Input("my_slider", "value"),
              ]
)

def graph_update(state, dropdown_x_value, dropdown_y_value, slider):
    print(dropdown_x_value)
    print(dropdown_y_value)
    mask = (
        (df.state == state)
    )
    filtered_data = df.loc[mask, :]
    print("slider", slider)
    my_date = start_dates[slider]
    filtered_data = df[df["start_date"]==my_date]

    fig = px.scatter(filtered_data, x = filtered_data['{}'.format(dropdown_x_value)], y = filtered_data['{}'.format(dropdown_y_value)])
    
    fig.update_layout(title = 'COVID-19 Cases over time',
                      xaxis_title = dropdown_x_value,
                      yaxis_title = dropdown_y_value
                      )
    
    return fig



if __name__ == '__main__': 
    app.run_server()


