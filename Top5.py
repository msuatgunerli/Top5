import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import os

st.write('''
# MSG
FBref data from the 2017-2018 season to 2020-2021 season across the top 5 european leagues.
''')

os.chdir('D:\Desktop\Programming\GitHub\msg\Portfolio\Soccer - FBref Top 5 App')
os.chdir('./data/Finalizer')

@st.cache(allow_output_mutation=True)
def excelreader():  # show_spinner=False
    df_raw = pd.read_excel('Top 5 Leagues Raw Player Stats All Time.xlsx')
    df_raw.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_raw.fillna(0, inplace=True)
    df_raw.set_index('Player_ID_Full', inplace = True)
    df_raw = df_raw.select_dtypes(include=[np.number])
    df_PAdj = pd.read_excel('Top 5 Leagues PAdj Player Stats All Time.xlsx')
    df_PAdj.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_PAdj.fillna(0, inplace=True)
    df_PAdj.set_index('Player_ID_Full', inplace = True)
    df_PAdj = df_PAdj.select_dtypes(include=[np.number])
    return df_raw, df_PAdj

df_raw, df_PAdj = excelreader()
os.chdir('../../')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Per 90 Stat Leaders
st.write('''
## Per 90 Stat Leaders
Displays the top 10 players for each metric filtered by minutes played.
''')

selection1 = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',
                         ['Raw', 'Possession Adjusted'], key='<selectbox1a>')

def statleaders(selection1):
    if selection1 == 'Raw':
        df = df_raw.copy()
    elif selection1 == 'Possession Adjusted':
        df = df_PAdj.copy()
    collist1 = list(df.columns)
    col = st.selectbox('Select a Metric (per 90)', sorted(collist1), key='<collist1a>')
    min_slider = st.slider('Minutes Played', min_value=400, max_value=int(df['Minutes Played'].max()), value=400, step=90, key="<slider1a>")
    df = df[df['Minutes Played'] >= min_slider]
    df['Player'] = df.index.str.split('_').str[0]
    df['Team'] = df.index.str.split('_').str[3]
    df['Season'] = df.index.str.split('_').str[6]
    leaders = df[['Player','Team', 'Season', col]]
    leaders.sort_values(by=col, ascending=False, inplace=True)
    return(leaders)

leaders = (statleaders(selection1))
leaders.reset_index(drop=True, inplace=True)
leaders.index = leaders.index + 1
st.dataframe(leaders.head(10), height=350)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Scatter Charts
st.write('''
## Scatter Charts
Displays the two selected metrics.
''')

selection2a = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',
                           ['Raw', 'Possession Adjusted'], key='<selectbox2a>')
if selection2a == 'Raw':
    dfscatter = df_raw.copy()
elif selection2a == 'Possession Adjusted':
    dfscatter = df_PAdj.copy()

selection2b = st.selectbox('Normal or Scaled Stats',
                           ['Normal', 'Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox2b>')
if selection2b == 'Normal':
    dfscatter = dfscatter
    min_slider2 = st.slider('Minutes Played', min_value=400, max_value=int(dfscatter['Minutes Played'].max()), value=400, key="<slider2a1>")
elif selection2b == 'Scale Across Top 5 Leagues':
    dfscatter = dfscatter
    scaler = MinMaxScaler()
    dfscatter = pd.DataFrame(scaler.fit_transform(dfscatter), index=dfscatter.index, columns=dfscatter.columns)
    min_slider2 = st.slider('Minutes Played', min_value=float(0), max_value=float(dfscatter['Minutes Played'].max()), value=float(0), key="<slider2a2>")
elif selection2b == 'Scale Leagues Independently':
    dfscatter = dfscatter
    dfscatter['Competition'] = dfscatter.index.str.split('_').str[4]
    leaguedict = {'Premier League' : 0, 'La Liga' : 0.25, 'Bundesliga' : 0.5, 'Serie A': 0.75, 'Ligue 1' : 1}
    dfscatter = dfscatter.replace({"Competition": leaguedict})
    dfscatter = dfscatter.groupby('Competition').transform(lambda x: minmax_scale(x.astype(float)))
    min_slider2 = st.slider('Minutes Played', min_value=float(0), max_value=float(dfscatter['Minutes Played'].max()), value=float(0), key="<slider2a3>")

dfscatter['Player'] = dfscatter.index.str.split('_').str[0]
dfscatter['Position'] = dfscatter.index.str.split('_').str[2]
dfscatter['Team'] = dfscatter.index.str.split('_').str[3]
dfscatter['Competition'] = dfscatter.index.str.split('_').str[4]
dfscatter['Season'] = dfscatter.index.str.split('_').str[6]

collist2 = list(dfscatter.columns)
xvar = st.selectbox('Select a Metric for the x-axis',
                    sorted(collist2), key='<collist2a>')
yvar = st.selectbox('Select a Metric for the y-axis',
                    sorted(collist2), key='<collist2b>')
#Pos = st.multiselect('Select a Position', sorted(list(set(dfscatter['Position']))), default=sorted(list(set(dfscatter['Position']))), key='<collist2c>')
container1 = st.container()

all1 = st.checkbox("Select/Deselect all", key="<checkbox2>")
if all1:
    Pos = container1.multiselect("Select Positions:", sorted(list(set(dfscatter['Position']))), sorted(list(set(dfscatter['Position']))))
else:
    Pos = container1.multiselect("Select Positions:",sorted(list(set(dfscatter['Position']))))

if len(Pos) > 0:
    def Scatter2D(xvar, yvar):
        if Pos == None:
            dfsel = dfscatter.copy()
            dfsel = dfsel[dfsel['Minutes Played'] >= min_slider2]
        elif Pos != None:
            dfsel = dfscatter.copy()
            dfsel = dfsel[dfsel['Position'].isin(Pos)]
            dfsel = dfsel[dfsel['Minutes Played'] >= min_slider2]
        
        container2 = st.container()
        all2 = st.checkbox("Select/Deselect all", key='<checkbox3>')
        if all2:
            Team = container2.multiselect("Select Teams:", sorted(
                list(set(dfsel['Team']))), sorted(list(set(dfsel['Team']))))
        else:
            Team = container2.multiselect("Select Teams:", sorted(list(set(dfsel['Team']))))
                                                        
        if Team == None:
            dfsel = dfsel.copy()
        elif Team != None:
            dfsel = dfsel.copy()
            dfsel = dfsel[dfsel['Team'].isin(Team)]

        words = ['Yd', '%', 'per 90', 'Team', 'Age', '90s Played', 'Competition']
        xvar = xvar
        if any(x in xvar for x in words):
            xtitle = xvar
        else:
            xtitle = xvar + ' per 90'
        yvar = yvar
        if any(y in yvar for y in words):
            ytitle = yvar
        else:
            ytitle = yvar + ' per 90'

        try:
            fig = px.scatter(dfsel, x=xvar, y=yvar, color="Competition",
                            color_discrete_sequence=px.colors.qualitative.Set1, hover_data=['Player', 'Team', 'Minutes Played', 'Season'])
            fig.update_layout(template='plotly_dark', title=dict(text=(ytitle + ' vs. ' + xtitle), font=dict(size=22), yanchor='top', xanchor='center', y=1, x=0.5),
                            xaxis_title=(xtitle), yaxis_title=(ytitle))
            fig.update_layout(autosize=False, width=1000, height=800, legend=dict(orientation="h", yanchor='top', xanchor='center', y=1.05, x=0.4))
            fig.layout.xaxis.tickformat = '.2f'
            fig.layout.yaxis.tickformat = '.2f'
            #fig.update_layout(xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
            fig.update_layout(dict(updatemenus=[dict(type = "buttons",direction = "left", buttons=list([dict(args=["visible", "legendonly"],
                label="Deselect All", method="restyle"), dict(args=["visible", True], label="Select All", method="restyle")]),
                pad={"r": 0, "t": 0},showactive=False,x=1,xanchor="right",y=1,yanchor="top"),]))

            #from pandas.api.types import is_numeric_dtype
            # if is_numeric_dtype(dfsel[xvar]):
            #     fig.add_vline(np.mean(dfsel[xvar]), opacity=1, line_color='black', line_width = 2.5)
            # if is_numeric_dtype(dfsel[yvar]):
            #     fig.add_hline(np.mean(dfsel[yvar]), opacity=1, line_color='black', line_width = 2.5)
            
            st.plotly_chart(fig)
            
        except:
            st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a team or multiple teams."}</h1>', unsafe_allow_html=True)

    Scatter2D(xvar, yvar)
elif len(Pos) == 0:
    st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a position or multiple positions."}</h1>', unsafe_allow_html=True)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Head-to-Head
st.write('''
## Head-to-Head
Compares two selected players.
''')

def HeadtoHead():

    selection3a = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',
                            ['Raw', 'Possession Adjusted'], key='<selectbox3a>')
    if selection3a == 'Raw':
        dfcomp = df_raw.copy()
    elif selection3a == 'Possession Adjusted':
        dfcomp = df_PAdj.copy()
    
    selection3b = st.selectbox('Normal or Scaled Stats',
                           ['Normal', 'Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox3b>')

    if selection3b == 'Normal':
        dfcomp = dfcomp
    elif selection3b == 'Scale Across Top 5 Leagues':
        dfcomp = dfcomp
        scaler = MinMaxScaler()
        dfcomp = pd.DataFrame(scaler.fit_transform(dfcomp), index=dfcomp.index, columns=dfcomp.columns)
    elif selection3b == 'Scale Leagues Independently':
        dfcomp = dfcomp
        dfcomp['Competition'] = dfcomp.index.str.split('_').str[4]
        leaguedict = {'Premier League' : 0, 'La Liga' : 0.25, 'Bundesliga' : 0.5, 'Serie A': 0.75, 'Ligue 1' : 1}
        dfcomp = dfcomp.replace({"Competition": leaguedict})
        dfcomp = dfcomp.groupby('Competition').transform(lambda x: minmax_scale(x.astype(float)))

    dfcomp['Selector'] = (dfcomp.index.str.split('_').str[0]) + ', ' + (dfcomp.index.str.split('_').str[6]) + ', ' + (dfcomp.index.str.split('_').str[3])
    Player1 = st.selectbox('Select Player 1', sorted(dfcomp.Selector), key='<dfcomp1>')
    Player2 = st.selectbox('Select Player 2', sorted(dfcomp.Selector), key='<dfcomp2>')

    if Player1 == Player2:
        st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select two different players."}</h1>', unsafe_allow_html=True)
    elif Player1 != Player2:
        dfPlayer1 = dfcomp[dfcomp['Selector'] == Player1]
        dfPlayer1 = dfPlayer1.iloc[:, :-1]
        dfPlayer2 = dfcomp[dfcomp['Selector'] == Player2]
        dfPlayer2 = dfPlayer2.iloc[:, :-1]

        dfPlayers = pd.concat([dfPlayer1.T, dfPlayer2.T], axis=1)
        dfPlayers = dfPlayers.T

        # Select template with st.selectionbox - Offensive, Passing, Scoring, etc... assign return to template then dfPlayers[template]
        metrics = st.multiselect("Select Metrics:", sorted(dfPlayers.columns))
        if len(metrics) == 0:
            st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a metric or multiple metrics."}</h1>', unsafe_allow_html=True)
        elif len(metrics) > 0:
            dfPlayers = dfPlayers[metrics]
            dfPlayers = dfPlayers.T

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            dfPlayers['Max'] = dfPlayers.max(axis=1)
            dfPlayers['Player1'] = dfPlayers.iloc[:, 0] / dfPlayers['Max']
            dfPlayers['Player2'] = dfPlayers.iloc[:, 1] / dfPlayers['Max']
            dfPlayers = dfPlayers.round(decimals=2)
            
            fig = make_subplots(rows=1, cols=2,specs=[[{}, {}]],horizontal_spacing=0.0);
            fig.update_layout(autosize=True, width=700)
            fig['layout'].update(margin=dict(l=0, r=0, b=0, t=100))
            xcoords = np.linspace(1,11,10)
            fig.add_trace(go.Bar(x=dfPlayers['Player1'], y=dfPlayers.index, marker={'color': 'dodgerblue'}, 
                                    orientation='h', showlegend=False, text=dfPlayers[dfPlayers.columns[0]], hoverinfo='none'), row=1, col=1)
            fig.update_xaxes(autorange="reversed", row=1, col=1)
            fig.add_trace(go.Bar(x=dfPlayers['Player2'], y=dfPlayers.index, marker={'color': 'crimson'}, 
                                    orientation='h', showlegend=False, text=dfPlayers[dfPlayers.columns[1]], hoverinfo='none'), row=1, col=2)
            fig.update_yaxes(tickfont=dict(size=20), row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=1, col=2)
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(template='plotly_dark')
            fig.update_traces(textposition='auto')
            fig.update_layout(uniformtext_minsize=28, uniformtext_mode='hide')
            st.plotly_chart(fig)
   
HeadtoHead()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------