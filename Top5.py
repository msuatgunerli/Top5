import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from scipy.spatial import distance
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import os

st.set_page_config(layout= "wide")

st.write('''
# Top5 Web App @msuatgunerli
FBref data from the 2017-2018 season to 2020-2021 season across the top 5 european leagues.
''')
os.chdir('./data/Finalizer')

@st.cache(allow_output_mutation=True)
def excelreader():  # show_spinner=False
    df_raw = pd.read_csv('Top 5 Leagues Raw Player Stats All Time.csv')
    df_raw.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_raw.fillna(0, inplace=True)
    df_raw.set_index('Player_ID_Full', inplace = True)
    df_raw = df_raw.select_dtypes(include=[np.number])
    df_PAdj = pd.read_csv('Top 5 Leagues PAdj Player Stats All Time.csv')
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

    min_slider = st.slider('Minutes Played', min_value=400, max_value=int(df['Minutes Played'].max()), value=400, step=10, key="<slider1a>")
    df = df[df['Minutes Played'] >= min_slider]
    collist1 = list(df.columns)
    col = st.selectbox('Select a Metric (per 90)', sorted(collist1), key='<collist1a>')
    
    df['Player'] = df.index.str.split('_').str[0]
    df['Position'] = df.index.str.split('_').str[2]
    df['Team'] = df.index.str.split('_').str[3]
    df['Season'] = df.index.str.split('_').str[6]
    container0 = st.container()
    all0 = st.checkbox("Select/Deselect all", key="<checkbox0>")
    if all0:
        Pos = container0.multiselect("Select Positions:", sorted(list(set(df['Position']))), sorted(list(set(df['Position']))), key = '<multiselect0a')
    else:
        Pos = container0.multiselect("Select Positions:",sorted(list(set(df['Position']))), key = '<multiselect0b')
    if len(Pos) == 0:
        st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a position or multiple positions."}</h1>', unsafe_allow_html=True)
    elif len(Pos) > 0:
        df = df[df['Position'].isin(Pos)]
        leaders = df[['Player','Team', 'Season', col, 'Minutes Played']]
        leaders.sort_values(by=col, ascending=False, inplace=True)
        leaders.reset_index(drop=True, inplace=True)
        leaders.index = leaders.index + 1
        st.dataframe(leaders.head(10), height=350)

leaders = (statleaders(selection1))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Scatter Charts
# TODO: Add highlighting for a particular player as a selectbox 
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

selection2b = st.selectbox('Attribute Scaling',
                           ['No Scaling', 'Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox2b>')
if selection2b == 'No Scaling':
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

container1 = st.container()
all1 = st.checkbox("Select/Deselect all", key="<checkbox2>")
if all1:
    Pos = container1.multiselect("Select Positions:", sorted(list(set(dfscatter['Position']))), sorted(list(set(dfscatter['Position']))), key='<multiselect1a')
else:
    Pos = container1.multiselect("Select Positions:", sorted(list(set(dfscatter['Position']))), key='<multiselect1b')

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
            Team = container2.multiselect("Select Teams:", sorted(list(set(dfsel['Team']))), key='<multiselect2a')
                                                        
        if len(Team) == 0:
            st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a team or multiple teams."}</h1>', unsafe_allow_html=True)
        elif Team != None:
            dfsel = dfsel.copy()
            dfsel = dfsel[dfsel['Team'].isin(Team)]
            container3 = st.container()
            all3 = st.checkbox("Select/Deselect all", key="<checkbox21>")
            if all3:
                Season = container3.multiselect("Select Seasons:", sorted(list(set(dfscatter['Season']))), sorted(list(set(dfscatter['Season']))), key='<multiselect21a')
            else:
                Season = container3.multiselect("Select Seasons:", sorted(list(set(dfscatter['Season']))), key='<multiselect21b')
            if Season == None:
                dfsel = dfsel.copy()
            elif Season != None:
                dfsel = dfsel.copy()
                dfsel = dfsel[dfsel['Season'].isin(Season)]

            words = ['Yd', '%', 'per 90', 'Team', 'Age', '90s Played', 'Competition', 'Minutes Played']
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
                st.write('Displaying ' + str(len(dfsel)) + ' Players')
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
                fig.update_layout(width=1600, height=900)
                st.plotly_chart(fig)
            
            except:
                st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a season or multiple seasons."}</h1>', unsafe_allow_html=True)

    Scatter2D(xvar, yvar)
elif len(Pos) == 0:
    st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a position or multiple positions."}</h1>', unsafe_allow_html=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Player Percentile Ranks
st.write('''
## Player Percentile Ranks
Displays the best and worst attributes of a player.
''')

selection4 = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',
                         ['Raw', 'Possession Adjusted'], key='<selectbox4a>')

if selection4 == 'Raw':
    df4 = df_raw.copy()
elif selection4 == 'Possession Adjusted':
    df4 = df_PAdj.copy()

df4['Position'] = df4.index.str.split('_').str[2]

container4 = st.container()
all4 = st.checkbox("Select/Deselect all", key="<checkbox4>")
if all4:
    Pos = container4.multiselect("Select Positions:", sorted(list(set(df4['Position']))), sorted(list(set(df4['Position']))), key = '<multiselect4a')
else:
    Pos = container4.multiselect("Select Positions:",sorted(list(set(df4['Position']))), key = '<multiselect4b')

if len(Pos) == 0:
    st.markdown(f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a position or multiple positions."}</h1>', unsafe_allow_html=True)
    df4 = df4[df4['Position'].isin(Pos)]
elif len(Pos) > 0:
    df4 = df4[df4['Position'].isin(Pos)]

df4.drop(['Position'], axis = 1, inplace = True)

def PPR(df4, Pos):
    if len(Pos) == 0:
        st.write('')

    elif len(Pos) > 0: 
        selection4b = st.selectbox('Attribute Scaling',
                                ['Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox5b>')

        if selection4b == 'Scale Across Top 5 Leagues':
            df4 = df4
            scaler = MinMaxScaler()
            df4 = pd.DataFrame(scaler.fit_transform(df4), index=df4.index, columns=df4.columns)
        
        elif selection4b == 'Scale Leagues Independently':
            df4 = df4
            df4['Competition'] = df4.index.str.split('_').str[4]
            leaguedict = {'Premier League': 0, 'La Liga': 0.25,
                        'Bundesliga': 0.5, 'Serie A': 0.75, 'Ligue 1': 1}
            df4 = df4.replace({"Competition": leaguedict})
            df4 = df4.groupby('Competition').transform(lambda x: minmax_scale(x.astype(float)))

        df4['Player'] = df4.index.str.split('_').str[0]
        df4['Position'] = df4.index.str.split('_').str[2]
        df4['Team'] = df4.index.str.split('_').str[3]
        df4['Competition'] = df4.index.str.split('_').str[4]
        df4['Season'] = df4.index.str.split('_').str[6]

        df4['Selector'] = (df4.index.str.split('_').str[0]) + ', ' + (df4.index.str.split('_').str[2])+ ', ' + (df4.index.str.split('_').str[6]) + ', ' + (df4.index.str.split('_').str[3])
        Player1 = st.selectbox('Select Player 1', sorted(df4.Selector), key='<df4selector1>')
        player_stats = df4[df4['Selector'] == Player1]
        player_stats = player_stats.round(decimals=3)
        df_player1 = pd.DataFrame(player_stats.iloc[0])
        df_player1.columns = df_player1.iloc[-6]
        df_player1 = df_player1[:-6]
        df_player1.sort_values(by=df_player1.columns[0], ascending=False, inplace = True)
        df_player1.rename(columns={df_player1.columns[0]: "Player Percentile Rank"}, inplace=True)

        len_options = [5, 10, 15, 20]
        len_option = st.selectbox('Select the number of attributes displayed', len_options, key='<df4sel1>')

        top_att = 100*df_player1.head(len_option)
        bottom_att = 100*df_player1.tail(len_option).iloc[::-1]
        nonzero_bottom_att = 100 * df_player1.loc[~(df_player1 == 0).all(axis=1)].tail(len_option).iloc[::-1]
        return Player1, top_att, bottom_att, nonzero_bottom_att
    
try: 
    Player1, PPR1, PPR2, PPR3 = PPR(df4, Pos) 
    df4_options = ['Best', 'Worst', 'Worst Non-zero']
    df4_option = st.selectbox('Select Attributes to view', df4_options, key='<df4sel2>')

    if df4_option == 'Best':
        PPR1.reset_index(inplace = True)
        PPR1.index = PPR1.index + 1
        PPR1.rename(columns={PPR1.columns[0]: str(Player1)}, inplace=True)
        st.dataframe(PPR1, height = 1000)
    elif df4_option == 'Worst':
        PPR2.reset_index(inplace=True)
        PPR2.index = PPR2.index + 1
        PPR2.rename(columns={PPR2.columns[0]: str(Player1)}, inplace=True)
        st.dataframe(PPR2, height=1000)
    elif df4_option == 'Worst Non-zero':
        PPR3.reset_index(inplace=True)
        PPR3.index = PPR3.index + 1
        PPR3.rename(columns={PPR3.columns[0]: str(Player1)}, inplace=True)
        st.dataframe(PPR3, height=1000)
except:
    st.write('')
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

    selection3b = st.selectbox('Attribute Scaling',
                               ['No Scaling', 'Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox3b>')

    if selection3b == 'No Scaling':
        dfcomp = dfcomp
    elif selection3b == 'Scale Across Top 5 Leagues':
        dfcomp = dfcomp
        scaler = MinMaxScaler()
        dfcomp = pd.DataFrame(scaler.fit_transform(
            dfcomp), index=dfcomp.index, columns=dfcomp.columns)
    elif selection3b == 'Scale Leagues Independently':
        dfcomp = dfcomp
        dfcomp['Competition'] = dfcomp.index.str.split('_').str[4]
        leaguedict = {'Premier League': 0, 'La Liga': 0.25,
                      'Bundesliga': 0.5, 'Serie A': 0.75, 'Ligue 1': 1}
        dfcomp = dfcomp.replace({"Competition": leaguedict})
        dfcomp = dfcomp.groupby('Competition').transform(
            lambda x: minmax_scale(x.astype(float)))

    dfcomp['Selector'] = (dfcomp.index.str.split('_').str[0]) + ', ' + (dfcomp.index.str.split('_').str[2]) + \
        ', ' + (dfcomp.index.str.split('_').str[6]) + \
        ', ' + (dfcomp.index.str.split('_').str[3])
    Player1 = st.selectbox('Select Player 1', sorted(
        dfcomp.Selector), key='<dfcomp1>')
    Player2 = st.selectbox('Select Player 2', sorted(
        dfcomp.Selector), key='<dfcomp2>')

    if Player1 == Player2:
        st.markdown(
            f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select two different players."}</h1>', unsafe_allow_html=True)
    elif Player1 != Player2:
        dfPlayer1 = dfcomp[dfcomp['Selector'] == Player1]
        dfPlayer1 = dfPlayer1.iloc[:, :-1]
        dfPlayer2 = dfcomp[dfcomp['Selector'] == Player2]
        dfPlayer2 = dfPlayer2.iloc[:, :-1]

        dfPlayers = pd.concat([dfPlayer1.T, dfPlayer2.T], axis=1)
        dfPlayers = dfPlayers.T

        # Select template with st.selectionbox - Offensive, Passing, Scoring, etc... assign return to template then dfPlayers[template]
        metrics = st.multiselect("Select Metrics:", sorted(
            dfPlayers.columns), key='<multiselect3a')
        if len(metrics) == 0:
            st.markdown(
                f'<h1 style="color:#ff5454;font-size:18px;">{"Error: Please select a metric or multiple metrics."}</h1>', unsafe_allow_html=True)
        elif len(metrics) > 0:
            dfPlayers = dfPlayers[metrics]
            dfPlayers = dfPlayers.T
            dfPlayers_actual = dfPlayers.copy()
            dfPlayers_actual = dfPlayers_actual.round(decimals = 2)

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            for row in dfPlayers.iterrows():
                if ((row[1][0] < 0) and (row[1][1] < 0)):
                    rowwisemax = max(abs(row[1]))
                    temp0 = abs(row[1][0])
                    temp1 = abs(row[1][1])
                    row[1][0] = temp1 / rowwisemax
                    row[1][1] = temp0 / rowwisemax
                else:
                    rowwisemax = max(abs(row[1]))
                    temp0 = abs(row[1][0])
                    temp1 = abs(row[1][1])
                    row[1][0] = temp0/ rowwisemax
                    row[1][1] = temp1 / rowwisemax
            
            dfPlayers = dfPlayers.round(decimals=2)
            fig = make_subplots(rows=1, cols=2, specs=[
                                [{}, {}]], horizontal_spacing=0.0, subplot_titles=(f'<b>{Player1}<b>', f'<b>{Player2}<b>'))
            fig.update_annotations(font_size=32)
            fig.update_layout(autosize=True, height = 950, width=1600)
            fig['layout'].update(margin=dict(l=0, r=0, b=0, t=100))
            xcoords = np.linspace(1, 11, 10)
            fig.add_trace(go.Bar(x=dfPlayers[dfPlayer1.index[0]], y=dfPlayers.index, marker={'color': 'dodgerblue'},
                                 orientation='h', showlegend=False, text=dfPlayers_actual[dfPlayers_actual.columns[0]], hoverinfo='none'), row=1, col=1)
            fig.update_xaxes(autorange="reversed", row=1, col=1)
            fig.add_trace(go.Bar(x=dfPlayers[dfPlayer2.index[0]], y=dfPlayers.index, marker={'color': 'crimson'},
                                 orientation='h', showlegend=False, text=dfPlayers_actual[dfPlayers_actual.columns[1]], hoverinfo='none'), row=1, col=2)
            fig.update_yaxes(tickfont=dict(size=20), row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=1, col=2)
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(template='plotly_dark')
            fig.update_traces(textposition='auto')
            fig.update_layout(uniformtext_minsize=28, uniformtext_mode='hide')
            fig.update_layout(xaxis1=dict(range=[0, 1.01]), xaxis2=dict(range=[0, 1.01]))
            st.plotly_chart(fig)

HeadtoHead()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PCA
st.write('''
## PCA
Lorem ipsum.
''')

selection6a = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',
                           ['Raw', 'Possession Adjusted'], key='<selectbox6a>')
if selection6a == 'Raw':
    dfcomp = df_raw.copy()
elif selection6a == 'Possession Adjusted':
    dfcomp = df_PAdj.copy()


def PCADep():
    dfscaled = dfcomp.copy()
    dfscaled = dfscaled.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    dfscaled1 = pd.DataFrame(scaler.fit_transform(
        dfscaled), index=dfscaled.index, columns=dfscaled.columns)

    pca = PCA(n_components=2, svd_solver='full')
    PCvalues = pd.DataFrame(pca.fit_transform(
        dfscaled1), columns=['PC1', 'PC2'])
    PCvalues.index = dfscaled1.index
    PCvalues['Player'] = PCvalues.index.str.split('_').str[0]
    PCvalues['Position'] = PCvalues.index.str.split('_').str[2]
    PCvalues['Team'] = PCvalues.index.str.split('_').str[3]
    PCvalues['Competition'] = PCvalues.index.str.split('_').str[4]
    PCvalues['Season'] = PCvalues.index.str.split('_').str[6]
    st.write("Cumulative Proportion of Variance Explained: " + str(100 * round(np.cumsum(pca.explained_variance_ratio_)[1],3)) + "%")
    st.write('Displaying ' + str(len(PCvalues)) + ' Players')
    fig = px.scatter(PCvalues, x='PC1', y='PC2', color="Position",
                     color_discrete_sequence=px.colors.qualitative.Set1, hover_data=['Player', 'Team', 'Season'])
    fig.update_layout(template='plotly_dark', title=dict(text=('PC2' + ' vs. ' + 'PC1'), font=dict(size=22), yanchor='top', xanchor='center', y=1, x=0.5),
                      xaxis_title=('PC1'), yaxis_title=('PC2'))
    fig.update_layout(autosize=False, width=1000, height=800, legend=dict(
        orientation="h", yanchor='top', xanchor='center', y=1.05, x=0.5))
    fig.layout.xaxis.tickformat = '.2f'
    fig.layout.yaxis.tickformat = '.2f'
    fig.update_layout(dict(updatemenus=[dict(type="buttons", direction="left", buttons=list([dict(args=["visible", "legendonly"],
                                                                                                  label="Deselect All", method="restyle"), dict(args=["visible", True], label="Select All", method="restyle")]),
                                             pad={"r": 0, "t": 0}, showactive=False, x=1, xanchor="right", y=1, yanchor="top"), ]))
    fig.update_layout(width=1600, height=900)
    st.plotly_chart(fig)
    return(PCvalues)

def PCAIndep():
    dfscaled = dfcomp.copy()
    dfscaled['Competition'] = dfscaled.index.str.split('_').str[4]
    leaguedict = {'Premier League': 0, 'La Liga': 0.25,
                  'Bundesliga': 0.5, 'Serie A': 0.75, 'Ligue 1': 1}
    dfscaled = dfscaled.replace({"Competition": leaguedict})
    dfscaled = dfscaled.select_dtypes(include=[np.number])
    dfscaled1 = dfscaled.groupby('Competition').transform(
        lambda x: minmax_scale(x.astype(float)))

    pca = PCA(n_components=2, svd_solver='full')
    PCvalues = pd.DataFrame(pca.fit_transform(
        dfscaled1), columns=['PC1', 'PC2'])
    PCvalues.index = dfscaled1.index
    PCvalues['Player'] = PCvalues.index.str.split('_').str[0]
    PCvalues['Position'] = PCvalues.index.str.split('_').str[2]
    PCvalues['Team'] = PCvalues.index.str.split('_').str[3]
    PCvalues['Competition'] = PCvalues.index.str.split('_').str[4]
    PCvalues['Season'] = PCvalues.index.str.split('_').str[6]
    st.write("Cumulative Proportion of Variance Explained: " + str(100 * round(np.cumsum(pca.explained_variance_ratio_)[1],3)) + "%")
    st.write('Displaying ' + str(len(PCvalues)) + ' Players')
    fig = px.scatter(PCvalues, x='PC1', y='PC2', color="Position",
                     color_discrete_sequence=px.colors.qualitative.Set1, hover_data=['Player', 'Team', 'Season'])
    fig.update_layout(template='plotly_dark', title=dict(text=('PC2' + ' vs. ' + 'PC1'), font=dict(size=22), yanchor='top', xanchor='center', y=0.975, x=0.5),
                      xaxis_title=('PC1'), yaxis_title=('PC2'))
    fig.update_layout(autosize=False, width=1000, height=800, legend=dict(
        orientation="h", yanchor='top', xanchor='center', y=1.05, x=0.5))
    fig.layout.xaxis.tickformat = '.2f'
    fig.layout.yaxis.tickformat = '.2f'
    fig.update_layout(dict(updatemenus=[dict(type="buttons", direction="left", buttons=list([dict(args=["visible", "legendonly"],
                                                                                                  label="Deselect All", method="restyle"), dict(args=["visible", True], label="Select All", method="restyle")]),
                                             pad={"r": 0, "t": 0}, showactive=False, x=1, xanchor="right", y=1, yanchor="top"), ]))
    fig.update_layout(width=1600, height=900)
    st.plotly_chart(fig)
    return(PCvalues)


selection6b = st.selectbox('Attribute Scaling',
                           ['Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox6b>')

if selection6b == 'Scale Across Top 5 Leagues':
    PCvalues = PCADep()

elif selection6b == 'Scale Leagues Independently':
    PCvalues = PCAIndep()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# KNN Scaled Across Top 5 Leagues
st.write('''
## Nearest Neighbors
Compares two selected players.
''')

selection5a = st.selectbox('Raw or Possession-Adjusted Stats (use PAdj on non-percentage defensive metrics only)',['Raw', 'Possession Adjusted'], key='<selectbox5a>')
if selection5a == 'Raw':
    dfcomp = df_raw.copy()
elif selection5a == 'Possession Adjusted':
    dfcomp = df_PAdj.copy()

dfcomp['Selector'] = (dfcomp.index.str.split('_').str[0]) + ', ' + (dfcomp.index.str.split('_').str[2]) + ', ' + (dfcomp.index.str.split('_').str[6]) + \
        ', ' + (dfcomp.index.str.split('_').str[3])
similarity_player = st.selectbox('Select Player', sorted(dfcomp.Selector), key='<dfcomp5>')
df_similarity_player = dfcomp[dfcomp['Selector'] == similarity_player]
similarity_player_ID = df_similarity_player.index[0]
df_similarity_player.drop(['Selector'], axis=1, inplace=True)

def PlayerSimilarityDep(similarity_player_ID, dfcomp):
    dfscaled = dfcomp.copy()
    dfscaled.drop(['Selector'], axis=1, inplace = True)
    dfscaled = dfscaled.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    dfscaled1 = pd.DataFrame(scaler.fit_transform(dfscaled),index=dfscaled.index, columns=dfscaled.columns)
    df_similarity_player = dfscaled1[dfscaled1.index == similarity_player_ID].iloc[0]
    dfscaled1 = dfscaled1[dfscaled1.index != similarity_player_ID] # Comment out to include the selected player 

    distance_euc = dfscaled1.apply(lambda x: distance.euclidean(x, df_similarity_player), axis=1)
    distance_euc = pd.DataFrame(data={"Euc Distance": distance_euc})
    distance_euc.sort_values("Euc Distance", ascending=True, inplace=True)
    distance_euc['Euc Similarity'] = 1/(1 + distance_euc['Euc Distance'])
    scaler = MinMaxScaler()
    distance_euc['Euc Similarity'] = scaler.fit_transform(distance_euc[['Euc Similarity']]) # Use in conjunction with the 2 other commented out lines to scale the similarity scores

    distance_cos = dfscaled1.apply(lambda x: distance.cosine(x, df_similarity_player), axis=1)
    distance_cos = pd.DataFrame(data={"Cos Distance": distance_cos})
    distance_cos.sort_values("Cos Distance", inplace=True)
    distance_cos['Cos Similarity'] = (1-distance_cos['Cos Distance'])
    overall_similarity = pd.concat([distance_cos, distance_euc], axis=1)
    overall_similarity['Overall Similarity'] = 100*overall_similarity['Cos Similarity'] * overall_similarity['Euc Similarity']
    overall_similarity.sort_values(by=['Overall Similarity'], ascending=False, inplace=True)
    return dfscaled1, distance_euc.head(10), distance_cos.head(10), overall_similarity.head(10)

def PlayerSimilarityIndep(similarity_player_ID, dfcomp):
    dfscaled = dfcomp.copy()
    dfscaled.drop(['Selector'], axis=1, inplace = True)
    dfscaled['Competition'] = dfscaled.index.str.split('_').str[4]
    leaguedict = {'Premier League': 0, 'La Liga': 0.25,'Bundesliga': 0.5, 'Serie A': 0.75, 'Ligue 1': 1}
    dfscaled = dfscaled.replace({"Competition": leaguedict})
    dfscaled = dfscaled.select_dtypes(include=[np.number])
    
    scaler = MinMaxScaler()
    dfscaled1 = dfscaled.groupby('Competition').transform(lambda x: minmax_scale(x.astype(float)))
    df_similarity_player = dfscaled1[dfscaled1.index == similarity_player_ID].iloc[0]
    dfscaled1 = dfscaled1[dfscaled1.index != similarity_player_ID] # Comment out to include the selected player 

    distance_euc = dfscaled1.apply(lambda x: distance.euclidean(x, df_similarity_player), axis=1)
    distance_euc = pd.DataFrame(data={"Euc Distance": distance_euc})
    distance_euc.sort_values("Euc Distance", ascending=True, inplace=True)
    distance_euc['Euc Similarity'] = 1/(1 + distance_euc['Euc Distance'])
    scaler = MinMaxScaler()
    distance_euc['Euc Similarity'] = scaler.fit_transform(distance_euc[['Euc Similarity']])

    distance_cos = dfscaled1.apply(lambda x: distance.cosine(x, df_similarity_player), axis=1)
    distance_cos = pd.DataFrame(data={"Cos Distance": distance_cos})
    distance_cos.sort_values("Cos Distance", inplace=True)
    distance_cos['Cos Similarity'] = (1-distance_cos['Cos Distance'])
    overall_similarity = pd.concat([distance_cos, distance_euc], axis=1)
    overall_similarity['Overall Similarity'] = 100*overall_similarity['Cos Similarity'] * overall_similarity['Euc Similarity']
    overall_similarity.sort_values(by=['Overall Similarity'], ascending=False, inplace=True)
    return dfscaled1, distance_euc.head(10), distance_cos.head(10), overall_similarity.head(10)


selection4b = st.selectbox('Attribute Scaling',
                                ['Scale Across Top 5 Leagues', 'Scale Leagues Independently'], key='<selectbox4b>')

if selection4b == 'Scale Across Top 5 Leagues':
    scaled_features, euclidean, cosine, overall = PlayerSimilarityDep(similarity_player_ID, dfcomp)
    overall['Selector'] = (overall.index.str.split('_').str[0]) + ', ' + (overall.index.str.split('_').str[2]) + ', ' + (overall.index.str.split('_').str[6]) + \
        ', ' + (overall.index.str.split('_').str[3])
    overall.index = overall['Selector']
    overall.drop(['Selector'], axis=1, inplace=True)
    overall.insert(0, 'Player', overall.index)
    overall.reset_index(inplace=True, drop=True)
    overall.index = overall.index + 1
    st.dataframe(overall, height=800)

elif selection4b == 'Scale Leagues Independently':
    scaled_features, euclidean, cosine, overall = PlayerSimilarityIndep(
        similarity_player_ID, dfcomp)
    overall['Selector'] = (overall.index.str.split('_').str[0]) + ', ' + (overall.index.str.split('_').str[2]) + ', ' + (overall.index.str.split('_').str[6]) + \
        ', ' + (overall.index.str.split('_').str[3])
    overall.index = overall['Selector']
    overall.drop(['Selector'], axis=1, inplace=True)
    overall.insert(0, 'Player', overall.index)
    overall.reset_index(inplace = True, drop = True)
    overall.index = overall.index + 1
    st.dataframe(overall, height=800)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PCvalues['Selector'] = (PCvalues.index.str.split('_').str[0]) + ', ' + (PCvalues.index.str.split('_').str[2]) + \
    ', ' + (PCvalues.index.str.split('_').str[6]) + ', ' + (PCvalues.index.str.split('_').str[3])
comparison_results = overall['Player'].to_list()

similarity_player_ID_string = (similarity_player_ID.split('_')[0]) + ', ' + (similarity_player_ID.split('_')[2]) + \
    ', ' + (similarity_player_ID.split('_')[6]) + ', ' + (similarity_player_ID.split('_')[3])
comparison_results.append(similarity_player_ID_string)
df_comparison = pd.DataFrame(columns=['Selector','Player','Position','Season','Team','PC1','PC2',])
df_comparison.set_index('Selector', inplace=True)
for id in comparison_results:
    df_comparison = df_comparison.append(PCvalues[PCvalues.Selector == id])

fig1 = px.scatter(PCvalues, x='PC1', y='PC2', color_discrete_sequence = ['white'], hover_data=['Position', 'Player', 'Team', 'Season'], opacity = 0.05)
fig2 = px.scatter(df_comparison[df_comparison.index == similarity_player_ID], x='PC1', y='PC2', color_discrete_sequence=[
                  'green'], hover_data=['Position', 'Player', 'Team', 'Season'], opacity=1)
fig2.update_traces(marker=dict(size=10, color='green'))
fig3 = px.scatter(df_comparison[df_comparison.index != similarity_player_ID], x='PC1', y='PC2', color_discrete_sequence=['red'], hover_data=['Position','Player', 'Team','Season'], opacity = 1)
fig4 = go.Figure(data=fig1.data + fig3.data + fig2.data)
fig4.update_layout(template='plotly_dark', title=dict(text=('PC2' + ' vs. ' + 'PC1'), font=dict(size=22), yanchor='top', xanchor='center', y=0.925, x=0.5),
                  xaxis_title=('PC1'), yaxis_title=('PC2'))
#fig4.update_layout(autosize=False, width=1000, height=800, legend=dict(orientation="h", yanchor='top', xanchor='center', y=1, x= 0.5))
fig4.layout.xaxis.tickformat = '.2f'
fig4.layout.yaxis.tickformat = '.2f'
fig4.update_layout(width=1600, height=900)
st.plotly_chart(fig4)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
