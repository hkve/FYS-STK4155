import numpy as np
import pandas as pd
import os
from datetime import datetime
import copy
from collections import namedtuple, OrderedDict


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split

'''
Work in progress!
'''

data_path = ""


#   Utilities or something idk

def season2year(season : str) -> int:
    y0, y1 = 1993, 2021
    season.replace("-", "/")
    year = int(season.split("/")[0])
    if year < 100:
        if year < 23:
            year += 2000
        else:
            year += 1900
    return year

def year2season(year : int) -> str:
    yy0 = str(year)[2:] 
    yy1 = str(year+1)[2:]
    season = yy0 + "/" + yy1
    return season

def describe_features():
    d = {}
    # objects:
    d["Season"]     = "Premier League season"
    d["Datetime"]   = "Match day"
    d["FTR"]        = "Full time result"
    d["HTR"]        = "Half time result"
    d["Referee"]    = "Match official"
    # feature names
    d["HomeTeam"]   = "Host's full team name"
    d["FTHG"]       = "Goals scored by home team at full time"
    d["HTHG"]       = "Goals scored by home team at half time"
    d["HS"]         = "Shots by home team"
    d["HST"]        = "Shots on target by home team"
    d["HC"]         = "Corners given to home team"
    d["HF"]         = "Fouls commited by home team"
    d["HY"]         = "Yellow cards given to home team"
    d["HR"]         = "Red cards given to home team"
    d["HP"]         = "Home team points at full time"
    d["HHP"]        = "Home team points at half time"
    d["AwayTeam"]   = "Visitor's full team name"
    d["FTAG"]       = "Goals scored by away team at full time"
    d["HTAG"]       = "Goals scored by away team at half time"
    d["AS"]         = "Shots by away team"
    d["AST"]        = "Shots on target by away team"
    d["AC"]         = "Corners given to away team"
    d["AF"]         = "Fouls commited by away team"
    d["AY"]         = "Yellow cards given to away team"
    d["AR"]         = "Red cards given to away team"
    d["AP"]         = "Away team points at full time"
    d["HAP"]        = "Away team points at half time"

    # df = pd.DataFrame(data=d.items()).transpose()
    #print(df)
    for key in d.keys():
        print(f"{key:10s} = {d[key]}")


# describe_features()

#   Building database (per season)

# Link to kaggle site is coming (I somehow lost it...)
raw_data = pd.read_csv(data_path + "EPL_results.csv", encoding="ISO-8859-1")


def preprocess(
        data : pd.DataFrame, 
        first_season : str = "03/04", 
        last_season : str = "21/22") -> pd.DataFrame:
    
    data.DateTime = pd.to_datetime(data.DateTime).dt.date

    # get desired season data:

    y0, y1 = 1993, 2021
    first_year = season2year(first_season)
    last_year = season2year(last_season)

    year = y0 
    drop_years = ""
    while year < int(first_year):
        drop_years += str(year) + "|"
        year += 1
    year = y1
    while year > int(last_year):
        drop_years += str(year) + "|"
        year -= 1
    
    data.drop(data.loc[data["Season"].str.contains(drop_years.strip("|"), regex=True)].index, inplace=True)

    # sort and translate(?) results:

    data.sort_values(["DateTime", "HomeTeam", "AwayTeam", "Referee"], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    home_points = {"H":3, "D":1, "A":0}
    away_points = {"A":3, "D":1, "H":0}
    data["HP"] = data["FTR"].map(home_points)
    data["AP"] = data["FTR"].map(away_points)
    data["HHP"] = data["HTR"].map(home_points)
    data["HAP"] = data["HTR"].map(away_points)

    return data


first_season = "03/04"
last_season = "21/22"
best_season = "19/20" # obviously

packed_data = {}

objects = ["Season", "DateTime", "FTR", "HTR", "Referee"]
feature_names = ["FTG", "HTG", "S", "ST", "C", "F", "Y", "R", "P", "HP"]

home = ["HomeTeam", "FTHG", "HTHG"]
away = ["AwayTeam", "FTAG", "HTAG"]

for ft in feature_names[2:-1]:
    home.append("H"+ft)
    away.append("A"+ft)
home.append("HHP")
away.append("HAP")

data = preprocess(raw_data, first_season, last_season)
data = data.reindex(columns=objects+home+away)

home_data = data[objects+home]
away_data = data[objects+away]

packed_data["data"] = data
packed_data["home_data"] = home_data
packed_data["away_data"] = away_data
packed_data["home"] = home
packed_data["away"] = away
packed_data["objects"] = objects
packed_data["feature_names"] = feature_names



raw_season_data = {}

for year in range(2013, 2020+1):
    # pls send help
    yy = year - 2000
    if yy < 9:
        yy0 = "0" + str(yy)
        yy1 = "0" + str(yy+1)
    elif yy == 9:
        yy0 = "09"
        yy1 = "10"
    else: 
        yy0 = str(yy)
        yy1 = str(yy+1)
   
    raw_season_data[yy0+"/"+yy1] = data.loc[data["Season"] == str(year)+"-"+str(year+1)[2:]]




"""
Obs: The following is dangerously similar to: https://armantee.github.io/predicting/
should fix
"""

def home_dict(idk, match_num : int) -> pd.DataFrame:

    match = {}
    match["Result"] = idk["FTR"].values[0]
    match["Goals"] = idk["FTHG"].values[0]
    match["GoalsConceded"] = idk["FTAG"].values[0]
    match["Ground"] = "H"

    match["MatchNumber"] = match_num # GW?
    match["Date"] = idk["DateTime"].values[0]
    match["Opponent"] = idk["AwayTeam"].values[0]
    match["YellowCards"] = idk["HY"].values[0]
    match["YellowCardsAgainst"] = idk["AY"].values[0]
    match["RedCards"] = idk["HR"].values[0]
    match["RedCardsAgainst"] = idk["AR"].values[0]
    match["Corners"] = idk["HC"].values[0]
    match["CornersAgainst"] = idk["AC"].values[0]
    match["FoulsCommited"] = idk["HF"].values[0]
    match["FoulsAgainst"] = idk["AF"].values[0]
    match["Shots"] = idk["HS"].values[0]
    match["ShotsOnTarget"] = idk["HST"].values[0]
    match["ShotsAgainst"] = idk["AS"].values[0]
    match["ShotsAgainstOnTarget"] = idk["AST"].values[0]
    # .....
    match["BigChancesCreated"] = match["ShotsOnTarget"] + match["Goals"]


    if match["Result"] == "H":
        match.update({"Win":1, "Draw":0, "Lose":0})
    elif match["Result"] == "A":
        match.update({"Win":0, "Draw":0, "Lose":1})
    elif match["Result"] == "D":
        match.update({"Win":0, "Draw":1, "Lose":0})
    return pd.DataFrame(match, index=[match_num, ])

def away_dict(idk, match_num : int) -> pd.DataFrame:

    match = {}
    match["Result"] = idk["FTR"].values[0]
    match["Goals"] = idk["FTAG"].values[0]
    match["GoalsConceded"] = idk["FTHG"].values[0]
    match["Ground"] = "A"

    match["MatchNumber"] = match_num # GW?
    match["Date"] = idk["DateTime"].values[0]
    match["Opponent"] = idk["HomeTeam"].values[0]
    match["YellowCards"] = idk["AY"].values[0]
    match["YellowCardsAgainst"] = idk["HY"].values[0]
    match["RedCards"] = idk["AR"].values[0]
    match["RedCardsAgainst"] = idk["HR"].values[0]
    match["Corners"] = idk["AC"].values[0]
    match["CornersAgainst"] = idk["HC"].values[0]
    match["FoulsCommited"] = idk["AF"].values[0]
    match["FoulsAgainst"] = idk["HF"].values[0]
    match["Shots"] = idk["AS"].values[0]
    match["ShotsOnTarget"] = idk["AST"].values[0]
    match["ShotsAgainst"] = idk["HS"].values[0]
    match["ShotsAgainstOnTarget"] = idk["HST"].values[0]
    # etc...
    match["BigChancesCreated"] = match["ShotsOnTarget"] + match["Goals"]

    if match["Result"] == "A":
        match.update({"Win":1, "Draw":0, "Lose":0})
    elif match["Result"] == "H":
        match.update({"Win":0, "Draw":0, "Lose":1})
    elif match["Result"] == "D":
        match.update({"Win":0, "Draw":1, "Lose":0})
    return pd.DataFrame(match, index=[match_num, ])




def build_snapshot_table(raw_data : dict, snapshots : dict = {}) -> dict:
    for ht in raw_data.keys():
        snapshots[ht] = {}
        for at in list(set(raw_data[ht]['AwayTeam'])):
            snapshots[ht][at] = pd.DataFrame()
            tab = raw_data[ht][(raw_data[ht]["AwayTeam"] == at) | (raw_data[ht]["HomeTeam"] == at)]
            for match_num in range(len(tab)):
                if at == tab.iloc[match_num]["AwayTeam"]:
                    snapshots[ht][at] = pd.concat([snapshots[ht][at], away_dict(tab[match_num:match_num+1], match_num+1)])
                elif at == tab.iloc[match_num]["HomeTeam"]:
                    snapshots[ht][at] = pd.concat([snapshots[ht][at], home_dict(tab[match_num:match_num+1], match_num+1)])
    return snapshots


def build_season_stats(teamdf : pd.DataFrame, team_name : str) -> dict:
    stats = {
        "Team": team_name,
        "W": int(sum(teamdf["Win"])), # wins
        "L": int(sum(teamdf["Lose"])), # losses
        "D": int(sum(teamdf["Draw"])), # draws
        "G": int(sum(teamdf["Goals"])), # goals for
        "GC": int(sum(teamdf["GoalsConceded"])), # goals against
        "Yc": int(sum(teamdf["YellowCards"])), # yellow cards recieved
        "Rc": int(sum(teamdf["RedCards"])), # red cards recieved
        "avg_C": np.mean(teamdf["Corners"]), # average corners won
        "avg_CA": np.mean(teamdf["CornersAgainst"]), # average corners given away
        "avg_F": np.mean(teamdf["FoulsCommited"]), # average fouls commited
        "avg_FA": np.mean(teamdf["FoulsAgainst"]), # average fouls commited by opponent
        "avg_G" : np.mean(teamdf["Goals"]), # average goals for
        "avg_GC": np.mean(teamdf["GoalsConceded"]), # avgerage goales against
        "avg_S": np.mean(teamdf["Shots"]), # average shots for
        "avg_SC": np.mean(teamdf["ShotsAgainst"]), # average shots against
        "avg_BCC": np.mean(teamdf["BigChancesCreated"]), # average big chances created
        } 
    return stats

def build_team_summary(teamdf : pd.DataFrame, team_name : str) -> pd.DataFrame:
    homedf = teamdf.query("Ground == 'H'")
    awaydf = teamdf.query("Ground == 'A'")
    summary = build_season_stats(teamdf, team_name)
    home = build_season_stats(homedf, team_name)
    away = build_season_stats(awaydf, team_name)

    home = dict(zip(["home_" + i for i in home.keys()], home.values()))
    away = dict(zip(["away_" + i for i in away.keys()], away.values()))

    summary = {**summary, **home, **away}
    summary = {**summary, **{"Pts":summary["W"]*3 + summary["D"], "GD": summary["G"]-summary["GC"]}}

    return pd.DataFrame(summary, index=[team_name,])

def build_season_table(snapshots):
    table = {}
    for s in snapshots.keys():
        table[s] = pd.DataFrame()
        
        for ht in snapshots[s].keys():
            table[s] = pd.concat([table[s], build_team_summary(snapshots[s][ht], ht)])
        table[s] = table[s].sort_values(by=["Pts", "GD"], ascending=False)
        table[s]["Position"] = range(1, 20+1) 
    return table


snapshots = build_snapshot_table(raw_season_data)
table = build_season_table(snapshots)

print(table[best_season])



"""
Idea is to get
* previous season stats
* previous match stats 
* previous match stats for opponent

Then build features. Following: https://armantee.github.io/predicting/ 


Copy-paste (ish) for now, just to test:
"""



def prev_game_features(prevGame,gameType,md,side):
    if type(prevGame) == type(0):
        return prev_game_zeros(prevGame, gameType, md, side)
    try:
        mp = datetime.strptime(prevGame['Date'], "%d/%m/%y")
    except TypeError:
        mp = prevGame['Date']
    return {side+'_'+gameType+'_gamesPlayed':prevGame['MatchNumber'],
                    side+'_'+gameType+'_daysRested':(md-mp).days,
                    side+'_'+gameType+'_prevGame_BigChancesCreated': prevGame['BigChancesCreated'],
                    side+'_'+gameType+'_prevGame_Corners': prevGame['Corners'],
                    side+'_'+gameType+'_prevGame_CornersAgainst': prevGame['CornersAgainst'],
                    side+'_'+gameType+'_prevGame_Draw': prevGame['Draw'],
                    side+'_'+gameType+'_prevGame_FoulsAgainst': prevGame['FoulsAgainst'],
                    side+'_'+gameType+'_prevGame_FoulsCommited': prevGame['FoulsCommited'],
                    side+'_'+gameType+'_prevGame_Goals': prevGame['Goals'],
                    side+'_'+gameType+'_prevGame_GoalsConceded': prevGame['GoalsConceded'],
                    side+'_'+gameType+'_prevGame_Lose': prevGame['Lose'],
                    side+'_'+gameType+'_prevGame_RedCards': prevGame['RedCards'],
                    side+'_'+gameType+'_prevGame_RedCardsAgainst': prevGame['RedCardsAgainst'],
                    side+'_'+gameType+'_prevGame_Shots': prevGame['Shots'],
                    side+'_'+gameType+'_prevGame_ShotsAgainst': prevGame['ShotsAgainst'],
                    side+'_'+gameType+'_prevGame_ShotsAgainstOnTarget': prevGame['ShotsAgainstOnTarget'],
                    side+'_'+gameType+'_prevGame_ShotsOnTarget' : prevGame['ShotsOnTarget'],
                    side+'_'+gameType+'_prevGame_Win': prevGame['Win'],
                    side+'_'+gameType+'_prevGame_YellowCards': prevGame['YellowCards'],
                    side+'_'+gameType+'_prevGame_YellowCardsAgainst': prevGame['YellowCardsAgainst']
                    }

def prev_game_zeros(prevGame,gameType,md,side):
    return  {   side+'_'+gameType+'_gamesPlayed':0,
                side+'_'+gameType+'_daysRested':60,
                side+'_'+gameType+'_prevGame_BigChancesCreated':0,
                side+'_'+gameType+'_prevGame_Corners':0,
                side+'_'+gameType+'_prevGame_CornersAgainst': 0,
                side+'_'+gameType+'_prevGame_Draw':0,
                side+'_'+gameType+'_prevGame_FoulsAgainst':0,
                side+'_'+gameType+'_prevGame_FoulsCommited': 0,
                side+'_'+gameType+'_prevGame_Goals':0,
                side+'_'+gameType+'_prevGame_GoalsConceded':0,
                side+'_'+gameType+'_prevGame_Lose': 0,
                side+'_'+gameType+'_prevGame_RedCards':0,
                side+'_'+gameType+'_prevGame_RedCardsAgainst':0,
                side+'_'+gameType+'_prevGame_Shots':0,
                side+'_'+gameType+'_prevGame_ShotsAgainst':0,
                side+'_'+gameType+'_prevGame_ShotsAgainstOnTarget':0,
                side+'_'+gameType+'_prevGame_ShotsOnTarget':0,
                side+'_'+gameType+'_prevGame_Win': 0,
                side+'_'+gameType+'_prevGame_YellowCards':0,
                side+'_'+gameType+'_prevGame_YellowCardsAgainst':0
                }

def prev_games_stats(prevGames,gameType,count,md,side):
    return {side+'_'+gameType+'_avgRestDays':calc_avg_restTime(prevGames['Date'].values),
        side+'_'+gameType+'_avgBigChancesCreated':np.mean(prevGames['BigChancesCreated']),
        side+'_'+gameType+'_avgCorners':np.mean(prevGames['Corners']),
        side+'_'+gameType+'_avgPoints':((3*sum(prevGames['Win']))+sum(prevGames['Draw']))/count,
        side+'_'+gameType+'_avgYellowCards':np.mean(prevGames['YellowCards']),
        side+'_'+gameType+'_avgRedCards':np.mean(prevGames['RedCards']),
        side+'_'+gameType+'_avgGoals':np.mean(prevGames['Goals']),
        side+'_'+gameType+'_avgGoalsConceded':np.mean(prevGames['GoalsConceded']),
        side+'_'+gameType+'_numWins':sum(prevGames['Win']),
        side+'_'+gameType+'_numLosses':sum(prevGames['Lose']),
        side+'_'+gameType+'_numDraws':sum(prevGames['Draw'])
        }

def prev_season_stats(prevSeason,gameType,side):
    return {
        side+'_season_'+'Position':prevSeason['Position'].values[0],
        side+'_season_'+'Draws':prevSeason['D'].values[0],
        side+'_season_'+'Wins':prevSeason['W'].values[0],
        side+'_season_'+'Losses':prevSeason['L'].values[0],
        side+'_season_'+'GD':prevSeason['GD'].values[0],
        side+'_season_'+'Points':prevSeason['Pts'].values[0],
        side+'_season_'+'RedCards':prevSeason['Rc'].values[0],
        side+'_season_'+'YellowCards':prevSeason['Yc'].values[0],
        side+'_season_'+'avg_BigChancesCreated':prevSeason['avg_BCC'].values[0],
        side+'_season_'+'avg_Corners':prevSeason['avg_C'].values[0],
        side+'_season_'+'avg_CornersAgaints':prevSeason['avg_CA'].values[0],
        side+'_season_'+'avg_Fouls':prevSeason['avg_F'].values[0],
        side+'_season_'+'avg_FoulsAgainst':prevSeason['avg_FA'].values[0],
        side+'_season_'+'avg_Goals':prevSeason['avg_G'].values[0],
        side+'_season_'+'avg_GoalsAgainst':prevSeason['avg_GC'].values[0] ,
        side+'_season_'+'avg_Shots':prevSeason['avg_S'].values[0] ,
        side+'_season_'+'avg_ShotsAgainst':prevSeason['avg_SC'].values[0],
        side+'_'+gameType+'_season_'+'Draws':prevSeason[gameType+'_'+'D'].values[0],
        side+'_'+gameType+'_season_'+'Wins':prevSeason[gameType+'_'+'W'].values[0],
        side+'_'+gameType+'_season_'+'Losses':prevSeason[gameType+'_'+'L'].values[0],
        side+'_'+gameType+'_season_'+'RedCards':prevSeason[gameType+'_'+'Rc'].values[0],
        side+'_'+gameType+'_season_'+'YellowCards':prevSeason[gameType+'_'+'Yc'].values[0],
        side+'_'+gameType+'_season_'+'avg_BigChancesCreated':prevSeason[gameType+'_'+'avg_BCC'].values[0],
        side+'_'+gameType+'_season_'+'avg_Corners':prevSeason[gameType+'_'+'avg_C'].values[0],
        side+'_'+gameType+'_season_'+'avg_CornersAgaints':prevSeason[gameType+'_'+'avg_CA'].values[0],
        side+'_'+gameType+'_season_'+'avg_Fouls':prevSeason[gameType+'_'+'avg_F'].values[0],
        side+'_'+gameType+'_season_'+'avg_FoulsAgainst':prevSeason[gameType+'_'+'avg_FA'].values[0],
        side+'_'+gameType+'_season_'+'avg_Goals':prevSeason[gameType+'_'+'avg_G'].values[0],
        side+'_'+gameType+'_season_'+'avg_GoalsAgainst':prevSeason[gameType+'_'+'avg_GC'].values[0],
        side+'_'+gameType+'_season_'+'avg_Shots':prevSeason[gameType+'_'+'avg_S'].values[0],
        side+'_'+gameType+'_season_'+'avg_ShotsAgainst':prevSeason[gameType+'_'+'avg_SC'].values[0]
    }

def prev_vs_stats(prevGames_h,prevGames_a):
    return {
    'hs_prev_vs_away_Win':prevGames_h['Win'].values[0],
    'hs_prev_vs_away_Lose':prevGames_h['Lose'].values[0],
    'hs_prev_vs_away_Draw':prevGames_h['Draw'].values[0],
    'hs_prev_vs_away_Goals':prevGames_h['Goals'].values[0],
    'hs_prev_vs_away_BigChancesCreated':prevGames_h['BigChancesCreated'].values[0],
    'hs_vs_away_avgBigChancesCreated':np.mean(prevGames_h['BigChancesCreated'].values[0]),
    'hs_vs_away_avgCorners':np.mean(prevGames_h['Corners'].values[0]),
    'hs_vs_away_avgYellowCards':np.mean(prevGames_h['YellowCards'].values[0]),
    'hs_vs_away_avgRedCards':np.mean(prevGames_h['RedCards'].values[0]),
    'hs_vs_away_avgGoals':np.mean(prevGames_h['Goals'].values[0]),
    'as_prev_vs_home_Goals':prevGames_a['Goals'].values[0],
    'as_prev_vs_home_BigChancesCreated':prevGames_a['BigChancesCreated'].values[0],
    'as_vs_home_avgBigChancesCreated':np.mean(prevGames_a['BigChancesCreated'].values[0]),
    'as_vs_home_avgCorners':np.mean(prevGames_a['Corners'].values[0]),
    'as_vs_home_avgYellowCards':np.mean(prevGames_a['YellowCards'].values[0]),
    'as_vs_home_avgRedCards':np.mean(prevGames_a['RedCards'].values[0]),
    'as_vs_home_avgGoals':np.mean(prevGames_a['Goals'].values[0])
    }

def calc_avg_restTime(dates):
    avgD=0
    dates=copy.deepcopy(dates)
    for i in range(len(dates)):
        try:
            dates[i] = datetime.strptime(dates[i], "%d/%m/%y")
        except TypeError:
            dates[i] = dates[i]
    for i in range(len(dates))[1:]:
        avgD+=(dates[i]-dates[i-1]).days
    try:
        avgD = avgD/(len(dates)-1)
    except:
        avgD = 60
    return avgD

def get_targets(dataDict):
        matchDict={}
        matchDict = {"Result": dataDict["FTR"]}

        if matchDict['Result']=='H':
            matchDict.update({'Win':1,'Draw':0, 'Lose':0})
        elif matchDict['Result']=='A':
            matchDict.update({'Win':0,'Draw':0, 'Lose':1})
        else:
            matchDict.update({'Win':0,'Draw':1, 'Lose':0})
        return matchDict



### Stats to consider ###
def build_features(seasonData,snapshots,rawData,lookback=5):
    features= pd.DataFrame()
    for i in list(rawData.keys())[1:]:
        y_i = season2year(i)
        prev_i = year2season(y_i-1)
        for j, r in rawData[i].iterrows():
            ht = r['HomeTeam']
            at = r['AwayTeam']
            #Get HomeSide Data
            try:
                md = datetime.strptime(r['DateTime'], '%d/%m/%y')
            except TypeError:
                md = r['DateTime']
            hs_mn = snapshots[i][ht].query('Ground == "H" and Opponent =="'+at+'"')['MatchNumber'].values[0]
            if hs_mn>1:
                hs_prev_game_single = snapshots[i][ht].loc[hs_mn-1,]
            else:
                hs_prev_game_single=0
            hs_prev_games = snapshots[i][ht].loc[hs_mn-lookback:hs_mn-1,]
            hs_prev_home_games= snapshots[i][ht].query('Ground == "H"')
            hs_hg_mn = list(hs_prev_home_games.index).index(hs_mn)
            hs_prev_home_games = hs_prev_home_games.iloc[hs_hg_mn-lookback:hs_hg_mn,]
            # get previous seasons summaries

            if ht in seasonData[prev_i]['Team']:
                hs_prev_season_sum = seasonData[prev_i].loc[seasonData[prev_i]['Team']==ht]
            else:
                hs_prev_season_sum = seasonData[prev_i].loc[seasonData[prev_i]['Position']==15]
            #get previous seasons snapshots
            try:
                hs_prevSeason = snapshots[prev_i][ht]
            except:
                ##if prev season doesnt exist, pick number 15th as average performance
                hs_prevSeason = snapshots[prev_i][seasonData[prev_i].iloc[14,]['home_Team']]
            hs_prevSeason_vs_away = hs_prevSeason.query('Opponent =="' + at + '"')
            if not len(hs_prevSeason_vs_away):
                    hs_prevSeason_vs_away = hs_prevSeason.query('Opponent =="' + seasonData[prev_i].iloc[15,]['home_Team'] + '"')
            if not len(hs_prevSeason_vs_away):
                    hs_prevSeason_vs_away = hs_prevSeason.query('Opponent =="' + seasonData[prev_i].iloc[14,]['home_Team'] + '"')
            hs_vs_away = snapshots[i][ht].query('MatchNumber <'+str(hs_mn)+' and Opponent == "'+str(at)+'"')
            hs_vs_away =pd.concat([hs_prevSeason_vs_away, hs_vs_away])
            #Get Away Side Data
            as_mn = snapshots[i][at].query('Ground == "A" and Opponent =="'+ht+'"')['MatchNumber'].values[0]
            if as_mn>1:
                as_prev_game_single = snapshots[i][ht].loc[as_mn-1,]
            else:
                as_prev_game_single=0
            as_prev_games = snapshots[i][at].loc[as_mn-lookback:as_mn-1,]
            as_prev_away_games= snapshots[i][at].query('Ground == "A"')
            as_ag_mn = list(as_prev_away_games.index).index(as_mn)
            as_prev_away_games = as_prev_away_games.iloc[as_ag_mn-lookback:as_ag_mn,]
            if at in seasonData[prev_i]['Team']:
                as_prev_season_sum = seasonData[prev_i].loc[seasonData[prev_i]['Team']==at]
            else:
                as_prev_season_sum = seasonData[prev_i].loc[seasonData[prev_i]['Position']==15]
            try:
                as_prevSeason = snapshots[prev_i][at]
            except:
                ##if prev season doesnt exist, pick number 15th as average performance
                as_prevSeason = snapshots[prev_i][seasonData[prev_i].iloc[14,]['home_Team']]
            as_prevSeason_vs_away = as_prevSeason.query('Opponent =="' + ht + '"')
            if not len(as_prevSeason_vs_away):
                as_prevSeason_vs_away = as_prevSeason.query('Opponent =="' + seasonData[prev_i].iloc[15,]['home_Team'] + '"')
            if not len(as_prevSeason_vs_away):
                    as_prevSeason_vs_away = as_prevSeason.query('Opponent =="' + seasonData[prev_i].iloc[14,]['home_Team'] + '"')
            as_vs_away = snapshots[i][at].query('MatchNumber <'+str(as_mn)+' and Opponent == "'+str(ht)+'"')
            as_vs_away =pd.concat([as_prevSeason_vs_away, as_vs_away])
            features=pd.concat([features, pd.DataFrame({**prev_game_features(hs_prev_game_single,'home',md,'hs'),
            **prev_game_features(as_prev_game_single,'away',md,'as'),
            **prev_games_stats(hs_prev_games,'any',lookback,md,'hs'),
            **prev_games_stats(hs_prev_home_games,'home',lookback,md,'hs'),
            **prev_games_stats(as_prev_games,'any',lookback,md,'hs'),
            **prev_games_stats(as_prev_away_games,'away',lookback,md,'as'),
            **prev_season_stats(hs_prev_season_sum, 'home', 'hs'),
            **prev_season_stats(as_prev_season_sum, 'away', 'as'),
            **prev_vs_stats(hs_vs_away, as_vs_away),
            **get_targets(r)
            },index=[j,])])
    return features


### Complete features/Targets without changing NANS
complete_features = build_features(table,snapshots,raw_season_data)
complete_targets = complete_features['Result']
complete_targets_OH= complete_features[(['Win','Lose','Draw'])]

## dropped NANS (drop first 5 games of the season)
complete_features_dropna = complete_features.dropna(axis=0)
complete_targets_dropna = complete_features_dropna['Result']
complete_targets_OH_dropna = complete_features_dropna[(['Win','Lose','Draw'])]
complete_features_dropna=complete_features_dropna.drop((['Win','Lose','Draw','Result']),axis=1)


## NANS  = 0
complete_features_fillna = complete_features.fillna(0)
complete_targets_OH_fillna = complete_features_fillna[(['Win','Lose','Draw'])]
complete_targets_fillna = complete_features_fillna['Result']
complete_features_fillna=complete_features_fillna.drop((['Win','Lose','Draw','Result']),axis=1)


print("Total number of records: {}".format(len(complete_features)))
print("Total number of Features: {}".format(len(complete_features.keys())))

print("Total number of records after dropping NANS: {}".format(len(complete_features_dropna)))
# print("Total number of records after filling NANS: {}".format(len(complete_features_fillna)))

# print(complete_features_fillna)
# print(complete_features.isna())

# for col in complete_features.columns:
    # if np.any(np.isnan(complete_features[col].tail(2))):
        # print("HERE")
    # print(col)#, " ... ", complete_features[col].tail(1))
# print(complete_features.head())
# print(complete_features.tail(15))
# print(complete_features_dropna.head())
# print(complete_features_dropna.tail())




X_train, X_test, y_train, y_test = train_test_split(complete_features_dropna, complete_targets_OH_dropna)

X_train, X_test, y_train, y_test = train_test_split(complete_features_fillna,complete_targets_OH_fillna, test_size=0.2, random_state=123)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)



# Building the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.3))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(.2))
model.add(Dense(16, activation='tanh'))
model.add(Dropout(.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
hist=model.fit(X_train, y_train, epochs=800, batch_size=15, verbose=0, validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)

print(score)