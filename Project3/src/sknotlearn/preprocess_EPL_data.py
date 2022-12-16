import numpy as np
import pandas as pd
import os
from datetime import datetime
import copy
from collections import namedtuple, OrderedDict
import pathlib as pl



'''
Build a dataset containing the following features:
    Team attributes: (same for all matches in a season)
        * annual wages 
        * previous season understats (xG, xGA, etc.) 
            > for promoted teams, choose data from relegated teams (?)
    Match features:
        * team: team attributes
        * team: understats from previous match
        * team: ground (home/away)
        * opponent: team attributes
'''




### Some global variables
DATE_FORMAT = "%Y-%m-%d"    


### Choose league and year
YEAR = 2019
prevYEAR = YEAR-1
LEAGUE = "EPL"

### Define some filenames
MATCH_DATA_CSV = "EPL1920"
prevTEAMS_WAGES_CSV = "salary18"
TEAMS_WAGES_CSV = "salary19"
PATH = pl.Path(__file__).parent

### Extra features to consider (must be present in 'MATCH_DATA_CSV' with H/A-prefixes)
EXTRA_FEATURES = ["S", "ST", "C", "F", "Y", "R"]



''' Comments:
    > Leaving the variable names specific to our case
    > Still possible to change league and season by changing the global variables

    > Names of teams according to "understat_per_game.csv" 
'''



""" (1) Extract relevant data """

def READ():
    global understats_EPL18, understats_EPL19_per_game
    ### Read understats of previous season (18/19) and chosen season (19/20) of league (EPL)
    understats = pd.read_csv(PATH/"understat.com.csv").rename(index=int, columns={"Unnamed: 0": "league", "Unnamed: 1": "year"}) 
    # English Premier League:
    understats_EPL = understats[(understats["league"] == LEAGUE)].drop(["league", "matches"], axis=1)
    # English Premier League 2018/19:
    understats_EPL18 = understats_EPL[(understats_EPL["year"] == prevYEAR)].drop(["year"], axis=1)

    ### Read understats per game of the chosen season (19/20) of league (EPL)
    understats_per_game = pd.read_csv(PATH/"understat_per_game.csv").rename(columns={"h_a": "ground"})
    understats_per_game.date = pd.to_datetime(understats_per_game.date, format="%Y-%m-%d").dt.strftime(DATE_FORMAT)
    # English Premier League:
    understats_EPL_per_game = understats_per_game.loc[(understats_per_game["league"] == LEAGUE)].drop((["league"]), axis=1)
    # English Premier League 2019/20:
    understats_EPL19_per_game = understats_EPL_per_game.loc[(understats_EPL_per_game["year"] == YEAR)].drop((["year"]), axis=1)
    understats_EPL19_per_game = understats_EPL19_per_game.sort_values(by="date")
    understats_EPL19_per_game.reset_index(drop=True, inplace=True)


### Get team attributes for the previous season (18/19) and chosen season (19/20) of league (EPL)
def get_wages(csv_filename : str) -> pd.DataFrame:
    # read wages file and fix dataframe:
    filename = csv_filename.replace(".csv", "")+".csv"
    wages = pd.read_csv(PATH/filename).rename(columns={"Squad": "team", "# Pl": "n_contracts", "Annual Wages": "annual_wages"}).drop((["Rk", "Weekly Wages", "% Estimated"]), axis=1)
    # fill in for teams (should I use merge??):
    for i in wages.index:
        sal = wages.at[i, "annual_wages"]
        sal = sal.split("(")[0].replace("£", "").strip()
        wages.at[i, "annual_wages"] = float(sal)*1e-6
    # use name convention:
    names = {"Manchester Utd": "Manchester United", "Newcastle Utd": "Newcastle United", "Wolves":"Wolverhampton Wanderers", "Sheffield Utd": "Sheffield United", "Leicester City": "Leicester", "Norwich City": "Norwich"}
    for shortname in names.keys():
        wages = wages.replace(shortname, names[shortname])
    return wages


### Get basic match data [...]
def get_match_data(csv_filename : str) -> pd.DataFrame:
    filename = csv_filename.replace(".csv", "")+".csv"
    match_data = pd.read_csv(PATH/filename).rename(columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team"})
    # drop irrelevant coloumns
    init_date_format = "%d/%m/%Y"
    # final_date_format = "%d%m$Y"
    match_data.date = pd.to_datetime(match_data.date, format=init_date_format).dt.strftime(DATE_FORMAT)
    match_data = match_data.sort_values(by="date")
    match_data.reset_index(drop=True, inplace=True)
    names = {"Man City": "Manchester City", "Wolves": "Wolverhampton Wanderers", "Man United": "Manchester United", "Newcastle": "Newcastle United"}
    for shortname in names.keys():
        match_data = match_data.replace(shortname, names[shortname])
    return match_data



""" (2) Combine season understats and attributes """

### Build team profile for this season (19/20)
def build_team_profiles(
                season_attributes : pd.DataFrame, 
                prev_season_understats : pd.DataFrame, 
                prev_season_attributes : pd.DataFrame,
                avg_X_teams : int = 5) -> pd.DataFrame:

    # add suffix to metrics
    prev_season_attributes = prev_season_attributes.add_suffix("_ps").rename(columns={"team_ps":"team"})
    prev_season_understats = prev_season_understats.add_suffix("_ps").rename(columns={"team_ps":"team"})

    # deal with promoted teams' understats:
    assumed_understats = prev_season_understats.loc[prev_season_understats["position_ps"] > (20-avg_X_teams)] # avg X of lower table half
    
    # put together all attributes:
    attributes = season_attributes.merge(prev_season_attributes, how="left", on="team", suffixes=("", "_ps"))
    releg_attributes = season_attributes.merge(season_attributes, how="left", on="team", suffixes=("", "_ps"))
    attributes = attributes.fillna(releg_attributes)
    # merge attributes and prev. season understats:
    season_stats = attributes.merge(prev_season_understats, how="left", on="team", suffixes=("", "_ps"))
    season_stats = season_stats.fillna(np.mean(assumed_understats, axis=0), axis=0)
    # make it easy to merge with dataframe later:
    season_stats["opp_team"] = season_stats["team"] 

    return season_stats



""" (3) Combine game understats with opponent's understats """


### Put together per-game stats of current season (19/20) and plain match data
def get_opponent(   
            understats_per_game : pd.DataFrame, 
            match_data : pd.DataFrame) -> tuple: # how do I say "tuple[pd.DataFrame, pd.DataFrame]"??

    
    # run through all matches in the season to pair up understats
    for match_no in range(len(match_data)):
        # create a match ID easy access to correspoinding elements:
        match_id = 20190000 + int(match_no)
        match_data.at[match_no, "match_id"] = match_id
        
        # find match day and squads: 
        match_day = match_data.at[match_no, "date"]
        home_team = match_data.at[match_no, "home_team"]
        away_team = match_data.at[match_no, "away_team"]
        weekday = pd.to_datetime(match_day, format=DATE_FORMAT).isoweekday()
        
        # isolate two per-game understats:
        understats_home_team = understats_per_game.loc[(understats_per_game["ground"] == "h") & (understats_per_game["team"] == home_team) & (understats_per_game["date"] == match_day)]
        understats_away_team = understats_per_game.loc[(understats_per_game["ground"] == "a") & (understats_per_game["team"] == away_team) & (understats_per_game["date"] == match_day)]
        
        # provide information to the dataframes:
        idx_home_team = understats_home_team.index.values[0]
        idx_away_team = understats_away_team.index.values[0]
        understats_per_game.at[idx_home_team, "opp_team"] = away_team
        understats_per_game.at[idx_away_team, "opp_team"] = home_team
        understats_per_game.at[idx_home_team, "match_id"] = match_id
        understats_per_game.at[idx_away_team, "match_id"] = match_id
        
        understats_per_game.at[idx_home_team, "day"] = weekday
        understats_per_game.at[idx_away_team, "day"] = weekday

        # additional data
        for feature in EXTRA_FEATURES:
            understats_per_game.at[idx_home_team, feature] = match_data.at[match_no, "H" + feature]
            understats_per_game.at[idx_away_team, feature] = match_data.at[match_no, "A" + feature]

    
    # sort by match ID:
    understats_per_game = understats_per_game.sort_values(["match_id", "ground"], ascending=[True, False]).reset_index(drop=True, inplace=False)
    match_data = match_data.sort_values(by="match_id").reset_index(drop=True, inplace=False)
    return understats_per_game, match_data



""" (4) Combine prev. game stats with team stats """


### Let last match's KPI-s be features for current match [...]
def get_team_record(
            understats_per_game : pd.DataFrame, 
            team : str) -> pd.DataFrame:
    
    # get team per-game stats:
    teamdf = understats_per_game.loc[(understats_per_game["team"] == team)]
    teamdf = teamdf.reset_index(drop=True, inplace=False)
    # prepare new dataframes:
    keep = ["match_id", "team", "opp_team", "date", "ground", "result", "day"]
    pre_game = pd.DataFrame()
    kpi = teamdf.drop((["match_id", "team", "opp_team", "date"]), axis=1)
    # fill new feature object using brute (?) force:
    for game in teamdf.index[1:]:
        # this game:
        for cat in keep:
            pre_game.at[game, cat] = teamdf.at[game, cat]
        # days rested:
        rest = (pd.to_datetime(teamdf.at[game, "date"], format=DATE_FORMAT) - pd.to_datetime(teamdf.at[game-1, "date"], format=DATE_FORMAT)).days
        pre_game.at[game, "days_rest"] = rest
        # previous game:
        for cat in kpi.columns:
            pre_game.at[game, cat + "_pg"] = kpi.at[game-1, cat]
    return pre_game



""" (5) Sort features (stats) and targets (result) """

### Build features to our desired form or something idk [...]
def build_features(
        match_data : pd.DataFrame, 
        understats_per_game : pd.DataFrame, 
        team_profiles : pd.DataFrame) -> pd.DataFrame:


    ### (3) 
    understats_per_game, match_data = get_opponent(understats_per_game, match_data)
    
    ### (4)
    pergame_stats = pd.DataFrame()
    # retrieve stats from last match:
    for team in team_profiles["team"]:
        pergame_stats = pd.concat([pergame_stats, get_team_record(understats_per_game, team)])

    ### (5)
    pergame_stats = pergame_stats.sort_values(by=["match_id", "ground"], ascending=[True, False]).reset_index(drop=True)
    # associate team stats to match p.o.v.:
    pergame_stats = pergame_stats.merge(team_profiles, how="left", on="team", suffixes=("", "_ps")).drop(["opp_team_ps"], axis=1)
    # associate oppositon stats to opp match p.o.v.:
    pergame_stats = pergame_stats.merge(team_profiles, how="left", on="opp_team", suffixes=("", "_opp")).drop(["team_opp"], axis=1)

    return pergame_stats


def RUN():
   
    ### (1)
    READ()
    wages_EPL18 = get_wages(prevTEAMS_WAGES_CSV)
    wages_EPL19 = get_wages(TEAMS_WAGES_CSV)
    match_data_EPL19 = get_match_data(MATCH_DATA_CSV)

    ### (2)
    # put together team attributes for previous season (18/19):
    attributes_EPL18 = pd.concat([wages_EPL18]) # not very intersting atm
    # put together team attributes for this season (19/20):
    attributes_EPL19 = pd.concat([wages_EPL19]) # not very intersting atm

    team_profiles_EPL19 = build_team_profiles(attributes_EPL19, understats_EPL18, attributes_EPL18)

    global stats_EPL19_per_game

    stats_EPL19_per_game = build_features(match_data_EPL19, understats_EPL19_per_game, team_profiles_EPL19)
 

### Write to file??

""" Additional functionalities: """

### Encode (and scale?) data
def translate_data(
        data : pd.DataFrame,
        omit_features=[None]) -> pd.DataFrame:

    ### Encode data of non-float types
    from sklearn.preprocessing import OrdinalEncoder
    def encoder(data, features):
        oe = OrdinalEncoder()
        data[features] = oe.fit_transform(data[features])
        return data

    container = namedtuple("container", ["x", "y"])
    
    features = data.copy()
    for feat in omit_features:
        if feat is not None:
            features[feat] = data.drop(feat)

    features = encoder(features, ["team", "opp_team", "result", "result_pg", "ground", "ground_pg", "date"])
    ind_feats = features.drop(["result"], axis=1)
    return container(ind_feats, features["result"])



def scale_data():
    pass





### Simple way of collecting data (temp.) 
def load_EPL(encoded : bool = False) -> pd.DataFrame:
    RUN()
    if encoded:
        return translate_data(stats_EPL19_per_game)
    else:
        return stats_EPL19_per_game

### Get latest (W, D, L)-distribution
def get_result_distribution(previous_season : bool = True):
    res = np.ones(3)
    if previous_season:
        # this does not make any sense atm, will fix at some point
        W = understats_EPL18["wins"].sum()
        D = understats_EPL18["draws"].sum()
        L = understats_EPL18["loses"].sum()

        res = np.array([W, D, L])

    return res/np.sum(res)

    
### (NB! Very not general) Get decription of feature codes (xG, ST etc.)
def get_feature_description(md_filename : str = None) -> dict:

    # helper dicts:
    _key_match_features = {
        "ground":       "Home (= h) or away (= a) pitch of team",
        "day":          "Weekday (= 1, .., 7 = Mon., ..., Sun.)",
        "days_rest":    "Number of days the team have been resting from league matches"
    }
    
    _extra_match_stats = {
        "S":    "Shots",
        "ST":   "Shots on target",
        "C":    "Corners won",
        "F":    "Fouls committed",
        "Y":    "Yellow cards recieved",
        "R":    "Red cards recieved"

    }

    _understats = {
        "xG":           "Expected goals",
        "xGA":          "Expected goals against",
        "npxG":         "Expected goals, not counting penalties or own goals",
        "npxGA":        "Expected goals against, not counting penalties or own goals",
        "deep":         "Passes completed within an estimated 20 yards of goal (crosses excluded)",
        "deep_allowed": "Opponent passes completed within an estimated 20 yards of goal (crosses excluded)",
        "scored":       "Goals scored",
        "missed":       "Goals conceded",
        "xpts":         "Expected points",
        "wins":         "Whether the team has won (1) or not (0)",
        "draws":        "Whether the team has drawn (1) or not (0)",
        "loses":        "Whether the team has lost (1) or not (0)",
        "pts":          "League points gained (= 3, 1, 0 for w, d, l)",
        "npxGD":        "The difference between 'for' and 'against' expected goals without penalties and own goals",
        "ppda_coef":    "Passes allowed per defensive action (PPDA) in the opposition half",
        "oppda_coef":   "Opponent passes allowed per defensive action (OPPDA) in the opposition half",
        "xG_diff":      "Difference betweeen xG and actual goals scored",
        "xGA_diff":     "Difference between expected goals against and missed",
        "xpts_diff":    "Difference between actual and expected points"
    }

    _match_understats = {
        "result":       "Full time result (= w, d, l = win, draw, loss) for the team",
        **_understats,
        "ppda_att":     "PPDA attacking actions",
        "ppda_def":     "PPDA defensive actions",
        "oppda_att":    "OPPDA attacking actions",
        "oppda_def":    "OPPDA defensive actions"
    }

    ### Match features (only those that we know before game start)
    match_day = {
        "match_id":     "Match ID",
        "team":         "Full team name",
        "opp_team":     "Oppoent's full team name",
        "date":         "Date of match day (%s)"%DATE_FORMAT,
        **_key_match_features
    }

    ### Team season stats (only those that we know or can assume before the league is ended)
    season = {
        "annual_wages": "Million £ used on player wages",
        "n_contracts":  "Players under contract"
    }

    ### Previous match stats
    prev_match = {
        **_key_match_features,
        **_extra_match_stats,
        **_match_understats
    }

    ### Previous season stats
    prev_season = {
        "position": "League position",
        "matches":  "(omitted) Macthes played",
        **_understats
    }
    
    # create dataframes for easy suffix-fix:
    match_day_pd        = pd.DataFrame.from_dict(match_day, orient="index").transpose()
    season_pd           = pd.DataFrame.from_dict(season, orient="index").transpose()
    prev_match_pd       = pd.DataFrame.from_dict(prev_match, orient="index").transpose().add_suffix("_pg")
    prev_season_pd      = pd.DataFrame.from_dict(prev_season, orient="index").transpose().add_suffix("_ps")
    # FIXME

    ### Previous season stats of opponent
    season_opp_pd       = season_pd.add_suffix("_opp")
    prev_season_opp_pd  = prev_season_pd.add_suffix("_opp")


    # back to dict: (why have I made this so difficult??)
    match_day       = match_day_pd.to_dict()
    season          = season_pd.to_dict()
    prev_match      = prev_match_pd.to_dict()
    prev_season     = prev_season_pd.to_dict()
    season_opp      = season_opp_pd.to_dict()
    prev_season_opp = prev_season_opp_pd.to_dict()

    ### Combine to one large frame/dict
    info = {**match_day, **season, **prev_match, **prev_season, **season_opp, **prev_season_opp}
    nfeatures = len(info)

    if md_filename is not None:

        # much back and forth here, but it works now, so idk 

        header0 = f"# Features in dataset from {LEAGUE} season {YEAR}/{YEAR+1}\n"
        
        tab1 = match_day_pd.transpose()
        tab2 = season_pd.transpose()
        tab3 = prev_match_pd.transpose()
        tab4 = prev_season_pd.transpose()
        tab5 = season_opp_pd.transpose()
        tab6 = prev_season_opp_pd.transpose()

        head1 = "## Match information"
        head2 = f"## Team attributes ({YEAR})"
        head3 = f"## Team's previous league match stats"
        head4 = f"## Team's previous season stats and attributes ({prevYEAR})"
        head5 = f"## Opponent attributes ({YEAR})"
        head6 = f"## Opponent's previous season stats and attributes ({prevYEAR})"

        tables = [tab1, tab2, tab3, tab4, tab5, tab6]
        titles = [head1, head2, head3, head4, head5, head6]

        filename = md_filename.strip(".md") + ".md"
        with open(PATH/filename, "w") as outfile:
            outfile.write(header0 + "\n")
            outfile.write(f"_Total number of features: {nfeatures}_ \n")
            for tab, head in zip(tables, titles):
                tab.columns = ["Description"]
                tab = tab.to_markdown()
                outfile.write(head + "\n" + tab + "\n")
    
    return info



if __name__ == "__main__":

    print("\n>>>\n")
    
    RUN()

    info = get_feature_description("EPL_notes.md")


    data = stats_EPL19_per_game.copy()
    container = translate_data(data)

    cols = container.x.columns
    
    for col in cols:
        print(col)
    print("#features = ", len(cols))
    print(np.size(container.y.to_numpy()))

    # from sklearn.model_selection import train_test_split   

    # trainx, testx, trainy, testy = train_test_split(container.x, container.y)
    # # print(trainx)

    # cols = trainx.columns

    # prevseason = []
    # prevgame = []
    # opp = []
    # other = []
 
    # for col in cols:
    #     col = str(col)
    #     if col.split("_")[-1] == "ps":
    #         prevseason.append(col)
    #     elif col.split("_")[-1] == "pg":
    #         prevgame.append(col)
    #     elif col.split("_")[-1] == "opp":
    #         opp.append(col)
    #     else:
    #         other.append(col)

    # print("This game: \n", other)
    # print("Previous season: \n", prevseason)
    # print("Previous game: \n", prevgame)
    # print("Opponent: \n", opp)
    


    print("\n---\n")



### Data credit:
'''
Basic match data:
    https://www.football-data.co.uk/englandm.php
Team player wages:
    https://fbref.com/en/comps/9/2018-2019/2018-2019-Premier-League-Stats
Match/season metrics: 
    https://www.kaggle.com/datasets/slehkyi/extended-football-stats-for-european-leagues-xg?select=understat_per_game.csv

'''