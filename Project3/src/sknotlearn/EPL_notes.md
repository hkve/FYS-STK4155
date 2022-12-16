# Features in dataset from EPL season 2019/2020

_Total number of features: 86_ 
## Match information
|           | Description                                                   |
|:----------|:--------------------------------------------------------------|
| match_id  | Match ID                                                      |
| team      | Full team name                                                |
| opp_team  | Oppoent's full team name                                      |
| date      | Date of match day (%Y-%m-%d)                                  |
| ground    | Home (= h) or away (= a) pitch of team                        |
| day       | Weekday (= 1, .., 7 = Mon., ..., Sun.)                        |
| days_rest | Number of days the team have been resting from league matches |
## Team attributes (2019)
|              | Description                    |
|:-------------|:-------------------------------|
| annual_wages | Million £ used on player wages |
| n_contracts  | Players under contract         |
## Team's previous league match stats
|                 | Description                                                                               |
|:----------------|:------------------------------------------------------------------------------------------|
| ground_pg       | Home (= h) or away (= a) pitch of team                                                    |
| day_pg          | Weekday (= 1, .., 7 = Mon., ..., Sun.)                                                    |
| days_rest_pg    | Number of days the team have been resting from league matches                             |
| S_pg            | Shots                                                                                     |
| ST_pg           | Shots on target                                                                           |
| C_pg            | Corners won                                                                               |
| F_pg            | Fouls committed                                                                           |
| Y_pg            | Yellow cards recieved                                                                     |
| R_pg            | Red cards recieved                                                                        |
| result_pg       | Full time result (= w, d, l = win, draw, loss) for the team                               |
| xG_pg           | Expected goals                                                                            |
| xGA_pg          | Expected goals against                                                                    |
| npxG_pg         | Expected goals, not counting penalties or own goals                                       |
| npxGA_pg        | Expected goals against, not counting penalties or own goals                               |
| deep_pg         | Passes completed within an estimated 20 yards of goal (crosses excluded)                  |
| deep_allowed_pg | Opponent passes completed within an estimated 20 yards of goal (crosses excluded)         |
| scored_pg       | Goals scored                                                                              |
| missed_pg       | Goals conceded                                                                            |
| xpts_pg         | Expected points                                                                           |
| wins_pg         | Whether the team has won (1) or not (0)                                                   |
| draws_pg        | Whether the team has drawn (1) or not (0)                                                 |
| loses_pg        | Whether the team has lost (1) or not (0)                                                  |
| pts_pg          | League points gained (= 3, 1, 0 for w, d, l)                                              |
| npxGD_pg        | The difference between 'for' and 'against' expected goals without penalties and own goals |
| ppda_coef_pg    | Passes allowed per defensive action (PPDA) in the opposition half                         |
| oppda_coef_pg   | Opponent passes allowed per defensive action (OPPDA) in the opposition half               |
| xG_diff_pg      | Difference betweeen xG and actual goals scored                                            |
| xGA_diff_pg     | Difference between expected goals against and missed                                      |
| xpts_diff_pg    | Difference between actual and expected points                                             |
| ppda_att_pg     | PPDA attacking actions                                                                    |
| ppda_def_pg     | PPDA defensive actions                                                                    |
| oppda_att_pg    | OPPDA attacking actions                                                                   |
| oppda_def_pg    | OPPDA defensive actions                                                                   |
## Team's previous season stats and attributes (2018)
|                 | Description                                                                               |
|:----------------|:------------------------------------------------------------------------------------------|
| position_ps     | League position                                                                           |
| matches_ps      | (omitted) Macthes played                                                                  |
| xG_ps           | Expected goals                                                                            |
| xGA_ps          | Expected goals against                                                                    |
| npxG_ps         | Expected goals, not counting penalties or own goals                                       |
| npxGA_ps        | Expected goals against, not counting penalties or own goals                               |
| deep_ps         | Passes completed within an estimated 20 yards of goal (crosses excluded)                  |
| deep_allowed_ps | Opponent passes completed within an estimated 20 yards of goal (crosses excluded)         |
| scored_ps       | Goals scored                                                                              |
| missed_ps       | Goals conceded                                                                            |
| xpts_ps         | Expected points                                                                           |
| wins_ps         | Whether the team has won (1) or not (0)                                                   |
| draws_ps        | Whether the team has drawn (1) or not (0)                                                 |
| loses_ps        | Whether the team has lost (1) or not (0)                                                  |
| pts_ps          | League points gained (= 3, 1, 0 for w, d, l)                                              |
| npxGD_ps        | The difference between 'for' and 'against' expected goals without penalties and own goals |
| ppda_coef_ps    | Passes allowed per defensive action (PPDA) in the opposition half                         |
| oppda_coef_ps   | Opponent passes allowed per defensive action (OPPDA) in the opposition half               |
| xG_diff_ps      | Difference betweeen xG and actual goals scored                                            |
| xGA_diff_ps     | Difference between expected goals against and missed                                      |
| xpts_diff_ps    | Difference between actual and expected points                                             |
## Opponent attributes (2019)
|                  | Description                    |
|:-----------------|:-------------------------------|
| annual_wages_opp | Million £ used on player wages |
| n_contracts_opp  | Players under contract         |
## Opponent's previous season stats and attributes (2018)
|                     | Description                                                                               |
|:--------------------|:------------------------------------------------------------------------------------------|
| position_ps_opp     | League position                                                                           |
| matches_ps_opp      | (omitted) Macthes played                                                                  |
| xG_ps_opp           | Expected goals                                                                            |
| xGA_ps_opp          | Expected goals against                                                                    |
| npxG_ps_opp         | Expected goals, not counting penalties or own goals                                       |
| npxGA_ps_opp        | Expected goals against, not counting penalties or own goals                               |
| deep_ps_opp         | Passes completed within an estimated 20 yards of goal (crosses excluded)                  |
| deep_allowed_ps_opp | Opponent passes completed within an estimated 20 yards of goal (crosses excluded)         |
| scored_ps_opp       | Goals scored                                                                              |
| missed_ps_opp       | Goals conceded                                                                            |
| xpts_ps_opp         | Expected points                                                                           |
| wins_ps_opp         | Whether the team has won (1) or not (0)                                                   |
| draws_ps_opp        | Whether the team has drawn (1) or not (0)                                                 |
| loses_ps_opp        | Whether the team has lost (1) or not (0)                                                  |
| pts_ps_opp          | League points gained (= 3, 1, 0 for w, d, l)                                              |
| npxGD_ps_opp        | The difference between 'for' and 'against' expected goals without penalties and own goals |
| ppda_coef_ps_opp    | Passes allowed per defensive action (PPDA) in the opposition half                         |
| oppda_coef_ps_opp   | Opponent passes allowed per defensive action (OPPDA) in the opposition half               |
| xG_diff_ps_opp      | Difference betweeen xG and actual goals scored                                            |
| xGA_diff_ps_opp     | Difference between expected goals against and missed                                      |
| xpts_diff_ps_opp    | Difference between actual and expected points                                             |
