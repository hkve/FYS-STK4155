import pandas as pd
import pathlib as pl

def load_fifa(filename="players_21.csv", n=None, random_state=321):
    path = pl.Path(__file__).parent / filename
    assert path.exists(), f"Missing {filename = } at {path.parent = }"
    df = pd.read_csv(path)

    info_cols = [
        "short_name",
        "overall"
    ]

    attr_cols = [
        "attacking_crossing",
        "attacking_finishing",
        "attacking_heading_accuracy",
        "attacking_short_passing",
        "attacking_volleys",
        "skill_dribbling",
        "skill_curve",
        "skill_fk_accuracy",
        "skill_long_passing",
        "skill_ball_control",
        "movement_acceleration",
        "movement_sprint_speed",
        "movement_agility",
        "movement_reactions",
        "movement_balance",
        "power_shot_power",
        "power_jumping",
        "power_stamina",
        "power_strength",
        "power_long_shots",
        "mentality_aggression",
        "mentality_interceptions",
        "mentality_positioning",
        "mentality_vision",
        "mentality_penalties",
        "mentality_composure",
        "defending_standing_tackle",
        "defending_sliding_tackle",
        "goalkeeping_diving",
        "goalkeeping_handling",
        "goalkeeping_kicking",
        "goalkeeping_positioning",
        "goalkeeping_reflexes"
    ]

    mics_cols = [
        "international_reputation"
    ]

    cols = info_cols + attr_cols + mics_cols
    df = df[cols].dropna(axis=0)

    if not n or n > len(df): n = len(df)

    df["freq"] = 1./df.groupby('international_reputation')['international_reputation'].transform('count')
    df = df.sample(n=n, weights=df["freq"], random_state=random_state).drop(["freq"], axis=1)

    return df