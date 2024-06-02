#submission.py 

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from xgboost import XGBClassifier # Use this class
import joblib



def clean_df(df, background_df=None):
    # Selecting variables for modelling
    keepcols = [
        "nomem_encr", "outcome_available", "age_bg","ca20g012", "ca20g013", "ca20g078",
        "cd20m034", "cf20m024", "cf20m025", "cf20m026", "cf20m029", "cf20m030",
        "cf20m128", "cf20m129", "cf20m130", "cf20m166", "cf20m454", "cf20m455",
        "cf20m513", "cf20m514", "cf20m515", "cf20m516", "cf20m517", "cf20m518",
        "cf20m519", "cf20m520", "cf20m521", "ch20m004", "ch20m219",  "ch20m269",
        "ch15h011", "ch16i011", "ch17j011",  "ch18k011", "ch19l011", "ch20m011",
        "cr18k101","cr18k102", "cr18k103", "cr18k104", "cr18k105", "cr20m162", "cv10c135",
        "cv10c136", "cv10c137", "cv10c138", "cv20l109", "cv20l110", "cv20l111",
        "cv20l112", "cv20l113", "cv20l114", "cv20l115", "cv20l124", "cv20l125",
        "cv20l126", "cv20l127", "cv20l128", "cv20l129", "cv20l130", "cv20l143",
        "cv20l144", "cv20l145", "cv20l146", "cv20l151", "cv20l152", "cv20l153",
        "ch07a018", "ch08b018", "ch10d018","ch15h018","ch16i018","ch17j018","ch18k018","ch19l018","ch20m018",
        "cv20l154", "birthyear_bg", "belbezig_2020", "gender_bg", "migration_background_bg",
        'cr08a030','cr09b030','cr10c030','cr11d030', 'cr12e030','cr13f030','cr14g030','cr15h030',
        'cr16i030','cr17j030','cr18k030', 'cr19l030','cr20m030',
        ##partner
        'partner_2007','partner_2008','partner_2009','partner_2010','partner_2011','partner_2012','partner_2013','partner_2014','partner_2015','partner_2016','partner_2017','partner_2018','partner_2019','partner_2020',
        "nettohh_f_2020","nettohh_f_2019", "oplmet_2020", "sted_2020", "woning_2020","cw10c062",
        ##nettoinkomen
        'nettoink_f_2013','nettoink_f_2014','nettoink_f_2015','nettoink_f_2016','nettoink_f_2017','nettoink_f_2018','nettoink_f_2019','nettoink_f_2020',
        #langugage
        'cr08a086','cr09b086','cr10c086','cr11d086','cr12e086','cr13f086','cr14g086','cr15h086','cr16i086','cr17j086','cr18k086','cr19l086','cr20m086', #flemish
        'cr08a085','cr09b085','cr10c085','cr11d085','cr12e085','cr13f085','cr14g085','cr15h085','cr16i085','cr17j085','cr18k085','cr19l085','cr20m085',#turkish
        'cr08a084','cr09b084','cr10c084','cr11d084','cr12e084','cr13f084','cr14g084','cr15h084','cr16i084','cr17j084', 'cr18k084','cr19l084','cr20m084', #indo
        'cr08a083','cr09b083','cr10c083','cr11d083','cr12e083','cr13f083','cr14g083','cr15h083','cr16i083','cr17j083','cr18k083','cr19l083','cr20m083', #fris
        'cr08a080','cr09b080','cr10c080','cr11d080','cr12e080','cr13f080','cr14g080','cr15h080','cr16i080','cr17j080','cr18k080','cr19l080','cr20m080'#arab
        ]

    # Keeping data with selected variables
    df = df[keepcols]

    # Keep only rows with available outcomes
    df = df[df["outcome_available"] == 1].copy()

    # Impute savings with range midpoints and other rules
    df["ca20g012"] = np.where(df["ca20g078"] == 0, 0, df["ca20g012"])
    df["ca20g012"] = np.where(df["ca20g013"] == 1, -1200, df["ca20g012"])
    savings_map = {2: 150, 3: 375, 4: 625, 5: 875, 6: 1750, 7: 3750, 8: 6250,
                   9: 8750, 10: 10750, 11: 12750, 12: 15500, 13: 18500, 14: 22500, 15: 62500}
    df["ca20g012"] = df.apply(lambda row: savings_map.get(row["ca20g013"], row["ca20g012"]), axis=1)
    df["ca20g012"] = np.where(df["ca20g013"] == 999, np.nan, df["ca20g012"])
    df["ca20g012"] = np.where(df["ca20g012"] < -9999999997, np.nan, df["ca20g012"])

    # Apply various transformation rules
    df["cf20m025"] = np.where(df["cf20m024"] == 2, 2, df["cf20m025"])
    df["cf20m030"] = np.where(df["cf20m024"] == 2, 2, df["cf20m030"])
    df["cf20m129"] = np.where(df["cf20m128"] == 2, 0, df["cf20m129"])
    df["cf20m130"] = np.where(df["cf20m128"] == 2, 31, np.where(df["cf20m130"] == 2025, 5, df["cf20m130"]))
    df["cf20m166"] = np.where(df["cf20m166"] == 99, np.nan, df["cf20m166"])
    df["cf20m455"] = np.where(df["cf20m454"] == 2, 0, df["cf20m455"])

    # Scale adjustments and means
    df[["cf20m515", "cf20m516", "cf20m518", "cf20m519", "cf20m520", "cf20m521"]] = 8 - df[["cf20m515", "cf20m516", "cf20m518", "cf20m519", "cf20m520", "cf20m521"]]

    df["child_feeling"] = df[["cf20m513", "cf20m514", "cf20m515", "cf20m516", "cf20m517",
                              "cf20m518", "cf20m519", "cf20m520", "cf20m521"]].mean(axis=1, skipna=True)

    gendered_religiosity_columns = ["cr18k101", "cr18k102", "cr18k103", "cr18k104", "cr18k105"]
    df[gendered_religiosity_columns] = df[gendered_religiosity_columns].applymap(lambda x: 3 if x == 1 else (1 if x == 2 else 2))
    df["cr18k102"] = 4 - df["cr18k102"]
    df["cr18k105"] = 4 - df["cr18k105"]
    df["gendered_religiosity"] = df[gendered_religiosity_columns].mean(axis=1, skipna=True)

    df["cr20m162"] = np.where(df["cr20m162"] == -9, np.nan, df["cr20m162"])

    df["ch20m269"] = df["ch20m269"].fillna(0)

    traditional_fertility_columns = ["cv10c135", "cv10c136", "cv10c137", "cv10c138"]
    df["traditional_fertility"] = df[traditional_fertility_columns].mean(axis=1, skipna=True)

    df["cv20l109"] = 6 - df["cv20l109"]
    traditional_motherhood_columns = ["cv20l109", "cv20l110", "cv20l111"]
    df["traditional_motherhood"] = df[traditional_motherhood_columns].mean(axis=1, skipna=True)

    traditional_fatherhood_columns = ["cv20l112", "cv20l113", "cv20l114", "cv20l115"]
    df[["cv20l112", "cv20l114", "cv20l115"]] = 6 - df[["cv20l112", "cv20l114", "cv20l115"]]
    df["traditional_fatherhood"] = df[traditional_fatherhood_columns].mean(axis=1, skipna=True)

    traditional_marriage_columns = ["cv20l124", "cv20l125", "cv20l126", "cv20l127", "cv20l128", "cv20l129", "cv20l130"]
    df[["cv20l126", "cv20l127", "cv20l128", "cv20l129", "cv20l130"]] = 6 - df[["cv20l126", "cv20l127", "cv20l128", "cv20l129", "cv20l130"]]
    df["traditional_marriage"] = df[traditional_marriage_columns].mean(axis=1, skipna=True)

    working_mother_columns = ["cv20l143", "cv20l144", "cv20l145", "cv20l146"]
    df["working_mother"] = df[working_mother_columns].mean(axis=1, skipna=True)

    sexism_columns = ["cv20l151", "cv20l152", "cv20l153", "cv20l154"]
    df["sexism"] = df[sexism_columns].mean(axis=1, skipna=True)

    anxiety_columns = ["ch15h011", "ch16i011", "ch17j011",  "ch18k011", "ch19l011", "ch20m011"]
    df['anxiety'] = df[anxiety_columns].mean(axis =1, skipna= True)

    long_standing = ["ch07a018", "ch08b018", "ch10d018","ch15h018","ch16i018","ch17j018","ch18k018","ch19l018","ch20m018"]
    df['long_standing'] = df[long_standing].mean(axis = 1, skipna= True) 

    god_belief = ['cr08a030','cr09b030','cr10c030','cr11d030','cr12e030','cr13f030','cr14g030','cr15h030','cr16i030','cr17j030','cr18k030','cr19l030','cr20m030']
    df['god'] = df[god_belief].mean(axis = 1, skipna= True) 

    netto_columns = ['nettoink_f_2013','nettoink_f_2014','nettoink_f_2015','nettoink_f_2016','nettoink_f_2017','nettoink_f_2018','nettoink_f_2019','nettoink_f_2020']
    df['netto'] = df[netto_columns].mean(axis = 1, skipna= True)

    language_columns = ['cr08a086','cr09b086','cr10c086','cr11d086','cr12e086','cr13f086','cr14g086','cr15h086','cr16i086','cr17j086','cr18k086','cr19l086','cr20m086', #flemish
        'cr08a085','cr09b085','cr10c085','cr11d085','cr12e085','cr13f085','cr14g085','cr15h085','cr16i085','cr17j085','cr18k085','cr19l085','cr20m085',#turkish
        'cr08a084','cr09b084','cr10c084','cr11d084','cr12e084','cr13f084','cr14g084','cr15h084','cr16i084','cr17j084', 'cr18k084','cr19l084','cr20m084', #indo
        'cr08a083','cr09b083','cr10c083','cr11d083','cr12e083','cr13f083','cr14g083','cr15h083','cr16i083','cr17j083','cr18k083','cr19l083','cr20m083', #fris
        'cr08a080','cr09b080','cr10c080','cr11d080','cr12e080','cr13f080','cr14g080','cr15h080','cr16i080','cr17j080','cr18k080','cr19l080','cr20m080']
    
    language = df[language_columns].mean(axis=1)

    language[language>0]=1

    language = language.fillna(1)

    df['language'] = language


    partner_columns =['partner_2007','partner_2008','partner_2009','partner_2010','partner_2011','partner_2012','partner_2013','partner_2014','partner_2015','partner_2016','partner_2017','partner_2018','partner_2019','partner_2020']

    partner = df[partner_columns].mean(axis=1)

    partner[partner>0]=1

    partner = partner.fillna(0)
    
    df['partner']=partner
    
    df["employee"] = np.where(df["belbezig_2020"] == 1, 1, 0)
    df["freelance"] = np.where(df["belbezig_2020"] == 3, 1, 0)
    df["student"] = np.where(df["belbezig_2020"] == 7, 1, 0)
    df["homemaker"] = np.where(df["belbezig_2020"] == 8, 1, 0)

    df["migration_background_bg"] = df["migration_background_bg"].apply(lambda x: 1 if x in [0, 101, 201] else (0 if x in [102, 202] else x))

    df["oplmet_2020"] = df["oplmet_2020"].apply(lambda x: 2 if x in [1, 2, 8, 9] else (np.nan if x == 7 else x))

    df["woning_2020"] = df["woning_2020"].apply(lambda x: 1 if x == 1 else (np.nan if x == 0 else 0))

    # Drop unnecessary columns
    dropcols = ["outcome_available", "ca20g078", "ca20g013", "cf20m128", "cf20m454",
                "cf20m513", "cf20m514", "cf20m515", "cf20m516", "cf20m517", "cf20m518",
                "cf20m519", "cf20m520", "cf20m521", "cr18k101", "cr18k102", "cr18k103",
                "cr18k104", "cr18k105", "cv10c135", "cv10c136", "cv10c137", "cv10c138",
                "cv20l109", "cv20l110", "cv20l111", "cv20l112", "cv20l113", "cv20l114",
                "cv20l115", "cv20l124", "cv20l125", "cv20l126", "cv20l127", "cv20l128",
                "cv20l129", "cv20l130", "cv20l143", "cv20l144", "cv20l145", "cv20l146",
                "ch15h011", "ch16i011", "ch17j011",  "ch18k011", "ch19l011", "ch20m011",
                "cv20l151", "cv20l152", "cv20l153", "cv20l154", "belbezig_2020", 
                "ch07a018", "ch08b018", "ch10d018","ch15h018","ch16i018","ch17j018","ch18k018","ch19l018","ch20m018",
                "migration_background_bg",'cr08a030','cr09b030','cr10c030','cr11d030','cr12e030','cr13f030','cr14g030',
                'cr15h030', 'cr16i030','cr17j030','cr18k030','cr19l030','cr20m030',
               'partner_2007','partner_2008','partner_2009','partner_2010','partner_2011','partner_2012','partner_2013','partner_2014','partner_2015','partner_2016','partner_2017','partner_2018','partner_2019','partner_2020',#partner
                'cr08a086','cr09b086','cr10c086','cr11d086','cr12e086','cr13f086','cr14g086','cr15h086','cr16i086','cr17j086','cr18k086','cr19l086','cr20m086', #flemish
        'cr08a085','cr09b085','cr10c085','cr11d085','cr12e085','cr13f085','cr14g085','cr15h085','cr16i085','cr17j085','cr18k085','cr19l085','cr20m085',#turkish
        'cr08a084','cr09b084','cr10c084','cr11d084','cr12e084','cr13f084','cr14g084','cr15h084','cr16i084','cr17j084', 'cr18k084','cr19l084','cr20m084', #indo
        'cr08a083','cr09b083','cr10c083','cr11d083','cr12e083','cr13f083','cr14g083','cr15h083','cr16i083','cr17j083','cr18k083','cr19l083','cr20m083', #fris
        'cr08a080','cr09b080','cr10c080','cr11d080','cr12e080','cr13f080','cr14g080','cr15h080','cr16i080','cr17j080','cr18k080','cr19l080','cr20m080',#arab
        'nettoink_f_2013','nettoink_f_2014','nettoink_f_2015','nettoink_f_2016','nettoink_f_2017','nettoink_f_2018','nettoink_f_2019','nettoink_f_2020'#netto
        ]
    
    df.drop(columns=dropcols, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df['oplmet_2020'] = df['oplmet_2020'].astype('category')

    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    imputed_data = pd.DataFrame(imputer.fit_transform(df))
    imputed_data.columns =  df.columns
    return imputed_data




def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load('model.joblib')

    # Preprocess the fake / holdout data
    #df = clean_df(df, background_df)

    df = clean_df(df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    #vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    #predictions = model.predict(df[vars_without_id])

    nomem_encr = df['nomem_encr']

    df.drop(['nomem_encr'], axis=1,inplace=True)

    predictions = model.predict(df)

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": nomem_encr, "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict

