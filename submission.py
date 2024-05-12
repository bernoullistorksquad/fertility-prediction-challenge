#submission.py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from category_encoders import CatBoostEncoder   
from sklearn.ensemble import AdaBoostClassifier
import joblib
# load PreFer_train_data.csv (may take a while)
#PreFer_train_data = pd.read_csv('~/Desktop/Pre-Fer/5e8ab08c-b634-4948-8b47-8792d36d753f/training_data/PreFer_train_data.csv')

# load codebook
#PreFer_codebook = pd.read_csv('~/Desktop/Pre-Fer/5e8ab08c-b634-4948-8b47-8792d36d753f/codebooks/PreFer_codebook.csv')

# load PreFer_train_outcome.csv
#PreFer_train_outcome = pd.read_csv('~/Desktop/Pre-Fer/5e8ab08c-b634-4948-8b47-8792d36d753f/training_data/PreFer_train_outcome.csv')

#PreFer_fake_data = pd.read_csv('~/Desktop/Pre-Fer/5e8ab08c-b634-4948-8b47-8792d36d753f/other_data/PreFer_fake_data.csv')

#PreFer_fake_outcome = pd.read_csv('~/Desktop/Pre-Fer/5e8ab08c-b634-4948-8b47-8792d36d753f/other_data/PreFer_fake_outcome.csv')

def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    # selecting the important variables 

    keep_these = ['nomem_encr', 'outcome_available', 'cf14g_m', 'cf17j_m', 'cf14g001', 'cf17j001', 'cf14g003', 
                  'cf17j003', 'cf14g004', 'cf17j004', 'cf14g024', 'cf17j024', 'cf14g128', 'cf14g388', 'cf17j388', 'cf14g389', 'cf17j389', 'cf14g390', 
                  'cf17j390', 'cf14g391', 'cf17j391', 'cf14g392', 'cf17j392',  'cf14g394',  
                  'cf14g396', 'cf14g397', 'cf17j397', 'cf14g432', 'cf17j432', 'cf17j454', 'cf17j471', 'birthyear_bg', 'gender_bg', 
                  'migration_background_bg', 'age_bg', 'belbezig_2007', 'belbezig_2008', 'belbezig_2009', 'belbezig_2010', 'belbezig_2011', 'belbezig_2012', 
                  'belbezig_2013', 'belbezig_2014', 'belbezig_2015', 'belbezig_2016', 'belbezig_2017', 'belbezig_2018', 'belbezig_2019', 'belbezig_2020', 
                  'brutohh_f_2008', 'brutohh_f_2009', 'brutohh_f_2010', 'brutohh_f_2011', 'brutohh_f_2012', 'brutohh_f_2013', 'brutohh_f_2014', 'brutohh_f_2015',
                  'brutohh_f_2016', 'brutohh_f_2017', 'brutohh_f_2018', 'brutohh_f_2019', 'brutohh_f_2020', 'brutoink_2009', 'brutoink_2014', 'brutoink_f_2008', 
                  'brutoink_f_2009', 'brutoink_f_2010', 'brutoink_f_2011', 'brutoink_f_2012', 'brutoink_f_2013', 'brutoink_f_2014', 'brutoink_f_2015', 
                  'brutoink_f_2016', 'brutoink_f_2017', 'brutoink_f_2018', 'brutoink_f_2019', 'brutoink_f_2020', 'burgstat_2007', 'burgstat_2008', 
                  'burgstat_2009', 'burgstat_2010', 'burgstat_2011', 'burgstat_2012', 'burgstat_2013', 'burgstat_2014', 'burgstat_2015', 'burgstat_2016', 
                  'burgstat_2017', 'burgstat_2018', 'burgstat_2019', 'burgstat_2020', 'netinc_2008', 'netinc_2009', 'netinc_2010', 'netinc_2011', 'netinc_2012', 
                  'netinc_2013', 'netinc_2014', 'netinc_2015', 'netinc_2016', 'netinc_2017', 'netinc_2018', 'netinc_2019', 'netinc_2020', 'nettohh_f_2008', 
                  'nettohh_f_2009', 'nettohh_f_2010', 'nettohh_f_2011', 'nettohh_f_2012', 'nettohh_f_2013', 'nettohh_f_2014', 'nettohh_f_2015', 'nettohh_f_2016', 
                  'nettohh_f_2017', 'nettohh_f_2018', 'nettohh_f_2019', 'nettohh_f_2020', 'nettoink_2007', 'nettoink_2008', 'nettoink_2009', 'nettoink_2010', 
                  'nettoink_2011', 'nettoink_2012', 'nettoink_2013', 'nettoink_2014', 'nettoink_2015', 'nettoink_2016', 'nettoink_2017', 'nettoink_2018',
                  'nettoink_2019', 'nettoink_2020', 'nettoink_f_2008', 'nettoink_f_2009', 'nettoink_f_2010', 'nettoink_f_2011', 'nettoink_f_2012', 
                  'nettoink_f_2013', 'nettoink_f_2014', 'nettoink_f_2015', 'nettoink_f_2016', 'nettoink_f_2017', 'nettoink_f_2018', 'nettoink_f_2019', 
                  'nettoink_f_2020', 'oplcat_2007', 'oplcat_2008', 'oplcat_2009', 'oplcat_2010', 'oplcat_2011', 'oplcat_2012', 'oplcat_2013', 'oplcat_2014',
                  'oplcat_2015', 'oplcat_2016', 'oplcat_2017', 'oplcat_2018', 'oplcat_2019', 'oplcat_2020', 'oplmet_2007', 'oplmet_2008', 'oplmet_2009', 
                  'oplmet_2010', 'oplmet_2011', 'oplmet_2012', 'oplmet_2013', 'oplmet_2014', 'oplmet_2015', 'oplmet_2016', 'oplmet_2017', 'oplmet_2018', 
                  'oplmet_2019', 'oplmet_2020', 'oplzon_2007', 'oplzon_2008', 'oplzon_2009', 'oplzon_2010', 'oplzon_2011', 'oplzon_2012', 'oplzon_2013', 
                  'oplzon_2014', 'oplzon_2015', 'oplzon_2016', 'oplzon_2017', 'oplzon_2018', 'oplzon_2019', 'oplzon_2020', 'partner_2007', 'partner_2008', 
                  'partner_2009', 'partner_2010', 'partner_2011', 'partner_2012', 'partner_2013', 'partner_2014', 'partner_2015', 'partner_2016', 'partner_2017', 'partner_2018', 
                  'partner_2019', 'partner_2020', 'sted_2008', 'sted_2009', 'sted_2010', 'sted_2011', 'sted_2012', 'sted_2013', 'sted_2014', 'sted_2015', 'sted_2016', 'sted_2017', 
                  'sted_2018', 'sted_2019', 'sted_2020', 'woning_2007', 'woning_2008', 'woning_2009', 'woning_2010', 'woning_2011', 'woning_2012', 'woning_2013', 'woning_2014', 
                  'woning_2015', 'woning_2016', 'woning_2017', 'woning_2018', 'woning_2019', 'woning_2020', 'woonvorm_2007', 'woonvorm_2008', 'woonvorm_2009', 'woonvorm_2010', 
                  'woonvorm_2011', 'woonvorm_2012', 'woonvorm_2013', 'woonvorm_2014', 'woonvorm_2015', 'woonvorm_2016', 'woonvorm_2017', 'woonvorm_2018', 'woonvorm_2019', 
                  'woonvorm_2020']

    df = df[keep_these]

    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    imputed_df = pd.DataFrame(imputer.fit_transform(df))
    imputed_df.columns = df.columns

    ## This script contains a bare minimum working example
    # Create new variable with age
    #df["age"] = 2024 - df["birthyear_bg"]

    # Imputing missing values in age with the mean
#    df["age"] = df["age"].fillna(df["age"].mean())

    # Selecting variables for modelling
    #keepcols = [
    #    "nomem_encr",  # ID variable required for predictions,
    #    "age"         # newly created variable
    #    ,"gender_bg"  # <--------ADDED VARIABLE
    #] 

    # Keeping data with variables selected
    #df = df[keepcols]

    return imputed_df


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
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    #vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    #predictions = model.predict(df[vars_without_id])

    nomem_encr = df['nomem_encr']
    df = df.drop(['nomem_encr','outcome_available','birthyear_bg'], axis=1)
    predictions = model.predict(df)

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": nomem_encr, "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict

