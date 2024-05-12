#training.py
def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    #random.seed(1) # not useful here because logistic regression deterministic
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df=  model_df[~model_df['new_child'].isna()] 

    #Drop the useless variables 
    model_df.drop(['outcome_available', 'nomem_encr','birthyear_bg'], axis=1, inplace=True)
    #model_df = model_df[~model_df['new_child'].isna()]  
    y = model_df['new_child']
    model_df.drop('new_child',axis=1, inplace=True)# Your features here

    for cols in model_df.columns:
        model_df[cols] = model_df[cols].astype('category',copy=False)

    encoder = CatBoostEncoder(random_state=1915743)# Complete this

    encoder.fit(model_df,y)# Fit the encoder

    X = encoder.transform(model_df) # Do not change this

    # Logistic regression model
    model = AdaBoostClassifier(n_estimators=300, random_state=42,learning_rate = 0.96)

    model.fit(X,y)
    # Fit the model
    #model.fit(model_df[['age', 'gender_bg']], model_df['new_child']) # <------- ADDED VARIABLE

    # Save the model
    joblib.dump(model, "model.joblib")

