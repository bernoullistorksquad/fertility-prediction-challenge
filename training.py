#training.py
def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """

    cleaned_df = clean_df(cleaned_df)
    
    ## This script contains a bare minimum working example
    #random.seed(1) # not useful here because logistic regression deterministic
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df=  model_df[~model_df['new_child'].isna()] 

    #Drop the useless variables 
    model_df.drop(['nomem_encr'], axis=1, inplace=True)
    #model_df = model_df[~model_df['new_child'].isna()]  
    y = model_df['new_child']

    model_df.drop('new_child',axis=1, inplace=True)# Your features here

    xgb_optimization_parameters = {'learning_rate':  uniform(0.01,1), 
                               'n_estimators': range(100,200,10),
                               'min_child_weight': uniform(0,8),
                               'eta': uniform(0.01,0.5), 
                               'max_depth':range(1,17,2),
                               'gamma': uniform(0,10),
                               'colsample_bylevel':uniform(0,1),
                               'subsample': uniform(0.01,1),
                               'reg_alpha': uniform(0,1)}  


    xgb_searcher = RandomizedSearchCV(XGBClassifier(random_state = 1915743), xgb_optimization_parameters, random_state=42,n_iter=50) # Complete this


    xgb_search = xgb_searcher.fit(model_df,y)

    xgbclassifier = xgb_search.best_estimator_ 

    model = xgbclassifier 

    model.fit(model_df,y)
    # Fit the model
    #model.fit(model_df[['age', 'gender_bg']], model_df['new_child']) # <------- ADDED VARIABLE

    # Save the model
    joblib.dump(model, "model.joblib")
