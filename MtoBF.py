import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

class MtoBF:

    def train(save=False):

        df = pd.read_csv('Data\BodyFat - Extended Edited.csv', sep=',')
        df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})

        X = df.drop(['BodyFat', 'Original'], axis = 1)
        y = df['BodyFat']
        print(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        k = 10
        kf = KFold(n_splits=k, shuffle=True)
        
        '''
        Ridge (l2) regression performs better than standard Linear 
        Regression for out task.

        Below is done to identify the best alpha value (l2 regression hyperparameter)
        through K-fold Cross Validation- the best alpha=1.0
        '''
        
        # # Prepare grid search over alpha values
        # alpha_range = np.logspace(-6, 6, 13)  # alpha values from 1e-6 to 1e6
        # param_grid = {'alpha': alpha_range}

        # # Set up GridSearchCV
        # grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_absolute_error', cv=kf)

        # # Fit grid search on the data
        # grid_search.fit(X, y)

        # # Get the best alpha value and corresponding score
        # best_alpha = grid_search.best_params_['alpha']
        # best_score = grid_search.best_score_

        # print(f"Best Alpha Value: {best_alpha:.4e}")
        # print(f"Best Cross-Validation MSE: {-best_score:.4f}")
        
        '''
        This K-fold Cross Validation is done to see the general loss (ridge)
        and Mean Absolute Error (MAE) over 10 folds.
        '''

        fold_results = {'fold': [], 'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

        with tf.device('/GPU:0'):
            # Loop through folds
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"\nFold {fold + 1}/{k}")
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Define model
                model = Ridge(alpha=1.0)

                # Train model
                model.fit(X_train, y_train)

                # Predict on training and validation sets
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                # Calculate train and validation loss (MSE)
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)

                # Calculate train and validation MAE
                train_mae = mean_absolute_error(y_train, train_pred)
                val_mae = mean_absolute_error(y_val, val_pred)

                fold_results['fold'].append(fold + 1)
                fold_results['train_loss'].append(train_loss)
                fold_results['val_loss'].append(val_loss)
                fold_results['train_mae'].append(train_mae)
                fold_results['val_mae'].append(val_mae)

                print(f"Fold {fold + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                

            avg_train_loss = np.mean(fold_results['train_loss'])
            avg_val_loss = np.mean(fold_results['val_loss'])
            avg_train_mae = np.mean(fold_results['train_mae'])
            avg_val_mae = np.mean(fold_results['val_mae'])

            print("\nCross-Validation Results:")
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            print(f"Average Val Loss: {avg_val_loss:.4f}")
            print(f"Average Train MAE: {avg_train_mae:.4f}")
            print(f"Average Val MAE: {avg_val_mae:.4f}")
            
            '''
            Finally, if 'save=True' we, train a regression 
            model over all the available data and save it.
            '''
            
            if save:
                final_model = Ridge(alpha=1.0)
                final_model.fit(X, y)
                
                joblib.dump(final_model, 'MtoBF.pkl')
                joblib.dump(scaler, 'MtoBF_scaler.pkl')
                print("\nModel saved to 'MtoBF_.pkl'")
                print("\nScaler saved to 'MtoBF_scaler.pkl'")
            else:
                print("---Save Flag set to 'False'---")
                
    def test(test_data):
        
        model = joblib.load('Models/MtoBF.pkl')
        scaler = joblib.load('Models/MtoBF_scaler.pkl')
        
        test_data_scaled = scaler.transform(test_data)
        prediction = model.predict(test_data_scaled)
        
        return prediction
        
        
            
if __name__ == "__main__":
    MtoBF.train(save=False)