import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, average_precision_score

# Function to calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of the Earth in kilometers. You can use 3956 for miles.

    # Distance in kilometers
    return c * r

def clean_data(df):
    cat = ['cc_num', 'trans_date_trans_time','category', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    
    X = df[cat].copy()
    y = df['is_fraud']
    
    # Convert 'trans_date_trans_time' to pandas datetime object
    X['trans_date_trans_time'] = pd.to_datetime(X['trans_date_trans_time'])
    
    # Create a new binary variable for AM (1 for morning, 0 for afternoon/evening)
    X['is_AM'] = (X['trans_date_trans_time'].dt.hour < 12).astype(int)
    
    # Create a new column for working hours vs. non-working hours (1 for working hours, 0 for non-working hours)
    X['is_working_hours'] = X['trans_date_trans_time'].apply(lambda x: 1 if 9 <= x.hour < 17 else 0)
    
    # Create a new column for the days of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday)
    X['day_of_week'] = X['trans_date_trans_time'].dt.dayofweek
    
    # Store the original index
    original_index = X.index
    
    # Sort the DataFrame by 'cc_num' and 'trans_date_trans_time' to ensure transactions are in chronological order
    X.sort_values(by=['cc_num', 'trans_date_trans_time'], inplace=True)
    
    # Calculate the time difference between consecutive transactions for each 'cc_num'
    X['time_since_prev_transaction'] = X.groupby('cc_num')['trans_date_trans_time'].diff()
    
    # Convert the time difference to seconds for better representation
    X['time_since_prev_transaction'] = X['time_since_prev_transaction'].dt.total_seconds()
    
    # Fill any missing time differences with 0 (for the first transaction of each 'cc_num')
    X['time_since_prev_transaction'].fillna(-1, inplace=True)
    
    # Sort back to the original order using the original index
    X = X.loc[original_index]
    
    # Create a new column for the sine and cosine transformations of the day_of_week
    X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
    X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
    
    # Calculate the distance and add a new column 'distance'
    X['distance'] = haversine(X['lat'], X['long'], X['merch_lat'], X['merch_long'])
    
    # Drop the original categories
    X.drop(['cc_num', 'lat', 'long', 'merch_lat', 'merch_long', 'trans_date_trans_time', 'day_of_week'], axis=1, inplace=True)
    
    # Perform one-hot encoding on the 'category' column
    X_encoded = pd.get_dummies(X, columns=['category'])
    
    return X_encoded, y

def KFold(model, cw, X, y):
    
    # Perform cross-validation predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
    
    cf = classification_report(y, y_pred)
    print("==============================")
    print(f"Class weights: {cw}")
    print(cf)
    
def predict(y_pred, class_weights, model, X_test, y_test, idx):
    
    # Calculate recall, precision, F1
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Predict probabilities for positive class (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_prob)

    # Find all fraud transactions and sum their corresponding 'amt' values
    fraud_amt = X_test.loc[(y_test == 1), 'amt'].sum()
    
    # Find true positives and sum their corresponding 'amt' values
    detected_amt = X_test.loc[(y_test == 1) & (y_pred == 1), 'amt'].sum()
    
    # Find false positives and count them
    fp_count = sum((y_test == 0) & (y_pred == 1))
    
    # Calculate the total cost
    total_cost_fp1 = 75 * sum(y_pred == 1) # Overhead for every detection
    total_cost_fp2 = X_test.loc[(y_test == 0) & (y_pred == 1), 'amt'].sum() * 0.0175  # Processing fees lost due to false detection

    total_cost_fp = 0.5 * (total_cost_fp1 + total_cost_fp2)
    
    # Calculate the total savings
    total_savings = detected_amt - total_cost_fp

    print("==============================")
    try:
        print("Class weight:", class_weights[idx])
    except:
        pass
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 score:", f1)
    print("AUPRC:", auprc)
    print("Total Fraud Amount:", fraud_amt)
    print("Total Detected Fraud Amount:", detected_amt)
    print("Total Cost of Detection:", total_cost_fp)
    print("Total Savings:", total_savings)
    print("Total False Detections:", fp_count)
    print()
    
    return total_savings

def feature_importance(model, X):
    # Get feature importances and create a DataFrame to store them
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    
    # Sort the DataFrame in descending order of importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print(importance_df)