import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read files
workout_routine = pd.read_excel('Workout_Routine.xlsx')
player_injuries = pd.read_csv('player_injuries_impact.csv')
athlete_recovery = pd.read_csv('Athlete_recovery_dataset.csv')
sleep_data = pd.read_csv('Sleep_Efficiency.csv')

training_vars = [
    'Sleep_Duration', 'Sleep_Score', 'Sleep_Quality', 
    'Soreness', 'Stress', 'Distance', 
    'Acceleration_Count', 'Max_Acceleration', 
    'Deceleration_Count', 'Max_Deceleration'
]

performance_metrics = [
    'Max_Speed', 
    'FIFA rating',  # from player_injuries
    'Recovery_Success',  # from athlete_recovery
    'Confidence_Score'  # from athlete_recovery
]

# Prepare data for analysis
def prepare_regression_data(data, training_vars, target_metric):
    columns_to_use = training_vars + [target_metric]
    subset = data[columns_to_use].copy()
    
    subset = subset.apply(pd.to_numeric, errors='coerce')
    
    subset_clean = subset.dropna()
    
    return subset_clean[training_vars], subset_clean[target_metric]

def analyze_training_impact(data, training_vars, target_metric):
    try:
        X, y = prepare_regression_data(data, training_vars, target_metric)
    except KeyError:
        print(f"Metric {target_metric} not found in the dataset.")
        return None
    

    if len(X) < 10:
        print(f"Not enough data for {target_metric}")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)
    
    # R-squared
    r_squared = reg.score(X_test_scaled, y_test)
    
    # Coefficient DataFrame
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': reg.coef_
    })
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Variable', y='Coefficient', data=coef_df)
    plt.title(f'Impact of Training Variables on {target_metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{target_metric}_impact.png')
    plt.close()
    
    return {
        'coefficients': coef_df,
        'intercept': reg.intercept_,
        'r_squared': r_squared
    }

# Perform analysis
results = {}
datasets = {
    'Workout Routine': workout_routine,
    'Player Injuries': player_injuries,
    'Athlete Recovery': athlete_recovery,
    'Sleep Data': sleep_data
}

for dataset_name, dataset in datasets.items():
    print(f"\nAnalyzing {dataset_name} dataset:")
    for metric in performance_metrics:
        try:
            result = analyze_training_impact(dataset, training_vars, metric)
            if result:
                results[f"{dataset_name} - {metric}"] = result
        except Exception as e:
            print(f"Error analyzing {metric} in {dataset_name}: {e}")

# Print results
print("\nTraining Impact Analysis Results:")
for name, analysis in results.items():
    print(f"\n{name} Impact:")
    print(analysis['coefficients'])
    print(f"R-squared: {analysis['r_squared']:.4f}")

with open('training_impact_results.txt', 'w') as f:
    for name, analysis in results.items():
        f.write(f"{name} Impact:\n")
        f.write(str(analysis['coefficients']) + "\n")
        f.write(f"R-squared: {analysis['r_squared']:.4f}\n\n")

print("\nResults have been saved to training_impact_results.txt")