import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def generate_realistic_performance_data():
    """
    Generate a more physiologically realistic dataset for athlete performance
    with improved predictive relationships for goals, heart rate, and sleep quality.
    """
    np.random.seed(42)
    
    # Sample size
    n = 500
    
    # Constrain training parameters
    training_frequency = np.random.uniform(1, 7, n)  # Days per week
    training_hours = np.clip(training_frequency * np.random.uniform(1, 5, n), 1, 20)  # Hours per week
    training_intensity = np.random.uniform(1, 10, n)  # Intensity 1-10
    
    # Goals per game - adjusted to have a base of 0.5 and realistic range (0 to 4)
    goals = np.clip(
        0.5 +
        0.2 * training_intensity +         
        0.1 * (training_hours / 20) +        
        0.1 * (training_frequency / 7) +     
        np.random.normal(0, 0.1, n),         
        0, 4
    )
    
    speed = np.clip(
        28 +
        0.3 * training_intensity +
        0.2 * (training_hours / 20) +
        np.random.normal(0, 1, n),
        28, 35
    )
    
    heart_rate = np.clip(
        80 +
        0.2 * training_intensity +           
        0.1 * (training_hours / 20) +        
        0.1 * training_frequency +           
        np.random.normal(0, 1, n),
        80, 110
    )
    
    fatigue_level = np.clip(
        5 +
        0.4 * training_intensity +
        0.3 * (training_hours / 20) +
        np.random.normal(0, 0.5, n),
        1, 10
    )

    strength = np.clip(
        5 +
        0.2 * training_intensity +
        0.15 * (training_hours / 20) +
        np.random.normal(0, 0.4, n),
        1, 10
    )
    

    sleep_quality = np.clip(
        6 +
        -0.2 * training_intensity +          
        0.2 * (training_hours / 20) +          
        0.1 * (training_frequency / 7) +       
        np.random.normal(0, 0.3, n),
        1, 10
    )
    
    recovery_rate = np.clip(
        6 +
        0.2 * training_intensity +
        0.15 * (training_hours / 20) +
        np.random.normal(0, 0.4, n),
        1, 10
    )
    
    injury_risk = np.clip(
        30 +
        1.5 * training_intensity +
        1 * (training_hours / 20) +
        np.random.normal(0, 2, n),
        5, 80
    )
    
    performance_score = np.clip(
        50 +
        1.5 * speed +
        1.2 * strength +
        1 * recovery_rate +
        -0.4 * injury_risk +
        -0.3 * fatigue_level +
        np.random.normal(0, 4, n),
        0, 100
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'training_hours': training_hours,
        'training_intensity': training_intensity,
        'training_frequency': training_frequency,
        'goals_per_game': goals,
        'speed_km_h': speed,
        'heart_rate': heart_rate,
        'strength': strength,
        'sleep_quality': sleep_quality,
        'recovery_rate': recovery_rate,
        'injury_risk_percentage': injury_risk,
        'fatigue_level': fatigue_level,
        'performance_score': performance_score
    })
    
    return df

def generate_performance_analysis(df):
    """
    Generate performance analysis with regression line visualizations
    """
    # Performance metrics to analyze
    performance_metrics = [
        'goals_per_game', 
        'speed_km_h', 
        'heart_rate',
        'strength', 
        'sleep_quality', 
        'recovery_rate', 
        'injury_risk_percentage',
        'fatigue_level',
        'performance_score'
    ]
    
    # Training variables
    training_vars = ['training_hours', 'training_intensity', 'training_frequency']
    
    equations = {}
    
    plt.figure(figsize=(25, 30))
    
    for i, metric in enumerate(performance_metrics, 1):
        X = df[training_vars]
        y = df[metric]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Linear Regression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Calculate R-squared
        r_squared = model.score(X_test_scaled, y_test)
        
        # Generate equation
        feature_names = training_vars
        equation = f"{metric} = {model.intercept_:.2f}"
        for j, var in enumerate(feature_names):
            equation += f" + {model.coef_[j]:.2f} * normalized({var})"
        
        equations[metric] = {
            'equation': equation,
            'r_squared': r_squared
        }
        
        plt.subplot(5, 2, i)
        
        for j, var in enumerate(training_vars):
            sorted_indices = df[var].argsort()
            sorted_x = df[var].iloc[sorted_indices]
            sorted_y = df[metric].iloc[sorted_indices]
            
            plt.scatter(sorted_x, sorted_y, alpha=0.3, label=f'{var}')
            
            z = np.polyfit(sorted_x, sorted_y, 1)
            p = np.poly1d(z)
            plt.plot(sorted_x, p(sorted_x), linestyle='--')
        
        plt.title(f'{metric.replace("_", " ").title()} vs Training Variables')
        plt.xlabel('Training Variables')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        
        plt.text(0.05, 0.95, 
                 f"R² = {r_squared:.2f}\n{equation}", 
                 transform=plt.gca().transAxes, 
                 verticalalignment='top', 
                 fontsize=8, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('performance_regression_analysis.png')
    plt.close()

    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Heatmap of Performance Metrics')
    plt.tight_layout()
    plt.savefig('performance_correlation_heatmap.png')
    plt.close()
    
    return equations

def export_performance_equations(performance_equations, filename='performance_analysis_results.txt'):
    with open(filename, 'w') as f:
        f.write("Performance Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        for metric, details in performance_equations.items():
            f.write(f"{metric.replace('_', ' ').title()} Performance Equation:\n")
            f.write(f"Equation: {details['equation']}\n")
            f.write(f"R-squared: {details['r_squared']:.4f}\n\n")
        
        f.write("\nNote:\n")
        f.write("- Equations represent the relationship between training variables and performance metrics\n")
        f.write("- R-squared indicates the proportion of variance explained by the model\n")
        f.write("- Training variables are normalized to ensure fair comparison\n")
        f.write("\nConstraints:\n")
        f.write("- Training Frequency: 1-7 days per week\n")
        f.write("- Training Hours: 1-20 hours per week (1-5 hours per training day)\n")
        f.write("- Training Intensity: 1-10 scale\n")
    
    print(f"Results saved to {filename}")

def main():
    performance_df = generate_realistic_performance_data()
    
    performance_equations = generate_performance_analysis(performance_df)
    export_performance_equations(performance_equations)
    
    print("\nPerformance Metrics Statistics:")
    print(performance_df.describe())

if __name__ == "__main__":
    main()
