import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Full paths to your JSON result files (replace with actual paths)
model_files = [
    'src/cnn/cnn_model_results.json',
    'src/gradientboosting/gradient_model_results.json',
    'src/knn/knn_model_results.json',
    'src/mlp/mlp_model_results.json',
    'src/randomforest/randomforest_model_results.json',
    'src/resnet/resnet_model_results.json',
    'src/svr/svr_model_results.json',
    'src/xgboost/xgboost_model_results.json'
]

# Dictionary to hold the results of all models
all_results = {
    'Model': [],
    'MSE': [],
    'MAE': [],
    'RMSE': [],
    'R2': [],
    'Accuracy': []
}

# Load each JSON file and extract the results
for model_file in model_files:
    with open(model_file, 'r') as f:
        data = json.load(f)
    
    # Extract model name from the file name (or assign manually)
    model_name = os.path.basename(model_file).split('_')[0].capitalize()
    
    # Append the results to the dictionary
    all_results['Model'].append(model_name)
    all_results['MSE'].append(data.get('Mean Squared Error', None))
    all_results['MAE'].append(data.get('Mean Absolute Error', None))
    
    # Calculate RMSE from MSE
    mse = data.get('Mean Squared Error', None)
    rmse = np.sqrt(mse) if mse is not None else None
    all_results['RMSE'].append(rmse)

    r2 = data.get('R-squared', None)  # Correct key for R-squared
    all_results['R2'].append(r2 if r2 is not None and r2 >= 0 else None)  # Only non-negative R² values
    all_results['Accuracy'].append(data.get('Accuracy', None))  # Correct key for Accuracy

# Convert to arrays for easier plotting
models = all_results['Model']
mse_values = all_results['MSE']
mae_values = all_results['MAE']
rmse_values = all_results['RMSE']
r2_values = all_results['R2']
accuracy_values = all_results['Accuracy']

# Replace None with 0 for plotting (or any other suitable value)
accuracy_values = [0 if val is None else val for val in accuracy_values]
rmse_values = [0 if val is None else val for val in rmse_values]
r2_values = [0 if val is None else val for val in r2_values]

# Plotting the comparison
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))

# Plot RMSE
ax5.bar(models, rmse_values, color='lightcoral')
ax5.set_title('Root Mean Squared Error (RMSE)')
ax5.set_ylabel('RMSE')
ax5.set_xticklabels(models, rotation=45, ha='right')

# Plot Accuracy
ax4.bar(models, accuracy_values, color='orange')
ax4.set_title('Accuracy (%)')
ax4.set_ylabel('Accuracy (%)')
ax4.set_xticklabels(models, rotation=45, ha='right')

# Plot MSE
ax1.bar(models, mse_values, color='skyblue')
ax1.set_title('Mean Squared Error (MSE)')
ax1.set_ylabel('MSE')
ax1.set_xticklabels(models, rotation=45, ha='right')

# Plot MAE
ax2.bar(models, mae_values, color='lightgreen')
ax2.set_title('Mean Absolute Error (MAE)')
ax2.set_ylabel('MAE')
ax2.set_xticklabels(models, rotation=45, ha='right')

# Plot R-squared (R²) if available and non-negative
ax3.bar(models, [val if val is not None else 0 for val in r2_values], color='salmon')
ax3.set_title('R-squared (R²)')
ax3.set_ylabel('R²')
ax3.set_xticklabels(models, rotation=45, ha='right')

# Display the plots
plt.suptitle('Model Performance Comparison')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Add space at the bottom to prevent overlap
plt.show()
