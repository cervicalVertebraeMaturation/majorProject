from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
import joblib
import numpy as np
import pandas as pd
from segmentation import extract_features  # Import the extract_features function

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/segmented_images', exist_ok=True)

# Load model and scaler
scaler = joblib.load('src/xgboost/xgboost_scaler.pkl')
model = joblib.load('src/xgboost/xgboost_regressor.pkl')

# Excel path
features_excel_path = "output/robo/extracted_features1.xlsx"

# Save features to Excel function
def save_features_to_excel(features_df, actual_age, floored_predicted_age, image_name):
    features_df.insert(0, 'Model Name', "XGboost")
    features_df.insert(1, 'Image Name', os.path.basename(image_name))
    features_df.insert(2, 'Actual Age', actual_age)
    features_df.insert(3, 'Predicted Age', floored_predicted_age)

    if os.path.exists(features_excel_path):
        existing_data = pd.read_excel(features_excel_path)
        combined_data = pd.concat([existing_data, features_df], ignore_index=True)
    else:
        combined_data = features_df

    combined_data.to_excel(features_excel_path, index=False)

# Login route with session handling
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username:
            error_message = "**Username cannot be empty."
            return render_template('login.html', error=error_message)

        if not password:
            error_message = "**Password cannot be empty."
            return render_template('login.html', error=error_message)

        # Password check
        if password == "team12":
            session['logged_in'] = True
            return redirect(url_for('upload_file'))
        else:
            error_message = "**Invalid username or password. Please try again."
            return render_template('login.html', error=error_message)

    return render_template('login.html')

# Protect upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Extract features and segment the image
            features_df, segmented_image_path = extract_features(image_path)

            # Extract actual age from the image name
            actual_age = int(file.filename[-6:-4])

            # Scale features and predict age
            new_features_scaled = scaler.transform(features_df)
            predicted_age = model.predict(new_features_scaled)
            floored_predicted_age = int(np.floor(predicted_age[0]))

            # Save to Excel
            save_features_to_excel(features_df, actual_age, floored_predicted_age, file.filename)

            # Render result page
            segmented_image_filename = os.path.basename(segmented_image_path)
            return render_template('result.html', 
                                   actual_age=actual_age,
                                   predicted_age=floored_predicted_age,
                                   segmented_image=segmented_image_filename,
                                   image_name=segmented_image_filename)
    
    return render_template('index.html')  # Image upload page

if __name__ == '__main__':
    app.run(debug=True)
