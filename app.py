import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
from livereload import Server

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ROWS_PER_PAGE = 10  # Number of rows to display per page

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store data and model
model = joblib.load('models/adaboost.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
df = None
X_train = None
X_test = None
y_train = None
y_test = None
accuracy = None
training_result = None
testing_result = None
classification_results = None
hasil = None
preview_data = None
data_review = None

# def load_model(model_path):
#     global model, vectorizer
#     try:
#         with open(model_path, 'rb') as f:
#             model = pickle.load(f)
        
#         # Check if vectorizer is part of the model file
#         if isinstance(model, tuple) and len(model) == 2:
#             model, vectorizer = model
#         return True
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return False

# model = load_model('models/adaboost.pkl')
# Download NLTK resources
@app.before_request
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processing_data(df, text_column, label_column):
    
    
    
    
    # Step 4: Transform text data to numerical features
    X = vectorizer.transform(df['full_text'])

    # Step 5: Predict sentiments
    sentiments = model.predict(X)

    # Step 6: Map numerical predictions to sentiment labels
    # Assuming: 0 = negative, 1 = neutral, 2 = positive
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['sentiment'] = [sentiment_map.get(s, 'neutral') for s in sentiments]

    # Step 7: Save the results
    df.to_csv('polri1_with_sentiment.csv', index=False)
    

    
    return "Success"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kelola_data', methods=['GET', 'POST'])
def kelola_data():
    preview_data = None
    total_rows = 0
    total_columns = 0
    current_page = request.args.get('page', 1, type=int)
    total_pages = 1
    
    # Check if a file was previously uploaded
    if 'uploaded_file' in session:
        try:
            df = pd.read_csv(session['uploaded_file'])
            total_rows, total_columns = df.shape
            total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
            # Make sure current page is valid
            if current_page < 1:
                current_page = 1
            elif current_page > total_pages:
                current_page = total_pages
                
            # Calculate start and end indices for pagination
            start_idx = (current_page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            # Get slice of dataframe for current page
            preview_data = df.iloc[start_idx:end_idx].to_html(classes='table table-striped')
        except Exception as e:
            flash(f'Error reading CSV: {str(e)}')
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Save the filename to the session
            session['uploaded_file'] = filepath
            
            # Read CSV for preview
            try:
                df = pd.read_csv(filepath)
                total_rows, total_columns = df.shape
                total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
                current_page = 1
                
                # Get first page of data
                preview_data = df.iloc[0:ROWS_PER_PAGE].to_html(classes='table table-striped')
                flash('File successfully uploaded')
            except Exception as e:
                flash(f'Error reading CSV: {str(e)}')
                
    return render_template('kelola_data.html', 
                           preview_data=preview_data,
                           current_page=current_page,
                           total_pages=total_pages,
                           total_rows=total_rows,
                           total_columns=total_columns)

@app.route('/processing')
def processing():
    global preview_data
    preview_data = None
    total_rows = 0
    total_columns = 0
    current_page = request.args.get('page', 1, type=int)
    total_pages = 1

    # Check if a file has been uploaded in this session
    if 'uploaded_file' in session:
        try:
            # Read the CSV file for preview
            df = pd.read_csv(session['uploaded_file'])

            total_rows, total_columns = df.shape
            total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
            # Make sure current page is valid
            if current_page < 1:
                current_page = 1
            elif current_page > total_pages:
                current_page = total_pages
                
            # Calculate start and end indices for pagination
            start_idx = (current_page - 1) * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            # Get slice of dataframe for current page
            preview_data = df.iloc[start_idx:end_idx].to_html(classes='table table-striped')
        except Exception as e:
            flash(f'Error reading CSV: {str(e)}')


    if data_review is not None:
        data = pd.read_csv("hasilweb.csv")

        total_rows, total_columns = data.shape
        total_pages = math.ceil(total_rows / ROWS_PER_PAGE)
            
        # Make sure current page is valid
        if current_page < 1:
            current_page = 1
        elif current_page > total_pages:
            current_page = total_pages
                
        # Calculate start and end indices for pagination
        start_idx = (current_page - 1) * ROWS_PER_PAGE
        end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
        # Get slice of dataframe for current page
        preview_data = data.iloc[start_idx:end_idx].to_html(classes='table table-striped')

    
    return render_template('processing.html', preview_data=preview_data, training_result=training_result,
                           testing_result=testing_result,
                           classification_results=classification_results, hasil=hasil, current_page=current_page, total_pages=total_pages, total_rows=total_rows, total_columns=total_columns)

@app.route('/process_data', methods=['POST'])
def process_data():
    global X_train, X_test, y_train, y_test, accuracy, training_result, testing_result, classification_results, vectorizer, hasil, data_review, preview_data
    data = pd.read_csv(session['uploaded_file'])

    for i in range(len(data)):
        # Vectorize the text
        text_vec = vectorizer.transform([data['full_text'][i]])
            
        # Make prediction
        prediction = model.predict(text_vec)[0]
        data.loc[i, 'sentiment'] = prediction
        

    data = data[['full_text', 'sentiment']]
    X = data['full_text']
    y = data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Calculate metrics
    accuracy = accuracy_score(X, y)
    report = classification_report(X, y, output_dict=True)
    conf_matrix = confusion_matrix(X, y)

    classification_results = {
        'accuracy': accuracy * 100,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
    }

    hasil = True

    data.to_csv('hasilweb.csv', index=False)

    data_review = data.to_html(classes='table table-striped')

    preview_data = None

    return redirect(url_for('processing'))

@app.route('/hasil_klasifikasi')
def hasil_klasifikasi():
    return render_template('hasil_klasifikasi.html')

@app.route('/uji_coba', methods=['GET', 'POST'])
def uji_coba():
    prediction = None
    text = None
    if request.method == 'POST':
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not trained yet'})
    
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'})
        
        # Vectorize the text
        text_vec = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_vec)[0]

    return render_template('uji_coba.html', result=prediction, text=text)

if __name__ == '__main__':
    server = Server(app.wsgi_app)

    # Watch for changes in the templates and static files
    server.watch('templates/')
    server.watch('static/')
    server.serve(port=5000, host='0.0.0.0', debug=True)
    # app.run(debug=True)