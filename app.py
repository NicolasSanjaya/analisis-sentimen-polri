import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ROWS_PER_PAGE = 10  # Number of rows to display per page

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    return render_template('processing.html')

@app.route('/hasil_klasifikasi')
def hasil_klasifikasi():
    return render_template('hasil_klasifikasi.html')

if __name__ == '__main__':
    app.run(debug=True)