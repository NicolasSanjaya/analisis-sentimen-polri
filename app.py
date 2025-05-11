import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
from livereload import Server
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
from collections import Counter
import re
import nltk

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

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# @app.before_request
# def download_nltk_resources():
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         nltk.download('punkt')
    
#     try:
#         nltk.data.find('corpora/stopwords')
#     except LookupError:
#         nltk.download('stopwords')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processing_data(df):
    # Preprocessing
    df = df.dropna(subset='full_text')
    df = df.drop_duplicates()

    # 1. Mengambil kolom yang dibutuhkan
    df['full_text'] = df['full_text']

    print(df.head())

    # 2. Case folding - mengubah teks menjadi lowercase
    df['full_text'] = df['full_text'].str.lower()

    # 3. Cleaning - menghapus URL, mention, hashtag, dan karakter non alfanumerik
    def clean_text(text):
        # Menghapus URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Menghapus mention
        text = re.sub(r'@\w+', '', text)
        # Menghapus hashtag
        text = re.sub(r'#\w+', '', text)
        # Menghapus RT dan FAV
        text = re.sub(r'\brt\b', '', text)
        # Menghapus karakter non alfanumerik kecuali spasi
        text = re.sub(r'[^\w\s]', '', text)
        # Menghapus angka
        text = re.sub(r'\d+', '', text)
        # Menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        # Mengganti &amp dengan kata dan
        text = text.replace('amp', 'dan')
        return text

    df['full_text'] = df['full_text'].apply(clean_text)

    # 4. Tokenizing
    df['stemmed_text'] = df['full_text'].apply(word_tokenize)

    # 5. Stopword removal
    factory = StopWordRemoverFactory()
    stopwords_id = factory.get_stop_words()
    additional_stopwords = [
        # Singkatan umum
        'yg', 'dgn', 'nya', 'utk', 'dlm', 'bkn', 'tdk', 'org', 'krn', 'jg', 'sdh', 'spy', 'trs', 'tsb',
        'skrg', 'sih', 'gak', 'ga', 'tuh', 'spt', 'bgs', 'tp', 'klo', 'kl', 'dr', 'pd', 'sm', 'bwt',
        'kmrn', 'sy', 'lg', 'gue', 'gw', 'aja', 'deh', 'sih', 'jd', 'bs', 'bisa', 'kok', 'kyk', 'ni',
        'yah', 'sih', 'gt', 'loh', 'bgt', 'udh', 'dpt', 'udah', 'nih', 'gini', 'gitu', 'gmn', 'thd', 'sgt',

        # Kata kerja umum
        'adalah', 'merupakan', 'menjadi', 'memiliki', 'terdapat', 'menurut', 'memang', 'seperti',

        # Kata hubung
        'dan', 'atau', 'tetapi', 'namun', 'serta', 'karena', 'sebab', 'jika', 'apabila', 'maka',
        'sehingga', 'agar', 'supaya', 'ketika', 'selama', 'sebelum', 'sesudah', 'sejak', 'sampai',
        'hingga', 'meskipun', 'walaupun', 'seolah', 'andai', 'kendati', 'seandainya', 'kalau',

        # Kata ganti dan petunjuk
        'saya', 'aku', 'kamu', 'kau', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita', 'beliau',
        'ini', 'itu', 'tersebut', 'begini', 'begitu', 'demikian',

        # Kata depan
        'di', 'ke', 'dari', 'pada', 'kepada', 'oleh', 'untuk', 'bagi', 'tentang', 'dengan', 'dalam',
        'antara', 'terhadap', 'akan', 'mengenai',

        # Kata keterangan
        'sangat', 'amat', 'terlalu', 'sekali', 'banget', 'sungguh', 'cukup', 'agak', 'hampir', 'hanya',
        'saja', 'pun', 'lagi', 'juga', 'sedang', 'masih', 'telah', 'sudah', 'belum', 'pernah',
        'selalu', 'sering', 'jarang', 'kadang', 'mungkin', 'barangkali', 'tentu', 'pasti',

        # Kata bilangan
        'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh',
        'sebelas', 'dua belas', 'puluh', 'ratus', 'ribu', 'juta', 'pertama', 'kedua', 'ketiga',
        'keempat', 'kelima', 'beberapa', 'sebagian', 'semua', 'seluruh', 'banyak', 'sedikit',

        # Kata tanya
        'apa', 'siapa', 'kapan', 'bila', 'bilamana', 'dimana', 'kemana', 'bagaimana', 'mengapa',
        'kenapa', 'berapa', 'mana',

        # Lainnya
        'ya', 'tidak', 'bukan', 'belum', 'sudah', 'mari', 'ayo', 'silakan', 'tolong'
    ]

    stopwords_id.extend(additional_stopwords)

    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stopwords_id and len(word) > 2]

    df['stemmed_text'] = df['stemmed_text'].apply(remove_stopwords)

    # 6. Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stem_tokens(tokens):
        return [stemmer.stem(token) for token in tokens]

    df['stemmed_text'] = df['stemmed_text'].apply(stem_tokens)
    df['stemmed_text'] = df['stemmed_text'].apply(lambda x: ' '.join(x))
    df.to_csv('data.csv', index=False)
    return df


# Generate charts as base64 images
def create_bar_chart(sentiment_counts):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    categories = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    # Define colors based on sentiment
    color_map = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    colors = [color_map.get(cat, '#2196F3') for cat in categories]
    
    ax = sns.barplot(x=categories, y=values, palette=colors)
    plt.title('Distribusi Sentimen', fontsize=16)
    plt.xlabel('Kategori Sentimen', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_pie_chart(sentiment_counts):
    plt.figure(figsize=(8, 8))
    categories = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    # Define colors based on sentiment
    color_map = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    colors = [color_map.get(cat, '#2196F3') for cat in categories]
    
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Proporsi Sentimen', fontsize=16)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100, 
                         colormap='viridis').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

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
    # data = pd.read_csv('data.csv')

    # Preprocess the data
    data = processing_data(data)

    print(data.head())

    for i in range(len(data)):
        # Vectorize the text
        text_vec = vectorizer.transform([data['stemmed_text'][i]])
        
        # Make prediction
        prediction = model.predict(text_vec)[0]
        data.loc[i, 'sentiment'] = prediction
        
    print(data.head())
    
    data = data[['full_text', 'stemmed_text', 'sentiment']]
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
    df = pd.read_csv('hasilweb.csv')
    
    # Generate charts
    # Wordcloud
    wordcloud_img = create_wordcloud(' '.join(df['full_text']))
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    accuracies = {
        'training_accuracy': training_result['accuracy'],
        'testing_accuracy': testing_result['accuracy'],
        'classification_report': classification_results['report'],
        'confusion_matrix': classification_results['confusion_matrix']
    }
    examples = {
        'positive': df[df['sentiment'] == 'positive'].sample(5).to_dict(orient='records'),
        'neutral': df[df['sentiment'] == 'neutral'].sample(5).to_dict(orient='records'),
        'negative': df[df['sentiment'] == 'negative'].sample(5).to_dict(orient='records')
    }

    bar_chart = create_bar_chart(sentiment_counts)
    pie_chart = create_pie_chart(sentiment_counts)
    
    return render_template('hasil_klasifikasi.html',
                          accuracies=accuracies,
                          examples=examples,
                          bar_chart=bar_chart,
                          pie_chart=pie_chart,
                          wordcloud=wordcloud_img)

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