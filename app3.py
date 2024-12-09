import json, os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, static_url_path='/static')

# Direktori data
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Memuat file CSV
csv_file_path = 'data/exercises.csv'
df = pd.read_csv(csv_file_path)

# Pra-pemrosesan data
# Menghapus nama folder dari nama file gambar
df['images'] = df['images'].apply(lambda x: [image.strip(" '") for image in x.strip("[]").split(", ")])

# Mengganti nilai NaN dengan string kosong
df.fillna('', inplace=True)

# Memastikan semua kolom bertipe string
kolom_konversi = ['name', 'force', 'level', 'mechanic', 'equipment', 'primaryMuscles', 'secondaryMuscles', 'instructions', 'category']
for kolom in kolom_konversi:
    df[kolom] = df[kolom].astype(str)

# Menentukan prioritas untuk field input pengguna
field_prioritas = ['primaryMuscles', 'level', 'equipment', 'secondaryMuscles', 'force', 'mechanic', 'category']

# Menentukan bobot prioritas
bobot_prioritas = [20, 15, 10, 5, 3, 2, 1]

# Menggabungkan kolom relevan untuk rekomendasi
df['content'] = df[field_prioritas].apply(
    lambda baris: (
        ' '.join([str(nilai) * bobot for nilai, bobot in zip(baris, bobot_prioritas)])
    ),
    axis=1
)

# Membuat TF-IDF vectorizer untuk mengubah konten menjadi bentuk numerik
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# Menghitung kesamaan kosinus antar latihan
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')

@app.route('/beginner', methods=['GET', 'POST'])
def beginner():
    # Daftar otot utama yang tersedia
    primary_muscles = ["Chest", "Biceps", "Abdominals", "Quadriceps", "Middle Back", "Glutes", "Hamstrings", "Calves"]
    # Otot utama yang dipilih oleh pengguna (disimpan di cookie)
    selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle')
    if request.method == 'POST':
        # Menangkap input dari pengguna
        selected_primary_muscle = request.form.get('selectedPrimaryMuscle')
        response = redirect(url_for('recommend_exercises'))
        response.set_cookie('selectedPrimaryMuscle', selected_primary_muscle)
        return response
    return render_template('beginner.html', primary_muscles=primary_muscles, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_exercises():
    # Data latihan yang akan direkomendasikan
    data_latihan = []
    input_pengguna = {}
    otot_utama_terpilih = ""

    if request.method == 'POST':
        # Menangkap input dari form
        input_pengguna = {field: request.form.get(field, '') for field in field_prioritas}
        otot_utama_terpilih = request.cookies.get('selectedPrimaryMuscle', "")

        # Menggabungkan input pengguna untuk rekomendasi
        konten_pengguna = (
            otot_utama_terpilih * 20 + ' ' +
            ' '.join([input_pengguna.get(field, '') * bobot for field, bobot in zip(field_prioritas, bobot_prioritas)])
        )

        # Mengubah konten pengguna ke dalam bentuk TF-IDF
        user_tfidf_matrix = tfidf_vectorizer.transform([konten_pengguna])
        user_cosine_sim = linear_kernel(user_tfidf_matrix, tfidf_matrix)
        skor_kemiripan = user_cosine_sim[0]
        indeks_latihan = skor_kemiripan.argsort()[::-1][:5]  # Memilih 5 rekomendasi terbaik

        # Mendapatkan latihan berdasarkan indeks
        for indeks in indeks_latihan:
            latihan = df.iloc[indeks].to_dict()
            data_latihan.append(latihan)

        return render_template('recommendations.html', recommendations=data_latihan, user_input=input_pengguna, selectedPrimaryMuscle=otot_utama_terpilih)

    return render_template('recommendations.html', recommendations=data_latihan, user_input=input_pengguna, selectedPrimaryMuscle=otot_utama_terpilih)

if __name__ == '__main__':
    app.run(debug=True)
