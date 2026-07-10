from flask import Flask, render_template, request, make_response
import joblib, numpy as np, requests, sqlite3
from fpdf import FPDF

app = Flask(__name__)
model = joblib.load('crop_model.pkl')
API_KEY = "71f6389db8156be7a7e0f355bc7d7185"

def init_db():
    conn = sqlite3.connect('history.db')
    conn.execute('CREATE TABLE IF NOT EXISTS history (city TEXT, crop TEXT, fert TEXT)')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city']
    N, P, K, ph = float(request.form['N']), float(request.form['P']), float(request.form['K']), float(request.form['ph'])
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()
    
    if 'main' not in res: return "Error: City not found."
    temp, humidity = res['main']['temp'], res['main']['humidity']
    
    crop = model.predict(np.array([[N, P, K, temp, humidity, ph, 100]]))[0]
    
    # Fertilizer Calculation
    need_N, need_P, need_K = max(0, 50-N), max(0, 50-P), max(0, 50-K)
    fert = f"N:{need_N}kg, P:{need_P}kg, K:{need_K}kg"
    
    conn = sqlite3.connect('history.db')
    conn.execute('INSERT INTO history (city, crop, fert) VALUES (?, ?, ?)', (city, crop, fert))
    conn.commit()
    conn.close()
    
    return render_template('result.html', city=city, crop=crop, temp=temp, fert=fert)

@app.route('/download_pdf/<city>/<crop>/<fert>')
def download_pdf(city, crop, fert):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"Agri Report: {city}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Crop: {crop}", ln=True)
    pdf.cell(200, 10, txt=f"Fertilizer: {fert}", ln=True)
    return make_response(pdf.output(dest='S').encode('latin-1'), 200, {'Content-Type': 'application/pdf'})

if __name__ == '__main__':
    app.run(debug=True)