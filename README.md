
---

# 📊 Forecasting Community Representation in Media

A deep‐learning-powered toolkit for modeling and predicting how different communities (African, Asian, Hispanic, Indigenous, Middle Eastern, European) show up across **News**, **Social Media**, and **Entertainment** (TV & Movies).

🔗 **Live Demo**: [http://sidmediadr.diversityatlas.io:5000/](http://sidmediadr.diversityatlas.io:5000/)

---

## 🏆 Highlights

* **Per-Community LSTM Models**
  We train separate sequence models for each community, capturing long-range trends in representation percentages from 2004–2024.

* **Multi-Domain Predictions**
  Forecasts span three domains:

  1. 📰 News Channels (BBC, CNN, Fox News, Al Jazeera, ABC, …)
  2. 💬 Social Media (Twitter, Facebook, Instagram, YouTube, Reddit)
  3. 🎬 Entertainment (Hollywood, Bollywood, K-Dramas, Spanish TV, Nigerian Cinema)

* **Interactive Web UI**
  Flask + Jinja templates + Chart.js drive a responsive interface with:

  * Dropdowns for community & platform selection
  * Textarea for entering the last 20 years of data
  * Dynamic line charts (2025–2035) and plain-English summaries (e.g. “Fox News predicts a 13% dip for African communities in 2025”).

* **Rigorous Benchmarking**
  We compare LSTM to ARIMA and Prophet models using AMAPE & R²—our LSTM achieves sub-0.2 AMAPE and >0.95 R² in most cases.

---

## 📂 Repository Structure

```
├── Deployment.py         # Flask app entrypoint & model loader
├── templates/            # HTML templates (Flask_UI.html)
├── static/               # JS, CSS, Chart.js assets
├── models/               # Pretrained .keras LSTM files per community
├── data/                 # Sample CSVs & data-processing scripts
├── README.md             # (You are here)
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quickstart

1. **Clone & install**

   ```bash
   git clone https://github.com/Siddharth7269/Community-Future-Prediction.git  
   cd your-repo  
   python3 -m venv venv  
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

2. **Start your app**

   ```bash
   gunicorn --bind 0.0.0.0:5000 Deployment:app
   ```

   or, for development:

   ```bash
   FLASK_ENV=development flask run --host=0.0.0.0
   ```

3. **Browse**
   Open [http://localhost:5000](http://localhost:5000) and explore forecasts!

---

## 🔧 How It Works

1. **Load Models**
   On startup, Flask loads six LSTM models (one per community) from `models/`.

2. **User Input**
   The frontend collects:

   * Community (e.g. “asian”)
   * Platform (e.g. “Twitter”)
   * 20 historical representation values (comma-separated)

3. **Forecast Loop**
   In `Deployment.py`, we repeatedly predict the “next” year, append it, and slide the window—yielding an 11-year forecast (2025–2035).

4. **Results & Visualization**

   * JSON of `{year: value}` plus plain-English “dip/rise” messages
   * Chart.js renders an interactive line plot with legend & tooltips

---

## 🔍 Performance & Metrics

| Model   | Domain        | Avg AMAPE | Avg R² |
| ------- | ------------- | --------- | ------ |
| LSTM    | News          | 0.18      | 0.97   |
| LSTM    | Social Media  | 0.16      | 0.98   |
| LSTM    | Entertainment | 0.20      | 0.96   |
| ARIMA   | All           | 0.30      | 0.92   |
| Prophet | All           | 0.28      | 0.94   |

*Full evaluation tables and methodology are in the paper.*

---

## 🤝 Contributing

1. Fork & clone
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes & test
4. Submit a pull request!

---

## 📄 Cite This Project

```bibtex
@article{Siddharth2025diversityforecast,
  title   = {Forecasting Community Representation in Media with LSTM},
  author  = {Siddharth Yadav,Rezza Moieni,Nabi Zamani and Nicole Lee},
  journal = {DiversityAtlas Tech Report},
  year    = {2025},
  url     = {http://sidmediadr.diversityatlas.io:5000/}
}
```

---

## 💡 License

CulturalInfusion © (https://github.com/Siddharth7269)

