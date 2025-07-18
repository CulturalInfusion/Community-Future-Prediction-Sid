---

# 📊 Forecasting Community Representation in Media

A deep‐learning toolkit to model and predict how diverse communities (African, Asian, Hispanic, Indigenous, Middle Eastern, European) appear across **News**, **Social Media**, and **Entertainment** (TV & Movies).

🔗 **Live Demo**: [http://sidmediadr.diversityatlas.io:5000/](http://sidmediadr.diversityatlas.io:5000/)

---

## 🏆 Highlights

* **Per‑Community LSTM Models**
  - Separate sequence models for each community, capturing long‑term trends in representation (2004–2024).

* **Multi‑Domain Forecasts**
  1. 📰 **News Channels** (BBC, CNN, Fox News, Al Jazeera, ABC, …)
  2. 💬 **Social Media** (Twitter, Facebook, Instagram, YouTube, Reddit)
  3. 🎬 **Entertainment** (Hollywood, Bollywood, K‑Dramas, Spanish TV, Nigerian Cinema)

* **Interactive Web UI**
  - Flask with Jinja + Chart.js streams a responsive interface:
    * Dropdowns for community & platform
    * Textarea for entering 20 years of past data
    * Dynamic 2025–2035 line charts & plain‑English insights

* **Benchmarking Accuracy**
  - LSTM vs ARIMA vs Prophet, measured by AMAPE & R² — achieving sub‑0.2 AMAPE and >0.95 R² in most domains.

---

## 📂 Repository Structure

```
├── Deployment.py         # Flask app entrypoint & model loader
├── templates/            # HTML templates (Flask_UI.html)
├── static/               # JS, CSS, Chart.js assets
├── models/               # Pretrained .keras LSTM files by community
├── data/                 # Sample CSVs & preprocessing scripts
├── tests/                # Unit and integration tests
├── README.md             # (You are here)
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quickstart

1. **Clone & install**

   ```bash
   git clone https://github.com/Siddharth7269/Community-Future-Prediction-Sid.git  
   cd Community-Future-Prediction-Sid  
   python3 -m venv venv  
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

2. **Deploy on EC2**

   - **Instance**: Ubuntu 22.04 LTS on t3.medium
   - **Setup**:
     ```bash
     # Pull repo, create venv, install deps
     git pull origin main
     source venv/bin/activate
     pip install -r requirements.txt
     ```
   - **Run**:
     ```bash
     # Use Gunicorn for production
     gunicorn --bind 0.0.0.0:5000 Deployment:app --daemon
     ```
   - **Restart**:
     ```bash
     pkill gunicorn && gunicorn --bind 0.0.0.0:5000 Deployment:app --daemon
     ```

   > *Tip*: You can use a simple `deploy.sh` script or Dockerfile/Ansible playbook in `scripts/` to automate.

3. **Run tests**

   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```

---

## 🔧 How It Works

1. **Load Models**: Flask loads six LSTM models (one per community) from `models/` on startup.
2. **User Input**: Frontend captures:
   * Community (e.g. `asian`)
   * Platform (e.g. `Twitter`)
   * 20 historical representation percentages
3. **Forecast Loop**: In `Deployment.py`, the model iteratively predicts year‑by‑year (11 steps: 2025–2035).
4. **Visualization**: JSON `{year:value}` + English summaries are rendered by Chart.js.

---

## 🔍 Performance & Metrics

| Model   | Domain        | Avg AMAPE | Avg R² |
| ------- | ------------- | --------- | ------ |
| LSTM    | News          | 0.18      | 0.97   |
| LSTM    | Social Media  | 0.16      | 0.98   |
| LSTM    | Entertainment | 0.20      | 0.96   |
| ARIMA   | All           | 0.30      | 0.92   |
| Prophet | All           | 0.28      | 0.94   |

*Full evaluation tables in `/data/` and methodology details in the paper.*

---

## 🧪 Testing

- **Unit tests**: `tests/test_deployment.py`, `tests/test_models.py`
- **Integration**: `tests/test_ui.py`
- **Run**:
  ```bash
  pytest
  ```

---

## 🛠 Troubleshooting

- **“Address already in use”**: Kill existing Gunicorn or change port.
- **Model load errors**: Verify `.keras` files exist in `models/` with correct names.
- **Missing dependencies**: Check `pip install -r requirements.txt` inside the active venv.

---

## 🤝 Contributing

1. Fork & clone
2. `git checkout -b feature/YourFeature`
3. Implement & test
4. Submit a pull request!

---

## 📄 Cite This Project

```bibtex
@article{Siddharth2025diversityforecast,
  title   = {Forecasting Community Representation in Media with LSTM},
  author  = {Siddharth Yadav, Nicole Lee, and Rezza Moieni},
  journal = {DiversityAtlas Tech Report},
  year    = {2025},
  url     = {http://sidmediadr.diversityatlas.io:5000/}
}
```

---

## 💡 License

© CulturalInfusion 2025 — https://github.com/Siddharth7269/Community-Future-Prediction-Sid
