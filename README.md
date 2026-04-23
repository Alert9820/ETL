# ⚡ SmartETL — Automated Data Analytics System

> An end-to-end automated ETL pipeline with ML-powered analytics and an interactive dashboard. Upload any CSV — SmartETL handles the rest.

🔗 **Live Demo:** [etl-1-eq88.onrender.com](https://etl-1-eq88.onrender.com)

---

## 📸 Preview

```
Upload CSV → Auto Clean → Auto ML → Interactive Dashboard → Download Report
```

---

## 🚀 Features

### 🔍 Extract
- Auto-detects CSV encoding (UTF-8, Latin-1, CP1252, UTF-16)
- Auto-detects delimiter ( `,` `;` `\t` `|` )
- Auto-infers column types — numeric, categorical, datetime, binary, text
- Handles files up to 100K+ rows

### 🧹 Transform
- Missing value imputation — mean/mode/forward-fill based on column type
- Duplicate row removal
- Outlier removal using IQR (1st–99th percentile)
- Zero-variance column removal
- Auto feature engineering:
  - `Profit` = Revenue − Cost
  - `Profit_Margin_%` = (Profit / Revenue) × 100
  - `Total_Value` = Units × Price
  - Date parts — year, month, day, day-of-week

### 📦 Load
- Saves cleaned data as **CSV**
- Saves to **SQLite database**
- Downloadable report from dashboard

### 🤖 Auto ML
- Auto-detects problem type — **Regression** or **Classification**
- Trains 3 models simultaneously:
  - Ridge Regression
  - Random Forest
  - XGBoost
- Picks best model based on R² (regression) or Accuracy (classification)
- Shows Feature Importance + Actual vs Predicted chart

### 📊 Dashboard
- Stat cards — records, columns, revenue, profit, best model
- Distribution Explorer (per column histogram)
- Category Breakdown (bar chart)
- Correlation Heatmap
- ML Results — model comparison + feature importance
- Column Statistics — min, max, mean, median, std, sum
- Paginated cleaned data preview
- Live pipeline log

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy, SciPy |
| Machine Learning | Scikit-learn, XGBoost |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Deployment | Docker, Render.com |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
smart-etl/
├── main.py          # FastAPI backend — ETL engine + ML + API routes
├── index.html       # Frontend — Dashboard UI
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker container config
└── README.md
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- pip

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/smart-etl.git
cd smart-etl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open browser
http://localhost:8000
```

### Run with Docker

```bash
# Build image
docker build -t smart-etl .

# Run container
docker run -p 8000:8000 smart-etl

# Open browser
http://localhost:8000
```

---

## 🌐 Deploy on Render

1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Select **Docker** as runtime
5. Click **Deploy**

That's it — live in minutes! ✅

---

## 📊 Supported CSV Formats

SmartETL works with **any** CSV file. No configuration needed.

**Best results with columns like:**

| Column Type | Examples |
|---|---|
| Revenue / Sales | `Revenue`, `Sales`, `Income`, `Amount` |
| Cost / Expense | `Cost`, `Expense`, `Spend` |
| Date | `Date`, `Order_Date`, `Created_At` |
| Quantity | `Units`, `Qty`, `Quantity` |
| Category | `Region`, `Category`, `Department` |

---

## 🧪 Test Dataset

Don't have a CSV? Use the built-in **Demo Dataset** button on the dashboard — it loads a 20-row sales dataset instantly.

For larger tests, generate a dataset:

```python
# Generate 10K row sales CSV for testing
python generate_dataset.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve dashboard UI |
| `POST` | `/upload` | Upload CSV file |
| `GET` | `/results/{job_id}` | Get ETL + ML results |
| `GET` | `/download/{job_id}` | Download cleaned CSV |
| `GET` | `/health` | Health check |

---

## 💡 Use Cases

- **Data Analysts** — Clean and explore raw datasets instantly
- **Business Teams** — Upload sales/HR data, get insights without coding
- **Students** — Learn ETL pipeline concepts with a real working system
- **Developers** — Use the API to integrate ETL into other apps
- **Researchers** — Automate repetitive data cleaning tasks

---

## 🐛 Known Limitations

- In-memory session storage — results lost on server restart (Render free tier spins down)
- ML works best with structured numeric/categorical data
- Very large files (500K+ rows) may timeout on free tier

---

## 👨‍💻 Author

**Sunny**
- Built as a Data Analyst / Junior Data Engineer portfolio project
- Demonstrates: ETL design, data cleaning, ML automation, API development, Docker deployment

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <strong>⚡ SmartETL — Because data cleaning shouldn't be manual.</strong>
</div>

