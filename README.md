<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=CA%20Intelligence%20Suite&fontSize=48&fontColor=fff&animation=twinkling&fontAlignY=35&desc=AI-Powered%20Chartered%20Accountant%20Assistant&descAlignY=55&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()
[![Made with ❤](https://img.shields.io/badge/Made%20with-❤-red?style=for-the-badge)]()

<br/>

> 🧠 **Combining Machine Learning · NLP · Symbolic AI · Financial Analytics**
> into one intelligent Chartered Accountant platform.

<br/>

[✨ Features](#-features) · [🧠 How It Works](#-how-it-works) · [📊 Dashboard](#-dashboard-modules) · [🚀 Quick Start](#-quick-start) · [📸 Screenshots](#-screenshots)

</div>

---

## ✨ Features

<div align="center">

| 🤖 AI & ML | 💰 Finance | 🛡 Compliance | 📊 Reporting |
|:---:|:---:|:---:|:---:|
| Transaction Classification | GST Liability Calculation | Section 269ST Checks | PDF Report Generation |
| Anomaly Detection (IsoForest) | Balance Sheet Analysis | Compliance Scoring | Interactive Dashboard |
| TF-IDF NLP Features | Profit & Loss Assessment | Risk Alerting | Excel & CSV Support |
| Confidence Scoring | Debt-to-Asset Ratio | IQR Outlier Detection | CA Recommendations |
| Neural Network Predictions | Expense-to-Income Ratio | Leverage Analysis | KPI Summary Cards |

</div>

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CA Intelligence Pipeline                       │
└─────────────────────────────────────────────────────────────────────┘

  📁 Upload CSV/Excel                                                  
        │                                                              
        ▼                                                              
  🔧 Preprocessing & Feature Engineering                              
        │  ┌─────────────────┐   ┌──────────────────┐                
        │  │  TF-IDF (NLP)   │   │  Amount Scaler   │                
        │  └─────────────────┘   └──────────────────┘                
        │                                                              
        ▼                                                              
  🤖 TensorFlow Neural Network Classification                         
        │  → Predicts: Income / Expense / Asset / Liability / GST     
        │                                                              
        ▼                                                              
  📐 Backward Chaining Rule Engine                                    
        │  ┌──────────────┐  ┌──────────────────┐                    
        │  │  GST Engine  │  │  Balance Sheet   │                    
        │  └──────────────┘  └──────────────────┘                    
        │                                                              
        ▼                                                              
  🚨 Anomaly Detection (Isolation Forest + IQR)                       
        │                                                              
        ▼                                                              
  💡 CA Insights → 📄 PDF Report Generation                           
```

---

## 🔗 Backward Chaining Rule Engine

<table>
<tr>
<td width="50%">

**GST Payable Goal**
```
Goal: GST Payable
│
├── GST on Income
│      └── Income × GST Rate
│
└── GST on Expense
       └── Input Tax Credit
```

</td>
<td width="50%">

**Balance Sheet Goal**
```
Goal: Balance Sheet
│
├── Total Assets
│     └── Category == Asset
│
└── Total Liabilities
      └── Category == Liability
```

</td>
</tr>
</table>

---

## 📊 Dashboard Modules

<div align="center">

```
┌──────────────────────────────────────────────────────────────────┐
│                    🖥  Streamlit Dashboard                        │
├──────────────┬───────────────┬──────────────┬────────────────────┤
│ 📋 Preview   │ 🤖 AI Predict │ 📊 Dashboard │  📈 Visualizations │
│              │               │              │                    │
│ Upload &     │ Real-time     │ KPI Cards &  │  Charts, Trends &  │
│ Inspect Data │ Classification│ Summaries    │  Distribution Plots│
├──────────────┴───────────────┼──────────────┴────────────────────┤
│         💡 CA Insights       │         📄 Report Export          │
│                              │                                    │
│  Automated Recommendations   │   PDF / Excel Downloadable Report  │
└──────────────────────────────┴────────────────────────────────────┘
```

</div>

---

## 📂 Project Structure

```
📦 CA-Intelligence-Suite/
│
├── 🐍 app.py                  ← Streamlit Web App (Main Entry)
├── ⚙️  config.py               ← Global Configuration & Hyperparameters
├── 🔄 run_pipeline.py          ← Full Pipeline Orchestrator
├── 🧠 train_model.py           ← TensorFlow Model Training
├── 🔧 preprocess.py            ← Feature Engineering & NLP
├── 🏭 generate_dataset.py      ← Synthetic Dataset Generator
├── 📈 financial_analysis.py    ← Financial Engine & Rule System
│
├── 📁 datasets/                ← Raw & Processed Data
├── 📁 outputs/                 ← Model Predictions & Results
├── 📁 reports/                 ← Generated PDF Reports
│
└── 📁 model_artifacts/
    ├── 🤖 ca_model.h5           ← Trained Neural Network
    ├── 📝 tfidf_vectorizer.pkl  ← NLP Vectorizer
    ├── 📏 amount_scaler.pkl     ← Feature Scaler
    ├── 🔢 payment_mode_encoder.pkl
    ├── 🗂  features.pkl
    ├── 📊 metrics.json          ← Model Performance Metrics
    └── 🚨 anomaly_model.pkl     ← Isolation Forest Model
```

---

## 📊 Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           📊 CA Intelligence Suite — Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  💰 Total Income         :  ₹ 2,34,56,789
  💸 Total Expense        :  ₹ 1,89,23,456
  📈 Net Profit           :  ₹   45,33,333
  📉 Profit Margin        :       19.3%

  🧾 Net GST Payable      :  ₹    4,23,100

  🏦 Total Assets         :  ₹   67,45,000
  📋 Total Liabilities    :  ₹   23,12,000
  ⚖️  Debt-to-Asset Ratio  :       34.28%

  🚨 Anomalies Detected   :         3
  ✅ Compliance Score     :       91 / 100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔍 Financial Analysis Engine

<details>
<summary><b>📈 Profitability Analysis</b></summary>

- Net Profit / Loss Alerts
- Net Profit Margin Calculation
- Revenue Growth Monitoring
- Cost Optimization Flags

</details>

<details>
<summary><b>💸 Expense Intelligence</b></summary>

- Expense-to-Income Ratio Monitoring
- Top Expense Category Breakdown
- Cost Reduction Suggestions
- Budget Deviation Alerts

</details>

<details>
<summary><b>🧾 GST Assessment</b></summary>

- GST Liability Calculation (CGST / SGST / IGST)
- Input Tax Credit Analysis
- GST Risk Flagging
- Audit-Ready Reports

</details>

<details>
<summary><b>🚨 Risk & Compliance</b></summary>

- IQR-Based Statistical Outlier Detection
- Isolation Forest Anomaly Detection
- Section 269ST Cash Transaction Monitoring
- Compliance Scoring Engine (0–100)

</details>

<details>
<summary><b>🏦 Financial Health</b></summary>

- Debt-to-Asset Ratio
- Liquidity Assessment
- Leverage Risk Analysis
- Balance Sheet Reconciliation

</details>

---

## 🛠 Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:------|:----------:|:--------|
| 🐍 **Backend** | Python 3.10+ | Core Language |
| 🔢 **Data** | Pandas, NumPy | Data Processing & Wrangling |
| 🤖 **Deep Learning** | TensorFlow / Keras | Transaction Classification |
| 📝 **NLP** | Scikit-learn TF-IDF | Text Feature Extraction |
| 📐 **Rule Engine** | Backward Chaining | Financial Logic Reasoning |
| 🚨 **Anomaly** | Isolation Forest | Outlier / Fraud Detection |
| 📊 **Visualization** | Plotly, Matplotlib, Seaborn | Charts & Dashboards |
| 📄 **Reporting** | ReportLab | PDF Generation |
| 💾 **Serialization** | Joblib | Model Persistence |
| 🌐 **Web App** | Streamlit | Interactive Frontend |

</div>

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/CA-Intelligence-Suite.git
cd CA-Intelligence-Suite
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate Dataset

```bash
python generate_dataset.py
```

### 4️⃣ Train the AI Model

```bash
python train_model.py
```

### 5️⃣ Run Full Pipeline

```bash
python run_pipeline.py
```

### 6️⃣ Launch the Dashboard

```bash
streamlit run app.py
```

> 🌐 Open your browser at **http://localhost:8501**

---

## ⚙️ Configuration

All project-level settings are managed in `config.py`:

```python
# config.py — Configurable Parameters
DATASET_PATH      = "datasets/transactions.csv"
MODEL_PATH        = "model_artifacts/ca_model.h5"
CONFIDENCE_THRESH = 0.75        # Minimum prediction confidence
GST_RATE          = 0.18        # 18% GST
ANOMALY_CONTAMINATION = 0.05   # 5% expected anomaly rate
PLOTLY_THEME      = "plotly_dark"
```

---

## 📸 Screenshots

<div align="center">

> 🖥 **Data Upload & Preview**

![Data Preview](https://github.com/user-attachments/assets/8fc30313-6254-4caf-81da-381375156c32)

> 🤖 **AI Transaction Classification**

![AI Predictions](https://github.com/user-attachments/assets/c1bb5375-f113-4af2-9dec-c15311e0a06b)

> 📊 **Financial KPI Dashboard**

![Dashboard](https://github.com/user-attachments/assets/8639e211-20e8-4c15-8344-a48b3a962715)

> 📈 **Visualizations & Charts**

![Charts](https://github.com/user-attachments/assets/ac8f7da1-0a4f-4ea2-8049-e5bfce1de87c)

> 💡 **CA Insights & Recommendations**

![CA Insights](https://github.com/user-attachments/assets/1fd308f2-aa13-4e5f-999c-bda11135abdd)

> 📄 **Report Generation**

![Reports](https://github.com/user-attachments/assets/55b0f00e-9adb-4541-89cf-19ba8be15a6e)

</div>

---

## 🔮 Future Roadmap

```
2024 ─────────────────────────────────────────────────────── 2025+
  │                                                              │
  ▼                                                              ▼
  ✅ Core AI Engine         🔜 Generative AI Assistant           
  ✅ GST Calculator         🔜 Voice Query Interface             
  ✅ Anomaly Detection      🔜 OCR Invoice Processing            
  ✅ PDF Reports            🔜 Financial Forecasting             
  ✅ Streamlit Dashboard    🔜 Cloud Deployment (AWS/GCP)        
                            🔜 Multi-Language Support            
                            🔜 ERP Integration                  
```

---

## 🎯 Why CA Intelligence Suite?

<div align="center">

```
Traditional Accounting Workflow          CA Intelligence Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Spreadsheets                     ┐
📒 Ledgers                          │
📜 Tax Rules                        ├──► 🤖 ONE Intelligent Platform
✅ Compliance Checklists            │
📊 Financial Statements             ┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

         Hours of Manual Work  →  Minutes of AI Analysis
```

</div>

This project demonstrates the power of combining:

- 🤖 **Machine Learning** — for intelligent classification
- 📐 **Symbolic AI** — for rule-based financial reasoning
- 💰 **Financial Intelligence** — for real-world CA workflows
- 🔍 **Explainable AI** — for transparent, auditable decisions

---

## 👨‍💻 Author

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=80&text=Chandrashekhar&fontSize=32&fontColor=fff&animation=fadeIn" width="400"/>

<br/><br/>

🎓 **M.Sc Data Science**, 
**Tumkur University**

💡 *AI · Machine Learning · Financial Analytics · CA Automation*


**⭐ Star this repository if you found it helpful!**

**🤝 Contributions are welcome — raise an Issue!**

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

*Built with ❤️ by Chandrashekhar · Powered by AI & Financial Intelligence*

</div>
