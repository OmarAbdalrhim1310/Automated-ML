# 🤖 Omni-Capable Autonomous AI Data Scientist Agent

An end-to-end, production-grade automated machine learning (AutoML) platform and conversational business intelligence tool. This agent diagnoses datasets, self-executes preprocessing pipelines, creates executive dashboards, and runs hyperparameter tuning across multiple ML & DL frameworks without human intervention.

## 🌟 Key Features

- **Autonomous Data Diagnostic:** Instantly infers whether the uploaded matrix is Labeled (Supervised) or Unlabeled (Unsupervised).
- **Self-Directed Preprocessing:** Outlier clipping via Z-score, missing value imputation (Mean/Mode), and automatic feature encoding.
- **Executive Dashboard Pro:** Rendered with Plotly to showcase KPIs, correlation trends, and statistical distributions dynamically.
- **Tuned Modeling Arena:** Simultaneous training loops over Linear Models, Tree Ensembles, Boosting Frameworks, SVMs, and Deep Learning (MLP Neural Networks).
- **Self-Explaining Tuning Engine:** Leverages GridSearchCV/RandomizedSearchCV and formats a scientific breakdown explaining exactly *why* parameters were selected.
- **Conversational Analytics:** Chatbot interface powered by Groq (Llama 3) to query datasets, prompt charts, and audit models using natural language.

## 📂 Project Structure

```text
AI_Business_Analyst_Project/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Autonomous data cleaning module
│   └── automl_engine.py    # Hybrid multi-model hyper-tuning engine
│
├── app.py                  # Streamlit central user interface
├── requirements.txt        # Frozen pipeline environment dependencies
└── README.md               # Production documentation
```

## 🛠️ Installation & Setup

Follow these exact steps to compile and deploy the system locally inside a virtual environment:

### 1. Clone the repository & Navigate
```bash
git clone https://github.com
cd AI_Business_Analyst_Project
```

### 2. Establish a Stable Virtual Environment
```bash
py -3.11 -m venv venv
```

### 3. Activate the Environment
* **On Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
* **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the App

Execute the centralized server engine using the following command:
```bash
streamlit run app.py
```

## 🔑 Authentication Note
To fully unlock the conversational analytics and decision logs, you will need a **FREE Groq API Key** which can be acquired instantly from `://groq.com`.

## ⚙️ Tech Stack Compiled
- **Frontend & Deployment:** Streamlit
- **Data Engineering:** Pandas & NumPy
- **Visual Engines:** Plotly Express
- **Core AI Agent Interpreter:** PandasAI
- **Machine Learning & Deep Neural Layers:** Scikit-Learn
- **Large Language Model Host:** Groq Cloud (Llama 3 Architecture)
-