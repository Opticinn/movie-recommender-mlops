# 🎬 Movie Recommender V2: The "Zero-Dollar" Enterprise MLOps Pipeline 
<img width="1912" height="1076" alt="image" src="https://github.com/user-attachments/assets/4f49f781-fdc0-46ee-855f-cb587a20e359" />


![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?logo=supabase)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-F9AB00?logo=huggingface)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=github-actions)

> *"Building an ML model is 20% of the work. Engineering the system to sustain it in production is the other 80%."*

Welcome to Phase 2 of my Machine Learning journey! This project evolved from a standard Jupyter Notebook experiment into a **fully automated, production-ready MLOps ecosystem**. It’s designed to be smart, self-monitoring, and completely free to run.

---

## 🚀 Live Links
- **🎯 Try the Recommender App:** [Insert Streamlit App Link]
- **📊 View the Live MLOps Dashboard:** [Insert Dashboard Link]

---

## 🔥 What Makes This System Special?

### 🧠 1. The Brain: Hybrid Recommendation Engine
Moving beyond basic metadata, this system actually understands context.
* **Collaborative Filtering (SVD):** Learns from complex audience rating patterns and biases.
* **Content-Based NLP (SBERT):** Uses Sentence-Transformers to read and understand movie synopses.
* **The Result:** A dynamically weighted Hybrid Score that recommends movies based on both *what* the movie is about and *who* likes it.

### 🚨 2. The Nervous System: Proactive Monitoring
<img width="1919" height="1075" alt="image" src="https://github.com/user-attachments/assets/a9a3baba-08a6-4334-b99d-560525a540e6" />

I don't believe in staring at dashboards waiting for things to break. 

* **Real-time Drift Detection:** Continuously monitors for shifts in genre distributions and rating baselines.
* **Latency Tracking:** Ensures the Hybrid Engine serves predictions in under `100ms`.
* **Telegram Alerts:** If the system detects an anomaly (e.g., latency spikes or high data drift), it instantly fires a customized alert straight to my phone.

### 🤖 3. The Stress Test: Automated Traffic Bot
How do you validate an MLOps pipeline without thousands of real users? **You build your own.**
* **GitHub Actions Cron Job:** A custom Python script runs completely serverless 8 times a day.
* **Behavior Simulation:** It mimics real user search patterns, hits the Hybrid API, and logs the latency and prediction results into the database.
* **Continuous Validation:** This ensures the monitoring dashboard is always alive and the drift detection logic is actively stress-tested.

### ⚡ 4. The "Zero-Dollar" Architecture
Enterprise-grade infrastructure doesn't have to break the bank. This entire pipeline is architected to run flawlessly on free-tier services:
* **Database & API:** Supabase (PostgreSQL)
* **Compute & CI/CD:** GitHub Actions (2,000 free minutes/month)
* **Model Registry:** Hugging Face Hub
* **Frontend:** Streamlit Community Cloud

---

* **Alerting Proof:**
<img width="480" height="1068" alt="image" src="https://github.com/user-attachments/assets/64458efd-1946-46f0-8955-ecbb7625edbe" />

---

## 💡 What I Learned

Building this project was a deep dive into the reality of production-grade Machine Learning. Here are my key takeaways:

* **MLOps is the Backbone:** I learned that developing a high-performing model is only a small part of the journey. Engineering the surrounding infrastructure—monitoring, automation, and reliability—is what actually makes a project "production-ready."
* **Proactive over Reactive:** Relying on manual dashboard checks is a recipe for failure. I learned how to implement automated alerting systems (via Telegram) to catch issues like latency spikes and data drift before they affect the end-user experience.
* **The Reality of Data Drift:** Models are not "set and forget." I gained hands-on experience in detecting how real-world data distributions can shift over time and why constant validation against a baseline is crucial for maintaining recommendation quality.
* **Automation as a Force Multiplier:** By leveraging GitHub Actions for more than just simple testing—such as running daily traffic simulations—I learned how to maintain a dynamic system with zero manual oversight.
* **Architectural Resourcefulness:** I discovered the power of integrating diverse platforms (Supabase, Hugging Face, Streamlit) to build a sophisticated, enterprise-grade pipeline without requiring a massive budget or complex server management.

---

## 🛠️ Project Architecture

```text
├── .github/workflows/       # CI/CD & Automation (Daily Fetch, Traffic Bot, Testing)
├── app/
│   ├── monitoring.py        # Streamlit MLOps Dashboard
│   ├── recommender.py       # Main Streamlit UI
│   └── utils/               # Database, API, and Model Loading logic
├── scripts/
│   ├── simulate_traffic.py  # The Automated Traffic Bot
│   └── fetch_ratings.py     # Automated Data Ingestion
└── requirements.txt         # Dependency management
