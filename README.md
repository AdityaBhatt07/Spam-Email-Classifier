# 📧 Spam Email/SMS Classifier

A Machine Learning based Spam Email/SMS Classifier built using **scikit-learn** and **Streamlit**.
The app predicts whether a message is **SPAM** or **HAM (Not Spam)** and shows confidence scores.

Deployed using **Docker** on **AWS EC2**.

---

# 🚀 Features

* Spam/Ham message classification
* TF-IDF text vectorization
* Multiple ML models evaluated
* Interactive Streamlit web app
* Docker support
* AWS EC2 deployment

---

# 🛠️ Tech Stack

* Python
* scikit-learn
* pandas
* numpy
* Streamlit
* Docker
* AWS EC2

---

# 📂 Project Structure

```bash
spam-classifier/
│
├── app.py
├── train.py
├── spam.csv
├── requirements.txt
├── Dockerfile
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train Model

```bash
python train.py
```

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

App runs on:

```bash
http://localhost:8501
```

---

# 🐳 Docker

## Build Image

```bash
docker build -t spam-classifier .
```

## Run Container

```bash
docker run -p 8501:8501 spam-classifier
```

---

# ☁️ AWS EC2 Deployment

The application is deployed on AWS EC2 using Docker.

---

# 📌 Notes

* `model/` contains trained model files
* Do not upload `venv/` to GitHub

---

# 👨‍💻 Author

Built with Python, Machine Learning, and Streamlit.
.
