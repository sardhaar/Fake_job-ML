# 🕵️ Fake Job Postings Detection - SVM with TF-IDF

## 📌 Objective
This project aims to build a **machine learning model** to detect **fraudulent job postings** based on job description text and other attributes. Using **TF-IDF vectorization** and a **Support Vector Machine (SVM) classifier**, the model identifies whether a given job posting is **real or fake**.

This work addresses the growing issue of **online recruitment scams**, helping users and organizations ensure safer hiring practices.

---

## 🧪 Steps Performed

### 🔹 1. Data Loading and Preprocessing
- Dataset: *Fake Job Postings Dataset*  
- Loaded via pandas (`fake_job_postings.csv`)  
- Initial exploration with `.head()`, `.info()`, `.describe()`  
- Handled missing values and irrelevant fields  

### 🔹 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: Identified imbalance between real and fake postings  
- **Text Length Analysis**: Compared lengths of real vs. fake job descriptions  
- **Word Clouds & Common Terms**: Visualized frequently used words in fraudulent postings  

### 🔹 3. Text Preprocessing
- Lowercasing, punctuation removal, and stopword removal  
- Tokenization & Lemmatization  
- Conversion of text data into numerical form using **TF-IDF Vectorizer**  

### 🔹 4. Feature Engineering
- Extracted features from job descriptions and metadata  
- Transformed textual fields (`title`, `requirements`, `description`) into **TF-IDF vectors**  
- Combined with additional features (if available)  

### 🔹 5. Model Training
- Split data into **train (80%) and test (20%)**  
- Applied **SVM Classifier** for binary classification  
- Used **GridSearchCV** to fine-tune hyperparameters (`C`, `kernel`, `gamma`)  

### 🔹 6. Model Evaluation
- Metrics Used:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  
- Achieved strong performance with high precision on detecting fake postings  

### 🔹 7. Model Deployment (Flask API)
- Built a **Flask app (`app.py`)** to serve predictions  
- Accepts job posting text as input and returns **Real / Fake prediction**  
- Integrated with saved artifacts:  
  - `svm_fake_job_model.pkl` → Trained SVM model  
  - `tfidf_vectorizer.pkl` → Vectorizer used during training  

---

## 🔍 Key Observations
- Fake postings often use **generic, vague language** and **too-good-to-be-true offers**  
- Real postings show **clear structure** and more **detailed requirements**  
- **SVM performed well** compared to baseline models due to its robustness with high-dimensional text data  
- **TF-IDF was effective** in capturing important textual signals for classification  

---

## 🛠 Tech Stack
- **Python 3.11+**  
- **Pandas / NumPy** → Data handling  
- **Matplotlib / Seaborn** → Visualization  
- **Scikit-learn** → ML modeling (SVM, TF-IDF, metrics)  
- **NLTK / Regex** → Text preprocessing  
- **Flask** → Model deployment  

---

## 📁 Files Included
- `Train.ipynb` → Full notebook with preprocessing, training, and evaluation  
- `app.py` → Flask app for deployment  
- `fake_job_postings.csv` → Dataset used  
- `svm_fake_job_model.pkl` → Saved SVM model  
- `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer  
- `requirements.txt` → Dependencies  
- `sample.txt` → Sample job posting for testing API  

---

## 🚀 How to Run the Project

### 1. Clone the repo
git clone <https://github.com/sardhaar/Fake_job-ML.git>
cd fake-job-detection
## 🚀 How to Run the Project

### 2. Install dependencies
```bash
pip install -r requirements.txt
3. Run the Flask app
bash
Copy code
python app.py
4. Test the API
Send a job posting text through the UI or API endpoint

Get prediction:

✅ Real

❌ Fake

🧑‍💻 Author
Tarun sada
Python Developer | Data Scientist | ML Engineer | Full Stack Developer | Ethical Hacker Enthusiast | Prompt Engineer

📌 GitHub Repo: [https://github.com/sardhaar/Fake_job-ML.git]

⚡ This project demonstrates the practical application of NLP + ML in detecting fraudulent online content, showcasing end-to-end workflow from data preprocessing → model training → deployment