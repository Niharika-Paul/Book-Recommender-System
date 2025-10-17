# 📚 Book Recommendation System using Matrix Factorization and Neural Networks

## 🧩 Overview
This project implements a **Book Recommendation System** that predicts user preferences and suggests top-rated books using collaborative filtering and neural network approaches.  
It combines **Matrix Factorization (ALS)** and **Neural Networks (MLP + CNN)** to model both numerical and visual data, inspired by research on the role of book covers in predicting ratings.

## 🧠 Motivation
Recommender systems are essential in helping users navigate large datasets on platforms like Goodreads and Amazon.  
The goal of this project is to design a hybrid model capable of improving recommendation accuracy while maintaining diversity in book discovery.

## 📊 Dataset
The project uses Goodreads data containing:
- **Books metadata:** title, author, genre, year, ISBN  
- **User ratings:** ≈2,000,000 ratings from ≈15,000 users across ≈8,000 books  
- **Cover images:** 5,928 images resized to 32x32 pixels  

**Data Split:**  
- Train: 65%  
- Validation: 15%  
- Test: 20%

## ⚙️ Methodology
### 1. Baseline Models
- **Popularity-based Model:** Ranks books by total ratings.
- **Naïve Bayes:** Predicts user interest based on text from book snippets.

### 2. TF-IDF (Item–Item Affinity)
- Converts book snippets to TF-IDF vectors.
- Uses **Cosine Similarity** to infer user ratings from similar books.

### 3. Matrix Factorization (ALS)
- Decomposes the user–book rating matrix into latent features.
- Optimized parameters (latent factors `d=3`, regularization=0.05) achieved **RMSE = 0.834**.

### 4. Neural Networks
- **MLP:** Predicts ratings using categorical and numerical features.
- **CNN:** Learns visual patterns from book covers.
- **Combined Model (MLP + CNN):** Outperformed individual models by ~1.5%.

## 🧾 Results
| Model                | RMSE      | nDCG     | Diversity (Div10) | Remarks            |
|----------------------|-----------|----------|------------------ |------------------- |
| Popularity           | High      | Low      | Low               | Simple baseline    |
| TF-IDF               | 0.85      | High     | **Best**          | High diversity     |
| Matrix Factorization | **0.834** | High     | Low               | Best RMSE          |
| CNN Only             | 1.02      | Low      | Moderate          | Weak standalone    |
| MLP + CNN            | **0.82**  | **Best** | Good              | Best overall model |

## 🚀 Future Work
- Improve diversity in Matrix Factorization results  
- Build hybrid ensemble models combining MF and NN  
- Use attention-based networks for deeper personalization  

## 🧰 Libraries Used
- Python, NumPy, Pandas, Scikit-learn, SciPy  
- Keras / TensorFlow  
- OpenCV, NLTK, Matplotlib  

## 📁 Project Structure
```
📦 Book-Recommender
 ┣ 💻 ML_book_recommender.ipynb
 ┣ 📄 Book_Recommender_Project_Report.pdf
 ┣ 📘 research_paper.pdf
 ┣ 🖼️ poster
 ┗ 📄 README_Book_Recommender.md
```

---

### 📚 Reference
*Logé, C., & Yoffe, A. (2016). Building the Optimal Book Recommender and Measuring the Role of Book Covers in Predicting User Ratings.*

---
> *This work was completed as part of the Machine Learning Mini Project for the course Machine Learning, at PES University, Bangalore.
