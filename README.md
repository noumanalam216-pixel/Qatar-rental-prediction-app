---

## 📂 Dataset Features

The dataset contains rental listings with the following features:

| Feature | Description |
|------|------|
| location | Area in Qatar (Doha, Lusail, etc.) |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| area | Property size in square meters |
| property_type | Type of property |
| area_per_room | Feature engineered variable |

Target variable:


price (monthly rent in QAR)


---

## ⚙️ Feature Engineering

Additional feature created:


area_per_room = area / bedrooms


This helps the model better understand **space efficiency per room**.

---

## 🔧 Model Pipeline

The project uses a **Scikit-Learn Pipeline**:


ColumnTransformer
├── Numerical Features
└── Categorical Features (OneHotEncoder)

↓
GradientBoostingRegressor


Benefits:

- automatic preprocessing
- consistent training and prediction pipeline
- prevents data leakage

---

## 📈 Model Performance

Evaluation metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

Example results:


MAE : 0.20
RMSE : 0.26
R² : 0.64


Average rent prediction error:


≈ 1958 QAR


---

## 🖥️ Streamlit Web Application

The web app allows users to:

1. Select location
2. Enter number of bedrooms
3. Enter number of bathrooms
4. Enter property area
5. Predict monthly rent

Output includes:

- predicted rent
- estimated price range
- rent per square meter

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle

---

## 📁 Project Structure


Qatar-Rental-Properties-Prediction
│
├── app.py
├── rent_model.pkl
├── qatar_rentals_training.ipynb
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run Locally

Clone the repository:

```bash
git clone https://github.com/yourusername/qatar-rental-price-predictor.git

Go to project folder:

cd qatar-rental-price-predictor

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

💡 Future Improvements

Possible upgrades:

add more property features
include map-based location data
integrate real estate APIs
deploy as a full SaaS real estate tool
👨‍💻 Author

Nouman Ali

Machine Learning & Data Science Enthusiast