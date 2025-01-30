# Moi Translator App

Welcome to **Moi Translator App**, a language translation tool designed to help users learn and communicate in the **Moi Papua language**. This project leverages advanced machine learning techniques to enable two-way translations between Moi Papua and Indonesian.

## 🚀 Features

- **Two-Way Translation:** Moi Papua to Indonesian and vice versa.
- **Phrase-Based and Dictionary Translations:** Supports individual words and complete phrases.
- **Model Persistence:** Trained models are saved for efficient usage.
- **AI-Powered Engine:** Built with **TF-IDF Vectorizer** and **LinearSVC** for accurate and fast translations.

## 🎯 Purpose
This project aims to:

- Simplify the process of learning the Moi Papua language.
- Assist communication between Moi speakers and non-Moi speakers.
- Preserve the Moi language by making it more accessible through technology.

## 🛠️ Tech Stack

- **Backend:** Flask
- **Machine Learning:** Scikit-learn, TF-IDF Vectorizer, LinearSVC
- **Data Storage:** Excel-based dataset (`Kamus MOI.xlsx`)
- **Serialization:** Joblib

## 🧩 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Elfaria-Wistoria/MoiTrans.git
   cd moi-translator
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the `Kamus MOI.xlsx` file in the project directory.

5. Run the application:
   ```bash
   python app.py
   ```

## 🌟 Usage

1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Use the translation interface to input text and choose the translation direction.
3. View the translation results.

## 📊 Model Training

If a pre-trained model is not available, the application will automatically train a new model using the provided dataset (`Kamus MOI.xlsx`). The training includes:

- Text preprocessing
- Feature extraction using **TF-IDF Vectorizer**
- Model training with **LinearSVC**

The model is evaluated and saved for future use.

## 📄 API Endpoints

- **`GET /`**: Home page
- **`GET /translate`**: Translation interface
- **`POST /api/translate`**: API for text translation

Example API Request:
```bash
curl -X POST -F "text=nja aba dun trung" -F "direction=moi_to_indo" http://127.0.0.1:5000/api/translate
```

## 📚 Dataset

The dataset (`Kamus MOI.xlsx`) contains Moi Papua and Indonesian word/phrase pairs used for training the translation models.

## 🔧 Project Structure

```
.
├── app.py               # Flask application
├── translator.py         # Translation logic and ML models
├── requirements.txt      # Project dependencies
├── templates/            # HTML templates
├── static/               # Static files (CSS, JS)
└── models/               # Saved models
```

## 🏆 Results

- Accurate two-way translations between Moi Papua and Indonesian.
- Phrase-based and individual word translations supported.

## 🤝 Contributing

Contributions are welcome! Feel free to fork this project and submit pull requests.

## 📜 License

This project is licensed under the MIT License.

## 🌐 Connect

If you have any questions or feedback, feel free to connect with me on [Instagram](https://www.instagram.com/edwinsyah.u?igsh=MXgwM2p6dDIyZnNmdQ%3D%3D&utm_source=qr) or check out my portfolio.
