import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class AdvancedMoiTranslator:
   def __init__(self):
       self.moi_to_indo_dict = {}
       self.indo_to_moi_dict = {}
       self.phrase_moi_to_indo = {}
       self.phrase_indo_to_moi = {}
       self.is_fitted = False
       
       # Inisialisasi TF-IDF Vectorizer
       self.tfidf_moi = TfidfVectorizer(
           analyzer='char_wb',
           ngram_range=(2, 4),
           max_features=5000,
           min_df=1,
           lowercase=True
       )
       
       self.tfidf_indo = TfidfVectorizer(
           analyzer='char_wb',
           ngram_range=(2, 4),
           max_features=5000,
           min_df=1,
           lowercase=True
       )
       
       self.model_moi_to_indo = None
       self.model_indo_to_moi = None

   def preprocess_text(self, text):
       if not isinstance(text, str):
           return ''
       
       text = text.lower()
       text = re.sub(r'\s+', ' ', text)
       text = text.strip()
       
       return text

   def create_phrase_dictionaries(self, df):
       for _, row in df.iterrows():
           moi = self.preprocess_text(row['Moi Papua'])
           indo = self.preprocess_text(row['Indonesia'])
           
           if len(moi.split()) > 1:
               self.phrase_moi_to_indo[moi] = indo
           if len(indo.split()) > 1:
               self.phrase_indo_to_moi[indo] = moi
               
           self.moi_to_indo_dict[moi] = indo
           self.indo_to_moi_dict[indo] = moi

   def train(self, file_path):
       print("Memulai proses training...")
       
       try:
           # Load dataset
           df = pd.read_excel(file_path)
           df.columns = ['Moi Papua', 'Indonesia']
           
           # Preprocess data
           df['Moi Papua'] = df['Moi Papua'].apply(self.preprocess_text)
           df['Indonesia'] = df['Indonesia'].apply(self.preprocess_text)
           
           # Handle multiple translations
           expanded_rows = []
           for _, row in df.iterrows():
               moi = row['Moi Papua']
               indo_translations = row['Indonesia'].split(';')
               for indo in indo_translations:
                   expanded_rows.append({
                       'Moi Papua': moi,
                       'Indonesia': indo.strip()
                   })
           
           df = pd.DataFrame(expanded_rows)
           
           # Create dictionaries
           self.create_phrase_dictionaries(df)
           
           print("Mempersiapkan fitur...")
           # Fit and transform the data
           X_moi = self.tfidf_moi.fit_transform(df['Moi Papua'])
           X_indo = self.tfidf_indo.fit_transform(df['Indonesia'])
           
           # Train models
           print("Melatih model...")
           self.model_moi_to_indo = LinearSVC(random_state=42)
           self.model_indo_to_moi = LinearSVC(random_state=42)
           
           self.model_moi_to_indo.fit(X_moi, df['Indonesia'])
           self.model_indo_to_moi.fit(X_indo, df['Moi Papua'])
           
           self.is_fitted = True
           print("Training selesai!")
           
           # Evaluate
           self._evaluate_models(df)
           return True
           
       except Exception as e:
           print(f"Error dalam training: {str(e)}")
           return False

   def _evaluate_models(self, df):
       if not self.is_fitted:
           print("Model belum di-training!")
           return
           
       try:
           X_moi = self.tfidf_moi.transform(df['Moi Papua'])
           X_indo = self.tfidf_indo.transform(df['Indonesia'])
           
           y_pred_indo = self.model_moi_to_indo.predict(X_moi)
           acc_moi_to_indo = accuracy_score(df['Indonesia'], y_pred_indo)
           
           y_pred_moi = self.model_indo_to_moi.predict(X_indo)
           acc_indo_to_moi = accuracy_score(df['Moi Papua'], y_pred_moi)
           
           print(f"\nAkurasi Moi -> Indonesia: {acc_moi_to_indo:.2%}")
           print(f"Akurasi Indonesia -> Moi: {acc_indo_to_moi:.2%}")
           
       except Exception as e:
           print(f"Error dalam evaluasi: {str(e)}")

   def translate(self, text, direction='moi_to_indo'):
       if not self.is_fitted:
           return "Error: Model belum di-training!"
           
       text = self.preprocess_text(text)
       
       if not text:
           return "Error: Teks tidak boleh kosong"
           
       try:
           if direction == 'moi_to_indo':
               # Check dictionaries first
               if text in self.phrase_moi_to_indo:
                   return self.phrase_moi_to_indo[text]
               if text in self.moi_to_indo_dict:
                   return self.moi_to_indo_dict[text]
               
               # Use model
               vector = self.tfidf_moi.transform([text])
               return self.model_moi_to_indo.predict(vector)[0]
               
           else:  # indo_to_moi
               if text in self.phrase_indo_to_moi:
                   return self.phrase_indo_to_moi[text]
               if text in self.indo_to_moi_dict:
                   return self.indo_to_moi_dict[text]
               
               vector = self.tfidf_indo.transform([text])
               return self.model_indo_to_moi.predict(vector)[0]
               
       except Exception as e:
           return f"Error dalam penerjemahan: {str(e)}"

   def save_model(self, path='models/moi_translator_model.joblib'):
       if not self.is_fitted:
           print("Model belum di-training!")
           return False
           
       try:
           # Create directory if it doesn't exist
           os.makedirs(os.path.dirname(path), exist_ok=True)
           
           model_data = {
               'tfidf_moi': self.tfidf_moi,
               'tfidf_indo': self.tfidf_indo,
               'model_moi_to_indo': self.model_moi_to_indo,
               'model_indo_to_moi': self.model_indo_to_moi,
               'moi_to_indo_dict': self.moi_to_indo_dict,
               'indo_to_moi_dict': self.indo_to_moi_dict,
               'phrase_moi_to_indo': self.phrase_moi_to_indo,
               'phrase_indo_to_moi': self.phrase_indo_to_moi,
               'is_fitted': self.is_fitted
           }
           
           joblib.dump(model_data, path)
           print(f"Model berhasil disimpan ke {path}")
           return True
           
       except Exception as e:
           print(f"Error dalam menyimpan model: {str(e)}")
           return False

   def load_model(self, path='models/moi_translator_model.joblib'):
       try:
           data = joblib.load(path)
           
           self.tfidf_moi = data['tfidf_moi']
           self.tfidf_indo = data['tfidf_indo']
           self.model_moi_to_indo = data['model_moi_to_indo']
           self.model_indo_to_moi = data['model_indo_to_moi']
           self.moi_to_indo_dict = data['moi_to_indo_dict']
           self.indo_to_moi_dict = data['indo_to_moi_dict']
           self.phrase_moi_to_indo = data['phrase_moi_to_indo']
           self.phrase_indo_to_moi = data['phrase_indo_to_moi']
           self.is_fitted = data.get('is_fitted', True)  # Handle older models
           
           print("Model berhasil dimuat!")
           return True
           
       except Exception as e:
           print(f"Error dalam memuat model: {str(e)}")
           return False

def test_translator():
   translator = AdvancedMoiTranslator()
   success = False
   
   # Try to load existing model
   if os.path.exists('models/moi_translator_model.joblib'):
       success = translator.load_model()
   
   # If loading failed or no model exists, train new model
   if not success:
       print("Melakukan training model baru...")
       if not os.path.exists("Kamus MOI.xlsx"):
           print("Error: File 'Kamus MOI.xlsx' tidak ditemukan!")
           return
       success = translator.train("Kamus MOI.xlsx")
       if success:
           translator.save_model()
   
   if not success:
       print("Gagal mempersiapkan model!")
       return
   
   test_cases = [
       ("nja aba dun trung", "moi_to_indo"),
       ("bete", "indo_to_moi"),
       ("abletsing", "moi_to_indo"),
       ("terbalik", "indo_to_moi")
   ]
   
   print("\nTest Translations:")
   for text, direction in test_cases:
       translation = translator.translate(text, direction)
       source_lang = "Moi Papua" if direction == "moi_to_indo" else "Indonesia"
       target_lang = "Indonesia" if direction == "moi_to_indo" else "Moi Papua"
       print(f"\n{source_lang}: {text}")
       print(f"{target_lang}: {translation}")

if __name__ == "__main__":
   test_translator()