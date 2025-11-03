from fastapi import FastAPI
from pydantic import BaseModel
import joblib, re

app = FastAPI(title="Chat Spam & Toxic Detector API")

# ====== Load model và vectorizer (.pkl) ======
model = joblib.load("chat_filter_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ====== Hàm xử lý văn bản ======
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹà-ỹ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ====== API schema ======
class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "✅ API đang hoạt động! Gửi POST /predict để kiểm tra."}

@app.post("/predict")
def predict_message(msg: Message):
    cleaned = clean_text(msg.text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return {"text": msg.text, "prediction": pred}
