# 🦠 Malaria Cell Classifier

CNN (Convolutional Neural Network) yordamida qon hujayrasi tasvirini tahlil qilib, uni **parazit yuqqan (Parasitic)** yoki **sog'lom (Uninfected)** deb tasniflovchi web ilova.

Foydalanuvchi hujayra rasmini yuklaydi, model esa natija va ishonchlilik darajasini (confidence) qaytaradi.

> ⚠️ Bu o'quv (educational) loyiha. Natijalar tibbiy tashxis o'rnini bosa olmaydi.

---

## ✨ Imkoniyatlar

- Rasm yuklash orqali tezkor bashorat
- Natija: `Parasitic` / `Uninfected` va ishonchlilik foizi
- FastAPI asosidagi yengil backend
- Oddiy HTML/CSS interfeys

## 🧠 Model

| | |
|---|---|
| Arxitektura | CNN (Keras / TensorFlow) |
| Kirish o'lchami | 128 × 128 RGB |
| Chiqish | Binary classification (sigmoid, chegara 0.5) |
| Model fayli | `malaria_cnn_model.h5` |

Rasm yuklanganda `RGB` ga o'tkaziladi, 128×128 ga o'zgartiriladi va `[0, 1]` oralig'iga normalizatsiya qilinadi.

## 🛠️ Texnologiyalar

- **Python 3.10+**
- **TensorFlow / Keras** — model
- **FastAPI + Uvicorn** — web server
- **Jinja2** — HTML shablonlar
- **Pillow, NumPy** — rasmni qayta ishlash

## 🚀 Ishga tushirish

```bash
git clone https://github.com/ifarohiddin/malaria-cell-classifier.git
cd malaria-cell-classifier

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload
```

So'ng brauzerda [http://127.0.0.1:8000](http://127.0.0.1:8000) manzilini oching.

## 📁 Loyiha tuzilmasi

```text
malaria-cell-classifier/
├── main.py                  # FastAPI ilova (/ va /predict)
├── malaria_cnn_model.h5     # O'qitilgan CNN model
├── malaria_model.pkl        # Qo'shimcha model fayli
├── requirements.txt
├── templates/
│   └── index.html           # Asosiy sahifa
└── static/
    └── style.css
```

## 🔌 API

| Method | Endpoint | Tavsif |
|---|---|---|
| `GET` | `/` | Rasm yuklash sahifasi |
| `POST` | `/predict` | Rasmni qabul qilib, natijani qaytaradi (`file` maydoni) |

## 🌱 Keyingi rejalar

- [ ] Modelning aniqlik (accuracy, precision, recall) natijalarini README'ga qo'shish
- [ ] Docker bilan ishga tushirish
- [ ] Demo skrinshot qo'shish
- [ ] Training notebook'ini repoga qo'shish

## 👤 Muallif

**Isomidinov Farohiddin** — [@ifarohiddin](https://github.com/ifarohiddin)
