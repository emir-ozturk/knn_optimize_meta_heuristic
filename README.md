# KNN Optimize Meta Heuristic API

Bu proje, meta-heuristik algoritmalar kullanarak KNN parametrelerini optimize eden FastAPI backend uygulamasıdır.

## 🎯 Özellikler

- **Meta-Heuristik Algoritmalar**: PSO, GA, DE, SA, WOA
- **Yüksek İterasyon Desteği**: Maksimum 10.000 iterasyon
- **Büyük Popülasyon**: 500'e kadar popülasyon boyutu
- **Dosya Desteği**: CSV ve Excel formatları
- **Clean Architecture**: SOLID prensiplerine uygun modüler yapı

## 📊 Parametreler

- `max_iterations`: Maksimum iterasyon sayısı (1-10.000, varsayılan: 50)
- `population_size`: Popülasyon boyutu (5-500, varsayılan: 20)
- `algorithm`: Optimizasyon algoritması (PSO, GA, DE, SA, WOA)
- `target_column`: Hedef değişken sütun adı

## 🚀 Hızlı Başlangıç

```bash
pip install -r requirements.txt
python -m app.main
```

API dokümantasyonu: http://localhost:8000/docs