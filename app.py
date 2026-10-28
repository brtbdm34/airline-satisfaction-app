import streamlit as st
import pandas as pd
import joblib
import os
import kagglehub
import matplotlib.pyplot as plt


# 1. Kaydedilmiş modelleri ve test verisini yüklüyoruz
@st.cache_resource
def load_resources():
    saved_data = joblib.load('best_airline_model.pkl')

    if isinstance(saved_data, dict) and 'all_models' in saved_data:
        all_models = saved_data['all_models']
        champion_name = saved_data['name']
        champion_model = saved_data['model']
    else:
        champion_model = saved_data
        champion_name = "Şampiyon Model"
        all_models = None

    path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")

    test_df = pd.read_csv(os.path.join(path, "test.csv"))
    test_df = test_df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    test_df['Arrival Delay in Minutes'] = test_df['Arrival Delay in Minutes'].fillna(
        test_df['Arrival Delay in Minutes'].median())

    for col in test_df.select_dtypes(include=['object']).columns:
        test_df[col] = test_df[col].astype('category').cat.codes

    return champion_model, champion_name, all_models, test_df


champion_model, champion_name, all_models, test_data = load_resources()

st.title("✈️ Havayolu Yolcu Memnuniyet Analiz ve Doğrulama Paneli")
st.write(
    "Bu panel, 4 farklı algoritmanın performansını hem toplu karşılaştırma hem de tekil model inceleme modlarıyla test etmeni sağlar.")

# Gerçek memnun ve memnun olmayan yolcuları ayıralım
satisfied_samples = test_data[test_data['satisfaction'] == 1]
dissatisfied_samples = test_data[test_data['satisfaction'] == 0]

st.sidebar.header("🧪 Test ve Model Ayarları")

# Görünüm Modu Seçimi (Tümünü Karşılaştır veya Tekil Model İncele)
view_mode = st.sidebar.radio(
    "Çalışma Modu:",
    ("🏆 Tüm Modelleri Karşılaştır", "🔍 Tekil Model İncele (Hoca Sunumu İçin)")
)

test_type = st.sidebar.radio(
    "Test Etmek İstediğiniz Yolcu Profili:",
    ("Gerçekte MEMNUN Olan Yolcu", "Gerçekte MEMNUN OLMAYAN Yolcu")
)

if st.sidebar.button("🔀 Yeni Yolcu Örneği Seç"):
    st.rerun()

# Yolcu seçimi
if test_type == "Gerçekte MEMNUN Olan Yolcu":
    sample = satisfied_samples.sample(1)
    gercek_durum = "MEMNUN :)"
else:
    sample = dissatisfied_samples.sample(1)
    gercek_durum = "MEMNUN DEĞİL :("

X_sample = sample.drop(columns=['satisfaction'])

# Yolcunun profil özellikleri
st.subheader("📋 Seçilen Yolcunun Profil Özellikleri")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Yolcu Yaşı", int(X_sample['Age'].values[0]))
    st.metric("Uçuş Mesafesi", int(X_sample['Flight Distance'].values[0]))
with col2:
    st.metric("Wi-Fi Puanı (0-5)", int(X_sample['Inflight wifi service'].values[0]))
    st.metric("Online Biniş Puanı (0-5)", int(X_sample['Online boarding'].values[0]))
with col3:
    st.metric("Koltuk Konforu (0-5)", int(X_sample['Seat comfort'].values[0]))
    st.metric("Varış Gecikmesi (dk)", int(X_sample['Arrival Delay in Minutes'].values[0]))

st.info(f"📌 **Veri Setindeki Gerçek Durum:** Yolcu bu uçuş sonucunda **{gercek_durum}**.")

st.markdown("---")

# 1. MOD: Tüm Modelleri Karşılaştır ve SHAP Göster
if view_mode == "🏆 Tüm Modelleri Karşılaştır":
    st.subheader("📊 Tüm Algoritmaların Karşılaştırmalı Tahmin Matbucu")

    if all_models:
        comparison_results = []
        for name, data in all_models.items():
            model_inst = data['model']
            pred = model_inst.predict(X_sample)[0]
            proba = model_inst.predict_proba(X_sample)[0]

            durum = "MEMNUN :)" if pred == 1 else "MEMNUN DEĞİL :("
            guven = proba[1] * 100 if pred == 1 else proba[0] * 100

            comparison_results.append({
                "Model Adı": name,
                "Model Tahmini": durum,
                "Güven Oranı (%)": f"%{guven:.2f}",
                "Test Başarısı (Accuracy)": f"%{data['acc'] * 100:.2f}"
            })

        result_df = pd.DataFrame(comparison_results)
        st.table(result_df)
        st.success(f"🏆 Otomatik Seçilen Şampiyon Model: **{champion_name}**")

        # SHAP Görseli
        st.subheader("🧠 SHAP (Açıklanabilir Yapay Zekâ) Karar Analizi")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption=f"{champion_name} - SHAP Özet Grafiği", use_container_width=True)
        else:
            st.warning("⚠️ SHAP grafiği için önce `main.py` dosyasını çalıştırın.")

# 2. MOD: İstediğin Modeli Tekil Olarak İncele
else:
    st.subheader("🔍 Tekil Model Derinlemesine İnceleme Modu")

    if all_models:
        selected_model_name = st.selectbox(
            "İncelemek ve Sunmak İstediğiniz Algoritmayı Seçin:",
            list(all_models.keys())
        )

        selected_data = all_models[selected_model_name]
        sel_model = selected_data['model']
        sel_acc = selected_data['acc']

        pred = sel_model.predict(X_sample)[0]
        proba = sel_model.predict_proba(X_sample)[0]

        durum = "MEMNUN :)" if pred == 1 else "MEMNUN DEĞİL :("
        guven = proba[1] * 100 if pred == 1 else proba[0] * 100

        st.markdown(f"### Seçilen Model: `{selected_model_name}`")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Modelin Bu Yolcu İçin Tahmini", durum, f"Güven: %{guven:.2f}")
        with col_m2:
            st.metric("Modelin Genel Doğruluk Oranı", f"%{sel_acc * 100:.2f}")

        if pred == 1:
            st.success(
                f"✅ **{selected_model_name}** bu yolcuyu **%{guven:.2f}** ihtimalle **Memnun** olarak değerlendirdi.")
        else:
            st.error(
                f"⚠️ **{selected_model_name}** bu yolcuyu **%{guven:.2f}** ihtimalle **Memnun Değil** olarak değerlendirdi.")

        st.info(
            "💡 Sunum sırasında bu ekrandan **Decision Tree** ile **XGBoost** veya **Random Forest** arasında geçiş yaparak her bir algoritmanın aynı yolcu profiline nasıl tepki verdiğini jüriye gösterebilirsin.")
    else:
        st.warning("Tekli model modundasınız. Lütfen `main.py` çalıştırarak çoklu model yapısını aktif edin.")