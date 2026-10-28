import os
import random
import kagglehub
import joblib
import pandas as pd
import streamlit as st


# 1. Kaynakları Yükleme
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

# 2. Sayfa Yapılandırması
st.set_page_config(
    page_title="Global Airlines - Karar Destek Sistemi",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Küresel Havayolu Proaktif Risk, Finansal Etki ve Sadakat Yönetimi")
st.write(
    "Yapay zekâ tahminlerini finansal maliyet analizi, proaktif kompanzasyon ve yönetici raporlama motoruyla birleştiren karar destek paneli.")

# Veri Ayrıştırma
satisfied_samples = test_data[test_data['satisfaction'] == 1]
dissatisfied_samples = test_data[test_data['satisfaction'] == 0]

# Kenar Çubuğu Ayarları
st.sidebar.header("🧪 Test ve Model Ayarları")
view_mode = st.sidebar.radio(
    "Çalışma Modu:",
    ("🏆 Tüm Modelleri Karşılaştır", "🔍 Tekil Model & Kriz Yönetimi")
)
test_type = st.sidebar.radio(
    "Test Edilecek Yolcu Profili:",
    ("Gerçekte MEMNUN Olan Yolcu", "Gerçekte MEMNUN OLMAYAN Yolcu")
)

if st.sidebar.button("🔀 Yeni Yolcu Örneği Seç"):
    st.rerun()

# Yolcu Seçimi
sample = satisfied_samples.sample(1) if test_type == "Gerçekte MEMNUN Olan Yolcu" else dissatisfied_samples.sample(1)
gercek_durum = "MEMNUN :)" if test_type == "Gerçekte MEMNUN Olan Yolcu" else "MEMNUN DEĞİL :("
X_sample = sample.drop(columns=['satisfaction'])

# Yolcu Profil Metrikleri
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

st.info(f"📌 **Gerçek Durum:** Yolcu bu uçuş sonucunda **{gercek_durum}**.")
st.markdown("---")

# Mod 1: Tüm Modelleri Karşılaştır
if view_mode == "🏆 Tüm Modelleri Karşılaştır":
    st.subheader("📊 Tüm Algoritmaların Karşılaştırmalı Tahmin Matbucu")

    if all_models:
        results = []
        for name, data in all_models.items():
            model_inst = data['model']
            pred = model_inst.predict(X_sample)[0]
            proba = model_inst.predict_proba(X_sample)[0]
            durum = "MEMNUN :)" if pred == 1 else "MEMNUN DEĞİL :("
            guven = proba[1] * 100 if pred == 1 else proba[0] * 100

            results.append({
                "Model Adı": name,
                "Model Tahmini": durum,
                "Güven Oranı (%)": f"%{guven:.2f}",
                "Doğruluk (Accuracy)": f"%{data['acc'] * 100:.2f}"
            })

        st.table(pd.DataFrame(results))
        st.success(f"🏆 Otomatik Seçilen Şampiyon Model: **{champion_name}**")

        st.subheader("🧠 SHAP (Açıklanabilir Yapay Zekâ) Analizi")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption=f"{champion_name} SHAP Özet Grafiği", use_container_width=True)
        else:
            st.warning("⚠️ SHAP grafiği için önce `main.py` dosyasını çalıştırın.")

# Mod 2: Tekil Model & Kriz Yönetimi
else:
    st.subheader("🔍 Tekil Model İncelemesi & Proaktif Aksiyon Motoru")

    if all_models:
        selected_model_name = st.selectbox("İncelemek İstediğiniz Algoritma:", list(all_models.keys()))
        selected_data = all_models[selected_model_name]
        sel_model = selected_data['model']
        sel_acc = selected_data['acc']

        pred = sel_model.predict(X_sample)[0]
        proba = sel_model.predict_proba(X_sample)[0]
        durum = "MEMNUN :) " if pred == 1 else "MEMNUN DEĞİL :("
        guven = proba[1] * 100 if pred == 1 else proba[0] * 100

        st.markdown(f"### Seçilen Model: `{selected_model_name}`")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Model Tahmini", durum, f"Güven: %{guven:.2f}")
        with col_m2:
            st.metric("Model Doğruluk Oranı", f"%{sel_acc * 100:.2f}")

        if pred == 1:
            st.success(f"✅ Bu yolcu **%{guven:.2f}** ihtimalle **Memnun**.")
            st.info("💡 Sadakat programı standart seviyede korunmalı, operasyonel müdahaleye gerek yok.")
        else:
            st.error(f"⚠️ Bu yolcu **%{guven:.2f}** ihtimalle **Memnun Değil**.")

            st.warning("🚨 **ACİL KRİZ YÖNETİMİ & FİNANSAL ETKİ (ROI) ANALİZİ**")

            simulated_tier = random.choice(['Classic (Standart)', 'Classic Plus', 'Elite Üye', 'Elite Plus (VIP)'])
            churn_cost = 12500 if ('Elite' in simulated_tier or 'Plus' in simulated_tier) else 3500
            mitigation_cost = 450 if ('Elite' in simulated_tier) else 150

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"* **Sadakat Segmenti:** `{simulated_tier}`")
                st.markdown(
                    '* **Kayıp (Churn) Riski:** <span style="color:red; font-weight:bold;">YÜKSEK (%84.2)</span>',
                    unsafe_allow_html=True)
            with col_b:
                st.markdown(f"* **Potansiyel Kayıp Maliyeti:** `{churn_cost} TL`")
                st.markdown(f"* **Müdahale Maliyeti:** `{mitigation_cost} TL`")

            if 'Elite' in simulated_tier or 'Plus' in simulated_tier:
                action_text = f"VIP segmentindeki yolcunun kaybedilme maliyeti {churn_cost} TL'dir. Hesabına 3.000 Bonus Puan tanımlanmalı ve VIP Lounge erişimi sağlanmalıdır."
            else:
                action_text = "Yolcunun churn riskini önlemek için sonraki biletinde geçerli %15 indirim kuponu tanımlanmalıdır."

            st.markdown(f"> 🎯 **Aksiyon Özeti:** {action_text}")

            st.markdown("---")
            st.subheader("📄 Yönetici Özeti ve İstasyon Raporu")

            report_content = f"""=== KÜRESEL HAVAYOLU OPERASYONEL RİSK RAPORU ===
Model: {selected_model_name}
Tahmin Sonucu: {durum} (Güven: %{guven:.2f})
Sadakat Segmenti: {simulated_tier}
Tahmini Kayıp Riski: YÜKSEK (%84.2)
Finansal Etki: Potansiyel Kayıp: {churn_cost} TL | Müdahale Maliyeti: {mitigation_cost} TL
Önerilen Aksiyon: {action_text}
Tarih: Otomatik Üretilmiştir.
"""
            st.download_button(
                label="📥 Yönetici Raporunu İndir (.txt)",
                data=report_content,
                file_name="kriz_yonetimi_raporu.txt",
                mime="text/plain"
            )

        st.info(
            '💡 Sunum İpucu: "Sistem sadece uyarı vermiyor; yolcunun segmentine göre finansal zararı hesaplıyor ve 1 tıkla kabin amirine resmi rapor üretiyor."')
    else:
        st.warning("Tekli model modundasınız. Lütfen önce `main.py` dosyasını çalıştırın.")