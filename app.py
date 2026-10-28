import os
import random
import kagglehub
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


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
    champion_name = 'Şampiyon Model'
    all_models = None

  path = kagglehub.dataset_download('teejmahal20/airline-passenger-satisfaction')

  test_df = pd.read_csv(os.path.join(path, 'test.csv'))
  test_df = test_df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
  test_df['Arrival Delay in Minutes'] = test_df[
      'Arrival Delay in Minutes'
  ].fillna(test_df['Arrival Delay in Minutes'].median())

  for col in test_df.select_dtypes(include=['object']).columns:
    test_df[col] = test_df[col].astype('category').cat.codes

  return champion_model, champion_name, all_models, test_df


champion_model, champion_name, all_models, test_data = load_resources()

st.set_page_config(
    page_title='Global Airlines Proaktif Yolcu Deneyimi ve Sadakat Sistemi',
    page_icon='✈️',
    layout='wide',
)

# Üst Başlık (Görseldeki gibi temiz ve kurumsal)
st.title('✈️ KÜRESEL HAVAYOLU PROAKTİF RİSK VE SADAKAT YÖNETİMİ')
st.markdown(
    '**Geliştirici:** Berat Badem | **Sistem:** Kurumsal Karar Destek ve'
    ' Proaktif CRM Motoru'
)
st.write(
    'Bu sistem; yapay zekâ tahminlerini finansal maliyet analizi, proaktif'
    ' kompanzasyon ve yönetici raporlama motoruyla birleştirir.'
)

# Gerçek memnun ve memnun olmayan yolcuları ayıralım
satisfied_samples = test_data[test_data['satisfaction'] == 1]
dissatisfied_samples = test_data[test_data['satisfaction'] == 0]

st.sidebar.header('🧪 Test ve Model Ayarları')

# Görünüm Modu Seçimi
view_mode = st.sidebar.radio(
    'Çalışma Modu:',
    (
        '🏆 Tüm Modelleri Karşılaştır',
        '🔍 Tekil Model, Finansal Etki & Kriz Yönetimi',
    ),
)

test_type = st.sidebar.radio(
    'Test Etmek İstediğiniz Yolcu Profili:',
    ('Gerçekte MEMNUN Olan Yolcu', 'Gerçekte MEMNUN OLMAYAN Yolcu'),
)

if st.sidebar.button('🔀 Yeni Yolcu Örneği Seç'):
  st.rerun()

# Yolcu seçimi
if test_type == 'Gerçekte MEMNUN Olan Yolcu':
  sample = satisfied_samples.sample(1)
  gercek_durum = 'MEMNUN :)'
else:
  sample = dissatisfied_samples.sample(1)
  gercek_durum = 'MEMNUN DEĞİL :('

X_sample = sample.drop(columns=['satisfaction'])

st.markdown('---')

# 1. MOD: Tüm Modelleri Karşılaştır ve SHAP Göster
if view_mode == '🏆 Tüm Modelleri Karşılaştır':
  st.subheader('📊 Tüm Algoritmaların Karşılaştırmalı Tahmin Matbucu')

  if all_models:
    comparison_results = []
    for name, data in all_models.items():
      model_inst = data['model']
      pred = model_inst.predict(X_sample)[0]
      proba = model_inst.predict_proba(X_sample)[0]

      durum = 'MEMNUN :)' if pred == 1 else 'MEMNUN DEĞİL :('
      guven = proba[1] * 100 if pred == 1 else proba[0] * 100

      comparison_results.append({
          'Model Adı': name,
          'Model Tahmini': durum,
          'Güven Oranı (%)': f'%{guven:.2f}',
          'Test Başarısı (Accuracy)': f"%{data['acc'] * 100:.2f}",
      })

    result_df = pd.DataFrame(comparison_results)
    st.table(result_df)
    st.success(f'🏆 Otomatik Seçilen Şampiyon Model: **{champion_name}**')

    # SHAP Görseli
    st.subheader('🧠 SHAP (Açıklanabilir Yapay Zekâ) Karar Analizi')
    if os.path.exists('shap_summary.png'):
      st.image(
          'shap_summary.png',
          caption=f'{champion_name} SHAP Özet Grafiği',
          use_container_width=True,
      )
    else:
      st.warning('⚠️ SHAP grafiği için önce `main.py` dosyasını çalıştırın.')

# 2. MOD: Tekil İnceleme + Görsel İstediğin Gibi Tasarlanan Sekmeli Yapı
else:
  st.subheader('🔍 Tekil Model Derinlemesine İnceleme & Proaktif Aksiyon Motoru')

  if all_models:
    selected_model_name = st.selectbox(
        'İncelemek ve Sunmak İstediğiniz Algoritmayı Seçin:',
        list(all_models.keys()),
    )

    selected_data = all_models[selected_model_name]
    sel_model = selected_data['model']
    sel_acc = selected_data['acc']

    pred = sel_model.predict(X_sample)[0]
    proba = sel_model.predict_proba(X_sample)[0]

    durum = 'MEMNUN :)' if pred == 1 else 'MEMNUN DEĞİL :('
    guven = proba[1] * 100 if pred == 1 else proba[0] * 100

    st.markdown(f'### Seçilen Model: `{selected_model_name}`')

    # --- HESAPLAMALAR ---
    tier_list = [
        'Classic (Standart)',
        'Classic Plus',
        'Elite Üye',
        'Elite Plus (VIP)',
    ]
    simulated_tier = random.choice(tier_list)

    churn_cost = (
        12500
        if ('Elite' in simulated_tier or 'Plus' in simulated_tier)
        else 3500
    )
    mitigation_cost = 450 if ('Elite' in simulated_tier) else 150

    if 'Elite' in simulated_tier or 'Plus' in simulated_tier:
      action_text = (
          f'VIP segmentindeki bu yolcunun kaybedilme maliyeti {churn_cost}'
          ' TLdir. Acil olarak hesabına 3.000 Bonus Puan (Mil) tanımlanmalı ve'
          ' VIP Lounge erişimi sağlanmalıdır.'
      )
    else:
      action_text = (
          'Yolcunun churn riskini önlemek için sonraki biletinde geçerli %15'
          ' indirim kuponu tanımlanmalıdır.'
      )

    # --- SEKMELİ YAPI (st.tabs) ---
    tab1, tab2, tab3, tab4 = st.tabs([
        '📋 Yolcu Profili',
        '🚨 Kriz & Aksiyon',
        '💰 Finansal ROI',
        '📄 Rapor İndir',
    ])

    # --- TAB 1: Yolcu Profili (Görsel Tasarıma Birebir Uyumlu) ---
    with tab1:
      st.markdown(
          '### Yolcu Örneği Analizi & Özellikleri'
      )

      c1, c2, c3 = st.columns([2, 1.2, 1])

      # Sol Kolon: Parametre Kartları Grid Yapısı
      with c1:
        st.markdown('**Seçilen Yolcunun Profil Özellikleri**')
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
          st.markdown(
              f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 10px;">
                        <span style="font-size: 11px; color: #6c757d;">🧳 Uçuş Mesafesi</span><br>
                        <b style="font-size: 15px; color: #0056b3;">{int(X_sample['Flight Distance'].values[0])} km</b>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 10px;">
                        <span style="font-size: 11px; color: #6c757d;">⏱️ Varış Gecikmesi</span><br>
                        <b style="font-size: 15px; color: #dc3545;">{int(X_sample['Arrival Delay in Minutes'].values[0])} dk</b>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef;">
                        <span style="font-size: 11px; color: #6c757d;">💻 Online Biniş</span><br>
                        <b style="font-size: 15px; color: #0056b3;">Puan: {int(X_sample['Online boarding'].values[0])}/5</b>
                    </div>
                    """,
              unsafe_allow_html=True,
          )
        with sub_c2:
          st.markdown(
              f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 10px;">
                        <span style="font-size: 11px; color: #6c757d;">👤 Yolcu Yaşı</span><br>
                        <b style="font-size: 15px; color: #0056b3;">{int(X_sample['Age'].values[0])}</b>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 10px;">
                        <span style="font-size: 11px; color: #6c757d;">💺 Koltuk Konforu</span><br>
                        <b style="font-size: 15px; color: #0056b3;">Puan: {int(X_sample['Seat comfort'].values[0])}/5</b>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #e9ecef;">
                        <span style="font-size: 11px; color: #6c757d;">📶 Wi-Fi Hizmeti</span><br>
                        <b style="font-size: 15px; color: #0056b3;">Puan: {int(X_sample['Inflight wifi service'].values[0])}/5</b>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

      # Orta Kolon: Yapay Zekâ Tahmin Sonucu Kutusu
      with c2:
        st.markdown('**Yapay Zekâ Tahmin Sonucu**')
        if pred == 1:
          st.markdown(
              f"""
                    <div style="background-color: #d1e7dd; padding: 18px; border-radius: 8px; border: 1px solid #badbcc; text-align: center; margin-bottom: 10px;">
                        <h3 style="color: #0f5132; margin: 0; font-size: 20px;">✅ MEMNUN :)</h3>
                        <p style="color: #0f5132; margin: 5px 0 0 0; font-size: 14px;">Güven: %{guven:.1f}</p>
                    </div>
                    <p style="font-size: 11px; color: #6c757d;">✔️ Yolcu yüksek olasılıkla memnun. Mevcut hizmet kalitesi korunmalı.</p>
                    """,
              unsafe_allow_html=True,
          )
        else:
          st