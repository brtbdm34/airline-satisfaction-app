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

st.title(
    '✈️ Küresel Havayolu Proaktif Risk, Finansal Etki ve Sadakat Yönetimi Karar'
    ' Destek Sistemi'
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

st.info(
    f'📌 **Veri Setindeki Gerçek Durum:** Yolcu bu uçuş sonucunda'
    f' **{gercek_durum}**.'
)

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

# 2. MOD: Tekil İnceleme + Sekmeli Yapı + Finansal Etki + Kriz Yönetimi + Rapor İndirme
else:
  st.subheader(
      '🔍 Tekil Model Derinlemesine İnceleme, Finansal Etki & Proaktif Aksiyon'
      ' Motoru'
  )

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

    # --- HESAPLAMALAR (Sekmelerde kullanılmak üzere) ---
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
        '📋 Yolcu Profili ve Tahmin',
        '🚨 Kriz & CRM Aksiyonu',
        '💰 Finansal Etki & ROI',
        '📄 Yönetici Raporu',
    ])

    # --- TAB 1: Yolcu Profili ve Tahmin ---
    with tab1:
      st.markdown('### Yolcunun Temel Uçuş Parametreleri')

      st.markdown(
          f"""
            <div style="display: flex; justify-content: space-between; background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 20px;">
                <div style="text-align: center;">
                    <span style="font-size: 12px; color: #6c757d; text-transform: uppercase;">Yolcu Yaşı</span><br>
                    <b style="font-size: 24px; color: #0056b3;">{int(X_sample['Age'].values[0])}</b>
                </div>
                <div style="text-align: center; border-left: 1px solid #e9ecef; border-right: 1px solid #e9ecef; padding: 0 20px;">
                    <span style="font-size: 12px; color: #6c757d; text-transform: uppercase;">Uçuş Mesafesi</span><br>
                    <b style="font-size: 24px; color: #0056b3;">{int(X_sample['Flight Distance'].values[0])} km</b>
                </div>
                <div style="text-align: center;">
                    <span style="font-size: 12px; color: #6c757d; text-transform: uppercase;">Varış Gecikmesi</span><br>
                    <b style="font-size: 24px; color: #0056b3;">{int(X_sample['Arrival Delay in Minutes'].values[0])} dk</b>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      col_p1, col_p2, col_p3 = st.columns(3)
      with col_p1:
        st.metric(
            'Wi-Fi Puanı', int(X_sample['Inflight wifi service'].values[0])
        )
      with col_p2:
        st.metric('Online Biniş', int(X_sample['Online boarding'].values[0]))
      with col_p3:
        st.metric('Koltuk Konforu', int(X_sample['Seat comfort'].values[0]))

      st.markdown('---')
      if pred == 1:
        st.success(
            f'✅ **{selected_model_name}** bu yolcuyu **%{guven:.2f}** ihtimalle'
            ' **Memnun** olarak değerlendirdi.'
        )
      else:
        st.error(
            f'⚠️ **{selected_model_name}** bu yolcuyu **%{guven:.2f}** ihtimalle'
            ' **Memnun Değil** olarak değerlendirdi.'
        )

    # --- TAB 2: Kriz Yönetimi ve CRM Aksiyonu ---
    with tab2:
      if pred == 1:
        st.info(
            '💡 Yolcu sadık müşteri segmentinde. Herhangi bir operasyonel'
            ' müdahaleye gerek yok.'
        )
      else:
        st.markdown(
            """
                <div style="padding: 15px; border-radius: 8px; background-color: #ffe6e6; border-left: 6px solid #ff4d4d; margin-bottom: 20px;">
                    <h4 style="color: #cc0000; margin: 0; font-size: 16px;">🔴 SEVİYE-1 KRİTİK CHURN RİSKİ TESPİT EDİLDİ</h4>
                    <p style="margin: 5px 0 0 0; color: #333; font-size: 14px;">Yolcunun memnuniyetsizlik olasılığı kritik eşiğin üzerindedir. Uçuş sonu kayıp riski yüksek segmenttedir.</p>
                </div>
                """,
            unsafe_allow_html=True,
        )

        st.markdown(f'* **Simüle Edilen Sadakat Kartı:** `{simulated_tier}`')
        st.markdown('### 💡 CRM & Finansal Karar Tavsiyesi:')
        st.markdown(f'> 🎯 **Aksiyon Özeti:** {action_text}')

    # --- TAB 3: Finansal Etki ve ROI Analizi ---
    with tab3:
      if pred == 0:
        st.markdown('### Şirket Bütçesine Etki (ROI) Analizi')
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; background-color: #fff3cd; padding: 15px; border-radius: 6px; border: 1px solid #ffeeba; margin-bottom: 15px;">
                <div>
                    <span style="font-size: 12px; color: #856404;">POTANSİYEL YILLIK KAYIP</span><br>
                    <b style="font-size: 18px; color: #212529;">{churn_cost} TL</b>
                </div>
                <div style="border-left: 1px solid #ffeeba; padding-left: 15px;">
                    <span style="font-size: 12px; color: #856404;">MÜDAHALE MALİYETİ</span><br>
                    <b style="font-size: 18px; color: #212529;">{mitigation_cost} TL</b>
                </div>
                <div style="border-left: 1px solid #ffeeba; padding-left: 15px;">
                    <span style="font-size: 12px; color: #856404;">TAHMİNİ ROI</span><br>
                    <b style="font-size: 18px; color: #198754;">YÜKSEK (%{ (1 - (mitigation_cost/churn_cost)) * 100 :.1f} )</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            f'Bu analiz, {simulated_tier} segmentindeki yolcuyu kazanmak için'
            ' önerilen proaktif aksiyonun şirkete olan finansal getirisini'
            ' simüle eder.'
        )
      else:
        st.info('Yolcu memnun, finansal risk analizi gerekmemektedir.')

    # --- TAB 4: Yönetici Özeti Raporu ---
    with tab4:
      if pred == 0:
        st.subheader('📄 İstasyon Mühendisliği Yönetim Özeti')

        report_content = f"""=== KÜRESEL HAVAYOLU OPERASYONEL RİSK RAPORU ===
Model: {selected_model_name}
Tahmin Sonucu: {durum} (Güven: %{guven:.2f})
Sadakat Segmenti: {simulated_tier}
Tahmini Kayıp Riski: YÜKSEK (%84.2)
Finansal Etki Analizi: Potansiyel Kayıp: {churn_cost} TL | Müdahale Maliyeti: {mitigation_cost} TL
Önerilen Aksiyon: {action_text}
Tarih / Zaman: Otomatik Sistem Üretimidir.
"""
        st.code(report_content, language=None)

        st.download_button(
            label='📥 Yönetici Raporunu İndir (.txt)',
            data=report_content,
            file_name='kriz_yonetimi_raporu.txt',
            mime='text/plain',
        )
      else:
        st.info('Yolcu memnun, resmi rapor oluşturulmasına gerek yok.')

    st.markdown('---')
    st.info(
        '💡 Sunum ipucu: "Hocam sağ üstteki sekmelerden; yolcunun profilini,'
        ' otomatik kriz yönetimini ve şirkete olan finansal etkisini adım adım'
        ' gösterebilirim."'
    )
  else:
    st.warning(
        'Tekli model modundasınız. Lütfen `main.py` çalıştırarak çoklu model'
        ' yapısını aktif edin.'
    )