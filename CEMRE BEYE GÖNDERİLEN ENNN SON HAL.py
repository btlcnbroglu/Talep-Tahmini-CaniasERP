import requests
import xml.etree.ElementTree as ET  # XML işleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from flask import Flask, Response # XML çıktısı için

predictions_df_global = None # Global değişken; tahmin DataFrame’ini saklamak için


# 1. Web Servisten Veri Çekme ve DataFrame Oluşturma
def fetch_webservice_data():
    url = "http://localhost:5000/api/WebService/CallService"
    payload = {
        "Client": "00",
        "Language": "T",
        "DBServer": "CANIAS",
        "DBName": "IAS803RDBDEV",
        "ApplicationServer": "localhost:27499",
        "Username": "BETULTEST",
        "Password": "B12345.",
        "Encrypted": False,
        "Compression": False,
        "LCheck": "",
        "VKey": "",
        "ServiceId": "WEBTESTPANDAS2",
        "Parameters": "<PARAMETERS><PARAM>param1</PARAM></PARAMETERS>",
        "Compressed": False,
        "Permanent": False,
        "ExtraVariables": "",
        "RequestId": 0
    }
    headers = {
        "Content-Type": "application/json",
        "accept": "text/plain"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
    except Exception as e:
        print("HTTP isteği gönderilirken hata oluştu:", e)
        return None

    if response.status_code == 200:
        print("İstek başarılı, veri alındı.")
        try:
            json_response = response.json()
        except Exception as e:
            print("Yanıtı JSON formatına çeviremedik:", e)
            return None

        if "Data" not in json_response or "Response" not in json_response["Data"] or "Value" not in \
                json_response["Data"]["Response"]:
            print("Beklenen veri yapısı bulunamadı.")
            return None

        xml_data = json_response["Data"]["Response"]["Value"]

        try:
            root = ET.fromstring(xml_data)
            records = []
            for element in root.findall('.//element'):
                record = {}
                for child in element:
                    record[child.tag] = child.text
                records.append(record)

            if not records:
                print("XML verisinden herhangi bir kayıt elde edilemedi.")
                return None

            df = pd.DataFrame(records)
            return df
        except Exception as e:
            print("XML verisi DataFrame'e dönüştürülemedi:", e)
            return None
    else:
        print("İstek başarısız. Durum kodu:", response.status_code)
        print("Hata mesajı:", response.text)
        return None


# 2. Talep Tahmini Çalışması (LSTM ve XGBoost)
#############################################
def create_dataset_lstm(dataset, look_back=12):
    X, Y = [], []
    if len(dataset) <= look_back:
        print("Veri seti, look_back değerinden küçük. Dataset oluşturulamıyor.")
        return np.array(X), np.array(Y)
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def create_dataset_xgb(dataset, look_back=12):
    X, Y = [], []
    if len(dataset) <= look_back:
        print("Veri seti, look_back değerinden küçük. Dataset oluşturulamıyor.")
        return np.array(X), np.array(Y)
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def forecast_demand(sales_data, look_back=12, apply_log_transform=False):
    predictions_df = None

    if apply_log_transform:
        sales_data['Satış Miktarı'] = np.log1p(sales_data['Satış Miktarı'])

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(sales_data.values)
    except Exception as e:
        print("Veri ölçeklendirme sırasında hata:", e)
        return None

    X_lstm, Y_lstm = create_dataset_lstm(scaled_data, look_back=look_back)
    if X_lstm.size == 0 or Y_lstm.size == 0:
        print("LSTM için oluşturulan veri seti boş.")
        return None
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
    X_xgb, Y_xgb = create_dataset_xgb(sales_data.values, look_back=look_back)
    if X_xgb.size == 0 or Y_xgb.size == 0:
        print("XGBoost için oluşturulan veri seti boş.")
        return None

    tss = TimeSeriesSplit(n_splits=5)
    metrics = {
        'LSTM': {'r2': [], 'corr': [], 'mape': [], 'y_true': [], 'y_pred': []},
        'XGBoost': {'r2': [], 'corr': [], 'mape': [], 'y_true': [], 'y_pred': []}
    }
    best_models = {
        'LSTM': {'r2': float('-inf'), 'model': None, 'weights': None, 'corr': None, 'mape': None, 'fold': -1},
        'XGBoost': {'r2': float('-inf'), 'model': None, 'corr': None, 'mape': None, 'fold': -1}
    }

    fold_number = 1

    for train_index, test_index in tss.split(X_lstm):
        print(f"\n=== Fold {fold_number} ===")
        try:
            X_train_lstm, X_test_lstm = X_lstm[train_index], X_lstm[test_index]
            Y_train_lstm, Y_test_lstm = Y_lstm[train_index], Y_lstm[test_index]
            X_train_xgb, X_test_xgb = X_xgb[train_index], X_xgb[test_index]
            Y_train_xgb, Y_test_xgb = Y_xgb[train_index], Y_xgb[test_index]
        except Exception as e:
            print(f"Veri setlerini bölme sırasında hata (fold {fold_number}):", e)
            fold_number += 1
            continue

        try:
            model_lstm = Sequential([
                Input(shape=(look_back, 1)),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(25),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model_lstm.fit(X_train_lstm, Y_train_lstm, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])
            predicted_lstm = model_lstm.predict(X_test_lstm)
            predicted_lstm_inv = scaler.inverse_transform(predicted_lstm)
            Y_test_inv = scaler.inverse_transform(Y_test_lstm.reshape(-1, 1))
            if apply_log_transform:
                predicted_lstm_inv = np.expm1(predicted_lstm_inv)
                Y_test_inv = np.expm1(Y_test_inv)
            r2_lstm = r2_score(Y_test_inv, predicted_lstm_inv)
            corr_matrix_lstm = np.corrcoef(Y_test_inv.reshape(-1), predicted_lstm_inv.reshape(-1))
            pearson_corr_lstm = corr_matrix_lstm[0, 1]
            mape_lstm = mean_absolute_percentage_error(Y_test_inv, predicted_lstm_inv)
            print("LSTM:")
            print(f"R^2: {r2_lstm:.4f}")
            print(f"Korelasyon Katsayısı: {pearson_corr_lstm:.4f}")
            print(f"MAPE: {mape_lstm * 100:.2f}%")
            metrics['LSTM']['r2'].append(r2_lstm)
            metrics['LSTM']['corr'].append(pearson_corr_lstm)
            metrics['LSTM']['mape'].append(mape_lstm)
            metrics['LSTM']['y_true'].extend(Y_test_inv.flatten())
            metrics['LSTM']['y_pred'].extend(predicted_lstm_inv.flatten())
            if r2_lstm > best_models['LSTM']['r2']:
                best_models['LSTM']['r2'] = r2_lstm
                best_models['LSTM']['model'] = model_lstm
                best_models['LSTM']['weights'] = model_lstm.get_weights()
                best_models['LSTM']['corr'] = pearson_corr_lstm
                best_models['LSTM']['mape'] = mape_lstm
                best_models['LSTM']['fold'] = fold_number
        except Exception as e:
            print(f"LSTM modeli eğitilirken hata (fold {fold_number}):", e)

        try:
            model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200,
                                         learning_rate=0.05, max_depth=5, random_state=42)
            model_xgb.fit(X_train_xgb, Y_train_xgb)
            predicted_xgb = model_xgb.predict(X_test_xgb)
            r2_xgb = r2_score(Y_test_xgb, predicted_xgb)
            corr_matrix_xgb = np.corrcoef(Y_test_xgb, predicted_xgb)
            pearson_corr_xgb = corr_matrix_xgb[0, 1]
            mape_xgb = mean_absolute_percentage_error(Y_test_xgb, predicted_xgb)
            print("XGBoost:")
            print(f"R^2: {r2_xgb:.4f}")
            print(f"Korelasyon Katsayısı: {pearson_corr_xgb:.4f}")
            print(f"MAPE: {mape_xgb * 100:.2f}%")
            metrics['XGBoost']['r2'].append(r2_xgb)
            metrics['XGBoost']['corr'].append(pearson_corr_xgb)
            metrics['XGBoost']['mape'].append(mape_xgb)
            metrics['XGBoost']['y_true'].extend(Y_test_xgb)
            metrics['XGBoost']['y_pred'].extend(predicted_xgb)
            if r2_xgb > best_models['XGBoost']['r2']:
                best_models['XGBoost']['r2'] = r2_xgb
                best_models['XGBoost']['model'] = model_xgb
                best_models['XGBoost']['corr'] = pearson_corr_xgb
                best_models['XGBoost']['mape'] = mape_xgb
                best_models['XGBoost']['fold'] = fold_number
        except Exception as e:
            print(f"XGBoost modeli eğitilirken hata (fold {fold_number}):", e)

        fold_number += 1

    print("\n=== Cross-Validation Sonuçları ===")
    for model_name in metrics:
        print(f"\n{model_name}:")
        for i in range(len(metrics[model_name]['r2'])):
            print(f"Fold {i + 1}: R^2 = {metrics[model_name]['r2'][i]:.4f}, "
                  f"Korelasyon = {metrics[model_name]['corr'][i]:.4f}, "
                  f"MAPE = {metrics[model_name]['mape'][i] * 100:.2f}%")

    if best_models['LSTM']['r2'] > best_models['XGBoost']['r2']:
        best_model_name = 'LSTM'
        best_model_details = best_models['LSTM']
    else:
        best_model_name = 'XGBoost'
        best_model_details = best_models['XGBoost']

    print(f"\n=== En İyi Model ===")
    print(f"Model: {best_model_name}")
    print(f"Fold: {best_model_details['fold']}")
    print(f"R^2: {best_model_details['r2']:.4f}")
    print(f"Korelasyon Katsayısı: {best_model_details['corr']:.4f}")
    print(f"MAPE: {best_model_details['mape'] * 100:.2f}%")

    try:
        if best_model_name == 'LSTM':
            best_model = Sequential([
                Input(shape=(look_back, 1)),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(25),
                Dense(1)
            ])
            best_model.compile(optimizer='adam', loss='mean_squared_error')
            best_model.set_weights(best_model_details['weights'])
            future_inputs = scaled_data[-look_back:]
            future_inputs = future_inputs.reshape((1, look_back, 1))
            future_predictions = []
            for i in range(6):
                prediction = best_model.predict(future_inputs)
                future_predictions.append(prediction[0, 0])
                prediction = prediction.reshape(1, 1, 1)
                future_inputs = np.concatenate((future_inputs[:, 1:, :], prediction), axis=1)
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            if apply_log_transform:
                future_predictions = np.expm1(future_predictions)
        elif best_model_name == 'XGBoost':
            best_model = best_model_details['model']
            future_inputs = sales_data.values[-look_back:]
            future_predictions = []
            for i in range(6):
                prediction = best_model.predict(future_inputs.reshape(1, -1))
                future_predictions.append(prediction[0])
                prediction = prediction.reshape(1, 1)
                future_inputs = np.concatenate((future_inputs[1:], prediction), axis=0)
            future_predictions = np.array(future_predictions).reshape(-1, 1)

        last_date = sales_data.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
        predictions_df = pd.DataFrame({
            "Tarih": [date.strftime('%Y-%m') for date in future_dates],
            "Tahmin": np.round(future_predictions).astype(int).flatten()
        })
        print("\n=== Gelecek 6 Aylık Tahminler (DataFrame) ===")
        print(predictions_df.to_string(index=False))
    except Exception as e:
        print("Gelecek tahminleri hesaplanırken hata oluştu:", e)

    try:
        # Gerçek vs Tahmin grafiklerini çizme
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(sales_data.index[look_back:look_back + len(metrics['LSTM']['y_true'])],
                 metrics['LSTM']['y_true'], label='Gerçek Değerler')
        plt.plot(sales_data.index[look_back:look_back + len(metrics['LSTM']['y_pred'])],
                 metrics['LSTM']['y_pred'], label='LSTM Tahminleri')
        plt.title('LSTM: Gerçek vs Tahmin Değerleri')
        plt.xlabel('Tarih')
        plt.ylabel('Satış Miktarı')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(sales_data.index[look_back:look_back + len(metrics['XGBoost']['y_true'])],
                 metrics['XGBoost']['y_true'], label='Gerçek Değerler')
        plt.plot(sales_data.index[look_back:look_back + len(metrics['XGBoost']['y_pred'])],
                 metrics['XGBoost']['y_pred'], label='XGBoost Tahminleri')
        plt.title('XGBoost: Gerçek vs Tahmin Değerleri')
        plt.xlabel('Tarih')
        plt.ylabel('Satış Miktarı')
        plt.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Gerçek vs Tahmin grafiklerinin çizilmesinde hata:", e)

    try:
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        sns.boxplot(data=[metrics['LSTM']['r2'], metrics['XGBoost']['r2']], palette="Set2")
        plt.xticks([0, 1], ['LSTM', 'XGBoost'])
        plt.ylabel('R² Değeri')
        plt.title('Cross-Validation R² Değerleri')

        plt.subplot(1, 3, 2)
        sns.boxplot(data=[metrics['LSTM']['corr'], metrics['XGBoost']['corr']], palette="Set3")
        plt.xticks([0, 1], ['LSTM', 'XGBoost'])
        plt.ylabel('Korelasyon Katsayısı')
        plt.title('Cross-Validation Korelasyon Katsayıları')

        plt.subplot(1, 3, 3)
        sns.boxplot(data=[np.array(metrics['LSTM']['mape']) * 100, np.array(metrics['XGBoost']['mape']) * 100],
                    palette="Set1")
        plt.xticks([0, 1], ['LSTM', 'XGBoost'])
        plt.ylabel('MAPE (%)')
        plt.title('Cross-Validation MAPE Değerleri')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Cross-Validation metrik grafiklerinin çizilmesinde hata:", e)

    try:
        plt.figure(figsize=(10, 5))
        last_date = sales_data.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
        future_dates_str = [date.strftime('%Y-%m') for date in future_dates]
        plt.plot(future_dates_str, np.round(future_predictions).flatten(), marker='o', label='Gelecek Tahminler')
        plt.title(f'En İyi Model: {best_model_name} ile Gelecek 6 Aylık Tahminler')
        plt.xlabel('Tarih')
        plt.ylabel('Tahmin Edilen Satış Miktarı')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Gelecek tahmin grafiklerinin çizilmesinde hata:", e)

    return predictions_df


# 3. DataFrame’i XML'e Dönüştürme Fonksiyonu
#############################################
def dataframe_to_xml(df, root_tag='Results', row_tag='Record'):
    root = ET.Element(root_tag)
    for index, row in df.iterrows():
        record = ET.SubElement(root, row_tag)
        for col in df.columns:
            child = ET.SubElement(record, col)
            child.text = str(row[col])
    return ET.tostring(root, encoding='utf-8').decode('utf-8')


# 4.XML Servisini Oluşturma
#############################################
app = Flask(__name__)
@app.route('/xml_taleptahmini')
def xml_taleptahmini():
    global predictions_df_global
    if predictions_df_global is not None:
        xml_str = dataframe_to_xml(predictions_df_global)
        return Response(xml_str, mimetype='application/xml')
    else:
        return Response("Tahmin verisi henüz oluşturulmadı.", mimetype='text/plain')


#############################################
# 5. Ana Program Akışı
#############################################
if __name__ == "__main__":
    try:
        df_web = fetch_webservice_data()
        if df_web is not None:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)

            print("\nOluşturulan DataFrame:")
            print(df_web.to_string())

            if 'QUANTITY' not in df_web.columns or 'VALIDFROM' not in df_web.columns:
                print("Uyarı: 'QUANTITY' veya 'VALIDFROM' sütunu bulunamadı. Lütfen sütun isimlerini kontrol edin.")
            else:
                sales_data = df_web[['VALIDFROM', 'QUANTITY']].copy()
                try:
                    sales_data['VALIDFROM'] = pd.to_datetime(sales_data['VALIDFROM'], format='%d.%m.%Y',errors='coerce')
                except Exception as e:
                    print("Tarih verileri datetime formatına çevrilemedi:", e)
                sales_data.sort_values(by='VALIDFROM', inplace=True)
                sales_data.set_index('VALIDFROM', inplace=True)
                sales_data.rename(columns={'QUANTITY': 'Satış Miktarı'}, inplace=True)
                sales_data['Satış Miktarı'] = pd.to_numeric(sales_data['Satış Miktarı'], errors='coerce')
                if sales_data.isnull().values.any():
                    sales_data.fillna(method='ffill', inplace=True)
                if sales_data.empty:
                    print("Satış verileri içeren DataFrame boş.")
                else:
                    predictions_df_global = forecast_demand(sales_data, look_back=12, apply_log_transform=False)
                    if predictions_df_global is not None:
                        print("Tahminler başarıyla oluşturuldu. XML servisi başlatılıyor...")
                        print(
                            "XML sonuçlarına şu URL üzerinden ulaşabilirsiniz: http://127.0.0.1:5001/xml_taleptahmini")
                        app.run(debug=True, port=5001, use_reloader=False)
                    else:
                        print("Tahminler oluşturulamadı.")
        else:
            print("Web servisten veri alınamadı.")
    except Exception as e:
        print("Program çalışırken beklenmeyen bir hata oluştu:", e)


