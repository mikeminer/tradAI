<img width="955" height="666" alt="image" src="https://github.com/user-attachments/assets/bf53b921-bf84-4069-8e36-aa0065bf70b4" />



Scegli profilo: Scalping (1m/3m) o Swing (15m/1h).

Premi AVVIA → popup CPU/GPU (GPU compare solo se torch.cuda.is_available() è True).

Poi fa 2 fasi:

# 1) Setup (modello)

Se trova già i file model_*.pt, scaler_*.joblib, meta_*.joblib → li carica e parte.

Se non li trova → scarica lo storico KuCoin, crea il dataset e fa train/val/test + walk-forward OOS.

In questa fase fa tuning automatico (thr, horizon, epochs, confidenza).
Massimo 30 minuti: se non raggiunge il 90% si ferma comunque e usa il best trovato.

# 2) Realtime

Ogni pochi secondi:

prende le ultime candele (2 timeframe) + orderbook/funding/open interest (live)

calcola le feature, le normalizza, passa al modello

produce 3 probabilità: P(LONG), P(SHORT), P(NO TRADE)

decide il segnale:

NO TRADE se il modello lo prevede o se LONG/SHORT non supera la soglia di confidenza

altrimenti LONG o SHORT

se LONG/SHORT: calcola anche Take Profit (basato su ATR% + confidenza)

aggiorna grafici + rete neurale 3D (LONG verde, SHORT rosso, NO TRADE blu)

ARRESTA ferma il loop realtime in sicurezza.

# 🚀 tradai.py

### ETH AI Trading Dashboard — KuCoin

**LONG · SHORT · NO TRADE · Neural Network 3D · Auto-Training · Walk-Forward · Take Profit AI**

---

## 📌 Descrizione

**tradai.py** è una dashboard di trading avanzata in Python per **Windows**, progettata per analizzare il mercato **ETH/USDT su KuCoin** tramite intelligenza artificiale.

Il sistema è in grado di:

* scaricare automaticamente **dati storici reali**
* addestrare autonomamente una **rete neurale profonda**
* validare i risultati con **train / validation / test**
* eseguire **walk-forward out-of-sample**
* continuare il training finché non trova una configurazione stabile
* lavorare in **tempo reale**
* fornire segnali:

  * 🟢 **LONG**
  * 🔴 **SHORT**
  * 🔵 **NO TRADE**
* suggerire **Take Profit dinamico**
* mostrare una **rete neurale 3D animata**
* permettere la scelta **CPU o GPU NVIDIA (CUDA)** all’avvio
* fermare il sistema in sicurezza tramite **pulsante ARRESTA**

---

## ⚠️ Disclaimer

> Questo progetto è **solo a scopo educativo e sperimentale**.
> Non costituisce consulenza finanziaria.
> Il trading comporta rischio di perdita del capitale.
> Usare sempre **paper trading**, test indipendenti e gestione del rischio.

---

## ⭐ Raccomandato: Trading Tools – Python Launcher & Analytics

Per una gestione professionale degli script Python (ambienti, avvio rapido, debugging, analytics), è **fortemente consigliato** utilizzare questo launcher:

```
Trading Tools – Python Launcher & Analytics
https://github.com/mikeminer/Pythonlauncher-2
```

Questo tool consente:

* gestione semplice di Python 3.11
* avvio script con un click
* ambienti isolati
* organizzazione dei trading tools
* riduzione drastica degli errori pip / path / versioni

---

## 🧠 Funzionalità principali

### 🔹 Intelligenza Artificiale

* Rete neurale profonda multilayer
* 3 output:

  * LONG
  * SHORT
  * NO TRADE
* Decisione basata su probabilità reali (softmax)

### 🔹 Dataset avanzato

* OHLCV multi-timeframe
* Feature tecniche (RSI, MACD, volatilità, ritorni, log-features)
* Order Book (live)
* Funding rate (live)
* Open Interest (live)

### 🔹 Training serio

* Split cronologico:

  * Train
  * Validation
  * Test
* Nessun leakage temporale
* Walk-forward automatico
* Auto-tuning dei parametri:

  * soglia neutral (thr)
  * orizzonte futuro
  * epoche
  * confidence threshold
* Il training **non si ferma** finché non trova una configurazione valida

### 🔹 NO TRADE reale

Il sistema può decidere di **non operare** quando:

* il modello prevede NO TRADE
* la confidenza LONG/SHORT è sotto soglia

Questo riduce drasticamente:

* overtrading
* rumore di mercato
* segnali casuali

---

## 🎯 Take Profit AI

Quando il segnale è LONG o SHORT:

* TP calcolato tramite:

  * ATR %
  * confidenza del modello
  * profilo operativo (scalping / swing)
* Output mostrato come:

  * percentuale
  * distanza in prezzo
  * target stimato

---

## 🧬 Visualizzazione 3D

Rete neurale visualizzata in tempo reale:

| Output   | Colore   |
| -------- | -------- |
| LONG     | 🟢 Verde |
| SHORT    | 🔴 Rosso |
| NO TRADE | 🔵 Blu   |

Ogni tick aggiorna:

* input
* layer interni
* output neurale

---

## 🖥️ Interfaccia utente

Comandi volutamente **minimali**:

* selezione profilo:

  * Scalping (1m / 3m)
  * Swing (15m / 1h)
* pulsante **AVVIA**
* pulsante **ARRESTA**
* popup scelta **CPU / GPU NVIDIA**

Tutto il resto è automatico.

---

## ⚙️ Requisiti

* Windows 10 / 11
* Python 3.11 (consigliato)
* Connessione Internet
* Account KuCoin (solo lettura, nessuna API key richiesta)
* GPU NVIDIA opzionale (CUDA)

---

## 📦 Installazione

Sono forniti **4 file requirements già pronti**:

| Uso             | File                          |
| --------------- | ----------------------------- |
| Globale CPU     | `requirements_global_cpu.txt` |
| Globale GPU     | `requirements_global_gpu.txt` |
| Python 3.11 CPU | `requirements_py311_cpu.txt`  |
| Python 3.11 GPU | `requirements_py311_gpu.txt`  |

### Esempio (Python 3.11 CPU)

```powershell
py -3.11 -m pip install -r requirements_py311_cpu.txt
```

### Esempio (Python 3.11 GPU NVIDIA)

```powershell
py -3.11 -m pip install -r requirements_py311_gpu.txt
```

---

## ▶️ Avvio

```powershell
py -3.11 tradai.py
```

All’avvio apparirà un popup:

* CPU
* GPU NVIDIA (solo se CUDA è disponibile)

---

## ❓ Se vedi solo CPU

Significa che PyTorch non rileva CUDA.

Verifica:

```powershell
py -3.11 -c "import torch; print(torch.cuda.is_available())"
```

Se restituisce `False`, installa la versione CUDA di PyTorch o aggiorna i driver NVIDIA.

---

## 📁 File generati automaticamente

Durante l’esecuzione vengono creati:

* `model_*.pt` → rete neurale
* `scaler_*.joblib` → normalizzazione
* `meta_*.joblib` → parametri ottimali trovati

Questi file permettono:

* riavvio rapido
* nessun retraining inutile
* continuità operativa

---

## 🧪 Modalità consigliata

✔ Avviare inizialmente in osservazione
✔ Confrontare segnali con grafici reali
✔ Usare paper trading
✔ Analizzare winrate, drawdown, frequenza trade

---

## 🔒 Sicurezza

* Nessuna chiave API privata
* Nessuna operazione reale eseguita
* Nessun ordine inviato agli exchange
* Sistema puramente analitico

---

## 🗺️ Roadmap futura (facoltativa)

* Export segnali CSV
* Paper trading automatico
* Alert Telegram / Discord
* Modelli sequence (LSTM / Transformer)
* Modalità multi-asset
* Backtest grafico avanzato

---

## 📜 Licenza

Scegli liberamente (MIT consigliata).

---

## 👤 Autore

**pappardelle**
Ricerca autonoma su AI, finanza quantitativa e sistemi decisionali automatizzati.
