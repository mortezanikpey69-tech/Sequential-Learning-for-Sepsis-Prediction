# Sequential-Learning-for-Sepsis-Prediction
#LSTM and TRANSFORMER MODELS FOR SEPSIS PREDICTION:
Early detection of sepsis in ICU patients is critical, as timely intervention can substantially reduce mortality and improve patient outcomes. This study presents a systematic framework for sepsis prediction using publicly available ICU time-series data from over 40,000 patients. Unlike studies focused solely on maximizing predictive accuracy, our primary aim is to evaluate and compare sequential modeling approaches—including LSTM, BiLSTM, and Transformer architectures—under the challenges of class imbalance and real-world ICU data.
We implemented rigorous preprocessing pipelines, handling missing values, normalizing vital signs and laboratory measurements, and reducing noise to ensure reproducibility and scalability. Models were trained and evaluated using stratified train/validation/test splits, with performance metrics including precision, recall, F1-score, and AUROC. Threshold optimization was emphasized to balance sensitivity and specificity, directly impacting clinical decision-making.

#
#
#
#
#
#
# ---------------------- libraries and load data----------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from google.colab import drive
warnings.filterwarnings('ignore')

# mount Google Drive
drive.mount('/content/drive')
# بارگذاری داده‌های پردازش شده
def load_processed_data():
    with open('/content/drive/MyDrive/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    return data
# استفاده از داده‌های بارگذاری شده
data = load_processed_data()
X_seq = data['X_seq']
y_seq = data['y_seq']
patient_lengths = data['patient_lengths']
features = data['features']
patients = data['patients']

# ---------------------- LSTM MODEL ----------------------
# ---------------------- Padding ----------------------
padding_value = -999.0
max_timesteps = max(patient_lengths)
X_pad = pad_sequences(X_seq, maxlen=max_timesteps, padding='post', dtype='float32', value=padding_value)

# ---------------------- Normalization هوشمند ----------------------
mask = X_pad != padding_value
X_for_scaling = X_pad.copy()
X_for_scaling[~mask] = np.nan
means = np.nanmean(X_for_scaling, axis=(0,1))
stds = np.nanstd(X_for_scaling, axis=(0,1))
stds[stds==0] = 1.0
X_scaled = (X_pad - means)/stds
X_scaled[~mask] = padding_value

# ---------------------- تقسیم داده ------------------
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_seq, test_size=0.3, random_state=42, stratify=y_seq)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---------------------- Focal Loss ----------------------
def focal_loss(gamma=2., alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0-K.epsilon())
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        alpha_factor = y_true*alpha + (1-y_true)*(1-alpha)
        modulating_factor = K.pow(1.0-p_t, gamma)
        return -K.mean(alpha_factor * modulating_factor * K.log(p_t))
    return loss


n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

model = Sequential([
    Masking(mask_value=padding_value, input_shape=(n_timesteps, n_features)),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    Bidirectional(LSTM(64, dropout=0.2)),  # آخرین LSTM بدون return_sequences=True
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(0.001),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)


# ---------------------- وزن کلاس‌ها ----------------------
neg, pos = np.bincount(y_train.astype(int))
total = len(y_train)
w0 = (1 / neg) * total / 2.0
w1 = (1 / pos) * total / 2.0
max_ratio = 6.0
if w1 / w0 > max_ratio:
    w1 = w0 * max_ratio
class_weight = {0: w0, 1: w1}
print(f"Class weights: {class_weight}")


# ---------------------- Callbacks ----------------------
early_stop = EarlyStopping(monitor='val_recall', patience=15, restore_best_weights=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=5, min_lr=1e-7, verbose=1, mode='max')
checkpoint = ModelCheckpoint('deep_powerful_lstm_best.keras', monitor='val_recall', save_best_only=True, verbose=1, mode='max')

# ---------------------- آموزش مدل ----------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=64,
                    class_weight=class_weight,
                    callbacks=[early_stop, reduce_lr, checkpoint],
                    verbose=1,
                    shuffle=True)





# ---------------------- Transformer model ----------------------
# ------------------- Padding -------------------
PADDING_VALUE = -99.0
MAX_LEN = min(256, np.max(patient_lengths))  # سقف انتخابی (مثلا 256)

N = len(X_seq)
F = len(features)

X_padded = np.full((N, MAX_LEN, F), PADDING_VALUE, dtype=np.float32)
for i, arr in enumerate(X_seq):
    T = arr.shape[0]
    if T >= MAX_LEN:
        X_padded[i] = arr[-MAX_LEN:, :]   # آخرین MAX_LEN تایم‌استپ
    else:
        X_padded[i, :T, :] = arr

# ------------------- Normalization -------------------
# برای نرمال‌سازی، فقط روی مقادیر واقعی (نه پدینگ) استانداردسازی می‌کنیم
scaler = StandardScaler()
mask = (X_padded != PADDING_VALUE)
scaler.fit(X_padded[mask].reshape(-1, 1))   # fit روی مقادیر واقعی
X_norm = X_padded.copy()
X_norm[mask] = scaler.transform(X_padded[mask].reshape(-1, 1)).ravel()

print("X_norm shape:", X_norm.shape)
print("Sample patient (before/after norm):")
print("Before:", X_padded[0, :5, :3])
print("After :", X_norm[0, :5, :3])

# ------------------- Attention Mask -------------------
attn_mask = np.arange(MAX_LEN)[None, :] < patient_lengths[:, None]
print("attn_mask shape:", attn_mask.shape)

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(
    X_norm, y_seq, attn_mask, test_size=0.2, random_state=42, stratify=y_seq
)

X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(
    X_temp, y_temp, m_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# ---------------------- Positional Encoding اصلاح شده ----------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_len, d_model)

    def positional_encoding(self, seq_len, d_model):
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2*(i//2))/d_model)
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# ---------------------- مدل Transformer ----------------------
def build_transformer(n_timesteps, n_features, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.2):
    inputs = Input(shape=(n_timesteps, n_features))

    # تبدیل اولیه به ابعاد مدل
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(n_timesteps, d_model)(x)

    # بلوک Transformer
    for _ in range(2):  # دو بلوک
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        ff_output = Dense(ff_dim, activation='relu')(x)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# ---------------------- وزن کلاس‌ها ----------------------
neg, pos = np.bincount(y_train.astype(int))
total = len(y_train)
w0 = (1 / neg) * total / 2.0
w1 = (1 / pos) * total / 2.0
max_ratio = 6.0
if w1 / w0 > max_ratio:
    w1 = w0 * max_ratio
class_weight = {0: w0, 1: w1}
print(f"Class weights: {class_weight}")

# ---------------------- کامپایل ----------------------
model = build_transformer(n_timesteps=256, n_features=27, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.2)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
# ---------------------- ModelCheckpoint با val_recall ----------------------
mc = ModelCheckpoint(
    'best_transformer.keras',  # فرمت جدید Keras
    monitor='val_recall',      # متریک مورد نظر
    mode='max',                # چون دنبال بیشترین رکال هستیم
    save_best_only=True,
    verbose=1
)

# ---------------------- EarlyStopping ----------------------
es = EarlyStopping(
    monitor='val_recall',
    mode='max',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# ---------------------- ReduceLROnPlateau ----------------------
rlr = ReduceLROnPlateau(
    monitor='val_recall',
    mode='max',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# ---------------------- آموزش ----------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    class_weight=class_weight,
                     callbacks=[mc, es, rlr])




#perdict code

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

thresholds = [0.2, 0.4, 0.5, 0.6]

summary = []

for t in thresholds:
    y_pred = (y_pred_prob > t).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    summary.append({
        'Threshold': t,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-score': round(f1, 4),
        'Specificity': round(specificity, 4),
        'Accuracy': round(accuracy, 4),
        'AUROC': round(roc_auc_score(y_test, y_pred_prob), 4),
        'Average Precision': round(average_precision_score(y_test, y_pred_prob), 4)
    })

summary_df = pd.DataFrame(summary)
print(summary_df)



