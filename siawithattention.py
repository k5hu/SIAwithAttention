import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D)


########################################
# use discoal to simulate data
########################################

def run_discoal(sample_size=198, num_replicates=1000, region_length=100000,
                Ne=10000, mu=2e-8, r=2e-8, selective=False):
    """
    use discoal to simulate data
    selective=False as Neutral 
    selective=True for selective sweeps
    """
    theta = 4 * Ne * mu * region_length
    rho = 4 * Ne * r * region_length

    cmd = [
        "discoal", str(sample_size), str(num_replicates), str(region_length),
        "-t", str(theta),
        "-r", str(rho)
    ]

    if selective:
        cmd += ["-ws", "0.05", "-a", "1000", "-x", "0.5"]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, text=True)
    output = result.stdout
    return output


########################################
# get genotype matrix
########################################

def parse_ms_output(ms_output):
    //
    segsites: X
    positions: ...
    0/1  Genotype line...

    Return a list that every element is a simulated result(genotype_matrix, positions)
    genotype_matrix shape as (sample size, mutation size)
    """
    lines = ms_output.strip().split('\n')
    results = []
    i = 0
    sample_size = 198
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("//"):
            i += 1
            if i >= len(lines):
                break
            seg_line = lines[i].strip()
            if not seg_line.startswith("segsites:"):
                i += 1
                continue
            segsites = int(seg_line.split(":")[1].strip())
            i += 1
            if i >= len(lines):
                break
            pos_line = lines[i].strip()
            if segsites > 0 and pos_line.startswith("positions:"):
                positions = pos_line.split()[1:]
                positions = np.array(positions, dtype=float)
                i += 1
                if i + sample_size > len(lines):
                    break
                geno_matrix = []
                for _ in range(sample_size):
                    hap = lines[i].strip()
                    i += 1
                    geno_matrix.append([int(x) for x in hap])
                geno_matrix = np.array(geno_matrix)
                results.append((geno_matrix, positions))
            else:
                i += 1
        else:
            i += 1
    return results


########################################
# extract ARG features from genome data
########################################

def extract_ARG_features_from_geno(geno_matrix, time_steps=50):
    if geno_matrix.size == 0:
	# no selection, return all 0s.
        return np.zeros((time_steps, 3))

    segsites = geno_matrix.shape[1]
    bins = np.linspace(0, segsites, time_steps + 1, dtype=int)
    features = []
    for i in range(time_steps):
        start, end = bins[i], bins[i + 1]
        if end == start:
            feats = [0, 0, 0]
        else:
            sub_geno = geno_matrix[:, start:end]
            DAF = sub_geno.mean(axis=0)
            feats = [DAF.mean(), DAF.var(), DAF.shape[0]]
        features.append(feats)
    features = np.array(features)
    return features


########################################
# create labels
########################################

def create_labels(num_samples, task_type="classification"):
    if task_type == "classification":
        labels = np.random.randint(0, 2, size=num_samples)
    else:
        labels = np.random.rand(num_samples)
    return labels


########################################
# data preprocess
########################################

def preprocess_data(X, y, task_type="classification", test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    if task_type == "classification":
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
    return X_train, X_test, y_train, y_test


########################################
# construct SIA (LSTM + Attention)
########################################

def build_sia_model_with_attention(input_shape, task_type="classification"):
    inputs = Input(shape=input_shape)  # (T, F)

    x = LSTM(100, return_sequences=True, activation='tanh')(inputs)
    x = Dropout(0.2)(x)

    # first attention
    att1 = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = x + att1
    x = LayerNormalization()(x)

    x = LSTM(100, return_sequences=True, activation='tanh')(x)
    x = Dropout(0.2)(x)

    # Second attention
    att2 = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = x + att2
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    if task_type == "classification":
        outputs = Dense(2, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae']

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=loss, metrics=metrics)
    return model


########################################
# contract SIA(LSTM with no Attention)
########################################

def build_sia_model_without_attention(input_shape, task_type="classification"):
    inputs = Input(shape=input_shape)  # (T, F)
    x = LSTM(100, return_sequences=True, activation='tanh')(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(100, return_sequences=True, activation='tanh')(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)

    if task_type == "classification":
        outputs = Dense(2, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae']

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=loss, metrics=metrics)
    return model


########################################
# train model
########################################

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    return history


########################################
# Evaluation（ROC curve）
########################################

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


########################################
# predictive_uncertainty
########################################

def predictive_uncertainty(model, X, n_samples=20):
    preds = []
    for _ in range(n_samples):
        pred = model(X, training=True).numpy()
        preds.append(pred)
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    return mean_pred, std_pred


########################################
# main
########################################

if __name__ == "__main__":
    task_type = "classification"

    # use discoal to generate neutral output
    neutral_output = run_discoal(selective=False)
    neutral_results = parse_ms_output(neutral_output)  # 列表[(geno_matrix, positions), ...]

    # use discoal generate selective sweeps
    sweep_output = run_discoal(selective=True)
    sweep_results = parse_ms_output(sweep_output)

    # combine neutral and selective sweeps  and take first 500 neutral and first 500 selective.
    # every replicate equals a sample. Set num_replicates=1000
    # neutral_results length=1000, sweep_results length=1000
    # each take first 500
    num_neu = 500
    num_swp = 500
    combined_results = neutral_results[:num_neu] + sweep_results[:num_swp]

    # extract ARG features
    # with every geno_matrix, sue extract_ARG_features_from_geno and get (T, F)= (50, 3)
    # final shape (N, T, F)
    X_list = []
    for geno_matrix, pos in combined_results:
        feats = extract_ARG_features_from_geno(geno_matrix, time_steps=50)
        X_list.append(feats)
    X = np.array(X_list)  # (N, 50, 3), N=1000

    # create label：first 500 neutral(0)，last 500 selective(1)
    y = np.array([0] * num_neu + [1] * num_swp)

    # preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(X, y, task_type=task_type)
    X_val, y_val = X_test, y_test

    # train model with attention
    model_with_att = build_sia_model_with_attention(input_shape=(X.shape[1], X.shape[2]), task_type=task_type)
    print("Model with attention summary:")
    model_with_att.summary()
    train_model(model_with_att, X_train, y_train, X_val, y_val, epochs=5, batch_size=64)

    # train model without attention
    model_without_att = build_sia_model_without_attention(input_shape=(X.shape[1], X.shape[2]), task_type=task_type)
    print("Model without attention summary:")
    model_without_att.summary()
    train_model(model_without_att, X_train, y_train, X_val, y_val, epochs=5, batch_size=64)

    # evaluate
    fpr_with, tpr_with, auc_with = evaluate_classification(model_with_att, X_test, y_test)
    fpr_without, tpr_without, auc_without = evaluate_classification(model_without_att, X_test, y_test)

    # ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr_with, tpr_with, color='darkorange', lw=2, label='With Attention (AUC = %0.2f)' % auc_with)
    plt.plot(fpr_without, tpr_without, color='blue', lw=2, label='Without Attention (AUC = %0.2f)' % auc_without)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()

    # predict uncertainty
    mean_pred_with, std_pred_with = predictive_uncertainty(model_with_att, X_test, n_samples=10)
    mean_pred_without, std_pred_without = predictive_uncertainty(model_without_att, X_test, n_samples=10)
    print("With attention - Mean predictions shape:", mean_pred_with.shape, "Std dev shape:", std_pred_with.shape)
    print("Without attention - Mean predictions shape:", mean_pred_without.shape, "Std dev shape:", std_pred_without.shape)
    print("Done.")
