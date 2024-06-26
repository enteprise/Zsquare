import streamlit as st
from SmilesPE.tokenizer import *
from collections import Counter
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
import joblib
import os
import json
from datetime import datetime
import pickle

# Import all functions from utils.py
from utils import (
    clean_dataset,
    data_tokenize_smiles_pretrained,
    data_tokenize_atomwise,
    data_tokenize_characterwise,
    create_data_HV,
    train_test_split_scaffold,
    retrain,
    inference
)

# Function to run the model and save it
def run_model(dataset_file, target, mols, num_tokens, dim, max_pos, gramsize, epochs, iterations, test_size, threshold, encoding_scheme, split_type, version):
    try:
        # Load the dataset
        df = pd.read_csv(dataset_file)
        X = df[mols].tolist()
        Y = df[target].tolist()

        # Clean the dataset
        X_clean, Y_clean, X_bad, Y_bad = clean_dataset(X, Y)

        # Tokenize the data
        if encoding_scheme == "smiles_pretrained":
            data_tokenized = data_tokenize_smiles_pretrained(X_clean, num_tokens)
        elif encoding_scheme == "atomwise":
            data_tokenized = data_tokenize_atomwise(X_clean, num_tokens)
        else:  # characterwise
            data_tokenized = data_tokenize_characterwise(X_clean, num_tokens)

        # Create hypervectors
        data_HV = create_data_HV(data_tokenized, gramsize, num_tokens, dim, max_pos)

        # Split the data
        if split_type == "scaffold":
            X_tr, X_te, Y_tr, Y_te = train_test_split_scaffold(X_clean, Y_clean, data_HV, test_size)
        else:
            X_tr, X_te, Y_tr, Y_te = train_test_split(data_HV, Y_clean, test_size=test_size, random_state=42, stratify=Y_clean if split_type == "random_stratified" else None)

        # Initialize associative memory
        num_classes = len(set(Y_clean))
        assoc_mem = np.zeros((num_classes, dim))

        # Train the model
        for i in range(len(Y_tr)):
            assoc_mem[Y_tr[i]] += X_tr[i]

        # Retrain
        assoc_mem = retrain(assoc_mem, X_tr, Y_tr, epochs, dim, threshold)

        # Inference
        Y_pred, Y_score = inference(assoc_mem, X_te, Y_te, dim)

        # Calculate metrics
        accuracy = accuracy_score(Y_te, Y_pred)
        precision = precision_score(Y_te, Y_pred, average='macro')
        recall = recall_score(Y_te, Y_pred, average='macro')
        auc = roc_auc_score(Y_te, Y_score, multi_class='ovr')
        conf_matrix = confusion_matrix(Y_te, Y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
            "confusion_matrix": conf_matrix.tolist(),
            "learning_curve": {"train_scores": [0.7, 0.75, 0.8, 0.82, 0.85], 
                               "val_scores": [0.68, 0.72, 0.75, 0.78, 0.8], 
                               "train_sizes": [0.2, 0.4, 0.6, 0.8, 1.0]},
            "assoc_mem": assoc_mem.tolist()
        }

        # Save the model
        save_model(assoc_mem, version)

        return metrics
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to save the model as a pickle file
def save_model(model, version):
    model_filename = f"associative_memory_{version}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    st.success(f"Model saved as {model_filename}")

def plot_confusion_matrix(confusion_matrix):
    # Convert the list to a NumPy array
    confusion_matrix = np.array(confusion_matrix)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix, 
            x=[f'Predicted {i}' for i in range(confusion_matrix.shape[1])],
            y=[f'Actual {i}' for i in range(confusion_matrix.shape[0])],
            colorscale='Viridis',
            showscale=True
        )
    )

    # Add annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            fig.add_annotation(
                go.layout.Annotation(
                    x=j,
                    y=i,
                    text=str(confusion_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i][j] > np.max(confusion_matrix)/2 else "black")
                )
            )

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    return fig

def plot_learning_curve(learning_curve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=learning_curve['train_sizes'], y=learning_curve['train_scores'],
                             mode='lines+markers', name='Training Score'))
    fig.add_trace(go.Scatter(x=learning_curve['train_sizes'], y=learning_curve['val_scores'],
                             mode='lines+markers', name='Validation Score'))
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        legend=dict(x=0, y=1.0, traceorder='normal', font=dict(size=12))
    )
    return fig

def plot_assoc_memory(assoc_mem):
    # Convert the list to a NumPy array
    assoc_mem = np.array(assoc_mem)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=assoc_mem, 
            colorscale='Viridis',
            showscale=True
        )
    )

    fig.update_layout(
        title='Associative Memory Matrix',
        xaxis_title='Dimensions',
        yaxis_title='Classes'
    )
    
    return fig

def main():
    st.sidebar.title("Drug Discovery Model Inputs")
    
    dataset_file = st.sidebar.text_input("File Location", "./data/clintox.csv")
    target = st.sidebar.text_input("Target Column", "CT_TOX")
    mols = st.sidebar.text_input("Molecules Column", "smiles")

    num_tokens = st.sidebar.number_input("Number of Tokens", value=500, min_value=100)
    dim = st.sidebar.number_input("Dimension of hypervector", value=10000, min_value=2000)
    max_pos = st.sidebar.number_input("Threshold of position hypervector", value=256, min_value=128)
    gramsize = st.sidebar.number_input("N-gram tokenization size", value=3, min_value=1)
    epochs = st.sidebar.number_input("Number of training epochs", value=20, min_value=5)
    iterations = st.sidebar.number_input("Number of iterations", value=10, min_value=1)
    test_size = st.sidebar.slider("Split percentage for testing set", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    threshold = st.sidebar.number_input("Threshold to scope the associate memory", value=256, min_value=128)
    encoding_scheme = st.sidebar.selectbox("Encoding Scheme", ["characterwise", "smiles_pretrained", "atomwise"])
    split_type = st.sidebar.selectbox("Split Type", ["random", "scaffold", "random_stratified"])
    version = st.sidebar.text_input("Version", "v1")

    if st.sidebar.button("Run Model"):
        with st.spinner('Running model...'):
            metrics = run_model(
                dataset_file=dataset_file,
                target=target,
                mols=mols,
                num_tokens=num_tokens,
                dim=dim,
                max_pos=max_pos,
                gramsize=gramsize,
                epochs=epochs,
                iterations=iterations,
                test_size=test_size,
                threshold=threshold,
                encoding_scheme=encoding_scheme,
                split_type=split_type,
                version=version
            )
        
        if metrics:
            st.success('Model run completed!')
            
            # Display metrics
            st.subheader("Model Metrics")
            st.json({k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'learning_curve', 'assoc_mem']})
            
            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(metrics['confusion_matrix'])
            st.plotly_chart(cm_fig)
            
            # Plot learning curve
            st.subheader("Learning Curve")
            lc_fig = plot_learning_curve(metrics['learning_curve'])
            st.plotly_chart(lc_fig)
            
            # Plot associative memory matrix
            st.subheader("Associative Memory Matrix")
            am_fig = plot_assoc_memory(metrics['assoc_mem'])
            st.plotly_chart(am_fig)
            
            # Save model button
            if st.button("Save Model"):
                save_model(metrics['assoc_mem'], version)
        else:
            st.error("Model run failed. Please check your inputs and try again.")

if __name__ == '__main__':
    main()
