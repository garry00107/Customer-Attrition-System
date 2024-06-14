import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import ttk
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load data
credit = pd.read_csv("/Users/garrry/Downloads/CC_GENERAL.csv")


# Data Preprocessing
credit['CREDIT_LIMIT'].fillna(credit['CREDIT_LIMIT'].median(), inplace=True)
credit['MINIMUM_PAYMENTS'].fillna(credit['MINIMUM_PAYMENTS'].median(), inplace=True)

# Feature Engineering
credit['Monthly_avg_purchase'] = credit['PURCHASES'] / credit['TENURE']
credit['Monthly_cash_advance'] = credit['CASH_ADVANCE'] / credit['TENURE']

def purchase_type(row):
    if row['ONEOFF_PURCHASES'] == 0 and row['INSTALLMENTS_PURCHASES'] == 0:
        return 'none'
    if row['ONEOFF_PURCHASES'] > 0 and row['INSTALLMENTS_PURCHASES'] > 0:
        return 'both_oneoff_installment'
    if row['ONEOFF_PURCHASES'] > 0 and row['INSTALLMENTS_PURCHASES'] == 0:
        return 'one_off'
    if row['ONEOFF_PURCHASES'] == 0 and row['INSTALLMENTS_PURCHASES'] > 0:
        return 'installment'

credit['purchase_type'] = credit.apply(purchase_type, axis=1)
credit['limit_usage'] = credit['BALANCE'] / credit['CREDIT_LIMIT']
credit['payment_minpay'] = credit['PAYMENTS'] / credit['MINIMUM_PAYMENTS']

# Log transformation
cr_log = credit.drop(['CUST_ID', 'purchase_type'], axis=1).applymap(lambda x: np.log(x + 1))

# Standardizing data
sc = StandardScaler()
cr_scaled = sc.fit_transform(cr_log)

# Apply PCA
pc = PCA(n_components=6)
reduced_cr_pca = pc.fit_transform(cr_scaled)

# KMeans Clustering with optimal number of clusters (update this based on elbow/silhouette method results)
optimal_clusters = 2  # Change this based on the elbow/silhouette method results
kmeans_pca = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_pca.fit(reduced_cr_pca)

# GUI for Clustering
def get_cluster_label(new_data, model):
    new_df = pd.DataFrame(new_data, index=[0])
    new_data_scaled = sc.transform(new_df)
    new_data_reduced = pc.transform(new_data_scaled)
    cluster_label = model.predict(new_data_reduced)
    return cluster_label[0]

def cluster():
    new_data = {
        'PURCHASES': float(purchases_entry.get()),
        'CASH_ADVANCE': float(cash_advance_entry.get()),
        'CREDIT_LIMIT': float(credit_limit_entry.get()),
        'TENURE': int(tenure_entry.get()),
        'PAYMENTS': float(payments_entry.get()),
        'PURCHASES_FREQUENCY': float(purchases_frequency_entry.get()),
        'PURCHASES_INSTALLMENTS_FREQUENCY': float(purchases_installments_frequency_entry.get()),
        'CASH_ADVANCE_FREQUENCY': float(cash_advance_frequency_entry.get()),
    }
    cluster_label = get_cluster_label(new_data, kmeans_pca)
    cluster_label_text.set(f'Cluster: {cluster_label}')

# GUI
root = tk.Tk()
root.title("Credit Card Segmentation")

mainframe = ttk.Frame(root, padding="20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# GUI Input Fields
ttk.Label(mainframe, text="PURCHASES:").grid(column=1, row=1, sticky=tk.W)
purchases_entry = ttk.Entry(mainframe)
purchases_entry.grid(column=2, row=1, sticky=tk.W)

ttk.Label(mainframe, text="CASH ADVANCE:").grid(column=1, row=2, sticky=tk.W)
cash_advance_entry = ttk.Entry(mainframe)
cash_advance_entry.grid(column=2, row=2, sticky=tk.W)

ttk.Label(mainframe, text="CREDIT LIMIT:").grid(column=1, row=3, sticky=tk.W)
credit_limit_entry = ttk.Entry(mainframe)
credit_limit_entry.grid(column=2, row=3, sticky=tk.W)

ttk.Label(mainframe, text="TENURE:").grid(column=1, row=4, sticky=tk.W)
tenure_entry = ttk.Entry(mainframe)
tenure_entry.grid(column=2, row=4, sticky=tk.W)

ttk.Label(mainframe, text="PAYMENTS:").grid(column=1, row=5, sticky=tk.W)
payments_entry = ttk.Entry(mainframe)
payments_entry.grid(column=2, row=5, sticky=tk.W)

ttk.Label(mainframe, text="PURCHASES FREQUENCY:").grid(column=1, row=6, sticky=tk.W)
purchases_frequency_entry = ttk.Entry(mainframe)
purchases_frequency_entry.grid(column=2, row=6, sticky=tk.W)

ttk.Label(mainframe, text="PURCHASES INSTALLMENTS FREQUENCY:").grid(column=1, row=7, sticky=tk.W)
purchases_installments_frequency_entry = ttk.Entry(mainframe)
purchases_installments_frequency_entry.grid(column=2, row=7, sticky=tk.W)

ttk.Label(mainframe, text="CASH ADVANCE FREQUENCY:").grid(column=1, row=8, sticky=tk.W)

