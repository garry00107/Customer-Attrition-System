import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import tkinter as tk
from tkinter import ttk
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf, precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

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

# PCA
sc = StandardScaler()
cr_scaled = sc.fit_transform(cr_log)
pc = PCA(n_components=6)
reduced_cr = pc.fit_transform(cr_scaled)

# KMeans Clustering
km_4 = KMeans(n_clusters=4, random_state=123)
km_4.fit(reduced_cr)

# Calculate Calinski-Harabasz Index
ch_score = calinski_harabasz_score(reduced_cr, km_4.labels_)
print("Calinski-Harabasz Index:", ch_score)

# Silhouette Coefficient (already in the code)
silhouette_avg = silhouette_score(reduced_cr, km_4.labels_)
print("Silhouette Coefficient:", silhouette_avg)

# GUI for Clustering (remaing code remains the same)


# GUI for Clustering
def get_cluster_label(new_data):
    new_df = pd.DataFrame(new_data, index=[0])
    cluster_label = kmeans.predict(new_df)
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
    cluster_label = get_cluster_label(new_data)
    cluster_label_text.set(f'Cluster: {cluster_label}')

# Sample Data for GUI
sample_data = {
    'PURCHASES': [1000, 0, 100, 500],
    'CASH_ADVANCE': [0, 1000, 200, 0],
    'CREDIT_LIMIT': [5000, 2000, 3000, 4000],
    'TENURE': [12, 24, 36, 48],
    'PAYMENTS': [100, 1000, 500, 200],
    'PURCHASES_FREQUENCY': [0.5, 0, 1, 0.2],
    'PURCHASES_INSTALLMENTS_FREQUENCY': [0.5, 0, 1, 0.2],
    'CASH_ADVANCE_FREQUENCY': [0, 0.5, 0.2, 0],
}
df_sample = pd.DataFrame(sample_data)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_sample)

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
cash_advance_frequency_entry = ttk.Entry(mainframe)
cash_advance_frequency_entry.grid(column=2, row=8, sticky=tk.W)

# Cluster Label Display
cluster_label_text = tk.StringVar()
cluster_label_text.set("Cluster: ")
ttk.Label(mainframe, textvariable=cluster_label_text).grid(column=2, row=9, sticky=tk.W)

# Cluster Button
ttk.Button(mainframe, text="Cluster", command=cluster).grid(column=2, row=10, sticky=tk.W)

# Padding
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()