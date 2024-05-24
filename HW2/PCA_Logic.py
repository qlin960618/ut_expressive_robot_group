import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#----------Read and Format Data----------'
# Pfad zur Excel-Datei
excel_file = 'C:\\Users\\Valdr\\Desktop\\Test.xlsx'

# Reading the Excel for now
df = pd.read_excel(excel_file, usecols='C:DU', dtype=str, skiprows=5, nrows=152)

# Formating
for col in df.columns:
    df[col] = df[col].str.replace('[,.]', '', regex=True).astype(int)
    
# Data to Array
X = df.to_numpy()    

#----------Add Dot----------#

# Funktion zum Hinzufügen des Kommas nach der vordersten Stelle
def add_dot(num):
    num_str = str(num)
    if len(num_str) > 2:
        if num_str[0] != '-':
            return num_str[:2] + '.' + num_str[2:]
        if num_str[0] == '-':
            return num_str[:3] + '.' + num_str[3:]
    return num_str

# Anwenden der Funktion auf jedes Element in X_Array
format_X = np.array([[add_dot(num) for num in row] for row in X])

#----------Split the Characteristics----------#

# Anzahl der Merkmale (jedes Merkmal besteht aus 3 Spalten: X, Y, Z)
num_merkmale = df.shape[1] // 3

# Erstellen von Variablen x1, x2, ..., x41 für jedes Merkmal
for i in range(num_merkmale):
    start_index = i * 3
    end_index = start_index + 3
    globals()[f'format_X{i+1}'] = format_X[:, start_index:end_index]

# Ausgabe des ersten Merkmals
print(format_X1)


#----------PCA----------#
# Standardisierung der Daten (ohne Formatierung)
X_std = StandardScaler().fit_transform(format_X1)

# Durchführung von PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_std)

# Plot der Hauptkomponenten mit zeitlicher Farbskala
plt.figure()
scatter = plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=np.arange(len(principalComponents)), cmap='viridis', s=50)
plt.colorbar(scatter, label='Index der Beobachtungen')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.title('PCA der Daten')
plt.grid(True)
plt.show()


# Transformationsmatrix abrufen
transformationsmatrix = pca.components_

# Ausgabe der Transformationsmatrix
print("Transformationsmatrix:")
print(transformationsmatrix)