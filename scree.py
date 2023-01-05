import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
#define URL where dataset is located

#read in data
data = pd.read_csv('coords.csv')
#define columns to use for PCA
df = data.iloc[:, 1:133]
print(df)

#define scaler
scaler = StandardScaler()

#create copy of DataFrame
scaled_df=df.copy()

#created scaled version of DataFrame
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)



#define PCA model to use
pca = PCA(n_components=132)

#fit PCA model to data
pca_fit = pca.fit(scaled_df)



PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
