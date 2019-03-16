# gentlemandata
  <img align="left" width="240" height="200" src="logo/GentlemanData.png">
  GENtleMANDATA: A toolkit to GENerate synthetic MANifold geometry DATAsets 

---



## Installation
Python 3 is required.

```
git clone https://github.com/georgepar/gentlemandata
cd gentlemandata
pip install -r requirements.txt
pip install -e .
```


## Usage  

Supported shapes to generate

```
data_type in {
    'sphere', 
    'cut-sphere', 
    'ball',
    'random', 
    'spiral', 
    'spiral-hole',
    'swissroll',
    'swisshole',
    'toroid-helix',
    's-curve',
    'punctured-sphere',
    'gaussian', 
    'clusters-3d',
    'twin-peaks',
    'corner-plane'
}
```

Working example using the `DataBuilder`:

```
from gentlemandata import shapes
data_type = 'spiral' # shape to generate
dim = 3  # shape dimension. Leave 3
distance = 'geodesic' # d_goal is calculated using geodesic or euclidean distance. Useful for MDS
npoints = 1000 # Number of points to generate
n_neighbors = 12 # Neighbors are used for geodesic distance calculation
noise_std = 0 # Amount of noise in the data

xs, d_goal, color = (shapes.DataBuilder()
                 .with_dim(dim)
                 .with_distance(distance)
                 .with_noise(noise_std)
                 .with_npoints(npoints)
                 .with_neighbors(n_neighbors)
                 .with_type(data_type)
                 .build())

# Then you can use data with sklearn. e.g. x_reduced = PCA(n_components=2).fit_transform(xs) to get a 2d representation
# Or x_reduced = MDS(n_components=2, dissimilarity='precomputed').fit_transform(d_goal)


from sklearn.decomposition import PCA

x_r = PCA(n_components=2).fit_transform(xs)

import  matplotlib.pyplot as plt
fig = plt.figure()

ax = plt.axes(projection='3d')
ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original data")
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(x_r[:, 0], x_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title('Projected data')
plt.show()

```
