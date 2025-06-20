# Geometric Shapes Projection Library

A comprehensive Python library for projecting points onto various geometric shapes in 2D, 3D, and high-dimensional spaces. This library provides efficient algorithms for calculating projections, distances, and containment tests across multiple geometric primitives.

## 🌟 Features

- **Multi-dimensional Support**: Handle 2D to n-dimensional geometric shapes
- **Comprehensive Shape Library**: Spheres, ellipsoids, cylinders, tori, disks, and more
- **Advanced Projections**: Multiple projection strategies for high-dimensional data
- **Performance Optimized**: Efficient algorithms with comprehensive testing
- **Rich Visualizations**: Built-in plotting and analysis tools
- **Educational Framework**: Perfect for learning computational geometry

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/geometric-shapes-library.git
cd geometric-shapes-library

# Install dependencies
pip install numpy matplotlib scipy scikit-learn
```

## 🚀 Quick Start

```python
import numpy as np
from geometric_shapes import Sphere, Ellipsoid, Cylinder
from torus_disk import Torus, DiskWithHole, ThickDisk
from multi_sphere import MultiSphere
from hyper_torus import HyperTorus
from high_dimensional_torus import HighDimensionalTorus

# Create a sphere and project a point
sphere = Sphere(center=[0, 0, 0], radius=2)
point = [3, 3, 3]
projection = sphere.project_to_surface(point)
distance = sphere.distance_to_surface(point)

print(f"Original point: {point}")
print(f"Projection: {projection}")
print(f"Distance: {distance:.3f}")
```

## 📚 Available Shapes

### Basic 3D Shapes

#### Sphere
```python
from geometric_shapes import Sphere

sphere = Sphere(center=[0, 0, 0], radius=2)
projection = sphere.project_to_surface([3, 3, 3])
inside = sphere.is_inside([1, 1, 1])
```

#### Ellipsoid
```python
from geometric_shapes import Ellipsoid

ellipsoid = Ellipsoid(center=[0, 0, 0], semi_axes=[3, 2, 1])
projection = ellipsoid.project_to_surface([4, 3, 2])
```

#### Cylinder
```python
from geometric_shapes import Cylinder

cylinder = Cylinder(
    center=[0, 0, 0], 
    axis_direction=[0, 0, 1], 
    radius=1.5, 
    height=4
)
projection = cylinder.project_to_surface([3, 2, 2])
```

### Advanced Shapes

#### Torus
```python
from torus_disk import Torus

torus = Torus(
    center=[0, 0, 0],
    major_radius=3,
    minor_radius=1,
    axis=[0, 0, 1]
)
projection = torus.project_to_surface([4, 2, 1])
```

#### Disk with Hole (CD Model)
```python
from torus_disk import DiskWithHole

disk = DiskWithHole(
    center=[0, 0, 0],
    outer_radius=6,
    inner_radius=0.75,
    normal=[0, 0, 1]
)
projection = disk.project_to_surface([4, 3, 0.5])
```

#### Thick Disk (Realistic CD)
```python
from torus_disk import ThickDisk

cd = ThickDisk(
    center=[0, 0, 0],
    outer_radius=6.0,     # 120mm diameter
    inner_radius=0.75,    # 15mm hole
    thickness=0.12        # 1.2mm thick
)
projection, surface_type, distance = cd.project_to_surface([4, 3, 0.5])
print(f"Projected to: {surface_type}")
```

### Multi-Shape Systems

#### Multi-Sphere System
```python
from multi_sphere import MultiSphere

# Create sphere system
multi_sphere = MultiSphere()
multi_sphere.create_predefined_configuration('cube_corners', radius=0.5, scale=3.0)

# Different projection methods
closest_proj, idx = multi_sphere.project_to_surface([0, 0, 1], 'closest')
weighted_proj, weights = multi_sphere.project_to_surface([0, 0, 1], 'weighted')
centroid_proj = multi_sphere.project_to_surface([0, 0, 1], 'centroid')
```

### High-Dimensional Shapes

#### HyperTorus (n-dimensional)
```python
from hyper_torus import HyperTorus

# 5D torus
hyper_torus = HyperTorus(
    radii=[3, 2, 1.5, 1.2, 1.0],
    center=[0, 0, 0, 0, 0]
)

# Project 5D point
point_5d = [4, 3, 2, 1, 2.5]
projection = hyper_torus.project_to_surface(point_5d)
```

#### High-Dimensional Torus with Multiple Methods
```python
from high_dimensional_torus import HighDimensionalTorus

# 6D torus with advanced projection methods
hd_torus = HighDimensionalTorus(
    center=[0, 0, 0, 0, 0, 0],
    major_radius=3,
    minor_radius=1,
    n_dims=6
)

# Compare different projection strategies
point = [4, 3, 2, 1, 5, 2]
methods = ['first_three', 'best_fit', 'random_sample', 'pca']

for method in methods:
    proj = hd_torus.project_to_surface(point, method)
    dist = hd_torus.distance_to_surface(point, method)
    print(f"{method}: distance = {dist:.3f}")
```

## 🎨 Visualization

All shape classes include built-in visualization capabilities:

```python
import matplotlib.pyplot as plt

# Basic shape visualization
fig = plt.figure(figsize=(15, 5))

# Sphere
ax1 = fig.add_subplot(131, projection='3d')
sphere = Sphere(center=[0, 0, 0], radius=2)
surface_points = sphere.generate_surface_points(400)
ax1.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
           alpha=0.1, color='blue', s=1)

# Add projections
test_point = [3, 3, 3]
projection = sphere.project_to_surface(test_point)
ax1.scatter(*test_point, color='red', s=100, label='Original')
ax1.scatter(*projection, color='green', s=100, label='Projection')
ax1.plot([test_point[0], projection[0]], 
         [test_point[1], projection[1]], 
         [test_point[2], projection[2]], 'r--')

plt.show()
```

## 🧪 Comprehensive Testing

Run the complete test suite using the provided Jupyter notebook:

```bash
jupyter notebook comprehensive_testing_notebook.ipynb
```

Or run individual module tests:

```python
# Test individual modules
from hyper_torus import run_comprehensive_tests
results = run_comprehensive_tests()

# Test with the main testing framework
from geometric_shapes import test_geometric_shapes
test_geometric_shapes()
```

## 📊 Performance Benchmarking

The library includes built-in performance analysis:

```python
import time
import numpy as np

# Benchmark projection performance
def benchmark_shape(shape, points, n_iterations=100):
    times = []
    for _ in range(n_iterations):
        start_time = time.time()
        for point in points:
            shape.project_to_surface(point)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times)
    }

# Test performance
shapes = {
    'Sphere': Sphere(center=[0, 0, 0], radius=2),
    'Torus': Torus(center=[0, 0, 0], major_radius=2.5, minor_radius=0.8),
    'HyperTorus': HyperTorus([2, 1.5, 1, 0.8], [0, 0, 0, 0])
}

test_points = np.random.randn(50, 3) * 2

for name, shape in shapes.items():
    result = benchmark_shape(shape, test_points)
    print(f"{name}: {result['mean_time']*1000:.2f} ms")
```

## 🔧 Advanced Usage

### Custom Shape Configuration

```python
# Create custom multi-sphere configurations
multi_sphere = MultiSphere()

# Add individual spheres
multi_sphere.add_sphere(center=[2, 0, 0], radius=1)
multi_sphere.add_sphere(center=[-2, 0, 0], radius=1)
multi_sphere.add_sphere(center=[0, 2, 0], radius=1)

# Test projection methods
point = [0, 0, 2]
results = multi_sphere.compare_projection_methods(point)
```

### High-Dimensional Analysis

```python
# Generate structured test data
hd_torus = HighDimensionalTorus(n_dims=8, major_radius=3, minor_radius=1)
test_points = hd_torus.generate_test_points(n_points=100, noise_level=1.5)

# Analyze projection quality across methods
methods = ['first_three', 'best_fit', 'random_sample']
for method in methods:
    distances = []
    for point in test_points:
        dist = hd_torus.distance_to_surface(point, method)
        distances.append(dist)
    
    print(f"{method}: mean={np.mean(distances):.3f}, std={np.std(distances):.3f}")
```

### Parameter Sensitivity Analysis

```python
# Analyze how parameters affect projections
test_point = [2, 1.5, 1]

# Sphere radius sensitivity
radii = [0.5, 1, 1.5, 2, 2.5, 3]
for radius in radii:
    sphere = Sphere(center=[0, 0, 0], radius=radius)
    distance = sphere.distance_to_surface(test_point)
    print(f"Radius {radius}: distance = {distance:.3f}")
```

## 📁 File Structure

```
geometric-shapes-library/
├── README.md
├── requirements.txt
├── comprehensive_testing_notebook.ipynb
├── geometric_shapes.py          # Basic 3D shapes
├── hyper_torus.py              # n-dimensional torus
├── multi_sphere.py             # Multi-sphere systems
├── torus_disk.py               # Torus and disk shapes
├── high_dimensional_torus.py   # Advanced high-D projections
├── examples/
│   ├── basic_usage.py
│   ├── performance_analysis.py
│   └── visualization_examples.py
└── tests/
    ├── test_geometric_shapes.py
    ├── test_hyper_torus.py
    └── test_performance.py
```

## 🎯 Common Use Cases

### 1. **Machine Learning Data Projection**
```python
# Project high-dimensional data onto geometric manifolds
from high_dimensional_torus import HighDimensionalTorus

# Create manifold for data projection
manifold = HighDimensionalTorus(n_dims=10, major_radius=5, minor_radius=2)

# Project dataset
projected_data = []
for data_point in dataset:
    proj = manifold.project_to_surface(data_point, method='pca', data_points=dataset)
    projected_data.append(proj)
```

### 2. **3D Graphics and Game Development**
```python
# Collision detection and surface projections
sphere = Sphere(center=player_position, radius=collision_radius)
if sphere.is_inside(object_position):
    # Handle collision
    safe_position = sphere.project_to_surface(object_position)
```

### 3. **Scientific Computing**
```python
# Model physical objects with realistic dimensions
cd_model = ThickDisk(
    center=[0, 0, 0],
    outer_radius=6.0,    # Standard CD: 120mm diameter
    inner_radius=0.75,   # 15mm center hole
    thickness=0.12       # 1.2mm thickness
)

# Calculate physical properties
volume = cd_model.get_volume()      # cm³
surface_area = cd_model.get_surface_area()  # cm²
```

### 4. **Data Analysis and Clustering**
```python
# Multi-sphere clustering analysis
cluster_system = MultiSphere()
cluster_system.create_predefined_configuration('random', radius=1.0, scale=5.0)

# Assign points to nearest cluster
for data_point in dataset:
    projection, cluster_id = cluster_system.project_to_surface(data_point, 'closest')
    cluster_assignments[data_point] = cluster_id
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-shape`
3. **Add tests** for your new functionality
4. **Update documentation** as needed
5. **Submit a pull request**

### Adding New Shapes

To add a new geometric shape:

1. Create a new class with the standard interface:
   ```python
   class NewShape:
       def __init__(self, ...):
           pass
       
       def project_to_surface(self, point):
           pass
       
       def distance_to_surface(self, point):
           pass
       
       def is_inside(self, point):
           pass
       
       def generate_surface_points(self, n_points=100):
           pass
   ```

2. Add comprehensive tests
3. Update the notebook with examples
4. Add visualization capabilities

## 📋 Requirements

- **Python** >= 3.7
- **NumPy** >= 1.18.0
- **Matplotlib** >= 3.2.0
- **SciPy** >= 1.4.0 (optional, for advanced features)
- **scikit-learn** >= 0.22.0 (optional, for PCA methods)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Written in collaboration with Claude Sonnet
- Inspired by computational geometry research
- Built for educational and practical applications
- Designed with performance and usability in mind