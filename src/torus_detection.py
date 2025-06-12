import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def torus_detection_explained(data, peak_threshold_factor=0.1, radial_weight=0.7, vertical_weight=0.3):
    """Improved torus detection with parameterized settings."""

    if data.shape[1] < 3:
        raise ValueError("Data must have at least 3 dimensions for torus detection.")

    print("🍩 Analyzing toroidal structure")

    center = np.mean(data, axis=0)
    centered_data = data - center

    print("\n🔄 Step 1: Radial distance analysis")
    radial_distances = np.linalg.norm(centered_data[:, :2], axis=1)
    print(f"Radial distance range: {np.min(radial_distances):.2f} - {np.max(radial_distances):.2f}")

    print("\n📊 Step 2: Peak detection (histogram analysis)")
    hist, bins = np.histogram(radial_distances, bins=30, density=True)
    peaks_indices, _ = find_peaks(hist, height=peak_threshold_factor * np.max(hist))
    peaks = [(bins[i] + bins[i+1])/2 for i in peaks_indices]

    print(f"Number of peaks found: {len(peaks)}")
    print(f"Peak positions: {peaks}")

    print("\n🌊 Step 3: Z-axis uniformity")
    z_values = centered_data[:, 2]
    z_normalized = (z_values - np.mean(z_values)) / np.std(z_values)
    z_std_normalized = np.std(z_normalized)
    z_uniformity = np.exp(-z_std_normalized)
    
    print(f"Normalized Z standard deviation: {z_std_normalized:.2f}")
    print(f"Z uniformity score: {z_uniformity:.3f}")

    bimodality_score = min(len(peaks)/2.0, 1.0)
    torus_score = radial_weight * bimodality_score + vertical_weight * z_uniformity

    print("\n🎯 Final Results:")
    print(f"Bimodality score: {bimodality_score:.3f}")
    print(f"Final torus score: {torus_score:.3f}")

    # Visualization code (unchanged for brevity)

    return {
        'score': torus_score,
        'n_peaks': len(peaks),
        'peaks': peaks,
        'z_uniformity': z_uniformity
    }
