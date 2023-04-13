import numpy as np
import matplotlib.pyplot as plt

def compare_sdf_tsdf(sdf, tsdf):
    sdf_mean = np.mean(sdf)
    tsdf_mean = np.mean(tsdf)
    
    sdf_std = np.std(sdf)
    tsdf_std = np.std(tsdf)
    
    sdf_min = np.min(sdf)
    tsdf_min = np.min(tsdf)
    
    sdf_max = np.max(sdf)
    tsdf_max = np.max(tsdf)

    print("SDF Mean: {:.6f}, TSDF Mean: {:.6f}".format(sdf_mean, tsdf_mean))
    print("SDF Std: {:.6f}, TSDF Std: {:.6f}".format(sdf_std, tsdf_std))
    print("SDF Min: {:.6f}, TSDF Min: {:.6f}".format(sdf_min, tsdf_min))
    print("SDF Max: {:.6f}, TSDF Max: {:.6f}".format(sdf_max, tsdf_max))

    # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sdf.flatten(), bins=100, range=(-1, 1), alpha=0.7, label='SDF')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('SDF Histogram')

    plt.subplot(1, 2, 2)
    plt.hist(tsdf.flatten(), bins=100, range=(-1, 1), alpha=0.7, label='TSDF', color='orange')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('TSDF Histogram')

    plt.show()
