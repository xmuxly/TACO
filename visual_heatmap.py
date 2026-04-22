import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

def visualize_feature_heatmap(feature_path, save_dir):
    """
    Visualize the mean feature map across all channels and all batches as a single heatmap
    Args:
        feature_path: Path to the .npy file containing features
        save_dir: Directory to save the visualization
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load features
    features = np.load(feature_path)  # Shape: [batch, channels, height, width]
    print(f"Processing file: {os.path.basename(feature_path)}")
    print(f"Features shape: {features.shape}")
    
    # Calculate mean across all channels and all batches
    mean_features = np.mean(features, axis=(0, 1))  # Shape: [height, width]
    
    # Normalize to [0, 1] range
    mean_features = (mean_features - mean_features.min()) / (mean_features.max() - mean_features.min())
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    im = plt.imshow(mean_features, cmap='jet')
    plt.colorbar(im, label='Feature Intensity')
    plt.title(f'Mean Feature Map - {os.path.basename(feature_path)}')
    
    # Save the visualization
    save_name = os.path.basename(feature_path).replace('.npy', '_heatmap.png')
    plt.savefig(os.path.join(save_dir, save_name), 
               dpi=300, bbox_inches='tight')
    plt.close()

def visualize_channel_heatmaps(feature_path, output_dir="channel_heatmaps"):
    """
    Loads features and generates a heatmap for each channel, averaged across batches.

    Args:
        feature_path (str): Path to the .npy file containing features (shape [batch, channels, height, width]).
        output_dir (str): Directory to save the generated heatmaps.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load features
    features = np.load(feature_path)  # Shape: [batch, channels, height, width]
    print(f"Processing file: {os.path.basename(feature_path)}")
    print(f"Features shape: {features.shape}")

    num_channels = features.shape[1]
    height = features.shape[2]
    width = features.shape[3]

    print(f"Generating heatmaps for {num_channels} channels...")

    for channel_idx in range(num_channels):
        # Extract features for the current channel across all batches
        channel_features = features[:, channel_idx, :, :] # Shape: [batch, height, width]

        # Calculate mean across the batch dimension for the current channel
        mean_channel_features = np.mean(channel_features, axis=0) # Shape: [height, width]

        # Generate heatmap
        plt.figure(figsize=(width/10, height/10)) # Adjust figure size based on heatmap dimensions
        sns.heatmap(mean_channel_features, cmap='viridis', cbar=True)
        plt.title(f'Channel {channel_idx} Heatmap (Averaged Across Batches)')
        plt.xlabel('Width')
        plt.ylabel('Height')

        # Save the heatmap
        heatmap_filename = f'channel_{channel_idx}_heatmap.png'
        heatmap_filepath = os.path.join(output_dir, heatmap_filename)
        plt.savefig(heatmap_filepath, bbox_inches='tight')
        plt.close() # Close the plot to free up memory

        if (channel_idx + 1) % 10 == 0 or (channel_idx + 1) == num_channels:
            print(f"Generated heatmap for channel {channel_idx + 1}/{num_channels}")

    print(f"Finished generating heatmaps. Saved to '{output_dir}'")

def main():
    # 设置输入和输出目录
    input_dir = '/home/ssd1/code/anchor_SG/output_heatmap_visual'
    save_dir = '/home/ssd1/code/anchor_SG/output_heatmap_visual/heatmap_visualizations'
    
    # 获取所有特征图文件
    feature_files = sorted(glob.glob(os.path.join(input_dir, '*_aligned_loc_feature.npy')))
    
    print(f"Found {len(feature_files)} feature files to process")
    
    # 处理每个文件
    for feature_file in feature_files:
        try:
            visualize_feature_heatmap(feature_file, save_dir)
            print(f"Successfully processed {os.path.basename(feature_file)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(feature_file)}: {str(e)}")
    
    print("All files processed!")

if __name__ == '__main__':
    main() 