from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = [
    'DiscretePointsDataset'
]

class DiscretePointsDataset(Dataset):
    def __init__(self, num_samples = 1000, size = 64, min_points=5, max_points = 15, min_distance = 5, border = 4):
        """
        Generates synthetic data with sparse bright points.
        
        Creates a set of discrete target image points with variable counts.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            size (int): Spatial size (pixel count) of the (square) image.
            min_points (int): Minimum number of bright points per image.
            max_points (int): Maximum number of bright points per image.
            min_distance: Minimum allowed distance between points in the image.
            border: No bright points allowed within the border

        Output:
            T_list: list of len = num_samples. Items in list (1, size, size) torch tensor, 
                    with pixel value 1 if a bright spot, 0 otherwise.
            point_counts: list of actual number of points per sample (for debugging)
        """

        self.num_samples = num_samples
        self.size = size
        self.min_points = min_points
        self.max_points = max_points
        self.T_list = [] 
        self.point_counts = []  # Track actual point counts

        for _ in range(num_samples): # create a sample
            # Create an amplitude image with all zeros
            T_np = np.zeros((size, size), dtype=np.float32)
            num_points = np.random.randint(min_points, max_points + 1)
            bright_spots = [] # contains the coordinate points for each sample

            while len(bright_spots) < num_points: 
                # generate coords for new point
                x = np.random.randint(0 + border, size - border)
                y = np.random.randint(0 + border, size - border)
                # check if new point is too close to other points
                dist_to_new_spot = [np.linalg.norm(np.array([x, y]) - coord) for coord in bright_spots]
                dist_to_new_spot = np.array(dist_to_new_spot)
    
                if len(bright_spots) == 0 or all(dist_to_new_spot > min_distance): 
                    #if no sites yet, or all distance to spot are sufficiently large
                    bright_spots.append(np.array([x,y]))
                    T_np[x, y] = 1.0
                else: 
                    #if new point is too close
                    continue
        
            # Convert to torch tensors
            T_t = torch.tensor(T_np, dtype=torch.float32)
            self.T_list.append(T_t.unsqueeze(0)) # Add a channel dimension: shape becomes (1, size, size)
            self.point_counts.append(len(bright_spots))  # Store actual count
    
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.T_list[idx]
    
    def get_point_count(self, idx):
        """Get the actual number of points for a given sample"""
        return self.point_counts[idx]



