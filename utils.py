from typing import Union

import numpy as np
import open3d as o3d

#############################
## Pointcloud Helper Class ##
#############################

class PointCloudProcessor:

    def __init__(self, pcd: Union[str, np.ndarray], color: np.ndarray = None):
        if isinstance(pcd, str):
            self.pcd = o3d.io.read_point_cloud(pcd)
        elif isinstance(pcd, np.ndarray):
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pcd)
        else:
            raise TypeError("Unsupported input type for PointCloudProcessor")
        
        if color is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(color)
        
    def process(self):
        """Process the point cloud"""

        # Denoise
        pcd_denoised = self._denoise(self.pcd)
        
        # Downsample
        pcd_downsampled = self._downsample(pcd_denoised)

        # Smooth
        # pcd_smoothed = self._smooth(pcd_downsampled)
        pcd_smoothed = pcd_downsampled

        # Restore original index
        idx = self._restore_idx(pcd_smoothed)

        return pcd_smoothed, idx

    def save_pcd(self, pcd: o3d.geometry.PointCloud, filename: str):
        """Save the point cloud to a file"""

        o3d.io.write_point_cloud(filename, pcd, print_progress=True)

    def _denoise(self, pcd, nb_neighbors=50, std_ratio=0.05):
        """Statistical Outlier Removal"""

        pcd_denoised, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio,
            print_progress=True
        )
        return pcd_denoised

    def _reconstruct_surface(self, pcd: o3d.geometry.PointCloud):

        #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([0.05, 0.05, 0.05]))
        return mesh

    def _smooth(self, pcd: o3d.geometry.PointCloud):
        """
        First reconstruct surface. Second, smooth the surface. Third, resample the surface.
        """

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        mesh = self._reconstruct_surface(pcd)
        mesh_smoothed = mesh.filter_smooth_laplacian(number_of_iterations=10)
        pcd_smoothed = mesh_smoothed.sample_points_poisson_disk(number_of_points=len(pcd.points))
        return pcd_smoothed

    def _downsample(self, pcd, n_samples=2048):
        """Farthest Point Sampling"""

        pcd_downsampled = pcd.farthest_point_down_sample(num_samples=n_samples)
        return pcd_downsampled

    def _restore_idx(self, pcd):
        """Restore the original index of the point cloud"""

        # Build KD-tree on the original point cloud
        tree = o3d.geometry.KDTreeFlann(self.pcd)

        proc_pts = np.asarray(pcd.points)
        nn_idx = np.zeros(len(pcd.points), dtype=np.int64)

        # For each processed point, find nearest original point
        for i, p in enumerate(proc_pts):
            _, idx, _ = tree.search_knn_vector_3d(p, 1)
            nn_idx[i] = idx[0]

        return nn_idx

    
