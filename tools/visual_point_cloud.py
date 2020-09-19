import open3d as o3d
import numpy as np

pc_load_pathname = '/home/caizhongang/playground/kitti/test_crop_car/007401-000.bin'
pc = np.fromfile(pc_load_pathname, dtype=np.float32).reshape(-1, 3)

pcd = o3d.geometry.PointCloud() 
import pdb; pdb.set_trace()
pcd.points = o3d.utility.Vector3dVector(pc) 
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0]) 
visual = [pcd, axis] 
o3d.visualization.draw_geometries(visual)
