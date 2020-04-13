import open3d as o3d
import pykitti.utils as pk_utils
import kitti_util as utils
import numpy as np

 
# waymo_kitti
pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_old/lidar/000000000000200.bin'
label_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_old/label_all/000000000000200.txt'
calib_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_old/calib/000000000000200.txt'

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            print('line', line)
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data



def corners_to_lines(qs):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )

    return line_set


def transform_pc(pc, R0, V2C):
    tf = np.matmul(R0, V2C)
    assert pc.shape[1] == 3
    pc = np.transpose(pc)
    ones = np.ones((1, pc.shape[1]))
    pc = np.vstack([pc, ones])
    tf_pc = np.matmul(tf, pc)
    #tf_pc = tf_pc[:3, :] / tf[3, :]
    tf_pc = np.transpose(tf_pc)
    return tf_pc


def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = utils.rotz(obj.ry)
    print('R', R)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    #corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    corners_2d = None
    return corners_2d, np.transpose(corners_3d)


def main():
    pc = pk_utils.load_velo_scan(pc_pathname)[:, :3]
    objs = utils.read_label(label_pathname)
    #calib = pk_utils.read_calib_file(calib_pathname)
    calib = read_calib_file(calib_pathname)    

    #V2C = np.array(calib['Tr_velo_to_cam_0']).reshape(4,4)[:3, :]    
    V2C = np.eye(4)[:3, :]    
    R0 = np.array(calib['R0_rect']).reshape(3,3)
    P0 = np.array(calib['P0']).reshape(3,4)

    pc = transform_pc(pc, R0, V2C) 

    bboxes3d = []
    for o in objs:
        bbox2d, bbox3d = compute_box_3d(o, P0)
        bboxes3d.append(bbox3d)        
    
    # draw
    pcd = o3d.geometry.PointCloud()
    print(pc)
    pcd.points = o3d.utility.Vector3dVector(pc)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0,0,0])

    visual = [pcd, axis]    
    #visual = [pcd]

    for bbox3d in bboxes3d:
        visual.append(corners_to_lines(bbox3d))
    
    o3d.visualization.draw_geometries(visual)


if __name__ == '__main__':
    main()

