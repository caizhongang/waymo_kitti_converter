import open3d as o3d
import pykitti.utils as pk_utils
import kitti_util as utils
import numpy as np
from calibration import Calibration
from objects3d_utils import get_objects_from_label

# kitti
# pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/velodyne/000002.bin'
# label_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti//label_2/000002.txt'
# calib_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/calib/000002.txt'

# use my own tool
pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/lidar/00000-00001.bin'
label_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/label_all/00000-00001.txt'
calib_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/waymo_kitti/calib/00000-00001.txt'


# def read_calib_file(filepath):
#     """Read in a calibration file and parse into a dictionary."""
#     data = {}
#
#     with open(filepath, 'r') as f:
#         for line in f.readlines():
#             print('line', line)
#             key, value = line.split(':', 1)
#             # The only non-float values in these files are dates, which
#             # we don't care about anyway
#             try:
#                 data[key] = np.array([float(x) for x in value.split()])
#             except ValueError:
#                 pass
#
#     return data



def corners_to_lines(qs):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]

    # print('draw bbox')
    # print(qs)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )

    return line_set


def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
    :param z_bottom: whether z is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry),  zeros],
                         [zeros,      zeros,        ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)



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


def main():
    # visualized in lidar frame

    pc = pk_utils.load_velo_scan(pc_pathname)
    print('pc, just loaded', pc.shape)
    pc = pc[:, :3]
    print('pc, after slicing', pc.shape)

    # objs = utils.read_label(label_pathname)
    # calib = pk_utils.read_calib_file(calib_pathname)
    # calib = read_calib_file(calib_pathname)

    # V2C = np.array(calib['Tr_velo_to_cam']).reshape(3,4)
    # R0 = np.array(calib['R0_rect']).reshape(3,3)
    # P0 = np.array(calib['P0']).reshape(3,4)

    # print(V2C, R0, P0)

    # pc = transform_pc(pc, R0, V2C)

    # print(pc.shape)

    obj_list = get_objects_from_label(label_pathname)
    calib = Calibration(calib_pathname)

    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    rots = np.array([obj.ry for obj in obj_list])
    loc_lidar = calib.rect_to_lidar(loc)

    # print('loc\n', loc)
    # print('loc_lidar\n', loc_lidar)

    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)

    corners = boxes3d_to_corners3d_lidar(gt_boxes_lidar)
    # corners_lidar = calib.rect_to_lidar(corners)

    # bboxes3d = []
    # for o in objs:
    #     bbox2d, bbox3d = utils.compute_box_3d(o, P0)
    #     bboxes3d.append(bbox3d)

    bboxes3d = []
    for i in range(corners.shape[0]):
        bboxes3d.append(corners[i])

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

