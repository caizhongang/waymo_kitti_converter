import cv2
import numpy as np

pc_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/velodyne/000002.bin'
img_pathname = '/home/alex/github/waymo_to_kitti_converter/tools/kitti/image_2/000002.png'


def cart_to_homo(mat):
    mat = np.vstack([mat, np.ones((1, mat.shape[1]))])
    return mat


def main():
    v2c = np.array([
        [7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
        [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
        [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01]])
    r0 = np.array([
        [9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03],
        [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03],
        [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]])
    px = np.array([
        [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
        [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
        [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

    pc = np.fromfile(pc_pathname, dtype=np.float32).reshape((-1, 4))[:, :3]

    # filter all behind image plane
    keep = []
    for i in range(pc.shape[0]):
        p = pc[i, :]
        if p[0] > 0:
            keep.append(p)
    pc = np.vstack(keep)

    img = cv2.imread(img_pathname)

    tmp = np.eye(4)
    tmp[:3, :3] = r0
    r0 = tmp

    pc = np.transpose(pc) # (n,3), (3,n)
    pc = cart_to_homo(pc)

    v2c = cart_to_homo(v2c)

    print(px.shape, r0.shape, v2c.shape, pc.shape)
    pt = px @ r0 @ v2c @ pc
    pt = pt[:2] / pt[2]

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # draw
    for i in range(pt.shape[1]):
        x = pt[0, i]
        y = pt[1, i]
        distance = np.sqrt(np.sum(pc[:3, i] ** 2))
        color = cmap[np.clip(640/distance,0, 255).astype(np.int), :]
        cv2.circle(img, (int(x), int(y)), 1, tuple(color))

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == 27:  # exit
            break
        elif key != -1:
            print('Undefined key:', key)


if __name__ == '__main__':
    main()


