#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy

import numpy as np
import pcl
# from sklearn.decomposition import PCA

import tf
import tf2_ros
import geometry_msgs.msg
from arc_utilities import transformation_helper

from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt


kinect_mocap_name = {"kinect2_victor_head": "mocap_Kinect2VictorHead_Kinect2VictorHead",
                     "kinect2_roof":        "mocap_Kinect2BlockRoof_Kinect2BlockRoof",
                     "kinect2_tripodA":     "mocap_Kinect2TripodA_Kinect2TripodA"}


def to_unit(vec):
    norm = np.linalg.norm(vec)
    return vec/norm


def xy_to_quaternion(x, y):
    """
    use perpendicular vectors x,y to defined rotation quaternion of the plane
    :param x: np.array.shape = (3,)
    :param y: np.array.shape = (3,)
    :return: np.array.shape = (4,)
    """
    z = np.cross(x, y)
    x = to_unit(x)
    y = to_unit(y)
    z = to_unit(z)
    rotmat = np.vstack((x, y, z)).transpose()
    rotmat_h = np.identity(4)
    rotmat_h[0:3, 0:3] = rotmat
    quat = tf.transformations.quaternion_from_matrix(rotmat_h)
    return quat


class Projector:
    def __init__(self, model):
        """model[a,b,c,d] plane: ax+by+cz+d=0"""
        n = model[0:3]
        norm = np.linalg.norm(n)
        dist = -model[3]/norm
        normal_vec = n/norm

        if np.dot(normal_vec, np.array([0, 1, 0])) < 0:
            # nomral should point down
            normal_vec = -1.0 * normal_vec
            dist = -dist

        center = normal_vec*dist
        plane_y = np.cross(normal_vec, np.array([1, 0, 0]))
        plane_y /= np.linalg.norm(plane_y)
        plane_x = np.cross(plane_y, normal_vec)
        plane_x /= np.linalg.norm(plane_x)
        plane_to_world = np.identity(4)
        plane_to_world[0:3, 0] = plane_x
        plane_to_world[0:3, 1] = plane_y
        plane_to_world[0:3, 2] = normal_vec
        plane_to_world[0:3, 3] = center

        # center = np.array([0, 0, 0])
        # quat = xy_to_quaternion(plane_x, plane_y)
        # tf2_broadcast(center, quat, parent_frame_id='kinect2_victor_head_rgb_optical_frame', child_frame_id='table')

        world_to_plane = np.linalg.inv(plane_to_world)
        self.plane_to_world = plane_to_world # Mapping from localplane to world frame: p_world = T * p_local
        self.world_to_plane = world_to_plane # Mapping from world frame to local plane: p_local = T * p_world

    def to_plane(self, cloud_arr):
        """
        transform 3d points to 2d coordinate on plane
        :param cloud_arr: np.array().shape = (3, n)
        :return: np.array().shape = (2,n)
        """
        if cloud_arr.shape[0] != 3:
            raise RuntimeError('Incorrect cloud_arr shape')
        world_to_plane = self.world_to_plane
        cloud_h = np.vstack((cloud_arr, np.ones((1, cloud_arr.shape[1]))))
        cloud_plane = np.matmul(world_to_plane, cloud_h)
        plane_arr = cloud_plane[0:3]/cloud_plane[3]
        plane_arr = plane_arr[0:2]
        return plane_arr

    def to_world(self, mat):
        """
        transform 2d points on plane back to world frame
        :param mat: np.array().shape = (2,n)
        :return: np.array().shape = (3, n)
        """
        if mat.shape[0] != 2:
            raise RuntimeError('Incorrect cloud_arr shape')
        plane_to_world = self.plane_to_world
        mat_h = np.vstack((mat, np.zeros((1, mat.shape[1])), np.ones((1, mat.shape[1]))))
        mat_world_h = np.matmul(plane_to_world, mat_h)
        mat_world = mat_world_h[0:3]/mat_world_h[3]
        return mat_world

    def to_world_rot_only(self, mat):
        """
        transform vectors on 2d coordinate back to world frame
        :param mat: np.array().shape = (2,n)
        :return: np.array().shape = (3, n)
        """
        if mat.shape[0] != 2:
            raise RuntimeError('Incorrect cloud_arr shape')
        plane_to_world = self.plane_to_world
        rot_mat = plane_to_world[0:3, 0:3]
        mat = np.vstack((mat, np.zeros((1, mat.shape[1]))))
        mat_world = np.matmul(rot_mat, mat)
        return mat_world


def ransac_plane(orig_cloud, distance_threshold = 0.01):
    """
    find a plane in point cloud
    :param orig_cloud: PCL::PointCloud
    :param distance_threshold: float
    :return: [int], [a,b,c,d](ax+by+cz+d=0)
    """
    seg = orig_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    indices, model = seg.segment()
    return indices, model


def ransac_normal_plane(cloud, distance_threshold=0.01):
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)
    seg.set_normal_distance_weight(0.01)
    seg.set_max_iterations(200)
    indices, model = seg.segment()
    return indices, model


def robust_min_max(theta, plane, bins=400):
    """
    Return left and right edge of square wave like data
    :param theta: angle in RAD, start from x positive, counter clockwise
    :param plane: 2xn data
    :return: left edge, right edge, min derivative, max derivative
    """
    vec = np.array([[np.cos(theta), np.sin(theta)]])
    projected = np.matmul(vec, plane)
    hist, bins = np.histogram(projected.T, bins=bins)
    hist = gaussian_filter1d(hist, 1)
    kernel = np.array([-1, 0, 1])
    conv = np.convolve(hist, kernel)
    mini, maxi = conv[1:-1].argmin(), conv[1:-1].argmax()
    minbin = (bins[mini] + bins[mini+1]) / 2.0
    maxbin = (bins[maxi] + bins[maxi+1]) / 2.0
    minval = conv[mini]
    maxval = conv[maxi]
    return minbin, maxbin, minval, maxval


def component_min_max(xyvec, plane):
    projected = np.matmul(xyvec.transpose(), plane)
    # trimmed = stats.trimboth(projected, 0.01)
    trimmed = projected
    print(trimmed.min(), trimmed.max())
    return trimmed.min(), trimmed.max()


def stats_outlier_remove(cloud):
    """
    filter points 3stdev away
    :param cloud: PCL::PointCloud
    :return: PCL::PointCloud
    """
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(3)
    return fil.filter()


def robust_caliper_localization(plane, division_num=90, visualization=False):
    """
    rotating caliper with robust min&max
    :param plane: 2*n array
    :param division_num: divisions per 90 degree
    :return: x_axis(2,), y_axis(2,), center(2,), inlier_percent(float)
    """

    # find edge with robust caliper in each direction
    thetas = []
    min_arr = []
    max_arr = []
    for i in range(0, division_num*2):
        theta = np.pi / float(division_num*2) * float(i)
        minbin, maxbin, minval, maxval = robust_min_max(theta, plane)
        thetas.append(theta)
        min_arr.append(minval)
        max_arr.append(maxval)

    if visualization:
        plt.plot(thetas, min_arr)
        plt.plot(thetas, max_arr)
        plt.show()

    # use perpendicular as constrain, find sharpest edge
    thetas_np = np.array(thetas)
    min_np = np.array(min_arr)
    max_np = np.array(max_arr)
    diff_np = max_np - min_np
    length = diff_np.shape[0]
    folded_sum = diff_np[:length/2] + diff_np[length/2:]

    if visualization:
        plt.plot(thetas_np[:length/2], folded_sum)
        plt.show()

    # find edge length and center
    major_angle = thetas[folded_sum.argmax()]
    second_angle = major_angle + np.pi/2.0
    major_min, major_max, minval, maxval = robust_min_max(major_angle, plane)
    second_min, second_max, minval, maxval = robust_min_max(second_angle, plane)
    major_axis = np.array([np.cos(major_angle), np.sin(major_angle)])
    second_axis = np.array([np.cos(second_angle), np.sin(second_angle)])

    # visualize
    if visualization:
        llcorner = major_min * major_axis + second_min * second_axis
        urcorner = major_max * major_axis + second_max * second_axis
        ax = plt.subplot(aspect='equal')
        ax.scatter(plane[0], plane[1])
        rect = plt.Rectangle(llcorner, (major_max - major_min), (second_max - second_min), fill=False, color='r',
                             angle=(major_angle / np.pi) * 180.0)
        ax.add_patch(rect)
        plt.show()

    # decide x(perpendicular with short edge) and y
    x_angle = major_angle
    if (major_max - major_min) > (second_max - second_min):
        x_angle = second_angle
    x_axis = np.array([np.cos(x_angle), np.sin(x_angle)])
    if np.dot(x_axis, np.array([0, 1])) < 0:
        x_angle += np.pi
    y_angle = x_angle - np.pi / 2.0

    x_axis = np.array([np.cos(x_angle), np.sin(x_angle)])
    y_axis = np.array([np.cos(y_angle), np.sin(y_angle)])
    x_min, x_max, _, _ = robust_min_max(x_angle, plane)
    y_min, y_max, _, _ = robust_min_max(y_angle, plane)
    center = (x_max + x_min) / 2.0 * x_axis + (y_max + y_min) / 2.0 * y_axis

    print "x size: ", x_max - x_min, " meters. This should be close to 30 inches. ", 30.0 * 2.54 / 100.0
    print "y size: ", y_max - y_min, " meters. This should be close to 42 inches. ", 42.0 * 2.54 / 100.0

    # evaluate
    major_mat = major_axis.reshape((1, 2))
    second_mat = second_axis.reshape((1, 2))
    major_dot = np.matmul(major_mat, plane)[0]
    second_dot = np.matmul(second_mat, plane)[0]

    major_in = np.logical_and(major_dot <= major_max, major_dot >= major_min)
    second_in = np.logical_and(second_dot <= second_max, second_dot >= second_min)
    both_in = np.logical_and(major_in, second_in)
    total = both_in.shape[0]
    inlier_num = both_in.sum()
    inlier_percent = float(inlier_num) / float(total)

    return x_axis, y_axis, center, inlier_percent


def find_candidate_plane(pcl_cloud, table_ratio=1.428, ratio_threshold = 0.02):
    """
    Iteratively find a plane containing table
    :param pcl_cloud: PCL::PointCloud
    :param table_ratio: expected table ratio (width/height)
    :param ratio_threshold:
    :return: PCL::PointCloud, [a,b,c,d](ax+by+cz+d=0)
    """
    on_table = None
    table_model = None

    pca = PCA(n_components=3)
    i = 0
    while True:
        if i > 5:
            raise RuntimeError('Failed to find candidate table plane.')
        i += 1

        # find plane
        indices, model = ransac_normal_plane(pcl_cloud)
        inliers = pcl_cloud.extract(indices, negative=False)

        # remove points 3 stdev away
        # inliers_filtered = stats_outlier_remove(inliers)
        # inliers_filtered_arr = inliers_filtered.to_array()
        inliers_filtered_arr = inliers.to_array()

        # check if plane is table using ratio of two sides
        pca.fit(inliers_filtered_arr)
        sv = pca.singular_values_
        sv_ratio = sv[0]/sv[1]
        print('candidate ratio:', sv_ratio)
        is_table = abs(sv_ratio-table_ratio) < ratio_threshold

        # test
        proj = Projector(model)
        on_plane_filtered_arr_2d = proj.to_plane(inliers_filtered_arr.T)

        # find 2d pose and center of the table using PCA
        xy2d, center2d = pca_localization(on_plane_filtered_arr_2d)

        if is_table:
            # found table
            on_table = inliers
            table_model = model
            break
        else:
            # not table
            pcl_cloud = pcl_cloud.extract(indices, negative=True)
            print(pcl_cloud.width * pcl_cloud.height)

    return on_table, table_model


def pca_localization(plane):
    """
    :param plane: plane projected to 2 by n ndarray
    :return: ndarray([[xvec][yvec][center]).transpose()
    """
    pca = PCA(n_components=2)
    pca.fit(plane.transpose())
    comp = pca.components_
    xvec = None
    yvec = None
    plane_y = np.array([0, 1])
    if abs(np.dot(comp[0], plane_y)) > abs(np.dot(comp[1], plane_y)):
        # comp[0] is vertical
        yvec = comp[0]
        xvec = comp[1]
    else:
        xvec = comp[0]
        yvec = comp[1]

    xmin, xmax = component_min_max(xvec, plane)
    ymin, ymax = component_min_max(yvec, plane)
    xcenter = (xmin + xmax) / 2.0
    ycenter = (ymin + ymax) / 2.0
    center = xvec*xcenter + yvec*ycenter
    lowerleft = xvec*xmin + yvec*ymax

    # plt.scatter(orig_plane[0], orig_plane[1])
    plt.scatter(plane[0], plane[1])
    plt.arrow(center[0], center[1], xvec[0], xvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01,
              color='r')
    plt.arrow(center[0], center[1], yvec[0], yvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01,
              color='r')
    angle = np.arctan2(xvec[1], xvec[0]) * 180 / np.pi
    rect = plt.Rectangle(lowerleft, (xmax - xmin), (ymax - ymin), fill=False, color='r', angle=angle)
    ca = plt.gca()
    ca.add_patch(rect)
    plt.show()
    out = np.vstack((xvec, yvec)).transpose()
    return out, center.reshape((1, 2)).transpose()


# ===================capture pointcloud2=======================
def msg_to_pcl(msg):
    arr = ros_numpy.numpify(msg)
    arr1d = arr.reshape(arr.shape[0]*arr.shape[1])
    reca = arr1d.view(np.recarray)
    newarr = np.vstack((reca.x,reca.y,reca.z))
    filtered = newarr[:, ~np.isnan(newarr).any(axis=0)]
    cloud = pcl.PointCloud()
    cloud.from_array(filtered.T)
    # pdb.set_trace()
    return cloud
# ===================end capture pointcloud2=======================


# ===================visualization========================
def draw(plt, plane, center, xvec, yvec):
    plt.scatter(plane[0], plane[1])
    plt.arrow(center[0], center[1], xvec[0], xvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01, color='r')
    plt.arrow(center[0], center[1], yvec[0], yvec[1], shape='full', lw=3, length_includes_head=True, head_width=.01, color='r')
    plt.show()
# ================end visualization=======================



class TableExtraction:

    def __init__(self, kinect_name):
        self.kinect_name = kinect_name
        self.table_found = False

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        rospy.Subscriber(self.kinect_name + '/qhd/points', PointCloud2, self.callback)

    def callback(self, msg):
        # print(msg)
        if not self.table_found:
            print "invoking callback (extraction)"
            self.table_found = True
            cloud = msg_to_pcl(msg)
            self.extract_table_pose(msg.header.frame_id, cloud)
        else:
            self.print_camera_transform()

    def extract_table_pose(self, native_frame_id, pcl_cloud, inlier_percent_threshold=0.7):
        xy2d, center2d = None, None
        proj = None
        i = 0
        while True:
            if i > 5:
                raise RuntimeError('Failed to find candidate table plane.')

            # find plane
            indices, model = ransac_normal_plane(pcl_cloud)
            inliers = pcl_cloud.extract(indices, negative=False)

            # project to 2d
            proj = Projector(model)
            plane = proj.to_plane(inliers.to_array().T)

            # localize table
            x_axis, y_axis, center, inlier_percent = robust_caliper_localization(plane, visualization=False)

            if inlier_percent < inlier_percent_threshold:
                # not a table
                pcl_cloud = pcl_cloud.extract(indices, negative=True)
                print('Not a table, inlier percentage:', inlier_percent)
                print('Trimmed cloud size:', pcl_cloud.width * pcl_cloud.height)
            else:
                # found a table
                xy2d = np.vstack((x_axis, y_axis)).T
                center2d = center.reshape((1, 2)).T
                print('Found a table, inlier percentage:', inlier_percent)
                break

        # transform into 3d world frame
        xy3d = proj.to_world_rot_only(xy2d)
        center3d = proj.to_world(center2d)
        transp = xy3d.transpose()
        x = transp[0]
        y = transp[1]
        center = center3d.transpose()[0]
        quat = xy_to_quaternion(x, y)

        # start broadcast static transformation
        self.tf2_broadcast(center, quat,
                           parent_frame_id=native_frame_id,
                           child_frame_id='table_surface_from_kinect_calibration')

    def print_camera_transform(self):
        # TODO: Make these transform names (at least for the kinect) paramaters
        mocap_plate_to_table_surface = self.get_tf_transform(
            parent=kinect_mocap_name[self.kinect_name], child='table_surface')

        table_as_calibration_to_kinect_link = self.get_tf_transform(
            parent='table_surface_from_kinect_calibration', child=self.kinect_name + '_link')

        table_as_calibration_to_kinect_link = np.dot(
            np.array([[-1.0,  0.0, 0.0, 0.0],
                      [ 0.0, -1.0, 0.0, 0.0],
                      [ 0.0,  0.0, 1.0, 0.0],
                      [ 0.0,  0.0, 0.0, 1.0]]),
            table_as_calibration_to_kinect_link)

        mocap_plate_to_kinect2_link = np.matmul(mocap_plate_to_table_surface, table_as_calibration_to_kinect_link)

        trans, quat =  transformation_helper.ExtractFromMatrix(mocap_plate_to_kinect2_link)

        print "calced mocap to link:\n", mocap_plate_to_kinect2_link
        print "Trans: ", trans
        print "Quat:  ", quat
        print "\n"

        # import IPython
        # IPython.embed()

    def get_tf_transform(self, parent, child, verbose=False):
        try:
            while not rospy.is_shutdown() and \
                    not self.tf_buffer.can_transform(child, parent, rospy.Time(), rospy.Duration(secs=0, nsecs=500*1000*1000)):
                print "Waiting for TF frames ", parent, " and ", child
            transform = self.tf_buffer.lookup_transform(parent, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("No transform available: %s to %s", parent, child)
            return

        return transformation_helper.BuildMatrixRos(transform.transform.translation, transform.transform.rotation)

    def tf2_broadcast(self, trans, quat, parent_frame_id, child_frame_id):
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = parent_frame_id
        static_transformStamped.child_frame_id = child_frame_id
        # trans = [1.00619416, -0.19541004, -0.42314096]
        # trans = [-0.01299984, -0.13024425,  0.95459]
        # trans = [-0.00621743, -0.13153817,  0.9555135]
        static_transformStamped.transform.translation.x = trans[0]
        static_transformStamped.transform.translation.y = trans[1]
        static_transformStamped.transform.translation.z = trans[2]
        # quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(static_transformStamped)
        print("Start to broadcast tf2 '%s' to '%s'" % (parent_frame_id, child_frame_id))


def test(fname):
    pcl_cloud = pcl.load(fname)
    extractor = TableExtraction()
    extractor.extract_table_pose(pcl_cloud)


if __name__ == "__main__":
    rospy.init_node('table_calibration_tf2_broadcaster', anonymous=False)
    extractor = TableExtraction(kinect_name="kinect2_tripodA")
    # extractor = TableExtraction(kinect_name="kinect2_victor_head")
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
