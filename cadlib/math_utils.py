import math
import numpy as np


def rads_to_degs(rads):
    """Convert an angle from radians to degrees"""
    return 180 * rads / math.pi


def angle_from_vector_to_x(vec):
    """computer the angle (0~2pi) between a unit vector and positive x axis"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """convert a vector in cartesian coordinates to polar(spherical) coordinates"""
    vec = vec.round(6)
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / norm) # (0, pi)
    phi = np.arctan(vec[1] / (vec[0] + 1e-15)) # (-pi, pi) # FIXME: -0.0 cannot be identified here
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """convert a vector in polar(spherical) coordinates to cartesian coordinates"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """represent a coordinate system by its rotation from the standard 3D coordinate system

    Args:
        normal_3d (np.array): unit vector for normal direction (z-axis)
        x_axis_3d (np.array): unit vector for x-axis

    Returns:
        theta, phi, gamma: axis-angle rotation 
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """build a coordinate system by the given rotation from the standard 3D coordinate system"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d


def weighted_random_sample(a, b, num_samples=1):
    if a == b:
        return a
    """
    在 [a, b] 区间内按数值大小加权采样
    参数:
        a, b: 区间范围
        num_samples: 采样数量
    返回:
        采样结果（float 或 list）
    """
    # 离散化区间，决定了生成形状的可能性
    x = np.linspace(a, b, 10)

    # 用二次函数 拟合权重
    weights = ((x - (a+b)/2.0) / (b - (a+b)/2.0))**2 + 0.1  # 权重 = x，可以替换为其他函数（如 x^2）
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 按权重采样
    samples = np.random.choice(x, size=num_samples, p=weights)
    samples = np.round(samples, 3)
    return samples[0] if num_samples == 1 else samples.tolist()


if __name__ == "__main__":
    print(weighted_random_sample(0.0, 0.0))
    # count1, count2, count3, count4, count5 = 0, 0, 0, 0, 0
    # for i in range(50):
    #     k = weighted_random_sample(0.25,0.75)
    #     print(k)
    #     if k == 0.25:
    #         count1 += 1
    #     if k == 0.375:
    #         count2 += 1
    #     if k == 0.5:
    #         count3 += 1
    #     if k == 0.625:
    #         count4 += 1
    #     if k == 0.75:
    #         count5 += 1
    # print(count1)
    # print(count2)
    # print(count3)
    # print(count4)
    # print(count5)