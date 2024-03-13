import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# constants:
H = 10  # wire z value in absolute, [m]
I = 20  # average wire current in the U.S, [A]
mu0 = 4 * np.pi * 1e-7  # constant, [H/m]
f = 20e3  # Hz
sigma = 10e-3  # S/m
NUMBER_OF_SEGMENTS = 1000  # precision of manual integral
TOLERANCE = 1e-10
MIN_X = -20
MAX_X = 20
MIN_Y = -20
MAX_Y = 20
NUM_OF_X = 1000
NUM_OF_Y = 1000


@jit(nopython=True)
def is_point_above_line(x_point, y_point, azimuth_line, x_line, y_line):
    """
    :param x_point: x coordinates of point
    :param y_point: y coordinates of point
    :param x_line: x coordinates of a point on the line
    :param y_line: y coordinates of a point on the line
    :param azimuth_line: angle of the line from x-axis, in radians
    :return: 1 if the point is above the line, -1 below and 0 on the line
    """
    # Calculate the slope of the line
    slope = np.tan(azimuth_line)

    # Use point-slope form to find the equation of the line (y - y1 = m * (x - x1))
    expected_y = slope * (x_point - x_line) + y_line

    # Compare y-coordinates to determine position
    if y_point > expected_y:
        return 1
    elif y_point < expected_y:
        return -1
    else:
        return 0


@jit(nopython=True)
def divide_wire_np(start_point: np.array, end_point: np.array, num_segments: int = NUMBER_OF_SEGMENTS):
    """
    :param start_point: 1d numpy array [x, y, z], start point of the line
    :param end_point: 1d numpy array [x, y, z], end point of the line
    :param num_segments: int, number of segments to divide the line into
    :return: 2d numpy array where each row is [x, y, z], first point is start_point last point is end_point and
            NUMBER_OF_SEGMENTS points in the middle with equal differences
    """
    # Create arrays for start and end points
    # Calculate the increments for each coordinate
    increments = (end_point - start_point) / num_segments

    # Create an array to hold points along the wire
    points = np.zeros((num_segments + 1, 3))  # Each row contains (x, y, z)

    # Generate points along the wire
    for i in range(num_segments + 1):
        points[i] = start_point + i * increments

    return points


@jit(nopython=True)
def calculate_biot_savart_field_from_wire_at_point(r_point: np.array, r_wire_start: np.array, r_wire_end: np.array):
    """
    :param r_point: 1d numpy array [x, y, z], point to calculate mf at
    :param r_wire_start: 1d numpy array [x, y, z], start point of the wire
    :param r_wire_end: 1d numpy array [x, y, z], end point of the line
    :return: 1d numpy array [Bx, By, Bz], mf at point using biot-savart law
    """
    # Divide wire into NUMBER_OF_SEGMENTS
    divided_wire = divide_wire_np(start_point=r_wire_start, end_point=r_wire_end)
    # Calculate dl
    dl = divided_wire[1] - divided_wire[0]  # because this is a straight line dl is the same for all points in line
    # Initialize sum, 1d numpy array [Bx, By, Bz]
    integral_sum = np.array([0., 0., 0.])

    # Calculate integral using a loop
    for l in divided_wire:
        # calculate using biot-savart, (dl cross r') / |r'|^3
        r_tag = r_point - l
        r_tag_norm = np.linalg.norm(r_tag)
        cross_product = np.cross(dl, r_tag)
        # add to the integral
        integral_sum += (cross_product / (r_tag_norm ** 3))

    # return integral sum * (mu0 * I) / (4 * pi)
    return mu0 * I * integral_sum / (4 * np.pi)


@jit(nopython=True)
def calculate_mf_at_grid_biot_savart(X: np.array, Y: np.array, wires: np.array):
    """
    :param X: 2d numpy array, output of np.meshgrid
    :param Y: 2d numpy array, output of np.meshgrid
    :param wires: 2d numpy array, each row is a wire of format (r_wire_start, r_wire_end)
    :return: Bx, By, Bz, 2d numpy array of the total mf in each point in the grid
    """
    # initialize Bx, By, Bz
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(X)
    # loop over all coordinates
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            # get point from meshgrid
            x = X[i, j]
            y = Y[i, j]

            for wire in wires:
                # calculate total mf at point
                Bx_val, By_val, Bz_val = calculate_biot_savart_field_from_wire_at_point(r_point=np.array([x, y, 0]),
                                                                                        r_wire_start=wire[0],
                                                                                        r_wire_end=wire[1])
                # update Bx, By, Bz in index i, j
                Bx[i, j] += Bx_val
                By[i, j] += By_val
                Bz[i, j] += Bz_val
    return Bx, By, Bz


@jit(nopython=True)
def distance_from_line_xy(x_point, y_point, x_line, y_line, azimuth):
    """
    :param x_point: x coordinates of point
    :param y_point: y coordinates of point
    :param x_line: x coordinates of a point on the line
    :param y_line: y coordinates of a point on the line
    :param azimuth: angle of the line from x-axis, in radians
    :return: distance of the point from the line
    """
    # Calculate equation of the line: y = mx + c
    m = np.tan(azimuth)
    c = y_line - m * x_line

    # Calculate distance according to the formula: |Ax + By + C| / sqrt(A^2 + B^2)
    distance = np.abs(m * x_point - y_point + c) / np.sqrt(m ** 2 + 1)

    return distance


@jit(nopython=True)
def mf_at_point_from_wire(r_point: np.array, phi_wire, r_wire: np.array):
    """
    :param r_point: 1d numpy array [x, y, z], point to calculate mf at
    :param phi_wire: angle of the wire from x-axis, in radians
    :param r_wire: 1d numpy array [x, y, z], point on the wire
    :return: bx, by, bz, The mf at the point from the wire
    """
    # get the distance in xy plane
    x_point, y_point, z_point = r_point
    x_wire, y_wire, z_wire = r_wire
    xy_distance = distance_from_line_xy(x_point=r_point[0], y_point=r_point[1], x_line=x_wire, y_line=y_wire,
                                        azimuth=phi_wire)
    # calculate distance from wire
    r = np.sqrt(xy_distance ** 2 + (r_wire[2] - r_point[2]) ** 2)
    # calculate magnitude of B, you may consider soil
    B = (mu0 * I) / (2 * np.pi * r) * np.exp(
        -r * np.sqrt(np.pi * f * mu0 * sigma))  # magnitude of ef in soil
    # get z direction
    above_line = is_point_above_line(x_point=x_point, y_point=y_point, azimuth_line=phi_wire, x_line=x_wire,
                                     y_line=y_wire)
    if z_wire > z_point:
        above_line *= -1
    if r == 0:
        return 0, 0, 0
    else:
        # get sin and cos of phi, use tolerance
        sin = np.sin(phi_wire)
        cos = np.cos(phi_wire)
        tolerance = 1e-10
        if abs(cos) < tolerance:
            cos = 0
        if abs(sin) < tolerance:
            sin = 0
        # calculate bx, by, bz
        bxy = B * (z_point - z_wire) / r
        bz = B * xy_distance / r * above_line
        bx = bxy * sin
        by = bxy * -cos
        return bx, by, bz


@jit(nopython=True)
def total_mf_at_point(r_point: np.array, wires: np.array):
    """
    :param r_point: 1d numpy array [x, y, z], point to calculate mf at
    :param wires: 2d numpy array, each row is a wire of format (x_wire, y_wire, z_wire, phi_wire (in radians))
    :return: bx, by, bz, Total mf from all wires at point
    """
    # initialize bx, by, bz values
    bx_total, by_total, bz_total = 0, 0, 0
    # loop over all wires
    for wire in wires:
        # calculate mf from wire
        bx_tmp, by_tmp, bz_tmp = mf_at_point_from_wire(r_point=r_point, phi_wire=wire[-1], r_wire=wire[:-1])
        # update bx, by, bz
        bx_total += bx_tmp
        by_total += by_tmp
        bz_total += bz_tmp
    return bx_total, by_total, bz_total


@jit(nopython=True)
def calculate_mf_at_grid(X: np.array, Y: np.array, wires: np.array):
    """
    :param X: 2d numpy array, output of np.meshgrid
    :param Y: 2d numpy array, output of np.meshgrid
    :param wires: 2d numpy array, each row is a wire of format (x_wire, y_wire, z_wire, phi_wire (in radians))
    :return: Bx, By, Bz, 2d numpy array of the total mf in each point in the grid
    """
    # initialize Bx, By, Bz
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(X)
    # loop over all coordinates
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            # get point from meshgrid
            x = X[i, j]
            y = Y[i, j]

            # calculate total mf at point
            Bx_val, By_val, Bz_val = total_mf_at_point(r_point=np.array([x, y, 0]), wires=wires)

            # update Bx, By, Bz in index i, j
            Bx[i, j] = Bx_val
            By[i, j] = By_val
            Bz[i, j] = Bz_val
    return Bx, By, Bz


def create_grid(min_x_val=MIN_X, max_x_val=MAX_X, min_y_val=MIN_Y, max_y_val=MAX_Y, num_of_x=NUM_OF_X,
                num_of_y=NUM_OF_Y):
    """
    :param min_x_val: min value of x
    :param max_x_val: max value of x
    :param min_y_val: min value of y
    :param max_y_val: max value of y
    :param num_of_x: number of x values in the grid
    :param num_of_y: number of y values in the grid
    :return: X, Y. np.meshgrid of coordinates according to the params.
    """
    # create meshgrid
    x_range = np.linspace(min_x_val, max_x_val, num_of_x)
    y_range = np.linspace(min_y_val, max_y_val, num_of_y)
    return np.meshgrid(x_range, y_range)


def plot_mf_wire(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """
    :param Bx: 2d numpy array of Bx value at every point
    :param By: 2d numpy array of By value at every point
    :param Bz: 2d numpy array of Bz value at every point
    :param X: 2d numpy array, output of np.meshgrid
    :param Y: 2d numpy array, output of np.meshgrid
    :return: nothing, creates figures of Bx, By, Bz
    """
    # plot
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title('mf (Bx)')
    plt.contourf(X, Y, Bx, cmap='viridis')
    plt.colorbar(label='mf Strength')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.subplot(132)
    plt.title('mf (By)')
    plt.contourf(X, Y, By, cmap='viridis')
    plt.colorbar(label='mf Strength')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.subplot(133)
    plt.title('mf (Bz)')
    # plt.contourf(X, Y, np.abs(Bz), cmap='viridis')
    plt.contourf(X, Y, Bz, cmap='viridis')
    plt.colorbar(label='mf Strength')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.tight_layout()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Example 1, 2 wires, both at z=-H, both go through (0, 0), one is parallel to x-axis, the other to y-axis
    # Create lines list, format [(x_wire0, y_wire0, z_wire0, phi_wire0)]
    lines = np.array([(0, 0, -H, np.pi / 2), (0, 0, -H, 0)])
    # Create grid
    X, Y = create_grid()
    # Calculate mf
    Bx, By, Bz = calculate_mf_at_grid(X, Y, lines)
    # plot
    plot_mf_wire(Bx=Bx, By=By, Bz=Bz, X=X, Y=Y)
    plt.show()

    # Example 2, 2 wires, both at z=-H, one from (5, -5) to (5, 5) (parallel to y-axis), the other from (5,
    # 5) to (-5, 5) (parallel to x-axis) Create lines list, format is ([[[x_wire0_start, y_wire0_start, z_wire0_start],
    # [x_wire0_end, y_wire0_end, z_wire0_end]], [[x_wire1_start, y_wire1_start, z_wire1_start],
    # [x_wire1_end, y_wire1_end, z_wire1_end]]])
    lines_biot_savart = np.array([[[5, -5, -H], [5, 5, -H]], [[5, 5, -H], [-5, 5, -H]]])
    # Create grid
    X_bs, Y_bs = create_grid(num_of_x=100, num_of_y=100)
    # Calculate mf
    Bx_bs, By_bs, Bz_bs = calculate_mf_at_grid_biot_savart(X=X_bs, Y=Y_bs, wires=lines_biot_savart)
    # plot
    plot_mf_wire(Bx=Bx_bs, By=By_bs, Bz=Bz_bs, X=X_bs, Y=Y_bs)
    plt.show()
