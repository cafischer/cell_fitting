from __future__ import division
import numpy as np


def find_local_minimum_for_each_candidate(candidates, fitness, n_redrawn_candidates=50, radius=10 ** (-2)):
    local_minimum_per_candidate = dict()
    for candidate in candidates:
        best_candidate = candidate
        last_best_candidate = candidate
        while True:
            last_best_candidate = best_candidate
            new_candidates = [random_in_circle(best_candidate, radius) for i in range(n_redrawn_candidates)]
            fitness_new_candidates = [fitness(c) for c in new_candidates]
            best_candidate = new_candidates[np.argmin(fitness_new_candidates)]
            if fitness(best_candidate) >= fitness(last_best_candidate):
                break
        local_minimum_per_candidate[tuple(candidate)] = list(last_best_candidate)
    return local_minimum_per_candidate


def assign_candidates_to_attraction_basin(local_minimum_per_candidate, distance, delta=10 ** (-3)):
    attraction_basins = dict()
    for candidate in local_minimum_per_candidate.keys():
        inserted = False
        for minimum in attraction_basins.keys():
            if np.all(distance(local_minimum_per_candidate[candidate], minimum) <= delta):
                attraction_basins[minimum].append(list(candidate))
                inserted = True
        if not inserted:
            attraction_basins[tuple(local_minimum_per_candidate[candidate])] = [list(candidate)]
    return attraction_basins


def random_in_circle(middle, radius):
    return np.random.uniform(np.array(middle)-radius, np.array(middle)+radius)


def get_true_local_minima(x, order):
    """
    Minimum = all numbers before and after the minimum are sequentially greater equal.
    If equals occur at the trough only one minimum is assigned.
    :param x: Array in which to find the minima.
    :type x: ndarray
    :param order: How many points to consider before and after the minimum.
    :type order: int
    :return: Indices of the minima.
    :rtype: list
    """
    minima = list()
    slope = np.diff(x)
    i = 0
    while i < (len(x)-2*order):
        if np.all(slope[i:i+order] < 0) and np.all(slope[i+order:i+2*order] > 0):
            minima.append(i+order)
            i += 2*order
        else:
            i += 1
    return minima


def get_local_minima(x, order):
    """
    Minimum = all numbers before and after the minimum are sequentially greater equal.
    If equals occur at the trough only one minimum is assigned.
    :param x: Array in which to find the minima.
    :type x: ndarray
    :param order: How many points to consider before and after the minimum.
    :type order: int
    :return: Indices of the minima.
    :rtype: list
    """
    minima = list()
    slope = np.diff(x)
    for i in range(len(x)-2*order):
        if np.all(slope[i:i+order] <= 0) and np.all(slope[i+order:i+2*order] >= 0):
            minima.append(i+order)
    return minima


def get_local_minima_2d(mat, order):
    minima_x = list()
    minima_y = list()
    minima_ullr = list()
    minima_urll = list()
    minima = list()

    # check x, y direction
    for offset in range(mat.shape[0]):
        minima_x.extend([[offset, y] for y in get_local_minima(mat[offset, :], order=order)])
    for offset in range(mat.shape[1]):
        minima_y.extend([[x, offset] for x in get_local_minima(mat[:, offset], order=order)])

    # check diagonals
    x_index_mat, y_index_mat = np.indices((mat.shape[0], mat.shape[1]))
    x_index_mat_rot = np.rot90(x_index_mat, 3)
    y_index_mat_rot = np.rot90(y_index_mat, 3)

    len_mat = max(mat.shape[0], mat.shape[1])

    for offset in np.arange(-1*(len_mat-2), len_mat-1):
        diagonal_ullr = np.diagonal(mat, offset=offset)
        add_minima_from_diagonal(diagonal_ullr, x_index_mat, y_index_mat, minima_ullr, offset, order)

    for offset in np.arange(-1*(len_mat-2), len_mat-1):
        diagonal_urll = np.diagonal(np.rot90(mat, 3), offset=int(offset))
        add_minima_from_diagonal(diagonal_urll, x_index_mat_rot, y_index_mat_rot, minima_urll, offset, order)

    # find common minima
    for minimum in minima_x:
        if (removed_from_list(minima_y, minimum) and removed_from_list(minima_ullr, minimum)
                and removed_from_list(minima_urll, minimum)):
            minima.append(minimum)
    return minima


def add_minima_from_diagonal(diagonal, x_index_mat, y_index_mat, minima, offset, order):
    x_indices = np.diag(x_index_mat, k=offset)
    y_indices = np.diag(y_index_mat, k=offset)
    for min in get_local_minima(diagonal, order=order):
        minima.append([x_indices[min], y_indices[min]])


def removed_from_list(some_list, elem):
    try:
        some_list.remove(elem)
        return True
    except ValueError:
        return False


# tests

def test_assign_candidates_to_local_minima():
    candidates = np.linspace(-np.pi, 2*np.pi, 10)
    fitness = np.sin
    distance = lambda x, y: np.abs(x-y)
    minima = assign_candidates_to_attraction_basin(candidates, fitness, distance)
    minima_keys = minima.keys()
    assert np.isclose(minima_keys[minima_keys < 0], -1.5707963267948966, atol=10 ** (-3))
    assert np.isclose(minima_keys[minima_keys > 0], 4.71238898038469, atol=10 ** (-3))
    assert np.allclose(minima[minima_keys[minima_keys < 0]],
                       [-3.1415926535897931, -2.0943951023931957, -1.0471975511965979, 0.0, 1.0471975511965974])
    assert np.allclose(minima[minima_keys[minima_keys > 0]],
                       [2.0943951023931948, 3.1415926535897931, 4.1887902047863905, 5.2359877559829879,
                        6.2831853071795862])


def test_local_minima():
    assert get_local_minima(np.array([2, 1, 0, 1, 2]), 1) == [2]
    assert get_local_minima(np.array([2, 1, 0, 1, 2, 1, 0, 1, 1]), 1) == [2, 6]
    assert get_local_minima(np.array([2, 1, 0, 1, 2]), 2) == [2]
    assert get_local_minima(np.array([2, 1, 0, 1, 2]), 4) == []
    assert get_local_minima(np.array([2, 1, 0, 0, 2, 3, 0, 1]), 1) == [2, 6]
    assert get_local_minima(np.array([2, 1, 0, 0, 2, 3, 0, 1]), 2) == [2]


def test_minima2d():
    mat = np.array([[1, 2, 1, 4], [3, 0, 2, 4], [0, 1, 2, 4], [5, 4, 4, 5]])
    minima = get_local_minima_2d(mat, 1)
    assert minima == [[1, 1]]
    mat = np.array([[1, 1, 4, 2, 1], [2, 0, 5, 0, 1], [1, 3, 2, 1, 2]])
    minima = get_local_minima_2d(mat, 1)
    assert minima == [[1, 1], [1, 3]]


if __name__ == '__main__':
    #test_assign_candidates_to_local_minima()
    #test_local_minima()
    test_minima2d()

