# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
import numpy as np


def direction_to(src, target):
    """
    Returns the direction to move from src to target
    0 = center, 1 = up, 2 = right, 3 = down, 4 = left
    """
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2  # right
        else:
            return 4  # left
    else:
        if dy > 0:
            return 3  # down
        else:
            return 1  # up


def manhattan_distance(pos1, pos2):
    """
    Calculate Manhattan distance between two positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_adjacent_positions(pos, width, height):
    """
    Returns a list of valid adjacent positions (up, right, down, left)
    """
    x, y = pos
    adjacent = []

    # Check up, right, down, left
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            adjacent.append((nx, ny))

    return adjacent


def positions_in_range(center, distance, width, height):
    """
    Returns all positions within a given Manhattan distance from the center
    """
    x, y = center
    positions = []

    for dx in range(-distance, distance + 1):
        for dy in range(-distance, distance + 1):
            if abs(dx) + abs(dy) <= distance:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    positions.append((nx, ny))

    return positions


def is_visible(pos, unit_positions, sensor_range):
    """
    Determines if a position is visible by any unit in unit_positions
    given the sensor_range
    """
    for unit_pos in unit_positions:
        if manhattan_distance(pos, unit_pos) <= sensor_range:
            return True
    return False


def calculate_energy_gain(pos, energy_nodes, energy_functions):
    """
    Calculate the energy gain at a position given energy nodes and their functions
    """
    total_energy = 0
    for i, node_pos in enumerate(energy_nodes):
        if i < len(energy_functions):
            distance = manhattan_distance(pos, node_pos)
            energy = energy_functions[i](distance)
            total_energy += energy
    return total_energy