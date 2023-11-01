import math
import matplotlib.pyplot as plt


def calculate_distance(point1, point2):
    ''' Beregn afstanden mellem to punkter (f.eks. euklidisk afstand) '''
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2)**2)**0.5


def nearest_neighbor(points):
    # Find den korteste vej ved at bruge den nærmeste nabo-algoritme
    n = len(points)

    if n <= 2:
        return points

    path = [points[0]]
    unvisited = set(points[1:])

    while unvisited:
        nearest_point = min(
            unvisited, key=lambda point: calculate_distance(path[-1], point))
        path.append(nearest_point)
        unvisited.remove(nearest_point)

    # Tilføj sidste vej tilbage til startpunktet
    # path.append(points[0])
    return path


# Test algoritmen med nogle punkter
points = [(0, 0), (2, 4), (3, 1), (5, 3), (6, 6)]
shortest_path = nearest_neighbor(points)
print("Korteste rute:", shortest_path)


x, y = zip(*shortest_path)

plt.figure(figsize=(8, 6))
plt.scatter(*zip(*points), color='red', label='Punkter')
plt.plot(x, y, 'bo-', markersize=8, label='Rute')
plt.plot([shortest_path[-1][0], shortest_path[0][0]], [shortest_path[-1][1], shortest_path[0][1]], 'bo-')  # Tilslut sidste og første punkt
plt.legend()
plt.show()

