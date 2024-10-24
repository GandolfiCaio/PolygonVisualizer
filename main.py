"""
The file contains the coordinates of the vertices of two convex polygons in random order,
one of which is completely inside the other.
Display these polygons in different colors on the screen.
Display the perimeter of the inner polygon on the screen.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import re
import random
from datetime import datetime
import time

# Function to read points from a file
def read_points_from_file(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r"Point \d+: x=(-?\d+); y=(-?\d+);", line.strip())
            if match:
                x, y = map(int, match.groups())
                points.append([x, y])
    return np.array(points)

# Function to generate random points
def generate_random_points(num_points, x_range, y_range):
    points = set()

    while len(points) < num_points:
        point = (random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))
        points.add(point)

    points_array = np.array(list(points))

    # Call the save function after generating the points
    save_points_to_file(points_array)
    return np.array(list(points))

def save_points_to_file(points):
    filename = f"data_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, 'w') as f:
        for i, (x, y) in enumerate(points, 1):
            f.write(f"Point {i}: x={x}; y={y};\n")

    print(f"Points saved to file {filename}")

# Function to calculate the perimeter of a polygon
def calculate_perimeter(points, hull):
    perimeter = 0.0
    vertices = hull.vertices
    for i in range(len(vertices)):
        p1 = points[vertices[i]]
        p2 = points[vertices[(i + 1) % len(vertices)]]
        distance = np.linalg.norm(p2 - p1)
        perimeter += distance
    return perimeter

"""
Algorithm 1:
Name: Convex Hull
Feature: Finds the smallest convex polygon that encompasses all points and generates random internal points, avoiding the hull's vertices.
Complexity:
            1) Time: O(n log n)
            2) Space: O(n)
"""
def convex_hull_algorithm(points):
    hull = ConvexHull(points)  # Find the outer convex hull
    hull_vertices = hull.vertices  # Get the vertices of the outer hull

    # Create an array for internal points
    inner_points = []

    # Generate internal points, avoiding the outer hull vertices
    while len(inner_points) < len(points) // 2:
        random_point = points[np.random.choice(len(points))]
        # Check if the point is not a vertex of the outer hull
        if not np.any(np.all(random_point == points[hull_vertices], axis=1)):
            inner_points.append(random_point)

    inner_points = np.array(inner_points)  # Convert the list to a NumPy array
    inner_hull = ConvexHull(inner_points)  # Build the convex hull for the internal points

    return hull, inner_hull, inner_points

"""
Algorithm 2:
Name: Delaunay Triangulation
Feature: Divides the plane into triangles so that no point is inside the circumcircle of any triangle,
and excludes points that are hull vertices.
Complexity:
            1) Time: O(n log n)
            2) Space: O(n)
"""
def delaunay_algorithm(points):
    hull = ConvexHull(points)  # Find the outer convex hull
    delaunay = Delaunay(points[hull.vertices])  # Perform Delaunay triangulation

    # Select internal points that are not part of the outer hull
    inner_points = points[delaunay.find_simplex(points) >= 0]  # Internal points
    inner_points = inner_points[~np.any(np.all(inner_points[:, np.newaxis] == points[hull.vertices], axis=2),
                                        axis=1)]  # Exclude points that are hull vertices

    # Check if there are enough internal points
    if len(inner_points) < 3:
        raise ValueError("Not enough internal points to construct a polygon.")

    inner_hull = ConvexHull(inner_points[:len(inner_points) // 2])  # Take the first half of internal points for the inner hull

    return hull, inner_hull, inner_points[:len(inner_points) // 2]

# Input choice for point entry method
choice = input("Enter '1' for random points or '2' to read from a file: ").strip().lower()

if choice == '1':
    num_points = int(input("Enter the number of points (up to 100): "))
    x_range = (-100, 100)  # Range for x coordinates
    y_range = (-100, 100)  # Range for y coordinates
    points = generate_random_points(num_points, x_range, y_range)
elif choice == '2':
    filename = input("Enter the file name: ")  # File name
    points = read_points_from_file(filename)
else:
    raise ValueError("Invalid choice. Please enter '1' or '2'.")

# Check if there are enough points to construct polygons
if len(points) < 3:
    raise ValueError("Not enough points to construct polygons.")

# Algorithm selection
algorithm_choice = input(
    "Choose an algorithm: '1' for Convex Hull or '2' for Delaunay Triangulation: ").strip().lower()

# Measure algorithm execution time
start_time = time.time()  # Start time measurement

if algorithm_choice == '1':
    hull, inner_hull, inner_points = convex_hull_algorithm(points)
elif algorithm_choice == '2':
    hull, inner_hull, inner_points = delaunay_algorithm(points)
else:
    raise ValueError("Invalid algorithm choice. Please enter '1' or '2'.")

execution_time = time.time() - start_time  # End time measurement

# Calculate perimeters
outer_perimeter = calculate_perimeter(points, hull)
inner_perimeter = calculate_perimeter(inner_points, inner_hull)

print(f"Outer polygon perimeter: {outer_perimeter:.2f}")
print(f"Inner polygon perimeter: {inner_perimeter:.2f}")
print(f"Algorithm execution time: {execution_time:.6f} seconds")  # Display execution time

# Visualize polygons
plt.plot(points[:, 0], points[:, 1], 'o')

# Draw the outer convex polygon
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# Draw the inner convex polygon
for simplex in inner_hull.simplices:
    plt.plot(inner_points[simplex, 0], inner_points[simplex, 1], 'r-')

plt.show()
