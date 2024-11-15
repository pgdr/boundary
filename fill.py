import numpy as np
import noise
from collections import deque
import matplotlib.pyplot as plt
import random

# Constants
GRID_SIZE = 200
RECT_WIDTH = 20
RECT_HEIGHT = 12
RECT_COLOR = 2  # Red
BLACK = 0
WHITE = 1
YELLOW = 3

SCALE = 6
THRESHOLD = 0.01
OCTAVES = 5


# Generate the 2D grid with Perlin noise
def generate_perlin_grid(size):
    grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            value = noise.pnoise2(i / SCALE, j / SCALE, octaves=OCTAVES)
            grid[i, j] = BLACK if value < THRESHOLD else WHITE
    return grid


# Add red rectangle in the middle
def add_red_rectangle(grid, width, height):
    start_x = (GRID_SIZE - width) // 2
    start_y = (GRID_SIZE - height) // 2
    for i in range(start_x, start_x + width):
        for j in range(start_y, start_y + height):
            grid[i, j] = RECT_COLOR


# BFS from red cells
def bfs_from_red(grid):
    visited = np.zeros_like(grid, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    queue = deque()

    # Find all red cells and enqueue them
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == RECT_COLOR:
                queue.append((i, j))
                visited[i, j] = True

    # BFS
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < GRID_SIZE
                and 0 <= ny < GRID_SIZE
                and not visited[nx, ny]
                and grid[nx, ny] != WHITE  # Do not expand white cells
            ):
                visited[nx, ny] = True
                queue.append((nx, ny))
    return visited


# Identify cells with unvisited neighbors
def mark_cells_with_unvisited_neighbors(grid, visited):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if visited[x, y]:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < GRID_SIZE
                        and 0 <= ny < GRID_SIZE
                        and not visited[nx, ny]
                    ):
                        grid[x, y] = YELLOW
                        break


# Visualize the matrix using Matplotlib
def visualize_matrix(grid):
    colors = {BLACK: "black", WHITE: "white", RECT_COLOR: "red", YELLOW: "yellow"}
    cmap = plt.matplotlib.colors.ListedColormap(
        [colors[BLACK], colors[WHITE], colors[RECT_COLOR], colors[YELLOW]]
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap, origin="upper")
    plt.axis("off")
    plt.title("Grid Visualization")
    plt.show()


# Main
if __name__ == "__main__":
    grid = generate_perlin_grid(GRID_SIZE)
    add_red_rectangle(grid, RECT_WIDTH, RECT_HEIGHT)
    visited = bfs_from_red(grid)
    mark_cells_with_unvisited_neighbors(
        grid, visited
    )  # Mark cells with unvisited neighbors

    # Visualize the grid
    visualize_matrix(grid)
