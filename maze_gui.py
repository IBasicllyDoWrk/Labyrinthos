import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, LifoQueue, PriorityQueue
import time
import tkinter as tk
from tkinter import ttk, filedialog, Toplevel
from PIL import Image, ImageTk

ALGORITHMS = {
    'BFS': 'bfs',
    'DFS': 'dfs',
    'Dijkstra': 'dijkstra',
    'A*': 'astar',
    'Flood Fill': 'flood_fill'
}

def load_maze(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def get_neighbors(pos, maze):
    x, y = pos
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            if maze[nx][ny] == 255:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, end):
    path = []
    while end in came_from:
        path.append(end)
        end = came_from[end]
    return path[::-1]

def bfs(maze, start, end):
    q = Queue()
    q.put(start)
    came_from = {}
    visited = {start}
    while not q.empty():
        current = q.get()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                q.put(neighbor)
    return []

def dfs(maze, start, end):
    stack = LifoQueue()
    stack.put(start)
    came_from = {}
    visited = {start}
    while not stack.empty():
        current = stack.get()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.put(neighbor)
    return []

def dijkstra(maze, start, end):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {}
    cost = {start: 0}
    while not pq.empty():
        current_cost, current = pq.get()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            new_cost = cost[current] + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                pq.put((new_cost, neighbor))
    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {}
    g_score = {start: 0}
    while not pq.empty():
        _, current = pq.get()
        if current == end:
            return reconstruct_path(came_from, end)
        for neighbor in get_neighbors(current, maze):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                came_from[neighbor] = current
                pq.put((f_score, neighbor))
    return []

def flood_fill(maze, start, end):
    return bfs(maze, start, end)

def draw_path(maze, path):
    color_img = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    for x, y in path:
        color_img[x, y] = [0, 0, 255]
    return color_img

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Solver")
        self.root.configure(bg="#2e2e2e")

        self.selected_algorithm = tk.StringVar(value="BFS")

        self.load_button = tk.Button(root, text="Load Maze", command=self.load_maze_dialog, bg="#444", fg="white")
        self.load_button.pack(pady=5)

        self.dropdown = ttk.Combobox(root, textvariable=self.selected_algorithm, values=list(ALGORITHMS.keys()))
        self.dropdown.pack(pady=5)

        self.solve_button = tk.Button(root, text="Solve Selected", command=self.solve_selected, bg="#444", fg="white")
        self.solve_button.pack(pady=5)

        self.compare_button = tk.Button(root, text="Compare Algorithms", command=self.compare_all, bg="#444", fg="white")
        self.compare_button.pack(pady=5)

        self.canvas = tk.Canvas(root, bg="#2e2e2e")
        self.canvas.pack()

        self.status = tk.Label(root, text="Load a maze to begin", fg="white", bg="#2e2e2e")
        self.status.pack()

        self.start = None
        self.end = None
        self.maze = None
        self.image_on_canvas = None
        self.scale_factor = 1.0

    def load_maze_dialog(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.maze = load_maze(file_path)
            self.original_maze = self.maze.copy()
            self.display_maze()
            self.start = None
            self.end = None
            self.status.config(text="Click on white areas to select START and END")

    def display_maze(self):
        h, w = self.maze.shape
        canvas_size = 600
        self.scale_factor = canvas_size / max(h, w)
        resized = cv2.resize(self.maze, (int(w * self.scale_factor), int(h * self.scale_factor)), interpolation=cv2.INTER_NEAREST)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.canvas.config(width=int(w * self.scale_factor), height=int(h * self.scale_factor))
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x = int(event.x / self.scale_factor)
        y = int(event.y / self.scale_factor)
        if self.maze[y][x] == 0:
            self.status.config(text="Cannot select black area. Choose a white pixel.")
            return
        if self.start is None:
            self.start = (y, x)
            self.status.config(text=f"Start set at {self.start}. Click to set END.")
        elif self.end is None:
            self.end = (y, x)
            self.status.config(text=f"End set at {self.end}. Choose algorithm and click solve.")
        else:
            self.start = (y, x)
            self.end = None
            self.status.config(text=f"New start set at {self.start}. Click to set END.")

    def solve_selected(self):
        if self.start and self.end:
            algo = globals()[ALGORITHMS[self.selected_algorithm.get()]]
            self.maze = self.original_maze.copy()
            path = []
            for step in algo(self.maze.copy(), self.start, self.end):
                path.append(step)
                temp = draw_path(self.original_maze.copy(), path)
                self.animate_step(temp)
            self.status.config(text=f"{self.selected_algorithm.get()} done: {len(path)} steps")

    def animate_step(self, img):
        resized = cv2.resize(img, (int(self.maze.shape[1] * self.scale_factor), int(self.maze.shape[0] * self.scale_factor)), interpolation=cv2.INTER_NEAREST)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        self.canvas.update()
        time.sleep(0.01)

    def compare_all(self):
        if not self.start or not self.end:
            self.status.config(text="Please select both START and END points.")
            return
        results = []
        for name in ALGORITHMS:
            algo = globals()[ALGORITHMS[name]]
            self.maze = self.original_maze.copy()
            start_time = time.time()
            path = algo(self.maze.copy(), self.start, self.end)
            end_time = time.time()
            results.append((name, len(path), round(end_time - start_time, 5)))
        self.show_comparison_table(results)

    def show_comparison_table(self, results):
        table_window = Toplevel(self.root)
        table_window.title("Comparison")
        tree = ttk.Treeview(table_window, columns=("Algorithm", "Steps", "Time"), show='headings')
        tree.heading("Algorithm", text="Algorithm")
        tree.heading("Steps", text="Steps")
        tree.heading("Time", text="Time (s)")
        for res in results:
            tree.insert("", "end", values=res)
        tree.pack(expand=True, fill="both", padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
