# 🧩 Maze Solver Visualizer

A Python GUI application for solving and comparing different pathfinding algorithms on maze images. Built using Tkinter and OpenCV, this app lets you load a maze, select start and end points, and visualize how algorithms like BFS, DFS, Dijkstra, A*, and Flood Fill solve the maze.

---

## 📸 Features

- Load maze images from your system
- Select **Start** and **End** points by clicking on the maze
- Choose and visualize any of the 5 included algorithms:
  - **BFS** (Breadth-First Search)
  - **DFS** (Depth-First Search)
  - **Dijkstra’s Algorithm**
  - **A\*** (A Star)
  - **Flood Fill**
- Animated visualization of pathfinding
- Side-by-side performance comparison of all algorithms (steps & time)

---

## 🗂️ Maze Images

Usable maze images are provided in the [`maze/`](maze/) folder. These are:
- Binary images (black & white)
- Walls must be black (`0`) and paths white (`255`)
- Format: `.png`, `.jpg`, or any image OpenCV can read

You can also add your own mazes to the `maze/` folder. Make sure the image:
- Is clear and monochrome
- Has continuous paths from start to end
- Doesn't contain grayscale noise

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/maze-solver-visualizer.git
   cd maze-solver-visualizer
