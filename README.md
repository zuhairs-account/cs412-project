Project for CS412.

# Dynamic Single Source Shortest Path using Retroactive Priority Queues

## 📚 Overview

This project implements a **dynamic version of Dijkstra’s algorithm** using **retroactive priority queues**. The goal is to efficiently solve the **Dynamic Single Source Shortest Path (DSSSP)** problem, where the graph undergoes updates (insertions/deletions of edges/vertices) over time and shortest path queries must still be supported efficiently.

Traditional Dijkstra’s algorithm does not support dynamic changes efficiently. By incorporating retroactive data structures, we are able to track and handle historical changes while maintaining an optimized pathfinding process.

## 🚀 Motivation

Dynamic shortest path algorithms are crucial in real-time systems such as:
- Navigation and GPS systems
- Network routing protocols
- Dynamic logistics and supply chain optimizations

These systems require the shortest paths to be recalculated efficiently upon updates to the graph topology. Our proposed algorithm provides an efficient solution to this problem with better performance in terms of **update time**, **query time**, and **memory usage**.

## 🧠 Key Concepts

- **Retroactive Priority Queue**: A data structure that supports updates and queries in the past, allowing us to "rewind" time and make changes that affect the future state of the graph.
- **Dynamization of Dijkstra**: Incorporating dynamic behavior into Dijkstra’s algorithm to handle insertions and deletions without complete recomputation.
- **Efficient Update Handling**: Only affected vertices are updated after each change, reducing unnecessary recalculations.

## 🛠 Features

- Handles insertions and deletions of vertices and edges.
- Supports retroactive insert/delete operations.
- Supports shortest path queries at any time.
- Update time complexity: **O(n log m)**
- Optimized for time and memory usage.

