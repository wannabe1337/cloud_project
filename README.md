# Relay-Assisted Parallel Offloading Strategy for Multi-Source Tasks in Internet of Vehicles

## Overview

This repository contains a Python implementation of the research paper titled **"A Relay-Assisted Parallel Offloading Strategy for Multi-Source Tasks in Internet of Vehicles"**. This project was given as a course project in a team of 3 members. The paper proposes a novel approach to offloading computational tasks from vehicles (TaVs) to nearby Mobile Edge Computing (MEC) nodes, addressing challenges related to network bandwidth, task latency, and resource allocation.

## Research Paper Highlights

### Problem Addressed
- **Network Bandwidth Pressure**: Offloading tasks to MEC nodes alleviates pressure on network bandwidth.
- **Task Latency**: Reducing latency by processing tasks closer to the source.
- **Resource Allocation**: Ensuring fair allocation of resources among both stationary edge nodes (RSUs) and mobile edge nodes (vehicles).
- **Relay Assistance**: Using vehicles as relay nodes to assist in communication when edge nodes are out of range.

### Objectives
- Develop a 3-D road vehicle mobility model to predict vehicle movement and aid in designing the offloading strategy.
- Formulate an optimization problem to manage computing and communication resources effectively.
- Propose the RAPO (Relay-Assisted Parallel Offloading) strategy to minimize latency and efficiently handle multi-source tasks.

### Solution
- **System Models**: A 3-D scenario involving RSUs, TaVs generating tasks, and relay-assisted nodes.
- **Network Model**: Detailed representation of RSUs, vehicles, and their roles as service nodes and relay-assisted nodes.
- **Communication Model**: Utilizes a flat Rayleigh fading channel for upload links.
- **Computing Model**: Includes local and edge computing resources.

## Reference
 You can access the research paper [here](https://www.sciencedirect.com/science/article/abs/pii/S2213138824000158)
