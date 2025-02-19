# 🎃 Pumpkin Pulse 🎃

<div align="center">
  <img src="https://github.com/loliverhennigh/PumpkinPulse/blob/dev/assets/cover_image.png">
  <p><strong>🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃</strong></p>
</div>

Pumpkin Pulse is a python package for scalable GPU accelerated simulations of plasma. The objective is to allow for full device, first principle modeling of a variety of fusion reactors such as Field Reversed Configuration, Z-Pinch, and Inertial Confinement Fusion machines. Key features include,

- GPU acceleration using NVIDIA [Warp](https://github.com/NVIDIA/warp), (core kernels achieving 95% memory bandwidth utilization🎃).
- Scalable to multiple GPUs and nodes using MPI, (linear weak scaling proven to 8 nodes 🎃).
- Mesh refinement and coarsening, (adaptive mesh refinement 🎃).
- Full 3D electromagnetic field solver using the FDTD method, (Perfectly matched layers 🎃).
- Two fluid MHD model as well a Ideal MHD model.
- Particle in Cell (PIC) model for kinetic simulations, (Esirkepov current deposition 🎃).
- Scalable visualization via in-situ rendering image [Phantom Gaze](https://github.com/loliverhennigh/PhantomGaze). Currently working to bring in rendering kernels internal to Pumpkin Pulse.

# 🎃 Full Device First Principle Modeling 🎃

The core objective of Pumpkin Pulse is to allow for full device first principle modeling of a variety of fusion reactors.
This includes the ability to model the full device geometry, plasma dynamics (kinetic and or fluid models), and electromagnetic fields.
The goal is to allow for the simulation of the entire device from the plasma core to the vacuum vessel to capacitor coil discharge.
This is achieved by using a combination of the Finite Difference Time Domain (FDTD) method for the electromagnetic fields and the Particle in Cell (PIC) method or Finite Volume Method for kinetic or fluid descriptions of the plasma.
Performing Full Device modeling is typically seen as computationally prohibitive however by using GPU acceleration
and making use of the recent transition to more unified memory architectures,
simulations with billions of cells can be achieved on a single node.
Bellow is a table and of various machines and their corresponding cell counts and updates per second. These are estimates based on the current state of the code and memory capacity/bandwidth of the machines.

| Machine | Cell Count Two Fluid (Billions) | Two Fluid Cell Updates Per Second (Billions/s) | Cell Count PIC (Assuming 30 particles per cell) (Billions)| PIC Cell Updates Per Second (Billions/s)|
| --- | --- | --- | --- | --- |
| Gaming PC (1 RTX 5090), Pcie 5.0, 128 GB | 1.17 | 4.1 | 0.167 | 0.136 |
| [Digits](https://www.nvidia.com/en-us/project-digits/), 128 GB | 1.17 | Unkown (probably ~1-2) | 0.167 | Unkown (probably ~0.05) |
| [GH200](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/), 480 GB | 4.39 | 8.2 | 0.32 | 0.27 |
| [NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/), 30 TB | 274.7 | 1180.8 | 9.1 | 40.8 |



# ⚠️ WARNING⚠️ 

Currently Pumpkin Pulse is under going a major rewrite in the default `dev` branch.
This branch does not have all of the features mentioned above however in the `main` branch you can find the older much messier version of the code.
The `main` branch was basically a sketch of the core algorithms needed such as wall and collision modeling, perfect matched layers, particle sorting and current deposition, etc... You can see many of these present in the bellow gallery of videos.
I was planning on not making this rewrite public until it was ready, but here it is. Hope you find the work interesting!

# Installation

Main dependencies is [Warp](https://github.com/NVIDIA/warp) which is used to write core CUDA kernels. Pumpkin Pulse can be installed via following,

```
pip install .
```

# Gallery

Check out videos on my YouTube channel, [Oliver Henigh](https://www.youtube.com/@oliverhennigh451)

## Science!

### Field Reversed Configuration

[![Watch the video](http://img.youtube.com/vi/OGnGGQSjQHo/0.jpg)](https://www.youtube.com/watch?v=OGnGGQSjQHo)

###

## Scalability!

Working on more scalability things now! Stay tuned for a 

### 4 Billion cells on Gaming PC, (128 GB, 1 RTX 4090)

[![Watch the video](https://img.youtube.com/vi/gtStqHPDXeI/0.jpg)](https://www.youtube.com/watch?v=gtStqHPDXeI)

## 🎃 Spooky Fun 🎃!

### Imploding Pumpkins

[![Watch the video](http://img.youtube.com/vi/875d3_iFTWM/0.jpg)](https://www.youtube.com/watch?v=875d3_iFTWM)
### Exploding Bunnies

[![Watch the video](http://img.youtube.com/vi/FoYjATimtJo/0.jpg)](https://www.youtube.com/watch?v=FoYjATimtJo)

