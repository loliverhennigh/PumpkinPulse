# 🎃 Pumpkin Pulse 🎃

Pumpkin Pulse is a python package for scalable GPU accelerated simulations of plasma. The objective is to allow for full device, first principle modeling of a variety of fusion reactors such as Field Reversed Configuration, Z-Pinch, and Inertial Confinement Fusion machines. Key features include,

- GPU acceleration using NVIDIA [Warp](https://github.com/NVIDIA/warp), (core kernels achieving 95% memory bandwidth utilization🎃).
- Scalable to multiple GPUs and nodes using MPI, (linear weak scaling proven up to 8 nodes 🎃).
- Full 3D electromagnetic field solver using the FDTD method, (Perfectly matched layers 🎃).
- Two fluid MHD model as well a Ideal MHD model.
- Particle in Cell (PIC) model for kinetic simulations, (Esirkepov current deposition 🎃).
- Scalable visualization via in-situ rendering image [Phantom Gaze](https://github.com/loliverhennigh/PhantomGaze). Currently 

<div align="center">
  <img src="https://github.com/loliverhennigh/PumpkinPulse/blob/dev/assets/cover_image.png">
  <p><strong>🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃 🎃</strong></p>
</div>

# ⚠️ WARNING⚠️ 

If you are seeing this its probably because you work at Helion Energy and I put this in my resume.
Currently Pumpkin Pulse is under going a major rewrite in the default `dev` branch.
This branch does not have all of the features mentioned above however in the `main` branch you can find the older much messier version of the code.
I was planning on not making this rewrite public until it was ready, but here it is. Hope you find the work interesting!

# Installation

Main dependencies is [Warp](https://github.com/NVIDIA/warp) which is used to write core CUDA kernels. Pumpkin Pulse can be installed via following,

```
pip install .
```

# Architecture

Pumpkin Pu

# Gallery

Check out videos on my YouTube channel, [Oliver Henigh](https://www.youtube.com/@oliverhennigh451)

## Science!

###

## Scalability!

### 4 Billion cells on Gaming PC, (128 GB, 1 RTX 4090)

[![Watch the video](https://www.youtube.com/shorts/gtStqHPDXeI)

## Fun!

### Exploding Bunnies

[![Watch the video](http://img.youtube.com/vi/FoYjATimtJo/0.jpg)](https://www.youtube.com/watch?v=FoYjATimtJo)

### Imploding Pumpkins

[![Watch the video](http://img.youtube.com/vi/875d3_iFTWM/0.jpg)](https://www.youtube.com/watch?v=875d3_iFTWM)
