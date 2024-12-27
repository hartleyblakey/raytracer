# CS 375 Final Project


## Continued Work
![screenshot](/Final-Project/screenshots/sponza_dof_hdri.png?raw=true)
Added depth of field, and CPU raycasting to find focal distance. 

Added support for an equirectangular environment map
 - No importance sampling yet

## Final State
![screenshot](/Final-Project/screenshots/sponza.png?raw=true)
- Loads simple static GLTF files into vectors of triangle positions, cameras, base color texcoords, and base color images converted to rgba8
  - Not well tested but worked on sponza and the tiny blender exports I tried

  - **The paths to the loaded gltf model files are hardcoded**

- Builds a simple BVH around the triangle positions and extra data, based on the first 2.5 articles of "Build A BVH" by Jacco Bikker

- Raytraces the bvh in a compute shader with simple diffuse lighting, and a hardcoded sun light
  - A small fixed number of directional and point lights are read from the file and passed to the shader but are not used

- Simple flycam with scroll wheel zoom, starts out the same as the first camera in the gltf file

- Technically can be built for the web
  - Hardcoded resolution to 512x512, no extra effort put in beyond getting it to run
  - [hosted on blue here (requires webgpu)](https://blue.cs.sonoma.edu/~hblakey/CS-375/Final-Project/generated/index.html)


#### Source Files

- `shader.wgsl`:
  - Implements the BVH traverasl and raytracing

- `scene.rs`:
  - Loads the GLTF files
  - Builds the BVH

- `main.rs`:
  - Handles the main event loop
  - Initializes the raytracer and wgpu resources

- `gpu.rs`:
  - Contians helper functions and structs for opening the window and working with wgpu

- `input.rs`:
  - Implements a simple camera controller


#### Build Instructions
Requires rust and cargo, [which can be found here](https://www.rust-lang.org/tools/install)

Use cargo to run  the project:
```bash
cd Final-Project
cargo run
```

For the web build:

Requires wasm-bindgen, which can be installed with cargo:
```bash
cargo install wasm-bindgen-cli
```
```bash
cd Final-Project

cargo build --target wasm32-unknown-unknown 

wasm-bindgen --out-dir generated --web target/wasm32-unknown-unknown/debug/raytracer.wasm
```

# Initial proposal

## Overview

For this project I will write a simple raytracer that can render a static model from a gltf file. I will make a simple bvh to accelerate ray traversal, and would like to support textures.

[The current state of the raytracer is hosted here (requires webgpu)](https://blue.cs.sonoma.edu/~hblakey/CS-375/Final-Project/generated/index.html)

## Schedule

- Week 1
  
  Load triangle positions from a gltf file, add basic camera movement

- Week 2

  Build a bvh to accelerate ray traversal

- Week 3

  investigate texture support, at least add texture coordinates

- Week 4

  polish and prepare presentation

## Technical Specification

I intend to use rust and the wgpu webgpu implementation to write the raytracer. I have written shadertoy raytracers before, and I already completed the webgpu base/boilerplate for this project, which uses a compute shader to raytrace a flat storage buffer of randomly-generated triangles. It just uses a ray-triangle intersection function I found online, I can re-write it myself if required.

