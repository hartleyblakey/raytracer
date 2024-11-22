# CS 375 Final Project

## Overview

For this project I will write a simple raytracer that can render a static model from a gltf file. I will make a simple bvh to accelerate ray traversal, and would like to support textures.

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

I will use rust and the wgpu webgpu implementation to write the raytracer. I have written shadertoy raytracers before, and I already completed the webgpu base/boilerplate for this project, which uses a compute shader to raytrace a flat storage buffer of randomly-generated triangles. It just uses a ray-triangle intersection function I found online, I can re-write it myself if required.

