# WebGPU GLTF Raytracer

![screenshot](/screenshots/helmet.png?raw=true)

### Features

- Uses the gltf crate to load opaque static GLTF files into vectors of triangle positions, cameras, texcoords, and textures converted to rgba8

  - Only supports indexed triangles

  - Not well tested but worked on Sponza and the tiny blender exports I tried
  - Open a file by dragging it into the window, or by pressing "o" and selecting the file

- Builds a simple BVH around the triangle positions and extra data, based on the first 2.5 articles of "Build A BVH" by Jacco Bikker

- Raytraces the bvh in a compute shader
  - Basic path tracing with HDRI environment
  - Khronos PBR Neutral Tone Mapper for easy comparison with other renderers
  - Depth of field with hardcoded aperture size
  - No texture samplers, all textures are nearest filtered
  - Two UV sets supported per GLTF primitive

- Simple first person flycam
  - WASD movement, scroll wheel zoom, LMB focus
  - Starts out the same as the first camera in the gltf file

- Technically can be built for the web
  - Hardcoded resolution to 512x512, no extra effort put in beyond getting it to run
  - [hosted here (requires webgpu)](https://hartleyblakey.github.io/raytracer)


#### Source Files

- `shader.wgsl`:
  - Implements the BVH traversal and raytracing

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

- `index.html`:
  - Runs the compiled wasm app through a wasm-bindgen generated js interface


#### Build Instructions
Requires rust and cargo, [which can be found here](https://www.rust-lang.org/tools/install)

Use cargo to run  the project:
```bash
cd raytracer
cargo run
```

For the web build:

Requires wasm-bindgen, which can be installed with cargo:
```bash
cargo install wasm-bindgen-cli
```
Once cargo is installed:
```bash
cd raytracer

cargo build --target wasm32-unknown-unknown --release

wasm-bindgen --out-dir generated --web target/wasm32-unknown-unknown/release/raytracer.wasm
```
`index.html` looks for the generated files in `raytracer/site/build` by default

If you have [just](https://github.com/casey/just), you can run `just web` from anywhere in `raytracer/` to run these commands.