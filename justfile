# list available commands
list: 
    just --list

# compile to wasm and generate js bindings
web:
    cargo build --target wasm32-unknown-unknown --release
    wasm-bindgen --out-dir generated --web target/wasm32-unknown-unknown/release/raytracer.wasm

# compile for just native
native:
    cargo build --release

# compile for both web and native
build: web native