use std::{borrow::Cow};
use bytemuck::bytes_of;
use pollster::FutureExt;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

pub fn new_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    let mut builder = winit::window::WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    builder.build(&event_loop).unwrap()
}

pub struct Gpu<'a> {
    pub adapter: wgpu::Adapter, 
    pub device:  wgpu::Device, 
    pub queue:   wgpu::Queue, 
    pub surface: wgpu::Surface<'a>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window:  &'a Window,
}

pub struct Buffer {
    pub raw: wgpu::Buffer,
    pub usage: wgpu::BufferUsages,
    pub size: u64,
}

struct BufferView<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: u64,
    pub read_only: bool
}

impl<'a> Gpu<'a> {
    pub fn new_uniform_buffer<T: bytemuck::Pod>(&self, val: &T) -> Buffer {
        let size = size_of::<T>() as u64;
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let desc = wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size,
            usage
        };
        let buffer = self.device.create_buffer(&desc);
        self.queue.write_buffer(&buffer, 0, bytes_of(val));
        Buffer {
            raw: buffer,
            size,
            usage
        }
    }

    pub fn new_storage_buffer(&self, size: u64) -> Buffer {
        let usage = 
            wgpu::BufferUsages::STORAGE 
            | wgpu::BufferUsages::COPY_DST 
            | wgpu::BufferUsages::COPY_SRC;

        let desc = wgpu::BufferDescriptor {
            label: None,
            mapped_at_creation: false,
            size,
            usage
        };

        Buffer {
            raw: self.device.create_buffer(&desc),
            size,
            usage
        }
    }
    pub async fn new(window: &'a Window) -> Self {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let device_desc = wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
            required_limits: wgpu::Limits::default()
                .using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        };

        // Create the logical device and command queue
        let (device, queue) = adapter.request_device(&device_desc, None)
            .await
            .expect("Failed to create device");

        let mut surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();

        surface.configure(&device, &surface_config);

        Self {
            adapter,
            device,
            queue,
            surface,
            surface_config,
            window
        }
    }

    fn run(user_event_loop: fn(Event<()>, &winit::event_loop::EventLoopWindowTarget<()>) -> ()) {

    }
}


