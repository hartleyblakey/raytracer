use std::{borrow::Cow, cell::RefCell, collections::HashMap, hash::Hash, num::NonZero, rc::Rc};
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
pub struct ResourceManager {
    bind_group_layouts: HashMap<BindGroupLayoutEntries, wgpu::BindGroupLayout>,
}
impl ResourceManager {
    pub fn new() -> Self {
        Self {
            bind_group_layouts: HashMap::new(),
        }
    }

    pub fn get_bind_group_layout(&self, layout_entries: &BindGroupLayoutEntries) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layouts.get(layout_entries)
    }

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

impl Buffer {
    pub fn view<'a>(&'a self, offset: u64, size: u64) -> BufferView<'a> {
        BufferView {
            buffer: self,
            offset,
            size,
            read_only: false
        }
    }

    pub fn view_all<'a>(&'a self) -> BufferView<'a> {
        BufferView {
            buffer: self,
            offset: 0,
            size: self.size,
            read_only: false
        }
    }

    pub fn view_read<'a>(&'a self, offset: u64, size: u64) -> BufferView<'a> {
        BufferView {
            buffer: self,
            offset,
            size,
            read_only: true
        }
    }
}

pub struct BufferView<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: u64,
    pub read_only: bool
}

impl<'a> BufferView<'a> {
    fn binding(&self) -> wgpu::BufferBinding<'a> {
        wgpu::BufferBinding {
            buffer: &self.buffer.raw,
            offset: self.offset,
            size: NonZero::<u64>::new(self.size),
        }
    }
}

pub struct BGBuilder<'a> {
    layout_entries: BindGroupLayoutEntries,
    entries:        Vec<wgpu::BindGroupEntry<'a>>,
    device:         &'a wgpu::Device,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct BindGroupLayoutEntries {
    entries: Vec<wgpu::BindGroupLayoutEntry>
}

pub struct BindGroup {
    pub raw: wgpu::BindGroup,
    pub entries: BindGroupLayoutEntries,
}


// struct BufferView {
//     ty: wgpu::BufferBindingType,
//     dynamic_offset_stride: Option<NonZero<u32>>,
//     offset: u32,
//     size: u32
// }

impl<'a> BGBuilder<'a> {
    pub fn with_buffer(&mut self, view: &'a BufferView, visibility: wgpu::ShaderStages) -> &mut Self{
        let ty : wgpu::BufferBindingType = match view.buffer.raw.usage() {
            _ if view.buffer.raw.usage().contains(wgpu::BufferUsages::UNIFORM)  => wgpu::BufferBindingType::Uniform,
            _ if view.buffer.raw.usage().contains(wgpu::BufferUsages::STORAGE)  => wgpu::BufferBindingType::Storage { read_only: false },
            _ => panic!("Invalid buffer usage: expected uniform or storage"),
        };

        let layout_entry = wgpu::BindGroupLayoutEntry {
            binding: self.layout_entries.entries.len() as u32,
            count: None,
            visibility,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }
        };

        self.layout_entries.entries.push(layout_entry);

        let entry = wgpu::BindGroupEntry {
            binding: self.entries.len() as u32,
            resource: wgpu::BindingResource::Buffer(view.binding()),
        };
        self.entries.push(entry);
        self
    }

    pub fn finish<'b>(&mut self, manager: &'b mut ResourceManager) -> BindGroup {
        if !manager.bind_group_layouts.contains_key(&self.layout_entries) {
            let layout_desc = wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: self.layout_entries.entries.as_slice(),
            };
            manager.bind_group_layouts.insert(self.layout_entries.clone(), self.device.create_bind_group_layout(&layout_desc));
        }

        let layout = manager.get_bind_group_layout(&self.layout_entries).expect("hash get failed after insertion");

        let desc = wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: self.entries.as_slice(),
        };

        BindGroup {
            raw: self.device.create_bind_group(&desc),
            entries: self.layout_entries.clone()
        }
    }
}


impl<'a> Gpu<'a> {
    pub fn new_bind_group(&'a self) -> BGBuilder<'a> {
        BGBuilder {
            device: &self.device,
            entries: Vec::new(),
            layout_entries: BindGroupLayoutEntries{entries: Vec::new()},
        }
    }

    pub fn new_pipeline_layout(&self, resources: &ResourceManager, bind_groups: &[&BindGroup]) -> wgpu::PipelineLayout {
        let layouts: Vec<&wgpu::BindGroupLayout> = bind_groups.iter().map(|bg| resources.get_bind_group_layout(&bg.entries).unwrap()).collect();
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });
        layout
    }

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
            window,
        }
    }
    fn run(user_event_loop: fn(Event<()>, &winit::event_loop::EventLoopWindowTarget<()>) -> ()) {

    }
}


