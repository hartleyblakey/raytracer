use std::collections::HashSet;

use glam::{vec3, Mat3, Mat4, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

pub const FORWARD: Vec3 = vec3(1.0, 0.0, 0.0);
pub const UP: Vec3 = vec3(0.0, 0.0, 1.0);
pub const RIGHT: Vec3 = vec3(0.0, -1.0, 0.0);
pub const YAW_PITCH_ROLL: glam::EulerRot = glam::EulerRot::ZYX;

pub struct InputState {
    pub keys: HashSet<PhysicalKey>,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub lmb: bool,
    pub rmb: bool,
    pub scroll: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
    dir:    Vec3,
    fovy:   f32,
    origin: Vec3,
    focus:  f32,
}

#[derive(Copy, Clone)]
pub struct Camera {
    pitch: f32,
    yaw: f32,
    roll: f32,
    position: Vec3,
    focus: f32,
    speed: f32,
    fovy: f32,
    aspect: Option<f32>,
    moved: bool,
    lmb_last: bool,
    rmb_last: bool,
}

impl Camera {

    pub fn forward(&self) -> Vec3 {
        self.rot().mul_vec3(FORWARD).normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.rot().mul_vec3(UP)
    }

    pub fn right(&self) -> Vec3 {
        self.rot().mul_vec3(RIGHT)
    }

    pub fn rot(&self) -> Mat3{
        Mat3::from_euler(YAW_PITCH_ROLL, self.yaw, -self.pitch, self.roll)
    }

    pub fn to_gpu(&self) -> GpuCamera {
        GpuCamera {
            dir: self.forward(),
            fovy: self.fovy,
            origin: self.position,
            focus: self.focus
        }
    }
    
    pub fn check_moved(&mut self) -> bool {
        let tmp = self.moved;
        self.moved = false;
        tmp
    }

    pub fn rotate(&mut self, pitch: f32, yaw: f32) {
        self.moved = true;
        self.pitch += pitch;
        self.yaw += yaw;
        self.pitch = self.pitch.clamp(-f32::to_radians(80.0), f32::to_radians(80.0));
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.moved = true;
        self.position += delta;
    }

    pub fn zoom(&mut self, delta: f32) {
        self.moved = true;
        self.fovy += delta;
        self.fovy = self.fovy.clamp(f32::to_radians(1.0), f32::to_radians(179.0));
    }

    pub fn set_pos(&mut self, position: Vec3) {
        self.moved = true;
        self.position = position;
    }

    pub fn update(&mut self, input: &mut InputState, dt: f32) {
        use KeyCode::*;
        let s = if input.keys.contains(&PhysicalKey::Code(ControlLeft)) {
            0.1
        } else {
            1.0
        };
        for key in input.keys.iter() {
            if let PhysicalKey::Code(code) = key {
                let forward = self.forward();
                let forward = vec3(forward.x, forward.y, 0.0).normalize();
                match code {
                    KeyW =>      self.translate( forward      * self.speed * dt * s),
                    KeyS =>      self.translate(-forward      * self.speed * dt * s),
                    KeyA =>      self.translate(-self.right() * self.speed * dt * s),
                    KeyD =>      self.translate( self.right() * self.speed * dt * s),
                    ShiftLeft => self.translate(-UP           * self.speed * dt * s),
                    Space =>     self.translate( UP           * self.speed * dt * s),
                    _ => ()
                }
            }
        }

        // cancel camera rotation on first frame
        if input.rmb && !self.rmb_last {
            input.mouse_x = 0.0;
            input.mouse_y = 0.0;
        }

        if input.rmb && (input.mouse_x != 0.0 || input.mouse_y != 0.0) {
            self.rotate(-input.mouse_y as f32 * 0.003, -input.mouse_x as f32 * 0.003);
            
            input.mouse_x = 0.0;
            input.mouse_y = 0.0;
        }

        if input.scroll != 0.0 {
            self.zoom((input.scroll * 0.1) as f32);
            input.scroll = 0.0;
        }

        self.rmb_last = input.rmb;
        self.lmb_last = input.lmb;
    }

    pub fn default() -> Camera {
        Camera {
            pitch: f32::to_radians(-45.0),
            yaw: f32::to_radians(0.0),
            roll: 0.0,
            position: vec3(0.0, 0.0, 0.0) - FORWARD * 5.0 + UP * 5.0,
            focus: 1.0,
            speed: 10.0,
            fovy: f32::to_radians(90.0),
            aspect: None,
            moved: false,
            lmb_last: true,
            rmb_last: true,
        }
    }

    pub fn from_gltf(gltf: gltf::Camera, transform: &Mat4) -> Camera {
        // let dir = transform.to_scale_rotation_translation().1.mul_vec3(vec3(0.0, 0.0, 1.0));
        let origin = transform.transform_point3(vec3(0.0, 0.0, 0.0));
        let (_, rot, _) = transform.to_scale_rotation_translation();
        let (yaw, pitch, roll) = rot.to_euler(YAW_PITCH_ROLL);

        let (fovy, aspect) = match gltf.projection() {
            gltf::camera::Projection::Orthographic(orthographic) => {
                panic!("Orthographic cameras are not supported");
            },
            gltf::camera::Projection::Perspective(perspective) => {
                (perspective.yfov(), perspective.aspect_ratio())
            },
        };
        
        let focus = 1.0;
        let mut camera = Camera::default();
        camera.position = origin;
        camera.pitch = pitch;
        camera.yaw = yaw;
        camera.roll = roll;
        camera.focus = focus;
        camera.speed = 5.0;
        camera.fovy = fovy;
        camera.aspect = aspect;

        camera
    }
}
