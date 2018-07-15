extern crate cgmath as cg;
extern crate ggez;
extern crate rand;

use cg::prelude::*;
use ggez::conf;
use ggez::event;
use ggez::graphics;
use ggez::timer;
use ggez::{Context, GameResult};
use rand::prelude::*;

type Point2f = cg::Point2<f64>;
type Vector2f = cg::Vector2<f64>;

const NEIGHBORS_RANGE: f64 = 50.0;
const NEIGHBORS_RANGE_SQ: f64 = NEIGHBORS_RANGE * NEIGHBORS_RANGE;
const MAX_SPEED: f64 = 72.0;
const MAX_SPEED_SQ: f64 = MAX_SPEED * MAX_SPEED;
const MAX_FORCE: f64 = 1.2;
const MAX_FORCE_SQ: f64 = MAX_FORCE * MAX_FORCE;
const DESIRED_SEPARATION: f64 = 18.0;
const DESIRED_SEPARATION_SQ: f64 = DESIRED_SEPARATION * DESIRED_SEPARATION;

#[derive(Debug)]
struct World {
    boids: Vec<Boid>,
}

impl World {
    fn new(ctx: &mut Context, boids_count: u32) -> GameResult<World> {
        let boids = (0..boids_count).map(|_| Boid::new(ctx).unwrap()).collect();
        Ok(World { boids })
    }

    fn update(&mut self, time: f64, w: f64, h: f64) {
        for i in 0..self.boids.len() {
            let (first_half, second_half_with_current) = self.boids.split_at_mut(i);
            let (current, second_half) = second_half_with_current.split_at_mut(1);
            current[0].update(time, first_half.iter().chain(second_half.iter()), w, h);
        }
    }
}

#[derive(Debug)]
struct Boid {
    position: Point2f,
    velocity: Vector2f,
    rotation: f64,
}

#[inline]
fn euclidian_modulo(a: f64, b: f64) -> f64 {
    (a % b + b) % b
}

#[inline]
fn toroidal_delta(a: f64, b: f64, r: f64, l: f64) -> f64 {
    if a < r && b > l - r {
        0.0 - (a + (l - b))
    } else if a > l - r && b < r {
        (l - a) + b
    } else {
        b - a
    }
}

#[inline]
fn limit_force(f: Vector2f) -> Vector2f {
    if f.magnitude2() > MAX_FORCE_SQ {
        f.normalize_to(MAX_FORCE)
    } else {
        f
    }
}

impl Boid {
    fn new(ctx: &mut Context) -> GameResult<Boid> {
        let position = Point2f::new(
            random::<f64>() * f64::from(ctx.conf.window_mode.width),
            random::<f64>() * f64::from(ctx.conf.window_mode.height),
        );
        Ok(Boid {
            position,
            velocity: Vector2f::new(
                thread_rng().gen_range(0.0 - MAX_SPEED, MAX_SPEED),
                thread_rng().gen_range(0.0 - MAX_SPEED, MAX_SPEED),
            ),
            rotation: 0.0,
        })
    }

    #[inline]
    fn update<'a, I>(&'a mut self, time: f64, others: I, w: f64, h: f64)
    where
        I: Iterator<Item = &'a Boid> + Clone,
    {
        let (x, y) = (self.position.x, self.position.y);
        let neighbors = others.filter(|b| {
            let delta_x = toroidal_delta(x, b.position.x, NEIGHBORS_RANGE, w);
            let delta_y = toroidal_delta(y, b.position.y, NEIGHBORS_RANGE, h);
            let distance_sq = (delta_x * delta_x) + (delta_y * delta_y);
            distance_sq <= NEIGHBORS_RANGE_SQ
        });

        let acceleration = self.flock(neighbors);
        self.velocity += acceleration;
        if self.velocity.magnitude2() > MAX_SPEED_SQ {
            self.velocity = self.velocity.normalize_to(MAX_SPEED);
        }
        self.position += self.velocity * time;
        self.position.x = euclidian_modulo(self.position.x, w);
        self.position.y = euclidian_modulo(self.position.y, h);
        if self.velocity.magnitude2() > 0.001 {
            let to = self.position + self.velocity;
            self.rotation = (to.y - self.position.y).atan2(to.x - self.position.x);
        }
    }

    fn flock<'a, I>(&self, boids: I) -> Vector2f
    where
        I: Iterator<Item = &'a Boid> + Clone,
    {
        let mut separate_count = 0;
        let mut separation_sum = Vector2f::zero();
        let mut neighbors_count = 0;
        let mut velocity_sum = Vector2f::zero();
        let mut position_sum = Vector2f::zero();

        for boid in boids {
            neighbors_count += 1;

            // Separation
            let distance_sq = self.position.distance2(boid.position);
            if distance_sq > 0.0 && distance_sq < DESIRED_SEPARATION_SQ {
                separate_count += 1;
                separation_sum += (self.position - boid.position).normalize() / distance_sq.sqrt();
            }

            // Velocity
            velocity_sum += boid.velocity;

            // Position
            position_sum += boid.position.to_vec();
        }

        let separate_force = if separate_count > 0 {
            let mean = separation_sum / f64::from(separate_count);
            if mean.magnitude2() > 0.0 {
                limit_force(mean.normalize_to(MAX_SPEED) - self.velocity)
            } else {
                Vector2f::zero()
            }
        } else {
            Vector2f::zero()
        };

        let (align_force, cohere_force) = if neighbors_count > 0 {
            let neighbors_count_f64 = f64::from(neighbors_count);
            let velocity_mean = velocity_sum / neighbors_count_f64;
            let position_mean = position_sum / neighbors_count_f64;

            (
                if velocity_mean.magnitude2() > 0.0 {
                    limit_force(velocity_mean.normalize_to(MAX_SPEED) - self.velocity)
                } else {
                    Vector2f::zero()
                },
                self.steer_to(Point2f::from_vec(position_mean)),
            )
        } else {
            (Vector2f::zero(), Vector2f::zero())
        };

        (separate_force * 1.0) + (align_force * 1.0) + (cohere_force * 1.0)
    }

    fn steer_to(&self, target: Point2f) -> Vector2f {
        let mut desired = target - self.position;
        let desired_mag_sq = desired.magnitude2();
        if desired_mag_sq > 0.0001 {
            let desired_mag = desired_mag_sq.sqrt();
            desired = if desired_mag < 100.0 {
                desired.normalize_to(MAX_SPEED * (desired_mag / 100.0))
            } else {
                desired.normalize_to(MAX_SPEED)
            };

            limit_force(desired - self.velocity)
        } else {
            Vector2f::zero()
        }
    }
}

struct MainState {
    frames: usize,
    world: World,
    boid_mesh: graphics::Mesh,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<MainState> {
        use graphics::Point2;
        let world = World::new(ctx, 3000)?;
        let s = MainState {
            frames: 0,
            world,
            boid_mesh: graphics::MeshBuilder::new()
                .polygon(
                    graphics::DrawMode::Fill,
                    &[
                        Point2::new(-3.0, -2.0),
                        Point2::new(5.0, 0.0),
                        Point2::new(-3.0, 2.0),
                    ],
                )
                .build(ctx)?,
        };

        graphics::set_color(ctx, graphics::Color::new(0.0, 0.0, 0.0, 1.0))?;
        graphics::set_background_color(ctx, graphics::Color::new(1.0, 1.0, 1.0, 1.0));

        Ok(s)
    }
}

impl event::EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        let delta = timer::get_delta(ctx);
        self.world.update(
            delta.as_secs() as f64 + f64::from(delta.subsec_nanos()) / 1_000_000_000.0,
            f64::from(ctx.conf.window_mode.width),
            f64::from(ctx.conf.window_mode.height),
        );
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx);
        for boid in &self.world.boids {
            graphics::draw(
                ctx,
                &self.boid_mesh,
                graphics::Point2::new(boid.position.x as f32, boid.position.y as f32),
                boid.rotation as f32,
            )?;
        }
        graphics::present(ctx);

        self.frames += 1;
        if (self.frames % 100) == 0 {
            println!("FPS: {}", ggez::timer::get_fps(ctx));
        }

        Ok(())
    }
}

pub fn main() {
    let mut c = conf::Conf::new();
    c.window_mode.width = 1920;
    c.window_mode.height = 1080;
    let ctx = &mut Context::load_from_conf("helloworld", "ggez", c).unwrap();

    let state = &mut MainState::new(ctx).unwrap();
    if let Err(e) = event::run(ctx, state) {
        println!("Error encountered: {}", e);
    } else {
        println!("Game exited cleanly.");
    }
}
