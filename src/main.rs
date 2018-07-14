extern crate ggez;
extern crate nalgebra as na;

use ggez::conf;
use ggez::event;
use ggez::graphics;
use ggez::timer;
use ggez::{Context, GameResult};

type Point2f = na::Point2<f64>;
type Vector2f = na::Vector2<f64>;

const NEIGHBORS_RANGE: f64 = 50.0;
const NEIGHBORS_RANGE_SQ: f64 = NEIGHBORS_RANGE * NEIGHBORS_RANGE;
const MAX_SPEED: f64 = 72.0;
const MAX_FORCE: f64 = 1.2;
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
    mesh: graphics::Mesh,
}

fn euclidian_modulo(a: f64, b: f64) -> f64 {
    (a % b + b) % b
}

fn toroidal_delta(a: f64, b: f64, r: f64, l: f64) -> f64 {
    if a < r && b > l - r {
        0.0 - (a + (l - b))
    } else if a > l - r && b < r {
        (l - a) + b
    } else {
        b - a
    }
}

fn limit_force(f: Vector2f) -> Vector2f {
    if f.norm() > MAX_FORCE {
        f.normalize() * MAX_FORCE
    } else {
        f
    }
}

impl Boid {
    fn new(ctx: &mut Context) -> GameResult<Boid> {
        use graphics::Point2;
        let position =
            Point2f::from_coordinates(Vector2f::new_random().component_mul(&Vector2f::new(
                ctx.conf.window_mode.width as f64,
                ctx.conf.window_mode.height as f64,
            )));
        Ok(Boid {
            position,
            velocity: (Vector2f::new_random() * MAX_SPEED) - (Vector2f::new_random() * MAX_SPEED),
            rotation: 0.0,
            mesh: graphics::MeshBuilder::new()
                .polygon(
                    graphics::DrawMode::Fill,
                    &[
                        Point2::new(-3.0, -2.0),
                        Point2::new(5.0, 0.0),
                        Point2::new(-3.0, 2.0),
                    ],
                )
                .build(ctx)?,
        })
    }

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
        if self.velocity.norm() > MAX_SPEED {
            self.velocity = na::normalize(&self.velocity) * MAX_SPEED
        }
        self.position += self.velocity * time;
        self.position.x = euclidian_modulo(self.position.x, w);
        self.position.y = euclidian_modulo(self.position.y, h);
        if self.velocity.norm() > 0.001 {
            let to = self.position + self.velocity;
            self.rotation = (to.y - self.position.y).atan2(to.x - self.position.x);
        }
    }

    fn flock<'a, I>(&self, boids: I) -> Vector2f
    where
        I: Iterator<Item = &'a Boid> + Clone,
    {
        (self.separate(boids.clone()) * 1.0)
            + (self.align(boids.clone()) * 1.0)
            + (self.cohere(boids.clone()) * 1.0)
    }

    fn separate<'a, I>(&self, boids: I) -> Vector2f
    where
        I: Iterator<Item = &'a Boid>,
    {
        let mut count = 0;
        let sum = boids
            .filter_map(|b| {
                let distance = na::distance_squared(&self.position, &b.position);
                if distance > 0.0 && distance < DESIRED_SEPARATION_SQ {
                    count = count + 1;
                    Some((self.position - b.position).normalize() / distance)
                } else {
                    None
                }
            })
            .sum::<Vector2f>();
        if count > 0 {
            let mean = sum / count as f64;
            if mean.norm() > 0.0 {
                limit_force((mean.normalize() * MAX_SPEED) - self.velocity)
            } else {
                Vector2f::zeros()
            }
        } else {
            Vector2f::zeros()
        }
    }

    fn align<'a, I>(&self, boids: I) -> Vector2f
    where
        I: Iterator<Item = &'a Boid>,
    {
        let mut count = 0;
        let sum = boids
            .map(|b| {
                count = count + 1;
                b.velocity
            })
            .sum::<Vector2f>();

        if count > 0 {
            let mean = sum / count as f64;
            if mean.norm() > 0.0 {
                limit_force((mean.normalize() * MAX_SPEED) - self.velocity)
            } else {
                Vector2f::zeros()
            }
        } else {
            Vector2f::zeros()
        }
    }

    fn cohere<'a, I>(&self, boids: I) -> Vector2f
    where
        I: Iterator<Item = &'a Boid>,
    {
        let mut count = 0;
        let sum = boids
            .map(|b| {
                count = count + 1;
                b.position.coords
            })
            .sum::<Vector2f>();

        if count > 0 {
            self.steer_to(Point2f::from_coordinates(sum / count as f64))
        } else {
            Vector2f::zeros()
        }
    }

    fn steer_to(&self, target: Point2f) -> Vector2f {
        let mut desired = target - self.position;
        if let Some(d) = desired.try_normalize_mut(0.0001) {
            desired = if d < 100.0 {
                desired * (MAX_SPEED * (d / 100.0))
            } else {
                desired * MAX_SPEED
            };

            limit_force(desired - self.velocity)
        } else {
            Vector2f::zeros()
        }
    }
}

struct MainState {
    frames: usize,
    world: World,
}

impl MainState {
    fn new(ctx: &mut Context) -> GameResult<MainState> {
        let world = World::new(ctx, 1000)?;
        let s = MainState { frames: 0, world };
        Ok(s)
    }
}

impl event::EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        let delta = timer::get_delta(ctx);
        self.world.update(
            delta.as_secs() as f64 + delta.subsec_nanos() as f64 / 1_000_000_000.0,
            ctx.conf.window_mode.width as f64,
            ctx.conf.window_mode.height as f64,
        );
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx);
        graphics::set_color(ctx, graphics::Color::new(0.0, 0.0, 0.0, 1.0))?;
        graphics::set_background_color(ctx, graphics::Color::new(1.0, 1.0, 1.0, 1.0));
        for boid in &self.world.boids {
            graphics::draw(
                ctx,
                &boid.mesh,
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
