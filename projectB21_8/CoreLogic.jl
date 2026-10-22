# ============================================================================
# CoreLogic.jl - הגדרות בסיס, לוגיקה ואלגוריתמי הניווט של הרובוטים
# ============================================================================

using LinearAlgebra
using Random

# ======================== Configuration ========================
const GRID_WIDTH = 20.0
const GRID_HEIGHT = 20.0
const STEP_SIZE = 0.3
const COLLISION_RADIUS = 0.3
const TOWER_REACH_DIST = 0.5
const ROBOT_SIZE = 0.7
const SENSOR_RANGE = COLLISION_RADIUS * 1.1

# ── Enemy (Adversary) Tower Configuration ──
const ENEMY_DUTY_CYCLE = 1 / 4
const ENEMY_STRENGTH = 1.0
const ENEMY_CYCLE_BASE = 40

const ANOMALY_THRESHOLD = 2.0
const ENEMY_CONFIRM_STEPS = 3

# ======================== Data Structures ========================
mutable struct Vec2
    x::Float64
    y::Float64
end

struct Obstacle
    xmin::Float64
    ymin::Float64
    xmax::Float64
    ymax::Float64
end

struct SimulationEnv
    width::Float64
    height::Float64
    tower::Vec2
    enemy_tower::Vec2
    obstacles::Vector{Obstacle}
end

mutable struct Robot
    pos::Vec2
    angle::Float64
    state::Symbol
    hit_intensity::Float64
end

mutable struct FollowState
    i_H::Float64
    local_max_i::Float64
    steps_since_max::Int
    active::Bool
end

struct TrajectoryPoint
    x::Float64
    y::Float64
    step::Int
    state::Symbol
    angle::Float64
    intensity_val::Float64
    front_sensor::Bool
    left_sensor::Bool
    enemy_active::Bool
    high_intensity_memory::Float64
    anti_enemy_mode::Symbol
    reached::Bool
    confirm_counter::Int
    action::Symbol
end

mutable struct RobotState
    robot::Robot
    strategy::Symbol
    follow::FollowState
    high_intensity_memory::Float64
    prev_perceived::Float64
    anti_enemy::Symbol
    frozen_angle::Float64
    frozen_state::Symbol
    confirm_counter::Int
    reached::Bool
    reached_step::Int
    needs_orientation::Bool
end

# ======================== Utility Functions ========================
function distance(a::Vec2, b::Vec2)
    sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
end

function normalize_angle(a::Float64)
    while a < 0
        a += 2π
    end
    while a >= 2π
        a -= 2π
    end
    return a
end

function intensity(pos::Vec2, tower::Vec2)
    d = distance(pos, tower)
    return d < 0.01 ? Inf : 1.0 / d
end

function is_enemy_active(step::Int)
    if step < 20
        return false
    end
    active_steps = round(Int, ENEMY_DUTY_CYCLE * ENEMY_CYCLE_BASE)
    return ((step - 20) % ENEMY_CYCLE_BASE) < active_steps
end

function perceived_intensity(pos::Vec2, env::SimulationEnv, step::Int)
    target_signal = intensity(pos, env.tower)
    enemy_signal = 0.0
    if is_enemy_active(step)
        enemy_signal = intensity(pos, env.enemy_tower) * ENEMY_STRENGTH
    end
    return target_signal + enemy_signal
end

# ======================== Collision & Sensors ========================
function collides_with_obstacle(pos::Vec2, env::SimulationEnv)
    for obs in env.obstacles
        cx = clamp(pos.x, obs.xmin, obs.xmax)
        cy = clamp(pos.y, obs.ymin, obs.ymax)
        dx = pos.x - cx
        dy = pos.y - cy
        if dx * dx + dy * dy < COLLISION_RADIUS^2
            return true
        end
    end
    return false
end

function collides_with_boundary(pos::Vec2, env::SimulationEnv)
    pos.x < COLLISION_RADIUS || pos.x > env.width - COLLISION_RADIUS ||
        pos.y < COLLISION_RADIUS || pos.y > env.height - COLLISION_RADIUS
end

function is_valid_pos(pos::Vec2, env::SimulationEnv)
    !collides_with_obstacle(pos, env) && !collides_with_boundary(pos, env)
end

function check_front_sensor(robot::Robot, env::SimulationEnv)
    px = robot.pos.x + SENSOR_RANGE * cos(robot.angle)
    py = robot.pos.y + SENSOR_RANGE * sin(robot.angle)
    return !is_valid_pos(Vec2(px, py), env)
end

function check_left_sensor(robot::Robot, env::SimulationEnv)
    la = robot.angle + π / 2
    px = robot.pos.x + SENSOR_RANGE * cos(la)
    py = robot.pos.y + SENSOR_RANGE * sin(la)
    return !is_valid_pos(Vec2(px, py), env)
end

# ======================== Algorithm Primitives ========================
function u_ori!(robot::Robot, env::SimulationEnv, step::Int=0)
    best_angle = robot.angle
    max_intensity_found = -Inf
    for a in 0:5:359
        rad = deg2rad(a)
        probe_pos = Vec2(robot.pos.x + 0.05 * cos(rad), robot.pos.y + 0.05 * sin(rad))
        i_val = perceived_intensity(probe_pos, env, step)
        if i_val > max_intensity_found
            max_intensity_found = i_val
            best_angle = rad
        end
    end
    robot.angle = best_angle
end

function u_fwd!(robot::Robot, env::SimulationEnv)
    nx = robot.pos.x + STEP_SIZE * cos(robot.angle)
    ny = robot.pos.y + STEP_SIZE * sin(robot.angle)
    new_pos = Vec2(nx, ny)
    if is_valid_pos(new_pos, env)
        robot.pos = new_pos
        return true
    end
    return false
end

function u_fol!(robot::Robot, env::SimulationEnv)
    start_angle = robot.angle + (π / 2) + deg2rad(15)
    for i in 0:72
        test_angle = normalize_angle(start_angle - i * deg2rad(5))
        test_pos = Vec2(robot.pos.x + STEP_SIZE * cos(test_angle), robot.pos.y + STEP_SIZE * sin(test_angle))
        if is_valid_pos(test_pos, env)
            robot.angle = test_angle
            robot.pos = test_pos
            return
        end
    end
end

# ======================== Anomaly Detection ========================
function max_natural_intensity_change(prev_intensity::Float64)
    if prev_intensity < 0.001
        return 0.01
    end
    d = 1.0 / prev_intensity
    if d <= STEP_SIZE * 1.5
        return 1.5
    end
    val = STEP_SIZE / (d * (d - STEP_SIZE))
    return min(val, 1.5)
end

function is_spike_anomaly(cur_i::Float64, prev_i::Float64)
    delta = cur_i - prev_i
    delta > 0 || return false
    return delta > ANOMALY_THRESHOLD * max_natural_intensity_change(prev_i)
end

function is_drop_anomaly(cur_i::Float64, prev_i::Float64)
    delta = prev_i - cur_i
    delta > 0 || return false
    if prev_i < 0.001
        return false
    end
    # הורדת סף הירידה החדה ל-15% או שינוי שגדול משינוי טבעי בצעד בודד
    is_massive_drop = (delta / prev_i) > 0.15
    return is_massive_drop || (delta > 1.2 * max_natural_intensity_change(cur_i))
end

# ======================== Robot Step Logic ========================
function do_normal_step!(rs::RobotState, env::SimulationEnv, step::Int)
    robot = rs.robot
    if robot.state == :MoveToTower
        if rs.needs_orientation
            u_ori!(robot, env, step)
            rs.needs_orientation = false
        end
        if !u_fwd!(robot, env)
            i_H = perceived_intensity(robot.pos, env, step)
            robot.hit_intensity = i_H
            robot.state = :FollowObstacle
            robot.angle = normalize_angle(robot.angle - π / 2)
            rs.follow = FollowState(i_H, i_H, 0, true)
            rs.needs_orientation = true
        end
    elseif robot.state == :FollowObstacle
        u_fol!(robot, env)
        cur_i = perceived_intensity(robot.pos, env, step)
        if cur_i > rs.follow.local_max_i
            rs.follow.local_max_i = cur_i
            rs.follow.steps_since_max = 0
        else
            rs.follow.steps_since_max += 1
        end
        has_passed_hit_intensity = (rs.follow.local_max_i > rs.follow.i_H)
        is_descending_from_peak = (rs.follow.steps_since_max >= 3)
        if has_passed_hit_intensity && is_descending_from_peak
            robot.state = :MoveToTower
            rs.follow.active = false
            rs.needs_orientation = true
        end
    end
end

function update_cautious_step!(rs::RobotState, env::SimulationEnv, step::Int, cur_perceived::Float64)
    if rs.anti_enemy == :Normal
        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_enemy = :Paused
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            rs.confirm_counter = 0
            return
        end
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)
    elseif rs.anti_enemy == :Paused
        if cur_perceived < rs.prev_perceived - 0.005
            rs.confirm_counter = 1
        elseif is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.confirm_counter = 0
        elseif rs.confirm_counter > 0
            rs.confirm_counter += 1
            if rs.confirm_counter >= ENEMY_CONFIRM_STEPS
                rs.anti_enemy = :Normal
                rs.confirm_counter = 0
                rs.robot.state = rs.frozen_state
                rs.robot.angle = rs.frozen_angle
                if rs.frozen_state == :MoveToTower
                    rs.needs_orientation = true
                else
                    rs.follow.local_max_i = perceived_intensity(rs.robot.pos, env, step)
                    rs.follow.steps_since_max = 0
                end
                rs.high_intensity_memory = perceived_intensity(rs.robot.pos, env, step)
            end
        end
    end
end

function update_inertial_step!(rs::RobotState, env::SimulationEnv, step::Int, cur_perceived::Float64)
    if rs.anti_enemy == :Normal
        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_enemy = :InertialMode
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            if rs.frozen_state == :MoveToTower
                rs.robot.angle = rs.frozen_angle
                u_fwd!(rs.robot, env)
            else
                u_fol!(rs.robot, env)
            end
            return
        end
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)
    elseif rs.anti_enemy == :InertialMode
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_enemy = :Normal
            rs.robot.state = rs.frozen_state
            rs.high_intensity_memory = perceived_intensity(rs.robot.pos, env, step)
            if rs.frozen_state == :MoveToTower
                rs.needs_orientation = true
            else
                rs.follow.local_max_i = perceived_intensity(rs.robot.pos, env, step)
                rs.follow.steps_since_max = 0
            end
            do_normal_step!(rs, env, step)
            return
        end
        if rs.frozen_state == :MoveToTower
            rs.robot.angle = rs.frozen_angle
            u_fwd!(rs.robot, env)
        else
            u_fol!(rs.robot, env)
        end
    end
end

function update_robot_step!(rs::RobotState, env::SimulationEnv, step::Int)
    rs.reached && return
    if distance(rs.robot.pos, env.tower) < TOWER_REACH_DIST
        rs.reached = true
        rs.reached_step = step
        return
    end
    cur_perceived = perceived_intensity(rs.robot.pos, env, step)

    if rs.strategy == :Naive
        if cur_perceived > rs.high_intensity_memory
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)
    elseif rs.strategy == :Cautious
        update_cautious_step!(rs, env, step, cur_perceived)
    elseif rs.strategy == :Inertial
        update_inertial_step!(rs, env, step, cur_perceived)
    end

    if rs.strategy != :Naive && rs.anti_enemy == :Normal
        post_i = perceived_intensity(rs.robot.pos, env, step)
        if post_i > rs.high_intensity_memory
            rs.high_intensity_memory = post_i
        end
    end
    rs.prev_perceived = perceived_intensity(rs.robot.pos, env, step)
end

function make_traj_point(rs::RobotState, env::SimulationEnv, step::Int)
    # חישוב הפעולה החד-משמעית בזמן הריצה עצמה!
    current_action = :None
    if rs.reached
        current_action = :Arrived
    elseif rs.anti_enemy == :Paused
        current_action = :Stopped
    elseif rs.robot.state == :MoveToTower
        # פה אנחנו יודעים בדיוק אם הוא הסתובב או התקדם
        current_action = rs.needs_orientation ? :u_ori : :u_fwd
    else
        current_action = :u_fol
    end

    TrajectoryPoint(
        rs.robot.pos.x, rs.robot.pos.y, step,
        rs.robot.state, rs.robot.angle,
        perceived_intensity(rs.robot.pos, env, step),
        check_front_sensor(rs.robot, env),
        check_left_sensor(rs.robot, env),
        is_enemy_active(step),
        rs.high_intensity_memory,
        rs.anti_enemy,
        rs.reached,
        rs.confirm_counter,
        current_action # <--- שמירת הפעולה המדויקת באובייקט
    )
end

function generate_random_env()
    tower = Vec2(18.0, 18.0)
    robot_start = Vec2(2.0, 1.0)
    obstacles = Obstacle[]
    for _ in 1:rand(3:9)
        w = rand() * 3.0 + 1.5
        h = rand() * 3.0 + 1.5
        x = rand() * 10.0 + 4.0
        y = rand() * 10.0 + 4.0
        push!(obstacles, Obstacle(x, y, x + w, y + h))
    end
    enemy_pos = Vec2(0.0, 0.0)
    temp_env = SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, Vec2(0.0, 0.0), obstacles)
    while true
        cand = Vec2(rand() * (GRID_WIDTH - 2 * COLLISION_RADIUS) + COLLISION_RADIUS,
            rand() * (GRID_HEIGHT - 2 * COLLISION_RADIUS) + COLLISION_RADIUS)
        if is_valid_pos(cand, temp_env) && distance(cand, robot_start) > 6.0 && distance(cand, tower) > 8.0
            enemy_pos = cand
            break
        end
    end
    return SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, enemy_pos, obstacles)
end