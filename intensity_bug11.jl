# ============================================================================
# Intensity Bug Simulation — 3-Robot Comparative Viewer
# ============================================================================
#
# GPU-accelerated interactive viewer built with GLMakie.
#
# Three robots start from the same position and navigate toward a tower.
# A pirate tower spoofs the signal on a duty cycle. Each robot uses a
# different anti-pirate strategy:
#
#   Robot 1 (Naive,     Cyan):   No filter — blindly follows strongest signal.
#   Robot 2 (Cautious,  Orange): Pauses on anomaly, waits for confirmation
#                                 that the pirate stopped before resuming.
#   Robot 3 (Inertial,  Green):  Freezes angle on anomaly, continues blindly,
#                                 resumes sensor on pirate drop.
#
# Anomaly detection: a sudden intensity spike (inconsistent with physical
# movement) signals that the pirate turned ON. A sudden drop signals it
# turned OFF. Robots 2 & 3 use this to protect themselves.
#
# Controls:
#   ▶ Play / ⏸ Pause     — button or Space bar
#   0.5× / 1× / 2× / 10× — speed buttons
#   🖱 Scroll wheel        — scrub the timeline forward/backward
#   ← → Arrow keys        — step one frame at a time
#   |<< >>| buttons       — jump to start / end
#   Slider                — drag to any point in the simulation
#
# ALL SIMULATION LOGIC IS PRE-COMPUTED, then played back reactively.
#
# Requirements: GLMakie
#   using Pkg; Pkg.add("GLMakie")
# ============================================================================
using GLMakie
using GLMakie: xlims!, ylims!
using LinearAlgebra
using Printf

# ======================== Configuration ========================

const GRID_WIDTH = 20.0
const GRID_HEIGHT = 20.0
const STEP_SIZE = 0.3       # Movement step
const COLLISION_RADIUS = 0.3  # Robot collision radius (circular hitbox)
const TOWER_REACH_DIST = 0.5  # Distance to consider tower reached
const ROBOT_SIZE = 0.7       # Visual square side length (smaller for 3 robots)
const SENSOR_RANGE = COLLISION_RADIUS * 1.1  # Touch sensor probe distance

# ======================== Pirate Tower Configuration ========================
const PIRATE_DUTY_CYCLE = 1 / 4     # Options: 1/4, 1/3, or 1/2
const PIRATE_STRENGTH = 1.0     # Signal multiplier (1.0 = same as real tower)
const PIRATE_CYCLE_BASE = 40      # Total steps per duty cycle period

# ======================== Anomaly Detection Configuration ========================
# The detector compares each step's intensity change against the maximum
# physically possible change (derived from STEP_SIZE and current intensity).
# ANOMALY_THRESHOLD is a multiplier on that physical maximum.
# Higher → less sensitive (fewer false positives near the tower).
# Lower  → more sensitive (catches weaker pirate signals).
const ANOMALY_THRESHOLD = 2.0

# How many consecutive steps of stable signal Robot 2 (Cautious) requires
# after detecting the pirate drop before it resumes navigation.
const PIRATE_CONFIRM_STEPS = 3

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
    pirate_tower::Vec2
    obstacles::Vector{Obstacle}
end

mutable struct Robot
    pos::Vec2
    angle::Float64       # Radians, 0 = Right, π/2 = Up
    state::Symbol        # :MoveToTower, :FollowObstacle
    hit_intensity::Float64   # i_H: intensity recorded at the hit point
end

mutable struct FollowState
    prev_intensity::Float64  # i_L
    rising::Bool
    steps_in_follow::Int
    i_H::Float64
    active::Bool
end

# Trajectory point — stores per-step observable state for one robot
struct TrajectoryPoint
    x::Float64
    y::Float64
    step::Int
    state::Symbol             # Navigation state (:MoveToTower / :FollowObstacle)
    angle::Float64
    intensity_val::Float64    # Perceived intensity (may include pirate)
    front_sensor::Bool
    left_sensor::Bool
    pirate_active::Bool       # Whether pirate is transmitting this step
    high_intensity_memory::Float64   # Running max intensity
    anti_pirate_mode::Symbol  # :Normal, :Paused, :InertialMode
    reached::Bool             # Has this robot reached the real tower?
    confirm_counter::Int      # Confirmation countdown (Robot 2 only)
end

# Per-robot mutable simulation state — bundles everything the loop needs
mutable struct RobotState
    robot::Robot
    strategy::Symbol          # :Naive, :Cautious, :Inertial
    follow::FollowState
    high_intensity_memory::Float64
    prev_perceived::Float64   # Previous step's perceived intensity
    anti_pirate::Symbol       # :Normal, :Paused, :InertialMode
    frozen_angle::Float64     # Angle saved on entering anti-pirate mode
    frozen_state::Symbol      # Navigation state saved on entering anti-pirate mode
    confirm_counter::Int      # Steps of stable signal counted (Cautious only)
    reached::Bool
    reached_step::Int         # Step at which tower was reached (0 if not)
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

function intensity(pos::Vec2, tower::Vec2) #m(p) in the paper
    d = distance(pos, tower)
    d < 0.01 ? Inf : 1.0 / d
end

function is_pirate_active(step::Int)
    active_steps = round(Int, PIRATE_DUTY_CYCLE * PIRATE_CYCLE_BASE)
    return (step % PIRATE_CYCLE_BASE) < active_steps
end

# What the robot's sensor reads — max of real and (if active) pirate signal
function perceived_intensity(pos::Vec2, env::SimulationEnv, step::Int)
    real_i = intensity(pos, env.tower)
    if is_pirate_active(step)
        d_pirate = distance(pos, env.pirate_tower)
        pirate_i = d_pirate < 0.01 ? Inf : (1.0 / d_pirate) * PIRATE_STRENGTH
        return max(real_i, pirate_i)
    else
        return real_i
    end
end

# ======================== Collision Detection ========================

function collides_with_obstacle(pos::Vec2, env::SimulationEnv)
    for obs in env.obstacles
        cx = clamp(pos.x, obs.xmin, obs.xmax) #clamp limit the value between x.min to xmax
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

# ======================== Sensor Functions ========================

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

# Orient toward the strongest perceived signal source
function u_ori!(robot::Robot, env::SimulationEnv, step::Int=0)
    real_i = intensity(robot.pos, env.tower)
    if is_pirate_active(step)
        d_pirate = distance(robot.pos, env.pirate_tower)
        pirate_i = d_pirate < 0.01 ? Inf : (1.0 / d_pirate) * PIRATE_STRENGTH
        if pirate_i > real_i
            dx = env.pirate_tower.x - robot.pos.x
            dy = env.pirate_tower.y - robot.pos.y
            robot.angle = atan(dy, dx)
            return
        end
    end
    dx = env.tower.x - robot.pos.x
    dy = env.tower.y - robot.pos.y
    robot.angle = atan(dy, dx)
end

# Move one step forward; returns true if the move succeeded
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

# Nudge sideways toward the wall to maintain contact (u_fol)
function nudge_toward_wall!(robot::Robot, env::SimulationEnv)
    la = robot.angle + π / 2
    for d in (0.15, 0.10, 0.05)
        nudge_pos = Vec2(robot.pos.x + d * cos(la), robot.pos.y + d * sin(la))
        if is_valid_pos(nudge_pos, env)
            probe = Vec2(nudge_pos.x + SENSOR_RANGE * cos(la), nudge_pos.y + SENSOR_RANGE * sin(la))
            if !is_valid_pos(probe, env)
                robot.pos = nudge_pos
                return
            end
        end
    end
end

# Follow the obstacle boundary — try forward, then scan for wall-hugging direction
function boundary_step!(robot::Robot, env::SimulationEnv)
    fwd_pos = Vec2(robot.pos.x + STEP_SIZE * cos(robot.angle), robot.pos.y + STEP_SIZE * sin(robot.angle))
    if is_valid_pos(fwd_pos, env)
        la = robot.angle + π / 2
        probe = Vec2(robot.pos.x + SENSOR_RANGE * cos(la), robot.pos.y + SENSOR_RANGE * sin(la))
        if !is_valid_pos(probe, env)
            robot.pos = fwd_pos
            nudge_toward_wall!(robot, env)
            return true
        end
    end

    n_angles = 360
    angle_step = 2π / n_angles
    first_free_angle = -1.0
    first_free_pos = Vec2(0.0, 0.0)
    found_free = false

    for i in 0:(n_angles-1)
        test_angle = normalize_angle(robot.angle + π / 2 - i * angle_step)
        new_pos = Vec2(robot.pos.x + STEP_SIZE * cos(test_angle), robot.pos.y + STEP_SIZE * sin(test_angle))

        if is_valid_pos(new_pos, env)
            if !found_free
                first_free_angle = test_angle
                first_free_pos = new_pos
                found_free = true
            end

            la = test_angle + π / 2
            probe = Vec2(new_pos.x + SENSOR_RANGE * cos(la), new_pos.y + SENSOR_RANGE * sin(la))
            if !is_valid_pos(probe, env)
                robot.angle = test_angle
                robot.pos = new_pos
                nudge_toward_wall!(robot, env)
                return true
            end
        end
    end

    if found_free
        robot.angle = first_free_angle
        robot.pos = first_free_pos
        nudge_toward_wall!(robot, env)
        return true
    end
    return false
end

# ======================== Anomaly Detection ========================

# Maximum physically possible intensity change in one STEP_SIZE movement.
function max_natural_intensity_change(prev_intensity::Float64)
    if prev_intensity < 0.001
        return 0.01   # Very far from any tower — tiny natural changes
    end
    d = 1.0 / prev_intensity

    # Cap the theoretical maximum to prevent Inf which hides pirate drops.
    if d <= STEP_SIZE * 1.5
        return 1.5
    end

    val = STEP_SIZE / (d * (d - STEP_SIZE))
    return min(val, 1.5)   # Hard cap to ensure massive drops are always flagged
end

# Detect a positive spike (pirate turning ON)
# cur_i = current perceived intensity, prev_i = previous perceived intensity
function is_spike_anomaly(cur_i::Float64, prev_i::Float64)
    delta = cur_i - prev_i
    delta > 0 || return false
    return delta > ANOMALY_THRESHOLD * max_natural_intensity_change(prev_i)
end

# Detect a negative drop (pirate turning OFF)
# cur_i = current perceived intensity, prev_i = previous perceived intensity
function is_drop_anomaly(cur_i::Float64, prev_i::Float64)
    delta = prev_i - cur_i   # positive when signal dropped
    delta > 0 || return false
    if prev_i < 0.001
        return false   # Too small to judge
    end
    # A signal plummeting by more than 40% instantly is physically impossible
    # from natural movement unless the source was turned off.
    is_massive_drop = (delta / prev_i) > 0.40
    return is_massive_drop || (delta > ANOMALY_THRESHOLD * max_natural_intensity_change(prev_i))
end

# ======================== Robot Step Logic ========================

# Execute one normal navigation step (identical to the original algorithm)
function do_normal_step!(rs::RobotState, env::SimulationEnv, step::Int)
    robot = rs.robot
    if robot.state == :MoveToTower
        u_ori!(robot, env, step)
        if !u_fwd!(robot, env)
            # Hit an obstacle — switch to boundary following
            i_H = perceived_intensity(robot.pos, env, step)
            robot.hit_intensity = i_H
            robot.state = :FollowObstacle
            robot.angle = normalize_angle(robot.angle - π / 2)
            rs.follow = FollowState(i_H, false, 0, i_H, true)
        end
    elseif robot.state == :FollowObstacle
        boundary_step!(robot, env)
        rs.follow.steps_in_follow += 1
        cur_i = perceived_intensity(robot.pos, env, step)

        # Leave-obstacle condition: intensity rose above high_intensity_memory
        # and then started falling → we passed the leave point.
        if cur_i >= rs.follow.prev_intensity
            rs.follow.rising = true
        else
            if rs.follow.rising && rs.follow.steps_in_follow > 3
                if rs.follow.prev_intensity > rs.high_intensity_memory
                    rs.high_intensity_memory = rs.follow.prev_intensity
                    robot.state = :MoveToTower
                    u_ori!(robot, env, step)
                    rs.follow.active = false
                end
                rs.follow.rising = false
            else
                rs.follow.rising = false
            end
        end
        rs.follow.prev_intensity = cur_i
    end
end

# ── Cautious (Robot 2) ──
# On spike: pause immediately. On drop: start confirmation countdown.
# After PIRATE_CONFIRM_STEPS of stable signal, resume from where it was.
function update_cautious_step!(rs::RobotState, env::SimulationEnv, step::Int,
    cur_perceived::Float64)

    if rs.anti_pirate == :Normal
        # ── Check for anomalous positive spike ──
        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_pirate = :Paused
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            rs.confirm_counter = 0
            return  # Freeze — do not move
        end
        # ── Check for anomalous drop in Normal mode ──
        # This catches the pirate turning OFF when the ON was never detected
        # (e.g. pirate was active from step 0). Resets inflated memory.
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.high_intensity_memory = cur_perceived
        end
        # No spike anomaly → normal navigation
        do_normal_step!(rs, env, step)

    elseif rs.anti_pirate == :Paused
        # Robot is frozen, waiting for pirate to stop.
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            # Sharp drop detected — pirate likely turned off. Begin confirmation.
            rs.confirm_counter = 1
        elseif is_spike_anomaly(cur_perceived, rs.prev_perceived)
            # Another spike — pirate came back. Reset confirmation.
            rs.confirm_counter = 0
        elseif rs.confirm_counter > 0
            # Signal is stable — accumulate confidence
            rs.confirm_counter += 1
            if rs.confirm_counter >= PIRATE_CONFIRM_STEPS
                # ── Confirmed: pirate is off → resume previous activity ──
                rs.anti_pirate = :Normal
                rs.confirm_counter = 0
                # Restore the state the robot was in before pausing
                rs.robot.state = rs.frozen_state
                rs.robot.angle = rs.frozen_angle
                if rs.frozen_state == :MoveToTower
                    u_ori!(rs.robot, env, step)
                else
                    # Was following obstacle — refresh intensity baseline
                    rs.follow.prev_intensity = perceived_intensity(rs.robot.pos, env, step)
                    rs.follow.rising = false
                end
                # Reset high_intensity_memory to clean (pirate-free) value
                rs.high_intensity_memory = perceived_intensity(rs.robot.pos, env, step)
            end
        end
        # Stay frozen (no movement)
    end
end

# ── Inertial (Robot 3) ──
# On spike: freeze angle and keep current movement pattern.
#   - If MoveToTower: keep moving forward at frozen angle
#   - If FollowObstacle: keep following boundary (but skip leave check)
# On drop: resume sensor immediately and continue from current state.
function update_inertial_step!(rs::RobotState, env::SimulationEnv, step::Int,
    cur_perceived::Float64)
    if rs.anti_pirate == :Normal
        # ── Check for anomalous positive spike ──
        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_pirate = :InertialMode
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            # Immediately do one step in current movement pattern
            if rs.frozen_state == :MoveToTower
                rs.robot.angle = rs.frozen_angle
                u_fwd!(rs.robot, env)   # May fail if wall
            else  # :FollowObstacle
                boundary_step!(rs.robot, env)  # Keep hugging the wall
                rs.follow.steps_in_follow += 1
                # Skip leave-obstacle check — intensity is corrupted by pirate
            end
            return
        end
        # ── Check for anomalous drop in Normal mode ──
        # Catches pirate turning OFF when ON was never detected (first cycle).
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.high_intensity_memory = cur_perceived
        end
        # No spike anomaly → normal navigation
        do_normal_step!(rs, env, step)

    elseif rs.anti_pirate == :InertialMode
        # ── Check for signal drop (pirate turned off) ──
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            # Pirate dropped — reactivate sensor, resume previous activity
            rs.anti_pirate = :Normal
            rs.robot.state = rs.frozen_state
            # Reset high_intensity_memory to clean value
            rs.high_intensity_memory = perceived_intensity(rs.robot.pos, env, step)
            if rs.frozen_state == :MoveToTower
                u_ori!(rs.robot, env, step)
                u_fwd!(rs.robot, env)
            else  # :FollowObstacle
                # Refresh follow intensity baseline to current clean value
                rs.follow.prev_intensity = perceived_intensity(rs.robot.pos, env, step)
                rs.follow.rising = false
                # Continue boundary following normally
                do_normal_step!(rs, env, step)
            end
            return
        end
        # Still spoofed — continue current movement pattern
        if rs.frozen_state == :MoveToTower
            rs.robot.angle = rs.frozen_angle
            u_fwd!(rs.robot, env)   # May fail if wall — robot stays put
        else  # :FollowObstacle
            boundary_step!(rs.robot, env)  # Keep hugging the wall
            rs.follow.steps_in_follow += 1
            # Skip leave-obstacle check — intensity is corrupted by pirate
        end
    end
end

# ── Snapshot helper: build a TrajectoryPoint from current RobotState ──
function make_traj_point(rs::RobotState, env::SimulationEnv, step::Int)
    TrajectoryPoint(
        rs.robot.pos.x, rs.robot.pos.y, step,
        rs.robot.state, rs.robot.angle,
        perceived_intensity(rs.robot.pos, env, step),
        check_front_sensor(rs.robot, env),
        check_left_sensor(rs.robot, env),
        is_pirate_active(step),
        rs.high_intensity_memory,
        rs.anti_pirate,
        rs.reached,
        rs.confirm_counter
    )
end

# ── Master update: drives one step for a single robot ──
function update_robot_step!(rs::RobotState, env::SimulationEnv, step::Int)
    # Already at tower — nothing to do
    rs.reached && return

    # Check if tower just reached
    if distance(rs.robot.pos, env.tower) < TOWER_REACH_DIST
        rs.reached = true
        rs.reached_step = step
        return
    end

    # Compute perceived intensity at current position (before movement)
    cur_perceived = perceived_intensity(rs.robot.pos, env, step)

    # ── Strategy dispatch ──
    if rs.strategy == :Naive
        # Naive robot: always update high_intensity_memory (no filter)
        if cur_perceived > rs.high_intensity_memory
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)
    elseif rs.strategy == :Cautious
        update_cautious_step!(rs, env, step, cur_perceived)
    elseif rs.strategy == :Inertial
        update_inertial_step!(rs, env, step, cur_perceived)
    end

    # For R2/R3: update high_intensity_memory AFTER strategy dispatch,
    # only when still in Normal mode. This prevents pirate-boosted
    # values from inflating memory (the strategy already detected the
    # spike and switched to anti-pirate mode before we reach here).
    if rs.strategy != :Naive && rs.anti_pirate == :Normal
        post_i = perceived_intensity(rs.robot.pos, env, step)
        if post_i > rs.high_intensity_memory
            rs.high_intensity_memory = post_i
        end
    end

    # Update prev_perceived at the robot's FINAL position this step
    rs.prev_perceived = perceived_intensity(rs.robot.pos, env, step)
end

# ======================== Pre-compute Full Simulation ========================

function run_full_simulation(env::SimulationEnv, robot_start::Vec2, max_steps::Int)
    strategies = [:Naive, :Cautious, :Inertial]
    states = RobotState[]

    for strat in strategies
        robot = Robot(Vec2(robot_start.x, robot_start.y), 0.0, :MoveToTower, 0.0)
        u_ori!(robot, env, 0)
        init_i = perceived_intensity(robot.pos, env, 0)
        push!(states, RobotState(
            robot, strat,
            FollowState(0.0, false, 0, 0.0, false),  # follow
            init_i,          # high_intensity_memory
            init_i,          # prev_perceived
            :Normal,         # anti_pirate
            0.0,             # frozen_angle
            :MoveToTower,    # frozen_state
            0,               # confirm_counter
            false,           # reached
            0                # reached_step
        ))
    end

    # Trajectory storage — one vector per robot
    trajectories = [TrajectoryPoint[] for _ in 1:3]

    # Record initial state (step 0)
    step = 0
    for (i, rs) in enumerate(states)
        push!(trajectories[i], make_traj_point(rs, env, step))
    end

    # ── Main loop: advance all 3 robots each tick ──
    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for (i, rs) in enumerate(states)
            update_robot_step!(rs, env, step)
            push!(trajectories[i], make_traj_point(rs, env, step))
        end
    end

    reached_flags = [rs.reached for rs in states]
    reached_steps = [rs.reached_step for rs in states]
    return trajectories, reached_flags, reached_steps
end

# ======================== Display Helper ========================

# Human-readable state label for the HUD
function state_display_label(t::TrajectoryPoint)
    if t.reached
        return "★ Arrived"
    elseif t.anti_pirate_mode == :Paused
        if t.confirm_counter > 0
            return @sprintf("⏸ Confirm %d/%d", t.confirm_counter, PIRATE_CONFIRM_STEPS)
        else
            return "⏸ Paused"
        end
    elseif t.anti_pirate_mode == :InertialMode
        return "⤇ Inertial"
    elseif t.state == :MoveToTower
        return "→ Moving"
    else
        return "~ Following"
    end
end

# ======================== GLMakie Interactive Viewer ========================

function launch_viewer(env::SimulationEnv,
    trajs::Vector{Vector{TrajectoryPoint}},
    reached_flags::Vector{Bool},
    reached_steps::Vector{Int})

    n_frames = length(trajs[1])   # All 3 trajectories are the same length
    strategy_names = ["R1 Naive", "R2 Cautious", "R3 Inertial"]

    println("  Total frames: $n_frames")
    for i in 1:3
        status = reached_flags[i] ? "Yes ✓ (step $(reached_steps[i]))" : "No ✗"
        println("  $(strategy_names[i]) reached: $status")
    end

    # ── Pre-compute trajectory point arrays ──
    all_pts = [[Point2f(t.x, t.y) for t in traj] for traj in trajs]

    # ── Color Palette (Dark Modern Theme) ──
    bg_color = RGBf(0.059, 0.059, 0.137)
    grid_bg = RGBf(0.086, 0.094, 0.165)
    grid_line_color = RGBAf(1, 1, 1, 0.035)
    grid_border_col = RGBf(0.30, 0.32, 0.45)
    obstacle_fill = RGBf(0.27, 0.29, 0.40)
    obstacle_stroke = RGBf(0.40, 0.42, 0.55)
    tower_color = RGBf(1.0, 0.40, 0.40)
    tower_glow_col = RGBAf(1.0, 0.25, 0.25, 0.12)
    text_primary = RGBf(0.92, 0.94, 0.96)
    text_secondary = RGBf(0.60, 0.64, 0.76)
    text_dim = RGBf(0.40, 0.43, 0.55)
    btn_default = RGBf(0.14, 0.15, 0.24)
    btn_hover = RGBf(0.20, 0.22, 0.32)
    btn_active = RGBf(0.26, 0.36, 0.90)
    sensor_on = RGBf(0.10, 0.92, 0.30)
    sensor_off = RGBf(0.40, 0.42, 0.48)

    # Robot-specific colors
    R1_COLOR = RGBf(0.30, 0.79, 0.94)   # Cyan   — Naive
    R2_COLOR = RGBf(1.00, 0.60, 0.20)   # Orange — Cautious
    R3_COLOR = RGBf(0.30, 0.90, 0.30)   # Green  — Inertial
    robot_colors = [R1_COLOR, R2_COLOR, R3_COLOR]

    # ── Winner computation (static, for end-of-sim banner) ──
    winner_idx = -1
    winner_step = typemax(Int)
    for i in 1:3
        if reached_flags[i] && reached_steps[i] < winner_step
            winner_step = reached_steps[i]
            winner_idx = i
        end
    end
    result_parts = String[]
    for i in 1:3
        if reached_flags[i]
            push!(result_parts, "$(strategy_names[i]): ✓ Step $(reached_steps[i])")
        else
            push!(result_parts, "$(strategy_names[i]): ✗")
        end
    end
    result_str = join(result_parts, "   │   ")
    winner_str = winner_idx > 0 ?
                 " $(strategy_names[winner_idx]) wins!  (Step $winner_step)" :
                 "No robot reached the tower"

    # ═══════════════════ FIGURE & LAYOUT ═══════════════════

    fig = Figure(size=(1050, 1020), backgroundcolor=bg_color)

    # Row 1 — Title
    Label(fig[1, 1], " 3-Robot Pirate Tower Comparative Simulation",
        fontsize=20, color=text_primary, tellwidth=false,
        halign=:center, font=:bold)

    # Row 3 — Main simulation axis
    ax = Axis(fig[3, 1],
        aspect=DataAspect(),
        backgroundcolor=grid_bg,
        xgridvisible=false, ygridvisible=false,
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false)
    xlims!(ax, -1.0, 21.0)
    ylims!(ax, -1.0, 21.0)

    deactivate_interaction!(ax, :scrollzoom)
    deactivate_interaction!(ax, :rectanglezoom)
    deactivate_interaction!(ax, :limitreset)

    # Row 4 — Controls
    controls = GridLayout(fig[4, 1], tellwidth=false)

    # Row 5 — Timeline slider
    sl = Slider(fig[5, 1],
        range=1:n_frames,
        startvalue=1,
        color_active=btn_active,
        color_inactive=RGBf(0.18, 0.18, 0.28),
        color_active_dimmed=RGBf(0.18, 0.25, 0.55),
        linewidth=14)
    frame_idx = sl.value   # Single source of truth

    # ── State observables ──
    speed_val = Observable(1.0)
    is_playing = Observable(false)
    timer_ref = Ref{Union{Timer,Nothing}}(nothing)
    programmatic_update = Ref(false)

    # ─── Row 2 — Info panel (common line + 3 robot lines) ───
    info_panel = GridLayout(fig[2, 1], tellwidth=false)

    common_info = @lift begin
        t = trajs[1][$frame_idx]
        pir = t.pirate_active ? "⚡ ON" : "OFF"
        @sprintf("Step %d / %d   │   Pirate: %s", t.step, n_frames - 1, pir)
    end
    Label(info_panel[1, 1], common_info,
        fontsize=11, color=text_secondary, halign=:center, tellwidth=false)

    # Per-robot info lines
    for (i, (traj, col, name)) in enumerate(zip(trajs, robot_colors, strategy_names))
        let traj = traj, col = col, name = name, i = i
            info_obs = @lift begin
                t = traj[$frame_idx]
                st = state_display_label(t)
                iv = min(t.intensity_val, 9999.9)
                hi = min(t.high_intensity_memory, 9999.9)
                reached_str = ""
                if reached_flags[i] && t.reached
                    reached_str = @sprintf("   ✓ Step %d", reached_steps[i])
                end
                @sprintf("%s:  %s   │  I = %.4f   │  Hi = %.4f%s",
                    name, st, iv, hi, reached_str)
            end
            Label(info_panel[1+i, 1], info_obs,
                fontsize=10, color=col, halign=:center, tellwidth=false)
        end
    end
    rowgap!(info_panel, 2)

    # Row 6 — Hint bar
    hint_text = @lift @sprintf(
        "Frame %d / %d    │    Speed: %g×    │    Scroll to scrub · Space play/pause · ←→ step",
        $frame_idx, n_frames, $speed_val)
    Label(fig[6, 1], hint_text,
        fontsize=10, color=text_dim, tellwidth=false, halign=:center)

    # Fix row heights
    rowsize!(fig.layout, 1, Fixed(32))
    rowsize!(fig.layout, 2, Fixed(80))
    rowsize!(fig.layout, 3, Auto())
    rowsize!(fig.layout, 4, Fixed(50))
    rowsize!(fig.layout, 5, Fixed(25))
    rowsize!(fig.layout, 6, Fixed(22))
    rowgap!(fig.layout, 4)

    # ═══════════════════ CONTROL BUTTONS ═══════════════════

    btn_0_5x_col = @lift $speed_val == 0.5 ? btn_active : btn_default
    btn_1x_col = @lift $speed_val == 1.0 ? btn_active : btn_default
    btn_2x_col = @lift $speed_val == 2.0 ? btn_active : btn_default
    btn_10x_col = @lift $speed_val == 10.0 ? btn_active : btn_default

    common_btn = (;
        buttoncolor_hover=btn_hover,
        buttoncolor_active=btn_active,
        labelcolor=text_primary,
        labelcolor_hover=text_primary,
        cornerradius=8,
        height=38)

    btn_0_5x = Button(controls[1, 1], label="0.5x", width=62; buttoncolor=btn_0_5x_col, common_btn...)
    btn_1x = Button(controls[1, 2], label="1x", width=58; buttoncolor=btn_1x_col, common_btn...)
    btn_2x = Button(controls[1, 3], label="2x", width=58; buttoncolor=btn_2x_col, common_btn...)
    btn_10x = Button(controls[1, 4], label="10x", width=62; buttoncolor=btn_10x_col, common_btn...)

    btn_start = Button(controls[1, 6], label="|<<", width=42; buttoncolor=btn_default, common_btn...)
    btn_back = Button(controls[1, 7], label="<-", width=42; buttoncolor=btn_default, common_btn...)

    play_label = @lift $is_playing ? "|| Pause" : ">  Play"
    play_btn_col = @lift $is_playing ? RGBf(0.97, 0.15, 0.52) : btn_active
    btn_play = Button(controls[1, 8], label=play_label, width=100;
        buttoncolor=play_btn_col, common_btn...)

    btn_fwd = Button(controls[1, 9], label="->", width=42; buttoncolor=btn_default, common_btn...)
    btn_end = Button(controls[1, 10], label=">>|", width=42; buttoncolor=btn_default, common_btn...)

    colgap!(controls, 5)
    colgap!(controls, 4, 25)
    colgap!(controls, 5, 25)

    # ═══════════════════ DRAWING ═══════════════════

    # — Grid background —
    poly!(ax, Rect(0, 0, GRID_WIDTH, GRID_HEIGHT),
        color=grid_bg, strokecolor=grid_border_col, strokewidth=2)

    for i in 0:Int(GRID_WIDTH)
        lines!(ax, [Float64(i), Float64(i)], [0.0, GRID_HEIGHT],
            color=grid_line_color, linewidth=0.5)
    end
    for j in 0:Int(GRID_HEIGHT)
        lines!(ax, [0.0, GRID_WIDTH], [Float64(j), Float64(j)],
            color=grid_line_color, linewidth=0.5)
    end

    # — Obstacles —
    for obs in env.obstacles
        poly!(ax, Rect(obs.xmin, obs.ymin, obs.xmax - obs.xmin, obs.ymax - obs.ymin),
            color=obstacle_fill, strokecolor=obstacle_stroke, strokewidth=1.5)
    end

    # — Real Tower glow + star —
    scatter!(ax, [env.tower.x], [env.tower.y],
        markersize=55, marker=:circle, color=tower_glow_col)
    scatter!(ax, [env.tower.x], [env.tower.y],
        marker=:star5, markersize=26,
        color=tower_color, strokecolor=RGBf(0.75, 0.10, 0.10), strokewidth=2)

    # — Pirate Tower (green star, always visible) —
    pirate_vis_color = RGBf(0.0, 1.0, 0.4)
    scatter!(ax, [env.pirate_tower.x], [env.pirate_tower.y],
        markersize=55, marker=:circle, color=RGBAf(0.0, 1.0, 0.4, 0.08))
    scatter!(ax, [env.pirate_tower.x], [env.pirate_tower.y],
        marker=:star5, markersize=26,
        color=pirate_vis_color, strokecolor=RGBf(0.0, 0.6, 0.2), strokewidth=2)

    # — Pirate active glow ring (visible only when pirate is transmitting) —
    pirate_glow_vis = @lift trajs[1][$frame_idx].pirate_active
    scatter!(ax, [env.pirate_tower.x], [env.pirate_tower.y],
        markersize=80, marker=:circle,
        color=RGBAf(0.0, 1.0, 0.4, 0.18),
        visible=pirate_glow_vis)

    # — Pirate active warning text —
    text!(ax, Point2f(GRID_WIDTH / 2, GRID_HEIGHT - 0.8),
        text="⚡ PIRATE ACTIVE", fontsize=16,
        color=RGBAf(0.0, 1.0, 0.4, 0.92),
        align=(:center, :center), font=:bold,
        visible=pirate_glow_vis)

    # ── Draw trajectory trails (all 3 robots, behind the robot bodies) ──
    for (i, (traj, col)) in enumerate(zip(trajs, robot_colors))
        let traj = traj, col = col, pts = all_pts[i]
            vis_pts = @lift pts[1:$frame_idx]
            lines!(ax, vis_pts;
                color=RGBAf(col.r, col.g, col.b, 0.35),
                linewidth=2)
        end
    end

    # ── Draw robot shapes (on top of trails) ──
    for (i, (traj, col)) in enumerate(zip(trajs, robot_colors))
        let traj = traj, col = col
            # Glow halo
            scatter!(ax,
                @lift([traj[$frame_idx].x]),
                @lift([traj[$frame_idx].y]),
                markersize=28,
                color=@lift(RGBAf(col.r, col.g, col.b,
                    traj[$frame_idx].reached ? 0.05 : 0.15)))

            # Body (rotated square) — semi-transparent if reached or paused
            robot_verts = @lift begin
                t = traj[$frame_idx]
                h = ROBOT_SIZE / 2
                ca, sa = cos(t.angle), sin(t.angle)
                Point2f[(t.x + cx * ca - cy * sa,
                    t.y + cx * sa + cy * ca)
                        for (cx, cy) in ((-h, -h), (h, -h), (h, h), (-h, h))]
            end
            body_col = @lift begin
                if traj[$frame_idx].reached
                    RGBAf(col.r, col.g, col.b, 0.35)
                elseif traj[$frame_idx].anti_pirate_mode == :Paused
                    RGBAf(col.r, col.g, col.b, 0.55)
                else
                    RGBAf(col.r, col.g, col.b, 1.0)
                end
            end
            poly!(ax, robot_verts, color=body_col, strokecolor=:white, strokewidth=1.5)

            # Direction indicator
            lines!(ax,
                @lift([traj[$frame_idx].x,
                    traj[$frame_idx].x + ROBOT_SIZE * 0.55 * cos(traj[$frame_idx].angle)]),
                @lift([traj[$frame_idx].y,
                    traj[$frame_idx].y + ROBOT_SIZE * 0.55 * sin(traj[$frame_idx].angle)]),
                color=:white, linewidth=1.5)

            # Front sensor dot
            scatter!(ax,
                @lift([traj[$frame_idx].x + (ROBOT_SIZE / 2) * cos(traj[$frame_idx].angle)]),
                @lift([traj[$frame_idx].y + (ROBOT_SIZE / 2) * sin(traj[$frame_idx].angle)]),
                markersize=7,
                color=@lift(traj[$frame_idx].front_sensor ? sensor_on : sensor_off),
                strokecolor=:white, strokewidth=0.5)

            # Left sensor dot
            scatter!(ax,
                @lift([traj[$frame_idx].x + (ROBOT_SIZE / 2) * (-sin(traj[$frame_idx].angle))]),
                @lift([traj[$frame_idx].y + (ROBOT_SIZE / 2) * cos(traj[$frame_idx].angle)]),
                markersize=7,
                color=@lift(traj[$frame_idx].left_sensor ? sensor_on : sensor_off),
                strokecolor=:white, strokewidth=0.5)
        end
    end

    # ── Completion banner (visible only at last frame) ──
    banner_vis = @lift $frame_idx >= n_frames
    text!(ax, Point2f(GRID_WIDTH / 2, GRID_HEIGHT / 2 + 2.8),
        text=winner_str, fontsize=22,
        color=RGBAf(1.0, 0.84, 0.0, 0.92),
        align=(:center, :center), font=:bold,
        visible=banner_vis)
    text!(ax, Point2f(GRID_WIDTH / 2, GRID_HEIGHT / 2 + 1.2),
        text=result_str, fontsize=13,
        color=RGBAf(0.90, 0.90, 0.95, 0.85),
        align=(:center, :center),
        visible=banner_vis)

    # ═══════════════════ PLAYBACK ENGINE ═══════════════════

    function stop_playback!()
        if timer_ref[] !== nothing
            try
                close(timer_ref[])
            catch
            end
            timer_ref[] = nothing
        end
    end

    function start_playback!()
        stop_playback!()
        step_inc = speed_val[]
        interval = 1.0 / 30.0            # 30 fps
        current_frame_float = Float64(frame_idx[])
        timer_ref[] = Timer(interval; interval=interval) do t
            try
                current_frame_float = min(current_frame_float + step_inc, n_frames)
                nxt = round(Int, current_frame_float)
                if nxt > frame_idx[]
                    programmatic_update[] = true
                    set_close_to!(sl, nxt)
                    programmatic_update[] = false
                end
                if nxt >= n_frames
                    is_playing[] = false
                    stop_playback!()
                end
            catch
                is_playing[] = false
                stop_playback!()
            end
        end
    end

    function toggle_playback!()
        if is_playing[]
            is_playing[] = false
            stop_playback!()
        else
            # If at the very end, restart from the beginning
            if frame_idx[] >= n_frames
                programmatic_update[] = true
                set_close_to!(sl, 1)
                programmatic_update[] = false
            end
            is_playing[] = true
            start_playback!()
        end
    end

    # Auto-pause when the user drags the slider manually
    on(frame_idx) do _
        if !programmatic_update[] && is_playing[]
            is_playing[] = false
            stop_playback!()
        end
    end

    # Restart timer when speed changes during playback
    on(speed_val) do _
        if is_playing[]
            start_playback!()
        end
    end

    # ═══════════════════ EVENT HANDLERS ═══════════════════

    # Speed buttons
    on(btn_0_5x.clicks) do _
        speed_val[] = 0.5
    end
    on(btn_1x.clicks) do _
        speed_val[] = 1.0
    end
    on(btn_2x.clicks) do _
        speed_val[] = 2.0
    end
    on(btn_10x.clicks) do _
        speed_val[] = 10.0
    end

    # Play / Pause
    on(btn_play.clicks) do _
        toggle_playback!()
    end

    # Navigation buttons
    on(btn_start.clicks) do _
        if is_playing[]
            is_playing[] = false
            stop_playback!()
        end
        programmatic_update[] = true
        set_close_to!(sl, 1)
        programmatic_update[] = false
    end
    on(btn_end.clicks) do _
        if is_playing[]
            is_playing[] = false
            stop_playback!()
        end
        programmatic_update[] = true
        set_close_to!(sl, n_frames)
        programmatic_update[] = false
    end
    on(btn_back.clicks) do _
        programmatic_update[] = true
        set_close_to!(sl, max(1, frame_idx[] - 1))
        programmatic_update[] = false
    end
    on(btn_fwd.clicks) do _
        programmatic_update[] = true
        set_close_to!(sl, min(n_frames, frame_idx[] + 1))
        programmatic_update[] = false
    end

    # Scroll wheel — scrub timeline
    on(events(fig.scene).scroll) do (_, dy)
        dy == 0 && return Consume(false)
        if is_playing[]
            is_playing[] = false
            stop_playback!()
        end
        delta = round(Int, dy * 5)
        new_val = clamp(frame_idx[] + delta, 1, n_frames)
        programmatic_update[] = true
        set_close_to!(sl, new_val)
        programmatic_update[] = false
        return Consume(false)
    end

    # Keyboard — Space, Left, Right
    on(events(fig.scene).keyboardbutton) do event
        (event.action in (Keyboard.press, Keyboard.repeat)) || return Consume(false)
        if event.key == Keyboard.space
            toggle_playback!()
        elseif event.key == Keyboard.left
            if is_playing[]
                is_playing[] = false
                stop_playback!()
            end
            programmatic_update[] = true
            set_close_to!(sl, max(1, frame_idx[] - 1))
            programmatic_update[] = false
        elseif event.key == Keyboard.right
            if is_playing[]
                is_playing[] = false
                stop_playback!()
            end
            programmatic_update[] = true
            set_close_to!(sl, min(n_frames, frame_idx[] + 1))
            programmatic_update[] = false
        end
        return Consume(false)
    end

    # Clean up timer when window closes
    on(events(fig.scene).window_open) do is_open
        is_open || stop_playback!()
    end

    # ─── Display ───
    display(fig)

    println("\n Viewer ready!")
    println("    Play / Pause  :  button or Space")
    println("    Speed         :  0.5x / 1x / 2x / 10x buttons")
    println("    Scroll wheel  :  scrub timeline forward / backward")
    println("    Step          :  ← → arrow keys")
    println("    Slider        :  drag to any frame")

    return fig
end

# ======================== Main ========================

function run_intensity_bug()
    tower = Vec2(18.0, 18.0)
    robot_start = Vec2(2.0, 1.0)
    obstacles = Obstacle[]
    WALL_T = 0.3

    # =========================================================
    # SCENARIO SELECTOR
    # Change this number to test different obstacle layouts:
    # 1 = Simple, 2 = Default (Original), 3 = Random
    # =========================================================
    scenario = 3

    if scenario == 1
        # SCENARIO 1: Simple
        # Just a single, straightforward block in the center to navigate around.
        push!(obstacles, Obstacle(8.0, 8.0, 12.0, 12.0))

    elseif scenario == 2
        # SCENARIO 2: Default (The original staggered walls)
        push!(obstacles, Obstacle(4.0, 4.0 - WALL_T, 20.0, 4.0 + WALL_T))
        push!(obstacles, Obstacle(0.0, 8.0 - WALL_T, 16.0, 8.0 + WALL_T))
        push!(obstacles, Obstacle(4.0, 12.0 - WALL_T, 20.0, 12.0 + WALL_T))
        push!(obstacles, Obstacle(0.0, 16.0 - WALL_T, 16.0, 16.0 + WALL_T))

    elseif scenario == 3
        # SCENARIO 3: Random
        # Generates random rectangular obstacles every time you run the script.
        for _ in 1:rand(3:9)
            w = rand() * 3.0 + 1.5  # Random width between 1.5 and 4.5
            h = rand() * 3.0 + 1.5  # Random height between 1.5 and 4.5

            # Restrict spawn coordinates to keep the start (2,1) and tower (18,18) clear
            x = rand() * 10.0 + 4.0
            y = rand() * 10.0 + 4.0

            push!(obstacles, Obstacle(x, y, x + w, y + h))
        end
    end

    # Place pirate tower at a random valid location
    pirate_pos = Vec2(0.0, 0.0)
    temp_env = SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, Vec2(0.0, 0.0), obstacles)
    while true
        px = rand() * (GRID_WIDTH - 2 * COLLISION_RADIUS) + COLLISION_RADIUS
        py = rand() * (GRID_HEIGHT - 2 * COLLISION_RADIUS) + COLLISION_RADIUS
        candidate = Vec2(px, py)
        if is_valid_pos(candidate, temp_env) &&
           distance(candidate, robot_start) > 3.0 &&
           distance(candidate, tower) > 3.0
            pirate_pos = candidate
            break
        end
    end

    env = SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, pirate_pos, obstacles)

    @printf(" Pirate Tower placed at (%.2f, %.2f)\n", pirate_pos.x, pirate_pos.y)
    @printf(" Pirate config: duty=%.2f, strength=%.2f, cycle_base=%d\n",
        PIRATE_DUTY_CYCLE, PIRATE_STRENGTH, PIRATE_CYCLE_BASE)
    @printf(" Anomaly detection: threshold=%.1f, confirm_steps=%d\n",
        ANOMALY_THRESHOLD, PIRATE_CONFIRM_STEPS)

    println(" Running 3-robot comparative simulation (scenario $scenario)...")
    trajs, reached_flags, reached_steps = run_full_simulation(env, robot_start, 1000)
    println(" Simulation complete!")

    strategy_names = ["R1 Naive", "R2 Cautious", "R3 Inertial"]
    for i in 1:3
        status = reached_flags[i] ? "✓ (step $(reached_steps[i]))" : "✗"
        println("  $(strategy_names[i]): $(length(trajs[i])) frames — $status")
    end

    println("\n Launching interactive viewer...")
    fig = launch_viewer(env, trajs, reached_flags, reached_steps)

    return fig
end

# ─── Entry point ───
run_intensity_bug()
