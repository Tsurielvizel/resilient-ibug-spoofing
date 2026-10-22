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
using DataFrames
using CSV
using Random
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
    i_H::Float64             # Hit intensity for the current wall
    local_max_i::Float64     # Running peak along this specific wall
    steps_since_max::Int     # Descent tracker to confirm a true peak
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
    needs_orientation::Bool   # [FIX] Ensures straight lines!
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

function intensity(pos::Vec2, tower::Vec2) # m(p) in the paper
    d = distance(pos, tower)
    return d < 0.01 ? Inf : 1.0 / d
end

function is_pirate_active(step::Int)
    active_steps = round(Int, PIRATE_DUTY_CYCLE * PIRATE_CYCLE_BASE)
    return (step % PIRATE_CYCLE_BASE) < active_steps
end

# What the robot's sensor reads — SUPERPOSITION of real and (if active) pirate signal
function perceived_intensity(pos::Vec2, env::SimulationEnv, step::Int)
    target_signal = intensity(pos, env.tower)
    
    pirate_signal = 0.0
    if is_pirate_active(step)
        # We calculate the pirate intensity just like the target, multiplied by its strength
        pirate_signal = intensity(pos, env.pirate_tower) * PIRATE_STRENGTH
    end
    
    # Physical wave superposition (Addition instead of Max)
    return target_signal + pirate_signal
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

# Orient toward the steepest signal gradient (Gradient Alignment Sensor)
# The robot probes its immediate surroundings without knowing global coordinates.
function u_ori!(robot::Robot, env::SimulationEnv, step::Int=0)
    best_angle = robot.angle
    max_intensity_found = -Inf
    
    # Simulate the robot checking signal intensity in a full circle around itself
    for a in 0:5:359
        rad = deg2rad(a)
        # Probe a tiny distance (e.g., 0.05 units) in this direction
        probe_pos = Vec2(robot.pos.x + 0.05 * cos(rad), robot.pos.y + 0.05 * sin(rad))
        
        # Read the combined (superimposed) signal at the probe location
        i_val = perceived_intensity(probe_pos, env, step)
        
        if i_val > max_intensity_found
            max_intensity_found = i_val
            best_angle = rad
        end
    end
    
    # Lock the robot's heading to the steepest gradient
    robot.angle = best_angle
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

# u_fol: Robust radial sweep for wall following. Prevents stuttering and getting stuck.
function u_fol!(robot::Robot, env::SimulationEnv)
    # Start looking "over the left shoulder" to tightly wrap around convex corners
    start_angle = robot.angle + (π / 2) + deg2rad(15) 
    
    # Sweep clockwise until we find the first valid, clear step
    for i in 0:72 # 5 degree increments
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

# Execute one normal navigation step (Exact I-Bug algorithm logic)
# Execute one normal navigation step (Exact I-Bug algorithm logic)
function do_normal_step!(rs::RobotState, env::SimulationEnv, step::Int)
    robot = rs.robot
    if robot.state == :MoveToTower
        
        # [FIX] Only calculate gradient once to ensure straight line walking!
        if rs.needs_orientation
            u_ori!(robot, env, step)
            rs.needs_orientation = false
        end
        
        if !u_fwd!(robot, env)
            # Hit an obstacle — switch to boundary following
            i_H = perceived_intensity(robot.pos, env, step)
            robot.hit_intensity = i_H
            robot.state = :FollowObstacle
            robot.angle = normalize_angle(robot.angle - π / 2)
            
            # Initialize LOCAL memory for this specific obstacle
            rs.follow = FollowState(i_H, i_H, 0, true)
            rs.needs_orientation = true # We will need to re-orient when we leave the wall
        end
    elseif robot.state == :FollowObstacle
        u_fol!(robot, env) 
        
        cur_i = perceived_intensity(robot.pos, env, step)

        # Update Local Maximum Tracker
        if cur_i > rs.follow.local_max_i
            rs.follow.local_max_i = cur_i
            rs.follow.steps_since_max = 0 # Still climbing
        else
            rs.follow.steps_since_max += 1 # Descending
        end
        
        # Step 6: Leave condition based STRICTLY on local memory
        has_passed_hit_intensity = (rs.follow.local_max_i > rs.follow.i_H)
        is_descending_from_peak = (rs.follow.steps_since_max >= 3)
        
        if has_passed_hit_intensity && is_descending_from_peak
            robot.state = :MoveToTower
            rs.follow.active = false
            rs.needs_orientation = true # [FIX] Force the robot to find the target again!
        end
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
                    rs.needs_orientation = true # Re-orient to target after pause
                else
                    # [FIX] Refresh local peak so we don't falsely leave the wall due to the signal drop
                    rs.follow.local_max_i = perceived_intensity(rs.robot.pos, env, step)
                    rs.follow.steps_since_max = 0
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
# ── Inertial (Robot 3) ──
function update_inertial_step!(rs::RobotState, env::SimulationEnv, step::Int,
    cur_perceived::Float64)
    if rs.anti_pirate == :Normal
        if is_spike_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_pirate = :InertialMode
            rs.frozen_angle = rs.robot.angle
            rs.frozen_state = rs.robot.state
            
            # Immediately do one step in current movement pattern
            if rs.frozen_state == :MoveToTower
                rs.robot.angle = rs.frozen_angle
                u_fwd!(rs.robot, env)   
            else  # :FollowObstacle
                u_fol!(rs.robot, env)  # [FIX] Use u_fol! instead of the deleted boundary_step!
            end
            return
        end
        
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.high_intensity_memory = cur_perceived
        end
        do_normal_step!(rs, env, step)

    elseif rs.anti_pirate == :InertialMode
        if is_drop_anomaly(cur_perceived, rs.prev_perceived)
            rs.anti_pirate = :Normal
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
        
        # Still spoofed — continue current movement pattern
        if rs.frozen_state == :MoveToTower
            rs.robot.angle = rs.frozen_angle
            u_fwd!(rs.robot, env)   
        else  
            u_fol!(rs.robot, env)  # [FIX] Use u_fol! instead of the deleted boundary_step!
        end
    end
end

# ── Snapshot helper: build a TrajectoryPoint from current RobotState ──
function is_pirate_active(step::Int)
    # [FIX] Give the robots 20 clean steps at the start to establish a baseline
    if step < 20 return false end 
    active_steps = round(Int, PIRATE_DUTY_CYCLE * PIRATE_CYCLE_BASE)
    return ((step - 20) % PIRATE_CYCLE_BASE) < active_steps
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
# ======================== Environment Generation ========================

function generate_random_env()
    tower = Vec2(18.0, 18.0)
    robot_start = Vec2(2.0, 1.0)
    obstacles = Obstacle[]
    for _ in 1:rand(3:9)
        w = rand() * 3.0 + 1.5; h = rand() * 3.0 + 1.5
        x = rand() * 10.0 + 4.0; y = rand() * 10.0 + 4.0
        push!(obstacles, Obstacle(x, y, x + w, y + h))
    end
    pirate_pos = Vec2(0.0, 0.0)
    temp_env = SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, Vec2(0.0, 0.0), obstacles)
    while true
        cand = Vec2(rand() * (GRID_WIDTH - 2*COLLISION_RADIUS) + COLLISION_RADIUS, 
                    rand() * (GRID_HEIGHT - 2*COLLISION_RADIUS) + COLLISION_RADIUS)
        if is_valid_pos(cand, temp_env) && distance(cand, robot_start) > 6.0 && distance(cand, tower) > 8.0
            pirate_pos = cand; break
        end
    end
    return SimulationEnv(GRID_WIDTH, GRID_HEIGHT, tower, pirate_pos, obstacles)
end

# ======================== Batch CSV Engine (No GUI, Very Fast) ========================

function run_single_batch_sim(env::SimulationEnv, max_steps::Int = 3000)
    strategies = [:Naive, :Cautious, :Inertial]
    states = RobotState[]
    
    for strat in strategies
        robot = Robot(Vec2(2.0, 1.0), 0.0, :MoveToTower, 0.0)
        u_ori!(robot, env, 0)
        init_i = perceived_intensity(robot.pos, env, 0)
        push!(states, RobotState(
            robot, strat, FollowState(init_i, init_i, 0, false),
            init_i, init_i, :Normal, 0.0, :MoveToTower, 0, false, 0, true
        ))
    end
    
    step = 0
    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for rs in states update_robot_step!(rs, env, step) end
    end
    
    results = Dict{Symbol, NamedTuple}()
    for rs in states
        results[rs.strategy] = (success = rs.reached, steps = rs.reached ? rs.reached_step : max_steps)
    end
    return results
end

function run_batch_simulations(n_runs::Int = 100, filename::String = "simulation_results.csv")
    println("Starting fast batch simulation of $n_runs environments (No GUI)...")
    df = DataFrame(RunID = Int[], Seed = UInt64[], Strategy = String[], Success = Bool[], Steps = Int[])
    
    for run_id in 1:n_runs
        if run_id % 20 == 0 println("  Completed $run_id / $n_runs...") end
        current_seed = abs(rand(UInt64))
        Random.seed!(current_seed)
        env = generate_random_env()
        sim_results = run_single_batch_sim(env)
        for (strat, res) in sim_results
            push!(df, (run_id, current_seed, String(strat), res.success, res.steps))
        end
        Random.seed!() 
    end
    
    CSV.write(filename, df)
    println("Batch simulation finished. Results saved to $filename")
end

# ======================== GLMakie Replay Viewer (GUI) ========================

function state_display_label(t::TrajectoryPoint)
    if t.reached return "★ Arrived"
    elseif t.anti_pirate_mode == :Paused
        return t.confirm_counter > 0 ? @sprintf("⏸ Confirm %d/%d", t.confirm_counter, PIRATE_CONFIRM_STEPS) : "⏸ Paused"
    elseif t.anti_pirate_mode == :InertialMode return "⤇ Inertial"
    elseif t.state == :MoveToTower return "→ Moving"
    else return "~ Following" end
end

function run_viewer_simulation(env::SimulationEnv, max_steps::Int = 3000)
    strategies = [:Naive, :Cautious, :Inertial]
    states = RobotState[]
    
    for strat in strategies
        robot = Robot(Vec2(2.0, 1.0), 0.0, :MoveToTower, 0.0)
        u_ori!(robot, env, 0)
        init_i = perceived_intensity(robot.pos, env, 0)
        push!(states, RobotState(
            robot, strat, FollowState(init_i, init_i, 0, false),
            init_i, init_i, :Normal, 0.0, :MoveToTower, 0, false, 0, true
        ))
    end
    
    trajectories = [TrajectoryPoint[] for _ in 1:3]
    step = 0
    for (i, rs) in enumerate(states) push!(trajectories[i], make_traj_point(rs, env, step)) end

    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for (i, rs) in enumerate(states)
            update_robot_step!(rs, env, step)
            push!(trajectories[i], make_traj_point(rs, env, step))
        end
    end
    return trajectories, [rs.reached for rs in states], [rs.reached_step for rs in states]
end

function launch_viewer(env::SimulationEnv, trajs, reached_flags, reached_steps)
    n_frames = length(trajs[1])
    strategy_names = ["R1 Naive", "R2 Cautious", "R3 Inertial"]
    robot_colors = [RGBf(0.3, 0.79, 0.94), RGBf(1.0, 0.6, 0.2), RGBf(0.3, 0.9, 0.3)]
    
    all_pts = [[Point2f(t.x, t.y) for t in traj] for traj in trajs]
    
    fig = Figure(size=(1050, 1050), backgroundcolor=RGBf(0.059, 0.059, 0.137))
    Label(fig[1, 1], " 3-Robot Adversary Tower Comparative Replay", fontsize=20, color=RGBf(0.92, 0.94, 0.96), font=:bold)
    
    ax = Axis(fig[3, 1], aspect=DataAspect(), backgroundcolor=RGBf(0.086, 0.094, 0.165),
              xgridvisible=false, ygridvisible=false, xticksvisible=false, yticksvisible=false, 
              xticklabelsvisible=false, yticklabelsvisible=false)
    xlims!(ax, -1.0, 21.0); ylims!(ax, -1.0, 21.0)
    deactivate_interaction!(ax, :scrollzoom); deactivate_interaction!(ax, :rectanglezoom)

    controls = GridLayout(fig[4, 1], tellwidth=false)
    sl = Slider(fig[5, 1], range=1:n_frames, startvalue=1, color_active=RGBf(0.26, 0.36, 0.90), linewidth=14)
    frame_idx = sl.value
    speed_val = Observable(1.0); is_playing = Observable(false)
    timer_ref = Ref{Union{Timer,Nothing}}(nothing)
    programmatic_update = Ref(false)

    info_panel = GridLayout(fig[2, 1], tellwidth=false)
    common_info = @lift @sprintf("Step %d / %d   |   Adversary: %s", trajs[1][$frame_idx].step, n_frames - 1, trajs[1][$frame_idx].pirate_active ? "ON" : "OFF")
    Label(info_panel[1, 1], common_info, fontsize=12, color=RGBf(0.60, 0.64, 0.76))

    for i in 1:3
        let traj=trajs[i], col=robot_colors[i], name=strategy_names[i], idx=i
            info_obs = @lift begin
                t = traj[$frame_idx]
                reached_str = reached_flags[idx] && t.reached ? "   ✓ Step $(reached_steps[idx])" : ""
                state_str = String(t.anti_pirate_mode) != "Normal" ? String(t.anti_pirate_mode) : String(t.state)
                @sprintf("%s: %s | I = %.3f | Hi = %.3f %s", name, state_str, min(t.intensity_val, 99.9), min(t.high_intensity_memory, 99.9), reached_str)
            end
            Label(info_panel[1+i, 1], info_obs, fontsize=11, color=col)
        end
    end

    btn_play = Button(controls[1, 1], label=@lift($is_playing ? "|| Pause" : "> Play"), width=100)
    btn_1x = Button(controls[1, 2], label="1x"); btn_5x = Button(controls[1, 3], label="5x")
    
    poly!(ax, Rect(0, 0, 20.0, 20.0), color=RGBf(0.086, 0.094, 0.165), strokecolor=RGBf(0.30, 0.32, 0.45), strokewidth=2)
    for obs in env.obstacles poly!(ax, Rect(obs.xmin, obs.ymin, obs.xmax - obs.xmin, obs.ymax - obs.ymin), color=RGBf(0.27, 0.29, 0.40)) end
    
    scatter!(ax, [env.tower.x], [env.tower.y], marker=:star5, markersize=26, color=RGBf(1.0, 0.40, 0.40))
    scatter!(ax, [env.pirate_tower.x], [env.pirate_tower.y], marker=:star5, markersize=26, color=RGBf(0.0, 1.0, 0.4))
    scatter!(ax, [env.pirate_tower.x], [env.pirate_tower.y], markersize=80, marker=:circle, color=RGBAf(0.0, 1.0, 0.4, 0.18), visible=@lift(trajs[1][$frame_idx].pirate_active))

    for i in 1:3
        let pts=all_pts[i], col=robot_colors[i]
            lines!(ax, @lift(pts[1:$frame_idx]), color=RGBAf(col.r, col.g, col.b, 0.35), linewidth=2)
        end
    end

    for i in 1:3
        let traj=trajs[i], col=robot_colors[i], r_size=ROBOT_SIZE
            scatter!(ax, @lift([traj[$frame_idx].x]), @lift([traj[$frame_idx].y]), markersize=28,
                     color=@lift(RGBAf(col.r, col.g, col.b, traj[$frame_idx].reached ? 0.05 : 0.15)))

            robot_verts = @lift begin
                t = traj[$frame_idx]
                h = r_size / 2 
                ca, sa = cos(t.angle), sin(t.angle)
                Point2f[(t.x + cx * ca - cy * sa, t.y + cx * sa + cy * ca) for (cx, cy) in ((-h, -h), (h, -h), (h, h), (-h, h))]
            end
            body_col = @lift begin
                if traj[$frame_idx].reached RGBAf(col.r, col.g, col.b, 0.35)
                elseif String(traj[$frame_idx].anti_pirate_mode) == "Paused" RGBAf(col.r, col.g, col.b, 0.55)
                else RGBAf(col.r, col.g, col.b, 1.0) end
            end
            poly!(ax, robot_verts, color=body_col, strokecolor=:white, strokewidth=1.5)

            lines!(ax, @lift([traj[$frame_idx].x, traj[$frame_idx].x + (r_size * 0.55) * cos(traj[$frame_idx].angle)]),
                       @lift([traj[$frame_idx].y, traj[$frame_idx].y + (r_size * 0.55) * sin(traj[$frame_idx].angle)]), color=:white, linewidth=1.5)

            scatter!(ax, @lift([traj[$frame_idx].x + (r_size / 2) * cos(traj[$frame_idx].angle)]),
                         @lift([traj[$frame_idx].y + (r_size / 2) * sin(traj[$frame_idx].angle)]),
                     markersize=7, color=@lift(traj[$frame_idx].front_sensor ? RGBf(0.10, 0.92, 0.30) : RGBf(0.40, 0.42, 0.48)), strokecolor=:white, strokewidth=0.5)

            scatter!(ax, @lift([traj[$frame_idx].x - (r_size / 2) * sin(traj[$frame_idx].angle)]),
                         @lift([traj[$frame_idx].y + (r_size / 2) * cos(traj[$frame_idx].angle)]),
                     markersize=7, color=@lift(traj[$frame_idx].left_sensor ? RGBf(0.10, 0.92, 0.30) : RGBf(0.40, 0.42, 0.48)), strokecolor=:white, strokewidth=0.5)
        end
    end

    banner_vis = @lift $frame_idx >= n_frames
    text!(ax, Point2f(10.0, 12.8), text="Simulation Complete", fontsize=22, color=RGBAf(1.0, 0.84, 0.0, 0.92), align=(:center, :center), font=:bold, visible=banner_vis)

    function stop_p!() if timer_ref[] !== nothing close(timer_ref[]); timer_ref[] = nothing end end
    function start_p!()
        stop_p!()
        curr_f = Float64(frame_idx[])
        timer_ref[] = Timer(0.033; interval=0.033) do t
            curr_f = min(curr_f + speed_val[], n_frames)
            nxt = round(Int, curr_f)
            if nxt > frame_idx[]
                programmatic_update[] = true; set_close_to!(sl, nxt); programmatic_update[] = false
            end
            if nxt >= n_frames is_playing[] = false; stop_p!() end
        end
    end

    on(btn_play.clicks) do _
        if is_playing[] is_playing[]=false; stop_p!()
        else 
            if frame_idx[] >= n_frames programmatic_update[]=true; set_close_to!(sl,1); programmatic_update[]=false end
            is_playing[]=true; start_p!()
        end
    end
    on(btn_1x.clicks) do _ speed_val[] = 1.0 end
    on(btn_5x.clicks) do _ speed_val[] = 5.0 end
    on(frame_idx) do _ if !programmatic_update[] && is_playing[] is_playing[]=false; stop_p!() end end

    rowsize!(fig.layout, 1, Fixed(32)); rowsize!(fig.layout, 2, Fixed(100))
    display(fig)
    return fig
end

function watch_replay(run_id::Int, filename::String = "simulation_results.csv")
    if !isfile(filename)
        println("Error: CSV file not found. Run batch simulation first.")
        return
    end
    df = CSV.read(filename, DataFrame)
    target_rows = df[df.RunID .== run_id, :]
    if nrow(target_rows) == 0
        println("Error: RunID $run_id not found in $filename.")
        return
    end
    
    target_seed = UInt64(target_rows.Seed[1])
    println("Found RunID $run_id. Reconstructing environment from Seed...")
    
    Random.seed!(target_seed)
    env = generate_random_env()
    
    println("Running Simulation Viewer...")
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, 3000)
    
    Random.seed!() 
    launch_viewer(env, trajs, reached_flags, reached_steps)
end

# ============================================================================
# EXECUTION COMMANDS (Uncomment exactly ONE at a time)
# ============================================================================

# --- COMMAND 1: RUN BATCH (Collect Data to CSV quickly without UI) ---
 #run_batch_simulations(10)

# --- COMMAND 2: REPLAY A SPECIFIC RUN FROM THE CSV (Shows UI) ---
 watch_replay(3)