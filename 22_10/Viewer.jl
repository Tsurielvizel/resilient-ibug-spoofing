# ============================================================================
# Viewer.jl - מערכת תצוגה גרפית אינטראקטיבית ו-Replay
# ============================================================================

using GLMakie
using GLMakie: xlims!, ylims!
using Printf
using DataFrames
using CSV
using Random

function state_display_label(t::TrajectoryPoint)
    if t.reached
        return "★ Arrived"
    elseif t.anti_enemy_mode == :Paused
        return t.confirm_counter > 0 ? @sprintf("⏸ Confirm %d/%d", t.confirm_counter, ENEMY_CONFIRM_STEPS) : "⏸ Paused"
    elseif t.anti_enemy_mode == :InertialMode
        return "⤇ Inertial"
    elseif t.state == :MoveToTower
        return "→ Moving"
    else
        return "~ Following"
    end
end

function run_viewer_simulation(env::SimulationEnv, max_steps::Int=300)
    strategies = [:Naive, :Cautious, :Inertial]
    states = RobotState[]

    for strat in strategies
        robot = Robot(Vec2(2.0, 1.0), 0.0, :MoveToTower, 0.0)
        u_ori!(robot, env, 0)
        init_i = perceived_intensity(robot.pos, env, 0)
        push!(states, RobotState(
            robot, strat, FollowState(init_i, init_i, init_i, 0, false, false),
            init_i, init_i, :Normal, 0.0, :MoveToTower, 0, false, 0, false, :u_ori
        ))
    end

    trajectories = [TrajectoryPoint[] for _ in 1:3]
    step = 0
    for (i, rs) in enumerate(states)
        push!(trajectories[i], make_traj_point(rs, env, step))
    end

    while step < max_steps && !all(rs -> rs.reached, states)
        step += 1
        for (i, rs) in enumerate(states)
            update_robot_step!(rs, env, step)
            push!(trajectories[i], make_traj_point(rs, env, step))
        end
    end
    return trajectories, [rs.reached for rs in states], [rs.reached_step for rs in states]
end

function launch_viewer_navigator(filename::String="simulation_results.csv", start_run_id::Int=1)
    if !isfile(filename)
        println("Error: $filename not found.")
        return
    end

    println("Loading run index from $filename...")
    df_index = CSV.read(filename, DataFrame; 
                        select=[:RunID, :Seed], 
                        types=Dict(:RunID => Int, :Seed => String))
    unique_runs = unique(df_index, :RunID)
    
    seed_dict = Dict{Int, UInt64}()
    for row in eachrow(unique_runs)
        seed_dict[row.RunID] = parse(UInt64, string(row.Seed))
    end
    total_available = length(seed_dict)
    println("Loaded $total_available runs successfully!")

    current_run_id = Observable(start_run_id)

    function compute_sim(rid::Int)
        seed = get(seed_dict, rid, seed_dict[1])
        Random.seed!(seed)
        env = generate_random_env()
        trajs, reached_flags, reached_steps = run_viewer_simulation(env, 600)
        Random.seed!()
        return env, trajs, reached_flags, reached_steps
    end

    # חישוב ראשוני
    env_init, trajs_init, reached_flags_init, reached_steps_init = compute_sim(start_run_id)

    # יצירת ה-Observables
    env_obs = Observable(env_init)
    trajs_obs = Observable(trajs_init)
    reached_flags_obs = Observable(reached_flags_init)
    reached_steps_obs = Observable(reached_steps_init)
    n_frames = Observable(length(trajs_init[1]))

    fig = Figure(size=(1050, 1100), backgroundcolor=RGBf(0.059, 0.059, 0.137))
    
    title_obs = @lift @sprintf("3-Robot Comparative Replay — RunID: %d / %d", $current_run_id, total_available)
    Label(fig[1, 1], title_obs, fontsize=20, color=RGBf(0.92, 0.94, 0.96), font=:bold)

    # שורת כפתורי ניווט בין הרצות בראש החלון
    nav_grid = GridLayout(fig[2, 1], tellwidth=false)
    btn_prev_run = Button(nav_grid[1, 1], label="◀◀ Prev Run", width=120)
    tb_run_input = Textbox(nav_grid[1, 2], placeholder="Jump to ID...", width=140)
    btn_jump_run = Button(nav_grid[1, 3], label="Go", width=50)
    btn_next_run = Button(nav_grid[1, 4], label="Next Run ▶▶", width=120)
    colgap!(nav_grid, 10)

    # פאנל מידע
    info_panel = GridLayout(fig[3, 1], tellwidth=false)
    sl = Slider(fig[6, 1], range=@lift(1:$n_frames), startvalue=1, color_active=RGBf(0.26, 0.36, 0.90), linewidth=14)
    frame_idx = sl.value
    speed_val = Observable(1.0)
    is_playing = Observable(false)

    common_info = @lift @sprintf("Step %d / %d   |   Adversary: %s", 
        $trajs_obs[1][$frame_idx].step, $n_frames - 1, $trajs_obs[1][$frame_idx].enemy_active ? "ON" : "OFF")
    Label(info_panel[1, 1], common_info, fontsize=12, color=RGBf(0.60, 0.64, 0.76))

    strategy_names = ["R1 Naive", "R2 Cautious", "R3 Inertial"]
    robot_colors = [RGBf(0.3, 0.79, 0.94), RGBf(1.0, 0.6, 0.2), RGBf(0.3, 0.9, 0.3)]

    for i in 1:3
        let col = robot_colors[i], name = strategy_names[i], idx = i
            info_obs = @lift begin
                t = $trajs_obs[idx][$frame_idx]
                r_flags = $reached_flags_obs
                r_steps = $reached_steps_obs
                reached_str = r_flags[idx] && t.reached ? "   ✓ Step $(r_steps[idx])" : ""
                combat_mode = t.anti_enemy_mode == :Normal ? "Regular" : String(t.anti_enemy_mode)
                prim_action = String(t.action)
                @sprintf("%s | Mode: %-8s | Action: %-7s | I = %.3f | Hi = %.3f %s",
                    name, combat_mode, prim_action, min(t.intensity_val, 99.9), min(t.high_intensity_memory, 99.9), reached_str)
            end
            Label(info_panel[1+i, 1], info_obs, fontsize=13, color=col, font=:bold)
        end
    end

    # Axis מרכזי
    ax = Axis(fig[4, 1], aspect=DataAspect(), backgroundcolor=RGBf(0.086, 0.094, 0.165),
        xgridvisible=false, ygridvisible=false, xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false)
    xlims!(ax, -1.0, 21.0)
    ylims!(ax, -1.0, 21.0)
    deactivate_interaction!(ax, :scrollzoom)
    deactivate_interaction!(ax, :rectanglezoom)

    # גבולות המפה
    poly!(ax, Rect(0, 0, 20.0, 20.0), color=RGBf(0.086, 0.094, 0.165), strokecolor=RGBf(0.30, 0.32, 0.45), strokewidth=2)

    # מכשולים
    obs_rects = @lift [Rect(o.xmin, o.ymin, o.xmax - o.xmin, o.ymax - o.ymin) for o in $env_obs.obstacles]
    poly!(ax, obs_rects, color=RGBf(0.27, 0.29, 0.40))

    # מגדלים
    tower_pt = @lift [Point2f($env_obs.tower.x, $env_obs.tower.y)]
    scatter!(ax, tower_pt, marker=:star5, markersize=26, color=RGBf(1.0, 0.40, 0.40))

    enemy_pt = @lift [Point2f($env_obs.enemy_tower.x, $env_obs.enemy_tower.y)]
    scatter!(ax, enemy_pt, marker=:star5, markersize=26, color=RGBf(0.0, 1.0, 0.4))
    enemy_pulse_vis = @lift $trajs_obs[1][$frame_idx].enemy_active
    scatter!(ax, enemy_pt, markersize=80, marker=:circle, color=RGBAf(0.0, 1.0, 0.4, 0.18), visible=enemy_pulse_vis)

    # מסלולי הרובוטים
    for i in 1:3
        let col = robot_colors[i], idx = i
            pts = @lift [Point2f(t.x, t.y) for t in $trajs_obs[idx][1:$frame_idx]]
            lines!(ax, pts, color=RGBAf(col.r, col.g, col.b, 0.35), linewidth=2)
        end
    end

    # גופי הרובוטים
    r_size = ROBOT_SIZE
    for i in 1:3
        let col = robot_colors[i], idx = i
            robot_verts = @lift begin
                t = $trajs_obs[idx][$frame_idx]
                h = r_size / 2
                ca, sa = cos(t.angle), sin(t.angle)
                Point2f[(t.x + cx * ca - cy * sa, t.y + cx * sa + cy * ca) for (cx, cy) in ((-h, -h), (h, -h), (h, h), (-h, h))]
            end
            body_col = @lift $trajs_obs[idx][$frame_idx].reached ? RGBAf(col.r, col.g, col.b, 0.35) : RGBAf(col.r, col.g, col.b, 1.0)
            poly!(ax, robot_verts, color=body_col, strokecolor=:white, strokewidth=1.5)

            heading_x = @lift [$trajs_obs[idx][$frame_idx].x, $trajs_obs[idx][$frame_idx].x + (r_size * 0.55) * cos($trajs_obs[idx][$frame_idx].angle)]
            heading_y = @lift [$trajs_obs[idx][$frame_idx].y, $trajs_obs[idx][$frame_idx].y + (r_size * 0.55) * sin($trajs_obs[idx][$frame_idx].angle)]
            lines!(ax, heading_x, heading_y, color=:white, linewidth=1.5)
        end
    end

    # פקדי תנועה
    controls = GridLayout(fig[5, 1], tellwidth=false)
    btn_prev = Button(controls[1, 1], label="◀ -1 Step", width=90)
    btn_play = Button(controls[1, 2], label=@lift($is_playing ? "|| Pause" : "> Play"), width=90)
    btn_next = Button(controls[1, 3], label="+1 Step ▶", width=90)
    btn_1x = Button(controls[1, 4], label="1x", width=50)
    btn_5x = Button(controls[1, 5], label="5x", width=50)

    on(btn_play.clicks) do _
        is_playing[] = !is_playing[]
    end
    on(btn_next.clicks) do _
        sl.value[] = min(n_frames[], sl.value[] + 1)
    end
    on(btn_prev.clicks) do _
        sl.value[] = max(1, sl.value[] - 1)
    end
    on(btn_1x.clicks) do _
        speed_val[] = 1.0
    end
    on(btn_5x.clicks) do _
        speed_val[] = 5.0
    end

    # פונקציית טעינת הרצה חדשה ישירות ל-Observables
    function switch_to_run(rid::Int)
        is_playing[] = false
        env, trajs, reached_flags, reached_steps = compute_sim(rid)
        current_run_id[] = rid
        env_obs[] = env
        trajs_obs[] = trajs
        reached_flags_obs[] = reached_flags
        reached_steps_obs[] = reached_steps
        n_frames[] = length(trajs[1])
        set_close_to!(sl, 1)
    end

    on(btn_next_run.clicks) do _
        if current_run_id[] < total_available
            switch_to_run(current_run_id[] + 1)
        end
    end

    on(btn_prev_run.clicks) do _
        if current_run_id[] > 1
            switch_to_run(current_run_id[] - 1)
        end
    end

    on(btn_jump_run.clicks) do _
        raw_text = tb_run_input.displayed_string[]
        
        # נוודא שקיבלנו מחרוזת ושהיא אינה ריקה לפני ההמרה
        if raw_text !== nothing && !isempty(strip(raw_text))
            val = tryparse(Int, strip(raw_text))
            if val !== nothing && 1 <= val <= total_available
                switch_to_run(val)
                tb_run_input.displayed_string[] = "" # מנקה את התיבה אחרי הקפיצה
            end
        end
    end

    rowsize!(fig.layout, 1, Fixed(32))
    rowsize!(fig.layout, 2, Fixed(45))
    
    screen = display(fig)
    
    
    @async while events(fig).window_open[]
        if is_playing[]
            if sl.value[] < n_frames[]
                
                set_close_to!(sl, sl.value[] + 1)
            else
                is_playing[] = false
            end
        end
        sleep(0.05 / speed_val[]) 
    end
    # ----------------------------------------------------
    
    wait(screen) 
    
    return fig
end

function watch_replay(run_id::Int, filename::String="simulation_results.csv")
    if !isfile(filename)
        println("Error: CSV file not found. Run batch simulation first.")
        return
    end
    df = CSV.read(filename, DataFrame; types=Dict(:Seed => String, :RunID => Int))
    target_rows = df[df.RunID.==run_id, :]
    if nrow(target_rows) == 0
        println("Error: RunID $run_id not found in $filename.")
        return
    end

    target_seed = parse(UInt64, target_rows.Seed[1])
    println("Found RunID $run_id. Reconstructing environment from Seed...")

    Random.seed!(target_seed)
    env = generate_random_env()

    println("Running Simulation Viewer...")
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, 600)

    Random.seed!()
    launch_viewer(env, trajs, reached_flags, reached_steps)
end

function run_direct_simulation()
    println("Generating random environment and running simulation viewer...")
    env = generate_random_env()
    trajs, reached_flags, reached_steps = run_viewer_simulation(env, 600)
    launch_viewer(env, trajs, reached_flags, reached_steps)
end

# ============================================================================
# Fast Interactive Replay Navigator
# ============================================================================

function start_fast_navigator(filename::String="simulation_results.csv", start_run_id::Int=1)
    if !isfile(filename)
        println("Error: $filename not found.")
        return
    end

    println("Loading run index from $filename (reading only RunID and Seed)...")
    # הגדרת טיפוסים מפורשת כדי למנוע קריאה של Seed כמספר Float64
    df_index = CSV.read(filename, DataFrame; 
                        select=[:RunID, :Seed], 
                        types=Dict(:RunID => Int, :Seed => String))
    unique_runs = unique(df_index, :RunID)
    
    seed_dict = Dict{Int, UInt64}()
    for row in eachrow(unique_runs)
        seed_dict[row.RunID] = parse(UInt64, string(row.Seed))
    end
    println("Index loaded! Total runs available: $(length(seed_dict))")

    current_run_id = Observable(start_run_id)

    function load_env_for_id(rid::Int)
        if !haskey(seed_dict, rid)
            println("RunID $rid not found!")
            return nothing, nothing, nothing, nothing
        end
        seed = seed_dict[rid]
        Random.seed!(seed)
        env = generate_random_env()
        trajs, reached_flags, reached_steps = run_viewer_simulation(env, 600)
        Random.seed!()
        return env, trajs, reached_flags, reached_steps
    end

    println("\n=== Fast Keyboard Navigator Ready ===")
    
    while true
        rid = current_run_id[]
        env, trajs, reached_flags, reached_steps = load_env_for_id(rid)
        if env === nothing
            break
        end
        
        println("Displaying RunID: $rid ...")
        fig = launch_viewer(env, trajs, reached_flags, reached_steps)
        
        print("Enter next RunID to view (or press Enter for $(rid+1), 'q' to quit): ")
        user_input = readline()
        if lowercase(strip(user_input)) == "q"
            break
        elseif isempty(strip(user_input))
            current_run_id[] = rid + 1
        else
            parsed_id = tryparse(Int, strip(user_input))
            if parsed_id !== nothing
                current_run_id[] = parsed_id
            else
                current_run_id[] = rid + 1
            end
        end
    end
end