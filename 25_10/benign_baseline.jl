# ============================================================================
# benign_baseline.jl - No-attack reference baseline for Table 1
#
# Re-generates every environment from the seeds stored in simulation_results.csv
# and runs it again with the adversarial transmitter disabled (u(t) = 0).
# Under no attack, the Naive strategy is exactly the benign I-Bug execution.
#
# Usage:  julia -t auto benign_baseline.jl [n_runs]
# ============================================================================

include("CoreLogic.jl")
include("BatchLogger.jl")

using CSV, DataFrames, Random, Statistics, Printf

const INPUT_CSV = "simulation_results.csv"
const OUTPUT_CSV = "benign_results.csv"
const MAX_STEPS = 2000

# Switch for the adversary. `is_enemy_active` is redefined below so that the
# unmodified CoreLogic.jl is used for all other logic.
const ATTACK_ENABLED = Ref(true)
is_enemy_active(step::Int) = ATTACK_ENABLED[] && _attack_schedule_orig(step)
_attack_schedule_orig(step::Int) = step >= Initial_Delay &&
    ((step - Initial_Delay) % ENEMY_CYCLE_BASE) < round(Int, ENEMY_DUTY_CYCLE * ENEMY_CYCLE_BASE)

function main()
    n_req = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)

    df = CSV.read(INPUT_CSV, DataFrame; types=Dict(:Seed => String, :RunID => Int))
    wide = unstack(select(df, :RunID, :Seed, :Strategy, :Steps), [:RunID, :Seed], :Strategy, :Steps)
    succ = unstack(select(df, :RunID, :Seed, :Strategy, :Success), [:RunID, :Seed], :Strategy, :Success)
    sort!(wide, :RunID); sort!(succ, :RunID)
    n = min(n_req, nrow(wide))
    println("Runs: $n  (threads: $(Threads.nthreads()))")

    # --- Reproducibility check: the attacked runs must match the stored CSV ---
    ATTACK_ENABLED[] = true
    for i in 1:min(20, n)
        Random.seed!(parse(UInt64, wide.Seed[i]))
        env = generate_random_env()
        r = run_single_batch_sim(env, MAX_STEPS)
        for s in (:Naive, :Cautious, :Inertial)
            r[s].steps == wide[i, String(s)] ||
                error("Reproducibility check failed for RunID $(wide.RunID[i]) / $s")
        end
    end
    println("Reproducibility check passed (attacked runs match $INPUT_CSV).")

    # --- Benign runs (no adversary) ---
    ATTACK_ENABLED[] = false
    b_succ = Vector{Bool}(undef, n)
    b_steps = Vector{Int}(undef, n)
    c_steps_benign = Vector{Int}(undef, n)   # Cautious under no attack (sanity: must equal Naive)
    Threads.@threads for i in 1:n
        Random.seed!(parse(UInt64, wide.Seed[i]))
        env = generate_random_env()
        r = run_single_batch_sim(env, MAX_STEPS)
        b_succ[i] = r[:Naive].success
        b_steps[i] = r[:Naive].steps
        c_steps_benign[i] = r[:Cautious].steps
    end

    out = DataFrame(RunID=wide.RunID[1:n], Seed=wide.Seed[1:n],
        BenignSuccess=b_succ, BenignSteps=b_steps,
        CautiousSteps=wide.Cautious[1:n], CautiousSuccess=succ.Cautious[1:n])
    CSV.write(OUTPUT_CSV, out)

    # --- Summary ---
    s = b_steps[b_succ]
    p = mean(b_succ)
    ci = 1.96 * sqrt(p * (1 - p) / n)
    @printf("Benign success: %.2f%% +- %.2f%%  (%d / %d)\n", 100p, 100ci, sum(b_succ), n)
    @printf("Benign steps (succ.): mean %.1f, median %.1f [%d, %d], capped mean %.1f\n",
        mean(s), median(s), round(Int, quantile(s, 0.25)), round(Int, quantile(s, 0.75)), mean(b_steps))
    println("Cautious(no attack) == Naive(no attack) in all runs: ", all(c_steps_benign .== b_steps))

    # Theorem 1 bound check on runs where both benign and attacked-Cautious succeed.
    T_on = round(Int, ENEMY_DUTY_CYCLE * ENEMY_CYCLE_BASE)
    T_off = ENEMY_CYCLE_BASE - T_on
    Nc = ENEMY_CONFIRM_STEPS
    both = b_succ .& out.CautiousSuccess
    bound = b_steps .+ cld.(b_steps, T_off - Nc) .* (T_on + Nc)
    overhead = out.CautiousSteps .- b_steps
    println("Runs where benign succeeds but attacked Cautious fails: ", sum(b_succ .& .!out.CautiousSuccess))
    println("Runs where attacked Cautious succeeds but benign fails: ", sum(.!b_succ .& out.CautiousSuccess))
    println("Eq. (8) bound satisfied in all paired successes: ", all(out.CautiousSteps[both] .<= bound[both]))
    @printf("Cautious overhead vs benign (steps): mean %.1f, max %d, min %d\n",
        mean(overhead[both]), maximum(overhead[both]), minimum(overhead[both]))
end

main()
