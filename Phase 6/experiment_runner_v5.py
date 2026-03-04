from headless_simulation_core_v5 import run_headless_simulation
import time
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

experiments = [
    {
        "name": "Baseline_Co_Evolution",
        "config": {
            # Using all defaults
        },
        "results_file": "results/baseline_co_evolution.csv"
    },
    {
        "name": "Enhanced_Enemy_Intelligence",
        "config": {
            "enemy_hidden_size": 16,
            "enemy_pop_size": 6,
            "enemy_attack_damage": 7.0
        },
        "results_file": "results/enhanced_enemies.csv"
    },
    {
        "name": "Resource_Scarcity",
        "config": {
            "max_apples": 10,
            "max_heavy_apples": 3,
            "apple_respawn_rate": 0.05
        },
        "results_file": "results/resource_scarcity.csv"
    },
    {
        "name": "Large_Populations",
        "config": {
            "creature_pop_size": 80,
            "enemy_pop_size": 8,
            "generation_time": 7000
        },
        "results_file": "results/large_populations.csv"
    },
    {
        "name": "Modified_Energy_Dynamics",
        "config": {
            "starting_energy": 150.0,
            "energy_decay_rate": 0.15,
            "gather_energy_bonus": 40.0,
            "heavy_apple_reward": 80.0
        },
        "results_file": "results/modified_energy.csv"
    },
    {
        "name": "Advanced_Neural_Architecture",
        "config": {
            "creature_hidden_size": 24,
            "enemy_hidden_size": 16,
            "mutation_rate": 0.15
        },
        "results_file": "results/advanced_neural.csv"
    }
]

# Main execution
if __name__ == "__main__":
    TOTAL_GENERATIONS = 100  # Increased from Phase 2 to allow for co-evolution
    start_time = time.time()

    print(f"Starting {len(experiments)} experiments, each running for {TOTAL_GENERATIONS} generations.")
    print("Phase 3: Co-Evolution Experiments\n")

    for i, exp in enumerate(experiments):
        print(f"\n=== Running Experiment {i+1}/{len(experiments)}: {exp['name']} ===")
        print(f"Configuration changes:")
        for key, value in exp['config'].items():
            print(f"  - {key}: {value}")

        run_headless_simulation(
            config=exp['config'],
            total_generations=TOTAL_GENERATIONS,
            results_csv_path=exp['results_file']
        )
        
        print(f"=== Completed: {exp['name']} ===")
        print(f"Results saved to: {exp['results_file']}\n")

    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"\nAll experiments completed in {hours}h {minutes}m {seconds}s")
    print("Summary of experiments:")
    for exp in experiments:
        print(f"- {exp['name']}: {exp['results_file']}")