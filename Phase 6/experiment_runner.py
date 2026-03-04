import multiprocessing
import os
from time import perf_counter
# Import the functions from your other file
from core import run_experiment_job, DEFAULT_CONFIG

def define_experiments():
    """
    Define all the experimental configurations you want to run.
    Each dict will be passed to one process.
    """
    
    # A list to hold all job configuration packages
    all_experiments = []
    
    # Total generations for each experiment
    total_gens = 100

    # Experiment 1: Baseline (using the default config)
    all_experiments.append({
        "name": "Baseline",
        "generations": total_gens,
        "config": DEFAULT_CONFIG
    })

    # Experiment 2: High Mutation
    config_high_mut = DEFAULT_CONFIG.copy()
    config_high_mut["mutation_rate"] = 0.2
    config_high_mut["mutation_strength"] = 0.8
    all_experiments.append({
        "name": "High_Mutation",
        "generations": total_gens,
        "config": config_high_mut
    })
    
    # Experiment 3: Low Mutation
    config_low_mut = DEFAULT_CONFIG.copy()
    config_low_mut["mutation_rate"] = 0.05
    config_low_mut["mutation_strength"] = 0.3
    all_experiments.append({
        "name": "Low_Mutation",
        "generations": total_gens,
        "config": config_low_mut
    })

    # Experiment 4: Larger Population
    config_large_pop = DEFAULT_CONFIG.copy()
    config_large_pop["creature_pop_size"] = 100
    all_experiments.append({
        "name": "Large_Population",
        "generations": total_gens,
        "config": config_large_pop
    })
    
    # Experiment 5: More Enemies
    config_more_enemies = DEFAULT_CONFIG.copy()
    config_more_enemies["enemy_pop_size"] = 8
    all_experiments.append({
        "name": "More_Enemies",
        "generations": total_gens,
        "config": config_more_enemies
    })
    
    # Experiment 6: Higher Energy Cost
    config_high_cost = DEFAULT_CONFIG.copy()
    config_high_cost["energy_decay_rate"] = 0.2
    all_experiments.append({
        "name": "High_Energy_Cost",
        "generations": total_gens,
        "config": config_high_cost
    })
    
    # Add more experiments as needed...

    return all_experiments

if __name__ == "__main__":
    # This __name__ == "__main__" check is ESSENTIAL for multiprocessing
    
    # 1. Define all the jobs to run
    experiments_to_run = define_experiments()

    # 2. Set number of processes
    # For your M1 Mac, 4 is the ideal number to use the high-performance cores.
    NUM_PROCESSES = 4
    
    print(f"--- Starting {len(experiments_to_run)} experiments using {NUM_PROCESSES} parallel processes ---")
    
    # 3. Create the process pool and run the jobs
    start_time = perf_counter()
    
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # pool.map distributes the 'experiments_to_run' list
        # to the 'run_experiment_job' function across 4 processes.
        results = pool.map(run_experiment_job, experiments_to_run)
    
    end_time = perf_counter()
    
    # 4. All jobs are done
    print("--- All experiments have completed ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Results:")
    for result in results:
        print(f"  - {result}")
