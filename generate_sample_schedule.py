import json
import random

def generate_random_schedule(output_path="sample_cache_schedule.json"):
    # SD3.5 has 38 transformer blocks
    layer_names = [f"transformer_blocks.{i}" for i in range(38)]
    
    # Simulate a range of timesteps from 1000 to 0
    # In a real run with 15 steps, only 15 specific values will be hit,
    # but providing a broad range ensures the schedule is useful.
    timesteps = [str(t) for t in range(0, 1001, 1)] 

    schedule = {}
    
    for ts in timesteps:
        schedule[ts] = {}
        # Randomly decide if this timestep should have any caching at all
        if random.random() > 0.2: # 80% of timesteps have some caching
            for layer in layer_names:
                # 30% chance for each layer to be cached at this timestep
                schedule[ts][layer] = random.random() < 0.3
        else:
            # 20% of timesteps perform full computation for all layers
            for layer in layer_names:
                schedule[ts][layer] = False

    with open(output_path, "w") as f:
        json.dump(schedule, f, indent=2)
    
    print(f"Generated sample schedule with {len(timesteps)} timesteps to {output_path}")

if __name__ == "__main__":
    generate_random_schedule()
