# AetherSync Benchmark Harness

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

@jit
def tpu_batch_compensation(player_positions, event_positions, intensities, latencies):
    distances = jnp.linalg.norm(player_positions[:, None] - event_positions[None, :], axis=-1)
    delays = distances / 343.0
    att = intensities[None, :] / (distances**2 + 1e-6)
    return delays, att

def run_heavy_benchmark():
    print('=== Heavy AetherSync Benchmark (128 players, 5000 events) ===')
    num_players = 128
    num_events = 5000
    num_ticks = 100
    
    np.random.seed(42)
    players = jnp.array(np.random.rand(num_players, 3).astype(np.float32))
    events = jnp.array(np.random.rand(num_events, 3).astype(np.float32))
    intensities = jnp.ones(num_events, dtype=jnp.float32)
    latencies = jnp.array(np.random.uniform(15, 110, num_players).astype(np.float32))
    
    start = time.time()
    for _ in range(num_ticks):
        _ = tpu_batch_compensation(players, events, intensities, latencies)
    jax.block_until_ready(_)
    duration = (time.time() - start) * 1000
    
    avg_per_tick = duration / num_ticks
    print(f'Avg per tick: {avg_per_tick:.2f} ms')
    print('Benchmark complete.')

if __name__ == "__main__":
    run_heavy_benchmark()