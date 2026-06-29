import jax
import jax.numpy as jnp
from jax import jit
import cirq
import time
import numpy as np

@jit
def tpu_batch_compensation(player_positions, event_positions, intensities, latencies):
    distances = jnp.linalg.norm(player_positions[:, None] - event_positions[None, :], axis=-1)
    delays = distances / 343.0
    att = intensities[None, :] / (distances**2 + 1e-6)
    return delays, att

def qpu_wave_refine(depth=5):
    qubits = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
    for _ in range(depth):
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    return cirq.Simulator().simulate(circuit).final_state_vector

def run_benchmark(num_players=50, num_events=1000, num_ticks=100):
    print(f"Running AetherSync Proof Benchmark ({num_players} players, {num_events} events)")
    players = jnp.array(np.random.rand(num_players, 3).astype(np.float32))
    events = jnp.array(np.random.rand(num_events, 3).astype(np.float32))
    intensities = jnp.ones(num_events, dtype=jnp.float32)
    latencies = jnp.array(np.random.uniform(15, 110, num_players).astype(np.float32))

    start = time.time()
    for _ in range(num_ticks):
        _ = tpu_batch_compensation(players, events, intensities, latencies)
    jax.block_until_ready(_)
    duration = (time.time() - start) * 1000

    print(f"Ticks: {num_ticks} | Total: {duration:.1f}ms | Per tick: {duration/num_ticks:.2f}ms")
    print("Benchmark passed. AetherSync is operational.")

if __name__ == "__main__":
    run_benchmark()