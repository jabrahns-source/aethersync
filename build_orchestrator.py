#!/usr/bin/env python3
"""
Polyglot Build Orchestrator with Docker Support
"""
import subprocess
import sys
import os
import argparse

def run_cmd(cmd, cwd=None):
    print(f"→ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def build_zig():
    run_cmd("zig build-lib math_core.zig -dynamic -O ReleaseFast -femit-bin=libmath_core.so")

def build_python_deps():
    run_cmd("pip install --quiet jax jaxlib cirq numpy")

def build_sycl():
    if os.path.exists("/opt/intel/oneapi"):
        run_cmd("icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -o engine main.cpp -L. -lmath_core -lcudart")
    else:
        print("⚠️  oneAPI not found — skipping SYCL build")

def run_benchmark():
    run_cmd("python3 run_engine.py")

def build_in_docker():
    print("=== Building inside Docker ===")
    run_cmd("docker build -t latency-engine .")
    run_cmd("docker run --rm -it latency-engine")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="Build and run inside Docker")
    args = parser.parse_args()

    if args.docker:
        build_in_docker()
        return

    print("=== POLYGLOT BUILD ORCHESTRATOR ===")
    build_zig()
    build_python_deps()
    build_sycl()
    run_benchmark()
    print("\n✅ Build complete.")

if __name__ == "__main__":
    main()