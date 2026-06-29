FROM ubuntu:24.04

RUN apt-get update && apt-get install -y build-essential wget curl git python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz && tar -xf zig-linux-x86_64-0.13.0.tar.xz && mv zig-linux-x86_64-0.13.0 /usr/local/zig && ln -s /usr/local/zig/zig /usr/local/bin/zig

RUN pip3 install --no-cache-dir jax jaxlib cirq numpy

WORKDIR /app
COPY . .

CMD ["python3", "build_orchestrator.py"]