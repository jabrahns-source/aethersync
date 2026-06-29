const std = @import("std");

pub const LatencyTracker = struct {
    ewma_ms: f32 = 50.0,
    jitter_ms: f32 = 0.0,
    alpha: f32 = 0.18,
    last_measured: f32 = 50.0,
    predicted_next: f32 = 50.0,

    pub fn update(self: *LatencyTracker, measured: f32) void {
        if (measured <= 0) return;
        const err = @abs(measured - self.ewma_ms);
        self.jitter_ms = 0.25 * err + 0.75 * self.jitter_ms;
        const cr = @abs(measured - self.last_measured);
        var da = 0.18 * (1.0 + cr / 25.0);
        da = std.math.clamp(da, 0.06, 0.42);
        self.alpha = da;
        self.predicted_next = 0.7 * self.ewma_ms + 0.3 * measured;
        self.ewma_ms = da * measured + (1.0 - da) * self.ewma_ms;
        self.last_measured = measured;
    }
};

pub fn soundDelayNs(dist: f32) u64 {
    return @intFromFloat(dist / 343.0 * 1_000_000_000.0);
}

pub fn attenuation(intensity: f32, dist: f32) f32 {
    return intensity / (dist * dist + 1e-6);
}