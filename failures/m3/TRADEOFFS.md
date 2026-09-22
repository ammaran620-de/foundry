# M3 Measured Tradeoffs

## Thread Scaling

The INT8 CPU benchmark demonstrates that increasing thread count does not monotonically improve performance.

| Configuration | Mean | p95 | FPS |
|---|---:|---:|---:|
| 1 thread | 53.37 ms | 74.49 ms | 18.74 |
| 4 threads | 52.46 ms | 58.70 ms | 19.06 |
| 6 threads | 62.16 ms | 67.69 ms | 16.09 |
| 8 threads | 57.09 ms | 60.22 ms | 17.52 |
| 10 threads | 58.51 ms | 62.41 ms | 17.09 |
| 12 requested / 10 actual | 61.30 ms | 67.91 ms | 16.31 |

Observed tradeoff:
More requested CPU threads beyond 4 reduced throughput and increased latency on this machine.

## Classical CV Baselines

These measurements use a constrained synthetic moving-object scene.

| Method | Mean | p95 | FPS | Mean IoU |
|---|---:|---:|---:|---:|
| Background subtraction | 8.92 ms | 10.20 ms | 112.10 | 0.6512 |
| Background subtraction + morphology | 10.27 ms | 11.67 ms | 97.36 | 0.6407 |
| Optical flow | 351.59 ms | 533.06 ms | 2.84 | 0.5425 |
| Template matching | 14.86 ms | 22.78 ms | 67.29 | 1.0000 |

Observed tradeoffs:

### Background subtraction + morphology
Adding morphology increased mean latency from 8.92 ms to 10.27 ms and reduced mean IoU from 0.6512 to 0.6407 on this sequence.

### Optical flow
Optical flow produced substantially higher latency: 351.59 ms mean and 533.06 ms p95, with only 2.84 FPS on the same sequence.

### Template matching
Template matching reached perfect metrics on this deterministic synthetic scene, but the benchmark notes explicitly state that this is a constrained baseline and not a general substitute for neural object detection.

## Interpretation

These results are retained as measured engineering evidence.

They demonstrate that:
- more CPU threads are not automatically better;
- additional classical processing can add cost without improving the measured metric;
- an algorithm can achieve excellent results in a constrained benchmark without being a general solution;
- optimization decisions must be evaluated against the actual workload and failure conditions.
