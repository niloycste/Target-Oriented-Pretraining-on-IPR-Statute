# CPU optimization report

QA and MLM were benchmarked separately because MLM's vocabulary head has a different memory profile.
Effective batch size and every scientific training choice remained fixed.

| Stage | Micro | Accum | Threads | Workers | sec/step | Status |
|---|---:|---:|---:|---:|---:|---|
| qa | 8 | 4 | 24 | 0 | 24.990 | ok |
| qa | 16 | 2 | 24 | 0 | 22.964 | ok |
| qa | 32 | 1 | 24 | 0 | 23.012 | ok |
| qa | 16 | 2 | 8 | 0 | 32.033 | ok |
| qa | 16 | 2 | 16 | 0 | 25.354 | ok |
| qa | 16 | 2 | 32 | 0 | 28.104 | ok |
| mlm | 4 | 8 | 24 | 0 | 29.652 | ok |
| mlm | 8 | 4 | 24 | 0 | 16.493 | ok |
| mlm | 16 | 2 | 24 | 0 | 17.622 | ok |

QA recommendation: **16×2**, 24 threads, 0 workers.
MLM recommendation: **8×4**, 24 threads, 0 workers.
Measured QA speed-up versus 8×4 baseline: **1.088×**.
Projected 1,274-step SQuAD stage: **8.13 hours**.

Projected full 47-reader study: **299-353 wall hours** (**12.4-14.7 continuous days**).
This includes a 10-30% allowance beyond measured training compute for evaluation and I/O.

The final runner reads `cpu_benchmark.json` and applies the stage-specific configuration.
