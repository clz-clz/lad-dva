# 058 Qwen timeout investigation — 2026-09-13

Status: historical thinking-mode evidence only; superseded by the confirmed
no-thinking recovery plan. The200-row gate did not pass. This is not a
completed Stage A report.

The observations below were collected before `QWEN_ENABLE_THINKING=false` was
implemented. They remain useful for locating the old failure, but must not be
copied into a new no-thinking cache or used as proof that the new mode works.

## Verified environment

- A800 80GB, vLLM 0.28.0; response fingerprint `vllm-0.28.0-9f60fde2`.
- Qwen/Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`.
- Independent loopback tunnel8058 and fresh058 service artifacts; pinned SSH
  fingerprint verified. No026 service or billing operation was performed.
- Three live schema capability probes passed after the fresh service became ready.

## Actual failed request and controlled observations

Original120/600 provider/runner configuration, max20 provider calls, two SDK
retries: msra/BT/seed13 **zero-based row2**, requiring372 tags, exhausted all
three120-second attempts on Coder paths1,2,5. Logical duration was about361s.
The retained chain was `LiveBackboneError -> APITimeoutError -> ReadTimeout
-> ReadTimeout -> timeout`; no completed cache cell was published.

Path1 logical request ID: `223cc14e231148058dec5a239a8817fd`.
Allowlisted payload SHA-256:
`84d326d1ec824419f5a301dfe3502d97c99c39400b02338c02c40aface92b1c6`.
Full request bodies remain in local ignored diagnostic storage, not this repository.

| Observation | Time | Input/output tokens | Outcome |
|---|---:|---:|---|
| Exact Path1, idle service, concurrency1, no retries,120s |20.813s|10842/1146|HTTP completion, stop|
| Exact Path1, idle service, streaming observation up to600s |20.515s|10842/1124|stop; first token0.718s|
| Path1 in300/3600 cell, concurrency20 |268.515s|10842/1146|first attempt successful|
| Path2 in same cell |251.125s|10863/1120|first attempt successful|
| Path5 in same cell |251.719s|10869/1124|first attempt successful|

The streamed response contained1878 characters in the server's `reasoning`
field, with1123 tokens classified as reasoning usage. This server field can
carry schema-constrained JSON; the accounting does **not** establish that
those tokens were natural-language deliberation. Streaming output is not a
formal prediction. Failed nonstreaming attempts expose no usage, first-token
latency, or generated text; these remain unknown.

The same Path1 input/output token counts took about13 times longer in the
loaded cell than in isolation. This supports load-dependent latency for that
observation; it does not prove the cause of every timeout or exclude long
samples as a contributing factor. Other300s first-attempt timeouts on rows7
and8 completed on retry in55.859s and76.547s respectively. Do not infer their
failed-attempt output size from the successful retry.

##300/3600 diagnostic outcome

The independent1800-second executor deadline terminated the run before its
last request returned.657 logical requests completed; four HTTP attempts
timed out and recovered on the first SDK retry. One of658 logical requests
remained unresolved; no200-row cache or index was published. This was an
executor deadline, **not** exhaustion of that request's300-second SDK retries.

The outstanding request was msra/BT/seed13 row193, Coder Path5, requiring
only55 tags, with19390 prompt characters. Request ID:
`e38e927f32d3464c9c79b59022deeaf5`; payload SHA-256:
`f16f51683aaed1fd16efe87649a04604b65498f034356fd789045d7e72719840`.
While it was the only active request, server logs from11:36:36 through
11:37:56 showed approximately54–56 generated tokens/s. Thus it was continuing
to generate, rather than merely waiting for other requests. Its actual text,
final output size, and reasoning/content split are unknown. The next paid
diagnostic should replay this exact request with streaming evidence; do not
assume that it shares row2's load-dependent latency cause.

At11:38:06 and11:38:16 the server reported zero running/waiting requests and
zero KV usage, confirming the client shutdown drained the service. The058
instance itself was not shut down; request termination does not stop billing.

Full local suite:355 passed,1 skipped; compile and diff checks passed.
Independent read-only review reported no remaining Critical/Important issues.
Changes remain uncommitted pending the required live gates. No47-row smoke,
formal tag, GitHub push, or full Stage A launch was performed in this recovery.

## Recovery gates still pending

- Complete300/3600 diagnostic200-row cell (separate cache namespace).
- Fixed47-row representative smoke, final code verification, commit/push and
  new formal run tag.
- Actual cumulative spend and058 hourly rate, or an explicitly documented
  conservative estimate; use budget executor before formal Stage A.
- Final Stage A validation:45 cells, exactly200 rows each.

No automatic600/7200 escalation, output truncation, sample skipping, or
diagnostic-cache promotion has been performed. The next authorized
observation is an independent no-thinking replay of row193 Path5 followed by
row2, with no SDK retries and at most600 seconds of streaming observation.
