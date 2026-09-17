# ADR-0002: Serve the endpoint locally on the GTX 1650 behind a cloudflared tunnel

- **Status:** Accepted
- **Date:** 2026-09-17
- **Context:** medical-appointment, Nordic AI Cup 2026
- **Related:** `docs/azure-gpu-quota.md` (why not Azure)

## Context

`/predict` must be reachable from the internet and run entirely on our own
machine (no cloud inference APIs). Azure GPU was ruled out — the subscription's
GPU quota is 0 in all 33 T4 regions and raising it needs a paid support plan
(`docs/azure-gpu-quota.md`). The dev box is a WSL GTX 1650 with **4096 MiB VRAM**
and 19 GB RAM.

## Decision

Serve locally:

- **ASR:** `faster-whisper` `large-v3-turbo`, `compute_type=int8`, CUDA.
- **NLI:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` on CUDA.
- **Search:** `TOP_CLAUSES=1`, `SELECT=greedy_trim`, `MAX_CANDIDATES=120`,
  `MAX_RANGE_WORDS=10`, `τ=0.3`.
- **Exposure:** a **cloudflared named tunnel** (quick tunnels for dry runs).

## Evidence

VRAM and latency measured on the 1650 (int8, three clips ~95 s mean):

| model | RTF | 3 clips | worst clip | VRAM |
| --- | --- | --- | --- | --- |
| large-v3 | 3.6× | 77.8 s | 31.9 s | 1957 MiB |
| **large-v3-turbo** | **7.9×** | 35.6 s | 13.8 s | 1125 MiB |
| distil-large-v3 | 8.8× | 32.2 s | 12.4 s | 1061 MiB |
| medium.en | 4.6× | 61.3 s | 25.2 s | 1029 MiB |
| small.en | 7.6× | 37.2 s | 14.3 s | 389 MiB |

Chosen config, full pipeline: ASR 16.1 s mean / 21.6 s worst + NLI 19.2 s
(decision 10.5, localization 8.8) ⇒ **~35 s mean, ~41 s worst per
conversation**, **1575 MiB** resident. A validation dry run (19 conversations
via cloudflared) returned **all 200 OK, zero timeouts, mean 30.1 s, worst
46.6 s** — see `docs/tries.md` T023.

Turbo was checked against large-v3-fp16 on dose-bearing lines
(`one million IU four times daily for seven days`,
`fluconazole 50 milligrams for seven days`) with no number regressions.

## Alternatives considered

- **Azure GPU VM** — blocked (quota + paid support plan). CPU VM fallback
  possible but CPU ASR is far slower; not needed now.
- **Third-party GPU VM** (RunPod/Vast) — viable, but adds cost and moves the
  one-shot evaluation off hardware we control.
- **Quick tunnel** — fine for dry runs, but the ephemeral URL must be
  re-submitted each time; not acceptable for the one-shot evaluation.

## Consequences

- The **4 GB VRAM ceiling is binding**: `large-v3` fp16 cannot coexist with the
  NLI model, so int8 and turbo/distil are not optional.
- The named tunnel URL is stable; the machine must stay awake through the
  evaluation window.
- Latency headroom is tight (worst 46.6 s of 60 s). Freeing NLI work via a
  learned localizer (ADR-0003) is also a serving-latency win.
