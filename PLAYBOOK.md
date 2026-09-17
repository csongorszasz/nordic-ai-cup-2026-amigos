# Amigos playbook — Nordic AI Cup 2026

Deadline: **Sun 20 Sep, 16:00 CEST**. Per problem: unlimited validation attempts, **one** evaluation attempt, pressed by a human. Ranking is F1-style points per problem (25, 18, 15, …) summed, so a decent submission on all three beats a great one on two.

## Team workflow

- One branch per person, `main` stays the untouched template. Juan → `waffles`.
- Keep each problem's code inside its own folder so branches merge cleanly.
- Inference must run on **our own machine**. Hosted APIs are fine while building, never inside `/predict`.
- Nobody but the team presses Submit. The API key never goes into code, chats or AI tools.

## Hardware

| Machine | GPU | Use for |
|---|---|---|
| Juan's desktop | RTX 2060, 6 GB VRAM, 12 threads, 15 GB RAM | serving + light training |
| Teammate laptop | 8 GB VRAM | serving + light training |
| IDUN (NTNU) | big GPUs, SLURM jobs | heavy training / sweeps only, cannot serve |
| Azure (parked) | CPU only, 2 VMs in Belgium | fallback host; `amigos-sim` is a cheap place for survival |

**Exposing a laptop to the judges:** the evaluator must reach `http(s)://<host>/predict` from the internet. University/home networks usually block that, so use a tunnel:
- `tailscale funnel 9054` → public `https://<machine>.<tailnet>.ts.net/predict` (Tailscale already installed on Juan's machine), or
- `cloudflared tunnel --url http://localhost:9054` → temporary `https://*.trycloudflare.com` URL (changes on restart).

Test with **Verify** on the site before relying on it. The machine must stay awake and online for every validation and the evaluation attempt. Tunnels add latency, which matters for survival and drone (see below).

---

## 1. Survival simulator

**Task:** hivemind controlling all herbivore agents. Every tick you get each agent's status + observations, you return one action per agent. Survive the longest, eat fruit, don't get eaten.

**Dataset:** none. The simulator itself (`src/`, `local_playground.py`) is the environment, runs offline, seeded and deterministic on Linux. Unlimited data.

**Score:** mainly survival time; +small per fruit; −remaining energy of agents eaten. Evaluation = 3 games with fixed seeds, averaged.

**Limits:** ≤10 s per tick, and **≤600 s total waiting** across up to 30 000 ticks (≈20 ms/tick average *including network*). No GPU needed; latency is what matters.

**Key facts:** observations are fruit/agent/predator/tree (type, distance, angle) + edges (walls, vision only). Actions: `move_distance`, `move_direction`, `turn_angle`, `spawn_agent`. Walking costs 0.05/unit, sprinting 0.5/unit above `speed`, turning `|angle|/2·π`, spawning 100, living cost rises with age after 60–120 s. ⚠️ README says `move_direction` is absolute, `DTOs.py` comments say relative; check in the sim code before building on it.

### Inference pipeline
```
POST tick ──► parse StepResponse
          ──► per agent: build local world view (nearest predator / fruit / walls)
          ──► decide mode: FLEE > EAT > EXPLORE > REST
          ──► compute move_direction/distance, turn (avoid sprint unless fleeing)
          ──► spawn? (energy high, population low, not in danger)
          ──► list[ActionRequest]  (must be fast: a few ms)
```

### Training / tuning pipeline (no neural net needed)
```
rule-based policy with parameters (flee radius, spawn energy threshold, …)
  ──► run N headless sims per parameter set (local_playground, verbose=False, many seeds)
  ──► grid / random / evolutionary search over parameters (CPU, parallel; IDUN CPU jobs if needed)
  ──► pick best mean score across seeds ──► freeze into policy
```
Optional later: imitation or RL on top of the rule policy, only if the rules plateau.

---

## 2. Drone flyby

**Task:** from a drone at 600 m, detect 16 object classes in a 4K frame while only ever receiving a 960x540 crop, and steer the camera (zoom level 0/1/2 + centre) for the next frame.

**Dataset:** `drone-flyby/src/helsinki/` (406 MB), **25 frames** at 3840x2160 with JSON boxes in 4K pixels. 16 objects, **one instance per class**, 259 boxes total (8–12 per frame). Box size at 4K: 9–189 px wide, median ~53 px, i.e. **~13 px at zoom level 0**. Validation sequence (249 frames) may be recorded when we run it, but has no labels.

**Score:** COCO mAP@0.5, macro over classes, per full frame. Skipped frames = zero detections. No NMS by the scorer.

**Limits:** a frame is emitted every **333 ms**; slower answers make us skip frames. 3333 ms hard cap per request. Any invalid box/field zeroes the whole frame.

### Inference pipeline
```
POST frame ──► decode base64 PNG (960x540) + view geometry
           ──► detector (small YOLO, GPU, FP16) on the crop
           ──► map boxes view → global with utils.view_bbox_to_global
           ──► update world memory: merge with objects seen in previous frames,
               shift them by the drone motion (~13.9 m/frame), decay confidence
           ──► dedupe (own NMS), clip, validate (utils.validate_response)
           ──► camera policy: pick next level/centre within camera_constraints
               (sweep unexplored areas at L1, zoom L2 on low-confidence detections)
           ──► response (int camera coords, echo request_id + frame)
```

### Training pipeline
```
25 labelled 4K frames
  ──► hold out ~5 frames for local validation (by frame index, not random)
  ──► generate crops exactly like the evaluator: L0 (4K→960x540), L1 (1920x1080→½), L2 (native 960x540)
      at many random centres  → YOLO-format images + labels
  ──► augment: copy-paste object crops onto other backgrounds, flips, rotations, scale, colour jitter
  ──► fine-tune pretrained YOLO (n/s size) — fits in 6–8 GB; bigger runs on IDUN
  ──► test with local_evaluator.py (default = accuracy, --realtime = real score)
  ──► optional: pseudo-label recorded validation frames with the model, retrain
```

---

## 3. Medical appointment

**Task:** one MP3 consultation (English, 1–3.5 min, both speakers on one channel) + 10 yes/no questions. Return 10 booleans and, for each yes, the start/end second of the supporting passage.

**Dataset:** `medical-appointment/data/` (74 MB): **39 MP3s + `question_train.csv`**, 390 questions. Columns: `question_id, transcript_id, question, answer, label, question_type, evidence_start, evidence_end`. Types: 195 positive, 142 hard_negative, 53 off_topic. Evidence spans are short: **median 2.9 s, max 14 s**. No transcripts provided.

**Score:** `0.4 × accuracy + 0.6 × mean tIoU` (tIoU counts only for true yes answers). Yes/no is exactly balanced. Validation = 19 conversations, evaluation = 38.

**Limits:** 60 s per conversation *on average over the whole attempt*; 5 consecutive timeouts ends the attempt. No cloud APIs in the request path.

### Inference pipeline
```
POST ──► decode MP3
     ──► ASR with word timestamps (faster-whisper, GPU, e.g. large-v3-turbo / distil, float16)
     ──► segment transcript into short sentences with [start, end]
     ──► for each question:
           retrieve top-k candidate sentences (embeddings / BM25)
           judge yes/no against candidates with a local LLM or NLI model,
             with explicit number/unit/duration/body-part checks for hard negatives
     ──► balance prior: rank by yes-confidence, lean towards ~5 yes out of 10
     ──► for yes: evidence = tightest supporting sentence span (+ small tuned padding)
         for no: null, null
     ──► validate (exactly 10 of each list) ──► response
     (always return something before ~50 s: fallback answers if anything is slow)
```

### Training / tuning pipeline (mostly pretrained models)
```
39 labelled conversations
  ──► transcribe all once offline with the best Whisper you can (cache it)
  ──► measure: ASR model × answer model × prompt → accuracy and tIoU on the 390 questions
  ──► tune: retrieval k, span padding, yes-threshold, balance rule
  ──► optional: fine-tune a small cross-encoder/NLI on (question, sentence) pairs
      built from evidence spans (positives) and hard negatives, only if prompting plateaus
  ──► check VRAM: ASR + LLM together must fit (6 GB here, 8 GB on the laptop),
      and time per conversation stays well under 60 s
```

---

## Priorities

1. Pick which machine serves what and get a tunnel working; **Verify** each endpoint on the site.
2. One validation attempt per problem with the baselines, to prove the network path (and check survival's 600 s budget over the tunnel).
3. Build in parallel: survival rules + parameter search, medical ASR + retrieval + LLM, drone crop dataset + YOLO fine-tune.
4. Iterate on validation; freeze, re-validate the exact build, then a human submits well before Sunday 16:00.
