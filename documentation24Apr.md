# FCD Type II Lesion Segmentation: Engineering Debug Arc

**Project:** SynthSeg-based training for FCD Type II lesion segmentation on Bonn FCD-II dataset
**Environment:** Kaggle (T4/P100, 16GB VRAM, fp16-mixed, 12h session limit)
**Framework:** PyTorch Lightning 2.4 + learn2synth + cornucopia, 3D UNet (31.6M params)
**Scope of this document:** Two entangled bugs — a checkpoint-saving pathology and a NaN weight-corruption issue — and the diagnostic process that led to resolving both.

---

## Part 1: The Checkpoint Regression Bug

### Symptom

Resuming training from a checkpoint consistently loaded an older epoch than expected:

| Run | Expected resume epoch | Actual resume epoch | Epochs lost |
|-----|----------------------|---------------------|-------------|
| 1→2 | ~111                 | ~111                | 0 (normal)  |
| 2→3 | ~223                 | 167                 | 56          |
| 3→4 | ~282                 | 192                 | 90          |

Training was happening. Epochs were advancing. Numbered checkpoint files existed on disk. But every resume pulled from a stale state.

### Initial (wrong) hypothesis

The first instinct was to blame external factors: stale Kaggle dataset version attachment, SIGKILL mid-write corrupting the checkpoint file, or some interaction between Lightning 2.x and the learn2synth training loop.

**This was wrong.** It cost multiple debugging iterations before the real cause was found.

### Correct diagnosis

The issue was in how `ModelCheckpoint` handles `save_last=True` in Lightning 2.3+. The `last.ckpt` file is not independently written every epoch — it's linked to the top-k metric system. Specifically:

- `ModelCheckpoint` was configured with `save_top_k=3`, `monitor='eval_loss'`
- `last.ckpt` was being written via `save_last=True`
- But in Lightning 2.3+, `save_last` behaves as a reference to the most recent top-k checkpoint, not a separate unconditional save

**Evidence:**
```
last.ckpt mtime: epoch 103 (frozen)
numbered ckpts:  all from epoch 103 (most recent best eval_loss)
resume loaded:   epoch 103, not the actual current epoch 282
```

Because eval_loss stopped improving after epoch 103, no new top-k saves happened. And because `last.ckpt` was linked to top-k writes rather than being an independent save-every-epoch, it was stuck at epoch 103 too.

### Fix: `EveryEpochCheckpointCallback`

Bypass `ModelCheckpoint`'s save-logic entirely for the "resume" checkpoint. Write unconditionally on every `on_train_epoch_end`:

```python
class EveryEpochCheckpointCallback(Callback):
    """Writes resume.ckpt every epoch, unconditionally, bypassing
    ModelCheckpoint's save_last/top-k behavior."""
    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_path = os.path.join(CKPT_DIR, 'resume.ckpt')
        trainer.save_checkpoint(ckpt_path)
```

Resume logic was updated to prefer `resume.ckpt` over `last.ckpt`:
```python
RESUME_CKPT = f"{CKPT_DIR}/resume.ckpt"
INPUT_CKPT  = f"{CKPT_DIR}/last.ckpt"  # fallback only
```

**Verification (v23.1 log):** `resume.ckpt` mtime advanced every epoch across 107 consecutive writes (epochs 187–293). `last.ckpt` mtime lagged ~8 hours behind, confirming the original bug.

### Lesson

Trust the user's instinct when they've observed the system's behavior directly. The first theory (save-logic gating) was correct; the alternative theories (Kaggle dataset version, SIGKILL) sounded more sophisticated but weren't supported by the evidence on disk. `ls -la` was more diagnostic than reasoning about infrastructure.

---

## Part 2: The NaN Weight Corruption Issue

### Symptom

Training would run stably for 200–300 epochs, then suddenly produce NaN in the loss. Once NaN appeared, it persisted — every subsequent batch produced NaN, Dice went to 0, the run was effectively dead. This happened at non-deterministic epoch numbers (240, 267, 184 in different runs) suggesting a stochastic trigger, likely tied to random augmentation sampling.

### Hypothesis 1: fp16 overflow from aggressive synth augmentation (PARTIALLY CORRECT)

With `gamma=0.5, snr=10, bias_strength=0.5, gmm_fwhm=10`, the synthesis pipeline occasionally produced voxels exceeding fp16's max representable value (~65,504) after conv + InstanceNorm amplification. These would cascade into NaN activations and NaN gradients.

**Fix applied in v23.2:** Input clamping inside `SharedSynth.forward` before output flows back into the fp16 segnet:

```python
simg = torch.nan_to_num(simg, nan=0.0, posinf=10.0, neginf=-10.0)
simg = torch.clamp(simg, min=-10.0, max=10.0)
# Same for rimg
```

Also added `[CLAMP]` telemetry to log when real outliers were caught. Outliers up to ±27 were observed in `rimg`, confirming the hypothesis.

**Result:** Training reached epoch 267 before NaN recurred (vs. ~240 before). Clamp events fired but did not fully prevent NaN. The NaN was arriving through a different path — not from clamped inputs but from somewhere deeper in the forward or backward computation.

### Hypothesis 2: Unconditional `optim.step()` in learn2synth corrupting weights (WRONG)

Inspection of `learn2synth/train.py:228-248` showed `SynthSeg.train_step` does:
```python
optim.zero_grad()           # line 230
synth_pred = self.segnet(synth_image)
synth_loss = self.loss(synth_pred, synth_ref)
self.backward(synth_loss)   # line 237 — this is our Lightning manual_backward
optim.step()                # line 240 — UNCONDITIONAL
```

Theory: when `synth_loss` is NaN, `backward()` produces NaN gradients, and the unconditional `optim.step()` writes NaN into weights. To prevent this, intercept at the `manual_backward` level.

**Fix attempted in v23.3:** Override `manual_backward` on the LightningModule to skip calling `super().manual_backward()` when loss is non-finite:

```python
def manual_backward(self, loss, *args, **kwargs):
    if not torch.isfinite(loss).all():
        self._skip_step_nan = True
        # Do NOT call super() — skip backward entirely
        return
    self._skip_step_nan = False
    return super().manual_backward(loss, *args, **kwargs)
```

Expectation: with backward skipped, gradients remain at zero (from `zero_grad` at line 230), and `optim.step()` becomes a harmless no-op on zero-gradient. Weights stay clean.

**Result:** Training crashed at epoch 240 with:
```
AssertionError: No inf checks were recorded for this optimizer.
  File "/usr/local/lib/python3.12/dist-packages/torch/amp/grad_scaler.py", line 454
```

### Root cause of the v23.3 crash (Lightning + GradScaler internals)

Inspection of Lightning 2.4 source (via Claude Code) revealed the actual AMP control flow:

1. `LightningModule.manual_backward(loss)` → `Strategy.backward(loss, ...)`
2. `Strategy.backward` → `precision_plugin.pre_backward(loss, module)` — **this is where `scaler.scale(loss)` happens**
3. Then `precision_plugin.backward(scaled_loss, ...)` — where `scaled_loss.backward()` runs
4. Later, `optim.step()` routes through `MixedPrecision.optimizer_step` → `self.scaler.step(optimizer)`
5. `scaler.step()` calls `scaler.unscale_(optimizer)` which iterates parameters, finds non-None `.grad` tensors, runs `_amp_foreach_non_finite_check_and_unscale_`, and populates `optimizer_state["found_inf_per_device"]`
6. `scaler.step()` then asserts `len(optimizer_state["found_inf_per_device"]) > 0`

By skipping `super().manual_backward()`, we skipped step 2-3. No gradients were computed. In step 5, every parameter had `.grad = None`, so `_unscale_grads_` iterated nothing, returned an empty dict. Step 6 hit the assertion.

Critically: **Lightning's `MixedPrecision.optimizer_step` has a sanctioned skip path**, but it's gated on `automatic_optimization=True`:
```python
skip_unscaling = closure_result is None and model.automatic_optimization
```

Since we're using manual optimization, this path is never taken. There is no Lightning-approved way to skip an optimizer step while preserving GradScaler bookkeeping in manual-optimization mode.

### The actual truth: GradScaler already handles this case

The deeper insight from the Lightning/torch source analysis: `GradScaler.step()` is explicitly designed to skip `optimizer.step()` when gradients contain NaN/Inf. From `torch/amp/grad_scaler.py`:

```python
def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
    retval = None
    if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
        retval = optimizer.step(*args, **kwargs)   # skipped if any found_inf > 0
    return retval
```

When NaN/Inf gradients are detected:
- `optimizer.step()` is NOT called (parameters stay untouched)
- `scaler.update()` halves the scale factor via `backoff_factor=0.5`
- After `growth_interval` consecutive clean steps, scale grows back

**Hypothesis 2 was wrong.** The library's unconditional `optim.step()` goes through Lightning's AMP plugin, which routes through GradScaler, which already gates the step on gradient finiteness. The "unconditional" appearance was superficial; the actual protection was already in place.

By overriding `manual_backward` to skip, we:
1. Broke GradScaler's bookkeeping (caused the assertion crash)
2. Prevented GradScaler from seeing the NaN event (so the scale factor was never halved — the next iteration would use the same bad scale and trigger NaN again immediately)

### Fix: remove the override (v23.4)

```python
def manual_backward(self, loss, *args, **kwargs):
    # Logging only. GradScaler handles NaN grads automatically:
    #   - unscale_() detects NaN/Inf in gradients
    #   - _maybe_opt_step() skips optimizer.step() if any found
    #   - update() halves the scale factor for next iteration
    # Skipping super() would bypass this entire mechanism.
    if not torch.isfinite(loss).all():
        print(f"[NaN Notice] ⚠ Epoch {self.current_epoch}: non-finite loss "
              f"({loss.item():.3e}). GradScaler will skip optimizer step "
              f"and halve scale factor.")
    return super().manual_backward(loss, *args, **kwargs)
```

**Result (v23.4 log):** Training ran past epoch 292 cleanly. Zero `[NaN Notice]` events. Zero GradScaler skips. The combination of SharedSynth input clamping + GradScaler's built-in NaN handling was sufficient to prevent permanent corruption.

### The remaining weight-corruption scenario

After pushing past epoch 292, a new run showed NaN at epoch 184 with 68 parameters corrupted to NaN/Inf, despite GradScaler presumably working. Investigation showed:

- First NaN appeared in the `real` loss component (computed under `torch.no_grad()`, so no gradient contribution)
- One batch later, synth loss also went NaN
- Weight health diagnostic showed 68 NaN parameters at epoch end

This points to **a NaN source in the forward pass that GradScaler cannot catch.** GradScaler checks gradients, not activations. If a NaN appears during the forward pass — through an InstanceNorm dividing by near-zero variance, a division op, a `log(0)` — it can corrupt intermediate tensors that feed back into parameters through subsequent backwards. GradScaler's protection applies to gradient NaN, not forward-pass NaN.

This is a known limitation of AMP-based NaN protection: it's gradient-side, not forward-side.

### Lessons from the NaN saga

1. **Don't override framework behavior without understanding what it already does.** GradScaler was doing the right thing all along; the v23.3 override actively broke it.

2. **"Unconditional" operations in library source aren't always unconditional at runtime.** `optim.step()` in learn2synth routes through multiple Lightning plugins before it actually touches parameters. Surface-level code inspection is insufficient.

3. **Source inspection beats guessing.** The v23.3 fix was plausible and internally consistent, but would never have been tried if Claude Code had inspected Lightning's `MixedPrecision` plugin earlier. Using Claude Code to read actual library source (rather than reasoning about expected behavior) directly resolved the issue.

4. **Multiple NaN mechanisms require multiple defenses.** Input clamping catches fp16 overflow from augmentation. GradScaler catches gradient-side NaN. Neither catches forward-pass corruption from norm/division edge cases — that requires either weight-health rollback, gradient clipping, or forward-pass guards.

5. **The first batch of a NaN event contains the diagnostic information.** Subsequent batches show propagated NaN through corrupted weights; only the first shows where the NaN originated (synth vs. real branch, specific layer, specific augmentation parameters). Instrumentation should capture pre-forward inputs, per-layer activations, and gradient magnitudes on the very first NaN event.

---

## Part 3: Summary of All Code Changes

| Version | Purpose | Key change |
|---------|---------|------------|
| v22 (baseline) | Original 8-class setup | FreeSurfer label 99 = FCD |
| v23 | Switched to 6-class, cleaner data | FCD = label 21, class 5; dropped test subjects; val fraction 0.04 |
| v23.1 | Checkpoint regression fix | `EveryEpochCheckpointCallback` writes `resume.ckpt` every epoch |
| v23.2 | fp16 overflow guard | Input clamping + `nan_to_num` inside `SharedSynth.forward`; added weight-health, clamp telemetry, disk monitoring, resume-debug hooks |
| v23.3 | **Broken attempt:** `manual_backward` override | Skipped backward on NaN loss → crashed on GradScaler assertion |
| v23.4 | **Working fix:** remove override | `manual_backward` becomes logging-only; GradScaler handles NaN automatically |

## Part 4: Open Questions

1. **Forward-pass NaN source at epoch 184.** GradScaler caught gradient NaN in most runs, but a single run produced NaN in forward activations (68 corrupted parameters at epoch end). Root cause not fully identified. Candidates: InstanceNorm on degenerate channels, log(0) in DiceCE, or an edge case in the augmentation pipeline not caught by input clamping.

2. **Missing early-stop on permanent NaN.** Once weights are corrupted, training cannot recover, but the loop continues, wasting compute and overwriting the `resume.ckpt` with corrupt state. A weight-health check that halts training and flags for rollback would prevent this.

3. **DICE FCD regression.** Independent of the NaN issue: the model lost FCD segmentation capability between epoch 156 (Dice 0.71) and epoch 290 (Dice 0.04-0.14). Hypothesis: class collapse under aggressive augmentation combined with equal class weighting in DiceCE. Proposed fix: resume from epoch-156 checkpoint with manual class weights `[1, 1, 1, 1, 1, 5]` favoring FCD.