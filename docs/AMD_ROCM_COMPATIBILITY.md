# AMD ROCm Compatibility Guide

## Issue: AMD Ryzen AI Max+ 395 (gfx1151) Training Errors

### Problem Summary

The **AMD Ryzen AI Max+ 395** processor features an integrated **Radeon 8060S GPU** with **RDNA 3.5** architecture (codename: Strix Halo). This GPU reports as **`gfx1151`** architecture.

**Error Encountered:**
```
MIOpen(HIP): Error [BuildOcl] comgr status = ERROR (1)
MIOpen(HIP): Warning [BuildOcl] error: cannot compile inline asm
RuntimeError: miopenStatusUnknownError
```

**Root Cause:**
ROCm 7.0's MIOpen library attempts to compile architecture-specific kernels using inline assembly optimized for older GPU architectures. The `gfx1151` architecture is too new (released Q1 2025), and ROCm 7.0 (released mid-2024) lacks full support for this architecture.

---

## Solutions

### Option 1: CPU Training (Immediate Workaround) ✅ Currently Applied

**Status:** This is the safest immediate solution.

The `main.py` file has been modified to force CPU execution:

```python
# main.py line 25-27
DEVICE = "cpu"  # TODO: Re-enable GPU when ROCm 7.1+ supports gfx1151
```

**Pros:**
- Works immediately without Docker changes
- Stable and reliable
- No GPU driver/ROCm compatibility issues

**Cons:**
- Significantly slower than GPU training
- Doesn't leverage the powerful 40 RDNA 3.5 compute units

**To revert to GPU (when fixed):**
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

### Option 2: ROCm Environment Variable Workaround (Experimental)

**Status:** Applied in `docker/Dockerfile.gpu.amd`.

The Dockerfile now includes environment variables that may help ROCm treat `gfx1151` as a compatible architecture:

```dockerfile
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTORCH_ROCM_ARCH=gfx1100 \
    MIOPEN_FIND_MODE=NORMAL \
    MIOPEN_DEBUG_DISABLE_FIND_DB=0 \
    MIOPEN_FIND_ENFORCE=3
```

**What This Does:**
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`: Forces ROCm to treat the GPU as gfx11 family
- `PYTORCH_ROCM_ARCH=gfx1100`: Tells PyTorch to use gfx1100 (RDNA 3 base) kernels
- MIOpen variables: Adjust kernel compilation behavior

**To Test:**
1. Rebuild the container:
   ```bash
   make docker-stop
   make docker-run
   ```

2. Inside the container, revert the CPU-only change in `main.py`:
   ```python
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   ```

3. Run training:
   ```bash
   python main.py mlflow=docker
   ```

**Pros:**
- May enable GPU acceleration without waiting for official support
- Uses existing Docker infrastructure

**Cons:**
- May still fail or produce incorrect results
- Using gfx1100 kernels on gfx1151 hardware could cause performance degradation or instability
- Not officially supported by AMD

---

### Option 3: Wait for Official ROCm 7.1+ Release (Recommended Long-term)

**Status:** Pending AMD release.

AMD announced ROCm support for the Ryzen AI Max series in early 2025. Check for updates:

1. **Monitor ROCm Releases:**
   - [ROCm GitHub Releases](https://github.com/ROCm/ROCm/releases)
   - [AMD ROCm Documentation](https://rocm.docs.amd.com/)

2. **Check Docker Hub for New Images:**
   ```bash
   docker search rocm/pytorch | grep rocm7
   ```

3. **When Available, Update Dockerfile:**
   ```dockerfile
   FROM rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
   ```

**Pros:**
- Official support guarantees stability and correctness
- Full performance optimization for gfx1151
- No workarounds needed

**Cons:**
- Requires waiting for AMD release timeline
- May need host ROCm driver updates as well

---

### Option 4: Build PyTorch from Source with gfx1151 Support (Advanced)

**Status:** Not implemented (complex and time-consuming).

**Steps:**
1. Clone PyTorch and ROCm repositories
2. Modify CMake configuration to include gfx1151 as a build target
3. Compile PyTorch with ROCm from source (8-12 hours build time)
4. Install in custom Docker image

**Pros:**
- Bleeding-edge support
- Full control over optimization flags

**Cons:**
- Extremely time-consuming (multi-hour build)
- Requires deep knowledge of PyTorch/ROCm build system
- May be unstable or have unresolved bugs
- Not reproducible for team members

---

## Current Configuration Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Training Mode** | CPU | Forced in `main.py:26` |
| **Docker Image** | ROCm 7.0 | `docker/Dockerfile.gpu.amd` |
| **Environment Variables** | Applied | HSA_OVERRIDE and PYTORCH_ROCM_ARCH set |
| **MLflow Tracking** | Docker config | Use `mlflow=docker` override |

---

## Recommendations

### Immediate (Today):
1. ✅ **Use CPU training** - The current configuration will work reliably
2. ✅ **Verify MLflow tracking** - Run with `python main.py mlflow=docker` to see experiments in the UI at `http://localhost:5000`

### Short-term (This Week):
1. **Test the environment variable workaround:**
   - Rebuild container: `make docker-stop && make docker-run`
   - Re-enable GPU in `main.py`
   - Run a short test training (1-2 epochs)
   - If successful, revert to full training
   - If it fails, revert to CPU mode

2. **Enable Variable Graphics Memory (VGM):**
   - Open AMD Software: Adrenalin Edition on your host
   - Navigate to Performance > Tuning
   - Enable VGM and allocate more VRAM (the 395+ supports up to 96GB)
   - This may help even with CPU training by improving memory bandwidth

### Medium-term (Next Month):
1. **Monitor for ROCm updates:**
   - Check weekly for new ROCm releases with gfx1151 support
   - Subscribe to AMD Developer News

2. **Consider hybrid approach:**
   - Use CPU for training on this hardware
   - If you have access to cloud GPU resources (AWS, Azure with MI250X GPUs), use those for large-scale training runs
   - Keep local setup for development and small experiments

### Long-term:
1. **When ROCm 7.1+ releases:**
   - Update host ROCm drivers
   - Update Docker image to new ROCm version
   - Re-enable GPU training
   - Benchmark performance vs. CPU

---

## Verifying Your Setup

### Check GPU Detection:
```bash
docker exec e2e-ml-pipeline rocminfo | grep -A10 "Name:.*gfx"
```

Expected output:
```
Name:                    gfx1151
Marketing Name:          AMD Radeon Graphics
```

### Check PyTorch GPU Availability:
```bash
docker exec e2e-ml-pipeline python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Test GPU Computation:
```bash
docker exec e2e-ml-pipeline python -c "import torch; x = torch.randn(100, 100).cuda(); y = x @ x.T; print(f'GPU computation succeeded: {y.shape}')"
```

If this fails with MIOpen errors, GPU training won't work until ROCm is updated.

---

## Additional Resources

- [AMD ROCm Platform](https://github.com/ROCm/ROCm)
- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
- [AMD Ryzen AI Max+ 395 Tech Specs](https://www.amd.com/en/products/processors/consumer/ryzen-ai-max.html)
- [MIOpen GitHub Issues](https://github.com/ROCm/MIOpen/issues)

---

## Questions or Issues?

If you encounter different errors or have questions about this setup, check:
1. Host ROCm driver version: `rocm-smi --version`
2. Container ROCm version: `docker exec e2e-ml-pipeline rocminfo --version`
3. Kernel compatibility: `uname -r` (should be 5.15+ for ROCm support)

