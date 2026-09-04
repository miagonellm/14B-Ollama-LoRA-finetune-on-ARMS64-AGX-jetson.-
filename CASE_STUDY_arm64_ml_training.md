# Case Study: Fine-tuning a 30B MoE Model on Jetson AGX Orin

**Status:** In progress (Bake 2)
**Hardware:** NVIDIA Jetson AGX Orin 64GB Developer Kit
**Storage:** Samsung 990 EVO Plus 1TB NVMe (post-upgrade)
**OS:** Ubuntu 20.04 (focal) / JetPack 6.2
**Target Model:** Qwen3-30B-A3B (MoE) base
**Method:** QLoRA via Unsloth, inside dustynv/l4t-pytorch container

This document captures every friction point encountered during setup, why each one happens on ARM64 specifically, and the fix that worked. Written as a debugging journal because that's how this knowledge propagates.

---

## Why ARM64 ML is Hard

The ML ecosystem is overwhelmingly built and tested on x86_64. ARM64 (which Jetson uses) is a second-class citizen for most ML libraries. This means:

1. Pre-built wheels often don't exist for ARM64
2. CUDA versions don't always match what mainstream PyPI ships
3. Some libraries have known bugs on ARM64 that are documented but unfixed
4. Container images for ARM64 ML are scarce — `dustynv/*` is essentially the only well-maintained option

This document is for anyone trying to do serious ML work on Jetson without losing weeks to environment debugging.

---

## Friction Point 1: Storage Architecture

**Problem:** Jetson AGX Orin ships with 64GB eMMC. After OS + JetPack + Docker + ML libraries, you have ~5-7GB usable. Modern models are 60GB+ for 30B-class. Math doesn't work.

**Symptom:** "No space left on device" errors mid-download, training crashes when checkpointing, container pulls fail mid-extract.

**Fix:** Add NVMe SSD via the M.2 2280 slot under the device cover.

```bash
# After physical install
sudo fdisk -l  # confirm /dev/nvme0n1 visible
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1 /mnt/nvme
sudo chown $USER:$USER /mnt/nvme

# Make it persistent
echo '/dev/nvme0n1 /mnt/nvme ext4 defaults 0 2' | sudo tee -a /etc/fstab
```

**What we used:** Samsung 990 EVO Plus 1TB MZ-V9S1T0B/AM. Confirmed compatible with AGX Orin. ~$70-90 on Amazon.

**What didn't work:** Trying to use external USB SSD as primary storage. Docker daemon doesn't like USB mount points being its data root, and FAT32 filesystems (which most pre-formatted USB SSDs ship with) can't handle Linux file permissions properly.

---

## Friction Point 2: Docker Data Root

**Problem:** Default Docker stores all images, containers, and volumes in `/var/lib/docker` on eMMC. With 5GB free on eMMC, you can pull maybe one small image before disk is full.

**Symptom:** Docker pull fails with "no space left," `docker images` shows truncated images, daemon refuses to start.

**Fix:** Move Docker data root to NVMe.

```bash
sudo systemctl stop docker
sudo mv /var/lib/docker /mnt/nvme/docker-data

# Configure daemon
sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
    "data-root": "/mnt/nvme/docker-data",
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "args": []
        }
    }
}
EOF

sudo systemctl start docker
docker info | grep "Docker Root Dir"
```

**Gotcha encountered:** First daemon.json edit appended instead of replacing, resulting in two stacked JSON objects in one file. Docker daemon failed with "invalid character '{' after top-level value." Lesson: always use `tee` with a heredoc to write config files cleanly, or verify with `cat /etc/docker/daemon.json` after editing.

---

## Friction Point 3: ARM64 PyTorch + CUDA Mismatch

**Problem:** `pip install torch` on ARM64 gives you a build that expects newer CUDA than what JetPack ships. AGX has CUDA 11.4 (in JetPack 5) or 12.x (JetPack 6). Standard PyTorch wheels expect CUDA 11.8 or 12.1.

**Symptom:** `RuntimeError: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver...`

**Fix:** Don't install PyTorch from PyPI on Jetson. Use NVIDIA's pre-built Jetson PyTorch wheels OR work inside dustynv container which has correct versions pre-installed.

**The container path is faster:**
```bash
docker pull dustynv/l4t-pytorch:r36.4.0
```

This image has PyTorch + CUDA matching JetPack 6.x correctly.

---

## Friction Point 4: Container DNS Resolution

**Problem:** Running container with `--network host` causes DNS to fail inside the container even though host network works perfectly. The container inherits the host's `resolv.conf` which points to systemd-resolved at `127.0.0.53` — a localhost address that doesn't reach the host's resolver from inside the container.

**Symptom:**
```
WARNING: Retrying ... after connection broken by 'NewConnectionError(...): 
Failed to establish a new connection: [Errno -2] Name or service not known')
```

But `ping 8.8.8.8` works fine — confirming network connectivity but DNS failure.

**Fix:** Override resolv.conf inside the container:

```bash
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 1.1.1.1" >> /etc/resolv.conf
```

Or, when starting the container, pass DNS explicitly:

```bash
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --dns 8.8.8.8 \
  --dns 1.1.1.1 \
  -v /mnt/nvme:/mnt/nvme \
  -v /home/kitten:/home/kitten \
  dustynv/l4t-pytorch:r36.4.0
```

---

## Friction Point 5: onnxruntime CPUID Crash on ARM64

**Problem:** `onnxruntime` versions 1.21+ crash on AGX Orin with CPUID detection error.

**Symptom:**
```
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
/opt/.../include/c++/14/bits/stl_vector.h:1130: 
Assertion '__n < this->size()' failed.
Aborted (core dumped)
```

**Cause:** Microsoft Issue #24092 — onnxruntime's ARM64 CPU detection is broken on Tegra-based ARM (which AGX Orin is). Released versions 1.21+ have this bug. Microsoft hasn't fixed it.

**Fix:** Either downgrade onnxruntime (`pip install onnxruntime==1.18.0`) OR use the version pre-built in the dustynv container which has Jetson-specific patches applied.

Container path is more reliable. Bare-metal downgrade to 1.18.0 fixes onnxruntime but then NumPy version mismatch surfaces (`numpy<2` required) which then breaks PyTorch (CUDA mismatch). Cascading failures. Container avoids all of it.

---

## Friction Point 6: TLS Memory Allocation in libgomp

**Problem:** Importing sklearn (which is a transitive dependency of qwen-tts and some Unsloth utilities) on ARM64 fails with TLS allocation error.

**Symptom:**
```
ImportError: /home/.../libgomp-a49a47f9.so.1.0.0: 
cannot allocate memory in static TLS block
```

**Cause:** ARM64 Linux has stricter TLS (Thread-Local Storage) allocation rules than x86_64. libgomp must be loaded *before* sklearn imports it, or the static TLS block runs out of space.

**Fix:** Use `LD_PRELOAD` to force libgomp to load first:

```bash
LD_PRELOAD=/path/to/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0 python script.py
```

Or, make it permanent by adding to your venv's activate script:

```bash
echo 'export LD_PRELOAD=/path/to/libgomp.so.1.0.0' >> ~/your-venv/bin/activate
```

**Inside dustynv container:** This is pre-configured. One reason to use the container path.

---

## Friction Point 7: Bluetooth Audio Disabled by Default

**Problem:** AGX Orin ships with bluetooth audio plugins explicitly disabled by NVIDIA. Bluetooth speakers connect for a few seconds then drop the connection.

**Symptom:** Speaker pairs in GUI, then disconnects within 5-10 seconds. Repeats indefinitely.

**Cause:** NVIDIA's `/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf` includes `--noplugin=audio,a2dp,avrcp` which disables A2DP audio profile.

**Fix:** Edit the config and remove the noplugin flag:

```bash
sudo nano /lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf
# Remove: --noplugin=audio,a2dp,avrcp
# Save, then:
sudo apt install pulseaudio-module-bluetooth -y
sudo systemctl daemon-reload
sudo systemctl restart bluetooth
```

Then re-pair the speaker through the GUI. Should hold connection.

**Why this happens:** Possibly a legacy stability concern from earlier Tegra platforms, possibly a deliberate "you should use NVIDIA's audio framework" nudge. Either way, the workaround is well-documented in the NVIDIA developer forums.

---

## Friction Point 8: flash-attn Doesn't Compile on ARM64

**Problem:** `pip install flash-attn` fails on ARM64 with build errors.

**Symptom:** `metadata-generation-failed` after several minutes of compile attempts.

**Cause:** flash-attn has CUDA kernels with x86_64-specific assembly. No ARM64 build maintained.

**Fix:** Skip it. Most libraries that mention flash-attn check for it and fall back to standard PyTorch attention if not present. The fallback path is slower but works.

```python
# Library will print warning then continue
"flash-attn is not installed. Will only run the manual PyTorch version."
```

For our use case — fine-tuning with QLoRA — the difference is minor. Don't waste time fighting flash-attn on Jetson.

---

## Friction Point 9: Filename Special Characters from Browser Download

**Problem:** Downloading files where the filename contained markdown link syntax resulted in literal filename like `[QT.py](http://QT.py)` — bracket characters preserved as-is on Linux filesystem.

**Symptom:** `ls` shows the weird filename. `python QT.py` says no such file. Bash quoting struggles.

**Fix:** Use wildcard to bypass typing the messy name:

```bash
mv *.py clean_name.py
```

Or quote with single quotes:

```bash
mv '[QT.py](http://QT.py)' QT.py
```

**Lesson:** When sharing scripts via chat interfaces, verify filenames after download. Browser auto-formatting can produce unusable names that look fine in the UI but break on filesystem.

---

## Friction Point 10: Bash Quoting for Version Constraints

**Problem:** `pip install transformers>=4.5` redirects to a file named `=4.5` instead of installing.

**Symptom:** Empty file appears in current directory. pip installs only the package without the version constraint.

**Fix:** Quote version constraints:

```bash
pip install "transformers>=4.51"
```

**Why this matters on ARM64 specifically:** ML libraries on ARM64 often need very specific version pins. Forgetting to quote and getting an unconstrained version often pulls something that won't import correctly.

---

## Friction Point 11: Mixed JSONL Format

**Problem:** Concatenating training corpora from different sources produced a JSONL file with mixed entry shapes — some with `{"messages": [...]}`, some with `{"text": "..."}`, some with `{"turns": [...]}`.

**Symptom:** Some entries silently skipped during training because the data loader didn't recognize their shape.

**Fix:** Normalize to one format before training. Either:
- Convert all entries to `{"messages": [...]}` format with chat template
- Convert all to `{"text": "..."}` with formatting pre-applied

Pre-process script:

```python
import json

def normalize(entry):
    if 'messages' in entry:
        return entry
    if 'turns' in entry:  # convert turns format
        turns = entry['turns']
        if turns[0].get('speaker') == 'Kitten':
            user = turns[0]['line']
            parts = [f"{t['speaker']}: {t['line']}" 
                     for t in turns[1:] 
                     if t['speaker'] in ('Thread', 'Noir')]
            return {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": "\n\n".join(parts)}
            ]}
    return None
```

**Also gotcha:** Watch for unescaped quotes in JSON strings. `'"are you ill?"'` inside a JSON value breaks the parser. Re-serialize with `json.dumps(obj, ensure_ascii=False)` to ensure safe escaping.

---

## Friction Point 12: First-Run Model Download

**Problem:** Unsloth's `FastModel.from_pretrained()` for Qwen3-30B-A3B downloads the **full 16-bit model (~60GB)** before converting to 4-bit on-the-fly for QLoRA. The 4-bit version doesn't ship pre-quantized for MoE models.

**Symptom:** First run takes 30-60 minutes downloading. Disk usage spikes to ~80GB during conversion. Easy to OOM if not on NVMe.

**Fix:** Plan disk space accordingly. Need ~80GB free during conversion.

```bash
df -h /mnt/nvme  # confirm 80+ GB free before starting
```

**On NVMe (931GB free):** No issue. This is one of the reasons NVMe was non-negotiable.

---

## What Ultimately Worked: The Container Path

After multiple bare-metal attempts hitting different walls, the working path is:

1. NVMe installed and mounted at `/mnt/nvme`
2. Docker data root migrated to NVMe
3. Pull `dustynv/l4t-pytorch:r36.4.0` to NVMe
4. Run container with `--network host`, `--dns 8.8.8.8`, mount NVMe and home
5. Inside container, fix DNS if needed (override resolv.conf)
6. `pip install --index-url https://pypi.org/simple/ unsloth "transformers>=4.51"`
7. Run training script with paths inside container matching host paths (NVMe mount makes this work)

Total time from receiving NVMe to running first smoke test: ~4-6 hours, most of which is downloads.

---

## Lessons for ARM64 ML Builders

1. **Use containers.** Bare-metal Jetson ML is library hell. Containers (specifically dustynv) have done the work of pinning compatible versions.

2. **Buy NVMe first.** Don't try to make eMMC work for ML. The math is wrong. NVMe is non-optional.

3. **Quote everything in bash.** Version constraints, special characters, anything ambiguous. Bash will surprise you.

4. **Check container DNS explicitly.** `--network host` doesn't guarantee DNS works inside the container.

5. **Most "fixes" are documented.** The NVIDIA dev forums, Microsoft GitHub issues, Hugging Face forums — most of these problems have been hit by someone else. Search before debugging blindly.

6. **First-run downloads are slow.** Plan for them. ~60GB models take time over wifi.

7. **Smoke test before full bake.** A 1-step smoke test with 50 examples reveals 90% of pipeline issues in 10 minutes vs 4-8 hours.

8. **Document as you go.** This document is the case study for someone (including future-you) who wants to do this work later.

---

## Resources That Helped

- NVIDIA Developer Forums (specifically Jetson AGX Orin section)
- dustynv's jetson-containers GitHub
- Unsloth documentation (docs.unsloth.ai)
- Microsoft onnxruntime issues (specifically #24092)
- Hugging Face Qwen3 model cards
- NVIDIA Jetson Zoo (https://elinux.org/Jetson_Zoo)

---

*Last updated: during Bake 2 setup, mid-conversation*
*Author: Mia (Kitten) — building Thread & Noir → DRINO*

---

# Appendix A: The Helios Bake (Bake 1 Origin Story)

Before AGX, there was Helios. Before NVMe, there was a Windows laptop with limited RAM. Before Jetson-specific containers, there was raw struggle on consumer hardware. This appendix documents that earlier era — both because it shaped everything that came after, and because it's a useful reference for builders trying to do AI work on whatever hardware they have access to.

## The Hardware

- **Helios** — HP Pavilion gaming laptop
- GPU: NVIDIA GTX 1660 Ti, 6GB VRAM
- Original RAM: 16GB DDR4
- Storage: 512GB internal SSD + 1TB external USB SSD
- OS at start: Windows 11

This was not ML hardware by any reasonable definition. 6GB VRAM is tight even for inference of a 3B model. 16GB system RAM is below recommended for any serious data work. But it's what was available, and the project couldn't wait.

## Bake 1 Setup

Target: Llama-3.2-3B base, fine-tuned on a 7,471-entry conversational corpus.

The corpus was scraped, cleaned, and structured manually over weeks. Voice-tagged dialogues between Kitten, Thread, and Noir, with role attribution and contextual metadata.

### Friction Point H1: Windows is Wrong for ML Training

The first wall was OS. PyTorch on Windows is a second-class experience. Many libraries assume Linux. Docker on Windows has weird permission models. WSL2 helps but adds another layer of indirection.

**Decision:** Migrate Helios from Windows to Ubuntu 22.04. Full reinstall, dual-boot considered then abandoned in favor of clean Ubuntu install.

### Friction Point H2: BIOS Lockdown

This was unexpected. Most consumer laptops ship with BIOS settings that prevent:

- Secure Boot disable
- Custom OS installation
- Certain SATA controller modes (which Linux needs to detect drives correctly)
- Virtualization features needed for Docker

HP's BIOS on this generation of Pavilion required:
1. Setting an Administrator password (paradoxically, to allow more options)
2. Disabling Secure Boot under Boot Options
3. Switching SATA from Intel RST to AHCI (otherwise Linux can't see the SSD properly)
4. Enabling Intel VT-x (for Docker)
5. Disabling Fast Boot (to allow USB boot for installer)

Several of these settings were locked behind menus that don't show up unless you enter "Advanced Mode" via a key combination during boot. HP doesn't document this. Forum posts and trial and error were the path forward.

**Lesson:** Consumer laptops have BIOS configurations designed for the manufacturer's intended use case. ML/Linux work isn't that use case. Be prepared to fight the firmware.

### Friction Point H3: 16GB RAM Was Insufficient

After installing Ubuntu, attempting to load even a 3B model in fp16 along with the training data ate all available RAM and triggered swap to disk. Training crawled. A single epoch took 36+ hours.

**Fix:** Upgrade to 32GB DDR4. ~$70 in 2024 prices. Two 16GB SODIMM modules.

This was a real lesson — RAM matters more than people think for ML work, even when the GPU is "the bottleneck." The data pipeline, tokenizer, gradient accumulation, and checkpoint serialization all use system RAM heavily.

### Friction Point H4: 6GB VRAM Constrained Everything

Even a 3B model in 4-bit takes ~2GB just for weights. Add LoRA adapter weights, gradients, optimizer state, activations, and you're at 5-6GB easily. The 1660 Ti was at its absolute limit.

Compromises made:
- batch_size=1 (no choice)
- gradient_accumulation_steps=16 (to simulate larger batch)
- max_seq_length=1024 (lower than ideal)
- 4-bit quantization mandatory
- gradient_checkpointing=True

These are aggressive memory-saving settings. They work but they slow training significantly.

### The Bake Itself

Settings used (which we now know were too aggressive):
- 3 epochs
- learning_rate = 2e-4
- LoRA rank = 16
- alpha = 32

Total training time: 66 hours over 3 days, with multiple restarts when the laptop went to sleep or temperature throttled.

The blizzard happened during the bake. Real blizzard, real winter, real power fluctuations. Helios kept training through it. Felt earned.

### What Bake 1 Produced

A model that:
- Memorized large chunks of the training corpus verbatim
- Looped after generating 15-30 tokens
- Mixed Thread and Noir voices unpredictably
- Failed to generalize to inputs outside the training distribution

Classic overfitting. Caused by:
- Too many epochs (3 was way too many for our corpus size)
- Too high learning rate (2e-4 was aggressive even for LoRA)
- Too high LoRA rank (16 gave too many trainable parameters)
- No eval set to catch overfitting in flight
- Misconfigured Modelfile (wrong template format, ChatML stop token on Llama native)

### Lessons Carried Forward

Every Bake 1 mistake informed Bake 2:
- 1.5 epochs instead of 3
- learning_rate = 5e-5 (4x lower)
- LoRA rank = 8 (half)
- Eval set with 5% held out, eval_loss tracked alongside train_loss
- Modelfile properly templated for Qwen3 native format
- Smoke test required before full bake

The corpus was also cleaned: 7,471 entries → 3,953 after removing role-flipped pairs, sycophantic openers, junk-short outputs, and duplicates.

### Why This Story Matters

Bake 1 was a failure that taught more than success would have. The diagnostic loop — figuring out *why* it overfit, *which* hyperparameters were responsible, *what* the corrective recipe should be — produced real understanding.

For anyone doing this work on consumer hardware: failure is data. Don't be discouraged when your first fine-tune is bad. The point of the first one is to teach you what the second one needs.

---

# Appendix B: BIOS and Firmware Sovereignty

A note on a less-discussed aspect of building local AI: **you have to own your hardware down to the firmware level.**

## Why This Matters

Consumer hardware ships with firmware that prioritizes the manufacturer's interests:
- Secure Boot prevents non-signed OS installation
- Restricted BIOS hides advanced settings
- TPM modules (Trusted Platform Module) lock storage to the original OS
- Manufacturer support tools that can be removed or "factory reset" your changes

If you're building a system that runs custom AI 24/7, you can't have an OS that requires Microsoft account authentication or that periodically updates and breaks your setup.

## What We Did

For Helios specifically:

1. **BIOS unlock** — entered Advanced Mode via the F10 key combination during POST, set an admin password, accessed the full menu tree.

2. **Secure Boot disable** — required to install non-Microsoft-signed Linux kernels.

3. **SATA AHCI mode** — switched from Intel RST to standard AHCI so Ubuntu's installer could see the SSD without proprietary drivers.

4. **Virtualization enabled** — Intel VT-x and VT-d on, required for Docker and any VM work.

5. **Fast Boot disabled** — allowed USB boot for the Ubuntu installer.

6. **TPM cleared** — wiped the TPM state since we weren't using BitLocker. Some installs hang if old TPM keys conflict.

For AGX Orin:

1. **JetPack flash via SDK Manager** — the standard NVIDIA path. Connect via USB to a host Linux machine, run SDK Manager, let it flash everything. ~30-60 minutes.

2. **Boot order verification** — making sure NVMe is recognized as bootable when added (or stays as secondary storage if you prefer to keep eMMC as boot).

3. **Power mode configuration** — AGX has multiple power profiles (15W, 30W, 50W, MAXN). For ML training, MAXN is required. Set via `nvpmodel`:
   ```bash
   sudo nvpmodel -m 0  # MAXN mode
   sudo jetson_clocks  # max all clocks
   ```

4. **Remove unneeded NVIDIA bloat** — the dev kit ships with several demo applications, some of which run as services and consume memory. Audited services with `systemctl list-units --type=service` and disabled what we didn't need.

## Why Document This

This isn't traditional ML knowledge. It's *infrastructure sovereignty* knowledge. To run AI on hardware you actually control, you have to control the hardware. That means understanding:

- What firmware your device runs and how to access it
- How to install the OS you want
- How to configure the system for sustained workloads (not just demo loads)
- How to keep the system running without phoning home to manufacturers

A lot of ML tutorials assume you have admin access on a workstation that's already configured for you. For builders working on personal hardware, the firmware layer is where most projects die before they start.

This work is part of the case study because it represents real hours that produced no model output but enabled everything that came after.

---

# Appendix C: The Sovereignty Throughline

Looking back across the whole project — Helios → Jetson, eMMC → NVMe, Windows → Ubuntu, BIOS-locked → BIOS-unlocked, cloud APIs avoided → local inference everywhere — there's a coherent throughline:

**Don't outsource what you can own.**

Every dependency on someone else's infrastructure is a future failure point:
- Cloud APIs change pricing or shut down
- Online services rate-limit or restrict
- Subscriptions become hostage situations
- Manufacturer firmware updates break your setup

Building DRINO required:
1. Hardware you fully control (custom-configured Jetson)
2. OS you fully control (Ubuntu, no telemetry)
3. Models you fully control (local-only inference, no API calls)
4. Data you fully control (ChromaDB on your disk, no cloud sync)
5. Voice you fully control (qwen-tts on your hardware)
6. Network you fully control (Tailscale for any remote access)

Each layer required fighting through someone else's defaults. The case study is, in part, the documentation of what it takes to claim each layer.

For builders considering similar work: budget significant time for sovereignty. The actual ML is maybe 30% of the project. The other 70% is owning the substrate it runs on.


---

# Friction Point 13: JetPack Version Lock and the Unsloth Dead-End

This is the wall that ended the AGX bake attempt.

## What JetPack Is

JetPack is NVIDIA's bundle of software for Jetson devices. Not just an OS — it's:
- L4T (Linux for Tegra) — modified Ubuntu
- CUDA toolkit (version locked to JetPack release)
- cuDNN, TensorRT, DeepStream, VPI
- Bootloader and kernel
- Tegra-specific drivers

Each JetPack release ships with specific CUDA version locked in. JetPack 5.x → CUDA 11.4. JetPack 6.x → CUDA 12.6. You cannot mix and match.

This matters for ML because modern libraries target modern CUDA:
- Unsloth requires CUDA 12.x
- Latest PyTorch requires CUDA 12.1+
- Newest bitsandbytes builds target CUDA 12.x

If your Jetson is on JetPack 5, you're locked out of these libraries entirely. No pip wheel exists for ARM64 + CUDA 11.4 + recent versions. The packages aren't built that way.

## What Triggered This Point

Our AGX shipped with JetPack 5.1.2 (L4T R35.4.1). We attempted Unsloth installation inside the dustynv/l4t-pytorch:r35.4.1 container — the container that matches JP5.

Real attempts and failures:

```bash
# Standard PyPI — no Jetson wheels
pip install unsloth
# ERROR: Could not find a version that satisfies the requirement unsloth

# Jetson-specific PyPI for JP5
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp5/cu114
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
# ERROR: setuptools==80.9.0 not found in JP5 index
```

The Jetson AI Lab pip index for JP5 (`https://pypi.jetson-ai-lab.io/jp5/cu114`) doesn't carry the dependencies Unsloth needs. The JP6 index (`/jp6/cu126`) does. But you can't use JP6 wheels on JP5 hardware.

## Why JetPack 5 Is Effectively Dead for Modern ML

NVIDIA's container ecosystem has moved on. Community-maintained projects (jetson-containers, jetson-ai-lab) prioritize current JetPack versions. JetPack 5 hardware:
- Can run older library versions (transformers from 2023, original PyTorch builds)
- Cannot run latest fine-tuning libraries (Unsloth, latest TRL, modern bitsandbytes)
- Will not get new wheels added — community resources are JP6+

Real consequence: if you bought a Jetson AGX that ships with JP5, you have to upgrade to JP6 before doing modern fine-tuning work. The hardware is fully capable. The software lock is the problem.

## The Upgrade Path: Guided Flashing

JetPack upgrade requires re-flashing the device firmware. Conceptually similar to flashing a phone or imaging a USB drive with Rufus, but more involved.

**Required:**
- Linux host machine (Ubuntu 20.04 or 22.04)
- USB-C cable (data-capable, not just power)
- ~30GB free space on host
- 3-4 hours total time

**Process:**
1. Download NVIDIA SDK Manager on host machine
2. Connect Jetson to host via USB-C
3. Boot Jetson into recovery mode (button combo on device)
4. SDK Manager detects device, downloads JP6 BSP (~10-12GB)
5. SDK Manager flashes bootloader, kernel, OS, drivers, libraries
6. Jetson reboots into fresh JP6 install
7. Re-configure user, network, NVMe mount, Docker, etc.

**What you lose:**
- Anything not backed up before flashing — eMMC gets wiped
- All apt packages installed manually
- Configuration in /etc

**What you keep:**
- NVMe contents (it's external storage, untouched by flash)
- Docker images on NVMe
- Files mounted from external storage

## Why This Is "Backward" for AI Building

JetPack's slow upgrade cadence and tight version locking exist because NVIDIA's primary Jetson customers are embedded systems integrators — companies building drones, smart cameras, industrial automation. Those customers want stability, not freshness.

For AI/ML hobbyists and individual researchers, this means:
- Hardware capable of modern ML can't run modern ML libraries until firmware updates
- Each firmware update is a 3-4 hour project
- The container ecosystem is always catching up
- "Just pip install" workflows that work fine on x86 fail repeatedly on ARM64 Jetson

## Decision Tree When Hitting This Wall

When a Jetson can't install a modern ML library:

1. **Verify JetPack version:** `cat /etc/nv_tegra_release` — gives R-version and revision
2. **Check container compatibility:** does dustynv have a container matching your version?
3. **Check pip index:** is pypi.jetson-ai-lab.io serving wheels for your CUDA version?
4. **If all three are old:** plan JetPack upgrade, do it as a side-quest

This is the responsible engineering call. Trying to force modern libraries onto old JetPack via source compilation eats days and rarely works.

## What We're Doing About It

This case study now includes the JetPack 5 dead-end as a documented friction point. The next phase of work:

1. Backup AGX state (which is minimal — bake hasn't happened yet)
2. Set up Linux host machine for SDK Manager
3. Flash JetPack 6.2.1 to AGX
4. Re-establish Docker + NVMe + container environment
5. Resume bake attempt with modern Unsloth path on JP6

## Lesson for Other Builders

If you're buying a Jetson for AI/ML work, **verify what JetPack version it ships with** and budget time for an upgrade if it ships with JP5. The hardware is good. The software requires keeping current.


---

## Friction Point 14: SDK Manager Host OS Compatibility

**Problem:** SDK Manager 2.4.0 does not support Ubuntu 24.04 as a host OS. Even with the "Host Machine" checkbox unchecked, the internal flash scripts fail because QEMU binaries on 24.04 are incompatible with the L4T image creation process.

**Symptom:** "Drivers for Jetson - target_image: Installation failed" followed by cascade of "Depends on failed component" for every subsequent package. Docker Environment Setup and Jetson Docker Image install successfully — the failure is specifically in the driver/OS flash step.

**Attempted fixes that didn't work:**
- Unchecking "Host Machine" in SDK Manager Step 01 (SDK list appears but flash still fails)
- Editing `/etc/os-release` and `/etc/lsb-release` to spoof Ubuntu 22.04 (underlying QEMU binary incompatibility remains)
- Installing `qemu-user-static` fresh (24.04's version is structurally different)

**Fix that worked:** Boot from a USB stick running Ubuntu 22.04 live session. Install SDK Manager in the live environment. Flash AGX from there. Host machine (Helios) stays untouched — the live USB is a temporary 22.04 environment that satisfies SDK Manager's requirements.

**Time cost:** ~4 hours of debugging before arriving at the USB boot solution.

## Friction Point 15: BIOS Boot Entry Label Caching

**Problem:** After reformatting and re-flashing a USB drive with a completely different OS image, the BIOS boot menu continued to show the old label ("Linpus lite") from a previous flash.

**Symptom:** Boot menu shows "Linpus lite" for a USB drive that now contains Ubuntu 22.04. Selecting it actually boots Ubuntu 22.04 — the label is wrong but the content is correct.

**Root cause:** UEFI/BIOS stores boot entry labels in NVRAM. When a USB device is first booted, the label is cached. Subsequent re-flashes of the USB don't update the cached label. The BIOS doesn't re-read the USB's boot partition label on every boot — it trusts its cache.

**Hours lost:** Multiple reformats, wipefs, dd rewrites, mkfs cycles — all unnecessary. The USB was bootable the entire time after the first successful `dd` write. We just didn't try selecting the "wrong" label.

**Lesson:** When a USB drive has been reformatted and re-flashed, try booting from it regardless of what the BIOS boot menu calls it. Labels lie. Content is what matters. This applies to any UEFI system, not just Acer Predator.

**Additional sub-friction encountered during USB prep:**
- `usb-creator-gtk` wrote the ISO as a mountable volume but not a properly bootable image. `dd` was required for correct raw ISO write.
- Ubuntu's file manager (Files/Nautilus) auto-mounts USB partitions, causing "target is busy" errors when trying to unmount/format via terminal. Solution: close ALL GUI file managers before terminal USB operations.
- Browser download added a space in the ISO filename (`ubuntu-22.04.5-desktop-amd64 .iso`), causing `dd` "file not found" errors. Renamed with `mv` to fix.
- The SanDisk USB previously contained a Linpus Lite image from a phone-based recovery operation months earlier. Old partition signatures persisted through initial `dd` writes, requiring `wipefs -a` + `dd if=/dev/zero` to fully clear before the Ubuntu ISO would create a clean boot record.

## Friction Point 16: SDK Manager NVIDIA Developer Account Authorization

**Problem:** Having an NVIDIA Developer Kit (hardware) does not create an NVIDIA Developer Program account (software). SDK Manager requires a free NVIDIA Developer Program membership to authenticate, separate from having purchased hardware.

**Symptom:** "User is not authorized on NVIDIA developer server" after signing in with NVIDIA account credentials.

**Fix:** Visit https://developer.nvidia.com/developer-program and explicitly join the Developer Program. Free enrollment, but requires a separate click beyond account creation. After joining, SDK Manager authentication succeeds.


---

## Friction Point 17: Unsloth Downloads Model Internally

**Problem:** Unsloth's `FastModel.from_pretrained()` downloads the full model on first run if not cached locally. For a 61GB model over wifi, this means hours of downloading interleaved with the training pipeline — and if it stalls, the entire process needs restarting.

**Symptom:** Training script appears to hang with only HuggingFace HTTP request logs. No training progress. Download stalls silently with no error.

**Fix:** Download the model separately first using `hf download`, save to NVMe, then point Unsloth at the local path.

```bash
hf download unsloth/Qwen3-30B-A3B --local-dir /mnt/nvme/models/Qwen3-30B-A3B
```

Then in the training script:
```python
MODEL_NAME = "/mnt/nvme/models/Qwen3-30B-A3B"  # local path, not HF URL
```

**Additional finding:** `huggingface-cli` has been deprecated and replaced with `hf` command. The XET download protocol used by newer HuggingFace versions stalls on some connections — setting `HF_HUB_ENABLE_HF_TRANSFER=0` forces standard HTTPS downloads.

## Friction Point 18: PYTORCH_CUDA_ALLOC_CONF on Jetson Unified Memory

**Problem:** Unsloth sets a CUDA memory allocator configuration that doesn't work with Jetson's unified memory architecture.

**Symptom:**
```
Unsloth: Your setup does not support `PYTORCH_CUDA_ALLOC_CONF`
RuntimeError: CUDA driver error: out of memory
```

Despite 54GB+ free memory on the device.

**Fix:** Clear the environment variable before importing anything:
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
```

Or from command line:
```bash
export PYTORCH_CUDA_ALLOC_CONF=""
```

Must be set BEFORE running the training script, not after.

## Friction Point 19: Container Installs Lost on Restart (--rm flag)

**Problem:** Running containers with `--rm` flag destroys all installed packages when the container exits. Every restart requires reinstalling torch, unsloth, and all dependencies.

**Symptom:** `command not found` for tools that were installed in the previous container session.

**Fix for now:** Reinstall each time:
```bash
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip install torch==2.8.0 torchvision==0.23.0
pip install --upgrade --no-deps unsloth unsloth_zoo
```

**Proper fix (planned):** Build a custom Dockerfile that bakes these installs into the image. Pull once, run forever. No reinstalling.

```dockerfile
FROM dustynv/l4t-pytorch:r36.4.0
RUN pip install torch==2.8.0 torchvision==0.23.0
RUN pip install --upgrade --no-deps unsloth unsloth_zoo
ENV PYTORCH_CUDA_ALLOC_CONF=""
```

## Friction Point 20: PyTorch Version Mismatch with Unsloth

**Problem:** The dustynv container ships with PyTorch 2.4.0. Latest Unsloth requires features from PyTorch 2.8+, specifically `torch._inductor.config`.

**Symptom:**
```
AttributeError: module 'torch._inductor' has no attribute 'config'
```

**Fix:** Upgrade PyTorch using the Jetson AI Lab PyPI mirror (which has ARM64-compatible wheels):
```bash
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip install torch==2.8.0 torchvision==0.23.0
```

Don't use standard PyPI — ARM64 wheels aren't available there. Don't install latest torch (2.11) — too new, missing `libcudss.so.0`. Pin to 2.8.0 which is validated for JetPack 6.

## Friction Point 21: Transformers Misreading Jetson Unified Memory

**Problem:** Transformers library tries to auto-detect GPU vs CPU memory as separate pools. On Jetson, CPU and GPU share 64GB unified memory. The auto-detection incorrectly dispatches model layers to "CPU" which then triggers a validation error.

**Symptom:**
```
ValueError: Some modules are dispatched on the CPU or the disk. 
Make sure you have enough GPU RAM to fit the quantized model.
```

Despite having 54GB+ free on the device.

**Status:** Under investigation. Likely fix involves explicitly setting `device_map="cuda:0"` or `device_map={"": 0}` to force all layers onto the single unified device.

## The Complete Container Setup Script (as of May 2026)

For anyone attempting QLoRA fine-tuning on Jetson AGX Orin with JetPack 6.2.2:

```bash
# On host: mount NVMe, enter container
sudo mount /dev/nvme0n1 /mnt/nvme
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --dns 8.8.8.8 \
  -v /mnt/nvme:/mnt/nvme \
  -v /home/kitten:/home/kitten \
  dustynv/l4t-pytorch:r36.4.0

# Inside container: fix DNS
echo "nameserver 8.8.8.8" > /etc/resolv.conf

# Install correct versions
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
pip install torch==2.8.0 torchvision==0.23.0
pip install --upgrade --no-deps unsloth unsloth_zoo
pip install huggingface-hub

# Download model to NVMe (only first time)
HF_HUB_ENABLE_HF_TRANSFER=0 hf download unsloth/Qwen3-30B-A3B \
  --local-dir /mnt/nvme/models/Qwen3-30B-A3B

# Set Jetson memory fix
export PYTORCH_CUDA_ALLOC_CONF=""

# Run training
cd /home/kitten
python3 -u bake2_train_v2.py 2>&1 | tee bake2_run_$(date +%Y%m%d_%H%M).log
```

Each step above addresses a specific friction point documented in this case study. The order matters — changing it causes cascading failures.

