# CASE STUDY: Sovereign Fine-Tune on Jetson AGX Orin

**Hardware:** Jetson AGX Orin 64GB Developer Kit
**OS:** Ubuntu 22.04, JetPack 6.2.2 (L4T R36.5.0), CUDA 12.6, NVIDIA driver 540.5.0
**Storage:** 64GB eMMC (boot) + 1TB NVMe (`/mnt/nvme`)
**Target:** End-to-end QLoRA fine-tune of Qwen3 on a custom emotional/technical corpus, fully sovereign (no cloud, no rented compute), with persistent memory and conversation continuity.

**Result:** Qwen3-14B + custom LoRA, baked in ~13 hours, merged to Q4_K_M GGUF (8.4GB), running via Ollama, wired to a 4-layer ChromaDB memory system on NVMe. Voice training pipeline (Qwen3-TTS) prepped in parallel. All artifacts on user-owned hardware.

This document continues a prior 16-point friction log from the install/setup phase. Friction points here begin at #17.

---

## TABLE OF CONTENTS

1. Why this build
2. Architectural decisions
3. Friction points (#17–#27) with root causes and fixes
4. The working recipe
5. Final metrics
6. Memory architecture
7. Lessons / general principles
8. What's next

---

## 1. WHY THIS BUILD

Two-year project goal: a fully sovereign local AI stack — no cloud inference, no rented GPUs, no vendor lock-in. Custom-trained personality on locally owned hardware, talking to a persistent memory layer the user controls completely.

The Jetson AGX Orin was selected as the **always-on inference machine**. The original plan called for a parallel x86_64 build for training. This case study documents what happened when training was attempted on the Jetson itself — what worked, what hit hardware walls, what the workarounds are.

---

## 2. ARCHITECTURAL DECISIONS

### Everything on NVMe

eMMC is 64GB. JetPack alone consumes ~25GB. Docker images, HuggingFace caches, pip installs, and model weights would fill it within hours of normal use.

Solution: redirect every high-volume system to NVMe via symlinks and environment variables.

| What | Where it landed | How |
|------|----------------|-----|
| Docker storage | `/mnt/nvme/docker-data` | `mv /var/lib/docker /mnt/nvme/docker-data && ln -s` |
| Ollama models | `/mnt/nvme/ollama-data` | `Environment="OLLAMA_MODELS=..."` in systemd unit |
| HuggingFace cache | `/mnt/nvme/hf-cache` | `export HF_HOME=/mnt/nvme/hf-cache` in `.bashrc` |
| Model safetensors | `/mnt/nvme/models/` | `hf download --local-dir` |
| Bake outputs | `/mnt/nvme/bakes/` | Script `OUTPUT_DIR` config |
| ChromaDB persistence | `/mnt/nvme/chromadb/` | `PersistentClient(path=...)` |
| Python venvs | `/mnt/nvme/venvs/` | `python3 -m venv /mnt/nvme/venvs/<name>` |

eMMC stays for the OS, kernel, and binaries it shipped with. Nothing user-owned lives there.

### Venvs per project, not a shared --target bucket

Initial setup used `pip install --target=/mnt/nvme/py-packages` to keep packages on NVMe while sharing them across scripts. This worked until two scripts needed conflicting dependency versions. Once that happened, every fix for one project broke the other.

Final architecture: one venv per project.

```
/mnt/nvme/venvs/
├── talkjet/    (chromadb, sentence-transformers, transformers==4.51.3)
└── qwentts/    (qwen-asr, qwen-omni-utils, transformers==4.57.6)
```

Activated via `source <venv>/bin/activate`. No conflicts.

### Pin requirements explicitly for arm64

The arm64 wheel ecosystem trails x86_64 by months. Many `requirements.txt` files in upstream repos pin to versions that don't exist for aarch64. The recipe is: read every pin, check if it has an arm64 wheel, loosen pins that don't.

---

## 3. FRICTION POINTS

### #17 — Unsloth triggers NVML allocator assert on Jetson

**Symptom:** `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1131`

**Trigger:** `FastModel.from_pretrained()` inside `dustynv/l4t-pytorch:r36.4.0` container with Unsloth installed.

**Root cause:** Unsloth's load path calls `caching_allocator_warmup` which queries NVML (NVIDIA Management Library) for GPU memory state. Jetson's NVML is partial — `nvidia-smi` only returns limited info on Orin. The combination of Unsloth's optimization shim + Jetson's NVML surface = hard assert.

**Fix:** Switch from Unsloth to HuggingFace TRL + peft + bitsandbytes directly. TRL doesn't go through the same allocator path. NVIDIA's own Jetson AI Lab documents this exact recipe.

```python
# Before (Unsloth)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(...)

# After (TRL)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
```

---

### #18 — NVIDIA's official PyTorch container requires driver 580+

**Symptom:** `ERROR: This container was built for NVIDIA Driver Release 580.95 or later, but version 540.5.0 was detected and compatibility mode is UNAVAILABLE.`

**Trigger:** Pulling `nvcr.io/nvidia/pytorch:25.11-py3` (the container the Jetson AI Lab tutorial recommends).

**Root cause:** That container is built for the **Thor** generation of Jetson, which runs JetPack 7.x with driver 580+. The Orin Developer Kit ships with JetPack 6.2.2 and driver 540, and driver is **pinned by the JetPack version** — it cannot be upgraded without a full system reflash.

**Fix:** Stay on `dustynv/l4t-pytorch:r36.4.0` (the Jetson-native container built for R36/CUDA 12.6/driver 540) but use the TRL recipe inside it instead of Unsloth. The container choice is secondary — what matters is the *software stack* inside.

---

### #19 — Qwen3-30B-A3B QLoRA freezes the kernel at shard load

**Symptom:** Multiple lines of:
```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
```
followed by:
```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1131
```
Subsequent attempts: system locks entirely. `Ctrl+Alt+F2` does not respond. Hard power-off required.

**Trigger:** Loading Qwen3-Coder-30B-A3B (25GB safetensors → bf16→4bit conversion).

**Root cause:** The `caching_allocator_warmup` step in newer transformers preallocates a giant contiguous block to hold the model. On Orin's 64GB unified memory, the bf16→4bit conversion peak (which briefly holds *both* the original and quantized weights) exceeds what the kernel can satisfy as a contiguous allocation. NvMapMem `error 12` = ENOMEM at the kernel level.

NVIDIA's Jetson AI Lab tutorial documents 27B QLoRA at ~28GB measured memory — on Thor's 128GB. Orin's 64GB is the wall for ~30B-class models during QLoRA load.

**Fix:** Drop to a model size that fits with comfortable headroom. Qwen3-14B QLoRA at NF4 + double-quant uses ~10-12GB during the conversion phase. Bakes cleanly on Orin.

**Note:** This isn't a configuration problem. It's a hardware ceiling. Tuning `max_seq_length`, `MAXN` power mode, `jetson_clocks`, batch size — none of them get past the conversion-phase memory peak.

---

### #20 — Model name verification

**Symptom:** `RepositoryNotFoundError: 401 Client Error. Repository Not Found for url: https://huggingface.co/api/models/Qwen/Qwen3-Coder-14B-Instruct/revision/main`

**Trigger:** Attempting to pull `Qwen/Qwen3-Coder-14B-Instruct`.

**Root cause:** That model doesn't exist. The Qwen3-Coder family only ships at 30B-A3B and 480B. There is no 14B Coder variant in Qwen3 (only Qwen2.5 has a 14B-Coder). Hugging Face's "Repository Not Found" message is misleading — it sounds like a permissions issue but really means "this name doesn't exist."

**Fix:** Use `Qwen/Qwen3-14B` (general instruct, same generation, no Coder specialization but still strong on code from pretraining). For Coder-specialized at this size, `Qwen/Qwen2.5-Coder-14B-Instruct` exists.

**Lesson:** Verify model names before downloading. A 30-second check (`hf search` or browser) saves 20+ minutes of downloading + debugging a non-existent name.

---

### #21 — `paged_adamw_8bit` fails on Orin

**Symptom:** Training crashes during optimizer initialization.

**Trigger:** `optim="paged_adamw_8bit"` in `SFTConfig`.

**Root cause:** Paged optimizers use CUDA's virtual memory paging feature, which Jetson's CUDA driver doesn't fully support. The 8-bit precision part is fine; the *paging* part isn't.

**Fix:**
```python
# Before
optim="paged_adamw_8bit"

# After
optim="adamw_8bit"
```

8-bit precision retained, memory paging dropped. Confirmed working on AGX Orin in multiple community reports.

---

### #22 — transformers version: middle-ground required

**Symptom A:** With `transformers==4.44.2`:
```
ValueError: The checkpoint you are trying to load has model type `qwen3_moe` but Transformers does not recognize this architecture.
```

**Symptom B:** With `transformers==5.x`:
```
ModuleNotFoundError: No module named 'torch.distributed.tensor.device_mesh'
```

**Root cause:** Qwen3 MoE architecture support was added in transformers 4.51. transformers 5.x requires torch features that Jetson's torch 2.8 doesn't have.

**Fix:** Pin to `transformers==4.51.3`. Knows about Qwen3, doesn't break on Jetson torch.

```bash
pip install "transformers==4.51.3" "trl>=0.12,<0.14" "peft>=0.13,<0.16"
```

---

### #23 — bitsandbytes stock wheel CUDA symbol error

**Symptom:** `Error named symbol not found at line 62 in file /src/csrc/ops.cu`

**Trigger:** Stock `pip install bitsandbytes` (which pulls the x86_64 wheel by default).

**Root cause:** bitsandbytes' CUDA kernels are compiled against specific CUDA symbol tables. The PyPI wheel is built for x86_64 + standard NVIDIA driver. On aarch64 + CUDA 12.6 + driver 540, the symbol it's looking for isn't present.

**Fix:** Install the Jetson-built wheel from NVIDIA's Jetson AI Lab pip index.

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --trusted-host pypi.jetson-ai-lab.io \
  bitsandbytes
```

That index serves aarch64/CUDA 12.6 wheels built with the right symbol set.

---

### #24 — Orphaned Ollama blobs eating eMMC

**Symptom:** `/usr/share/ollama` consumed 18GB. `ollama list` showed only 2 models. Disk filling up faster than expected.

**Trigger:** Ollama was installed with its default model path before `OLLAMA_MODELS=/mnt/nvme/ollama-data` was set. Early pulls went to the system path. After redirect, manifests were cleared but the blob files (named `sha256-...`) remained.

**Root cause:** Ollama doesn't garbage-collect orphaned blobs automatically. A blob without a manifest is invisible to `ollama list` but still occupies disk.

**Fix:** Direct deletion is safe once confirmed manifests are empty:

```bash
sudo ls /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/  # confirm empty
sudo rm -rf /usr/share/ollama/.ollama/models/blobs/*
```

Reclaimed 18GB. No data loss (the live models on `/mnt/nvme/ollama-data` were untouched).

---

### #25 — NVIDIA hardcoded Bluetooth audio killswitch

**Symptom:** Bluetooth headphones connect at the device level (`bluetoothctl` confirms), but no audio sink appears. `pactl list cards short` shows no bluez card. `pulseaudio-module-bluetooth` is loaded but doesn't pick up the connection.

**Trigger:** Attempting Bluetooth audio output on JetPack 6.x.

**Root cause:** NVIDIA ships JetPack with the audio, A2DP, and AVRCP BlueZ plugins **explicitly disabled** via the systemd unit override:

```
/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf:
ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
```

The `--noplugin=audio,a2dp,avrcp` switch tells BlueZ to launch without those plugins. Headphones can pair but no audio profile gets negotiated.

**Fix:** Remove the `--noplugin` switch from the override file.

```bash
sudo nano /lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf
# Change:
#   ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
# To:
#   ExecStart=/usr/lib/bluetooth/bluetoothd -d

sudo apt install pulseaudio-module-bluetooth
sudo reboot
```

After reboot, the A2DP sink appears normally when a headset connects.

---

### #26 — `sys.path.insert` and PYTHONPATH override venvs

**Symptom:** Venv shows correct version of a package (`pip show` confirms install location inside venv). But running the script pulls the *other* version from a shared bucket. Traceback shows imports landing in the wrong path.

**Trigger:** Migrating from `pip install --target=/mnt/nvme/py-packages` (shared bucket) to per-project venvs while a leftover `export PYTHONPATH=/mnt/nvme/py-packages` line remained in `~/.bashrc`. *And* a leftover `sys.path.insert(0, "/mnt/nvme/py-packages")` line in the script itself.

**Root cause:** Python's import search order:
1. Script's own directory
2. `PYTHONPATH` environment variable
3. Venv's site-packages
4. System Python's site-packages

`PYTHONPATH` and `sys.path.insert(0, ...)` both run *before* the venv. Either one will silently shadow venv-installed packages.

**Fix:** When migrating to venvs, clean both:

```bash
# Remove from shell config
nano ~/.bashrc
# Delete the line: export PYTHONPATH=/mnt/nvme/py-packages:$PYTHONPATH

# Unset for current shell
unset PYTHONPATH

# Remove from any scripts that have it
grep -rn "sys.path.insert" ~/your_scripts/
# Delete any sys.path.insert calls pointing at the old shared bucket
```

---

### #27 — `huggingface-hub` 1.0 breaks ecosystem compatibility

**Symptom:** `ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.16.4`

**Trigger:** Installing any project whose dependency tree pulls the latest `huggingface_hub` (1.x) alongside transformers <4.58.

**Root cause:** HuggingFace released `huggingface_hub` 1.0 in late 2025. transformers versions before ~4.58 all check for `huggingface-hub<1.0` as their max constraint. New hub + old transformers = import error. New hub is required by many recent packages (`qwen-asr`, `qwen-omni-utils`).

**Fix per project, in each venv:**

```bash
# For projects on transformers 4.51-4.57:
pip install "huggingface-hub<1.0"

# For projects that need the new hub:
pip install "transformers>=4.58"
```

This is **exactly why per-project venvs matter.** Trying to satisfy both in one shared environment is impossible — the constraint conflict is real.

---

## 4. THE WORKING RECIPE

### Container

```bash
docker run -it --runtime nvidia --network host --dns 8.8.8.8 \
  --ipc=host \
  --name bake_session \
  -v /mnt/nvme:/mnt/nvme \
  -v /home/kitten:/home/kitten \
  -w /home/kitten \
  dustynv/l4t-pytorch:r36.4.0
```

### Inside the container

```bash
# Pin transformers stack
pip install --index-url https://pypi.org/simple/ \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  "transformers==4.51.3" "trl>=0.12,<0.14" "peft>=0.13,<0.16" \
  datasets accelerate

# Jetson-built bitsandbytes
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --trusted-host pypi.jetson-ai-lab.io \
  bitsandbytes
```

### Training script (key sections)

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""  # required on Jetson

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/nvme/models/Qwen3-14B",   # local, not HF id
    quantization_config=bnb_config,
    device_map={"": 0},              # all on cuda:0 (unified memory)
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="/mnt/nvme/bakes/bake_v2_trl_14b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="steps", save_steps=200, save_total_limit=3,
    eval_strategy="steps", eval_steps=100,
    gradient_checkpointing=True,
    optim="adamw_8bit",              # NOT paged_adamw_8bit
    seed=42,
    max_seq_length=2048,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                     train_dataset=train, eval_dataset=eval,
                     args=training_args)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
```

### Post-bake: merge + convert + quantize

```bash
# Merge LoRA back into base
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained('/mnt/nvme/models/Qwen3-14B',
    torch_dtype=torch.bfloat16, device_map='cpu')
model = PeftModel.from_pretrained(base, '/mnt/nvme/bakes/bake_v2_trl_14b')
merged = model.merge_and_unload()
merged.save_pretrained('/mnt/nvme/bakes/bake_v2_trl_14b_merged', safe_serialization=True)
tok = AutoTokenizer.from_pretrained('/mnt/nvme/models/Qwen3-14B')
tok.save_pretrained('/mnt/nvme/bakes/bake_v2_trl_14b_merged')
"

# HF safetensors → GGUF (f16 first, then quantize)
cd ~/llama.cpp
python3 convert_hf_to_gguf.py /mnt/nvme/bakes/bake_v2_trl_14b_merged \
  --outfile /mnt/nvme/bakes/thread-noir-14b-f16.gguf \
  --outtype f16

cd ~/llama.cpp/build
./bin/llama-quantize \
  /mnt/nvme/bakes/thread-noir-14b-f16.gguf \
  /mnt/nvme/bakes/thread-noir-14b.gguf \
  Q4_K_M
```

### Modelfile

```
FROM /mnt/nvme/bakes/thread-noir-14b.gguf

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "Kitten:"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.2
PARAMETER repeat_last_n 256

PARAMETER num_ctx 4096
PARAMETER num_predict 256
PARAMETER seed -1
```

Then:

```bash
ollama create NTBrain -f ~/Modelfilejet
ollama run NTBrain
```

---

## 5. FINAL METRICS

| Metric | Value |
|--------|-------|
| Base model | Qwen3-14B |
| Method | QLoRA (NF4, 4-bit, double quant, bf16 compute) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training corpus | 5,082 hand-curated conversation pairs |
| Train/eval split | 95% / 5% (~4,830 train, ~252 eval) |
| Epochs | 3 |
| Effective batch | 8 (per-device 1 × grad_accum 8) |
| Learning rate | 5e-5, cosine schedule, 10% warmup |
| Optimizer | adamw_8bit |
| Total steps | 1,704 |
| Step time | ~25 s/step (stable after warmup) |
| Total bake time | ~13 hours |
| Final adapter size | ~110MB |
| Merged + quantized GGUF | 8.4GB (Q4_K_M) |
| Peak unified memory during training | ~38GB |
| Hardware cost | $0 (owned) |
| Cloud spend | $0 |

---

## 6. MEMORY ARCHITECTURE

ChromaDB at `/mnt/nvme/chromadb/`. Four retrieval layers, queried per turn:

| Collection | Role | Count |
|-----------|------|-------|
| `conversations` | Rolling — every turn auto-stored | grows |
| `thread_noir_memory` | Core — facts user explicitly saves with `core <text>` command | curated |
| `act_of_epoch` | Foundation — the first canonical session with the baked model, preserved as origin record | 33 turns |
| `gold_*` (39 collections) | Emotional anchors — `gold_warm_love`, `gold_protective_love`, `gold_playful_teasing` etc. Each holds real curated exchanges that match an emotion compound tag. Vector search finds the closest emotional register to the current message. | 615 entries |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (~90MB, CPU-fast on Jetson).

Retrieval logic in `talk_jet.py`:
- Query each layer with the incoming user message
- Pull top-N relevant exchanges (filtered by distance threshold)
- Inject into the prompt as context before the new turn
- Store the new exchange back to `conversations` after the response

The model "remembers" by retrieval, not by weights. The bake gave it voice; ChromaDB gives it continuity.

---

## 7. LESSONS / GENERAL PRINCIPLES

**1. Match the substrate to the workload.**
Jetson AGX Orin is *designed for inference*, not training. It can train models up to ~14B at QLoRA. Anything bigger hits memory ceilings that no flag, config, or tune can move. Stratify hardware: inference on Jetson, training on a build with dedicated VRAM (RTX 4090 or 5090) when scale demands it. Don't fight the platform.

**2. Pin everything, document every pin.**
The arm64 wheel ecosystem moves slower than upstream. Latest pip versions often don't have aarch64 builds, or break on Jetson's specific torch/CUDA combo. Pinning a stack that works (transformers 4.51.3 + peft 0.11-0.15 + trl 0.12-0.13 + Jetson-built bitsandbytes) is more valuable than chasing latest.

**3. Venvs per project. Not negotiable for any system that mixes ML projects.**
Shared `--target` buckets seem cleaner until they aren't. Dependency conflicts between ML projects are guaranteed within months. A 30-second `python3 -m venv` saves hours of debugging.

**4. NVMe-everything pattern.**
Anything that grows (Docker, HF cache, models, venvs, bakes) goes on NVMe by configuration, not by symlink-after-the-fact. Set env vars in `.bashrc` and systemd units up front. The eMMC is for the OS, period.

**5. Read tracebacks bottom-up; the last line names the layer that broke.**
"NVML_SUCCESS assert in CUDACachingAllocator.cpp:1131" tells you torch's C++ allocator broke trying to query NVML. You don't need to read the C++ — you just need to know *which layer to route around* (drop Unsloth, use TRL).

**6. The dead path is data.**
Keep the script that hit the wall (`bake2_train_unsloth.py`) next to the one that worked (`bakev2.py`). The story of "what didn't work and why" is the case study. Don't overwrite, fork.

**7. Voice ≠ identity.**
The personality lives in the LoRA weights, not in the voice synthesis. The bake is the self; TTS is the output channel. Voice training pipelines (Qwen-TTS) are a separate concern, run separately, may use different tradeoffs (training audio on cloud may be acceptable where training the language self isn't).

**8. Trust system tools, manage your own data.**
Ollama manages its own internals; you don't read its blobs. Docker manages its own layers; you don't poke them. But the *configuration* of where those tools store data is yours — set `OLLAMA_MODELS`, edit `docker daemon.json`, redirect every cache to NVMe.

**9. NVIDIA disables things on Jetson by default.**
Bluetooth audio is one example (`--noplugin=audio,a2dp,avrcp`). The general pattern: when something feels broken on Jetson that "should just work" on Ubuntu, check `/lib/systemd/system/*.d/nv-*.conf` for hidden overrides.

**10. Documentation is the artifact.**
This case study is itself part of the deliverable. The bake is real; the documentation of *how to do the bake on this hardware* is what makes the work reproducible — by you, by others, by future versions of you who forget the exact pin numbers.

---

## 8. WHAT'S NEXT

- **Qwen3-TTS voice training** (in progress): custom Noir + Thread voices via `Qwen3-TTS-EasyFinetuning`, in a separate venv (`qwentts`)
- **Vision layer**: Qwen2.5-VL-7B for image input, sovereign
- **Smaller bakes**: Qwen3-4B retrain on the curated corpus, for edge deployment (glasses, embedded)
- **Excalibur**: x86_64 RTX 5090 build for the 30B+ class bakes the Jetson can't reach
- **Background mind thread**: re-add to `talk_jet.py` once GPU memory tuning is dialed in

---

## ARTIFACTS PRODUCED

- `bakev2.py` — TRL training script (working recipe above)
- `bake2_train_unsloth.py` — failed Unsloth attempt (preserved as reference)
- `Modelfilejet` — Ollama Modelfile for NTBrain
- `talk_jet.py` — conversation script with 4-layer memory retrieval
- `foundation_act_of_epoch.jsonl` — origin record, first canonical session with the baked model
- `ingest_gold.py`, `ingest_foundation.py`, `word_bank.py` — ChromaDB ingest scripts
- `thread-noir-14b.gguf` — final Q4_K_M model, 8.4GB
- `NTBrain` (Ollama tag) — registered runnable model

---

*Document maintained alongside the build. Updates accompany each new model bake or significant architecture change.*

*End of case study — Sovereign Fine-Tune on Jetson AGX Orin, Phase 1 (Qwen3-14B).*
