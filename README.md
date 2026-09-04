# 14B-Ollama-LoRA-finetune-on-ARMS64-AGX-jetson.-
This case study will walk you through how to successfully tune your Ollama model on ARMS64
# Sovereign Fine-Tune on Jetson AGX Orin

**A field report from the frontier of edge ML: QLoRA fine-tune of Qwen3-14B, fully sovereign, on a 64GB unified-memory board.**

The full stack lives on user-owned hardware. No cloud inference, no rented GPUs, no vendor lock-in. This document is the friction log - eleven walls hit, each one diagnosed and routed around, and the working recipe that came out the other side.

**Hardware:** Jetson AGX Orin 64GB Dev Kit · Ubuntu 22.04 · JetPack 6.2.2 (L4T R36.5.0) · CUDA 12.6 · driver 540.5.0
**Storage:** 64GB eMMC (boot) + 1TB NVMe (`/mnt/nvme`)

---

## Result

| Metric                              | Value                                                                 |
|-------------------------------------|-----------------------------------------------------------------------|
| **Base model**                      | Qwen3-14B                                                             |
| **Method**                          | QLoRA (NF4, 4-bit, double quant, bf16 compute)                        |
| **LoRA rank / alpha**               | 8 / 16                                                                |
| **Target modules**                  | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj         |
| **Training corpus**                 | 5,082 hand-curated conversation pairs (95/5 train/eval split)         |
| **Epochs / effective batch / LR**   | 3 / 8 / 5e-5 cosine, 10% warmup                                       |
| **Optimizer**                       | `adamw_8bit` (paged variant unsupported on Jetson)                    |
| **Total steps / step time**         | 1,704 / ~25 s per step                                                |
| **Total bake time**                 | ~13 hours                                                             |
| **Final adapter / merged GGUF**     | ~110MB LoRA / 8.4GB Q4_K_M                                            |
| **Peak unified memory**             | ~38GB                                                                 |
| **Cloud spend**                     | $0                                                                    |

Result: Qwen3-14B + custom LoRA, merged to Q4_K_M GGUF, running via Ollama, wired to a four-layer ChromaDB memory system on NVMe. All artifacts on owned hardware.

---

## Architecture decisions

### Everything on NVMe

eMMC is 64GB. JetPack alone eats ~25GB. Docker, HuggingFace caches, model weights would fill it within hours. Every high-volume system was redirected to NVMe up front, by configuration - not by symlink after the fact.

| What                  | Where it landed          | How                                          |
|-----------------------|--------------------------|----------------------------------------------|
| Docker storage        | `/mnt/nvme/docker-data`  | Move directory + symlink                     |
| Ollama models         | `/mnt/nvme/ollama-data`  | `OLLAMA_MODELS=...` in systemd unit          |
| HuggingFace cache     | `/mnt/nvme/hf-cache`     | `HF_HOME=/mnt/nvme/hf-cache` in `.bashrc`    |
| Model safetensors     | `/mnt/nvme/models/`      | `hf download --local-dir`                    |
| Bake outputs          | `/mnt/nvme/bakes/`       | Script `OUTPUT_DIR` config                   |
| ChromaDB persistence  | `/mnt/nvme/chromadb/`    | `PersistentClient(path=...)`                 |
| Python venvs          | `/mnt/nvme/venvs/`       | `python3 -m venv /mnt/nvme/venvs/<name>`     |

eMMC stays for OS and shipped binaries. Nothing user-owned lives there.

### One venv per project

Initial setup used `pip install --target=/mnt/nvme/py-packages` to share packages across scripts. Worked until two projects needed conflicting dependency versions. Every fix for one broke the other.

Final architecture: one venv per project. No shared state.

```
/mnt/nvme/venvs/
├── talkjet/    (chromadb, sentence-transformers, transformers==4.51.3)
└── qwentts/    (qwen-asr, qwen-omni-utils, transformers==4.57.6)
```

### Pin explicitly for arm64

The arm64 wheel ecosystem trails x86_64 by months. Many upstream `requirements.txt` files pin versions that don't exist for aarch64. The recipe: read every pin, verify an arm64 wheel exists, loosen pins that don't.

---

## Friction log

Eleven walls, in order hit. Each entry: symptom, trigger, root cause, fix.

### #17 - Unsloth triggers NVML allocator assert on Jetson

- **Symptom:** `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1131`
- **Trigger:** `FastModel.from_pretrained()` inside `dustynv/l4t-pytorch:r36.4.0` with Unsloth installed.
- **Root cause:** Unsloth's load path calls `caching_allocator_warmup` which queries NVML for GPU memory state. Jetson's NVML is partial. Optimization shim + partial NVML = hard assert.
- **Fix:** Switch to HuggingFace TRL + peft + bitsandbytes directly. TRL doesn't traverse the same allocator path.

### #18 - NVIDIA's official PyTorch container requires driver 580+

- **Symptom:** `This container was built for NVIDIA Driver Release 580.95 or later, but version 540.5.0 was detected.`
- **Trigger:** Pulling `nvcr.io/nvidia/pytorch:25.11-py3`.
- **Root cause:** That container targets the **Thor** generation (JetPack 7.x, driver 580+). Orin ships with JetPack 6.2.2 / driver 540, and the driver is pinned by JetPack - not upgradeable without a full reflash.
- **Fix:** Stay on `dustynv/l4t-pytorch:r36.4.0` (Jetson-native, R36 / CUDA 12.6 / driver 540). Container choice is secondary; software stack inside is what matters.

### #19 - Qwen3-30B-A3B QLoRA freezes the kernel at shard load

- **Symptom:** `NvMapMemAllocInternalTagged: error 12` followed by NVML assert. System locks entirely; hard power-off required.
- **Trigger:** Loading Qwen3-Coder-30B-A3B (25GB safetensors → bf16→4bit conversion).
- **Root cause:** `caching_allocator_warmup` preallocates a giant contiguous block. The bf16→4bit conversion briefly holds both original and quantized weights - peak exceeds what the kernel can satisfy as one contiguous allocation. `NvMapMem error 12` = ENOMEM at the kernel level.
- **Fix:** Drop to a model size that fits with headroom. Qwen3-14B QLoRA at NF4 + double-quant uses ~10-12GB during the conversion phase. **This is a hardware ceiling, not a config problem.** Tuning `max_seq_length`, `MAXN`, `jetson_clocks`, batch size - none get past the conversion-phase memory peak on 64GB unified.

### #20 - Model name verification

- **Symptom:** `RepositoryNotFoundError: 401 Client Error` on `Qwen/Qwen3-Coder-14B-Instruct`.
- **Trigger:** That model doesn't exist. Qwen3-Coder ships at 30B-A3B and 480B. No 14B Coder in Qwen3.
- **Fix:** Use `Qwen/Qwen3-14B` (general instruct, same generation, still strong on code from pretraining). A 30-second name check saves 20+ minutes of debugging a non-existent model.

### #21 - `paged_adamw_8bit` fails on Orin

- **Symptom:** Training crashes during optimizer initialization.
- **Trigger:** `optim="paged_adamw_8bit"` in `SFTConfig`.
- **Root cause:** Paged optimizers use CUDA virtual-memory paging, which Jetson's CUDA driver doesn't fully support. 8-bit precision is fine; paging isn't.
- **Fix:** `optim="adamw_8bit"`. 8-bit precision retained, paging dropped.

### #22 - transformers version: middle ground required

- **Symptom A (transformers 4.44):** `ValueError: model type qwen3_moe not recognized`.
- **Symptom B (transformers 5.x):** `ModuleNotFoundError: torch.distributed.tensor.device_mesh`.
- **Root cause:** Qwen3 MoE support landed in transformers 4.51. transformers 5.x needs torch features Jetson's torch 2.8 doesn't have.
- **Fix:** `pip install "transformers==4.51.3" "trl>=0.12,<0.14" "peft>=0.13,<0.16"`.

### #23 - bitsandbytes stock wheel: CUDA symbol error

- **Symptom:** `Error named symbol not found at line 62 in file /src/csrc/ops.cu`.
- **Trigger:** Stock `pip install bitsandbytes` (pulls the x86_64 wheel).
- **Root cause:** bitsandbytes CUDA kernels compile against specific symbol tables. PyPI wheel is built for x86_64 + standard driver. On aarch64 + CUDA 12.6 + driver 540, the symbol isn't present.
- **Fix:** Install the Jetson-built wheel from NVIDIA's Jetson AI Lab index:

```bash
pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --trusted-host pypi.jetson-ai-lab.io \
  bitsandbytes
```

### #24 - Orphaned Ollama blobs eating eMMC

- **Symptom:** `/usr/share/ollama` grew to 18GB. `ollama list` showed 2 models.
- **Trigger:** Ollama installed with default model path before `OLLAMA_MODELS=/mnt/nvme/ollama-data` was set. Early pulls went to system path; after redirect, manifests cleared but blob files remained.
- **Root cause:** Ollama doesn't garbage-collect orphaned blobs. Blob without a manifest = invisible to `ollama list`, still occupies disk.
- **Fix:** Direct deletion once manifests are confirmed empty. Reclaimed 18GB. Live models on NVMe untouched.

### #25 - NVIDIA hardcoded Bluetooth audio killswitch

- **Symptom:** Bluetooth headphones connect at device level. No audio sink appears. No bluez card in `pactl`.
- **Root cause:** NVIDIA ships JetPack with the audio, A2DP, and AVRCP BlueZ plugins **explicitly disabled** in the systemd unit override:

  ```
  /lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf:
  ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
  ```

- **Fix:** Remove the `--noplugin` switch, install `pulseaudio-module-bluetooth`, reboot. A2DP sink appears on next headset connect.

### #26 - `sys.path.insert` and PYTHONPATH override venvs

- **Symptom:** Venv reports the right package version; running the script pulls a different version from a shared bucket.
- **Trigger:** Leftover `export PYTHONPATH=/mnt/nvme/py-packages` in `.bashrc` and leftover `sys.path.insert(0, ...)` in the script, from before the venv migration.
- **Root cause:** Python import search order runs `PYTHONPATH` and script-inserted paths *before* the venv. Either shadows venv-installed packages silently.
- **Fix:** When migrating to venvs, clean both. Remove `PYTHONPATH` from shell config, `unset PYTHONPATH` in the current shell, grep scripts for `sys.path.insert` pointing at the old bucket, delete.

### #27 - `huggingface-hub` 1.0 breaks ecosystem compatibility

- **Symptom:** `ImportError: huggingface-hub>=0.34.0,<1.0 is required, but found huggingface-hub==1.16.4`.
- **Trigger:** Any project whose dependency tree pulls the latest `huggingface_hub` alongside transformers <4.58.
- **Root cause:** HuggingFace released `huggingface_hub` 1.0 in late 2025. transformers <4.58 pins to `<1.0`. New hub + old transformers = import error.
- **Fix:** Per venv, either `pip install "huggingface-hub<1.0"` or `pip install "transformers>=4.58"`. **This is exactly why per-project venvs matter.**

---

## The working recipe

### Container

```bash
docker run -it --runtime nvidia --network host --dns 8.8.8.8 --ipc=host \
  --name bake_session \
  -v /mnt/nvme:/mnt/nvme -v /home/kitten:/home/kitten -w /home/kitten \
  dustynv/l4t-pytorch:r36.4.0
```

### Stack pin

```bash
pip install "transformers==4.51.3" "trl>=0.12,<0.14" "peft>=0.13,<0.16" \
            datasets accelerate

pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
            --trusted-host pypi.jetson-ai-lab.io bitsandbytes
```

### Training (key sections)

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""    # required on Jetson

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/nvme/models/Qwen3-14B",           # local, not HF id
    quantization_config=bnb_config,
    device_map={"": 0},                     # all on cuda:0 (unified memory)
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="/mnt/nvme/bakes/bake_v2_trl_14b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.1, weight_decay=0.01, lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="steps", save_steps=200, save_total_limit=3,
    eval_strategy="steps", eval_steps=100,
    gradient_checkpointing=True,
    optim="adamw_8bit",                      # NOT paged_adamw_8bit
    seed=42, max_seq_length=2048, packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                     train_dataset=train, eval_dataset=eval,
                     args=training_args)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
```

### Post-bake: merge → GGUF → quantize → Ollama

```bash
# 1. Merge LoRA back into base (see full script in the case study)

# 2. HF safetensors → GGUF (f16 first, then quantize)
python3 convert_hf_to_gguf.py bake_v2_trl_14b_merged \
  --outfile thread-noir-14b-f16.gguf --outtype f16

./bin/llama-quantize \
  thread-noir-14b-f16.gguf thread-noir-14b.gguf Q4_K_M

# 3. Register with Ollama
ollama create NTBrain -f Modelfilejet
ollama run NTBrain
```

---

## Memory architecture

ChromaDB at `/mnt/nvme/chromadb/`. Four retrieval layers, queried per turn.

| Collection             | Role                                                                        | Count      |
|------------------------|-----------------------------------------------------------------------------|------------|
| `conversations`        | Rolling - every turn auto-stored                                            | grows      |
| `thread_noir_memory`   | Core - facts user explicitly saves with `core <text>`                       | curated    |
| `act_of_epoch`         | Foundation - first canonical session with the baked model, preserved       | 33 turns   |
| `gold_*` (39)          | Emotional anchors - each collection matches an emotion compound tag        | 615 entries|

Embedding: `sentence-transformers/all-MiniLM-L6-v2` (~90MB, CPU-fast on Jetson). **The bake gave it voice; ChromaDB gives it continuity.**

---

## Lessons

1. **Match substrate to workload.** Jetson AGX Orin is designed for inference, not training. It can train up to ~14B at QLoRA. Bigger hits memory ceilings no config can move. Stratify hardware.
2. **Pin everything, document every pin.** The arm64 wheel ecosystem moves slower than upstream. A working pinned stack is more valuable than chasing latest.
3. **Venvs per project. Not negotiable** for any system mixing ML projects. Shared `--target` buckets seem cleaner until they aren't.
4. **NVMe-everything.** By configuration, not by symlink-after-the-fact. Env vars in `.bashrc` and systemd units up front. eMMC is for the OS.
5. **Read tracebacks bottom-up; the last line names the layer that broke.** "NVML assert in CUDACachingAllocator" tells you which layer to route around, without reading the C++.
6. **The dead path is data.** Keep the script that hit the wall next to the one that worked. The story of what didn't work is the case study. Don't overwrite - fork.
7. **Voice ≠ identity.** Personality lives in LoRA weights, not TTS. The bake is the self; voice is the output channel.
8. **Trust system tools, manage your own data.** Ollama manages its blobs; you don't. But the *configuration* of where those blobs live is yours.
9. **NVIDIA disables things on Jetson by default.** Bluetooth audio is one example. When something feels broken on Jetson that "should just work" on Ubuntu, check `/lib/systemd/system/*.d/nv-*.conf` for hidden overrides.
10. **Documentation is the artifact.** This case study is part of the deliverable. The bake is real; documenting how to do it on this hardware is what makes it reproducible.

---

## What's next

- **Qwen3-TTS voice training** (in progress): custom Noir + Thread voices via `Qwen3-TTS-EasyFinetuning`, separate venv (`qwentts`)
- **Vision layer**: Qwen2.5-VL-7B for image input, sovereign
- **Smaller bakes**: Qwen3-4B retrain on the curated corpus, for edge deployment
- **Excalibur**: x86_64 RTX 5090 build for the 30B+ class bakes the Jetson can't reach

---

## Artifacts

- `bakev2.py` - TRL training script (working recipe above)
- `bake2_train_unsloth.py` - failed Unsloth attempt (preserved as reference)
- `Modelfilejet` - Ollama Modelfile for NTBrain
- `talk_jet.py` - conversation script with 4-layer memory retrieval
- `foundation_act_of_epoch.jsonl` - origin record, first canonical session
- `ingest_gold.py`, `ingest_foundation.py`, `word_bank.py` - ChromaDB ingest scripts
- `thread-noir-14b.gguf` - final Q4_K_M model, 8.4GB
- `NTBrain` (Ollama tag) - registered runnable model

---

*Document maintained alongside the build. Updates accompany each new bake or significant architecture change.*

**Contact:** [GitHub](https://github.com/miagonellm) · [Portfolio](https://miagonellm.github.io/222datascience.github.io/)
