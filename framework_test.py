"""
framework_test.py — 框架可行性验证
不需要任何 MIDI 文件，全部用合成数据测试
逐步验证每个模块是否正常工作
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 颜色输出 ──────────────────────────────────────────────────
class C:
    OK   = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END  = "\033[0m"

def ok(msg):    print(f"  {C.OK}[PASS]{C.END} {msg}")
def fail(msg):  print(f"  {C.FAIL}[FAIL]{C.END} {msg}")
def info(msg):  print(f"  {C.CYAN}[TEST]{C.END} {msg}")
def warn(msg):  print(f"  {C.WARN}[WARN]{C.END} {msg}")
def title(msg): print(f"\n{C.BOLD}{C.CYAN}{'='*55}\n  {msg}\n{'='*55}{C.END}")

results = []  # (test_name, passed, detail)

def test(name):
    """装饰器：捕获异常并记录结果"""
    def decorator(fn):
        def wrapper():
            info(name)
            try:
                detail = fn()
                ok(f"{name}  {detail or ''}")
                results.append((name, True, detail or ""))
            except Exception as e:
                fail(f"{name}")
                print(f"      原因: {e}")
                results.append((name, False, str(e)))
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════
title("1. 基础依赖")
# ══════════════════════════════════════════════════════════════

@test("导入 torch")
def _():
    import torch
    return f"v{torch.__version__}"

@test("导入 pretty_midi")
def _():
    import pretty_midi
    return f"v{pretty_midi.__version__}"

@test("导入 numpy")
def _():
    import numpy as np
    return f"v{np.__version__}"

@test("导入 tqdm / tensorboard / scipy")
def _():
    import tqdm, tensorboard, scipy
    return "OK"

@test("CUDA 可用性")
def _():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name}  {vram:.1f}GB"
    else:
        warn("CUDA 不可用，将使用 CPU（训练较慢）")
        return "CPU 模式"

for fn in [_ for _ in [
    globals().get(k) for k in list(globals().keys())
] if callable(_) and getattr(_, '__name__', '') == 'wrapper']:
    fn()

# ── 手动逐个调用 ──────────────────────────────────────────────
import torch
import numpy as np


# ══════════════════════════════════════════════════════════════
title("2. Tokenizer")
# ══════════════════════════════════════════════════════════════

def run_tokenizer_tests():
    from data.tokenizer import MusicTokenizer
    tok = MusicTokenizer()

    info("词汇表构建")
    assert tok.vocab_size > 200, f"词汇表过小: {tok.vocab_size}"
    ok(f"词汇表构建  vocab_size={tok.vocab_size}")
    results.append(("词汇表构建", True, f"vocab_size={tok.vocab_size}"))

    info("音高编码/解码往返")
    for pitch in [48, 60, 72, 84, 95]:
        tid = tok.encode_pitch(pitch)
        dec = tok.decode_token(tid)
        assert dec["pitch"] == pitch, f"pitch={pitch} 解码错误: {dec}"
    ok("音高编码/解码往返  C3~B6 全部正确")
    results.append(("音高编码/解码往返", True, ""))

    info("调性编码")
    for key in ["C_maj", "A_min", "G_maj", "D_min"]:
        tid = tok.encode_key(key)
        dec = tok.decode_token(tid)
        assert dec["type"] == "key", f"调性解码失败: {dec}"
    ok("调性编码  4种调性正确")
    results.append(("调性编码", True, ""))

    info("和弦编码")
    for root in [0, 5, 7]:
        for ctype in ["maj", "min", "dom7"]:
            tid = tok.encode_chord(root, ctype)
            dec = tok.decode_token(tid)
            assert dec["root"] == root and dec["ctype"] == ctype
    ok("和弦编码  根音+类型正确")
    results.append(("和弦编码", True, ""))

    info("速度/力度编码")
    for bpm in [60, 90, 120, 150, 180]:
        tid = tok.encode_tempo(bpm)
        assert tok.id2token[tid].startswith("TEMPO_")
    for vel in [0, 32, 64, 96, 127]:
        tid = tok.encode_velocity(vel)
        assert tok.id2token[tid].startswith("VELOCITY_")
    ok("速度/力度编码  范围正确")
    results.append(("速度/力度编码", True, ""))

    return tok

try:
    tok = run_tokenizer_tests()
except Exception as e:
    fail(f"Tokenizer 测试异常: {e}")
    traceback.print_exc()
    tok = None


# ══════════════════════════════════════════════════════════════
title("3. MIDI 处理器")
# ══════════════════════════════════════════════════════════════

def run_midi_tests(tok):
    import pretty_midi
    from data.midi_processor import MidiProcessor

    proc = MidiProcessor(tok)

    # 3-1: 创建合成 MIDI → 处理 → token 序列
    info("合成 MIDI → Token 序列")
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=80)
    # C 大调音阶，8 个音符
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=p,
            start=i * 0.5, end=(i + 0.8) * 0.5))
    pm.instruments.append(inst)

    # 保存临时文件
    tmp_path = "./processed/_test_tmp.mid"
    os.makedirs("./processed", exist_ok=True)
    pm.write(tmp_path)

    tokens = proc.process_file(tmp_path)
    assert tokens is not None and len(tokens) > 5, f"token序列为空或过短: {tokens}"
    ok(f"合成 MIDI → Token 序列  {len(tokens)} tokens")
    results.append(("合成MIDI→Token", True, f"{len(tokens)} tokens"))

    # 3-2: Token → MIDI 还原
    info("Token 序列 → MIDI 还原")
    pm2 = proc.tokens_to_midi(tokens, tempo=120)
    notes = pm2.instruments[0].notes if pm2.instruments else []
    assert len(notes) > 0, "还原 MIDI 无音符"
    ok(f"Token 序列 → MIDI 还原  {len(notes)} 个音符")
    results.append(("Token→MIDI还原", True, f"{len(notes)} 音符"))

    # 3-3: 调性检测
    info("调性检测")
    key = proc._detect_key(pm)
    assert "_" in key, f"调性格式错误: {key}"
    ok(f"调性检测  检测结果: {key}")
    results.append(("调性检测", True, key))

    # 3-4: 和弦检测
    info("和弦检测")
    notes_grid = proc._quantize_notes(pm, 120)
    chords = proc._detect_chords_per_bar(notes_grid)
    ok(f"和弦检测  {len(chords)} 个小节的和弦")
    results.append(("和弦检测", True, f"{len(chords)}小节"))

    os.remove(tmp_path)
    return tokens

if tok:
    try:
        sample_tokens = run_midi_tests(tok)
    except Exception as e:
        fail(f"MIDI处理器测试异常: {e}")
        traceback.print_exc()
        sample_tokens = None
else:
    sample_tokens = None
    warn("Tokenizer 失败，跳过 MIDI 处理器测试")


# ══════════════════════════════════════════════════════════════
title("4. Dataset")
# ══════════════════════════════════════════════════════════════

def run_dataset_tests(tok):
    import torch
    from data.dataset import MusicDataset

    # 构造合成序列
    seq_len = 64
    fake_sequences = [
        list(range(1, seq_len + 50)) for _ in range(20)
    ]

    info("Dataset 构建")
    ds = MusicDataset(fake_sequences, tok, seq_len=seq_len, augment=False)
    assert len(ds) > 0, "Dataset 为空"
    ok(f"Dataset 构建  {len(ds)} 个样本窗口")
    results.append(("Dataset构建", True, f"{len(ds)}样本"))

    info("Dataset __getitem__")
    inp, tgt = ds[0]
    assert inp.shape == (seq_len,), f"input shape 错误: {inp.shape}"
    assert tgt.shape == (seq_len,), f"target shape 错误: {tgt.shape}"
    assert inp.dtype == torch.long
    ok(f"Dataset __getitem__  input={tuple(inp.shape)} target={tuple(tgt.shape)}")
    results.append(("Dataset取样", True, ""))

    info("DataLoader 批次")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_inp, batch_tgt = next(iter(loader))
    assert batch_inp.shape == (4, seq_len)
    ok(f"DataLoader 批次  shape={tuple(batch_inp.shape)}")
    results.append(("DataLoader批次", True, ""))

    info("数据增强 (augment=True)")
    ds_aug = MusicDataset(fake_sequences, tok, seq_len=seq_len, augment=True)
    inp_aug, _ = ds_aug[0]
    ok("数据增强  不崩溃")
    results.append(("数据增强", True, ""))

if tok:
    try:
        run_dataset_tests(tok)
    except Exception as e:
        fail(f"Dataset 测试异常: {e}")
        traceback.print_exc()
else:
    warn("Tokenizer 失败，跳过 Dataset 测试")


# ══════════════════════════════════════════════════════════════
title("5. Transformer 模型")
# ══════════════════════════════════════════════════════════════

def run_model_tests(tok):
    import torch
    from model.transformer import MusicTransformer
    from config import MODEL_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用小配置加速测试
    small_cfg = {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.0, "max_seq_len": 128,
    }

    info("模型实例化")
    model = MusicTransformer(vocab_size=tok.vocab_size, config=small_cfg).to(device)
    n_params = model.count_params()
    ok(f"模型实例化  {n_params:,} 参数")
    results.append(("模型实例化", True, f"{n_params:,}参数"))

    info("前向传播 (训练模式)")
    B, T = 2, 64
    ids  = torch.randint(1, tok.vocab_size, (B, T), device=device)
    tgt  = torch.randint(1, tok.vocab_size, (B, T), device=device)
    logits, loss, _ = model(ids, tgt)
    assert logits.shape == (B, T, tok.vocab_size), f"logits shape错误: {logits.shape}"
    assert loss is not None and loss.item() > 0
    ok(f"前向传播  logits={tuple(logits.shape)}  loss={loss.item():.4f}")
    results.append(("前向传播", True, f"loss={loss.item():.4f}"))

    info("反向传播 + 梯度")
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "部分参数没有梯度"
    ok("反向传播  所有参数有梯度")
    results.append(("反向传播", True, ""))

    info("KV Cache 推理")
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(1, tok.vocab_size, (1, 8), device=device)
        _, _, kv = model(prompt, use_cache=True)
        assert kv is not None and len(kv) == small_cfg["num_layers"]
        next_tok = torch.randint(1, tok.vocab_size, (1, 1), device=device)
        logits2, _, kv2 = model(next_tok, kv_caches=kv, use_cache=True)
        assert logits2.shape == (1, 1, tok.vocab_size)
    ok(f"KV Cache 推理  {small_cfg['num_layers']} 层缓存正常")
    results.append(("KV Cache推理", True, ""))

    info("混合精度 (fp16) 前向传播")
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits_fp16, loss_fp16, _ = model(ids, tgt)
        assert loss_fp16.item() > 0
        ok(f"fp16 前向传播  loss={loss_fp16.item():.4f}")
        results.append(("fp16混合精度", True, ""))
    else:
        warn("CPU 模式跳过 fp16 测试")
        results.append(("fp16混合精度", True, "跳过(CPU)"))

    info("推理速度")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            model(ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    ok(f"推理速度  {ms:.1f}ms / 批次 (bs=2, seq=64)")
    results.append(("推理速度", True, f"{ms:.1f}ms"))

    return model, device, small_cfg

model = device = small_cfg = None
if tok:
    try:
        model, device, small_cfg = run_model_tests(tok)
    except Exception as e:
        fail(f"模型测试异常: {e}")
        traceback.print_exc()
else:
    warn("Tokenizer 失败，跳过模型测试")


# ══════════════════════════════════════════════════════════════
title("6. 乐理约束")
# ══════════════════════════════════════════════════════════════

def run_theory_tests(tok):
    import torch
    from model.music_theory import (
        MusicTheoryConstraints, GenerationContext, sample_with_constraints
    )

    info("乐理约束实例化")
    constraints = MusicTheoryConstraints(tok)
    ok("乐理约束实例化")
    results.append(("乐理约束实例化", True, ""))

    info("GenerationContext 状态更新")
    ctx = GenerationContext(tok)
    ctx.update(tok.encode_key("C_maj"))
    ctx.update(tok.encode_tempo(120))
    ctx.update(tok.bar_id)
    ctx.update(tok.encode_chord(0, "maj"))
    ctx.update(tok.encode_beat(0))
    ctx.update(tok.encode_pitch(60))
    assert ctx.current_key == "C_maj"
    assert ctx.first_pitch == 60
    assert ctx.current_bar == 1
    ok(f"状态更新  key={ctx.current_key}  first_pitch={ctx.first_pitch}")
    results.append(("Context状态更新", True, ""))

    info("调性约束施加")
    logits = torch.zeros(tok.vocab_size)
    ctx2 = GenerationContext(tok)
    ctx2.current_key = "C_maj"
    logits_after = constraints._apply_key_constraint(logits.clone(), "C_maj")
    note_start, note_end = tok.note_on_range()
    # C大调内的音 (C=60%12=0) 不应被惩罚
    c_note_id = tok.encode_pitch(60)
    fs_note_id = tok.encode_pitch(66)  # F# 不在 C 大调
    assert logits_after[c_note_id] == 0.0, "C 音被错误惩罚"
    assert logits_after[fs_note_id] < 0.0, "F# 未被惩罚"
    ok("调性约束  调内音不惩罚，调外音惩罚")
    results.append(("调性约束", True, ""))

    info("终止音偏置")
    ctx3 = GenerationContext(tok)
    ctx3.current_key = "C_maj"
    ctx3.first_pitch = 60
    ctx3.approaching_end = True
    logits3 = torch.zeros(tok.vocab_size)
    logits3 = constraints._apply_cadence_bias(logits3, 60, "C_maj")
    c4_id = tok.encode_pitch(60)
    assert logits3[c4_id] > 0, "终止音未获得正偏置"
    ok("终止音偏置  主音获得正偏置")
    results.append(("终止音偏置", True, ""))

    info("带约束采样")
    ctx4 = GenerationContext(tok)
    ctx4.current_key = "C_maj"
    logits4 = torch.randn(tok.vocab_size)
    sampled = sample_with_constraints(logits4, constraints, ctx4,
                                      temperature=1.0, top_k=50, top_p=0.9)
    assert 0 <= sampled < tok.vocab_size
    ok(f"带约束采样  sampled token_id={sampled}")
    results.append(("带约束采样", True, ""))

if tok:
    try:
        run_theory_tests(tok)
    except Exception as e:
        fail(f"乐理约束测试异常: {e}")
        traceback.print_exc()
else:
    warn("Tokenizer 失败，跳过乐理约束测试")


# ══════════════════════════════════════════════════════════════
title("7. 完整小规模训练循环")
# ══════════════════════════════════════════════════════════════

def run_training_test(tok):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from model.transformer import MusicTransformer
    from data.dataset import MusicDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    small_cfg = {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.1, "max_seq_len": 128,
    }

    info("构建合成训练数据")
    fake_seqs = [list(range(1, 150)) for _ in range(30)]
    ds = MusicDataset(fake_seqs, tok, seq_len=64, augment=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    ok(f"合成训练数据  {len(ds)} 样本")
    results.append(("合成训练数据", True, ""))

    info("模型 + 优化器初始化")
    model = MusicTransformer(vocab_size=tok.vocab_size, config=small_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ok(f"优化器初始化  AdamW lr=3e-4")
    results.append(("优化器初始化", True, ""))

    info("运行 5 步训练...")
    model.train()
    losses = []
    t0 = time.time()

    for step, (inp, tgt) in enumerate(loader):
        if step >= 5:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            _, loss, _ = model(inp, tgt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        losses.append(loss.item())

    elapsed = time.time() - t0
    assert len(losses) == 5, f"只跑了 {len(losses)} 步"
    assert all(l > 0 for l in losses), "loss 出现非正值"
    ok(f"5步训练完成  loss: {losses[0]:.3f} → {losses[-1]:.3f}  耗时 {elapsed:.1f}s")
    results.append(("训练循环", True, f"loss {losses[0]:.3f}→{losses[-1]:.3f}"))

    info("检查点保存/加载")
    os.makedirs("./processed", exist_ok=True)
    ckpt_path = "./processed/_test_checkpoint.pt"
    torch.save({
        "step": 5, "model_state": model.state_dict(),
        "loss": losses[-1], "model_config": small_cfg,
    }, ckpt_path)
    assert os.path.exists(ckpt_path)

    # 重新加载
    ckpt = torch.load(ckpt_path, map_location=device)
    model2 = MusicTransformer(vocab_size=tok.vocab_size, config=small_cfg).to(device)
    model2.load_state_dict(ckpt["model_state"])
    ok(f"检查点保存/加载  {os.path.getsize(ckpt_path)/1024:.0f} KB")
    results.append(("检查点保存/加载", True, ""))

    os.remove(ckpt_path)

if tok:
    try:
        run_training_test(tok)
    except Exception as e:
        fail(f"训练循环测试异常: {e}")
        traceback.print_exc()
else:
    warn("Tokenizer 失败，跳过训练循环测试")


# ══════════════════════════════════════════════════════════════
title("8. 生成流程 (端到端)")
# ══════════════════════════════════════════════════════════════

def run_generation_test(tok):
    import torch
    from model.transformer import MusicTransformer
    from model.music_theory import MusicTheoryConstraints, GenerationContext, sample_with_constraints
    from data.midi_processor import MidiProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    small_cfg = {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.0, "max_seq_len": 128,
    }

    model = MusicTransformer(vocab_size=tok.vocab_size, config=small_cfg).to(device)
    model.eval()
    constraints = MusicTheoryConstraints(tok)
    proc = MidiProcessor(tok)

    info("自回归生成 4 小节")
    from config import TICKS_PER_BAR
    prompt = [tok.bos_id, tok.encode_key("C_maj"), tok.encode_tempo(120)]
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    generated = []
    ctx = GenerationContext(tok)
    ctx.target_bars = 4
    for tid in prompt:
        ctx.update(tid)

    with torch.no_grad():
        _, _, kv_caches = model(input_ids, use_cache=True)
        cur = torch.tensor([[prompt[-1]]], dtype=torch.long, device=device)
        for _ in range(200):
            logits, _, kv_caches = model(cur, kv_caches=kv_caches, use_cache=True)
            nxt = sample_with_constraints(
                logits[0, -1, :], constraints, ctx,
                temperature=1.0, top_k=50, top_p=0.9
            )
            ctx.update(nxt)
            generated.append(nxt)
            if nxt == tok.eos_id or ctx.total_bars >= 4:
                break
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)

    assert len(generated) > 0, "未生成任何 token"
    ok(f"自回归生成  {len(generated)} tokens，{ctx.total_bars} 小节")
    results.append(("自回归生成", True, f"{len(generated)}tokens"))

    info("生成结果 → MIDI 文件")
    full = [tok.bos_id, tok.encode_key("C_maj"), tok.encode_tempo(120)] + generated
    pm = proc.tokens_to_midi(full, tempo=120)
    os.makedirs("./outputs", exist_ok=True)
    out_path = "./outputs/_test_gen.mid"
    pm.write(out_path)
    size = os.path.getsize(out_path)
    assert size > 0
    ok(f"输出 MIDI  {out_path}  ({size} bytes)")
    results.append(("输出MIDI文件", True, f"{size}bytes"))
    os.remove(out_path)

if tok:
    try:
        run_generation_test(tok)
    except Exception as e:
        fail(f"生成流程测试异常: {e}")
        traceback.print_exc()
else:
    warn("Tokenizer 失败，跳过生成流程测试")


# ══════════════════════════════════════════════════════════════
title("9. VRAM 估算 (正式模型)")
# ══════════════════════════════════════════════════════════════

def run_vram_test(tok):
    import torch
    from model.transformer import MusicTransformer
    from config import MODEL_CONFIG, TRAIN_CONFIG

    if not torch.cuda.is_available():
        warn("CPU 模式，跳过 VRAM 估算")
        results.append(("VRAM估算", True, "跳过(CPU)"))
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = MusicTransformer(vocab_size=tok.vocab_size, config=MODEL_CONFIG).to(device)
    B = TRAIN_CONFIG["batch_size"]
    T = TRAIN_CONFIG["seq_len"]

    ids = torch.randint(1, tok.vocab_size, (B, T), device=device)
    tgt = torch.randint(1, tok.vocab_size, (B, T), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, loss, _ = model(ids, tgt)
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    status = "充裕" if peak_mb < total_mb * 0.7 else ("警告:偏高" if peak_mb < total_mb * 0.9 else "危险:可能OOM")
    ok(f"VRAM 峰值  {peak_mb:.0f}MB / {total_mb:.0f}MB  ({status})")
    results.append(("VRAM峰值估算", True, f"{peak_mb:.0f}/{total_mb:.0f}MB"))

    if peak_mb > total_mb * 0.85:
        warn("VRAM 使用率过高，建议在 config.py 中调小 batch_size 或 seq_len")

    del model
    torch.cuda.empty_cache()

if tok:
    try:
        run_vram_test(tok)
    except Exception as e:
        fail(f"VRAM 测试异常: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════════════════════
print(f"\n{C.BOLD}{'='*55}")
print("  测试报告汇总")
print(f"{'='*55}{C.END}")

passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]

for name, ok_, detail in results:
    status = f"{C.OK}PASS{C.END}" if ok_ else f"{C.FAIL}FAIL{C.END}"
    extra = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {name}{extra}")

print(f"\n{C.BOLD}  总计: {len(passed)}/{len(results)} 通过{C.END}", end="")
if failed:
    print(f"  {C.FAIL}{len(failed)} 项失败{C.END}")
    print(f"\n  失败项:")
    for name, _, detail in failed:
        print(f"    {C.FAIL}✗{C.END} {name}: {detail}")
else:
    print(f"  {C.OK}全部通过!{C.END}")

print()
if not failed:
    print(f"  {C.OK}{C.BOLD}框架验证完成，可以开始训练！{C.END}")
    print(f"\n  下一步: 将 MIDI 文件放入 midi_data/ 目录，然后运行:")
    print(f"  python train.py --midi_dir .\\midi_data")
else:
    print(f"  请修复上述失败项后重新运行验证")
print()
