import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F


def make_dummy_data(*shape, seed=42):
    np.random.seed(seed)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)


def assert_near(a, b, atol=1e-4, name=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    diff = np.abs(a - b).max()
    assert diff < atol, f"max diff = {diff:.2e} >= {atol}"
    print(f"{name}: max diff = {diff:.2e}  ->  PASSED")

# === 1. Simple MLP ===
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=64, h1=128, h2=64, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPLSTMPolicy(nn.Module):
    def __init__(self, in_dim=64, h1=256, h2 = 256, lstm_dim=32, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, lstm_dim)
        self.lstm = nn.LSTMCell(lstm_dim, lstm_dim)
        self.head = nn.Linear(lstm_dim, out_dim)

    def forward(self, x, hx, cx):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        hx, cx = self.lstm(x, (hx, cx))
        return self.head(hx), (hx, cx)

class SimpleCNN(nn.Module):
    def __init__(self, in_dim=20, cnn_channels=32, hidden_size=256, hidden_lstm_dim=128, out_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, cnn_channels, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=2, stride=1)
        self.fc = nn.Linear(cnn_channels, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_lstm_dim)
        self.head = nn.Linear(hidden_lstm_dim, out_dim)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        hx, cx = self.lstm(x, (hx, cx))
        return self.head(hx), (hx, cx)

def _conv2d_out_hw(H, W, kH, kW, s=1, p=0, d=1):
    Hout = (H + 2*p - d*(kH - 1) - 1)//s + 1
    Wout = (W + 2*p - d*(kW - 1) - 1)//s + 1
    return Hout, Wout


def batched_conv2d(x, W_stack, b, stride=1, padding=0, dilation=1, layer_groups=None):
    """
    x : [B, C, H, W]
    W : [B, out_c, in_c, kH, kW]
    b : [B, out_c]
    """
    B, Cin, H, W = x.shape
    Bw, Cout, CinW, kH, kW = W_stack.shape  
    assert Bw == B and b.shape == (B, Cout)
    if layer_groups is None:
        assert Cin % CinW == 0, f"Cin={Cin} not divisible by CinW={CinW}"
        layer_groups = Cin // CinW
    assert Cout % layer_groups == 0
    Cout_g = Cout // layer_groups
    Kg = CinW * kH * kW               # per-group kernel length
    K  = Cin  * kH * kW
    X_col = F.unfold(x, (kH,kW), padding=padding, stride=stride, dilation=dilation)
    B_,K_, L_ = X_col.shape
    assert K_ == K

    if layer_groups == 1:
        Wm = W_stack.reshape(B, Cout, K)
        out = torch.bmm(Wm, X_col) + b.unsqueeze(-1)
    else:
        print("wazoo")

    Hout, Wout = _conv2d_out_hw(H, W, kH, kW, s=stride, p=padding, d=dilation)
    out = out.view(B, Cout, Hout, Wout)
    return out

def per_sample_grouped_conv2d(x, W_stack, b_stack, version_ids, stride = 1, padding =0, dilation =1):
    """
    x : [B, C, H, W]
    W_stack : [B, out_c, in_c, kH, kW]
    b_stack : [B, out_c]
    """

    B, Cin, H, W = x.shape
    P, Cout, CinW, kH, kW = W_stack.shape
    assert Cin == CinW
    assert b_stack.shape == (P, Cout)

    device = x.device
    idx = torch.as_tensor(version_ids, device=device, dtype=torch.long)
    assert idx.shape[0] == B
    W_per = W_stack.index_select(0, idx)
    b_per = b_stack.index_select(0, idx)

    x_big = x.reshape(1, B*Cin, H,W).contiguous()
    W_big = W_per.reshape(B*Cout, Cin, kH, kW).contiguous()
    b_big = b_per.reshape(B*Cout).contiguous()

    out_big = F.conv2d(x_big, W_big, b_big, stride=stride, padding=padding, dilation=dilation, groups = B)

    _,_, Hout, Wout = out_big.shape
    out = out_big.view(B, Cout, Hout, Wout).contiguous()
    return out

def create_policies_mlp_lstm(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [MLPLSTMPolicy() for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies

def create_cnn_policies(num_policies=4, base_seed=42):
    policies = []
    for i in range(num_policies):
        torch.manual_seed(base_seed + i)  # ← DIFFERENT SEED
        p = SimpleCNN(in_dim=20).to('cuda')
        policies.append(p)
    return policies

'''def create_cnn_policies(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [SimpleCNN(in_dim=20) for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies
'''
def build_weight_stacks(policies):
    W = {}
    b = {}
    for name in ['fc1', 'fc2', 'fc3', 'head']:
        W[name] = torch.stack([p.state_dict()[f'{name}.weight'] for p in policies])
        b[name] = torch.stack([p.state_dict()[f'{name}.bias']   for p in policies])

    lstm_ih = torch.stack([p.state_dict()['lstm.weight_ih'] for p in policies])
    lstm_hh = torch.stack([p.state_dict()['lstm.weight_hh'] for p in policies])
    lstm_b_ih = torch.stack([p.state_dict()['lstm.bias_ih'] for p in policies])
    lstm_b_hh = torch.stack([p.state_dict()['lstm.bias_hh'] for p in policies])
    return {'W': W, 'b': b, 
            'lstm_ih': lstm_ih, 'lstm_hh': lstm_hh, 
            'lstm_b_ih': lstm_b_ih, 'lstm_b_hh': lstm_b_hh}

def build_weight_stacks_cnn(policies):
    W = {}
    b = {}
    for name in ['fc', 'head']:
        W[name] = torch.stack([p.state_dict()[f'{name}.weight'] for p in policies])
        b[name] = torch.stack([p.state_dict()[f'{name}.bias']   for p in policies])

    lstm_ih = torch.stack([p.state_dict()['lstm.weight_ih'] for p in policies])
    lstm_hh = torch.stack([p.state_dict()['lstm.weight_hh'] for p in policies])
    lstm_b_ih = torch.stack([p.state_dict()['lstm.bias_ih'] for p in policies])
    lstm_b_hh = torch.stack([p.state_dict()['lstm.bias_hh'] for p in policies])
    return {'W': W, 'b': b, 
            'lstm_ih': lstm_ih, 'lstm_hh': lstm_hh, 
            'lstm_b_ih': lstm_b_ih, 'lstm_b_hh': lstm_b_hh}


def optimized_loop_forward(obs_sorted, version_ids, policies):
    B = obs_sorted.shape[0]
    out = torch.empty(B, 4)
    start = 0
    for v in torch.unique(version_ids):
        mask = (version_ids == v)
        end = start + mask.sum().item()
        sub_obs = obs_sorted[start:end]
        with torch.no_grad():
            out[start:end] = policies[v](sub_obs)
        start = end
    return out

def optimized_loop_forward_lstm(obs_sorted, version_ids,h_sorted, c_sorted, policies):
    device = obs_sorted.device
    B = obs_sorted.shape[0]
    out = torch.empty(B, 4, device=device)
    new_h = torch.empty_like(h_sorted)
    new_c = torch.empty_like(c_sorted)
    start = 0
    for v in torch.unique(version_ids):
        mask = (version_ids == v)
        end = start + mask.sum().item()
        sub_obs = obs_sorted[start:end]
        sub_h = h_sorted[start:end]
        sub_c = c_sorted[start:end]
        with torch.no_grad():
            #out[start:end] = policies[v](sub_obs)
            act, (nh, nc) = policies[v](sub_obs, sub_h, sub_c)
        out[start:end] = act
        new_h[start:end] = nh
        new_c[start:end] = nc
        start = end
    return out, new_h, new_c

def stacked_forward_mlp(obs, version_ids, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)


    x = obs

    for name in ['fc1', 'fc2']:
        W = stacks['W'][name][idx]
        b = stacks['b'][name][idx]
        x = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b

        x = F.relu(x)

    # fc3
    W = stacks['W']['fc3'][idx]
    b = stacks['b']['fc3'][idx]
    out = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b


    return out

def stacked_forward_mlp_lstm(obs, version_ids, h, c, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)
    x = obs

    # MLP
    for name in ('fc1', 'fc2', 'fc3'):
        W = stacks['W'][name][idx]
        b = stacks['b'][name][idx]
        WT = W.transpose(-2,-1).contiguous()
        x = torch.bmm(x.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b
        if name != 'fc3':
            x = F.relu(x)

    # LSTM: 4 linear ops
    W_ih = stacks['lstm_ih'][idx].contiguous()      # [B, 4*H, D]
    W_hh = stacks['lstm_hh'][idx].contiguous()      # [B, 4*H, H]
    b_tot = (stacks['lstm_b_ih'][idx]+ stacks['lstm_b_hh'][idx]).contiguous()    # [B, 4*H]

    assert W_ih.dim() == 3 and W_hh.dim() == 3
    B, fourH, Dlstm = W_ih.shape
    B2, fourH2, Hdim = W_hh.shape
    assert B == B2
    assert fourH == fourH2
    Wcat_T = torch.empty(B, Dlstm+Hdim, fourH, device=W_ih.device, dtype=W_ih.dtype)
    Wcat_T[:, :Dlstm, :] = W_ih.transpose(-2, -1)
    Wcat_T[:, Dlstm:, :] = W_hh.transpose(-2, -1)
    Wcat_T = Wcat_T.contiguous()
    z = torch.cat([x, h], dim=1)                                          # [B, Dlstm+H]

    gates = torch.bmm(z.unsqueeze(1), Wcat_T).squeeze(1) + b_tot           # [B, 4H]
    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    new_c = f * c + i * g
    new_h = o * torch.tanh(new_c)

    # Head
    W = stacks['W']['head'][idx]
    b = stacks['b']['head'][idx]
    out = torch.bmm(new_h.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b

    return out, new_h, new_c

def conv_features_per_policy(obs_sorted, version_ids, policies):
    uniq, counts = torch.unique(version_ids, return_counts=True)
    start = 0
    feats = []
    for v,c in zip(uniq.tolist(), counts.tolist()):
        sub = obs_sorted[start:start+c]
        p = policies[v]
        x = F.relu(p.conv1(sub))
        x = F.relu(p.conv2(x))
        x = x.flatten(1)
        feats.append(x)
        start += c
        
    return torch.cat(feats, dim=0)
def stacked_forward_cnn_mlp_lstm(obs_sorted, version_ids,h, c, policies, stacks):
    idx = torch.as_tensor(version_ids, device=obs_sorted.device, dtype=torch.long)
    
    B = obs_sorted.size(0)
    x = conv_features_per_policy(obs_sorted, version_ids, policies)
    x = x.contiguous()

    W_fc = stacks['W']['fc'][idx]
    b_fc = stacks['b']['fc'][idx]
    x = torch.bmm(x.unsqueeze(1), W_fc.transpose(1, 2)).squeeze(1) + b_fc
    x = F.relu(x)

    W_ih = stacks['lstm_ih'][idx]      # [B, 4*H, D]
    W_hh = stacks['lstm_hh'][idx]      # [B, 4*H, H]
    b_ih = stacks['lstm_b_ih'][idx]    # [B, 4*H]
    b_hh = stacks['lstm_b_hh'][idx]    # [B, 4*H]
    gates = torch.bmm(x.unsqueeze(1), W_ih.transpose(-2, -1)).squeeze(1) + \
            torch.bmm(h.unsqueeze(1), W_hh.transpose(-2, -1)).squeeze(1) + \
            b_ih + b_hh

    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    new_c = f * c + i * g
    new_h = o * torch.tanh(new_c)

    W = stacks['W']['head'][idx]
    b = stacks['b']['head'][idx]
    out = torch.bmm(new_h.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b
    return out, new_h, new_c

def _evt(): return torch.cuda.Event(enable_timing=True)


@torch.no_grad()
def stacked_forward_mlp_lstm_timed(obs, version_ids, h, c, stacks):
    """
    Times:
      - gather_*         : per-sample weight/bias gathers
      - transpose_*      : one-time pretranspose for bmm
      - gemm_fc*         : MLP matmuls
      - gemm_lstm_fused  : fused gates matmul (one bmm)
      - lstm_pointwise   : all sigmoid/tanh/cell/hidden updates
      - gemm_head        : head matmul
    """
    assert obs.is_cuda and h.is_cuda and c.is_cuda
    device = obs.device
    idx = version_ids.to(device=device, dtype=torch.long)
    x = obs

    timings = {}
    def rec(name, start, end):
        torch.cuda.synchronize()
        timings[name] = start.elapsed_time(end)  # ms

    def time_section(name, fn):
        s, e = _evt(), _evt()
        s.record()
        out = fn()
        e.record(); rec(name, s, e)
        return out

    # ------------------- fc1 -------------------
    W_fc1 = time_section("gather_fc1",      lambda: stacks['W']['fc1'][idx].contiguous())  # [B, h1, Din]
    b_fc1 = time_section("gather_fc1_b",    lambda: stacks['b']['fc1'][idx].contiguous())  # [B, h1]
    W_fc1_T = time_section("transpose_fc1", lambda: W_fc1.transpose(1, 2).contiguous())    # [B, Din, h1]
    x = time_section("gemm_fc1",            lambda: torch.bmm(x.unsqueeze(1), W_fc1_T).squeeze(1) + b_fc1)
    x = time_section("act_fc1",             lambda: F.relu(x))

    # ------------------- fc2 -------------------
    W_fc2 = time_section("gather_fc2",      lambda: stacks['W']['fc2'][idx].contiguous())  # [B, h2, h1]
    b_fc2 = time_section("gather_fc2_b",    lambda: stacks['b']['fc2'][idx].contiguous())  # [B, h2]
    W_fc2_T = time_section("transpose_fc2", lambda: W_fc2.transpose(1, 2).contiguous())    # [B, h1, h2]
    x = time_section("gemm_fc2",            lambda: torch.bmm(x.unsqueeze(1), W_fc2_T).squeeze(1) + b_fc2)
    x = time_section("act_fc2",             lambda: F.relu(x))

    # ------------------- fc3 -------------------
    W_fc3 = time_section("gather_fc3",      lambda: stacks['W']['fc3'][idx].contiguous())  # [B, Dlstm, h2]
    b_fc3 = time_section("gather_fc3_b",    lambda: stacks['b']['fc3'][idx].contiguous())  # [B, Dlstm]
    W_fc3_T = time_section("transpose_fc3", lambda: W_fc3.transpose(1, 2).contiguous())    # [B, h2, Dlstm]
    x = time_section("gemm_fc3",            lambda: torch.bmm(x.unsqueeze(1), W_fc3_T).squeeze(1) + b_fc3)
    # x: [B, Dlstm]

    # ------------------- LSTM (fused gates → one bmm) -------------------
    # Gather once
    W_ih = time_section("gather_lstm_ih",   lambda: stacks['lstm_ih'][idx].contiguous())   # [B, 4H, Dlstm]
    W_hh = time_section("gather_lstm_hh",   lambda: stacks['lstm_hh'][idx].contiguous())   # [B, 4H, H]
    b_ih = time_section("gather_lstm_b_ih", lambda: stacks['lstm_b_ih'][idx].contiguous()) # [B, 4H]
    b_hh = time_section("gather_lstm_b_hh", lambda: stacks['lstm_b_hh'][idx].contiguous()) # [B, 4H]

    # Build concatenated weights once (pretranspose)
    Wcat_T = time_section(
        "transpose_lstm_cat",
        lambda: torch.cat([W_ih, W_hh], dim=2).transpose(1, 2).contiguous()
    )  # [B, Dlstm+H, 4H]
    b_tot = b_ih + b_hh  # [B, 4H]

    # One bmm for the gates
    def _gemm_lstm():
        z = torch.cat([x, h], dim=1)                                          # [B, Dlstm+H]
        return torch.bmm(z.unsqueeze(1), Wcat_T).squeeze(1) + b_tot           # [B, 4H]
    gates = time_section("gemm_lstm_fused", _gemm_lstm)

    # Pointwise (all activations + state updates in one timed block)
    def _lstm_pointwise():
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); g = torch.tanh(g); o = torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c
    new_h, new_c = time_section("lstm_pointwise", _lstm_pointwise)

    # ------------------- Head -------------------
    W_head = time_section("gather_head",      lambda: stacks['W']['head'][idx].contiguous())   # [B, O, H]
    b_head = time_section("gather_head_b",    lambda: stacks['b']['head'][idx].contiguous())   # [B, O]
    W_head_T = time_section("transpose_head", lambda: W_head.transpose(1, 2).contiguous())     # [B, H, O]
    out = time_section("gemm_head",           lambda: torch.bmm(new_h.unsqueeze(1), W_head_T).squeeze(1) + b_head)

    # Percentages
    total_ms = sum(timings.values()) if timings else 1.0
    pct = {k: 100.0 * v / total_ms for k, v in timings.items()}

    return out, new_h, new_c, timings, pct

def benchmark_mlp_lstm(num_iters=1000, batch_size=512, num_versions=40):
    policies = create_policies_mlp_lstm(num_policies=num_versions)
    policies_gpu = [p.to('cuda') for p in policies]
    stacks = build_weight_stacks(policies)

    # Generate data
    obs = torch.randn(batch_size, 64, device='cuda')
    h = torch.randn(batch_size, 32, device='cuda')
    c = torch.randn(batch_size, 32, device='cuda')
    versions = torch.randint(0, num_versions, size=(batch_size,), device='cuda')
    # === SORT for fair optimized loop ===
    sort_idx = torch.argsort(versions)
    obs_sorted = obs.index_select(0, sort_idx)
    h_sorted = h.index_select(0, sort_idx)
    c_sorted = c.index_select(0, sort_idx)
    versions_sorted = versions.index_select(0, sort_idx)
    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')


    # Warmup
    for _ in range(10):
        optimized_loop_forward_lstm(obs_sorted, versions_sorted, h_sorted, c_sorted, policies_gpu)
        stacked_forward_mlp_lstm(obs, versions, h, c, gpu_stacks)

    # === TIME OPTIMIZED LOOP ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        optimized_loop_forward_lstm(obs_sorted, versions_sorted, h_sorted, c_sorted, policies_gpu)
    torch.cuda.synchronize()
    opt_loop_time = time.time() - t0

    # === TIME STACKED ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        stacked_forward_mlp_lstm(obs, versions, h, c, gpu_stacks)
    torch.cuda.synchronize()
    stack_time = time.time() - t0

    print(f"\n=== FAIR BENCHMARK (iters={num_iters}, B={batch_size}, V={num_versions}) ===")
    print(f"Old Method: {opt_loop_time:.3f}s  ->  {opt_loop_time/num_iters*1000:.2f} ms/step")
    print(f"New Method: {stack_time:.3f}s  ->  {stack_time/num_iters*1000:.2f} ms/step")
    print(f"Speedup        : {opt_loop_time/stack_time:.1f}x")

def benchmark_cnn(num_iters=1000, batch_size=512, num_versions=40):
    policies = create_cnn_policies(num_policies=num_versions)
    policies_gpu = [p.to('cuda') for p in policies]
    stacks = build_weight_stacks_cnn(policies)

    B = 1024
    C, H, W = 20, 8, 8
    obs = torch.randn(B, C, H, W, device='cuda')
    versions = np.random.randint(0, 4, size=B)
    h = torch.randn(B, 128, device='cuda')
    c = torch.randn(B, 128, device='cuda')

    # ----- Sort for loop -----
    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    h_sorted = h.cpu()[sort_idx]
    c_sorted = c.cpu()[sort_idx]
    versions_sorted = versions[sort_idx]

    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')


    # Warmup
    for _ in range(10):
        optimized_loop_forward_lstm(obs_sorted.cpu(), torch.from_numpy(versions_sorted), h_sorted, c_sorted, [p.cpu() for p in policies])
        stacked_forward_cnn_mlp_lstm(obs_sorted.to('cuda'), torch.from_numpy(versions), h_sorted, c_sorted, policies, gpu_stacks)

    # === TIME OPTIMIZED LOOP ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        optimized_loop_forward(obs_sorted.cpu(), torch.from_numpy(versions_sorted), [p.cpu() for p in policies])
    torch.cuda.synchronize()
    opt_loop_time = time.time() - t0

    # === TIME STACKED ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        stacked_forward_cnn_mlp_lstm(obs_sorted, torch.from_numpy(versions), gpu_stacks)
    torch.cuda.synchronize()
    stack_time = time.time() - t0

    print(f"\n=== FAIR BENCHMARK (iters={num_iters}, B={batch_size}, V={num_versions}) ===")
    print(f"Old Method: {opt_loop_time:.3f}s  ->  {opt_loop_time/num_iters*1000:.2f} ms/step")
    print(f"New Method: {stack_time:.3f}s  ->  {stack_time/num_iters*1000:.2f} ms/step")
    print(f"Speedup        : {opt_loop_time/stack_time:.1f}x")


def test_mlp_lstm():
    print("\n=== MLP + LSTM TEST ===")
    policies = create_policies_mlp_lstm(4)
    policies_gpu = [p.to('cuda') for p in policies]
    stacks = build_weight_stacks(policies)
    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')
     # Generate data
    batch_size = 256
    obs = torch.randn(batch_size, 64, device='cuda')
    h = torch.randn(batch_size, 64, device='cuda')
    c = torch.randn(batch_size, 64, device='cuda')
    versions = torch.randint(0, 4, size=(batch_size,), device='cuda')
    # === SORT for fair optimized loop ===
    sort_idx = torch.argsort(versions)
    obs_sorted = obs.index_select(0, sort_idx)
    h_sorted = h.index_select(0, sort_idx)
    c_sorted = c.index_select(0, sort_idx)
    versions_sorted = versions.index_select(0, sort_idx)

    loop_out, loop_h, loop_c = optimized_loop_forward_lstm(
        obs_sorted, versions_sorted, h_sorted, c_sorted, policies_gpu
    )

    stack_out, stack_h, stack_c = stacked_forward_mlp_lstm(
        obs, versions, h, c, gpu_stacks
    )

    for _ in range(10):
        _ = stacked_forward_mlp_lstm_timed(obs, versions, h, c, gpu_stacks)

    torch.cuda.synchronize()
    iters = 200
    acc_ms = {}
    for _ in range(iters):
        _, _, _, t, _ = stacked_forward_mlp_lstm_timed(obs, versions, h, c, stacks)
        for k,v in t.items():
            acc_ms[k] = acc_ms.get(k, 0.0) + v
    torch.cuda.synchronize()

    total = sum(acc_ms.values())
    report = {k: (v/iters, 100.0*v/total) for k,v in acc_ms.items()}  # (ms/step, %)
    for k,(ms,pc) in sorted(report.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:16s} : {ms:7.3f} ms  | {pc:5.1f}%")

    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(batch_size, device=sort_idx.device)
    loop_out = loop_out.index_select(0, inv_idx)
    loop_h = loop_h.index_select(0, inv_idx)
    loop_c = loop_c.index_select(0, inv_idx)

    assert_near(loop_out.cpu().numpy(), stack_out.cpu().numpy(), atol=1e-4, name="Actions")
    assert_near(loop_h.cpu().numpy(), stack_h.cpu().numpy(), atol=1e-4, name="Hidden")
    assert_near(loop_c.cpu().numpy(), stack_c.cpu().numpy(), atol=1e-4, name="Cell")
    print("MLP + LSTM PASSED\n")

def test_cnn_mlp():
    print("=== CNN + MLP TEST (C=20, H=8, W=8) ===")
    policies = create_cnn_policies(4)
    policies_gpu = [p.to('cuda') for p in policies]
    stacks = build_weight_stacks_cnn(policies)
    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')

    B = 1024
    C, H, W = 20, 8, 8
    obs = torch.randn(B, C, H, W, device='cuda')
    versions = np.random.randint(0, 4, size=B)
    h = torch.randn(B, 128, device='cuda')
    c = torch.randn(B, 128, device='cuda')

    # ----- Sort for loop -----
    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    h_sorted = h[sort_idx]
    c_sorted = c[sort_idx]
    versions_sorted = versions[sort_idx]

    
    loop_out, loop_h, loop_c = optimized_loop_forward_lstm(
        obs_sorted, torch.from_numpy(versions_sorted).to('cuda'),h_sorted, c_sorted, [p.cpu() for p in policies]
    )

    # ----- Stacked forward -----
    stack_out, stack_h, stack_c = stacked_forward_cnn_mlp_lstm(
        obs_sorted, torch.from_numpy(versions).to('cuda'), h_sorted.to('cuda'), c_sorted.to('cuda'), policies_gpu, gpu_stacks
    )

    # ----- Unsort loop output -----
    unsort_idx = np.argsort(sort_idx)
    loop_out = loop_out[unsort_idx]
    stack_out = stack_out[unsort_idx]
    # ----- Verify -----
    assert_near(loop_out.detach().cpu().numpy(),
                stack_out.detach().cpu().numpy(),
                atol=1e-4,
                name="Actions")
    assert_near(loop_h.detach().cpu().numpy(),
                stack_h.detach().cpu().numpy(),
                atol=1e-4,
                name="Hidden")
    assert_near(loop_c.detach().cpu().numpy(),
                stack_c.detach().cpu().numpy(),
                atol=1e-4,
                name="Cell")
    print("CNN + MLP + LSTM PASSED\n")

if __name__ == "__main__":
    #test_mlp_lstm()
    #test_cnn_mlp()
    benchmark_mlp_lstm(num_iters=1000, batch_size=1024, num_versions=4)
    #benchmark_cnn(num_iters=1000, batch_size=1024, num_versions=4)


