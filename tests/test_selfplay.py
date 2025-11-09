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
    def __init__(self, in_dim=64, h1=128, h2=64, lstm_dim=32, out_dim=4):
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
    def __init__(self, in_dim=20, cnn_channels=32, hidden_size=256, out_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, cnn_channels, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=2, stride=1)
        self.head = nn.Linear(cnn_channels, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.head(x)
        return x

def batched_conv2d(x, W, b, stride=1, padding=0):
    """
    x : [B, C, H, W]
    W : [B, out_c, in_c, kH, kW]
    b : [B, out_c]
    """
    B, C, H, W_in = x.shape
    out_c, _, kH, kW = W.shape[1:]          # <-- kernel size from W

    # Unfold input
    x_unf = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
    x_unf = x_unf.transpose(1, 2)           # [B, OH*OW, C*kH*kW]

    # Flatten weights
    W_flat = W.view(B, out_c, -1)           # [B, out_c, C*kH*kW]
    W_flat = W_flat.transpose(1, 2)         # [B, C*kH*kW, out_c]

    # Matmul
    out = torch.bmm(x_unf, W_flat)

    # Add bias
    out = out + b.unsqueeze(1)

    # Reshape back using INPUT dimensions
    OH = (H + 2*padding - kH) // stride + 1
    OW = (W_in + 2*padding - kW) // stride + 1
    out = out.view(B, OH, OW, out_c).permute(0, 3, 1, 2)
    return out

def create_policies_mlp_lstm(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [MLPLSTMPolicy() for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies

def create_cnn_policies(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [SimpleCNN(in_dim=20) for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies

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
    for name in ['conv1', 'conv2', 'head']:
        W[name] = torch.stack([p.state_dict()[f'{name}.weight'] for p in policies])
        b[name] = torch.stack([p.state_dict()[f'{name}.bias']   for p in policies])

    return {'W': W, 'b': b}

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
    out = torch.empty_like(obs_sorted[:, :4])  # pre-allocate
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
        x = torch.bmm(x.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b
        if name != 'fc3':
            x = F.relu(x)

    # LSTM: 4 linear ops
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

    # Head
    W = stacks['W']['head'][idx]
    b = stacks['b']['head'][idx]
    out = torch.bmm(new_h.unsqueeze(1), W.transpose(-2, -1)).squeeze(1) + b

    return out, new_h, new_c

def stacked_forward_cnn_mlp(obs, version_ids, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)
    x = obs
    # -- conv 1 ---
    W = stacks['W']['conv1'][idx]
    b = stacks['b']['conv1'][idx]
    x = batched_conv2d(x, W, b, stride = 3, padding = 0)
    x = F.relu(x)

    # -- conv 2 ---
    W = stacks['W']['conv2'][idx]
    b = stacks['b']['conv2'][idx]
    x = batched_conv2d(x, W, b, stride = 1, padding = 0)
    x = F.relu(x)

    x = x.flatten(1)

    # Head
    W = stacks['W']['head'][idx]
    b = stacks['b']['head'][idx]

    out = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
    return out

# === 6. Benchmark ===
def benchmark(num_iters=1000, batch_size=512, num_versions=4):
    policies = create_policies_mlp_lstm(num_policies=num_versions)
    stacks = build_weight_stacks(policies)

    # Generate data
    obs = torch.randn(batch_size, 64, device='cuda')
    h = torch.randn(batch_size, 32, device='cuda')
    c = torch.randn(batch_size, 32, device='cuda')
    versions = np.random.randint(0, num_versions, size=batch_size)

    # === SORT for fair optimized loop ===
    sort_idx = np.argsort(versions)
    obs_sorted = obs.cpu()[sort_idx]
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
        stacked_forward_mlp_lstm(obs.to('cuda'), torch.from_numpy(versions), h, c, gpu_stacks)

    # === TIME OPTIMIZED LOOP ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        optimized_loop_forward_lstm(obs_sorted.cpu(), torch.from_numpy(versions_sorted), h_sorted, c_sorted, [p.cpu() for p in policies])
    torch.cuda.synchronize()
    opt_loop_time = time.time() - t0

    # === TIME STACKED ===
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        stacked_forward_mlp_lstm(obs, torch.from_numpy(versions), h, c, gpu_stacks)
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

    B = 256
    obs = torch.randn(B, 64, device='cuda')
    h = torch.randn(B, 32, device='cuda')
    c = torch.randn(B, 32, device='cuda')
    versions = np.random.randint(0, 4, size=B)

    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    h_sorted = h[sort_idx]
    c_sorted = c[sort_idx]
    versions_sorted = versions[sort_idx]

    loop_out, loop_h, loop_c = optimized_loop_forward_lstm(
        obs_sorted, torch.from_numpy(versions_sorted).to('cuda'),
        h_sorted, c_sorted, policies_gpu
    )

    stack_out, stack_h, stack_c = stacked_forward_mlp_lstm(
        obs, torch.from_numpy(versions).to('cuda'), h, c, gpu_stacks
    )

    unsort_idx = np.argsort(sort_idx)
    loop_out = loop_out[unsort_idx]
    loop_h = loop_h[unsort_idx]
    loop_c = loop_c[unsort_idx]

    assert_near(loop_out.cpu().numpy(), stack_out.cpu().numpy(), atol=1e-4, name="Actions")
    assert_near(loop_h.cpu().numpy(), stack_h.cpu().numpy(), atol=1e-4, name="Hidden")
    assert_near(loop_c.cpu().numpy(), stack_c.cpu().numpy(), atol=1e-4, name="Cell")
    print("MLP + LSTM PASSED\n")

def test_cnn_mlp():
    print("=== CNN + MLP TEST (C=20, H=8, W=8) ===")
    policies = create_cnn_policies(4)
    policies_gpu = [p.to('cuda') for p in policies]
    stacks = build_weight_stacks_cnn(policies)
    x = torch.randn(2, 20, 8, 8)
    conv = nn.Conv2d(20, 4, kernel_size=5, stride=3, padding=0, bias=True)
    conv_2 = nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0, bias=True)
    head = nn.Linear(4, 4)
    W = conv.weight  # [4, 20, 5, 5]
    b = conv.bias
    W_batch = W.unsqueeze(0).repeat(2, 1, 1, 1, 1)  # [2, 20, 3, 5, 5]
    b_batch = b.unsqueeze(0).repeat(2, 1)          # [2, 4]

    out1 = F.relu(conv(x))

    # Manual batched conv
    out2 = batched_conv2d(x, W_batch, b_batch, stride=3, padding=0)
    out2 = F.relu(out2)
    print(out1.shape, out2.shape)
    print(torch.allclose(out1, out2, atol=1e-6))

    out3 = F.relu(conv_2(out1))
    W_2 = conv_2.weight  # [4, 2, 2, 2]
    b_2 = conv_2.bias
    W_batch_2 = W_2.unsqueeze(0).repeat(2, 1, 1, 1, 1)  # [2, 4, 4, 2, 2]
    b_batch_2 = b_2.unsqueeze(0).repeat(2, 1)          # [2, 4]
    out4 = batched_conv2d(out1, W_batch_2, b_batch_2, stride=1, padding=0)
    out4 = F.relu(out4)
    print(out3.shape, out4.shape)
    print(torch.allclose(out3, out4, atol=1e-6))

    norm_flat = out3.flatten(1)
    norm_flat_2 = out4.flatten(1)
    print(norm_flat.shape, norm_flat_2.shape)
    print(torch.allclose(norm_flat, norm_flat_2, atol=1e-6))

    head_out = head(norm_flat)
    W_3 = head.weight  # [4, 4]
    b_3 = head.bias
    W_batch_3 = W_3.unsqueeze(0).repeat(2, 1, 1)
    b_batch_3 = b_3.unsqueeze(0).repeat(2, 1)
    head_out_2 = torch.bmm(norm_flat_2.unsqueeze(1), W_batch_3.transpose(-2, -1)).squeeze(1) + b_batch_3
    print(head_out.shape, head_out_2.shape)
    print(torch.allclose(head_out, head_out_2, atol=1e-6))

    # ----- Move stacks to GPU correctly -----
    gpu_stacks = {}
    for k, v in stacks.items():
        if isinstance(v, dict):
            gpu_stacks[k] = {kk: vv.to('cuda') for kk, vv in v.items()}
        else:
            gpu_stacks[k] = v.to('cuda')

    print("Policy 0 conv1 weight norm:", policies[0].conv1.weight.norm())
    print("Policy 1 conv1 weight norm:", policies[1].conv1.weight.norm())
    print("Stacked W conv1[0] norm:", stacks['W']['conv1'][0].norm())
    print("Stacked W conv1[1] norm:", stacks['W']['conv1'][1].norm())

    B = 256
    C, H, W = 20, 8, 8
    obs = torch.randn(B, C, H, W, device='cuda')
    versions = np.random.randint(0, 4, size=B)


    # ----- Sort for loop -----
    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    versions_sorted = versions[sort_idx]

    
    loop_out = optimized_loop_forward(obs_sorted,
                                      torch.from_numpy(versions_sorted).to('cuda'),
                                      policies_gpu)

    # ----- Stacked forward -----
    stack_out = stacked_forward_cnn_mlp(obs, torch.from_numpy(versions).to('cuda'), gpu_stacks)

    # ----- Unsort loop output -----
    unsort_idx = np.argsort(sort_idx)
    loop_out = loop_out[unsort_idx]
    breakpoint()
    # ----- Verify -----
    assert_near(loop_out.detach().cpu().numpy(),
                stack_out.detach().cpu().numpy(),
                atol=1e-6,
                name="Actions")
    print("CNN + MLP PASSED\n")

if __name__ == "__main__":
    #test_mlp_lstm()
    test_cnn_mlp()
    #benchmark(num_iters=1000, batch_size=1024, num_versions=4)


