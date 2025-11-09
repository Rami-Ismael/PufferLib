import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F


def make_dummy_data(*shape, seed=42):
    np.random.seed(seed)
    ary = np.random.rand(*shape).astype(np.float32) - 0.5
    return np.ascontiguousarray(ary)


def assert_near(a, b, atol=1e-4):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    diff = np.abs(a - b).max()
    assert diff < atol, f"max diff = {diff:.2e} >= {atol}"
    print(f"max diff = {diff:.2e}  ->  PASSED")

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

# === 2. Create policies ===
def create_policies(num_policies=4, seed=0):
    torch.manual_seed(seed)
    policies = [SimpleMLP() for _ in range(num_policies)]
    with torch.no_grad():
        for i, p in enumerate(policies):
            for param in p.parameters():
                param.add_(i * 0.1)
    return policies

# === 3. Build weight stacks ===
def build_weight_stacks(policies):
    W = {}
    b = {}
    for name in ['fc1', 'fc2', 'fc3']:
        W[name] = torch.stack([p.state_dict()[f'{name}.weight'] for p in policies])
        b[name] = torch.stack([p.state_dict()[f'{name}.bias']   for p in policies])
    return {'W': W, 'b': b}

# === 4. Optimized Looped Forward (Sorted + Grouped) ===
def optimized_loop_forward(obs_sorted, version_ids, policies):
    out = torch.empty_like(obs_sorted[:, :4])  # pre-allocate
    start = 0
    for v in torch.unique(version_ids):
        mask = (version_ids == v)
        end = start + mask.sum()
        sub_obs = obs_sorted[start:end]
        with torch.no_grad():
            out[start:end] = policies[v](sub_obs)
        start = end
    return out

# === 5. Stacked Forward (Your Method) ===
def stacked_forward(obs, version_ids, stacks):
    idx = torch.as_tensor(version_ids, device=obs.device, dtype=torch.long)


    x = obs

    for name in ['fc1', 'fc2']:
        W = stacks['W'][name][idx]
        b = stacks['b'][name][idx]
        x = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
        #x = torch.addmm(b, x, W.transpose(-2, -1))
        x = F.relu(x)

    # fc3
    W = stacks['W']['fc3'][idx]
    b = stacks['b']['fc3'][idx]
    out = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + b
    #out = torch.addmm(b, x, W.transpose(-2, -1))
    return out

# === 6. Benchmark ===
def benchmark(num_iters=1000, batch_size=512, num_versions=4):
    policies = create_policies(num_policies=num_versions)
    stacks = build_weight_stacks(policies)

    # Generate data
    obs = torch.randn(batch_size, 64, device='cuda')
    versions = np.random.randint(0, num_versions, size=batch_size)

    # === SORT for fair optimized loop ===
    sort_idx = np.argsort(versions)
    obs_sorted = obs.cpu()[sort_idx]
    versions_sorted = versions[sort_idx]

    gpu_stacks = {
        outer: {k: v.to('cuda') for k, v in inner.items()}
        for outer, inner in stacks.items()
    }

    # Warmup
    for _ in range(10):
        optimized_loop_forward(obs_sorted.cpu(), torch.from_numpy(versions_sorted), [p.cpu() for p in policies])
        stacked_forward(obs.to('cuda'), torch.from_numpy(versions), gpu_stacks)

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
        stacked_forward(obs, torch.from_numpy(versions), gpu_stacks)
    torch.cuda.synchronize()
    stack_time = time.time() - t0

    print(f"\n=== FAIR BENCHMARK (iters={num_iters}, B={batch_size}, V={num_versions}) ===")
    print(f"Old Method: {opt_loop_time:.3f}s  ->  {opt_loop_time/num_iters*1000:.2f} ms/step")
    print(f"New Method: {stack_time:.3f}s  ->  {stack_time/num_iters*1000:.2f} ms/step")
    print(f"Speedup        : {opt_loop_time/stack_time:.1f}x")

# === 7. MAIN ===
if __name__ == "__main__":
    # --- CORRECTNESS: GPU vs GPU ---
    print("=== CORRECTNESS (GPU vs GPU) ===")
    policies = create_policies(4)
    policies_gpu = [p.to('cuda') for p in policies] 

    stacks = build_weight_stacks(policies)
    gpu_stacks = {
        outer: {k: v.to('cuda') for k, v in inner.items()}
        for outer, inner in stacks.items()
    }

    B = 64
    obs = torch.randn(B, 64, device='cuda')
    versions = np.random.randint(0, 4, size=B)

    sort_idx = np.argsort(versions)
    obs_sorted = obs[sort_idx]
    versions_sorted = versions[sort_idx]

    # Loop on GPU
    loop_out = optimized_loop_forward(
        obs_sorted,
        torch.from_numpy(versions_sorted).to('cuda'),
        policies_gpu
    )

    # Stack on GPU
    stack_out = stacked_forward(
        obs,
        torch.from_numpy(versions).to('cuda'),
        gpu_stacks
    )

    # Unsort loop output
    unsort_idx = np.argsort(sort_idx)
    loop_out = loop_out[unsort_idx]

    assert_near(loop_out.cpu().numpy(), stack_out.cpu().numpy(), atol=1e-4)
    #benchmark(num_iters=1000, batch_size=1024, num_versions=4)


