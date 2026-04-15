# Qwen3-235B PD 分离部署指南

基于 vLLM v1 + LMCache + Mooncake (RDMA) 的 Prefill-Decode 分离推理服务。

## 架构

```
                        ┌──────────────┐
                        │  Proxy :9090 │
                        └──────┬───────┘
                 ┌─────────────┴──────────────┐
                 ▼                             ▼
┌────────────────────────────┐   ┌──────────────────────────┐
│  Prefill Cluster           │   │  Decode (1~2 nodes)      │
│  2 nodes × 8 GPU (Ray)    │   │  每节点独立 8 GPU         │
│  DP=16, TP=1, EP=16 (auto) │   │  DP=8, TP=1, EP=8 (auto) │
│                            │   │                          │
│  node-0: Ray head + vllm  │   │  node-2: Ray head + vllm │
│  node-1: Ray worker       │   │  node-3: Ray head + vllm │
│          :8100             │   │          :8200            │
└─────────────┬──────────────┘   └────────────┬─────────────┘
              │    Mooncake RDMA (KV cache)    │
              └────────────┬──────────────��───┘
                    ┌──────┴──────┐
                    │   Master    │
                    │ :50052/8080 │
                    └─────────────┘
```

**KV 缓存传输机制**: Prefill 通过 Mooncake RDMA 将 KV cache 存入共享存储,
Decode 自动从 Mooncake 检索对应的 KV cache, 无需 NIXL 直连端口。

## 节点规划

| 角色 | 节点 | IPv4 管理 | IPv6 RDMA (bond0) |
|------|------|----------|-------------------|
| Master + Prefill (Ray head) | node-0 (lj.zeyu.tw) | `192.168.0.42` | `fd03:4514:80:6240::1` |
| Prefill (Ray worker) | node-1 (lj1.zeyu.tw) | `192.168.0.40` | `fd03:4514:80:7b80::1` |
| Decode | node-2 (lj2.zeyu.tw) | `192.168.0.39` | `fd03:4514:80:6600::1` |
| Decode (可选) | node-3 (lj3.zeyu.tw) | `192.168.0.41` | `fd03:4514:80:5f00::1` |

## 文件说明

```
fe_rnic/
├── master.sh                          # Mooncake master 启动脚本
├── ray_start.sh                       # Ray 集群启动 (export env + ray start)
├── disagg_vllm_launcher.sh            # vLLM 启动入口 (prefill/decode/proxy)
├── disagg_proxy_server.py             # PD 分离代理 (来自 LMCache, 见下方说明)
├── benchmark.py                       # Metrics 收集 (RPS, TTFT, TBT, JCT)
├── gen_synthetic_data.py              # 合成数据生成 (10K~100K tokens)
├── clean.sh                           # 清理残留进程
├── configs/
│   ├── mooncake-prefiller-config.yaml # Prefiller mooncake 配置模板
│   └── mooncake-decoder-config.yaml   # Decoder mooncake 配置模板
└── README.md
```

## 前置条件

所有操作都在容器 `fe_rnic` 中进行:
```bash
docker exec -it -u zeyu fe_rnic bash
mamba activate fe_rnic
cd /home/zeyu/vllm/fe_rnic/fe_rnic
```

## 启动步骤

### Step 1: 启动 Mooncake Master

在 **node-0** 上:
```bash
bash master.sh
```
或后台运行:
```bash
nohup bash master.sh > /tmp/master.log 2>&1 &
```

### Step 2: 启动 Ray 集群 (Prefill)

在 **node-0** 上 (Ray head):
```bash
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.42 \
LOCAL_RDMA_IP=fd03:4514:80:6240::1 \
  bash ray_start.sh head
```

在 **node-1** 上 (Ray worker):
```bash
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.40 \
LOCAL_RDMA_IP=fd03:4514:80:7b80::1 \
  bash ray_start.sh worker 192.168.0.42
```

验证: 在 node-0 上运行 `ray status`，应看到 2 个节点、16 个 GPU。

### Step 3: 启动 Prefill

在 **node-0** 上:
```bash
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.42 \
LOCAL_RDMA_IP=fd03:4514:80:6240::1 \
  bash disagg_vllm_launcher.sh prefill
```

等待所有 16 个 `Application startup complete.` 输出。

### Step 4: 启动 Decode

Decode 节点需要独立的 Ray head (单节点 Ray 集群)。

**node-2:**
```bash
# 1. 启动 Ray head
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.39 \
LOCAL_RDMA_IP=fd03:4514:80:6600::1 \
MOONCAKE_ROLE=decoder \
  bash ray_start.sh head

# 2. 启动 decode
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.39 \
LOCAL_RDMA_IP=fd03:4514:80:6600::1 \
  bash disagg_vllm_launcher.sh decode
```

**(可选) node-3:**
```bash
# 1. 启动 Ray head
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.41 \
LOCAL_RDMA_IP=fd03:4514:80:5f00::1 \
MOONCAKE_ROLE=decoder \
  bash ray_start.sh head

# 2. 启动 decode
MASTER_IP=192.168.0.42 \
LOCAL_IP=192.168.0.41 \
LOCAL_RDMA_IP=fd03:4514:80:5f00::1 \
  bash disagg_vllm_launcher.sh decode
```

等待所有 8 个 `Application startup complete.` 输出。

### Step 5: 启动 Proxy

**单 decode (仅 node-2):**
```bash
PREFILL_PRIMARY_IP=192.168.0.42 \
DECODE_IPS=192.168.0.39 \
  bash disagg_vllm_launcher.sh proxy
```

**双 decode (node-2 + node-3):**
```bash
PREFILL_PRIMARY_IP=192.168.0.42 \
DECODE_IPS=192.168.0.39,192.168.0.41 \
  bash disagg_vllm_launcher.sh proxy
```

**启用 KV Overlap (可选):**

KV overlap 使 prefill 的 KV 传输与 GPU 计算 pipeline 化（layerwise RDMA WRITE 到 decode 节点）：
```bash
# Prefill 启动时加 ENABLE_KV_OVERLAP=true
ENABLE_KV_OVERLAP=true \
MASTER_IP=192.168.0.42 LOCAL_IP=192.168.0.42 LOCAL_RDMA_IP=fd03:4514:80:6240::1 \
  bash disagg_vllm_launcher.sh prefill

# Proxy 启动时加 DECODE_RDMA_IPS（decode 节点的 RDMA IPv6 地址）
PREFILL_PRIMARY_IP=192.168.0.42 \
DECODE_IPS=192.168.0.39,192.168.0.41 \
DECODE_RDMA_IPS=fd03:4514:80:6600::1,fd03:4514:80:5f00::1 \
  bash disagg_vllm_launcher.sh proxy
```

### Step 6: 验证

**completions 测试:**
```bash
curl http://192.168.0.42:9090/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, my name is", "max_tokens": 50, "model": "Qwen3-235B"}'
```

**chat completions 测试:**
```bash
curl http://192.168.0.42:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50, "model": "Qwen3-235B"}'
```

## 停止服务

```bash
bash clean.sh          # 清理本节点
bash clean.sh --all    # 清理所有 4 个节点 (需从 node-0 执行)
```

## 注意事项

1. **模型加载时间**: 首次启动每个节点约需 5-10 分钟加载模型权重。
2. **首次请求**: 第一个请求会触发 warmup/编译，可能需要 10-30 秒。
3. **僵尸进程**: 如果启动失败后重启，请先运行 `clean.sh` 清理残留进程，
   否则旧进程可能占用端口导致新请求超时。
4. **Mooncake 告警**: `Failed to get NUMA node` 和部分 `Buffer registration failed`
   是已知的非致命告警，不影响正常使用。
5. **Proxy**: `disagg_proxy_server.py` 来自
   `LMCache/examples/disagg_prefill/disagg_proxy_server.py`，
   修改：comment out 两处 `await wait_decode_kv_ready` (LMCache#1342)，
   新增 `--decoder-rdma-host` 参数支持 KV overlap。
6. **KV Overlap**: 通过 `ENABLE_KV_OVERLAP=true` 启用 layerwise KV 传输。
   Prefill 每计算一层就 RDMA WRITE KV 到 decode 节点的 Mooncake segment，
   与后续层的 GPU 计算 pipeline 化。需同时在 proxy 传入 `DECODE_RDMA_IPS`。
7. **Head NIC 分流**: 通过 `ENABLE_HEAD_NIC_SPLIT=true` 启用（需在 ray_start.sh 前设置）。
   部分 KV chunk 可通过机头 RNIC (mlx5_0) 传输。分流策略由
   `LMCache/lmcache/v1/kv_routing.py` 的 `route_kv_chunk()` 函数控制，
   默认全走机尾。修改函数返回值可临时切换为全机头实验。

## Benchmark

**快速测试:**
```bash
python benchmark.py --url http://192.168.0.42:9090 --num-requests 5 --max-tokens 20
```

**合成数据测试 (Poisson 发送):**
```bash
# 生成数据
python gen_synthetic_data.py --input-lens 1024,2048 --num-per-len 5 -o /tmp/data.jsonl

# 运行 benchmark
python benchmark.py --url http://192.168.0.42:9090 --dataset /tmp/data.jsonl --qps 1.0 \
  --output-dir /home/zeyu/exp_results/fe_rnic --tag test1
```

**收集的 Metrics:** RPS, TTFT (ms), TBT (ms), JCT (ms), avg output tokens
