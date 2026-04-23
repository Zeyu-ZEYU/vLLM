我的灵骏实验环境有四个node。通过ssh连接四个node的方式是：
- node 0: 需要通过proxy http://127.0.0.1:7890连接到lj.zeyu.tw，用户名zeyu。node 0的host alias是zlingjun，可以在ssh config里看到。
- node 1: 需要通过node 0作为跳板，连接到lj1.zeyu.tw，用户名zeyu。
- node 2: 需要通过node 0作为跳板，连接到lj2.zeyu.tw，用户名zeyu。
- node 3: 需要通过node 0作为跳板，连接到lj3.zeyu.tw，用户名zeyu。
- 注意：node 0 1 2 3相互内部访问（不是模型的流量，是命令管理相关的流量）时，只需要直接通过lj0.zeyu.tw lj1.zeyu.tw lj2.zeyu.tw lj3.zeyu.tw相互访问就行。

你只需要node 0和node 1这两个，2和3你不用碰。

你修改任何代码时，只能修改local本地此项目的里的代码，不能动node上的代码。
本地分支必须在mono_kernel上。不在的话，需要切换到mono_kernel分支上才能修改。
本地vLLM对应每个node上的/home/zeyu/vllm/mono_kernel目录。

实验背景介绍：
- 这是mono_kernel的论文实验项目。
- 每个node有机头网络RDMA以及机尾网络RDMA两个RDMA网络。机头网络是CPU亲和的管理网络（只有一个RNIC），机尾网络是GPU-to-GPU之间的网络。机尾RDMA每个GPU有一个RNIC，只不过实际每两个RNIC被bond为一个。你可以ssh进每个node确认RDMA信息。
- 一般我跑的模型Qwen3-VL-8B-Instruct模型。如果有GPU通信跨node，要能在启动时指定用什么网卡，一般用机头ipv4网卡就行。
- 目前所有的启动入口你都已经实现了，在本地项目的目录zeyu下。也就是在远程node的目录/home/zeyu/vllm/mono_kernel/zeyu里。

每次你修改了local本地某个repo的代码，一定要git add .然后commit。至于是git commit -m "XXX"还是git commit --amend将修改融合到某次之前的commit中，你根据情况来做。如果只是对某次commit的一些修复和细微调整一般都可以将新的修改融合到那个commit。如果commit的内容是一次比较独立的功能实现，就独立git commit一个新的commit。最后就是git push (--force)。所有这些操作一定要先确认是在fe_rnic分支上。然后每个node都要在对应的repo目录里，在fe_rnic的分支上，进行git pull或者git pull --rebase来更新最新修改。所以你的修改只能在local本地，远端四个node通过git pull (--rebase)的方式来同步更新。

必须保证四个node每个同步更新时都能成功更新。因为每个node在做git pull (--rebase)时，有可能会超时，所以超时一定要重新试一下git pull (--rebase)。

如何跑实际的实验：
- 只访问让你使用的node。
- 你可以通过ssh连到让你访问的node。
- 每个node都需要进容器fe_rnic，做docker exec -it -u zeyu fe_rnic bash。如果容器没有start，可以先docker start fe_rnic。
- 进入容器后要mamba activate mono_kernel。
- 工作目录都在/home/zeyu/vllm/mono_kernel这里。
- 模型文件都在/home/zeyu/models里。一般用Qwen3-VL-8B-Instruct。
- 每次实验前都要检查环境是否被清理且可用。实验结束时，无论成功失败，都要清理。

现在需要实现disaggregation方法。就是，就是Qwen3-VL-8B-Instruct这种多模态模型，可以在两个GPU上跑（可能是同个node也可能是不同的node）。类似PD分离。vision encoder和text prefill在同一个GPU上，text decode在另一个GPU上。然后也能收集到目前所有相同的metrics信息。尽量用最简单的方法实现。可以复用现在的metric收集脚本，只需在某处添加参数说明是以这种disaggregation的方式运行模型。你目前实现了这套方案，但没有被充分测试。zeyu/README_DISAGG.md记录了你说的运行方法。
测试方法是，只使用node 0和1。每个node一个GPU。node 0 GPU跑vision encoder和text prefill，node 1 GPU跑text decode。至少要测20个请求，保证全通，能正确收集到数据，数据只在node 0一处放。启动过程一定要实时监控，一旦有错立刻退出分析并修改重新测试。所有metrics一定要分析是否合理，不合理处一定要重新检查代码。GPU之间通信目前用机头网卡ipv4就行。可以实现支持机尾网卡，但是不需默认。未来需要机尾网卡时，再在启动时配置。
这份代码是要给美国学生用的，他要在其他cluster里测（他的cluster只有机头网络，未来会装机尾网卡），因此你的README_DISAGG.md一定要清晰易懂，用英文；代码修改任何地方也都是要英文。要方便他配置，分析metrics。

目前，学生在他自己的环境里跑出现问题，他发我了邮件：
Hi Zeyu,

On the prefill node, I'm able to successfully run the sanity check (step 1 from the README_DISAGG.md), but when I try to run step 2 (I use iface enp226s0f0 because eth0 is not available):

bash zeyu/disagg_run.sh --role prefill \
    --iface enp226s0f0 \
    --gpu 0 \
    --num-prompts 4 --max-tokens 64 \
    --model Qwen/Qwen3-VL-8B-Instruct

I get the error attached in the .txt. After doing some research, it still seems to be a vLLM PD-disaggregation compatibility problem between P2pNcclConnector and the hybrid KV cache behavior of Qwen3-VL-8B-Instruct, but I'm not 100% sure. Also, I'm not sure if its because I am changing iface - when I run the command with --iface eth0, then the command just immediately returns, likely because there is no eth0 available on the hpc cluster. Here is the ip addr information:

(vllm-newest) -bash-4.4$ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: enp226s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether d8:5e:d3:8f:4d:6a brd ff:ff:ff:ff:ff:ff
    inet 10.153.48.78/16 brd 10.153.255.255 scope global dynamic noprefixroute enp226s0f0
       valid_lft 370566209sec preferred_lft 370566209sec
3: enp226s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether d8:5e:d3:8f:4d:6b brd ff:ff:ff:ff:ff:ff
4: enp194s0f1np1: <BROADCAST,MULTICAST,SLAVE,UP,LOWER_UP> mtu 1500 qdisc mq master bond0 state UP group default qlen 1000
    link/ether e8:eb:d3:8a:15:5f brd ff:ff:ff:ff:ff:ff
5: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4092 qdisc mq state UP group default qlen 1000
    link/infiniband 00:00:10:47:fe:80:00:00:00:00:00:00:e8:eb:d3:03:00:8a:15:5e brd 00:ff:ff:ff:ff:12:40:1b:ff:ff:00:00:00:00:00:00:ff:ff:ff:ff
    inet 10.155.48.78/16 scope global ib0
       valid_lft forever preferred_lft forever
6: bond0: <BROADCAST,MULTICAST,MASTER,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether e8:eb:d3:8a:15:5f brd ff:ff:ff:ff:ff:ff
7: bond0.529@bond0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether e8:eb:d3:8a:15:5f brd ff:ff:ff:ff:ff:ff
    inet 10.250.133.102/22 brd 10.250.135.255 scope global bond0.529
       valid_lft forever preferred_lft forever



Very respectfully,
Marcus Muntean


然后，附件内容是：
(vllm-newest) -bash-4.4$bash zeyu/disagg_run.sh --role prefill     --iface enp226s0f0     --gpu 0     --num-prompts 4 --max-tokens 64     --model Qwen/Qwen3-VL-8B-Instruct
============================================================
  PD-disagg launcher: role=prefill
============================================================
  Host        : udc-an38-29
  Iface/IP    : enp226s0f0 = 10.153.48.78
  GPU         : 0
  Peer IP     : <unset (prefill binds & waits)>
  KV port     : 25555         (decode connects to kv_port+100 on peer)
  Ctrl port   : 25500
  Model       : Qwen/Qwen3-VL-8B-Instruct
  Prompts     : 4         max_tokens=64
  Output dir  : /home/xhm3ws/vLLM/zeyu/outputs/disagg_20260423_035759
                (this side will write under /home/xhm3ws/vLLM/zeyu/outputs/disagg_20260423_035759/prefill)
  Profiling   : pynvml-only (nsys OFF)

  >>> After this side is up and printing 'waiting for decode
  >>> READY...', go to the DECODE node and run:
  >>>
  >>>   bash zeyu/disagg_run.sh --role decode \
  >>>       --peer-ip 10.153.48.78 \
  >>>       --iface <decode-nic> --gpu <decode-gpu-idx> \
  >>>       --kv-port 25555 --ctrl-port 25500 \
  >>>       --num-prompts 4 --max-tokens 64 \
  >>>       --model Qwen/Qwen3-VL-8B-Instruct

============================================================
[launcher] Starting prefill ...
Prepared 4 requests.
[Prefill] Role=kv_producer  local_ip=10.153.48.78  peer_ip=<wait-for-connect>  prefill_kv_port=25555  decode_kv_port=25655  iface=enp226s0f0
[Prefill] Loading model ...
INFO 04-22 23:58:13 [utils.py:233] non-default args: {'seed': 42, 'max_model_len': 4096, 'gpu_memory_utilization': 0.85, 'max_num_seqs': 5, 'enforce_eager': True, 'limit_mm_per_prompt': {'image': 1}, 'mm_processor_kwargs': {'min_pixels': 784, 'max_pixels': 1003520}, 'kv_transfer_config': KVTransferConfig(kv_connector='P2pNcclConnector', engine_id='a7737f6c-24df-4557-92f6-be67515fd047', kv_buffer_device='cuda', kv_buffer_size=1000000000.0, kv_role='kv_producer', kv_rank=0, kv_parallel_size=2, kv_ip='10.153.48.78', kv_port=25555, kv_connector_extra_config={'send_type': 'PUT_ASYNC', 'nccl_num_channels': '8'}, kv_connector_module_path=None, enable_permute_local_kv=False, kv_load_failure_policy='fail'), 'model': 'Qwen/Qwen3-VL-8B-Instruct'}
INFO 04-22 23:58:13 [model.py:549] Resolved architecture: Qwen3VLForConditionalGeneration
INFO 04-22 23:58:13 [model.py:1665] Using max model len 4096
INFO 04-22 23:58:13 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 04-22 23:58:13 [vllm.py:786] Asynchronous scheduling is enabled.
WARNING 04-22 23:58:13 [vllm.py:844] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 04-22 23:58:13 [vllm.py:855] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 04-22 23:58:15 [vllm.py:1021] Cudagraph is disabled under eager mode
WARNING 04-22 23:58:15 [vllm.py:1228] Turning off hybrid kv cache manager because `--kv-transfer-config` is set. This will reduce the performance of vLLM on LLMs with sliding window attention or Mamba attention. If you are a developer of kv connector, please consider supporting hybrid kv cache manager for your connector by making sure your connector is a subclass of `SupportsHMA` defined in kv_connector/v1/base.py and use --no-disable-hybrid-kv-cache-manager to start vLLM.
INFO 04-22 23:58:15 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=255044) INFO 04-22 23:58:21 [core.py:109] Initializing a V1 LLM engine (v0.1.dev15312+g690e127b1) with config: model='Qwen/Qwen3-VL-8B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen3-VL-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=42, served_model_name=Qwen/Qwen3-VL-8B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=255044) INFO 04-22 23:58:22 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://10.250.133.102:36069 backend=nccl
(EngineCore pid=255044) INFO 04-22 23:58:22 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=255044) INFO 04-22 23:58:23 [gpu_model_runner.py:4726] Starting to load model Qwen/Qwen3-VL-8B-Instruct...
(EngineCore pid=255044) INFO 04-22 23:58:23 [cuda.py:390] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(EngineCore pid=255044) INFO 04-22 23:58:23 [mm_encoder_attention.py:230] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(EngineCore pid=255044) WARNING 04-22 23:58:23 [vllm.py:844] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=255044) WARNING 04-22 23:58:23 [vllm.py:855] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=255044) INFO 04-22 23:58:23 [vllm.py:1021] Cudagraph is disabled under eager mode
(EngineCore pid=255044) INFO 04-22 23:58:24 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=255044) INFO 04-22 23:58:24 [flash_attn.py:596] Using FlashAttention version 2
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:06<00:18,  6.05s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:12<00:12,  6.05s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:19<00:06,  6.63s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:23<00:00,  5.69s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:23<00:00,  5.92s/it]
(EngineCore pid=255044)
(EngineCore pid=255044) INFO 04-22 23:58:48 [default_loader.py:384] Loading weights took 23.72 seconds
(EngineCore pid=255044) INFO 04-22 23:58:49 [gpu_model_runner.py:4811] Model loading took 16.78 GiB memory and 25.177978 seconds
(EngineCore pid=255044) INFO 04-22 23:58:49 [gpu_model_runner.py:5747] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 5 image items of the maximum feature size.
(EngineCore pid=255044) INFO 04-22 23:58:52 [gpu_worker.py:436] Available KV cache memory: 22.46 GiB
(EngineCore pid=255044) INFO 04-22 23:58:52 [kv_cache_utils.py:1319] GPU KV cache size: 163,568 tokens
(EngineCore pid=255044) INFO 04-22 23:58:52 [kv_cache_utils.py:1324] Maximum concurrency for 4,096 tokens per request: 39.93x
(EngineCore pid=255044) INFO 04-22 23:58:52 [factory.py:64] Creating v1 connector with name: P2pNcclConnector and engine_id: a7737f6c-24df-4557-92f6-be67515fd047
(EngineCore pid=255044) WARNING 04-22 23:58:52 [base.py:189] Initializing KVConnectorBase_V1. This API is experimental and subject to change in the future as we iterate the design.
Traceback (most recent call last):
  File "/home/xhm3ws/vLLM/zeyu/run_qwen35_vision_offline.py", line 1176, in <module>
    main()
    ~~~~^^
  File "/home/xhm3ws/vLLM/zeyu/run_qwen35_vision_offline.py", line 432, in main
    run_prefill_role(args, examples)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/xhm3ws/vLLM/zeyu/run_qwen35_vision_offline.py", line 626, in run_prefill_role
    llm = LLM(**llm_kwargs)
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/entrypoints/llm.py", line 382, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~^
        engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/llm_engine.py", line 177, in from_engine_args
    return cls(
        vllm_config=vllm_config,
    ...<4 lines>...
        multiprocess_mode=enable_multiprocessing,
    )
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/llm_engine.py", line 111, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        multiprocess_mode=multiprocess_mode,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        log_stats=self.log_stats,
        ^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/core_client.py", line 101, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/core_client.py", line 710, in __init__
    super().__init__(
    ~~~~~~~~~~~~~~~~^
        asyncio_mode=False,
        ^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        log_stats=log_stats,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/core_client.py", line 535, in __init__
    with launch_core_engines(
         ~~~~~~~~~~~~~~~~~~~^
        vllm_config, executor_class, log_stats, addresses
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as (engine_manager, coordinator, addresses, tensor_queue):
    ^
  File "/home/xhm3ws/miniforge3/envs/vllm-newest/lib/python3.13/contextlib.py", line 148, in __exit__
    next(self.gen)
    ~~~~^^^^^^^^^^
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/utils.py", line 998, in launch_core_engines
    wait_for_engine_startup(
    ~~~~~~~~~~~~~~~~~~~~~~~^
        handshake_socket,
        ^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        coordinator.proc if coordinator else None,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/sfs/gpfs/tardis/home/xhm3ws/vLLM/vllm/v1/engine/utils.py", line 1057, in wait_for_engine_startup
    raise RuntimeError(
    ...<3 lines>...
    )
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
(vllm-newest) -bash-4.4$



请找出潜在bug，并修复代码，然后按我说的在我的node 0和1上充分测试，并更新zeyu/README_DISAGG.md。
