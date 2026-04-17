我的灵骏实验环境有四个node。通过ssh连接四个node的方式是：
- node 0: 需要通过proxy http://127.0.0.1:7890连接到lj.zeyu.tw，用户名zeyu。
- node 1: 需要通过node 0作为跳板，连接到lj1.zeyu.tw，用户名zeyu。
- node 2: 需要通过node 0作为跳板，连接到lj2.zeyu.tw，用户名zeyu。
- node 3: 需要通过node 0作为跳板，连接到lj3.zeyu.tw，用户名zeyu。

你只需要node 0和node 1这两个，2和3你不用碰。

你修改任何代码时，只能修改local本地此项目的里的代码，不能动node上的代码。
本地分支必须在mono_kernel上。不在的话，需要切换到mono_kernel分支上才能修改。
本地vLLM对应每个node上的/home/zeyu/vllm/mono_kernel目录。

实验背景介绍：
- 这是mono_kernel的论文实验项目。
- 每个node有机头网络RDMA以及机尾网络RDMA两个RDMA网络。机头网络是CPU亲和的管理网络（只有一个RNIC），机尾网络是GPU-to-GPU之间的网络。机尾RDMA每个GPU有一个RNIC，只不过实际每两个RNIC被bond为一个。你可以ssh进每个node确认RDMA信息。
- 一般我跑的模型Qwen3.5-9B模型。如果有GPU通信跨node，要能在启动时指定用什么网卡，一般用机头ipv4网卡就行。
- 目前所有的启动入口你都已经实现了，在本地项目的目录zeyu下。也就是在远程node的目录/home/zeyu/vllm/mono_kernel/zeyu里。

每次你修改了local本地某个repo的代码，一定要git add .然后commit。至于是git commit -m "XXX"还是git commit --amend将修改融合到某次之前的commit中，你根据情况来做。如果只是对某次commit的一些修复和细微调整一般都可以将新的修改融合到那个commit。如果commit的内容是一次比较独立的功能实现，就独立git commit一个新的commit。最后就是git push (--force)。所有这些操作一定要先确认是在fe_rnic分支上。然后每个node都要在对应的repo目录里，在fe_rnic的分支上，进行git pull或者git pull --rebase来更新最新修改。所以你的修改只能在local本地，远端四个node通过git pull (--rebase)的方式来同步更新。

必须保证四个node每个同步更新时都能成功更新。因为每个node在做git pull (--rebase)时，有可能会超时，所以超时一定要重新试一下git pull (--rebase)。

如何跑实际的实验：
- 只访问让你使用的node。
- 你可以通过ssh连到让你访问的node。
- 每个node都需要进容器fe_rnic，做docker exec -it -u zeyu fe_rnic bash。
- 进入容器后要mamba activate mono_kernel。
- 工作目录都在/home/zeyu/vllm/mono_kernel这里。
- 模型文件都在/home/zeyu/models里。一般用Qwen3.5-9B。
- 每次实验前都要检查环境是否被清理且可用。实验结束时，无论成功失败，都要清理。

现在需要实现disaggregation方法。就是，就是qwen3.5这种多模态模型，可以在两个GPU上跑（可能是同个node也可能是不同的node）。类似PD分离。vision encoder和text prefill在同一个GPU上，text decode在另一个GPU上。然后也能收集到目前所有相同的metrics信息。尽量用最简单的方法实现。可以复用现在的metric收集脚本，只需在某处添加参数说明是以这种disaggregation的方式运行模型。请先找最简单的这种disaggregation的方法，然后再进行修改。
另外，目前你已经实现了一版disaggregation，在commit 38309d1c58cead上。但是据反馈有bug。zeyu/README_DISAGG.md记录了你说的运行方法，我需要你进行测试以及修复。我需要的metric之前也说了，现在要能有每个请求的prefill计算时间（vision encoder time，text model prefill time两个）、KV传输时间（两个GPU之间传输时间）、decode TBT时间，decode的iteration次数（也就是output token数），JCT, vision encoder iteration的GPU使用率、GPU内存使用率、占用的SM数量、每个SM的使用率，prefill iteration的GPU使用率、GPU内存使用率、占用的SM数量、每个SM的使用率，decode iteration的GPU使用率、GPU内存使用率、占用的SM数量、每个SM的使用率，vision encoder iteration/prefill iteration/decode期间总的kernel overhead（总的绝对时间和相对于对应iteration/期间的百分比）。还有吞吐量RPS。
你先看哪些实现了，实现的对不对，再看哪些没实现，然后实现。
最后测试，只使用node 0和1。每个node一个GPU。node 0 GPU跑vision encoder和text prefill，node 1 GPU跑text decode。至少要测20个请求，保证全通，能正确收集到数据，数据只在node 0一处放。启动过程一定要实时监控，一旦有错立刻退出分析并修改重新测试。所有metrics一定要分析是否合理，不合理处一定要重新检查代码。GPU之间通信目前用机头网卡ipv4就行。可以实现支持机尾网卡，但是不需默认。未来需要机尾网卡时，再在启动时配置。
这份代码是要给美国学生用的，他要在其他cluster里测（他的cluster只有机头网络，未来会装机尾网卡），因此你的README_DISAGG.md一定要清晰易懂，用英文；代码修改任何地方也都是要英文。要方便他配置，分析metrics。
