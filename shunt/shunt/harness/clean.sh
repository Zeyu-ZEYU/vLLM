#!/usr/bin/env bash
# Stop every process of a Shunt run owned by this user on this host (vLLM
# servers and their workers, the Mooncake master, the proxy, the trace driver,
# RNIC samplers) and remove Mooncake shared-memory files. Run on every host
# before a run; the harness does it for you.
set -u
me=$(id -u)
for pat in '[v]llm serve' '[V]LLM::' '[m]ooncake_master' '[s]hunt.serving.proxy' \
           '[s]hunt.harness.replay' '[s]hunt.harness.bw_sampler'; do
  pgrep -u "$me" -f "$pat" | xargs -r kill -TERM 2>/dev/null
done
sleep 3
for pat in '[v]llm serve' '[V]LLM::' '[m]ooncake_master' '[s]hunt.serving.proxy' \
           '[s]hunt.harness.replay' '[s]hunt.harness.bw_sampler'; do
  pgrep -u "$me" -f "$pat" | xargs -r kill -KILL 2>/dev/null
done
find /dev/shm -maxdepth 1 -user "$me" -name 'mooncake*' -exec rm -rf {} + 2>/dev/null
echo "clean: done on $(hostname)"
