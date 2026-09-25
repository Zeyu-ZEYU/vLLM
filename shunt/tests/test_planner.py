"""Group plan: EAP per node, window, budgets, and KVLB per node."""
from shunt import native as N
from shunt.compute_model import ComputeModel
from shunt.config import DeploySpec, ModelSpec, PlanOptions
from shunt.planner import GroupInputs, plan_group


def _inputs(G=16):
    tau = [2e-3] * G
    tau[3] = 20e-3
    attn = [0.9 * x for x in tau]
    fresh = [4000] * G
    fresh[3] = 30000
    v_in = [1e6] * G
    v_in[5] = 2e9
    v_out = [2e6] * G
    return GroupInputs(tau, attn, fresh, v_in, v_out)


def test_plan_eap_and_kvlb():
    deploy = DeploySpec()
    cm = ComputeModel(ModelSpec(), deploy)
    plan = plan_group(_inputs(), cm, deploy, PlanOptions())
    assert plan.moves and all(s == 3 for s, _ in plan.moves[:1])
    assert all(deploy.node_of(s) == deploy.node_of(d) for s, d in plan.moves)
    assert plan.t_cmp == max(plan.tau_post) + plan.tau_ex
    assert plan.b_be == deploy.bw_port * plan.t_cmp
    hot = plan.inbound[5]
    assert hot.own <= plan.b_be + 1e-6 and (hot.borrow or hot.frontend)
    for w in range(16):
        assert abs(plan.inbound[w].total - _inputs().v_in[w]) < 1e-3


def test_plan_arms():
    deploy = DeploySpec()
    cm = ComputeModel(ModelSpec(), deploy)
    off = plan_group(_inputs(), cm, deploy, PlanOptions(eap=False, kvlb_budget=False))
    assert off.moves == [] and off.inbound[5].own == 2e9 and not off.inbound[5].borrow
    dbo = plan_group(_inputs(), cm, deploy, PlanOptions(dbo=True))
    assert dbo.b_be == deploy.bw_port * max(0.0, dbo.t_cmp - dbo.t_a2a)


def test_native_plan_matches_reference():
    if not N.available():
        return
    deploy = DeploySpec()
    cm = ComputeModel(ModelSpec(), deploy)
    a = plan_group(_inputs(), cm, deploy, PlanOptions())
    b = plan_group(_inputs(), cm, deploy, PlanOptions(), impl=N)
    assert a.moves == b.moves
    assert abs(a.t_cmp - b.t_cmp) < 1e-15
