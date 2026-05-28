# I stopped writing AI copilots. I started writing AI runtimes.

> A working-out-loud post about turning a 4-week portfolio project into an honest stand against the prevailing "wrap an LLM around your domain" pattern.

Most AI portfolio projects I see right now have the same shape: a vector store, a prompt template, a chat interface, a screenshot of an LLM explaining something. The chat is the artifact, and the chat is where the value tends to stop.

I spent the last few weeks building something with a different shape — a small *predictive-maintenance agent runtime* for a simulated microgrid battery. There is no chatbot. The LLM is one node in a deterministic state machine, and not even the most important one. The project exists to test a hypothesis: at least for industrial-flavoured use cases, **the engineering work that matters in 2026 is the runtime around the model, not the model itself**.

This post is the engineering account of how that argument landed in code. Repo: [`pdm-agent-mvp`](https://github.com/DISSIDIA-986/pdm-agent-mvp).

![Workflow](../figures/workflow.png)

---

## 1. The setup, and why I refused to write a copilot

I had already built [EMS-demo](https://github.com/DISSIDIA-986/EMS-demo) — a microgrid energy-management dashboard with battery storage, frequency response, energy arbitrage, carbon accounting. Standard full-stack: React, FastAPI, TimescaleDB. After looking around at what "AI on top of EMS" usually meant in 2026, I noticed every example I could find boiled down to:

> Take the dashboard. Add a chat box. The chat box answers questions about the dashboard.

That shape is fine for a SaaS-onboarding tutorial. It feels weaker as the basis for a system you'd actually deploy in an industrial setting — at least judging from how the industrial-AI literature describes what buyers want. The repeated themes are deterministic decisions where it counts, physics where data is sparse, audit logs where money is on the line, human approval where mistakes are expensive.

So I picked one corner of the EMS — predictive maintenance — and asked what the *agent* layer would look like if "the LLM said so" wasn't a defensible answer at 3am when a maintenance work order got cut. This is a 4-week MVP on simulated data, not a real industrial deployment; I'm not pretending otherwise. The question I cared about was whether the *patterns* hold up when you take that constraint seriously from the first line of code.

## 2. The shape of a runtime

Five components, in order of importance:

1. **A deterministic diagnostic** built on classical signal processing, not a learned model
2. **A state machine** that can pause for human approval mid-execution and resume hours later
3. **An idempotent persistence layer** so the state machine survives a process restart
4. **An audit log** so every state transition is queryable, not just the decisions
5. **An LLM** — narrowly scoped to one job: writing a natural-language summary on top of the deterministic output

The LLM is the last component for a reason. Everything earlier on the list has to work without it.

## 3. The diagnostic: physics first, ML never (for this scope)

The standard playbook for bearing-fault detection is order tracking + envelope spectrum + a learned classifier. I skipped the classifier. Here's why.

A learned classifier requires labelled data the project doesn't have, and adds an opaque box at the most critical decision point — "is this bearing failing?" Worse, every reviewer of the code asks the same question: "what does this confidence mean?" If the answer is "softmax of a neural net trained on N samples we don't have provenance for," the project's audit story is dead.

So I wrote it as deterministic physics:

```python
# Bearing fault frequencies for SKF 6205-2RS (CWRU drive-end geometry)
bpfi = 5.4152 * (rpm / 60)   # ball-pass freq inner race
bpfo = 3.5848 * (rpm / 60)   # ball-pass freq outer race
bsf  = 4.7135 * (rpm / 60)   # ball spin freq
ftf  = 0.3983 * (rpm / 60)   # fundamental train (cage) frequency
```

Diagnose flow:

1. **Bandpass** the raw signal into the bearing's resonance band (default 2-4 kHz)
2. **Hilbert envelope** — demodulates the impact train sitting on top of the resonance carrier
3. **Family score** — for each fault class, sum `(peak amplitude / background median)` across two harmonics and ±2 FTF sidebands, with the first harmonic dominating
4. **Decision gate** — predict a fault class only if (a) top family score ≥ 6 and (b) top beats runner-up by ≥1.3×
5. **Severity bucket** — `normal / watch / alert / critical` from the top score plus an impulse-kurtosis tripwire

![Envelope spectrum on a CWRU inner-race window](../figures/envelope_demo.png)

In the figure above, the red dashed line is the theoretical BPFI for SKF 6205 at 1797 RPM. The peak directly under it is the diagnostic's primary evidence. No training loop, no opaque weights — just bearing geometry and a Butterworth filter.

### What the family-score rewrite cost me, and why I did it anyway

I originally shipped v1 with a single-harmonic scan. It worked on synthetic data and got 74% accuracy on real CWRU windows. Then an adversarial reviewer (Codex, on its third pass) ran my code, looked at the per-window evidence on real ball-fault data, and posted this:

> 真实 CWRU 球故障的能量分散在 BSF/2×BSF 及 ±FTF 侧带族 — 当前实现等于只看单个 BPFI/BPFO/BSF 峰，球故障被结构性低估了。

He was right. I rewrote `_harmonic_family_score()` to sum over harmonics × sidebands, capping at the 2nd harmonic specifically because CWRU's bearing geometry creates a 3×BPFO ≈ 2×BPFI alias that would have polluted the family sum. Real accuracy went from 74% to 80% on the curated subset.

**Then** the reviewer made his other point — *ball fault detection is still 0/10 and it's not a threshold bug*. Here's the part of the project I find most useful as a portfolio signal:

> **I shipped the failure mode.** The README's evaluation section reports 0/10 ball detection alongside 10/10 inner-race and 10/10 outer-race. The error analysis file explains *why* — 0.007" ball defects produce FTF-modulated sidebands with weak fundamental energy; envelope-spectrum methods systematically under-detect them; the fix is order tracking or cepstrum, neither of which is in scope for a 4-week MVP.

Saying that out loud is more useful than picking a smaller eval set that hides the problem. It's also what made the family-score rewrite worth doing — without admitting the failure publicly, the algorithmic upgrade would have just looked like a 6-point accuracy bump.

![Confusion: diagnose v2 vs RMS baseline](../figures/confusion.png)

## 4. The state machine: LangGraph's `interrupt()`, not webhooks

Once the diagnostic emits a severity, the workflow has to decide what to do. Three deterministic branches:

- `normal` → log, exit, no work order
- `watch` → draft a work order, queue for later human triage, do not page
- `alert` / `critical` → draft, **interrupt the workflow**, wait for human approval, then persist the decision

LangGraph has a primitive for the third case — `interrupt(payload)` literally suspends the graph mid-execution. The graph state is checkpointed to SQLite. When the human comes back hours later (or after a process restart), `Command(resume={"approve": True, "decided_by": "human:alice"})` picks up exactly where it stopped.

This is **the** reason I chose LangGraph over N8N, Zapier, or Dify. The competition makes workflow orchestration a configuration concern. LangGraph makes it a code concern. For a runtime that has to defend itself in front of an auditor, "show me the code" beats "show me the config" every time. The trade-off is honest: N8N has better off-the-shelf OPC UA nodes, so if I were shipping this to a factory I would seriously consider the swap. For a portfolio that has to demonstrate engineering judgement, LangGraph wins.

There is a subtle gotcha that took me a Codex review pass to find: the in-memory checkpointer drops everything on process restart, and the `thread_id` is the only thing identifying a paused incident. If you re-use a `thread_id` for a new sample while the previous one is still paused, you silently overwrite the first incident's checkpoint and the first work order is orphaned. Fix: persist checkpoints via `SqliteSaver`, derive `thread_id` per-incident, and add a `assert_thread_unused()` guard.

## 5. Persistence: atomic state transitions or it isn't real

The work-order table has five legal statuses (`draft / pending_approval / approved / rejected / closed`). Every transition is a row update. The naïve implementation is:

```python
# WRONG — TOCTOU race
status = cursor.execute("SELECT status FROM work_orders WHERE id=?", (id,)).fetchone()[0]
if status == "draft":
    cursor.execute("UPDATE work_orders SET status='pending_approval' WHERE id=?", (id,))
```

Two concurrent approvers can both pass the `if`, both run the `UPDATE`, and the second one silently wins. The fix is a single atomic statement guarded on the expected starting state:

```python
cursor.execute(
    """UPDATE work_orders
       SET status = ?, decided_at = ?, decided_by = ?, decision_note = ?
       WHERE id = ? AND status IN ('pending_approval', 'draft')""",
    (new_status, ts, decided_by, note, wo_id),
)
if cursor.rowcount != 1:
    # the row either didn't exist or was already decided — investigate
```

This is paired with `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout = 10000`, and `BEGIN IMMEDIATE` for write transactions. The MVP only ever has one writer in the demo, so I never observed contention in practice. The pattern is here because removing it would make extending to multi-writer (multiple operators, a sweeper job) a fragile refactor instead of a one-line config change.

## 6. The audit log: not a feature, a constraint

Every transition writes a row to `audit_log` with the event name, a JSON payload, and a timestamp. There is no API for *not* writing it — the cursor that does the `UPDATE` also does the `INSERT INTO audit_log`, in the same transaction.

This isn't decorative. When the MCP server (see below) exposes `list_audit_trail(work_order_id)`, the response is the literal SQL rows in order — nothing summarised, nothing filtered. If I were extending this MVP toward a real deployment, the audit-log shape is the part I would change least.

## 7. The LLM, finally

The Anthropic Claude call happens in exactly one node: `_llm_summary(diagnosis, asset_id)`. Input: the full deterministic diagnosis JSON. Output: a 2-sentence operator-facing summary that calls out the dominant fault frequency and the evidence ratio. If `ANTHROPIC_API_KEY` is not in the environment, the node falls back to a deterministic template and the workflow runs identically.

This is the thing I'm proudest of: **the workflow is correct without the LLM**. The LLM is a UX layer over deterministic decisions, not the decision itself. If Anthropic raises prices, has an outage, or releases a new model that hallucinates differently, my system keeps running and producing correct work orders. The audit log shows whether the summary was LLM-generated or templated.

## 8. The sidecar, and why HTTP beat in-process

The diagnostic engine could have been imported directly into the EMS-demo's FastAPI process. Instead it runs as a separate FastAPI service the host calls over plain JSON HTTP.

I measured the cost: p50 round-trip overhead is 4-14 ms across 0.5/1/2-second windows on a local socket. For window-level diagnostics this is invisible. The benefit is process isolation: the diagnostic, the workflow, the OPC UA reader, and the MCP server all live in their own process spaces, and an OOM in one doesn't take down the others.

The sidecar speaks vanilla HTTP, not MCP directly. That was intentional — MCP is a higher-level wrapper, and the same diagnose() function is also exposed via an `mcp.server.fastmcp.FastMCP` server in `pdm_agent.mcp_server`. Two siblings into the same brain: one for HTTP clients and EMS-demo integration, one for Claude Code / Claude Desktop.

One detail of that MCP wrapper is worth pulling out because it shaped the whole security story: there is no `approve_work_order` tool. The earliest draft had one, and an adversarial reviewer pointed out the obvious — an LLM client can pass `decided_by="human:alice"` and silently approve maintenance work that no human ever saw. The fix was to demote the surface: instead of a write tool, the MCP server only exposes `propose_decision_for_work_order`, which writes to a separate `agent_recommendations` table that a human must consult out-of-band. The `proposed_by` field is regex-checked against `^agent:[a-zA-Z0-9._-]{1,64}$` so an LLM literally cannot impersonate a human, even by mistake. Status changes on the `work_orders` table only happen through a trusted-process caller of `WorkOrderStore.decide()`. This MVP does not ship the authenticated operator UI that would actually establish the human operator's identity at decision time — the gap is documented explicitly via the `pdm://security` resource so a hiring reviewer doesn't read this as a closed security loop.

## 9. The thing I keep coming back to

In absolute terms this is a small project: about 1.5k lines of Python, 65 tests, one curated CWRU subset, one bearing geometry, one fault class still failing the eval. A more impressive portfolio entry could include RL dispatch, a custom model trained on real bearing data, or an LLM-vs-LLM debate over the maintenance schedule.

None of that would change the position I'm testing. The most useful thing I took away from three rounds of adversarial review wasn't an algorithm — it was a habit: keep the LLM away from decision points that downstream systems need to audit. Put it at the summarisation point. Keep the physics, the state machine, the atomic SQL, and the audit log at the decision point.

For a simulated MVP that's a stylistic preference. For something that would actually go into a regulated industrial setting, I think it stops being optional.

---

**Read the code:**
- Repo: [`pdm-agent-mvp`](https://github.com/DISSIDIA-986/pdm-agent-mvp)
- Evaluation + failure-mode write-up: [`eval/error_analysis.md`](https://github.com/DISSIDIA-986/pdm-agent-mvp/blob/master/eval/error_analysis.md)
- The MCP wrapper this post talks about: [`src/pdm_agent/mcp_server.py`](https://github.com/DISSIDIA-986/pdm-agent-mvp/blob/master/src/pdm_agent/mcp_server.py)

**Open to AI Engineer / Full-Stack roles.** [Contact](https://github.com/DISSIDIA-986).

— Yupo, 2026-05
