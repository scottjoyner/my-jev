"""Trading lane for the typed decision model.

This package adapts the JEV decision-attention machinery to market decisions:
given causal market state, score the typed options ``buy`` / ``sell`` / ``hold``.

Authority boundary (unchanged from the System-One lane)
------------------------------------------------------
Learned outputs here are **advisory**. Nothing in this package places, sizes,
approves, or routes an order, and the portfolio-management execution paths
remain authoritative. The model emits a probability distribution over options
rather than an action, so consuming it always requires a downstream decision
about whether to act at all.

What is reused from the System-One lane
---------------------------------------
* ``my_jev.schema.DecisionRecord`` / ``QuestionSpec`` / ``TargetSpec`` — the
  typed question and target contract.
* ``my_jev.decision_attention.ExternalDecisionAttentionHead`` — the trainable
  head. It is deliberately backbone-agnostic: it consumes
  ``[batch, taps, tokens, hidden]`` state activations and
  ``[batch, options, taps, hidden]`` option activations and never sees the
  provider. That is what allows a numeric market encoder to stand in for the
  frozen ternary Bonsai 2 backbone without touching the head.

What is new here
----------------
A provider that emits taps from numeric market state, forward-return labels, and
purged/embargoed temporal splitting. Label leakage is the dominant risk in this
domain and the split logic treats it as a first-class concern.
"""

from __future__ import annotations

__all__ = ["OPTIONS", "DECISION_HORIZON_BARS"]

#: The typed option set. Fixed and ordered: the index of each option is the
#: label the model is trained to predict, so this tuple is part of the on-disk
#: contract and must not be reordered without a schema version bump.
OPTIONS: tuple[str, ...] = ("buy", "sell", "hold")

#: Bars ahead used to compute the forward return that defines the label. Kept
#: here because the split must embargo at least this many bars.
DECISION_HORIZON_BARS = 4