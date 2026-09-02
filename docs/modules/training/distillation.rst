.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-distillation-api:

Distillation API
================

Teacher scoring, offline dataset labeling, the offline distillation strategy
and loss terms, and the on-policy generation components for
knowledge-distillation workflows.

.. seealso::

   - **Training strategy API**: :ref:`training-strategy-api`
   - **Fine-tuning API**: :ref:`training-finetuning-api`
   - **Loss API**: :ref:`losses-api`


Scoring
-------

A scorer turns a :class:`~nvalchemi.data.Batch` into named teacher signals —
``energy``, ``forces``, ``stress``, ``node_energies``, and ``embeddings`` — each
mapped to a batch field, a level, and a canonical shape.
:class:`~nvalchemi.training.distillation.InProcessTeacherScorer` evaluates a
teacher loaded in the current process and leaves the scored batch exactly as it
found it, including neighbor tensors.

.. currentmodule:: nvalchemi.training.distillation

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherScorer
   InProcessTeacherScorer
   signal_fields
   scorer_fields
   signal_for_field

Scorers speak two public type aliases: ``SignalLevel``, the ``"node"`` or
``"system"`` level a signal is attached at, and ``TeacherLabels``, the
``{batch field: (detached tensor, level)}`` mapping
:meth:`~nvalchemi.training.distillation.TeacherScorer.label` returns; the
signal names themselves are published as
:data:`~nvalchemi.training.distillation.SUPPORTED_SIGNALS`. A custom scorer may
publish ``label_fields``, the batch fields its ``label()`` populates, which
consumers resolve through
:func:`~nvalchemi.training.distillation.scorer_fields` rather than reading the
attribute.


Labeling
--------

Offline labeling walks a dataset once, scores it, and writes the source fields
plus the teacher fields to a Zarr store that the ordinary reader and dataset
path consume. Runs are resumable, and the neighbor tensors are dropped because
a stored list records no cutoff for a consumer to check; ``keep_neighbors=True``
keeps a sparse one. Every chunk must write the schema the store
holds, and a store whose arrays disagree about how many samples it contains —
what an interrupted run leaves behind — is reported rather than resumed from a
misaligned offset.

.. autosummary::
   :toctree: generated
   :nosignatures:

   label_dataset


Strategy
--------

:class:`~nvalchemi.training.distillation.DistillationStrategy` is a
:class:`~nvalchemi.training.TrainingStrategy` over the named models
``"student"`` and ``"teacher"``. The teacher is frozen by omission from
``optimizer_configs``, the teacher signals are derived from the ``teacher_*``
targets the loss reads, and batches that arrive unlabeled are labeled on the fly
unless ``label_missing=False`` skips the teacher and lets the missing target
surface from the loss. ``training_fn`` stays a plain student forward, defaulting
to :func:`~nvalchemi.training.distillation.default_distillation_fn`, whose
``predicted_*`` keys are checked at construction against the outputs the student
actually computes — its ``active_outputs`` intersected with its declared
``outputs`` — so a student whose active set is narrowed is caught before the run
rather than on its first batch.

Training and validation batches go through one labeling seam: an internal hook
on ``BEFORE_FORWARD``, a stage both loops dispatch on the device-placed batch.
The teacher runs there with autocast disabled, so mixed-precision training does
not change the targets and an on-the-fly label matches the offline one exactly.
Pointing ``validation_config`` at a store written by
:func:`~nvalchemi.training.distillation.label_dataset` still avoids the teacher
pass entirely.

Checkpoints serialize every entry of ``models``, so each write duplicates the
frozen teacher's weights; size the checkpoint interval accordingly with a large
teacher.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn


On-policy generation
--------------------

On-policy distillation trains on frames the student itself generated.
:class:`~nvalchemi.training.distillation.OnPolicyConfig` describes one segment
loop: which propagator generates, how many steps a segment runs, how often the
teacher labels, and how much of each training batch is replayed. The propagator
is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so relaxation optimizers
generate paths exactly as integrators generate trajectories.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OnPolicyConfig

:class:`~nvalchemi.training.distillation.TeacherLabelHook` is the inline
labeling route: an ``AFTER_STEP`` dynamics hook that attaches ``teacher_*``
fields to the frame the propagator just resolved, at the level each signal
declares, and optionally mirrors a stripped copy of it into a
:class:`~nvalchemi.dynamics.sinks.DataSink`. It never touches the ``energy``
and ``forces`` the student wrote on the live batch, which drive the next step —
but it does strip them from the copy, along with the neighbor tensors and the
dynamics bookkeeping, so a stored frame is a training sample rather than a
propagator state and carries no self-label under a reference target's name. Do
not confuse it with the strategy's own private ``BEFORE_FORWARD`` labeling
seam, which labels batches on their way into a *training* step. Labeling is
idempotent per propagator step: a scorer publishing ``label_fields``, or one
whose signal names are all built-in, is skipped on a re-dispatch of the step it
already labeled, and a scorer publishing neither is skipped from its second
dispatch on, once the first pass has revealed what it writes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherLabelHook

Generated frames land in a
:class:`~nvalchemi.training.distillation.ReplayBuffer`, an in-memory dataset
behind a frozen key schema — appending a batch keeps only the keys both sides
hold, so one unlabeled frame would strip ``teacher_*`` from everything already
stored. :func:`~nvalchemi.training.distillation.build_mixed_loader` then draws
each training batch with an exact reference/replay composition, resolved to
whole samples of the batch size, and must be rebuilt after every segment
because the batch sampler reads the child dataset lengths once, at
construction. What the two sources have to agree on is their whole batch
schema, compared on a probe batch drawn from each side rather than on the field
names a Zarr-backed store and an in-memory buffer report differently. Collation
drops a field only one side holds and zero-fills a whole level only one side
holds, so both differences are rejected: the anchor has to be a teacher-labeled
dataset in the replay-frame shape — structure, propagator state, ``teacher_*``
labels — and one carrying reference ``energy`` or ``forces`` of its own is
rejected rather than mixed into batches that silently lose or fabricate them.
Supervising one batch from teacher labels and reference labels at once is
masked-composition work that comes later. ``ReplayEviction`` names the policy
retiring frames from a full buffer.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReplayBuffer
   build_mixed_loader

Setting ``on_policy`` on the strategy is what turns those pieces into a run.
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` then takes no
dataloader: it seeds a state batch from ``seed_dataset`` — or from a
``sampler``, which supersedes it and is therefore configured instead of it —
and repeats generate-label-train segments until ``num_steps`` optimizer steps
are done, drawing the ``1 - replay_ratio`` share of every batch from
``reference_dataset``, which is required unless the ratio is ``1`` and refused
when it is, because a ratio of ``1`` draws whole batches from the buffer and
would leave the anchor policed but never sampled. The seed batch is restamped
with fresh dynamics bookkeeping on the way in, so seeds loaded from a store an
earlier relaxation graduated do not arrive frozen at ``exit_status``, and the
anchor is probed once at construction for the fields the labeling hook strips —
a guaranteed mixture failure that would otherwise surface only after a whole
generation segment had been paid for. One segment is one epoch, so
``AFTER_EPOCH`` and epoch-cadence validation land at segment boundaries while
step-cadence validation fires inside them. ``OnPolicyConfig.seed`` keys the
mixture sampler, which is how replicate runs are made to draw independently.
The loop is single-process for now: nothing shards its loader or its seed
state, so it refuses to start on more than one rank rather than have every rank
regenerate and retrain the same frames, while offline distillation over a
labeled store distributes through ``DDPHook`` as usual. Generated frames are
drained to host memory and staged on the reference dataset's own device, so a
GPU-resident anchor and the buffer collate on one device; ``replay_device``
overrides that and is checked against the anchor at construction. The student is
held in evaluation mode to generate and flipped to training mode for the
training phase only, so generated frames cost no second-order graph and no
moving batch-norm statistics; a propagator model that merely *composes* the
student is held in evaluation mode for the whole loop instead, because the
training phase forwards ``models["student"]`` rather than the composition. The
propagator must hold the very module registered as
``models["student"]``, on its own or composed into a larger model — that object
identity is what makes each segment generate from the weights the previous one
trained, and it is checked at construction.

On-policy runs also relax the reserved ``teacher_`` namespace in exactly one
way. A loss target under that prefix normally has to name a built-in signal,
but a propagator scorer that declares the field in ``label_fields`` *supplies*
it: the labeling hook writes it onto every captured frame, ``reference_dataset``
has to carry it too — the generation/anchor parity check is what enforces
that — and validation data has to arrive with it, because the strategy's own
scorer produces built-in signals only and cannot backfill it. At least one
built-in ``teacher_*`` target, or an explicit ``teacher_signals``, is still
required alongside it. A scorer that declares no ``label_fields`` and no
built-in signals supplies nothing: its fields are unknowable until it has
scored a batch, so the strategy warns that it cannot check the anchor parity
yet and refuses a custom target read against it.

Because ``on_policy`` and
``reference_dataset`` hold live runtime objects, they are left out of
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`,
which warns, and a strategy rebuilt from that spec runs offline until they are
supplied again.


Losses
------

Every teacher signal shaped like a total energy, a force, or a stress is
consumed by a built-in loss term with its ``target_key`` pointed at the teacher
field — ``EnergyMSELoss(target_key="teacher_energy")``, and so on. Signals with
no supervised counterpart get their own term.

:class:`~nvalchemi.training.ComposedLossFunction` renormalizes its weights by
default, so composed weights are relative ratios: ``a + b + 0.2 * c`` runs at
``1/2.2``, ``1/2.2``, and ``0.2/2.2``. Build the composition with
``normalize_weights=False`` for literal coefficients, which also keeps a weight
schedule on one term from rescaling the others as it ramps.

.. autosummary::
   :toctree: generated
   :nosignatures:

   PerAtomEnergyMatchingLoss
