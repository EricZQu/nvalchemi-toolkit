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
``energy``, ``forces``, ``stress``, ``node_energies``, ``embeddings``, and
``hessian`` — each mapped to a batch field, a level, and a canonical shape.
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

Signals differ in cost. All the forward-pass ones share a single teacher pass;
``embeddings`` adds a second, because the model contract computes
representations in their own method rather than returning them from the forward
pass; and ``hessian`` adds an energy-only pass plus the two backward passes
:func:`~nvalchemi.training.distillation.hessian_vector_product` takes through
it. The ``hessian`` signal is the only one that writes two fields —
``teacher_hvp`` and the ``teacher_hvp_probe`` direction it was taken along,
which the student has to be differentiated along too for the two to be
comparable, so it is stored and travels with the label.
:meth:`~nvalchemi.training.distillation.InProcessTeacherScorer.label_hvp`
computes one product for a probe the caller chose.

.. autosummary::
   :toctree: generated
   :nosignatures:

   hessian_vector_product


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

A ``validation_config`` carrying its own ``loss_fn`` takes part in both checks:
its ``teacher_*`` targets widen the derived signal set, and its prediction keys
are checked the same way whenever the effective validation function
(``validation_fn`` falling back to ``training_fn``) is the stock one. Neither
re-runs on assignment, so pass ``validation_config`` to the constructor or name
the wider set in ``teacher_signals``. Every resolved signal — derived or
explicit — is a request for its fields on every batch: a batch counts as
labeled only when it carries every resolved field, so adding a validation loss
with a new ``teacher_*`` target puts a training store written before it back on
the teacher, batch after batch, at identical values.

Training and validation batches go through one labeling seam: an internal hook
on ``BEFORE_FORWARD``, a stage both loops dispatch on the device-placed batch.
The teacher runs there with autocast disabled, so mixed-precision training does
not change the targets and an on-the-fly label matches the offline one exactly
wherever the store returns the label dtype: a store round-trips every floating
field to the dataset's ``positions`` dtype, so over the usual float32 dataset
every student but a float64 one agrees on both paths, while a float64 student
reads float32 back and needs a ``dtype_policy``. Labels are never cast below single
precision, so a ``bfloat16`` or ``float16`` student gets float32 labels and
needs ``dtype_policy="prediction_to_target"`` on its loss terms. Pointing
``validation_config`` at a store written by
:func:`~nvalchemi.training.distillation.label_dataset` still avoids the teacher
pass entirely, and validating an EMA-averaged student against the live teacher
is ``ValidationConfig(use_ema="auto")``, reported as ``model_source="mixed"``;
``use_ema="always"`` currently also demands an inference-slot entry for the
frozen teacher and fails at the first validation pass without one.

Checkpoints serialize every entry of ``models``, so each write duplicates the
frozen teacher's weights; size the checkpoint interval accordingly with a large
teacher.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn

Two objectives need a prediction the student's forward pass does not return, and
each ships the training function that produces it. Both are module-level
functions, so a recipe using one still survives
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`, and
both are additive: they return the stock ``predicted_*`` outputs plus one key.
:func:`~nvalchemi.training.distillation.embedding_distillation_fn` runs the
student's ``compute_embeddings`` and routes the result through the
``"projector"`` model when one is registered;
:func:`~nvalchemi.training.distillation.hessian_distillation_fn` differentiates
the student's energy twice along the labeled probe. A recipe wanting both writes
one module-level function of its own — calling both costs the student forward
pass twice, which building the union out of
:func:`~nvalchemi.training.distillation.hessian_vector_product` and the
student's ``compute_embeddings`` avoids.

.. autosummary::
   :toctree: generated
   :nosignatures:

   embedding_distillation_fn
   hessian_distillation_fn


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
dispatch on, once the first pass has revealed what it writes. A segment also
forces a label on the frame it ends on, whatever the cadence; because the
registry fires on the pre-increment step count, that forced frame and the next
segment's first cadence dispatch would otherwise be one propagator step apart,
so a cadence dispatch landing immediately after a labeled step is passed over.
A forced label never is, which keeps an early-exiting segment and a run's final
frame intact.

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
drops a field only one side holds, zero-fills a whole level only one side
holds, and casts the second part of a mixed batch to the dtype the first
carries while which source leads a chunk is not fixed, so all three differences
are rejected — a field the two sides hold at different dtypes among them: the
anchor has to be a teacher-labeled
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
step-cadence validation fires inside them, and the run's closing validation is
skipped when a cadence already validated at the final step. The segment is also
the restart granularity: a checkpoint taken mid-segment, or an offline run
graduating from a partial epoch, resumes by counting that segment as finished
rather than replaying the batches it had left. A second call to ``run()`` on
one strategy keeps the replay buffer the first filled and reseeds only the
trajectory. ``OnPolicyConfig.seed`` keys the
mixture sampler, which is how replicate runs are made to draw independently.
The loop is single-process for now: nothing shards its loader or its seed
state, so it refuses to start on more than one rank rather than have every rank
regenerate and retrain the same frames, while offline distillation over a
labeled store distributes through ``DDPHook`` as usual. Generated frames are
drained to host memory and staged on the reference dataset's own device, so a
GPU-resident anchor and the buffer collate on one device; ``replay_device``
overrides that and is checked against the anchor at construction. That device is
the one the anchor actually emits on, read off a batch whenever no declaration
settles it — a :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset`
declares none, and a store opened without a device declares an index-less
``cuda`` that names whichever device is current. The student is
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
supplied again. Supplying them is a keyword argument on every rebuild entry
point:
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`,
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_checkpoint_dict`,
and
:meth:`~nvalchemi.training.distillation.DistillationStrategy.load_checkpoint`
all take ``on_policy`` and ``reference_dataset``. The segment loop travels with
the student it propagates, so the ``models`` the propagator was built around go
back in alongside it and the checkpoint's weights are restored into those very
objects; restoring with
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` into a strategy
that was constructed with the loop reaches the same place from the other end.
An objective defined only on generated batches — an ensemble term — makes this
mandatory rather than optional, since it refuses to rebuild offline-shaped at
all.


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


Representation, curvature, and ensemble objectives
--------------------------------------------------

Three further terms distill things a reference dataset has no column for. Each
needs more from the run than a target field, and each is checked at
construction.

:class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss` matches the
teacher's per-atom representation. Both sides come from
``compute_embeddings`` rather than from a forward pass, so the objective needs
:func:`~nvalchemi.training.distillation.embedding_distillation_fn`, and the
student is run twice per batch. Across architectures the two widths differ,
which the learnable
:class:`~nvalchemi.training.distillation.EmbeddingProjector` reconciles: give it
the student's width by the teacher's, register it as a ``"projector"`` model
with an ``optimizer_configs`` entry of its own, and the training function routes
the student's embeddings through it. The projection is applied to the student
and never to the teacher, whose embeddings stay fixed targets — a learnable map
on the target side is optimized to be easy to hit, and the pair would minimize
the objective by collapsing the teacher's representation. The projector is a
training-time artifact: the distilled model is the student alone, so nothing
about the student's own outputs depends on it.

.. code-block:: python

   from nvalchemi.training.distillation import (
       DistillationStrategy,
       EmbeddingMatchingLoss,
       EmbeddingProjector,
       embedding_distillation_fn,
   )

   projector = EmbeddingProjector(student_width, teacher_width)
   strategy = DistillationStrategy(
       models={"student": student, "teacher": teacher, "projector": projector},
       optimizer_configs={
           "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
           "projector": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
       },
       loss_fn=EnergyMSELoss(target_key="teacher_energy")
       + 0.1 * EmbeddingMatchingLoss(),
       training_fn=embedding_distillation_fn,
       num_steps=10_000,
   )

Two representations agree only up to whatever symmetry each architecture's
embedding space carries — a channel permutation, a rotation of an equivariant
block — which is what the projector absorbs and why a residual floor on this
term is normal. Weight it as a regularizer beside the terms carrying the
physical targets.

:class:`~nvalchemi.training.distillation.HessianMatchingLoss` matches the
curvature of the teacher's energy surface, which decides vibrational spectra and
integrator stability and which energies and forces do not pin down. Neither side
forms a Hessian: both are products with one random probe direction, two backward
passes each. The teacher's product and its probe are materialized onto the batch
by the ``hessian`` signal — offline through
:func:`~nvalchemi.training.distillation.label_dataset` or on the fly through the
strategy's labeling seam — and the student's comes from
:func:`~nvalchemi.training.distillation.hessian_distillation_fn`, which takes it
on a second student pass narrowed to the energy alone: a conservative student
derives its forces from the very graph the second derivative needs, and frees
that graph outside training mode, so the stock forward cannot be differentiated
again and the narrowed pass derives no forces to consume it. The student is
therefore run twice per batch here too, and every validation pass costs the
same. One probe constrains one direction, so coverage comes from
redrawing: an on-policy run gets a fresh probe every time it labels a frame,
while a store labeled once freezes one direction per structure. Because that
probe is standard normal per component, the graph-balanced value is a Hutchinson
estimate of ``||dH||_F^2 / 3V`` in (eV/A^2)^2, which for a near-converged
student runs one to two orders of magnitude above a force mean-squared error on
the same batch. Start the term a hundred to ten thousand times lighter than the
force term rather than at parity, and read a single batch's value as the noisy
one-sample estimate it is.

:class:`~nvalchemi.training.distillation.BoltzmannMatchingLoss` matches the
ensemble rather than the configuration: it is the relative entropy between the
teacher's and student's Boltzmann distributions at a temperature, blind to a
constant energy offset and to any error that does not change relative
populations. ``beta`` interpolates the forward (``0``, mass-covering) and
reverse (``1``, mode-seeking) directions. The estimator reads a batch as a
sample of the *student's* own ensemble, which is what makes the weights uniform
on the student side, so the strategy requires ``on_policy``, rejects a
relaxation or converging propagator — neither samples an equilibrium ensemble —
and warns when ``replay_ratio`` mixes anchor frames the student never visited
into the batch. Reweighting an off-policy sample back onto the student's
ensemble is not offered — the weights this form folds away as uniform are not
recoverable from a batch — so an existing dataset reaches the term as
``reference_dataset``, mixed into generated frames by ``replay_ratio``. The
batch also has to be one system's configurations, since energies of different
systems are not comparable at all; seed the run with replicas of one structure,
one walker per graph. What cannot be checked is the temperature: set the term's
and the thermostat's from the same number. The two directions are not
interchangeable in scale either: the forward one is bounded above by ``log B``
and its gradient vanishes once the softmax saturates — a student whose error
spreads over more than a few ``k_B T`` — so ``beta=0`` can read as converged
while the student is far off, and ``beta`` is better held at ``0.5`` or above
until it is within a couple of ``k_B T``. Reducing energies by ``k_B T`` also puts the
gradient of either direction at up to ``1/k_B T`` per configuration, about
39 eV^-1 at 300 K, well above what a pointwise energy term produces.

The recommended recipe is therefore ``replay_ratio=1`` *and* a bounded
``replay_capacity``: the ratio keeps anchor rows out of the batch, and the
capacity keeps stale generated ones out, since every segment's loader draws
uniformly over the whole replay buffer and an unbounded one retires nothing —
after ``N`` segments only about one ``N``-th of a batch came from the current
student. Size it to the frames one segment or a few segments yield. Validation
is the other off-policy path, and the strategy refuses it outright: a
``ValidationConfig`` without a ``loss_fn`` of its own reuses the training
objective, ensemble term included, on a held-out set the student never visited,
so give the validation config a pointwise loss instead.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EmbeddingMatchingLoss
   EmbeddingProjector
   HessianMatchingLoss
   BoltzmannMatchingLoss
