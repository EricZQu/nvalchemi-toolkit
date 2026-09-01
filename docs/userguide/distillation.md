(distillation_guide)=

# Distilling a Teacher Into a Student

Knowledge distillation trains a small model — the *student* — to reproduce the
predictions of a larger, frozen one — the *teacher*. For interatomic potentials
the motivation is throughput: a foundation teacher may be far too expensive to
drive long molecular dynamics, while a student that reproduces its energies and
forces on the states that matter runs orders of magnitude faster. The teacher
also removes the usual data bottleneck, because it can label any structure,
including ones no reference calculation was ever run on.

{py:class}`~nvalchemi.training.distillation.DistillationStrategy` is the entry
point. It is a {py:class}`~nvalchemi.training.TrainingStrategy` subclass, so
everything in {ref}`training_guide` — optimizers, schedulers, validation, hooks,
checkpoints — applies unchanged; this guide covers only what distillation adds.

This guide assumes that you already have:

- a teacher wrapped with {py:class}`~nvalchemi.models.base.BaseModelMixin`;
- a student that is trainable and declares the outputs your objective reads;
- a dataset of structures, with or without reference labels.

## The shape of a distillation run

`DistillationStrategy` takes a named-model mapping holding `"student"` and
`"teacher"`. The teacher is **frozen by omission**: it must not appear in
`optimizer_configs`, and that absence is what puts it in evaluation mode with
gradients disabled for the duration of the run. The student — and any auxiliary
model, such as a learned projection — must be given an optimizer config. The
strategy raises at construction if either half of that contract is broken.

Teacher knowledge reaches the loss as ordinary batch fields. Every signal the
teacher produces populates one `teacher_*` field, and a loss term consumes it by
pointing its `target_key` there:

```python
from nvalchemi.training import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.distillation import PerAtomEnergyMatchingLoss

loss_fn = (
    EnergyMSELoss(target_key="teacher_energy")
    + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    + 0.2 * PerAtomEnergyMatchingLoss()
)
```

Distillation therefore needs no special loss machinery: any built-in term
distills by naming a teacher field, and mixing teacher targets with reference
targets in one objective is ordinary loss composition — offline, where every
sample carries its own reference labels. An on-policy run cannot, for reasons
covered below. The signals available, and the field and shape each one lands as,
are:

| Signal | Batch field | Level | Shape |
| --- | --- | --- | --- |
| `energy` | `teacher_energy` | system | `(B, 1)` |
| `forces` | `teacher_forces` | node | `(V, 3)` |
| `stress` | `teacher_stress` | system | `(B, 3, 3)` |
| `node_energies` | `teacher_node_energies` | node | `(V,)` |
| `embeddings` | `teacher_node_embeddings` | node | `(V, D)` |

All but `embeddings` come from the teacher's forward pass;
`teacher_node_embeddings` comes from
{py:meth}`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, which costs
a second pass.

You do not normally declare which signals you want. `teacher_signals=None`, the
default, derives the set from the `teacher_*` targets the loss reads, so the two
cannot drift apart; an explicit set must cover the derived one and may request
more. Whatever the resolved set is, it is checked against the teacher's declared
`outputs` at construction — as are, for the stock training function, every loss
component's prediction key against the student's declared outputs. A
misconfigured run fails before it starts rather than on its first batch.

The training function stays a plain student forward pass. The default,
{py:func}`~nvalchemi.training.distillation.default_distillation_fn`, calls the
student and prefixes every output with `predicted_`. The teacher is never called
there, which is why the teacher can never enter the student's autograd graph and
why the recipe survives serialization.

{py:class}`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss` is the one
term distillation adds. It matches the teacher's per-atom energy decomposition,
a quantity no reference dataset carries. Per-atom energies are not physically
observable on their own, so treat the term as a regularizer on the student's
internal decomposition and keep a total-energy term weighted above it.

## Offline distillation

The offline path scores the dataset once and trains from the result. It is the
cheaper path by a wide margin whenever the same structures are visited more than
once, and it is where any distillation project should start.

### Label the dataset once

{py:func}`~nvalchemi.training.distillation.label_dataset` walks a dataset in
chunks, scores each chunk with a
{py:class}`~nvalchemi.training.distillation.TeacherScorer`, and writes the source
fields plus the teacher fields into a Zarr store:

```python
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset

scorer = InProcessTeacherScorer(teacher, ["energy", "forces", "node_energies"])
num_labeled = label_dataset(dataset, scorer, "labeled.zarr", batch_size=64)
```

{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer` owns the
teacher's evaluation contract so callers do not have to. It narrows the teacher's
`active_outputs` to exactly what the requested signals need, reuses the batch's
neighbor list when it is a known full list at the teacher's own cutoff and format
and otherwise builds one and rolls it back, picks the grad mode a teacher with
autograd outputs requires, restores the `requires_grad` flags it found, detaches
every tensor it returns, and normalizes each signal to the canonical shape above.
The batch it is handed is left exactly as it was found, which is what makes the
same scorer usable mid-training and mid-trajectory.

Labeling is resumable: by default an existing store is treated as a partial run
and continued from `len(store)`, with the field set of the resumed chunks checked
against the store's. The one thing not carried over is the dense neighbor
tensors, whose neighbor dimension is rebuilt per chunk and cannot append into a
fixed-width store array — rebuild them from the stored positions when reading.

### Train from the labeled store

Nothing about the consumption path is distillation-specific. The teacher fields
arrive as ordinary batch attributes at the levels they were written at, so reader,
dataset, and loader are the ones any training run uses, and no teacher forward
pass happens during training at all:

```python
import torch

from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.training import OptimizerConfig
from nvalchemi.training.distillation import DistillationStrategy

loader = DataLoader(
    Dataset(reader=AtomicDataZarrReader("labeled.zarr")), batch_size=32
)

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    },
    loss_fn=loss_fn,
    num_steps=10_000,
)
strategy.run(loader)
```

A complete, runnable version of this workflow is
{doc}`/examples/intermediate/08_offline_distillation`.

### Consuming the labeled store elsewhere

The store is a plain Zarr hierarchy, so the labels are not locked to this
toolkit. Each field is one array under the store's `core` group —
`teacher_energy` and `teacher_forces` sit next to `positions` and
`atomic_numbers` — per-atom arrays are concatenated across samples with the
per-sample offsets in `meta/atoms_ptr`, and the store's root attributes record
whether each field is per-atom or per-system. Any Zarr client can therefore read
a labeled store, which makes the labeling pass a reusable artifact: a student
written in another framework consumes the same teacher labels without running
the teacher again, and without importing `nvalchemi` at all.

### Labeling on the fly

A batch that arrives without the required `teacher_*` fields is labeled on the
fly instead, by an internal hook the strategy registers ahead of your own on
`BEFORE_FORWARD`. That keeps short runs and interactive sessions working with no
labeling pass at all, and it is why unlabeled validation data needs no
preparation. Set `label_missing=False` to turn it off, in which case an unlabeled
batch surfaces as a missing loss target.

On-the-fly labels are attached to the device-placed batch the strategy trains on,
which is a copy of the one you handed over, so they do not persist on your
object. A loader that replays the same systems every epoch therefore pays one
teacher pass per epoch — the reason a long run should label offline first.

## On-policy distillation

Offline distillation trains the student on whatever structures the dataset
happens to hold. Those are not the structures the student will visit once it is
driving dynamics itself, and the gap between the two is what makes a distilled
potential drift or blow up on long trajectories. On-policy distillation closes
it: the student's own propagator generates frames, the teacher labels them, and
the student trains on them.

Setting `on_policy` turns
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.run` into a
segment loop that takes no dataloader, because each segment builds its own. One
segment is three phases:

1. **Generate** — the propagator advances the live state batch by
   `segment_steps`, seeded on the first segment from `seed_dataset`.
2. **Label and capture** — a
   {py:class}`~nvalchemi.training.distillation.TeacherLabelHook` on the
   propagator scores every `label_frequency` steps and mirrors each labeled frame
   into a host-memory sink; the segment's final frame is labeled too, then the
   sink is drained into a
   {py:class}`~nvalchemi.training.distillation.ReplayBuffer`.
3. **Train** — a freshly built mixed loader draws `steps_per_segment` batches at
   the configured ratio, each going through the ordinary per-batch stages.

```python
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.training.distillation import OnPolicyConfig

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    },
    loss_fn=loss_fn,
    num_steps=10_000,
    on_policy=OnPolicyConfig(
        dynamics=NVTLangevin(student, dt=0.5, temperature=300.0, friction=0.01),
        teacher_scorer=scorer,
        seed_dataset=seed_dataset,
        replay_ratio=0.25,
        steps_per_segment=32,
        batch_size=16,
        segment_steps=50,
        label_frequency=10,
        replay_capacity=8192,
    ),
    reference_dataset=anchor_dataset,
)
strategy.run()
```

The propagator must hold the very module registered as `models["student"]`,
either directly or composed into a larger model, and that object identity is
checked at construction. It is the whole point of the loop: because the trainer
and the propagator share one module, every optimizer step is immediately visible
to the next generated frame, and each segment generates from a fresher policy
than the last. The run is sized in optimizer steps rather than epochs, since each
segment builds its own loader and there is no fixed epoch to convert; one segment
counts as one epoch for hooks and epoch-cadence validation.

The propagator is any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` — an
integrator generating trajectories, or an optimizer generating relaxation paths.
Nothing downstream of the config reads a velocity or a temperature. Seed
structures must carry whatever the chosen propagator declares in
`__needs_keys__`, which for the built-in integrators and optimizers alike means
`velocities` and `atomic_masses`. `seed_dataset` is propagated whole, as a single
batch, so it *is* the set of systems the run generates from — size it to the
device. A {py:class}`~nvalchemi.dynamics.sampler.SizeAwareSampler` can bin-pack
the initial batch instead, and is configured in place of a `seed_dataset` rather
than alongside one.

`label_frequency` is the throughput knob. The teacher is the expensive model, and
a segment that labels every tenth frame costs a tenth of the teacher passes while
still generating every frame at student speed.

A runnable three-segment loop is
{doc}`/examples/intermediate/09_onpolicy_distillation`.

### The mixture ratio

`replay_ratio` — λ — is the fraction of every training batch drawn from
generated frames; the rest comes from `reference_dataset`, the *anchor*. It is
the knob to reach for first, because it decides how far the run is allowed to
follow its own trajectory:

- **λ = 1** trains on generated data alone and needs no anchor. The student is
  pulled entirely toward wherever its own dynamics go, which is also the failure
  mode: if the trajectory drifts into configurations the teacher was never meant
  to describe, nothing pulls it back.
- **0 < λ < 1** keeps a fixed, teacher-labeled distribution in every batch. The
  anchor is the pull: it holds the student on data whose coverage you chose,
  while the generated share keeps closing the gap between the training
  distribution and the one the student actually visits.
- **λ = 0** is offline distillation. The loop rejects it rather than running
  generation whose frames it would never train on — drop `on_policy` and call
  `run(loader)` over the labeled store instead.

The composition is exact per batch rather than an average: with
`replay_ratio=0.25` and `batch_size=16`, every optimizer step sees twelve anchor
samples and four generated ones. The achievable granularity is `1 / batch_size`,
so the two knobs only mean something together — a ratio that rounds either source
down to zero samples of a batch is rejected at construction, with the smallest
batch size that works named in the error.

Between segments the buffer grows, and the batch sampler reads its child dataset
lengths once, at construction. The loop therefore rebuilds the loader every
segment; if you drive
{py:func}`~nvalchemi.training.distillation.build_mixed_loader` yourself, do the
same, or the newest — most on-policy — frames are never sampled.

### The anchor must be teacher-labeled

A mixed batch is one collated `Batch`, and collation is not a merge: it keeps
only the fields *both* sources hold and drops the rest, while a whole level only
one side holds is zero-filled for the other's samples. Either behavior would be
silent, so both are rejected instead — the anchor's schema is compared against
the buffer's, on a probe batch drawn from each side.

The schema the anchor has to match is the replay-frame contract: the structure,
whatever propagator state travels with it, and the `teacher_*` labels — with none
of the `energy`, `forces`, or `stress` the propagator wrote on the live frame,
which the labeling hook strips on the way into the buffer so a stored frame never
carries the student's self-label under a reference target's name.

The practical consequence is worth stating plainly: **a raw DFT-labeled dataset
cannot be used as the anchor.** Its `energy` and `forces` are reference labels,
not teacher labels, and it is refused rather than quietly mixed in. Run it through
{py:func}`~nvalchemi.training.distillation.label_dataset` first, requesting the
same signals the propagator's scorer produces, and it becomes mixable. The
teacher fields the two sources carry are checked against each other at
construction as well, so a mismatch there is caught before the first segment.

Supervising one batch from teacher labels and reference labels at once is
masked-composition work that is not modeled yet. Until it lands, an on-policy run
is supervised by the teacher throughout — which is also why annealing between
teacher and reference targets with a
{py:class}`~nvalchemi.training.losses.base.LossWeightSchedule` is an offline
technique, where every sample carries its own reference labels.

Both mixture sources are collated before the strategy moves the batch, so
generated frames are staged on the anchor's device unless `replay_device` names
another one. A run with no anchor keeps them in host memory. If your anchor is a
Zarr-backed {py:class}`~nvalchemi.data.datapipes.dataset.Dataset`, open it on the
device the run trains on.

## Non-conservative teachers

Some teachers predict forces from a dedicated head rather than as the negative
gradient of their energy. Such a force field is *non-conservative*: it
does not integrate to a potential energy surface, and its curl need not vanish.
That is a fine trade for a labeling model and a bad one for a model driving long
MD, so distilling a direct-force teacher into a student whose forces *are* an
energy gradient is a core use case here rather than a workaround.

Nothing in the distillation path gates on conservativeness — not the strategy,
not the scorer, not the losses. There is no flag to set. The scorer detaches
every signal it returns, so how the teacher produced a force never reaches the
student's autograd graph; a teacher force is a number in a batch field, exactly
like a DFT force from a dataset.

What happens to the non-conservative part is worth understanding, because it
decides what a conservative student can and cannot fit. Such a student cannot
represent that part at all: its force field is, by construction, minus the
gradient of a scalar, and gradient fields are curl-free. Minimizing a
force-matching objective therefore does not approximate the teacher's field as a
whole — it drives the student toward the closest curl-free field to it, in the
least-squares sense the loss defines. The teacher's non-conservative component is
projected out rather than badly fitted. That is usually the outcome you want: it
is the component that would have shown up as energy drift in the student's own
dynamics, and the residual it leaves is a measure of how non-conservative the
teacher was on the sampled states, not of how badly the student trained.

Two practical consequences. First, do not expect force-matching error against a
non-conservative teacher to go to zero — the floor is the size of the projected
component. Second, keep a total-energy term in the objective: a force-matching
term only ever sees the gradient of the student's energy, so with forces alone
its energy scale is unconstrained.

## Operational notes

**Validation data goes through the same seam.** The internal labeling hook fires
on `BEFORE_FORWARD`, a stage both the training loop and the validation loop
dispatch on the device-placed batch, so unlabeled validation data needs no
preparation and a caller-supplied `training_fn` is covered too. Pointing
`validation_config` at a store written by `label_dataset` still avoids the
teacher pass entirely.

**Composed weights are ratios, not coefficients.**
{py:class}`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
default, so the three-term objective above runs at `1/2.2`, `1/2.2`, and
`0.2/2.2`. Build the composition with `normalize_weights=False` for literal
weights, which also stops a weight schedule on one term from rescaling the others
as it ramps.

**Label dtype follows the student.** Teacher labels are cast to the student's
first floating-point parameter dtype, so a float64 teacher feeds a float32
student without a dtype error at the loss. The cast is resolved at construction;
a student whose dtype changes afterwards needs a `dtype_policy` on the loss terms
instead.

**Checkpoints duplicate the teacher.** `save_checkpoint` serializes every entry
of `models`, teacher included, so every
{py:class}`~nvalchemi.training.hooks.CheckpointHook` write stores a second copy of
the frozen teacher's weights. Size the checkpoint interval accordingly with a
large teacher; storing the teacher by reference is planned.

**Spec round-trip is offline-shaped.** `on_policy` and `reference_dataset` hold
live runtime objects — a propagator, a scorer, and datasets — that no spec can
describe, so
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
omits them and warns. A strategy rebuilt from the spec of an on-policy run is
therefore offline-shaped, and the on-policy pieces have to be re-supplied at
construction until full recipe serialization lands. Restarting an on-policy run
mid-segment is not modeled either: the propagator state is not checkpointed, so a
resumed run continues from a freshly seeded trajectory.

**Reserved knobs.** `replay_eviction="uncertainty"` is reserved for
committee-based frame selection and raises today; use the default `"fifo"`, and
bound `replay_capacity` on long runs. `weight_sync_frequency` must be `1`: the
propagator and the trainer share one module object, so an eager run is never out
of sync, and the knob only becomes meaningful once the propagator holds a
compiled or remote copy of the student.

## API reference

See {ref}`training-distillation-api` for the API reference for
{py:class}`~nvalchemi.training.distillation.DistillationStrategy`,
{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer`,
{py:func}`~nvalchemi.training.distillation.label_dataset`,
{py:class}`~nvalchemi.training.distillation.OnPolicyConfig`,
{py:class}`~nvalchemi.training.distillation.TeacherLabelHook`,
{py:class}`~nvalchemi.training.distillation.ReplayBuffer`, and
{py:class}`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss`.
