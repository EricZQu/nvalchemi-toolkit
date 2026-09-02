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
checkpoints — applies unchanged, except that resuming an on-policy run needs
`restore_checkpoint` rather than `load_checkpoint`, covered under the
operational notes; this guide covers only what distillation adds.

This guide assumes that you already have:

- a teacher wrapped with {py:class}`~nvalchemi.models.base.BaseModelMixin`;
- a student that is trainable and declares the outputs your objective reads;
- a dataset of structures, with or without reference labels.

For those prerequisites, see {ref}`models_guide`, {ref}`datapipes_guide`, and
{ref}`training_guide`.

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

| Signal | Teacher output | Batch field | Level | Shape |
| --- | --- | --- | --- | --- |
| `energy` | `energy` | `teacher_energy` | system | `(B, 1)` |
| `forces` | `forces` | `teacher_forces` | node | `(V, 3)` |
| `stress` | `stress` | `teacher_stress` | system | `(B, 3, 3)` |
| `node_energies` | `atomic_energies` | `teacher_node_energies` | node | `(V,)` |
| `embeddings` | `compute_embeddings` | `teacher_node_embeddings` | node | `(V, D)` |

All but `embeddings` come from the teacher's forward pass;
`teacher_node_embeddings` comes from
{py:meth}`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, which costs
a second pass. The *teacher output* column is the name that has to appear in the
teacher's `ModelConfig.outputs`, and it is the name the construction check
reports as missing — which is why requesting `node_energies` from a teacher that
does not decompose its energy fails naming `atomic_energies`, not the signal.

`embeddings` is the one signal no built-in objective can consume. Nothing in
{py:mod}`nvalchemi.training.losses` reads a `(V, D)` target, and the stock
training function produces no student-side embedding to compare one against, so
the signal is usable today for labeling stores and for custom objectives: a loss
term reading `teacher_node_embeddings` needs a student-side embedding
prediction, which means a `training_fn` that calls `compute_embeddings` itself.
The strategy says so if you try — a loss component whose prediction key is an
embedding is refused at construction with that instruction rather than with the
generic missing-output message.

You do not normally declare which signals you want. `teacher_signals=None`, the
default, derives the set from the `teacher_*` targets the losses read — the
training loss and, when `validation_config` carries a `loss_fn` of its own, the
validation loss too — so objective and teacher cannot drift apart; an explicit
set must cover the derived one and may request more. Whatever the resolved set
is, it is checked against the teacher's declared `outputs` at construction.
Every loss component's prediction key is checked the same way, for the stock
training function, but against the outputs the student *actually computes* —
its `active_outputs` narrowed to its declared ones — so a pretrained wrapper
whose active set was narrowed is caught here rather than on its first batch,
with an error that names `active_outputs` as the thing to widen. A validation
loss goes through the same prediction-key check whenever its effective
validation function — `validation_fn` falling back to `training_fn` — is the
stock one.

Neither check re-runs on assignment: a `validation_config` attached after
construction keeps the signals already resolved, so pass it to the constructor,
or name the wider set in `teacher_signals`. And every resolved signal is a
request for its fields on every batch, not a permission to carry them. A batch
counts as labeled only when it holds every resolved field, so a validation loss
reading a `teacher_*` target the training loss does not puts a store labeled
without that field back on the teacher, batch after batch — the same values, at
the price of a forward pass each time. A store meant to train with no teacher
pass at all has to be labeled with the same signal set the strategy resolves.

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

It is also the term that asks the most of both models, because the decomposition
has to exist on each side. `atomic_energies` is the name to look for: the teacher
must declare it in `ModelConfig.outputs` for the `node_energies` signal to be
scorable, and the student must both declare *and* compute it, since the term
reads `prediction_key="predicted_atomic_energies"` and the stock training
function prefixes the student's forward outputs. A student that emits only
`energy` and `forces` — including
{py:class}`~nvalchemi.models.demo.DemoModelWrapper` — therefore fails at
construction on the three-term objective above, naming the loss component and
the missing `atomic_energies`.

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

The one teacher it refuses is a composition that plans more than one
neighbor-list source. A
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` under its default
`neighbor_adaptation="auto"` builds the list at its largest cutoff and lets a
stage adapt it only while that cutoff is within `max_cutoff_ratio` — `1.5` by
default — of the stage's own, so an MLIP composed with a dispersion correction
at a much longer cutoff plans two lists. The scorer builds exactly one list per
batch, while such a composition resolves each stage's list out of a per-source
table only its own hooks produce, so it is rejected at construction — by the
scorer, and by `DistillationStrategy`, which builds one — with the two ways out
named: compose the teacher with `neighbor_adaptation="always"`, or with a
`max_cutoff_ratio` of at least the ratio of its largest to its smallest cutoff,
and it adapts that single list per stage.

Labeling is resumable: by default an existing store is treated as a partial run
and continued from `len(store)`, with every resumed chunk checked against the
store's own field set, levels, and dtypes, and a store an interrupted append
left inconsistent refused rather than resumed from a misaligned offset. The one
thing not carried over is the neighbor list, in either format. The dense
tensors cannot append into a fixed-width store array at all, and a sparse list
is dropped because the cutoff it was built at is a batch attribute the store
does not hold, so a reloaded list is one nothing downstream could check — which
is why the scorer rebuilds rather than trusts one. `keep_neighbors=True` stores
the sparse list anyway, at the cost of that check, and a store written under one
setting and resumed under the other is refused as field-set drift like any
other. Nothing rebuilds a list on the way back out of the store; the next
section wires that up.

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

Nothing in the training loop builds a neighbor list, so a graph student needs one
built for it — labeling drops the neighbor tensors from the store, and a
wrapped MLIP raises rather than building its own. Add a
{py:class}`~nvalchemi.hooks.NeighborListHook` at `BEFORE_FORWARD`, configured
from the student's own neighbor config:

```python
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import TrainingStage

neighbor_hook = NeighborListHook(
    student.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD
)
```

Pass it in `hooks=[neighbor_hook]`. The internal labeling hook is prepended ahead
of your own, so on-the-fly labeling happens first — which costs nothing here,
because the scorer builds and rolls back the teacher's own list regardless of
what is on the batch. Both snippets in this guide and both examples use
neighbor-free demo potentials, so the hook only becomes necessary when a real
MLIP takes the student's place.

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

The teacher runs with autocast disabled whatever precision context surrounds it,
so a mixed-precision training step does not quietly change the targets it is
supervised against. An on-the-fly label is bit-for-bit the label
`label_dataset` would have written, which is what makes the two paths
interchangeable and a mid-run switch between them harmless.

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
counts as one epoch for hooks and epoch-cadence validation, while a step-cadence
validation fires inside segments. The run closes with one terminal validation,
skipped when a cadence already validated at the final step, so a metric-driven
scheduler is never stepped twice on one set of metrics.

```{warning}
**On-policy distillation is single-process for now.** Each segment builds its own
loader from a rank-local replay buffer, and the seed state is not sharded, so
every rank would propagate the same trajectories, pay the same teacher bill, and
train on the same frames. The loop refuses to start in a world of more than one
rank rather than do that silently. Distributing the offline path is unaffected:
label the dataset with `label_dataset` and train the store with a
{py:class}`~nvalchemi.training.hooks.DDPHook`, which shards it as usual.
Rank-sharded generation is planned.
```

The propagator is any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` — an
integrator generating trajectories, or an optimizer generating relaxation paths.
Nothing downstream of the config reads a velocity or a temperature.

Seed structures do have to arrive carrying the fields the propagator reads
*before* its first `compute`. The segment loop propagates through plain
`BaseDynamics.run`, which primes nothing, and a step runs `pre_update` before
`compute`, so the propagator opens on batch fields nobody has written yet. A
seed missing one surfaces on the first propagator step, not at construction —
`AttributeError: 'Batch' has no attribute 'forces'`. The required set is read
off the propagator itself rather than guessed: its `__needs_keys__` are model
outputs, but each one lands in a batch field the step reads, and its
`__provides_keys__` are updated in place from what it finds there. For the
built-ins that comes to `forces` for every integrator and optimizer, plus
`stress` for NPT, NPH, and the variable-cell FIRE optimizers, which also need a
`cell` because nothing fills one in for an aperiodic structure. Zeros are
enough for the model outputs, since the first `compute` overwrites them; that
is what `build_systems(..., predictions=True)` writes in the example.
`velocities` and `atomic_masses` need no supplying —
{py:class}`~nvalchemi.data.AtomicData` fills both, unless a store they were
written to dropped them.

Seeds are therefore shaped differently from anchor frames, which must carry no
`energy` or `forces` at all. What a seed may safely carry is the *stale* half of
that: `status` and `system_id` describe the run that wrote them, and a store
filled by an earlier relaxation hands back structures already sitting at their
exit status, which the propagator would read as "already finished" and refuse to
move. The loop strips that bookkeeping from every seed batch and installs its
own, so seeding from a previous run's output is safe without a cleanup pass.

`seed_dataset` is propagated whole, as a single batch, so it *is* the set of
systems the run generates from — size it to the device. A
{py:class}`~nvalchemi.dynamics.sampler.SizeAwareSampler` can bin-pack the initial
batch instead, and is configured in place of a `seed_dataset` rather than
alongside one.

`label_frequency` is the throughput knob. The teacher is the expensive model, and
a segment that labels every tenth frame costs a tenth of the teacher passes while
still generating every frame at student speed. The cadence is counted against
the propagator's cumulative `step_count`, which carries across segments, and it
is read before the count is incremented, so a frequency `f` fires at steps
`0, f, 2f, ...` while a segment's forced last frame is one step later. With
`segment_steps` a multiple of `label_frequency` — the defaults, `100` and
`100`, among them — the two would land on adjacent frames at every boundary and
pay two teacher passes for what is effectively one frame, so the hook passes
over a cadence dispatch on the step right after a labeled one whenever the
frequency is above `1`; a forced label is never passed over, which keeps an
early-exiting segment and the run's final frame labeled. Under the defaults that
leaves one label per trajectory per segment, on its last frame — plus, in the
first segment of each `run()`, the frame after the first step, which the
cadence lands on at step count zero.

Size `replay_capacity` with that arithmetic in hand. A segment contributes one
frame per trajectory per labeled step, and FIFO eviction retires whole frames in
arrival order, so a capacity that is not a multiple of the trajectory count cuts
a segment's contribution mid-step and leaves the trajectories at the back of
the seed batch over-represented in every mixture drawn afterwards. Make it a
multiple of the number of seeds.

```{note}
Request the same signals on `OnPolicyConfig.teacher_scorer` that the loss reads.
With an anchor there is no choice: the generation scorer's teacher fields and the
anchor's stored ones are compared for equality at construction, so a scorer
narrower than the anchor is rejected outright. It constructs only against an
equally narrow anchor — and then every generated frame is scored twice, once
during generation and again on its way into a training step, so the run pays more
teacher passes than `label_frequency` implies. The strategy warns whenever the
generation scorer is narrower than the loss, anchor or not.
```

Any {py:class}`~nvalchemi.training.distillation.TeacherScorer` may drive
generation, not only the in-process one, and a custom scorer is worth declaring
`label_fields` on — the batch fields its `label()` writes. That declaration is
what the checks above read, through
{py:func}`~nvalchemi.training.distillation.scorer_fields`: a scorer whose signal
names are all built-in has knowable fields without it, but one with a signal
name of its own does not, and the strategy then warns that neither the anchor
parity nor the double-scoring check can run at construction, leaving a mismatch
to surface as a rejected mixture once a whole generation phase has been paid
for. A `label_fields` entry outside the `teacher_*` namespace is refused at
construction, because the hook must never overwrite the `energy` and `forces`
that drive the propagator's next step.

The declaration also opens the one door the reserved `teacher_` prefix has.
Offline, a loss target under that prefix has to name a built-in signal, and a
custom scorer's own field reaches the loss only by being named outside the
prefix. On-policy, a `teacher_*` target naming no built-in signal is accepted
when the propagator's scorer declares it in `label_fields`: generation writes it
onto every captured frame, so the replay buffer carries it into every mixed
batch. Such a *generation-supplied* target derives no signal — the strategy's
own scorer produces built-in signals only, so at least one built-in `teacher_*`
target, or an explicit `teacher_signals`, is still required alongside it — the
anchor has to carry the field too, which the parity check enforces, and
validation data has to arrive already carrying it, since nothing labels it on
the fly: a validation batch without it surfaces as the loss's missing-target
`KeyError`.

A runnable three-segment loop is
{doc}`/examples/intermediate/09_onpolicy_distillation`.

### Relaxation paths need a convergence lifecycle

```{note}
The `convergence` and `recycle_seeds` knobs in this section land with the
relaxation-generation change. Everything else in this guide describes the loop as
it stands today.
```

A relaxation propagator differs from an integrator in one way that matters here:
its trajectories *end*. A structure that reaches its minimum keeps being
propagated by a loop that does not know it has arrived, and the labeling hook
keeps mirroring it, so the replay buffer fills with near-duplicate frames of the
same minimum every `label_frequency` steps. Nothing errors — the run reports
plausible losses over a mixture those duplicates have quietly taken over. Set
`OnPolicyConfig.convergence` to give the trajectories an ending:

```python
from nvalchemi.dynamics import FIRE

on_policy = OnPolicyConfig(
    dynamics=FIRE(student, dt=0.1),
    teacher_scorer=scorer,
    seed_dataset=seed_dataset,
    convergence=0.05,
    recycle_seeds=True,
    replay_ratio=0.25,
    steps_per_segment=32,
    batch_size=16,
    segment_steps=50,
    label_frequency=10,
)
```

`convergence` takes an `fmax` threshold, as above, or a
{py:class}`~nvalchemi.dynamics.ConvergenceHook` for a criterion the shorthand
does not express. A hook passed whole has to migrate status — that is what
freezes a converged structure and later graduates it — to at least the
propagator's exit status, and it has to fire on every step, because a structure
is captured at the step it converges. The float shorthand wires all of that up;
prefer it unless the criterion genuinely needs a hook. Resolving it leaves the
field alone: a threshold stays the plain number a recipe can hold and
serialize, and `OnPolicyConfig.convergence_criterion` is the live hook it stands
for, built once and handed to the lifecycle by identity.

Two further checks wait for `run()`, because they read the propagator and the
seeds rather than the config. A propagator already carrying another
status-migrating {py:class}`~nvalchemi.dynamics.ConvergenceHook` is refused: the
lifecycle owns graduation for the run, and a second migrator graduating a
structure at its own threshold freezes it out of the path capture and leaves the
converged route nothing to store, so the trajectory ends in neither. A criterion
whose `source_status` no freshly stamped seed carries is refused too: the loop
strips the seeds' bookkeeping and stamps its own, so a criterion aimed at some
other status would freeze nothing and graduate nothing while the run reported
itself configured.

With it set, a converged structure freezes where it stopped, is stored once as
the minimum it reached, and graduates out of the active batch at the segment
boundary through the propagator's own refill. Frames then reach the buffer by two
routes that partition them: the labeling hook stores the structures still
relaxing, and the converged ones are labeled in a single teacher pass as their
sink drains onto the buffer's device. Neither route stores a structure twice.
The labeling route narrows to the structures still moving *before* the teacher
pass rather than after it, so a batch that has largely converged stops paying
the teacher passes `label_frequency` implies for structures that have stopped.

Graduation shrinks the batch, because `seed_dataset` is consumed whole to build
it. `recycle_seeds=True` restarts the dataset from its beginning so the
trajectory count holds — it is meaningful only alongside `convergence`, and only
for a `seed_dataset`, since a configured `SizeAwareSampler` backfills from its
own dataset under its own size budget instead. When the last trajectory finishes
with nothing left to seed a fresh one from, the run warns once and spends its
remaining training steps on the frames it already has.

### The mixture ratio

`replay_ratio` — λ — is the fraction of every training batch drawn from
generated frames; the rest comes from `reference_dataset`, the *anchor*. It is
the knob to reach for first, because it decides how far the run is allowed to
follow its own trajectory:

- **λ = 1** trains on generated data alone and takes no anchor. The student is
  pulled entirely toward wherever its own dynamics go, which is also the failure
  mode: if the trajectory drifts into configurations the teacher was never meant
  to describe, nothing pulls it back. Passing a `reference_dataset` anyway is
  rejected rather than ignored, because the anchor would be policed for schema
  and device and then never sampled.
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

Each rebuilt sampler keys its generator on `OnPolicyConfig.seed` plus the segment
index, so the reference draw is reproducible across runs without repeating within
one. That knob, not the global `torch` seed, is the mixture's randomness: an
ensemble or a seed-sensitivity sweep needs distinct values here, or every
replicate draws the same reference samples in the same order. Distinct is not
enough on its own, though. The sampler seeds its generator with the sum, so
consecutive values overlap by a shift of one segment — seed `0`'s second
segment draws exactly what seed `1`'s first segment draws — and replicates
meant to be independent want seeds spaced at least as far apart as the number
of segments a run takes, `num_steps // steps_per_segment`.

### The anchor must be teacher-labeled

A mixed batch is one collated `Batch`, and collation is not a merge: it keeps
only the fields *both* sources hold and drops the rest, while a whole level only
one side holds is zero-filled for the other's samples. Either behavior would be
silent, so both are rejected instead — the anchor's schema is compared against
the buffer's, on a probe batch drawn from the anchor and the buffer's own
frozen schema.

The schema the anchor has to match is the replay-frame contract: the structure,
whatever propagator state travels with it, and the `teacher_*` labels — with none
of the `energy`, `forces`, or `stress` the propagator wrote on the live frame,
which the labeling hook strips on the way into the buffer so a stored frame never
carries the student's self-label under a reference target's name.

```{warning}
**A raw DFT-labeled dataset cannot be used as the anchor.** Its `energy` and
`forces` are reference labels, not teacher labels, and it is refused rather than
quietly mixed in. Running it through
{py:func}`~nvalchemi.training.distillation.label_dataset` is necessary but not
sufficient: labeling carries every source field over, so the labeled store holds
`teacher_energy` and `teacher_forces` *alongside* the `energy` and `forces` it
started with, and the anchor check refuses it just the same.
```

The remedy is to strip the reference labels on the way in, because
`label_dataset` writes what the dataset hands it and applies no transform of its
own. A per-sample transform on the streaming
{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` is the general lever: it
runs on each sample after device transfer, and a field set to `None` there is
gone from the batch `load_batches` re-forms, because
{py:meth}`~nvalchemi.data.Batch.from_data_list` takes its key list from the
sample's non-`None` fields and the reader's `field_levels` only classifies the
keys that survive — it never restores one.

```python
from nvalchemi.data import AtomicData
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset


def strip_reference_labels(
    data: AtomicData, metadata: dict
) -> tuple[AtomicData, dict]:
    """Drop the reference labels a generated frame never carries."""
    data.energy = None
    data.forces = None
    data.stress = None
    return data, metadata


unlabeled = Dataset(
    reader=AtomicDataZarrReader("reference.zarr"),
    device="cpu",
    transforms=[strip_reference_labels],
)
label_dataset(unlabeled, scorer, "anchor.zarr", batch_size=64)
anchor = Dataset(reader=AtomicDataZarrReader("anchor.zarr"), device="cpu")
```

Assigning `None` is the deletion idiom, since `AtomicData` has no `__delitem__`
and `del data["energy"]` raises through the transform pipeline, and the
transform has to return the `(data, metadata)` pair it was handed. Two settings
defeat it. `skip_validation=True` builds the fused batch straight from raw
tensor dicts and never runs the per-sample pipeline, so the labels come back —
leave it at its default here. And the transform has to strip every sample alike,
because a batch whose first sample kept `forces` and whose later ones dropped
them fails collation on a batch-dimension mismatch.

The batch-level equivalent is there when the reference set fits in memory:

```python
from nvalchemi.data import Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, InMemoryDataset


def drop_reference_labels(batch: Batch) -> Batch:
    """Strip the reference labels a generated frame never carries."""
    for key in ("energy", "forces", "stress"):
        if key in batch:
            del batch[key]
    return batch


unlabeled = InMemoryDataset(
    reader=AtomicDataZarrReader("reference.zarr"),
    batch_transforms=[drop_reference_labels],
)
label_dataset(unlabeled, scorer, "anchor.zarr", batch_size=64)
anchor = Dataset(reader=AtomicDataZarrReader("anchor.zarr"), device="cpu")
```

That one runs once, as the resident batch is materialized, so every chunk
`label_dataset` reads is already stripped; `Batch`, unlike `AtomicData`, does
support `del`. Both satisfy the `load_batches` contract `label_dataset`
consumes, so choose on memory: the streaming form has no ceiling.

The second recipe is to label structures that never carried reference labels at
all, which is what `build_systems(..., predictions=False)` does in
{doc}`/examples/intermediate/09_onpolicy_distillation`.

Two checks enforce all of this, at different seams. At construction, the
`teacher_*` field sets of the two sources are compared for *equality* — an anchor
with no teacher labels, one narrower than the generation scorer, and one wider
all fail alike — and the anchor is probed for any field a generated frame can
never carry. That second check is the one that catches a store labeled over an
existing reference set, and it fires before a single teacher pass is paid.

What is left for the first segment's mixed loader is the full field, level, and
dtype comparison against real frames: an anchor *missing* something the frames
carry, holding a level they do not, or carrying a field at another dtype,
surfaces only once a segment has generated and labeled. That is a whole segment
of forward passes with an expensive teacher, so
it is worth knowing the frame schema up front. It is enumerable: a stored frame
is whatever the seed structures carry, minus everything run-local — `energy`,
`forces`, `stress`, the neighbor tensors, and the dynamics bookkeeping (`status`,
`system_id`) — plus one field per teacher signal. Seeds built from plain
{py:class}`~nvalchemi.data.AtomicData` with `energy` and `forces` zero-filled
therefore store `positions`, `atomic_numbers`, `atomic_masses`,
`atom_categories`, and `velocities`, plus `cell` and `pbc` for a periodic system.

To read it off the run rather than off this list, take one throwaway segment with
no anchor and compare:

```python
probe = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs=optimizer_configs,
    loss_fn=loss_fn,
    num_steps=1,
    on_policy=OnPolicyConfig(
        dynamics=dynamics,
        teacher_scorer=scorer,
        seed_dataset=seed_dataset,
        replay_ratio=1.0,
        steps_per_segment=1,
        batch_size=1,
        segment_steps=1,
        label_frequency=1,
    ),
)
probe.run()
print(sorted(probe.replay_buffer.schema))
```

The names come back as `level.field`, the form the mismatch is reported in, so
strip the level off each one to compare against the anchor's bare
`field_names`, or compare them against the same `level.field` names read off a
probe batch drawn from the anchor.

Dtype parity is the part of that comparison that is easy to break by accident.
Collation casts the second part of a mixed batch to the dtype of the first, and
which source leads a chunk is not fixed, so a float64 anchor beside float32
generated frames would change the targets' precision from chunk to chunk with
nothing to show for it; the loader rejects the pair instead. The trap is that
the two sides do not see the same labels even from one scorer: a store hands
every floating field back at the dtype of the dataset's `positions` — float32
for essentially every dataset — while a generated frame keeps whatever the
generation scorer emitted. Build that scorer with an explicit `cast_to` matching
what the store returns, and label the anchor with the same scorer. The example
needs no cast only because its teacher already computes at float32.

Supervising one batch from teacher labels and reference labels at once is
masked-composition work that is not modeled yet. Until it lands, an on-policy run
is supervised by the teacher throughout — which is also why annealing between
teacher and reference targets with a
{py:class}`~nvalchemi.training.losses.base.LossWeightSchedule` is an offline
technique, where every sample carries its own reference labels.

Both mixture sources are collated before the strategy moves the batch, so the
anchor's device pins the buffer to it: leaving `replay_device` unset stages
generated frames there, and naming a different one is rejected at construction
rather than discovered as a cross-device collation failure mid-run. That device
is the one the anchor actually *emits* on. A declaration settles it where one
exists — the `device=` a Zarr-backed
{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` was opened with, or the
device an `InMemoryDataset` already holds its batch on — and otherwise a probe
batch is drawn and measured: a
{py:class}`~nvalchemi.data.datapipes.multidataset.MultiDataset` declares no
device at all, and a store opened without one declares an index-less `cuda`
that names whichever device is current, so both are resolved against the batch
they hand back. The way to move the mixture is therefore to open the anchor on
the device the run trains on, which is what `Dataset` takes `device=` for.
`replay_device` decides anything only in a run with no anchor, where the frames
stay in host memory unless it says otherwise, because that is where the
segment's sink drained them.

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
dynamics. What it leaves behind is a floor rather than a verdict — the residual
is bounded below by how non-conservative the teacher was on the sampled states,
so a force error that stops falling is not by itself evidence of a bad run, and
only the part above that floor reports how well the student fit.

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
teacher pass entirely. Wrap that store in a
{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` — or any iterable of
`Batch` — before handing it to
{py:class}`~nvalchemi.training.ValidationConfig`: a bare `Dataset` iterates
`(AtomicData, metadata)` pairs rather than batches, and the validation loop
moves whole batches to the device. `every_n_epochs=1` validates at every segment
boundary; `every_n_steps` fires inside segments.

**Composed weights are ratios, not coefficients.**
{py:class}`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
default, so the three-term objective above runs at `1/2.2`, `1/2.2`, and
`0.2/2.2`. Build the composition with `normalize_weights=False` for literal
weights, which also stops a weight schedule on one term from rescaling the others
as it ramps.

**Label dtype follows the student, down to single precision.** Teacher labels
are cast to the student's first floating-point parameter dtype, so a float64
teacher feeds a float32 student without a dtype error at the loss. The cast
never goes below float32: a `bfloat16` or `float16` student gets float32
labels, for two reasons. A store hands every floating field back at the dtype
of the dataset's `positions`, float32 for essentially every dataset, so a
narrower label would disagree with what `label_dataset` persisted; and the
graph-balanced reductions the loss terms use accumulate in the residual's
dtype, where a `bfloat16` sum saturates at 256. Such a student therefore needs
`dtype_policy="prediction_to_target"` on its loss terms, which casts the
prediction to the label's dtype and computes the loss in float32. A float64
student is the mirror case: on-the-fly labels reach it at float64, but a store
returns float32, so training from a labeled store needs a `dtype_policy` on the
loss terms too — `"target_to_prediction"` widens the labels to its precision.
The cast is resolved at construction, so a student whose dtype changes
afterwards needs a `dtype_policy` as well.

```{note}
**Checkpoints duplicate the teacher.** `save_checkpoint` serializes every entry
of `models`, teacher included, so every
{py:class}`~nvalchemi.training.hooks.CheckpointHook` write stores a second copy of
the frozen teacher's weights. Size the checkpoint interval accordingly with a
large teacher; storing the teacher by reference is planned.
```

**Spec round-trip is offline-shaped.** `on_policy` and `reference_dataset` hold
live runtime objects — a propagator, a scorer, and datasets — that no spec can
describe, so
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
omits them and warns. A strategy rebuilt from the spec of an on-policy run is
therefore offline-shaped, and the on-policy pieces have to be re-supplied at
construction until full recipe serialization lands. The spec does name its own
class under `strategy_cls`, the key a checkpoint writes with the same value, so
a spec that travels alone still says which strategy rebuilds it, and
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`
refuses one naming a class that is not a `DistillationStrategy`.

**The segment is the restart granularity.** The propagator state is not
checkpointed yet — a restart that carries the trajectory, the propagator's step
count, and the replay frames lands with recipe serialization — so today a
resumed on-policy run continues from a freshly seeded trajectory, and a segment
a checkpoint interrupted part-way is counted as finished on the way in: its
`AFTER_EPOCH` hooks never fire, the batches it had left are not replayed, and
the run opens a fresh segment at the next epoch index rather than redrawing the
reference samples the interrupted one already trained on. An offline run
graduating to the segment loop from a partial epoch is closed the same way. The
replay buffer, in contrast, outlives a run: a second `run()` on one strategy —
continuing a finished run with a raised `num_steps` — appends to the frames the
first filled instead of regenerating them, while still reseeding its own
trajectory, so a `sampler` seed source the first call consumed raises on the
second.

Resuming an on-policy run takes a different API from the offline one.
`on_policy` and `reference_dataset` are excluded from the spec a checkpoint
writes, so
{py:meth}`~nvalchemi.training.TrainingStrategy.load_checkpoint`
returns an offline-shaped strategy whose `run()` rejects the `None` dataloader.
Rebuild the strategy with the same propagator, scorer, anchor dataset, and
hooks, then restore the counters, weights, optimizer state, and checkpointable
hook state into it in place with
{py:meth}`~nvalchemi.training.TrainingStrategy.restore_checkpoint`
— which, unlike `load_checkpoint`, takes no `hooks` override, because it
restores into the hooks the rebuilt strategy already holds:

```python
from nvalchemi.training.hooks import CheckpointHook

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs=optimizer_configs,
    loss_fn=loss_fn,
    num_steps=20,
    on_policy=on_policy,
    reference_dataset=anchor,
    hooks=[CheckpointHook("runs/on_policy/checkpoints", step_interval=3)],
)
strategy.restore_checkpoint("runs/on_policy/checkpoints")
strategy.run()
```

Attaching `on_policy` to an already-loaded strategy is not a substitute:
assignment is not validated, so the propagator would keep a student the
optimizer never updates and the run would silently stop being on-policy.
`num_steps` is an absolute target rather than a budget for the resumed leg, so a
run that already reached it resumes to nothing until the target is raised, and
the replay buffer starts empty on the new instance because it is not
checkpointed.

```{note}
**Reserved knobs.** `replay_eviction="uncertainty"` is reserved for
committee-based frame selection and raises today; use the default `"fifo"`, and
bound `replay_capacity` on long runs, as a multiple of the trajectory count.
`weight_sync_frequency` must be `1`: the
propagator and the trainer share one module object, so an eager run is never out
of sync, and the knob only becomes meaningful once the propagator holds a
compiled or remote copy of the student.
```

## API reference

See {ref}`training-distillation-api` for the API reference for
{py:class}`~nvalchemi.training.distillation.DistillationStrategy`,
{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer`,
{py:func}`~nvalchemi.training.distillation.label_dataset`,
{py:class}`~nvalchemi.training.distillation.OnPolicyConfig`,
{py:class}`~nvalchemi.training.distillation.TeacherLabelHook`,
{py:class}`~nvalchemi.training.distillation.ReplayBuffer`, and
{py:class}`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss`.
