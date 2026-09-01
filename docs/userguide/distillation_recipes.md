<!-- markdownlint-disable MD014 -->

(distillation_recipes_guide)=

# Reproducible Distillation Recipes

A distillation run is worth reproducing: the teacher is expensive, the student
is a product, and the number that decides whether the student ships comes from
a holdout the run itself never saw. This guide covers the machinery that makes
a run reproducible --- the JSON recipe the CLI authors and executes, the spec
round trip behind it, teacher-by-reference checkpoints, and restarting an
interrupted on-policy run --- and closes with the catalog mapping each
distillation objective to the literature it comes from and the API symbol that
implements it.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-distillation``
{ref}`agent skill <agent_skills>` for concise instructions on strategy setup,
labeling, on-policy configuration, losses, evaluation, and this CLI.
```

For the concepts --- what a teacher signal is, how the offline and on-policy
loops differ --- see {ref}`training-distillation-api`. This page assumes you
already have a teacher, a student, and a dataset.

## The recipe lifecycle

One recipe file carries a run from authoring to verdict. The six stages are:

1. **Spec.** `distill init` writes a validated `DistillationJobSpec` scaffold
   at a chosen student size. Edit it; it is ordinary JSON.
2. **Pre-flight.** `distill spec report` validates the recipe with the same
   helpers the runtime uses and renders what it intends to do --- derived
   teacher signals, batch composition, acceptance bars --- before a teacher is
   loaded onto a GPU.
3. **Run.** `distill spec run` builds the teacher, the student, the data, and
   the strategy, then runs it. Errors the strategy's own constructor raises
   surface as CLI errors rather than tracebacks.
4. **Checkpoint.** A `CheckpointHook` in `student.hooks` writes periodic
   checkpoints. The frozen teacher is stored *by reference*, not by weight.
5. **Restore.** `DistillationStrategy.load_checkpoint` (or
   `restore_checkpoint` into a constructed strategy) resumes the run. An
   on-policy run resumes its trajectory, its propagator counter, and its
   replay frames as well as its weights.
6. **Evaluate.** `distill evaluate` scores the trained student over the
   recipe's holdout, renders the acceptance report, optionally exports it as
   JSON, and exits non-zero on a missed bar so a sweep can gate on the command.

## The recipe file

`DistillationJobSpec` is the pydantic envelope every command reads. It forbids
unknown keys, so a typo is an error rather than a silently ignored setting.

| Member | Meaning |
| --- | --- |
| `mode` | `"offline"` over a teacher-labeled store, or `"on-policy"` |
| `teacher` | `SourceSpec`: where the frozen teacher comes from |
| `student` | `StudentSpec`: a constructor `spec`, or a `source` checkpoint |
| `dataset` | Training store --- the labeled dataset offline, the anchor on-policy |
| `output` | `run_dir`, and the `checkpoint_dir` hooks write under |
| `validation` | Optional validation cadence |
| `on_policy` | Segment-loop recipe; required in, and only read in, on-policy mode |
| `evaluation` | Holdout and acceptance bars `distill evaluate` gates on |
| `strategy` | The `DistillationStrategy.to_spec_dict()` bundle |
| `notes` | Free text rendered in the report |

A scaffold at the `small` tier, trimmed to its structure:

```json
{
  "name": "small-student-offline-distillation",
  "mode": "offline",
  "teacher": {"model": "mace", "model_id": "small-0b"},
  "student": {
    "tier": "small",
    "spec": {
      "cls_path": "my_package.my_module.MyStudentModel",
      "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_radial": 8}
    }
  },
  "dataset": {"path": "data/labeled.zarr", "format": "alchemi-zarr"},
  "output": {
    "run_dir": "runs/distill",
    "checkpoint_dir": "runs/distill/checkpoints"
  },
  "strategy": {
    "optimizer_configs": {"student": ["<OptimizerConfig spec>"]},
    "num_epochs": null,
    "num_steps": 1000,
    "devices": ["cuda"],
    "loss_fn_spec": "<ComposedLossFunction spec>",
    "training_fn":
      "nvalchemi.training.distillation.strategy.default_distillation_fn",
    "teacher_signals": null,
    "label_missing": true
  }
}
```

The default loss the scaffold writes matches the teacher's energy and forces
--- `EnergyMSELoss(target_key="teacher_energy")` plus
`ForceMSELoss(target_key="teacher_forces")` at weights `1.0` and `10.0`, with
`normalize_weights=False` so those are literal coefficients. The teacher
signals are *derived* from those `teacher_*` targets rather than declared, so
adding a term is all it takes to ask the teacher for another signal.

Run `distill schema` for the full JSON schema, which is what an editor or a
sweep generator should validate against.

## CLI usage

The group registers on the existing training entry point, beside `train` and
`finetune`, and is also installed as a `nvalchemi-distill` alias:

```bash
nvalchemi-training distill --help
nvalchemi-distill --help          # the same group
```

Author, review, run, gate:

```bash
nvalchemi-training distill init \
  --tier small \
  --teacher-model mace --teacher-id small-0b \
  --dataset data/labeled.zarr \
  --holdout-dataset data/holdout.zarr \
  --output-dir runs/distill \
  --out recipe.json

nvalchemi-training distill spec report recipe.json
nvalchemi-training distill spec run recipe.json

nvalchemi-training distill evaluate recipe.json \
  --student-checkpoint runs/distill/checkpoints \
  --json-out acceptance.json
```

`distill init --mode on-policy` additionally writes the segment loop, seeded
from `--seed-dataset` (defaulting to the training store):

```bash
nvalchemi-training distill init --mode on-policy \
  --dataset data/anchor.zarr \
  --seed-dataset data/seeds.zarr \
  --output-dir runs/onpolicy \
  --out onpolicy.json
```

`spec report` is worth reading before every run. It shows the teacher signals
the loss implies, the composition of one training batch, the paths that do not
exist on disk yet, a `checkpoint_dir` set without a `CheckpointHook` to write
into it, a `replay_ratio` of `1.0` that leaves the run with no anchor, and the
acceptance bars the recipe records. `spec run` renders the same card first
unless `--no-report` is passed.

### Student size tiers

`--tier` selects `small`, `base`, or `large`. A tier is a **size template and
nothing else** --- a width, a depth, and a radial-basis count written into
`student.spec.kwargs` for whatever constructor `--student-cls-path` names. It
never selects an architecture or a model family, and `student.tier` is recorded
only so a report and a sweep can say which size a run belongs to. Point the
tier at your own model and edit the numbers freely; the constructor is called
with exactly those keyword arguments.

## Teacher-by-reference checkpoints

The teacher is frozen for the whole run, so writing its weights into every
periodic checkpoint duplicates a model that never changed --- with a foundation
teacher, that duplication dominates the cost of checkpointing. Instead, a
strategy may declare that one of its models is stored *by reference*, and
`DistillationStrategy` declares the teacher.

A referenced model contributes a `model_references` entry in the checkpoint
manifest and its own `spec.json`, but **no** `checkpoints/{N}.pt` weight file:

```json
"model_references": {
  "teacher": {
    "rebuild": "spec",
    "source": "my_package.load_teacher(path='/models/teacher.pt')",
    "fingerprint": {
      "num_tensors": 42,
      "num_elements": 4501000,
      "digest": "9f2c..."
    }
  }
}
```

Loading rebuilds the teacher from that spec --- or takes the live one the
caller supplied --- and checks it against the `fingerprint`, which hashes each
state-dict entry's name and shape together with an evenly strided sample of its
values, read at `float64` on the host so device and reduced dtype do not change
the digest. Sampling is what keeps the check cheap for a large teacher; the
price is that it identifies a model rather than validating it, and a change
confined to the values between two samples can slip past. A source that no
longer holds the checkpointed weights raises `ValueError` at load, so a swapped
teacher fails loudly instead of quietly training a student against a different
one.

Naming the source is the teacher's own job. A wrapper built through a loading
factory publishes it as `checkpoint_spec()` --- the same spec the checkpoint
writes to `models/teacher/spec.json` and rebuilds from. A teacher constructed
in memory has no such source, because its spec would rebuild the architecture
with fresh weights; its weights are then serialized inline as before, and the
strategy warns once per run with the stored-value count so the size of the
duplication is explicit.

## Serializable versus runtime-only

`DistillationStrategy.to_spec_dict()` carries a whole on-policy run, including
`on_policy` and `reference_dataset`, as *references* rather than objects.
`OnPolicyConfig.to_spec_dict()` is the piece that does the work:

| Field | How it round-trips |
| --- | --- |
| Scalar knobs (`replay_ratio`, `steps_per_segment`, `batch_size`, `segment_steps`, `label_frequency`, `replay_capacity`, `replay_eviction`, `replay_device`, `seed`, `weight_sync_frequency`) | Verbatim |
| `dynamics` | `{"cls_path", "kwargs"}`; the student is rebound at build time |
| `teacher_scorer` | Signal set, cast dtype, and the model name `"teacher"` |
| `seed_dataset` | The store path and device it reads |
| `sampler` | **Runtime-only**: omitted with a warning |

Three things stay runtime-only, and all three are omitted rather than
approximated:

- **`sampler`.** It owns a live dataset and a size budget no path names.
  Re-supply it at rebuild, or configure `seed_dataset` instead.
- **A propagator's live collaborators** --- hooks, a convergence hook, sinks.
  A propagator carrying them serializes with a warning naming them, and a
  rebuilt propagator starts without them.
- **In-memory datasets.** A recipe references a dataset by the store it reads,
  so an `InMemoryDataset` raises with the fix in the message: write it with
  `label_dataset` and point the recipe at the path.

Rebuilding needs the models supplied, because a recipe names them by role
rather than serializing a second copy:

```python
config = OnPolicyConfig.from_spec_dict(
    recipe, student=student, teacher=teacher
)
strategy = DistillationStrategy.from_spec_dict(
    spec, models={"student": student, "teacher": teacher}
)
```

`DistillationStrategy.from_spec_dict` also accepts `on_policy`,
`reference_dataset`, and `sampler` overrides, which is how a run whose datasets
live in memory --- or whose propagator carries hooks --- is restored.

A piece the recipe cannot describe leaves the whole `on_policy` entry out of
the spec and says why, rather than writing a recipe that would rebuild into a
different run. A strategy rebuilt from such a spec is offline-shaped.

```{note}
The recipe describes the fields `OnPolicyConfig` declares today. Knobs added by
other work in flight --- a convergence lifecycle for relaxation propagators
among them --- gain their own spec entries as those changes land. Never add a
spec entry for a field the class does not declare.
```

## Restarting an interrupted run

An offline run restarts the way any `TrainingStrategy` does: weights, optimizer
and scheduler state, counters, and hook state come back, and the resumed run
reaches the weights an unbroken run would have.

An on-policy run needs more, because the propagator's position in configuration
space is not in any of that. The strategy therefore carries three extra things
through the checkpoint --- the live trajectory batch, the propagator's
cumulative step count, and the frames already in the replay buffer --- as an
internal checkpointable hook, so no checkpoint-format change is involved and a
run that never generates simply contributes an empty bundle.

That is enough for an exact continuation with the built-in integrators, whose
Langevin noise is drawn from a counter-based generator keyed on the step count:
restore the batch and the counter and the next step draws the noise it would
have drawn. A propagator carrying internal state of its own --- a relaxation
optimizer's history, say --- resumes with that state rebuilt from the restored
batch instead.

Restarting lands on a **segment boundary**. A checkpoint written part-way
through a segment's training phase restores the trajectory it held, but the
resumed run re-enters the segment from the top, so it pays one extra generation
phase for the segment it re-entered. The trajectory is continuous either way;
only the generate/train split shifts.

```python
strategy.restore_checkpoint(run_dir / "checkpoints")
strategy.run()
```

## Objective, literature, API

Each distillation objective supported here matches one teacher signal with one
loss term. The literature column names public work the objective comes from,
for orientation --- these are not implementations of a specific paper.

| Objective | Teacher signal | Public literature | API |
| --- | --- | --- | --- |
| Total energy matching | `energy` → `teacher_energy` | Hinton, Vinyals & Dean 2015 (response-based knowledge distillation); Behler & Parrinello 2007 (energy-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.EnergyMSELoss`, {py:class}`~nvalchemi.training.losses.EnergyMAELoss`, {py:class}`~nvalchemi.training.losses.EnergyHuberLoss` with `target_key="teacher_energy"` |
| Force matching | `forces` → `teacher_forces` | Ercolessi & Adams 1994 (the force-matching method); Czarnecki et al. 2017 (Sobolev training --- fitting a teacher's derivatives, not only its values) | {py:class}`~nvalchemi.training.losses.ForceMSELoss`, {py:class}`~nvalchemi.training.losses.ForceHuberLoss`, {py:class}`~nvalchemi.training.losses.ForceL2NormLoss` with `target_key="teacher_forces"` |
| Stress / virial matching | `stress` → `teacher_stress` | Thompson et al. 2015 (virial-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.StressMSELoss`, {py:class}`~nvalchemi.training.losses.StressHuberLoss` with `target_key="teacher_stress"` |
| Per-atom energy decomposition | `node_energies` → `teacher_node_energies` | Behler & Parrinello 2007 (atomic energy decomposition); Romero et al. 2015 (FitNets --- supervising a student on a teacher's intermediate targets) | {py:class}`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss` |

The generation side has a literature of its own: training a student on the
states its own policy visits, rather than only on states a reference
distribution supplies, is the argument of Ross, Gordon & Bagnell 2011 (DAgger),
and it is what {py:class}`~nvalchemi.training.distillation.OnPolicyConfig`
implements for configuration space.

```{note}
**Extension point.** The teacher already produces an `embeddings` signal
(`teacher_node_embeddings`) that no loss term at this revision consumes, and
curvature-level and distribution-matching objectives --- Hessian matching, and
matching the Boltzmann distribution a student samples rather than its
pointwise forces --- are separate work. Adding one means a
{py:class}`~nvalchemi.training.losses.BaseLossFunction` subclass whose
`target_key` names a `teacher_*` field, plus a signal in the scorer if the
field is new; the strategy derives the signal set from the loss and needs no
change. Add a row here when you add the term.
```

## See also

- {ref}`training-distillation-api` --- the full distillation API reference
- {ref}`training_guide` --- strategies, optimizers, checkpoints
- {ref}`losses_guide` --- composing and weighting loss terms
- {ref}`serialization_guide` --- how specs and checkpoints work in general
