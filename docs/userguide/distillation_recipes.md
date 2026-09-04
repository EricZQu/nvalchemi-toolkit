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
loops differ --- see {ref}`distillation_guide`; for the symbols behind them,
see {ref}`training-distillation-api`. This page assumes you already have a
teacher, a student, and a dataset.

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
4. **Checkpoint.** The `CheckpointHook` `init` writes into `student.hooks`
   saves periodic checkpoints. The frozen teacher is stored *once per
   checkpoint root*, not once per checkpoint.
5. **Restore.** `distill spec resume` --- or, from Python,
   `DistillationStrategy.load_checkpoint` or `restore_checkpoint` into a
   constructed strategy --- resumes the run. An on-policy run resumes its
   trajectory, its propagator counter, and its replay frames as well as its
   weights.
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
| `evaluation` | Holdout and the accuracy bars `distill evaluate` gates on |
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

`mode` decides which loop runs, and it is the only thing that does. The
`strategy` bundle a Python-side `DistillationStrategy.to_spec_dict()` produces
for an on-policy run carries its own `on_policy` and `reference_dataset`
entries; pasting one into an `"offline"` recipe is rejected rather than quietly
rebuilding the segment loop the recipe says it is not running. In on-policy
mode the top-level `on_policy` block is the one that is built.

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

nvalchemi-training distill spec resume runs/distill/checkpoints \
  --spec recipe.json

nvalchemi-training distill evaluate recipe.json \
  --student-checkpoint runs/distill/checkpoints \
  --json-out acceptance.json
```

`distill init --mode on-policy` additionally writes the segment loop, and
requires `--seed-dataset`:

```bash
nvalchemi-training distill init --mode on-policy \
  --dataset data/anchor.zarr \
  --seed-dataset data/seeds.zarr \
  --output-dir runs/onpolicy \
  --out onpolicy.json
```

The seed store has no default, and omitting the flag exits non-zero rather than
picking one. `--dataset` names the *anchor* --- the store the batch mixture
draws its `1 - replay_ratio` reference share from --- and an anchor cannot
stand in for a seed set. It carries no `forces`, which the propagator reads off
the seed batch before the student's first forward; and one that does carry
`energy` or `forces` of its own is rejected by the strategy at construction,
because the mixture would then zero-fill those targets for every replay row.
Point `--seed-dataset` at a store a dynamics sink or a labeled relaxation
wrote.

`spec report` is worth reading before every run. It shows the teacher signals
the loss implies, the composition of one training batch, the paths that do not
exist on disk yet, a `checkpoint_dir` set without a `CheckpointHook` to write
into it, a `replay_ratio` of `1.0` that leaves the run with no anchor, and the
acceptance bars the recipe records. `spec run` renders the same card first
unless `--no-report` is passed.

Its validation is the real thing rather than a summary of it: an `on_policy`
block is checked against `OnPolicyConfig`'s own field constraints, so a
`replay_ratio` above `1`, an unimplemented `replay_eviction`, a reserved
`weight_sync_frequency`, or a misspelled knob is refused at `spec report` ---
before a teacher reaches a device --- rather than surfacing as a traceback at
`spec run`. What still needs the models built is reported as a CLI error when
they are.

`spec resume` picks an interrupted run back up from its checkpoint directory
and the recipe that started it. The checkpoint carries the models, optimizer
and scheduler state, counters, and the on-policy trajectory; the recipe
supplies the runtime hooks and, offline, the dataloader.

It needs a checkpoint to exist, and `init` writes the hook that produces one. A
scaffold puts a {py:class}`~nvalchemi.training.hooks.CheckpointHook` in
`student.hooks`, pointed at `<output-dir>/checkpoints` --- the same path it
records as `output.checkpoint_dir` --- and saving every `num_steps // 10` steps,
or every step when that would round to less than one:

```json
"student": {
  "hooks": [
    {
      "spec": {
        "cls_path": "nvalchemi.training.hooks.checkpoint.CheckpointHook",
        "timestamp": "2026-09-04T10:30:46.756916+00:00",
        "checkpoint_dir": "runs/distill/checkpoints",
        "step_interval": 100
      },
      "stages": []
    }
  ]
}
```

Edit the interval like any other field; hooks are declared here exactly as they
are in {ref}`finetuning_guide` and {ref}`training_guide`. `timestamp` is the
ISO-8601 stamp every spec carries and `init` fills in --- it is a required
field, so a hand-written hook block needs one too. Because the hook is
written from the start, the `init` / `spec report` / `spec run` /
`evaluate --student-checkpoint` sequence above runs as it is written: the
directory `evaluate` is pointed at is the one the run wrote into. `spec report`
still warns when `output.checkpoint_dir` is set and no `CheckpointHook` writes
into it, which is what a recipe that dropped the hook earns.

### Student size tiers

`--tier` selects `small`, `base`, or `large`. A tier is a **size template and
nothing else** --- a width, a depth, and a radial-basis count written into
`student.spec.kwargs` for whatever constructor `--student-cls-path` names. It
never selects an architecture or a model family, and `student.tier` is recorded
only so a report and a sweep can say which size a run belongs to. Point the
tier at your own model and edit the numbers freely; the constructor is called
with exactly those keyword arguments.

### Acceptance bars a recipe may carry

`distill evaluate` scores the student over the recipe's holdout and does
nothing else, so `evaluation.thresholds` accepts only the four bars that
scoring pass can fill:

```json
"evaluation": {
  "holdout_path": "data/holdout.zarr",
  "targets": "teacher",
  "quantities": ["energy", "forces"],
  "thresholds": {
    "max_energy_per_atom_mae": 0.005,
    "max_forces_mae": 0.05,
    "max_stress_mae": 0.002,
    "min_force_cosine": 0.99
  }
}
```

Any other bar --- `max_energy_drift_per_atom_per_ns`,
`max_energy_drift_per_atom_per_step`, `max_momentum_drift`,
`max_extensivity_error_per_atom`, `max_rdf_jensen_shannon`,
`min_atoms_per_second`, `min_ns_per_day`, `min_drafter_acceptance_rate`,
`require_from_scratch_baseline`, or a `from_scratch_margin` off its default ---
is refused when the recipe is parsed, by `spec report` as much as by
`evaluate`. The refusal is not tidiness. `build_acceptance_report` fails a bar
that has no measurement behind it rather than skipping it, so a recipe carrying
one of these could never be accepted whatever the student did: the run would
end in a verdict formed against a number nobody took. Those bars need a
propagator and a timestep, a supercell builder, or a second trained model, and
a recipe names none of them.

Measure them in Python instead, and assemble one report at the end:

```python
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StabilityMonitor,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    extensivity_error,
    measure_throughput,
)

monitor = StabilityMonitor(timestep_fs=0.5, warmup_steps=200)
propagator.register_hook(monitor)
state = propagator.run(seed_batch, n_steps=2000)

report = build_acceptance_report(
    [
        StudentEvaluation(
            name="small",
            accuracy=evaluate_accuracy(
                student, holdout, targets="teacher", scorer=teacher
            ),
            stability=monitor.metrics(),
            throughput=measure_throughput(propagator, state, timestep_fs=0.5),
            extensivity=extensivity_error(student, state),
        )
    ],
    AcceptanceThresholds(
        max_forces_mae=0.05,
        max_energy_drift_per_atom_per_ns=0.005,
        min_ns_per_day=10.0,
        max_extensivity_error_per_atom=1e-4,
    ),
)
print(report.accepted)
```

`StabilityMonitor.metrics` is a method, not an attribute, and it needs at least
two recorded samples at two different steps. Every metric rebuilds from its own
`to_dict` export with `from_dict`, so a sweep can measure each student in its
own job --- `distill evaluate --json-out` for the accuracy half --- and form one
report at the end.

## Teacher checkpoints: stored once per checkpoint root

The teacher is frozen for the whole run, so writing its weights into every
periodic checkpoint duplicates a model that never changed --- with a foundation
teacher, that duplication dominates the cost of checkpointing. Instead, a
strategy may declare that one of its models is stored **once per checkpoint
root**, and `DistillationStrategy` declares the teacher.

The first checkpoint written under a root holds the teacher's weights at
`models/teacher/checkpoints/0.pt`. Every later checkpoint records a
`model_references` entry naming that index and writes no weight file of its
own, so a run's hundredth checkpoint costs the student's weights alone:

```json
"model_references": {
  "teacher": {
    "rebuild": "stored",
    "checkpoint_index": 0,
    "fingerprint": {
      "num_tensors": 42,
      "num_elements": 4501000,
      "digest": "9f2c..."
    }
  }
}
```

Loading reads the weights back from the index the entry names --- into the
rebuilt teacher, or into the live one the caller supplied --- and checks them
against the `fingerprint`, which hashes each state-dict entry's name, shape,
and dtype together with its values, read at `float64` on the host so the device
the weights were loaded on does not change the digest. How many values depends
on the tensor: one holding at most 4096 of them is hashed whole --- which covers
the per-element tables and the biases a change tends to hide in --- while a
larger one contributes 64 values spanning its whole index range, first and last
included, so no tensor ends in a blind tail. Precision is part of the identity:
a copy held at `bfloat16` is a different model to the fingerprint, and is
reported as one, because widening back to `float64` cannot recover what the
cast rounded off. Sampling the large tensors keeps the cost independent of a
foundation teacher's size; the price is that it identifies a model rather than
validating it, and a change confined to the values between two samples of one
large tensor can slip past. A stored copy that was replaced or truncated raises
`ValueError` at load rather than quietly training a student against a different
teacher.

The saved copy, not the teacher's origin, is what a restart reads --- which is
what makes the checkpoint tree self-contained. A teacher's `checkpoint_spec()`
names the factory call that built it, and the checkpoint still writes that to
`models/teacher/spec.json` to rebuild the *architecture* from, exactly as it
does for any other model. It is not trusted for the weights: a teacher loaded
from an nvalchemi checkpoint --- `teacher.model: "native-checkpoint"` in a
recipe, the ordinary way to distill a fine-tuned foundation model --- publishes
the spec of whatever it was originally built from, and rebuilding from that
alone would restore the wrong weights. Storing them once sidesteps the question
and costs one copy per checkpoint root.

One root holds one copy. Saving a *different* copy of a declared model into a
root that already holds one raises `ValueError` instead of storing it: the
`model_references` entry is root-global, so moving it would repoint every
checkpoint already written under that root at weights they were not written
against. A second teacher therefore needs its own checkpoint root --- which is
what a second run wants anyway. The repair path is untouched: a copy that still
matches the fingerprint is written again freely, which is how a root whose
stored weight file went missing is made whole, at a fresh index every
checkpoint under that root then reads from.

## Serializable versus runtime-only

`DistillationStrategy.to_spec_dict()` carries a whole on-policy run, including
`on_policy` and `reference_dataset`, as *references* rather than objects.
`OnPolicyConfig.to_spec_dict()` is the piece that does the work:

| Field | How it round-trips |
| --- | --- |
| Scalar knobs (`replay_ratio`, `steps_per_segment`, `batch_size`, `segment_steps`, `label_frequency`, `replay_capacity`, `replay_eviction`, `replay_device`, `seed`, `weight_sync_frequency`) | Verbatim |
| `dynamics` | `{"cls_path", "kwargs"}`; the student is rebound at build time. A `torch.dtype` or `torch.device` argument travels as its name (`"float64"`, `"cuda:0"`) and is read back for a constructor annotated to take one |
| `teacher_scorer` | Signal set, cast dtype, and the model name `"teacher"` |
| `seed_dataset` | The store path and device it reads |
| `sampler` | **Runtime-only**: omitted with a warning |

Three things stay runtime-only, and all three are omitted rather than
approximated:

- **`sampler`.** It owns a live dataset and a size budget no path names.
  Re-supply it at rebuild, or configure `seed_dataset` instead.
- **A propagator's live collaborators** --- hooks, a convergence hook, sinks,
  and a sampler the propagator holds itself. Serializing a propagator carrying
  any of them warns and names them, and a rebuilt propagator starts without
  them. The check reads the *live* propagator rather than the constructor
  arguments it can be introspected for, so a collaborator registered after
  construction counts, and so does one on a propagator a recipe built --- the
  shortcut below skips the introspection, not the warning. The segment loop's
  own `TeacherLabelHook` is excluded: the loop registers it for the length of a
  run and removes it afterwards, and a rebuilt loop registers its own, so
  naming it would fire at every mid-segment checkpoint and say nothing.

  What a rebuild loses is worth separating. A missing neighbor-list hook is
  loud, not silent: the model reads its neighbor tensors off the batch, and a
  batch carrying none raises `KeyError` on the first step. The genuinely silent
  losses are the others --- a convergence hook, so a relaxation runs its full
  `segment_steps` instead of graduating converged structures; sinks, so the
  frames the run was capturing are never written; a thermostat or logging hook,
  so the trajectory samples the wrong ensemble, or goes unrecorded.
- **In-memory datasets.** A recipe references a dataset by the store it reads,
  so an `InMemoryDataset` raises with the fix in the message: write it with
  `label_dataset` and point the recipe at the path.

A propagator that a recipe built round-trips as the recipe it was built from,
which is a shortcut past the introspection below --- not past the warning above.
Any other one is introspected, and it round-trips **only if every constructor
argument is readable off a same-named attribute**. One that normalizes an
argument into a private internal --- `self._dt_init` for a timestep converted
to internal time units, which every shipped integrator and optimizer does ---
is refused by name, not approximated: rebuilding it would fall back to the
constructor's own defaults for the arguments it hid, which is a different run.
Build such a propagator through `OnPolicyConfig.from_spec_dict`, which keeps
the reference it built from, or re-supply `dynamics` at construction.

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
have drawn. A propagator carrying internal state of its own is not continued
that far. A relaxation optimizer's adaptive history lives outside the batch, so
a resumed `FIRE` run re-initializes its timestep, its mixing coefficient, and
its uphill counter from the constructor arguments: the positions continue, the
acceleration restarts. A run that had climbed to near `dt_max` therefore takes
the same path a fresh relaxation from those positions would.

Restarting lands on a **segment boundary**. A segment a checkpoint interrupted
part-way is counted as finished on the way in: its `AFTER_EPOCH` hooks never
fire, the training batches it had left are *not* replayed, and the mixture
sampler advances past its epoch index instead of redrawing the reference
samples it already trained on. The resumed run opens a **fresh** segment --- and
since every segment begins by generating, a checkpoint taken part-way through a
training phase costs one extra generation phase, for frames the interrupted
segment had already generated once. The trajectory is continuous either way;
only the generate/train split shifts.

Two properties of the restart bundle are worth budgeting for.

**It is rank-local.** The bundle rides in a strategy checkpoint, which
`CheckpointHook` writes on rank zero alone, so it holds one rank's trajectory
and one rank's replay frames. It is consumed only when a single rank wrote it
and a single rank is restoring it. Restarting on a larger world --- or restoring
onto one rank a bundle written on a larger one --- drops it with a `UserWarning`
and reseeds each rank from its own share of the seed source, with a **cold
replay buffer**. Until the first segments refill it, the mixture is drawn from
the reference dataset alone, so budget those segments as cold. (The segment
loop still refuses to start on more than one rank at this revision; the guard
is what keeps the bundle honest for when it does.)

**A restore replaces the replay frames rather than merging them.** The bundle's
frames *are* the buffer as of the checkpoint, and the buffer outlives a `run()`
call, so a strategy restored while still holding the frames it generated would
otherwise carry the pre-checkpoint half of them twice. That is not a loss of
diversity --- the mixed loader draws with replacement --- but a weighting skew
toward the stale pre-restart states, which is exactly backwards for an
on-policy loop, on top of double the buffer memory and an eviction horizon
reached one restart early. `ReplayBuffer.clear()` is the public form of the
same operation.

```python
strategy.restore_checkpoint(run_dir / "checkpoints")
strategy.run()
```

From the CLI the same restart is one command, against the recipe the run
started from:

```bash
nvalchemi-training distill spec resume runs/onpolicy/checkpoints \
  --spec onpolicy.json
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

- {ref}`distillation_guide` --- teacher signals, the two loops, and the concepts
  this page builds on
- {ref}`training-distillation-api` --- the full distillation API reference
- {ref}`training_guide` --- strategies, optimizers, checkpoints
- {ref}`losses_guide` --- composing and weighting loss terms
- {ref}`serialization_guide` --- how specs and checkpoints work in general
