"""Synthetic regression checks for loader RNG, initialization and epoch-boundary resume."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import train
from training.checkpoint import load_checkpoint
from utils.tools import weights_init


class TinyMyoPS(nn.Module):
    """Exercise the real trainer/loss with dropout, without allocating the full model."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Dropout2d(0.2))
        self.heads = nn.ModuleList(nn.Conv2d(4, classes, 1) for classes in (2, 3, 3))

    def forward(self, c0, lge, t2):
        features = self.body(torch.cat((c0, lge, t2), 1))
        return tuple(head(features).softmax(1) for head in self.heads)


class SkipFirstScaler:
    """CPU simulation of a single AMP overflow, including persisted scaler state."""

    def __init__(self, *args, **kwargs):
        self.value = 1.0
        self.attempts = 0

    def is_enabled(self):
        return True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if self.attempts:
            optimizer.step()

    def update(self):
        if not self.attempts:
            self.value /= 2
        self.attempts += 1

    def state_dict(self):
        return {"value": self.value, "attempts": self.attempts}

    def load_state_dict(self, state):
        self.value, self.attempts = state["value"], state["attempts"]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    data = tmp_path / "data"
    splits = tmp_path / "splits"
    splits.mkdir()
    names = [f"case0002_slice{i:03d}" for i in range(5)]
    (splits / "train.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    rng = np.random.default_rng(41)
    label = np.zeros((128, 128), dtype=np.uint8)
    label[30:80, 30:80] = 1
    label[40:55, 40:55] = 2
    label[55:70, 55:70] = 3
    for modality in ("bSSFP", "LGE", "T2w"):
        folder = data / modality / "train_npz"
        folder.mkdir(parents=True)
        for name in names:
            image = rng.uniform(0.1, 0.9, size=label.shape).astype(np.float32)
            np.savez_compressed(folder / f"{name}.npz", image=image, label=label)
    monkeypatch.setattr(train, "MyoPSNetL", TinyMyoPS)
    initialized = []

    def record_initialization(module):
        initialized.append(type(module).__name__)
        weights_init(module)

    monkeypatch.setattr(train, "weights_init", record_initialization)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def run(name, *, workers=0, epochs_per_run=0, resume=None, extra=()):
        args = [
            "--data_root", str(data), "--split_dir", str(splits),
            "--run_dir", str(tmp_path / name), "--device", "cpu", "--amp", "none",
            "--epochs", "3", "--batch_size", "2", "--num_workers", str(workers),
            "--epochs_per_run", str(epochs_per_run), "--no-tensorboard", "--log_interval", "50",
        ]
        if resume is not None:
            args += ["--resume", str(resume)]
        result = train.main(args + list(extra))
        return result, tmp_path / name / "checkpoints/myopsnet_seed1234.pt"

    yield run, initialized, data
    torch.set_num_threads(old_threads)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("workers", [0, 2])
def test_real_training_loop_resume_matches_continuous(experiment, workers):
    run, initialized, _ = experiment
    _, continuous_path = run("continuous", workers=workers)
    assert initialized
    partial, resumed_path = run("resumed", workers=workers, epochs_per_run=1)
    assert partial["completed_epochs"] == 1
    assert partial["training_complete"] is False
    before = torch.load(resumed_path, map_location="cpu", weights_only=False)
    initial_rng = torch.Generator().manual_seed(1234).get_state()
    assert not torch.equal(before["loader_rng"], initial_rng)
    assert before["initialization"]["id"] == "native_kaiming_normal_v1"
    assert before["model_contract"]["head_classes"]["C0"] == ["background", "whole_myocardium"]
    initialization_calls = len(initialized)
    result, _ = run("resumed", workers=workers, resume=resumed_path)
    assert len(initialized) == initialization_calls  # Resume never reapplies native initialization.
    assert result["training_complete"] is True
    continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    assert resumed["global_step"] == 9  # Includes the final one-sample batch in every epoch.
    for key in ("model", "optimizer", "scheduler", "scaler", "loader_rng", "rng_state"):
        assert_nested_equal(continuous[key], resumed[key])
    assert resumed["resume_contract"] == continuous["resume_contract"]


def test_amp_skip_counter_survives_resume(experiment, monkeypatch):
    run, _, _ = experiment
    monkeypatch.setattr(torch.amp, "GradScaler", SkipFirstScaler)
    _, reference_path = run("reference")
    _, path = run("paused", epochs_per_run=1)
    paused = torch.load(path, map_location="cpu", weights_only=False)
    assert paused["amp_skipped_updates"] == 1
    assert paused["global_step"] == 2
    run("paused", resume=path)
    resumed = torch.load(path, map_location="cpu", weights_only=False)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    assert resumed["amp_skipped_updates"] == 1
    assert resumed["global_step"] == 8
    for key in ("model", "optimizer", "scheduler", "scaler"):
        assert_nested_equal(reference[key], resumed[key])


@pytest.mark.parametrize("override", [("--seed", "1098"), ("--epochs", "4"), ("--base_lr", "0.001")])
def test_resume_rejects_recipe_changes_without_overwriting_checkpoint(experiment, override):
    run, _, _ = experiment
    _, path = run("paused", epochs_per_run=1)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Resume changes training contract"):
        run("paused", resume=path, extra=override)
    assert path.read_bytes() == original


def test_resume_rejects_changed_training_cache(experiment):
    run, _, data = experiment
    _, path = run("paused", epochs_per_run=1)
    sample = data / "LGE/train_npz/case0002_slice000.npz"
    with np.load(sample) as values:
        image, label = values["image"].copy(), values["label"].copy()
    image[0, 0] += 0.01
    np.savez_compressed(sample, image=image, label=label)
    with pytest.raises(ValueError, match="train_cache_sha256"):
        run("paused", resume=path)


def test_legacy_checkpoint_remains_readable_but_cannot_resume_new_recipe(experiment):
    run, _, _ = experiment
    _, path = run("paused", epochs_per_run=1)
    legacy = load_checkpoint(path)
    legacy.pop("resume_contract")
    torch.save(legacy, path)
    assert load_checkpoint(path)["global_step"] == legacy["global_step"]
    with pytest.raises(ValueError, match="lacks the current RNG/initialization"):
        run("paused", resume=path)


def test_inconsistent_resume_counters_are_rejected(experiment):
    run, _, _ = experiment
    _, path = run("paused", epochs_per_run=1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["amp_skipped_updates"] = 2
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Inconsistent resume budget"):
        run("paused", resume=path)


def test_changed_training_source_is_rejected(experiment):
    run, _, _ = experiment
    _, path = run("paused", epochs_per_run=1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["resume_contract"]["source_sha256"]["network/model.py"] = "changed"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="source_sha256.network/model.py"):
        run("paused", resume=path)


def test_corrupt_rng_is_not_silently_ignored(experiment):
    run, _, _ = experiment
    _, path = run("paused", epochs_per_run=1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["rng_state"]["numpy_random"] = None
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Missing RNG state: numpy_random"):
        run("paused", resume=path)


def test_completed_run_does_not_reinitialize_or_overwrite(experiment):
    run, initialized, _ = experiment
    _, path = run("completed")
    original = path.read_bytes()
    calls = len(initialized)
    result, _ = run("completed", resume=path)
    assert result["training_complete"] is True
    assert len(initialized) == calls
    assert path.read_bytes() == original


def test_session_boundary_saves_even_between_periodic_checkpoints(experiment):
    run, _, _ = experiment
    result, path = run("paused", epochs_per_run=1, extra=("--save_interval", "10"))
    assert result["completed_epochs"] == 1
    assert path.is_file()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["planned_epochs"] == 3
    assert checkpoint["total_updates"] == 9


def test_defaults_match_shared_recipe():
    args = train.build_parser().parse_args([])
    assert (args.seed, args.num_workers, args.save_interval, args.accum_steps) == (1234, 4, 1, 1)
