import pytest

from deppseek.checkpoint import CheckpointStore


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return CheckpointStore(workspace / ".deppseek", workspace, "test-session"), workspace


def test_undo_restores_modified_and_removes_created(store):
    cp_store, ws = store
    (ws / "solver.py").write_text("dt = 0.01\n")

    checkpoint = cp_store.snapshot(
        [ws / "solver.py", ws / "new.py"], tool="multi_edit", description="tune"
    )
    (ws / "solver.py").write_text("dt = 0.001\n")
    (ws / "new.py").write_text("print('x')\n")
    cp_store.commit(checkpoint)

    cp_store.undo()
    assert (ws / "solver.py").read_text() == "dt = 0.01\n"
    assert not (ws / "new.py").exists()


def test_undo_walks_back_one_step_at_a_time(store):
    """Regression: undo() selected its target from a freshly-read manifest but
    wrote back a different list, so the `undone` flag was lost and the same
    checkpoint could be undone forever."""
    cp_store, ws = store
    (ws / "a.py").write_text("v1\n")
    for version in (2, 3):
        checkpoint = cp_store.snapshot([ws / "a.py"], tool="edit_file", description=f"v{version}")
        (ws / "a.py").write_text(f"v{version}\n")
        cp_store.commit(checkpoint)

    cp_store.undo()
    assert (ws / "a.py").read_text() == "v2\n"
    cp_store.undo()
    assert (ws / "a.py").read_text() == "v1\n"
    assert cp_store.undo() == "Nothing to undo."


def test_identical_content_is_stored_once(store):
    cp_store, ws = store
    (ws / "a.py").write_text("same\n")
    (ws / "b.py").write_text("same\n")
    checkpoint = cp_store.snapshot([ws / "a.py", ws / "b.py"], tool="t", description="d")
    cp_store.commit(checkpoint)

    blobs = [p for p in (ws / ".deppseek" / "objects").rglob("*") if p.is_file()]
    assert len(blobs) == 1


def test_undo_by_explicit_id(store):
    cp_store, ws = store
    (ws / "a.py").write_text("v1\n")
    first = cp_store.snapshot([ws / "a.py"], tool="edit_file", description="v2")
    (ws / "a.py").write_text("v2\n")
    cp_store.commit(first)

    assert "already undone" not in cp_store.undo(first.id)
    assert (ws / "a.py").read_text() == "v1\n"
    assert "already undone" in cp_store.undo(first.id)


def test_unknown_checkpoint_id_is_reported_not_raised(store):
    cp_store, _ = store
    assert "No checkpoint" in cp_store.undo("cp9999-000000")


def test_torn_manifest_line_does_not_hide_earlier_history(store):
    cp_store, ws = store
    (ws / "a.py").write_text("v1\n")
    checkpoint = cp_store.snapshot([ws / "a.py"], tool="edit_file", description="v2")
    (ws / "a.py").write_text("v2\n")
    cp_store.commit(checkpoint)

    with cp_store.manifest.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "truncated", "created')

    assert len(cp_store.history()) == 1


def test_prune_drops_unreferenced_objects(store):
    cp_store, ws = store
    (ws / "a.py").write_text("v1\n")
    for version in range(2, 6):
        checkpoint = cp_store.snapshot([ws / "a.py"], tool="edit_file", description=f"v{version}")
        (ws / "a.py").write_text(f"v{version}\n")
        cp_store.commit(checkpoint)

    before = len([p for p in (ws / ".deppseek" / "objects").rglob("*") if p.is_file()])
    cp_store.prune(keep=1)
    after = len([p for p in (ws / ".deppseek" / "objects").rglob("*") if p.is_file()])
    assert after < before


def test_oversized_file_is_recorded_as_skipped_not_silently_lost(store, monkeypatch):
    import deppseek.checkpoint as module

    monkeypatch.setattr(module, "MAX_SNAPSHOT_BYTES", 10)
    cp_store, ws = store
    (ws / "big.dat").write_text("x" * 100)

    checkpoint = cp_store.snapshot([ws / "big.dat"], tool="write_file", description="big")
    entry = checkpoint.files[0]
    assert entry.skipped_reason and "snapshot limit" in entry.skipped_reason

    (ws / "big.dat").write_text("y" * 100)
    cp_store.commit(checkpoint)
    assert "FAILED" in cp_store.undo()
