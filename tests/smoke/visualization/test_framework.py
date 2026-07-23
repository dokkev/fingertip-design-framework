"""Headless rendering smoke tests for the scientific figure framework."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import pytest

from visualization.adapters.phase4k import (
    CANONICAL_INPUT_FILES,
    input_checksums,
)
from visualization.framework import (
    load_figure_spec,
    load_visualization_dataset,
    render_figure,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PHASE4K_INPUT = REPOSITORY_ROOT / "output" / "phase4_mechanical_transfer_map"
NORMAL_INPUT = REPOSITORY_ROOT / "output" / "normal_indentation_full_field"


def test_visualization_import_does_not_load_kratos() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import visualization; "
                "assert not any(name.startswith('KratosMultiphysics') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def framework_outputs(tmp_path_factory):
    if not all(
        (PHASE4K_INPUT / name).is_file() for name in CANONICAL_INPUT_FILES
    ) or not (NORMAL_INPUT / "dataset_manifest.json").is_file():
        pytest.skip("generated visualization smoke artifacts are not available")
    before = input_checksums(PHASE4K_INPUT)
    root = tmp_path_factory.mktemp("scientific_figures")
    transfer_raw = json.loads(
        (REPOSITORY_ROOT / "examples/transfer_map_comparison.yaml").read_text()
    )
    transfer_raw["datasets"][0]["input_dir"] = str(PHASE4K_INPUT)
    vector_raw = json.loads(
        (REPOSITORY_ROOT / "examples/displacement_vector_atlas.yaml").read_text()
    )
    vector_raw["datasets"][0]["input_dir"] = str(NORMAL_INPUT)
    transfer_spec = load_figure_spec(transfer_raw)
    vector_spec = load_figure_spec(vector_raw)
    dataset = load_visualization_dataset(transfer_spec)
    vector_dataset = load_visualization_dataset(vector_spec)
    transfer_output = root / "transfer"
    vector_output = root / "vectors"
    render_figure(dataset, transfer_spec, output_directory=transfer_output)
    render_figure(
        vector_dataset, vector_spec, output_directory=vector_output
    )

    def tree_hash(directory: Path) -> dict[str, str]:
        return {
            str(path.relative_to(directory)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }

    first = {
        "transfer": tree_hash(transfer_output),
        "vectors": tree_hash(vector_output),
    }
    render_figure(dataset, transfer_spec, output_directory=transfer_output)
    render_figure(
        vector_dataset, vector_spec, output_directory=vector_output
    )
    second = {
        "transfer": tree_hash(transfer_output),
        "vectors": tree_hash(vector_output),
    }
    assert first == second
    assert before == input_checksums(PHASE4K_INPUT)
    return transfer_output, vector_output, dataset


def test_phase4k_adapter_preserves_descriptor_mask(framework_outputs) -> None:
    _, _, dataset = framework_outputs
    source = dataset.metadata["adapters"][0]["phase4k_audit"]
    assert source["valid_step_count"] == 384
    assert any(not case.descriptor_valid for case in dataset.contact_cases)
    assert all(case.codtm_valid for case in dataset.contact_cases)


def test_reference_mesh_matches_phase4k_counts(framework_outputs) -> None:
    _, _, dataset = framework_outputs
    mesh = dataset.mesh("baseline", "medium")
    assert len(mesh.node_ids) == 6774
    assert len(mesh.element_ids) == 13164
    assert mesh.provenance["phase4k_counts_matched"] is True
    assert mesh.provenance["fem_solve_performed"] is False


def test_headless_reference_figures_render(framework_outputs) -> None:
    transfer, vectors, _ = framework_outputs
    for path in (
        transfer / "transfer_map_comparison.png",
        vectors / "displacement_vector_atlas.png",
    ):
        image = mpimg.imread(path)
        assert path.stat().st_size > 10_000
        assert image.shape[0] > 700 and image.shape[1] > 900


def test_png_pdf_and_manifest_are_consistent(framework_outputs) -> None:
    transfer, vectors, _ = framework_outputs
    for directory in (transfer, vectors):
        manifest = json.loads((directory / "plot_manifest.json").read_text())
        assert {item["format"] for item in manifest["output_paths"]} == {
            "png",
            "pdf",
        }
        assert manifest["source_data_files"]
        for output in manifest["output_paths"]:
            path = Path(output["path"])
            assert path.stat().st_size == output["bytes"] > 0
        for source in manifest["source_data_files"]:
            assert Path(source).is_file()


def test_vector_manifest_records_actual_scales(framework_outputs) -> None:
    _, vectors, _ = framework_outputs
    manifest = json.loads((vectors / "plot_manifest.json").read_text())
    assert manifest["deformation_scale"] == 1.0
    assert manifest["arrow_scale"] == 1.0
    vector_panels = [
        item
        for item in manifest["panel_metadata"]
        if item["component"] == "DisplacementVectorPanel"
    ]
    assert len(vector_panels) == 3
    assert all(
        item["represented_vector"] == "physical displacement u=[u_x,u_y]"
        and item["normalized_arrows"] is False
        for item in vector_panels
    )
