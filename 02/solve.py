from pathlib import Path

import modal

here = Path(__file__).parent
COLS = 8

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("cupy-cuda12x==13.3.0", "numpy==2.1.3")
    .add_local_file(here / "src" / "markSafe.cu", "/root/markSafe.cu")
)

app = modal.App("advent-of-cude-02", image=image)

with image.imports():
    import cupy as cp


@app.function(gpu="a10g")
def process(reports: list[list[int]], verbose=False) -> int:
    d_reports = ragged_to_padded_array(reports)
    n_reports = len(d_reports)
    d_safe = cp.zeros(n_reports, dtype=cp.int8)

    if verbose:
        print(d_reports)

    d_mark_safe = kernel_from_path(Path("/root/markSafe.cu"), "markSafe")

    threads_per_block = 256
    block_dim_x = threads_per_block // COLS
    grid = (n_reports // block_dim_x + 1, 1)

    d_mark_safe(
        grid,
        (block_dim_x, COLS),
        (d_reports, d_safe, n_reports),
    )

    if verbose:
        print(d_safe)

    return int(sum(cp.asarray(d_safe, dtype=bool)))


@app.local_entrypoint()
def main(
    input_path: str = Path(__file__).parent / "data" / "test.txt",
    expected_output: int = None,
    verbose: bool = False,
):
    input_path = Path(input_path)
    reports = parse(input_path.read_text())
    output = process.remote(reports, verbose=verbose)
    print(output)
    if expected_output is not None:
        assert output == expected_output


def kernel_from_path(path: Path, kernel_name: str):
    return cp.RawKernel(path.read_text(), kernel_name)


def parse(text: str) -> list[list[int]]:
    return [list(map(int, line.split())) for line in text.splitlines()]


def ragged_to_padded_array(ragged_list, padding_value=-1, pad_to=COLS):
    result = cp.full((len(ragged_list), pad_to), padding_value, dtype=cp.int8)

    for i, sublist in enumerate(ragged_list):
        result[i, : len(sublist)] = cp.array(sublist, dtype=cp.int8)

    return result
