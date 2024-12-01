from pathlib import Path

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("cupy-cuda12x==13.3.0", "numpy==2.1.3")
    .add_local_file(
        Path(__file__).parent / "src" / "countOccurrences.cu",
        "/root/countOccurrences.cu",
    )
    .add_local_file(
        Path(__file__).parent / "src" / "computeSimilarityCounts.cu",
        "/root/computeSimilarityCounts.cu",
    )
)

app = modal.App("advent-of-cude-01", image=image)

with image.imports():
    import cupy as cp


@app.function(gpu="a10g")
def sort_diff(left: [int], right: [int]) -> int:
    d_left, d_right = to_gpu(left), to_gpu(right)
    d_left = sort(d_left)
    d_right = sort(d_right)

    return int(sum(abs(d_left - d_right)))


@app.function(gpu="a10g")
def similarity(left: [int], right: [int]) -> int:
    d_left, d_right = to_gpu(left), to_gpu(right)
    d_counts = count_occurrences(d_right)

    score = similarity_from_counts(d_left, d_counts)

    return int(score)


@app.local_entrypoint()
def main(
    input_path: str = Path(__file__).parent / "data" / "test.txt",
    expected_output: int = None,
    algorithm: str = "sort-diff",
):
    input_path = Path(input_path)
    left, right = parse(input_path.read_text())
    if algorithm == "sort-diff":
        output = sort_diff.remote(left, right)
    else:
        output = similarity.remote(left, right)
    print(output)
    if expected_output is not None:
        assert output == expected_output


def similarity_from_counts(d_ints, d_counts):
    kernel_name = "computeSimilarityCounts"
    d_compute_similarity = kernel_from_path(
        Path(__file__).parent / "computeSimilarityCounts.cu", kernel_name
    )

    n = len(d_ints)
    threads_per_block = 256
    grid = (n // threads_per_block + 1,)

    d_score = cp.zeros(1, dtype=cp.int32)
    d_compute_similarity(
        grid,
        (threads_per_block,),
        (d_ints, d_counts, d_score, len(d_ints), len(d_counts)),
    )

    return d_score


def count_occurrences(d_ints, do_sort=True):
    kernel_name = "countOccurrences"
    if do_sort:
        kernel_name += "Sorted"
        d_ints = sort(d_ints)
        mx_int = d_ints[-1]
    else:
        mx_int = d_ints.max()

    d_counts = cp.zeros(int(mx_int) + 1, dtype=cp.int32)
    d_count_occurrences = kernel_from_path(
        Path(__file__).parent / "countOccurrences.cu", kernel_name
    )

    n = len(d_ints)
    threads_per_block = 256
    grid = (n // threads_per_block + 1,)

    d_count_occurrences(
        grid,
        (threads_per_block,),
        (d_ints, d_counts, len(d_ints), len(d_counts)),
        shared_mem=4 * threads_per_block if do_sort else 0,  # just a compiler hint?
    )

    return d_counts


def sort(arr):
    arr.sort()
    return arr


def kernel_from_path(path: Path, kernel_name: str):
    return cp.RawKernel(path.read_text(), kernel_name)


def parse(text: str) -> (list[int], list[int]):
    ints = text.split()
    left, right = map(int, ints[::2]), map(int, ints[1::2])
    return list(left), list(right)


def to_gpu(ints):
    d_ints = cp.asarray(ints, dtype=cp.int32)

    return d_ints
