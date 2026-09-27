import os
import pytest
from evaluation.benchmark_runner import BenchmarkLock, BenchmarkRunner

def test_lock_acquire_success(tmp_path):
    """TEST A: One runner acquires the lock successfully."""
    lock_path = tmp_path / "benchmark.lock"
    lock = BenchmarkLock(str(lock_path))
    lock.acquire()
    assert os.path.exists(lock_path)
    lock.release()
    assert not os.path.exists(lock_path)

def test_lock_concurrent_acquire_fails(tmp_path):
    """TEST B: A second runner attempting acquisition while the first lock is held fails immediately."""
    lock_path = tmp_path / "benchmark.lock"
    lock1 = BenchmarkLock(str(lock_path))
    lock2 = BenchmarkLock(str(lock_path))

    lock1.acquire()

    with pytest.raises(RuntimeError, match="BENCHMARK_ALREADY_RUNNING"):
        lock2.acquire()

    lock1.release()

def test_lock_sequential_acquire_success(tmp_path):
    """TEST C: After the first runner releases the lock, a new runner can acquire it."""
    lock_path = tmp_path / "benchmark.lock"
    lock1 = BenchmarkLock(str(lock_path))
    lock2 = BenchmarkLock(str(lock_path))

    lock1.acquire()
    lock1.release()

    # Second should acquire fine
    lock2.acquire()
    lock2.release()

def test_lock_release_on_exception(tmp_path):
    """TEST D: The lock is released correctly when benchmark execution raises an exception."""
    lock_path = tmp_path / "benchmark.lock"

    def buggy_runner():
        lock = BenchmarkLock(str(lock_path))
        lock.acquire()
        try:
            raise ValueError("Something broke")
        finally:
            lock.release()

    with pytest.raises(ValueError, match="Something broke"):
        buggy_runner()

    # The lock should now be released and free to acquire
    lock2 = BenchmarkLock(str(lock_path))
    lock2.acquire()
    lock2.release()

def test_benchmark_runner_lock_integration(tmp_path):
    """TEST E: Normal existing benchmark behavior is unchanged when the lock is available."""
    output_dir = str(tmp_path / "reports")
    runner = BenchmarkRunner(output_dir=output_dir)

    # We won't run full benchmark since we are avoiding TG requests,
    # but we can mock _run_benchmark_internal
    import unittest.mock as mock

    with mock.patch.object(runner, "_run_benchmark_internal") as mock_run:
        runner.run_benchmark(overwrite=True)
        mock_run.assert_called_once()

    # Ensure lock is released after
    lock_path = os.path.join(output_dir, "benchmark.lock")
    assert not os.path.exists(lock_path)
