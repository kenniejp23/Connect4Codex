"""Isolate desktop inference so a blocked model cannot prevent game shutdown."""

import ctypes
import multiprocessing
import os
import signal
import threading
import time

import torch

from c4a0.training import TrainingGen
from c4a0.training_v2 import is_v2_run, load_attempt_model, load_champion_model
from c4a0.tournament import ModelID, ModelPlayer, RandomPlayer, UniformPlayer


def _serve(factory, args, connection, parent_pid):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        return
    torch.set_num_threads(1)
    try:
        players = factory(*args)
    except Exception as error:
        connection.send((False, str(error)))
        return
    try:
        while True:
            model_id, positions = connection.recv()
            try:
                result = players[int(model_id)].forward_numpy(positions)
                connection.send((True, result))
            except Exception as error:
                connection.send((False, str(error)))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


class ProcessEvaluator:
    def __init__(self, factory, args, timeout_seconds=30):
        self._factory, self._args = factory, args
        self._timeout = timeout_seconds
        self._process = None
        self._connection = None
        self._closed = threading.Event()
        self._lifecycle = threading.Lock()

    def _start(self):
        with self._lifecycle:
            if self._closed.is_set():
                raise RuntimeError("Game evaluation was closed")
            if self._process is None or not self._process.is_alive():
                self._stop_worker()
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe()
                self._process = context.Process(
                    target=_serve,
                    args=(self._factory, self._args, child, os.getpid()),
                    name="c4a0-interactive-inference",
                    daemon=True,
                )
                try:
                    self._process.start()
                except BaseException:
                    self._connection.close()
                    self._connection = None
                    self._process = None
                    raise
                finally:
                    child.close()
            assert self._connection is not None
            return self._connection

    def forward_numpy(self, model_id, positions):
        connection = self._start()
        deadline = time.monotonic() + self._timeout
        try:
            connection.send((model_id, positions))
            while not self._closed.is_set():
                if connection.poll(0.05):
                    success, result = connection.recv()
                    if not success:
                        raise RuntimeError(result)
                    return result
                if time.monotonic() >= deadline:
                    with self._lifecycle:
                        self._stop_worker()
                    raise RuntimeError(
                        f"Interactive inference exceeded its {self._timeout:g}-second deadline; retry to reload the players"
                    )
            raise RuntimeError("Game evaluation was closed")
        except (EOFError, BrokenPipeError, OSError) as error:
            raise RuntimeError(
                "Interactive inference process exited; retry to reload the players"
            ) from error

    def _stop_worker(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def close(self):
        self._closed.set()
        with self._lifecycle:
            self._stop_worker()


def build_players(red, gold, directory, device):
    return {
        0: _player(red, 0, directory, device),
        1: _player(gold, 1, directory, device),
    }


def _player(spec: str, model_id: int, directory: str, device: str):
    identifier = ModelID(model_id)
    if spec in {"Human", "Uniform"}:
        return UniformPlayer(identifier)
    if spec == "Random":
        return RandomPlayer(identifier)
    if is_v2_run(directory):
        if spec == "Latest Model":
            model = load_champion_model(directory)
        elif spec.startswith(("Attempt ", "Generation ")):
            number = int(spec.split(maxsplit=1)[1])
            model = load_attempt_model(directory, number)
        else:
            raise ValueError(f"Unknown player type: {spec}")
        return ModelPlayer(identifier, model, torch.device(device))
    generations = TrainingGen.load_all(directory)
    if not generations:
        raise FileNotFoundError(
            "No trained model is available. Choose Human, Random, or Uniform."
        )
    if spec == "Latest Model":
        generation = generations[0]
    elif spec.startswith("Generation "):
        number = int(spec.removeprefix("Generation "))
        generation = next((item for item in generations if item.gen_n == number), None)
        if generation is None:
            raise ValueError(f"Generation {number} was not found")
    else:
        raise ValueError(f"Unknown player type: {spec}")
    return ModelPlayer(
        identifier,
        generation.get_model(directory),
        torch.device(device),
    )
