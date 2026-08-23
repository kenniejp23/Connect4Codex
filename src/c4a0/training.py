"""
Generation-based network training, alternating between self-play and training.
"""

import copy
from datetime import datetime
import os
import pickle
import shutil
import tempfile
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

from loguru import logger
from pydantic import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import DataLoader

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.utils import BestModelCheckpoint

import c4a0_cpp  # type: ignore
from c4a0_cpp import PlayGamesResult, BUF_N_CHANNELS, N_COLS, N_ROWS, Sample  # type: ignore


class TrainingGen(BaseModel):
    """
    Represents a single generation of training.
    """

    created_at: datetime
    gen_n: int
    n_self_play_games: Optional[int] = None
    n_mcts_iterations: int
    c_exploration: float
    c_ply_penalty: float
    self_play_batch_size: int
    training_batch_size: int
    parent: Optional[datetime] = None
    val_loss: Optional[float] = None
    solver_score: Optional[float] = None
    max_epochs: int = 100
    early_stopping_patience: int = 10
    network_config: Optional[ModelConfig] = None

    @staticmethod
    def _gen_folder(created_at: datetime, base_dir: str) -> str:
        return os.path.join(base_dir, created_at.isoformat())

    def gen_folder(self, base_dir: str) -> str:
        return TrainingGen._gen_folder(self.created_at, base_dir)

    @staticmethod
    def _generation_timestamps(base_dir: str) -> List[datetime]:
        """Return only fully published generation directories."""
        if not os.path.isdir(base_dir):
            return []
        timestamps: List[datetime] = []
        for name in os.listdir(base_dir):
            folder = os.path.join(base_dir, name)
            if name.startswith(".") or not os.path.isdir(folder):
                continue
            try:
                timestamp = datetime.fromisoformat(name)
            except ValueError:
                continue
            if os.path.isfile(os.path.join(folder, "metadata.json")):
                timestamps.append(timestamp)
        return sorted(timestamps, reverse=True)

    def save_all(
        self,
        base_dir: str,
        games: Optional[PlayGamesResult],
        model: ConnectFourNet,
    ):
        os.makedirs(base_dir, exist_ok=True)
        gen_dir = self.gen_folder(base_dir)
        if os.path.exists(gen_dir):
            raise FileExistsError(f"generation directory already exists: {gen_dir}")
        staging_dir = tempfile.mkdtemp(
            prefix=f".{self.created_at.isoformat()}.",
            suffix=".staging",
            dir=base_dir,
        )
        try:
            metadata_path = os.path.join(staging_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                f.write(self.model_dump_json(indent=2))

            play_result_path = os.path.join(staging_dir, "games.pkl")
            with open(play_result_path, "wb") as f:
                pickle.dump(games, f)

            model_path = os.path.join(staging_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # The directory becomes visible to readers only after every artifact exists.
            os.replace(staging_dir, gen_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def save_metadata(self, base_dir: str):
        gen_dir = self.gen_folder(base_dir)
        if not os.path.isdir(gen_dir):
            raise FileNotFoundError(f"generation directory not found: {gen_dir}")
        metadata_path = os.path.join(gen_dir, "metadata.json")
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".metadata.", suffix=".json", dir=gen_dir, text=True
        )
        try:
            with os.fdopen(descriptor, "w") as f:
                f.write(self.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, metadata_path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def load(base_dir: str, created_at: datetime) -> "TrainingGen":
        gen_folder = TrainingGen._gen_folder(created_at, base_dir)
        with open(os.path.join(gen_folder, "metadata.json"), "r") as f:
            return TrainingGen.model_validate_json(f.read())

    @staticmethod
    def load_all(base_dir: str) -> List["TrainingGen"]:
        timestamps = TrainingGen._generation_timestamps(base_dir)
        return [TrainingGen.load(base_dir, t) for t in timestamps]

    @staticmethod
    def load_latest(base_dir: str) -> "TrainingGen":
        timestamps = TrainingGen._generation_timestamps(base_dir)
        if not timestamps:
            raise FileNotFoundError("No existing generations")
        return TrainingGen.load(base_dir, timestamps[0])

    @staticmethod
    def load_latest_with_default(
        base_dir: str,
        n_mcts_iterations: int,
        c_exploration: float,
        c_ply_penalty: float,
        self_play_batch_size: int,
        training_batch_size: int,
        model_config: ModelConfig,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        n_self_play_games: Optional[int] = None,
    ):
        try:
            return TrainingGen.load_latest(base_dir)
        except FileNotFoundError:
            logger.info("No existing generations found, initializing root")
            gen = TrainingGen(
                created_at=datetime.now(),
                gen_n=0,
                n_self_play_games=n_self_play_games,
                n_mcts_iterations=n_mcts_iterations,
                c_exploration=c_exploration,
                c_ply_penalty=c_ply_penalty,
                self_play_batch_size=self_play_batch_size,
                training_batch_size=training_batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                network_config=model_config,
            )
            model = ConnectFourNet(model_config)
            gen.save_all(base_dir, None, model)
            return gen

    def get_games(self, base_dir: str) -> Optional[PlayGamesResult]:
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "games.pkl"), "rb") as f:
            return pickle.load(f)

    def get_model(self, base_dir: str) -> ConnectFourNet:
        """Gets the model for this generation."""
        gen_folder = self.gen_folder(base_dir)
        with open(os.path.join(gen_folder, "model.pkl"), "rb") as f:
            model = pickle.load(f)
            return model


class SolverConfig(BaseModel):
    solver_path: str
    book_path: str
    solutions_path: str


class TrainingCancelled(RuntimeError):
    """Raised when a cooperative training cancellation is requested."""


ProgressCallback = Callable[[str, Dict[str, Any]], None]


class TrainingProgressCallback(pl.Callback):
    def __init__(
        self,
        progress_callback: Optional[ProgressCallback],
        should_cancel: Optional[Callable[[], bool]],
        max_epochs: int,
    ) -> None:
        self.progress_callback = progress_callback
        self.should_cancel = should_cancel
        self.max_epochs = max_epochs

    def on_train_batch_end(
        self, trainer, _module, _outputs, _batch, _batch_idx
    ) -> None:
        if self.should_cancel is not None and self.should_cancel():
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, _module) -> None:
        if self.progress_callback is None:
            return
        metrics = {
            key: float(
                value.detach().cpu() if isinstance(value, torch.Tensor) else value
            )
            for key, value in trainer.callback_metrics.items()
            if key.startswith("train_")
        }
        self.progress_callback(
            "training",
            {
                "current": trainer.current_epoch + 1,
                "total": self.max_epochs,
                "metrics": metrics,
            },
        )


def train_single_gen(
    base_dir: str,
    device: torch.device,
    parent: TrainingGen,
    n_self_play_games: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    self_play_batch_size: int,
    training_batch_size: int,
    solver_config: Optional[SolverConfig] = None,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
) -> TrainingGen:
    """
    Trains a new generation from the given parent.
    First generate games using c4a0_cpp.play_games.
    Then train a new model based on the parent model using the generated samples.
    Finally, save the resulting games and model in the training directory.
    """
    gen_n = parent.gen_n + 1
    logger.info(f"Beginning new generation {gen_n} from {parent.gen_n}")
    if progress_callback is not None:
        progress_callback("self_play", {"generation": gen_n})
    if should_cancel is not None and should_cancel():
        raise TrainingCancelled("training cancelled")

    # TODO: log experiment metadata in MLFlow

    # Self play
    model = parent.get_model(base_dir)
    model.to(device)
    reqs = [c4a0_cpp.GameMetadata(id, 0, 0) for id in range(n_self_play_games)]  # type: ignore

    def self_play_progress(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                "self_play",
                {"generation": gen_n, "current": completed, "total": total},
            )

    try:
        games = c4a0_cpp.play_games(  # type: ignore
            reqs,
            self_play_batch_size,
            n_mcts_iterations,
            c_exploration,
            c_ply_penalty,
            lambda player_id, pos: model.forward_numpy(pos),  # type: ignore
            self_play_progress,
            should_cancel,
        )
    except RuntimeError:
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("training cancelled") from None
        raise
    if should_cancel is not None and should_cancel():
        raise TrainingCancelled("training cancelled")

    # Optionally judge generated policies against solver
    if solver_config is not None:
        if progress_callback is not None:
            progress_callback("solver", {"generation": gen_n})
        logger.info("Scoring policies against solver")
        solver_score = games.score_policies(
            solver_config.solver_path,
            solver_config.book_path,
            solver_config.solutions_path,
        )
        logger.info("Solver score: {}", solver_score)
    else:
        logger.info("Skipping scoring against solver")
        solver_score = None

    # Training
    logger.info("Beginning training")
    if progress_callback is not None:
        progress_callback("training", {"generation": gen_n, "current": 0})
    model = copy.deepcopy(model)
    train, test = games.split_train_test(0.8, 1337)  # type: ignore
    data_module = SampleDataModule(train, test, training_batch_size)
    best_model_cb = BestModelCheckpoint(monitor="val_loss", mode="min")
    if device.type == "cuda":
        accelerator = "gpu"
        devices: int | list[int] = [device.index if device.index is not None else 0]
    elif device.type in {"cpu", "mps"}:
        accelerator = device.type
        devices = 1
    else:
        raise ValueError(f"unsupported training device: {device}")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            best_model_cb,
            EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, mode="min"
            ),
            TrainingProgressCallback(
                progress_callback, should_cancel, max_epochs=max_epochs
            ),
        ],
        num_sanity_val_steps=0,
        default_root_dir=base_dir,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
    )
    trainer.gen_n = gen_n  # type: ignore
    model.train()  # Switch batch normalization to train mode for training bn params
    trainer.fit(model, data_module)
    if should_cancel is not None and should_cancel():
        raise TrainingCancelled("training cancelled")
    logger.info("Finished training")

    best_model = best_model_cb.get_best_model()
    gen = TrainingGen(
        created_at=datetime.now(),
        gen_n=parent.gen_n + 1,
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        parent=parent.created_at,
        val_loss=best_model_cb.best_score,
        solver_score=solver_score,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        network_config=parent.network_config,
    )
    gen.save_all(base_dir, games, best_model)
    if progress_callback is not None:
        progress_callback(
            "saved",
            {
                "generation": gen.gen_n,
                "val_loss": gen.val_loss,
                "solver_score": gen.solver_score,
            },
        )
    return gen


def training_loop(
    base_dir: str,
    device: torch.device,
    n_self_play_games: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    self_play_batch_size: int,
    training_batch_size: int,
    model_config: ModelConfig,
    max_gens: Optional[int] = None,
    solver_config: Optional[SolverConfig] = None,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    should_stop_after_generation: Optional[Callable[[], bool]] = None,
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
) -> TrainingGen:
    """Main training loop. Sequentially trains generation after generation."""
    logger.info("Beginning training loop")
    logger.info("device: {}", device)
    logger.info("n_self_play_games: {}", n_self_play_games)
    logger.info("n_mcts_iterations: {}", n_mcts_iterations)
    logger.info("c_exploration: {}", c_exploration)
    logger.info("c_ply_penalty: {}", c_ply_penalty)
    logger.info("self_play_batch_size: {}", self_play_batch_size)
    logger.info("training_batch_size: {}", training_batch_size)
    logger.info("model_config: \n{}", model_config.model_dump_json(indent=2))
    logger.info("max_gens: {}", max_gens)
    logger.info("max_epochs: {}", max_epochs)
    logger.info("early_stopping_patience: {}", early_stopping_patience)
    logger.info(
        "solver_config: {}", solver_config and solver_config.model_dump_json(indent=2)
    )

    gen = TrainingGen.load_latest_with_default(
        base_dir=base_dir,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        n_self_play_games=n_self_play_games,
    )

    while True:
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("training cancelled")
        gen = train_single_gen(
            base_dir=base_dir,
            device=device,
            parent=gen,
            n_self_play_games=n_self_play_games,
            n_mcts_iterations=n_mcts_iterations,
            c_exploration=c_exploration,
            c_ply_penalty=c_ply_penalty,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
            solver_config=solver_config,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
        )
        if max_gens is not None and gen.gen_n >= max_gens:
            return gen
        if should_stop_after_generation is not None and should_stop_after_generation():
            logger.info("Stopping after completed generation {}", gen.gen_n)
            return gen


SampleTensor = NewType(
    "SampleTensor",
    Tuple[
        torch.Tensor,  # Pos
        torch.Tensor,  # Policy
        torch.Tensor,  # Q Value with penalty
        torch.Tensor,  # Q Value without penalty
    ],
)


class SampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        training_data: List[Sample],
        validation_data: List[Sample],
        batch_size: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        training_data += [s.flip_h() for s in training_data]
        validation_data += [s.flip_h() for s in validation_data]
        self.training_data = [self.sample_to_tensor(s) for s in training_data]
        self.validation_data = [self.sample_to_tensor(s) for s in validation_data]

    @staticmethod
    def sample_to_tensor(sample: Sample) -> "SampleTensor":
        pos, policy, q_penalty, q_no_penalty = sample.to_numpy()
        pos_t = torch.from_numpy(pos)
        policy_t = torch.from_numpy(policy)
        q_penalty_t = torch.from_numpy(q_penalty)
        q_no_penalty_t = torch.from_numpy(q_no_penalty)
        assert pos_t.shape == (BUF_N_CHANNELS, N_ROWS, N_COLS)
        assert policy_t.shape == (N_COLS,)
        assert q_penalty_t.shape == ()
        assert q_no_penalty.shape == ()
        return SampleTensor((pos_t, policy_t, q_penalty_t, q_no_penalty_t))

    def train_dataloader(self):
        return DataLoader(
            self.training_data,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,  # type: ignore
            batch_size=self.batch_size,
        )


def parse_lr_schedule(floats: List[float]) -> Dict[int, float]:
    """Parses an lr_schedule sequence like "0 2e-3 10 8e-4" into a dict of {0: 2e-3, 10: 8e-4}."""
    assert len(floats) % 2 == 0, "lr_schedule must have an even number of elements"
    schedule = {}
    for i in range(0, len(floats), 2):
        threshold = int(floats[i])
        assert threshold == floats[i], (
            "lr_schedule must alternate between gen_id (int) and lr (float)"
        )
        lr = floats[i + 1]
        schedule[threshold] = lr
    return schedule
