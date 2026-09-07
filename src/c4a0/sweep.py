import functools
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from c4a0.training_common import TrainingCancelled
from c4a0.training import SampleDataModule, TrainingGen
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.config import NNSweepConfig
from c4a0_cpp import Sample  # type: ignore


def load_samples(base_dir: str, n_gens: int = 5) -> List[Sample]:
    if (Path(base_dir) / "run.sqlite3").is_file():
        raise ValueError("Neural sweeps require legacy training samples, not a V2 run")
    gens = TrainingGen.load_all(base_dir)[:n_gens]
    game_results = [gen.get_games(base_dir) for gen in gens]
    samples = [
        sample
        for game_result in game_results
        if game_result
        for results in game_result.results
        for sample in results.samples
    ]
    return samples


class _CancellationCallback(pl.Callback):
    def __init__(self, should_cancel):
        self.should_cancel = should_cancel

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._check()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check()

    def _check(self):
        if self.should_cancel is not None and self.should_cancel():
            raise TrainingCancelled("Neural sweep cancelled")


def objective(
    trial: optuna.Trial,
    samples: List[Sample],
    config: NNSweepConfig,
    should_cancel=None,
):
    if len(samples) < 4:
        raise ValueError("Neural sweeps require at least four legacy training samples")
    _CancellationCallback(should_cancel)._check()
    model_config = ModelConfig(
        n_residual_blocks=trial.suggest_int(
            "n_residual_blocks",
            config.residual_blocks_min,
            config.residual_blocks_max,
        ),
        conv_filter_size=trial.suggest_int(
            "conv_filter_size", config.filter_size_min, config.filter_size_max
        ),
        n_policy_layers=trial.suggest_int(
            "n_policy_layers", config.policy_layers_min, config.policy_layers_max
        ),
        n_value_layers=trial.suggest_int(
            "n_value_layers", config.value_layers_min, config.value_layers_max
        ),
        lr_schedule={
            0: trial.suggest_float(
                "learning_rate",
                config.learning_rate_min,
                config.learning_rate_max,
                log=True,
            )
        },
        l2_reg=trial.suggest_float(
            "l2_reg", config.l2_reg_min, config.l2_reg_max, log=True
        ),
    )
    model = ConnectFourNet(model_config)

    batch_size = trial.suggest_categorical("batch_size", config.batch_sizes)

    split_idx = int(0.8 * len(samples))
    train, test = samples[:split_idx], samples[split_idx:]
    data_module = SampleDataModule(train, test, batch_size)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            _CancellationCallback(should_cancel),
            EarlyStopping(monitor="val_loss", patience=4, mode="min"),
        ],
        enable_progress_bar=False,  # Disable progress bar for cleaner logs
        enable_model_summary=False,
        default_root_dir=str(Path(config.base_dir) / "sweeps"),
    )

    trainer.fit(model, data_module)

    val_loss = trainer.callback_metrics["val_loss"].item()
    trial.report(val_loss, step=trainer.current_epoch)
    return val_loss


def perform_hparam_sweep(base_dir: str, study_name: str = "sweep_hparam"):
    return perform_hparam_sweep_config(
        NNSweepConfig(base_dir=base_dir, study_name=study_name)
    )


def perform_hparam_sweep_config(
    config: NNSweepConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
):
    samples = load_samples(config.base_dir, config.n_gens)

    if len(samples) < 4:
        raise ValueError(
            "Neural sweeps require legacy training data with at least four samples"
        )
    _CancellationCallback(should_cancel)._check()

    storage_name = f"sqlite:///{config.study_name}.db"
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    def on_trial_complete(
        study: optuna.Study, _trial: optuna.trial.FrozenTrial
    ) -> None:
        if progress_callback is not None:
            progress_callback(len(study.trials), config.n_trials)
        if should_cancel is not None and should_cancel():
            study.stop()

    study.optimize(
        functools.partial(
            objective, samples=samples, config=config, should_cancel=should_cancel
        ),
        n_trials=config.n_trials,
        callbacks=[on_trial_complete],
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    logger.info("")
    logger.info("Study statistics:")
    logger.info(f"  Finished trials: {len(study.trials)}")
    logger.info(
        f"  Pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}"
    )
    logger.info(
        f"  Completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}"
    )
    return study
