from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil

import numpy as np
import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, EpochRandomSampler, collate_fn
from f5_tts.model.utils import default, exists
from f5_tts.train.lora_resume_contract import (
    RUNTIME_STATE_NAME,
    STATE_NAME,
    prunable_checkpoints,
    validate_checkpoint,
    write_sidecar,
)


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        use_lora: bool = False,  # LoRA fine-tuning mode (disables EMA, saves only adapter weights)
        resume_from: str = "",
        trust_resume_state: bool = False,
        resume_contract: dict | None = None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                    "bnb_optimizer": bnb_optimizer,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = None
            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model
        self.use_lora = use_lora
        self.resume_from = resume_from
        self.trust_resume_state = trust_resume_state
        self.resume_contract = resume_contract
        self._resume_state = {"completed_updates": 0, "data_epoch": 0, "batches_consumed_in_epoch": 0}
        self._resume_runtime_state = None
        self._saved_lora_updates: set[int] = set()
        if self.use_lora and self.resume_contract is None:
            raise ValueError("LoRA training requires a guarded resume contract")

        if self.is_main:
            if self.use_lora:
                self.ema_model = None
                print("LoRA mode: EMA disabled, only adapter weights will be saved")
            else:
                self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
                self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate)
        else:
            self.optimizer = AdamW(trainable_params, lr=learning_rate, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            if self.use_lora:
                if update < 1:
                    raise ValueError("A resumable LoRA checkpoint requires one completed optimizer update")
                unwrapped = self.accelerator.unwrap_model(self.model)
                final_dir = os.path.join(self.checkpoint_path, f"lora_{update}")
                if os.path.lexists(final_dir):
                    if update not in self._saved_lora_updates:
                        raise FileExistsError(f"Refusing to reuse unowned immutable LoRA checkpoint: {final_dir}")
                else:
                    lora_dir = os.path.join(self.checkpoint_path, f".lora_{update}.partial.{os.getpid()}")
                    os.mkdir(lora_dir)
                    unwrapped.save_pretrained(lora_dir)
                    training_state_path = os.path.join(lora_dir, "training_state.pt")
                    torch.save(
                        dict(
                            optimizer_state_dict=self.optimizer.state_dict(),
                            scheduler_state_dict=self.scheduler.state_dict(),
                            update=update,
                        ),
                        training_state_path,
                    )
                    scaler_state = (
                        self.accelerator.scaler.state_dict()
                        if self.accelerator.scaler is not None and hasattr(self.accelerator.scaler, "state_dict")
                        else None
                    )
                    torch.save(
                        {
                            "python_rng_state": random.getstate(),
                            "numpy_rng_state": np.random.get_state(),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "scaler_state": scaler_state,
                        },
                        os.path.join(lora_dir, RUNTIME_STATE_NAME),
                    )
                    with open(os.path.join(lora_dir, STATE_NAME), "x", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "completed_updates": int(update),
                                "data_epoch": int(self._data_epoch),
                                "batches_consumed_in_epoch": int(self._batches_consumed_in_epoch),
                            },
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    write_sidecar(
                        lora_dir,
                        contract=self.resume_contract,
                        completed_updates=update,
                        required_files=[
                            "adapter_config.json",
                            "adapter_model.safetensors",
                            "training_state.pt",
                            STATE_NAME,
                            RUNTIME_STATE_NAME,
                        ],
                    )
                    os.replace(lora_dir, final_dir)
                    self._saved_lora_updates.add(update)
                    directory_fd = os.open(self.checkpoint_path, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)

                pointer = os.path.join(self.checkpoint_path, "lora_last")
                temporary_pointer = os.path.join(self.checkpoint_path, f".lora_last.{os.getpid()}.tmp")
                if os.path.lexists(temporary_pointer):
                    os.unlink(temporary_pointer)
                os.symlink(os.path.basename(final_dir), temporary_pointer, target_is_directory=True)
                os.replace(temporary_pointer, pointer)

                if last:
                    print(f"Saved LoRA adapter at update {update} -> {final_dir}")
                if self.keep_last_n_checkpoints > 0:
                    for oldest in prunable_checkpoints(self.checkpoint_path, self.keep_last_n_checkpoints):
                        shutil.rmtree(oldest)
                        self._saved_lora_updates.discard(int(oldest.name.split("_")[1]))
                        print(f"Removed old LoRA checkpoint: {oldest.name}")
            else:
                checkpoint = dict(
                    model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    ema_model_state_dict=self.ema_model.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    update=update,
                )
                if last:
                    self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                    print(f"Saved last checkpoint at update {update}")
                else:
                    if self.keep_last_n_checkpoints == 0:
                        return
                    self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                    if self.keep_last_n_checkpoints > 0:
                        checkpoints = [
                            f
                            for f in os.listdir(self.checkpoint_path)
                            if f.startswith("model_")
                            and not f.startswith("pretrained_")
                            and f.endswith(".pt")
                            and f != "model_last.pt"
                        ]
                        checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                        while len(checkpoints) > self.keep_last_n_checkpoints:
                            oldest_checkpoint = checkpoints.pop(0)
                            os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                            print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return 0

        dir_contents = os.listdir(self.checkpoint_path)

        if self.use_lora:
            if not self.resume_from:
                return 0
            lora_dir, self._resume_state, _ = validate_checkpoint(
                self.resume_from,
                output_dir=self.checkpoint_path,
                expected_contract=self.resume_contract,
                trust_resume_state=self.trust_resume_state,
                world_size=self.accelerator.num_processes,
            )

            self.accelerator.wait_for_everyone()

            # Load LoRA adapter weights
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
            adapter_state = load_file(adapter_path, device="cpu")
            set_peft_model_state_dict(self.accelerator.unwrap_model(self.model), adapter_state)
            print(f"Loaded LoRA adapter from {lora_dir}")

            # Load optimizer/scheduler state for resumption
            training_state_path = os.path.join(lora_dir, "training_state.pt")
            training_state = torch.load(training_state_path, weights_only=True, map_location="cpu")
            if training_state.get("update") != self._resume_state["completed_updates"]:
                raise ValueError("LoRA optimizer state disagrees with the validated completed-update boundary")
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
            update = training_state["update"]
            del training_state
            self._resume_runtime_state = torch.load(
                os.path.join(lora_dir, RUNTIME_STATE_NAME), weights_only=False, map_location="cpu"
            )

            gc.collect()
            return update

        # Standard (non-LoRA) checkpoint loading
        if not any(filename.endswith((".pt", ".safetensors")) for filename in dir_contents):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in dir_contents:
            latest_checkpoint = "model_last.pt"
        else:
            all_checkpoints = [
                f
                for f in dir_contents
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if self.batch_size_type == "sample":
            sampler = EpochRandomSampler(train_dataset, resumable_with_seed) if exists(resumable_with_seed) else None
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=sampler is None,
                sampler=sampler,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
        skipped_epoch = int(self._resume_state["data_epoch"])
        skipped_batch = int(self._resume_state["batches_consumed_in_epoch"])
        restore_runtime_state = self._resume_runtime_state

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if hasattr(train_dataloader, "set_epoch"):
                train_dataloader.set_epoch(epoch)
            elif hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            if epoch == skipped_epoch and skipped_batch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
                self._batches_consumed_in_epoch = skipped_batch
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader
                self._batches_consumed_in_epoch = 0
            self._data_epoch = epoch

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                if restore_runtime_state is not None:
                    random.setstate(restore_runtime_state["python_rng_state"])
                    np.random.set_state(restore_runtime_state["numpy_rng_state"])
                    torch.set_rng_state(restore_runtime_state["torch_rng_state"])
                    if torch.cuda.is_available() and restore_runtime_state.get("cuda_rng_state") is not None:
                        torch.cuda.set_rng_state_all(restore_runtime_state["cuda_rng_state"])
                    scaler_state = restore_runtime_state.get("scaler_state")
                    if (
                        scaler_state is not None
                        and self.accelerator.scaler is not None
                        and hasattr(self.accelerator.scaler, "load_state_dict")
                    ):
                        self.accelerator.scaler.load_state_dict(scaler_state)
                    restore_runtime_state = None

                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main and self.ema_model is not None:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                self._batches_consumed_in_epoch += 1
                if self._batches_consumed_in_epoch >= len(train_dataloader):
                    self._data_epoch = epoch + 1
                    self._batches_consumed_in_epoch = 0

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                save_last = global_update % self.last_per_updates == 0
                save_numbered = global_update % self.save_per_updates == 0
                if (save_last or save_numbered) and self.accelerator.sync_gradients:
                    if save_numbered and self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

                    self.save_checkpoint(global_update, last=save_last)

            self._data_epoch = epoch + 1
            self._batches_consumed_in_epoch = 0

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
