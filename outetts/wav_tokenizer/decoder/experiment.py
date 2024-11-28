import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers
import yaml

from outetts.wav_tokenizer.decoder.discriminator_dac import DACDiscriminator
from outetts.wav_tokenizer.decoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from outetts.wav_tokenizer.decoder.feature_extractors import FeatureExtractor
from outetts.wav_tokenizer.decoder.heads import FourierHead
from outetts.wav_tokenizer.decoder.helpers import plot_spectrogram_to_numpy
from outetts.wav_tokenizer.decoder.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss, DACGANLoss
from outetts.wav_tokenizer.decoder.models import Backbone
from outetts.wav_tokenizer.decoder.modules import safe_log
from outetts.wav_tokenizer.decoder.pretrained_model import instantiate_class


class VocosExp(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        resume_config: str = None,
        resume_model: str = None,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])
        self.automatic_optimization = False

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.resume_config = resume_config
        self.resume_model = resume_model
        self.resume = resume

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.dac = DACDiscriminator()
        self.dacdiscriminator = DACGANLoss(self.dac)

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff
        self.validation_step_outputs = []

        # Добавляем метрики для отслеживания
        self.training_step_count = 0
        self.best_val_loss = float('inf')

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
            {"params": self.dac.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate)
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate)

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def forward(self, audio_input, **kwargs):
        features, _, commit_loss = self.feature_extractor(audio_input, **kwargs)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output, commit_loss

    def training_step(self, batch, batch_idx, **kwargs):
        opt_disc, opt_gen = self.optimizers()
        sch_disc, sch_gen = self.lr_schedulers()
        
        audio_input = batch
        self.training_step_count += 1

        # train discriminator
        if self.train_discriminator:
            opt_disc.zero_grad()
            with torch.no_grad():
                audio_hat, _ = self(audio_input, **kwargs)

            loss_dac = self.dacdiscriminator.discriminator_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac

            self.manual_backward(loss)
            opt_disc.step()
            sch_disc.step()

            self.log("train/discriminator/total", loss, prog_bar=True)
            self.log("train/discriminator/multi_period_loss", loss_mp)
            self.log("train/discriminator/multi_res_loss", loss_mrd)
            self.log("train/discriminator/dac", loss_dac)

        # train generator
        opt_gen.zero_grad()
        audio_hat, commit_loss = self(audio_input, **kwargs)
        
        if self.train_discriminator:
            loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

            self.log("train/generator/multi_period_loss", loss_gen_mp)
            self.log("train/generator/multi_res_loss", loss_gen_mrd)
            self.log("train/generator/feature_matching_mp", loss_fm_mp)
            self.log("train/generator/feature_matching_mrd", loss_fm_mrd)
            self.log("train/generator/loss_dac_1", loss_dac_1)
            self.log("train/generator/loss_dac_2", loss_dac_2)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = loss_dac_1 = loss_dac_2 = 0

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.mel_loss_coeff * mel_loss
            + 1000 * commit_loss
            + loss_dac_1
            + loss_dac_2
        )

        self.manual_backward(loss)
        opt_gen.step()
        sch_gen.step()

        # Логируем метрики обучения
        self.log("train/generator/total_loss", loss, prog_bar=True)
        self.log("train/mel_loss_coeff", self.mel_loss_coeff)
        self.log("train/generator/mel_loss", mel_loss)
        self.log("train/commit_loss", commit_loss)
        self.log("train/step", self.training_step_count)
        self.log("train/learning_rate", opt_gen.param_groups[0]['lr'])

        if self.global_step % 100 == 0 and self.global_rank == 0:
            self.logger.experiment.add_audio(
                "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
            )
            with torch.no_grad():
                mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            self.logger.experiment.add_image(
                "train/mel_target",
                plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        audio_hat, commit_loss = self(audio_input, **kwargs)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + 1000 * commit_loss

        # Логируем метрики валидации
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/commit_loss", commit_loss, sync_dist=True)

        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
            self.log("val/best_loss", self.best_val_loss, sync_dist=True)

        output = {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        if self.global_rank == 0:
            outputs = self.validation_step_outputs[0]
            audio_in = outputs["audio_input"]
            audio_pred = outputs["audio_pred"]
            
            self.logger.experiment.add_audio(
                "val/audio_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            self.logger.experiment.add_audio(
                "val/audio_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val/mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val/mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

    @property
    def global_step(self):
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


class WavTokenizer(VocosExp):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        resume: bool = False,
        resume_config: str = None,
        resume_model: str = None,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            backbone=backbone,
            head=head,
            resume_config=resume_config,
            resume_model=resume_model,
            sample_rate=sample_rate,
            initial_learning_rate=initial_learning_rate,
            num_warmup_steps=num_warmup_steps,
            mel_loss_coeff=mel_loss_coeff,
            mrd_loss_coeff=mrd_loss_coeff,
            pretrain_mel_steps=pretrain_mel_steps,
            decay_mel_coeff=decay_mel_coeff,
            evaluate_utmos=evaluate_utmos,
            evaluate_pesq=evaluate_pesq,
            evaluate_periodicty=evaluate_periodicty,
            resume=resume,
        )

    def training_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device)
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        if self.global_rank == 0:
            outputs = self.validation_step_outputs[0]
            audio_in = outputs["audio_input"]
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.add_audio(
                "encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate,
            )

        super().on_validation_epoch_end()
