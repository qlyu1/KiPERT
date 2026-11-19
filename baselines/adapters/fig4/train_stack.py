"""Official STACK losses and models, with a PIE split-specific dataset adapter."""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled')
os.makedirs('/dev/shm/pie_stack_checkpoint_tmp', exist_ok=True)
os.environ['TMPDIR'] = '/dev/shm/pie_stack_checkpoint_tmp'
from pathlib import Path
import sys, json, hashlib, gc, time, shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from stack.training.lightning import LightningGeneModel
from stack.finetune.lightning import LightningFinetunedModel
from stack_config import B, MODEL, FOLDS
from stack_data import loaders

def build_model(stage, pre_checkpoint=None):
    epochs = 10 if stage == 'pretrain' else 8
    scheduler = dict(type='cosine', T_max=epochs, warmup_epochs=1,
                     eta_min=1e-6 if stage == 'pretrain' else 5e-6)
    common = dict(model_config=MODEL, learning_rate=3e-5 if stage == 'pretrain' else 2e-5,
                  weight_decay=.003, scheduler_config=scheduler)
    if stage == 'pretrain':
        return LightningGeneModel(**common)
    model = LightningFinetunedModel(**common, n_kept_cell=32)
    if pre_checkpoint is not None:
        ckpt = torch.load(pre_checkpoint, map_location='cpu', weights_only=False)
        state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        result = model.model.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result
        expected = set(model.model.state_dict()) - set(state)
        assert set(result.missing_keys) == expected
        # Every pretrained parameter must transfer exactly. Only fine-tuning heads may be new.
        for k, value in state.items():
            assert torch.equal(value, model.model.state_dict()[k]), k
        print('PRETRAIN_TRANSFER', len(state), 'new_heads', sorted(expected), flush=True)
        del ckpt, state
    # Author's save hook prunes the frozen teacher; on_fit_start restores it from student.
    model.teacher_model.load_state_dict(model.model.state_dict(), strict=True)
    for key,value in model.model.state_dict().items():
        assert torch.equal(value,model.teacher_model.state_dict()[key]), key
    model.strict_loading = False
    return model

def main():
    fold, stage = sys.argv[1:3]
    assert fold in FOLDS and stage in ['pretrain', 'posttrain']
    smoke = '--smoke' in sys.argv
    if not smoke:
        assert (B/'reports/BENCHMARK_COMPLETE.json').exists()
        assert all((B/'runs/evaluation/STATE'/f/'COMPLETE.json').exists() for f in FOLDS)
        assert (B/'manifests/STACK_ADAPTER_VALIDATED.json').exists()
    out = B/'runs/stack'/fold/stage
    if smoke: out = B/'cache/stack_cpu_smoke'/fold/stage
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    torch.set_num_threads(4)
    pl.seed_everything(42, workers=True)
    train, val = loaders(fold, stage, 1 if smoke else 2, workers=0 if smoke else 2)
    pre = out.parent/'pretrain/best.ckpt'
    model = build_model(stage, pre if stage == 'posttrain' else None)
    best = ModelCheckpoint(dirpath=str(out), filename='best', monitor='val_loss',
        mode='min', save_top_k=1, save_weights_only=True, enable_version_counter=False)
    resume_dir = Path('/dev/shm/pie_stack_resume_20261107')/fold/stage
    if smoke: resume_dir = Path('/dev/shm/pie_stack_smoke_resume')/fold/stage
    resume_dir.mkdir(parents=True, exist_ok=True)
    resume = ModelCheckpoint(dirpath=str(resume_dir), save_top_k=0, save_last=True,
        every_n_epochs=1, save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator='cpu' if smoke else 'gpu', devices=1 if smoke else 2,
        strategy='auto' if smoke else 'ddp_find_unused_parameters_true',
        precision='32-true' if smoke else 'bf16-mixed', max_epochs=1 if smoke else (10 if stage=='pretrain' else 8),
        accumulate_grad_batches=1 if smoke else 8, gradient_clip_val=1.,
        callbacks=[best, resume, EarlyStopping(monitor='val_loss', mode='min', patience=10 if stage=='pretrain' else 5)],
        logger=CSVLogger(str(out), name='metrics'), enable_progress_bar=False,
        log_every_n_steps=1 if smoke else 10, num_sanity_val_steps=0 if smoke else 2,
        limit_train_batches=1 if smoke else 1., limit_val_batches=1 if smoke else 1.,
        default_root_dir=str(out))
    last = resume_dir/'last.ckpt'
    trainer.fit(model, train, val, ckpt_path=str(last) if last.exists() else None)
    trainer.strategy.barrier()
    if trainer.is_global_zero:
        ckpt_path = Path(best.best_model_path)
        assert ckpt_path.is_file() and torch.isfinite(best.best_model_score)
        digest = hashlib.file_digest(ckpt_path.open('rb'), 'sha256').hexdigest()
        (out/'COMPLETE.json').write_text(json.dumps({'fold':fold, 'stage':stage,
            'checkpoint':str(ckpt_path), 'sha256':digest, 'best_val_loss':float(best.best_model_score),
            'epochs_completed':trainer.current_epoch, 'global_steps':trainer.global_step,
            'smoke_only':smoke, 'completed_at':time.time()}, indent=2))
        if last.exists(): last.unlink()
        print('STACK_STAGE_COMPLETE', fold, stage, float(best.best_model_score), flush=True)

if __name__ == '__main__': main()
