import pandas as pd
import time
import torch
from pathlib import Path
from fastai.text.all import *
from fastai.callback.core import Callback
import pickle
import logging
import os

# ========== [超参数区域：只需修改这几项] ==========
DATA_FILE     = 'ChemBL-LM_train.csv'    # 输入文件名（TSV）
TEXT_COLUMN   = 'spe'                   # 哪一列是 token 流
BATCH_SIZE    = 256
SEQ_LEN       = 72
EPOCHS        = 40
PRINT_EVERY   = 10
LEARNING_RATE = 1e-3
MODEL_NAME    = 'smiles_spe_lm'
# ===============================================

# ---------- 统一输出目录（与脚本同目录） ----------
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / 'models'
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / 'training_log.txt'

# ---------- 设置日志 ----------
logging.basicConfig(filename=log_file, level=logging.INFO)
def log(msg):
    print(msg)
    logging.info(msg)

# ---------- 读取数据 ----------
data_path = script_dir / 'data'
df = pd.read_csv(data_path / DATA_FILE)
df = df.dropna(subset=[TEXT_COLUMN])
df['text'] = df[TEXT_COLUMN].astype(str)

train_df = df.sample(frac=0.98, random_state=42)
valid_df = df.drop(train_df.index)

# ---------- 构建数据加载器 ----------
dls = TextDataLoaders.from_df(pd.concat([train_df, valid_df]),
                              text_col='text',
                              is_lm=True,
                              valid_pct=0.02,
                              bs=BATCH_SIZE,
                              seq_len=SEQ_LEN)

# ---------- 打印进度 Callback ----------
class PrintDetailedProgress(Callback):    
    def __init__(self, total_epochs:int, print_every:int=10):
        self.total_epochs = total_epochs
        self.print_every = print_every

    def before_fit(self):
        self.start_time = time.time()
        self.total_batches = self.n_epoch * len(self.dls.train)
        self.batch_count = 0

    def before_epoch(self):
        self.epoch_start_time = time.time()
        log(f"\n=== Epoch [{self.epoch+1}/{self.total_epochs}] start ===")

    def before_batch(self):
        self.batch_start_time = time.time()

    def after_batch(self):
        self.batch_count += 1
        if self.train_iter % self.print_every == 0:
            now = time.time()
            batch_time = now - self.batch_start_time
            elapsed = now - self.start_time
            remaining = (self.total_batches - self.batch_count) * batch_time
            msg = (f"Epoch [{self.epoch+1}/{self.total_epochs}] "
                   f"Batch [{self.train_iter}/{len(self.dls.train)}] "
                   f"Loss: {self.loss.item():.4f} "
                   f"Batch time: {batch_time:.2f}s "
                   f"ETA: {remaining/60:.1f} min")
            log(msg)

# ---------- 每个 epoch 保存模型 ----------
class SaveEveryEpoch(Callback):
    def __init__(self, base_name='smiles_lm_model'):
        # 将模型保存到当前脚本所在目录的 models 文件夹
        script_dir = Path(__file__).resolve().parent
        self.model_dir = script_dir / 'models'
        self.base_name = base_name

    def before_fit(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def after_epoch(self):
        filename = f"{self.base_name}_epoch{self.epoch+1}"
        save_path = self.model_dir / filename
        self.learn.save(str(save_path), with_opt=False)
        log(f"[Saved] Model after epoch {self.epoch+1} → {save_path}")

# ---------- 创建 Learner ----------
lr_scaled = LEARNING_RATE * dls.bs / 48

learn = language_model_learner(
    dls, AWD_LSTM, drop_mult=0.5, pretrained=False,
    metrics=[accuracy, Perplexity()],
    cbs=[
        PrintDetailedProgress(total_epochs=EPOCHS, print_every=PRINT_EVERY),
        SaveEveryEpoch(base_name='smiles_lm_model')
    ]
)

# ---------- 训练 ----------
learn.unfreeze()
learn.fit_one_cycle(EPOCHS, lr_scaled, moms=(0.95, 0.85, 0.95))

# ---------- 保存最终模型、encoder、vocab ----------
with open(output_dir / 'vocab.pkl', 'wb') as f:
    pickle.dump(learn.dls.vocab, f)

torch.save(learn.model[0].state_dict(), output_dir / 'smiles_encoder.pth')
log("Training complete. Model and encoder saved.")
