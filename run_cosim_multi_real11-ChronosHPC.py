# run_cosim_chronos_only.py
# Closed-loop somente Chronos com gap real curto (REAL_GAP=20)
# Janelas de previsão intercaladas

import os, time, json, math, hashlib, logging, warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==== LOGGING ====
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    fh = logging.FileHandler("orchestrator_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)

# ==== CONFIG ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEP_TEST  = 100
REAL_GAP   = 20
MIN_TRAIN_SAMPLES = 2400

# versões Chronos
VERSIONS = [
    "chronos-t5-tiny","chronos-t5-mini","chronos-t5-small",
    "chronos-t5-base","chronos-t5-large",
    "chronos-bolt-tiny","chronos-bolt-mini",
    "chronos-bolt-small","chronos-bolt-base"
]

def _build_colors():
    base_colors = plt.rcParams.get("axes.prop_cycle", None)
    if base_colors: palette = base_colors.by_key().get("color", [])
    else: palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                     "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    return {v: palette[i % len(palette)] for i,v in enumerate(VERSIONS)}

# ==== DATA ====
DF_INPUT_PATH = "/home/jonas/CoSimOrq/00-Data/01-dict_input_temperature.csv"
DF_OUTPUT_PATH = "/home/jonas/CoSimOrq/00-Data/01-dict_output_temperature.csv"
df_input  = pd.read_csv(DF_INPUT_PATH)
df_output = pd.read_csv(DF_OUTPUT_PATH)
n_samples = min(len(df_input), len(df_output))
dates = pd.date_range("2023-01-01", periods=n_samples, freq="T")
df_input.index = dates; df_output.index = dates
assert n_samples > MIN_TRAIN_SAMPLES + STEP_TEST + REAL_GAP

# ==== OUTPUT ====
def make_run_id():
    meta = dict(step_test=STEP_TEST, real_gap=REAL_GAP, device=DEVICE,
                min_train=MIN_TRAIN_SAMPLES, ts=datetime.now().strftime("%Y%m%d_%H%M%S"))
    short = hashlib.md5(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:8]
    return f"chronos_ST{STEP_TEST}_RG{REAL_GAP}_{short}", meta

RUN_ID, META = make_run_id()
OUTPUT_ROOT = os.path.join(os.getcwd(),"outputs")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT,RUN_ID)
os.makedirs(OUTPUT_DIR,exist_ok=True)
with open(os.path.join(OUTPUT_DIR,"run_meta.json"),"w") as f: json.dump(META,f,indent=2)

# ==== Chronos ====
chronos_available = False
try:
    from chronos import BaseChronosPipeline as _Chronos
    chronos_available = True
except:
    try:
        from chronos import ChronosPipeline as _Chronos
        chronos_available = True
    except Exception as e:
        logger.warning(f"Chronos indisponível: {e}")

class ChronosWrapper:
    def __init__(self,version): self.version, self.pipeline, self.loaded = version,None,False
    def _dtype(self):
        if DEVICE=="cuda":
            try:
                if torch.cuda.is_bf16_supported(): return torch.bfloat16
            except: pass
        return torch.float32
    def load(self):
        if not chronos_available: return
        try:
            self.pipeline = _Chronos.from_pretrained(f"amazon/{self.version}",
                                                     device_map=DEVICE,torch_dtype=self._dtype())
            self.loaded=True; logger.info(f"{self.version} carregado")
        except Exception as e: logger.warning(f"Falha {self.version}: {e}")
    def predict(self,series,h):
        if not self.loaded: return np.full(h,np.nan)
        s = pd.Series(series).dropna()
        if len(s)<10: return np.full(h,np.nan)
        try:
            ctx = torch.tensor(s.values,dtype=torch.float32)
            ctx = torch.nan_to_num(ctx,0.0,0.0,0.0)
            if hasattr(self.pipeline,"predict_quantiles"):
                quants,_ = self.pipeline.predict_quantiles(context=ctx,prediction_length=h,quantile_levels=[0.5])
                return quants[0,:,0].cpu().numpy()
            else:
                p = self.pipeline.predict(ctx,prediction_length=h)
                return p.cpu().numpy() if hasattr(p,'cpu') else np.asarray(p)
        except: return np.full(h,np.nan)

# ==== Orchestrator ====
class ClosedLoopChronosOnly:
    def __init__(self):
        self.models, self.timelines, self.series_sim = {},{},{ }
        self.pred_windows, self.colors = [], _build_colors()
    def initialize(self):
        if chronos_available:
            for v in VERSIONS:
                w = ChronosWrapper(v); w.load()
                if w.loaded:
                    self.models[v]=w
                    tl = pd.DataFrame(np.nan,index=df_output.index,columns=df_output.columns)
                    ss = df_output.copy(); ss.iloc[MIN_TRAIN_SAMPLES:] = np.nan
                    self.timelines[v],self.series_sim[v] = tl, ss
    def _override_real(self,current):
        real = df_output.iloc[:current]
        for n in self.series_sim:
            self.series_sim[n].iloc[:current] = real.values
    def run(self):
        total = len(df_output)
        pred_start = MIN_TRAIN_SAMPLES
        while pred_start < total:
            pred_end  = min(pred_start+STEP_TEST,total)
            gap_start = pred_end
            gap_end   = min(pred_end+REAL_GAP,total)
            logger.info(f"Treino até {pred_start} | Prev[{pred_start}:{pred_end}] | GapReal[{gap_start}:{gap_end}]")
            self.pred_windows.append((pred_start,pred_end))
            self._override_real(pred_start)
            for n,m in self.models.items():
                ctx = self.series_sim[n].iloc[:pred_start]
                tl  = self.timelines[n]
                for col in df_output.columns:
                    pred = m.predict(ctx[col], pred_end-pred_start)
                    L = min(len(pred),pred_end-pred_start)
                    if L>0:
                        j = df_output.columns.get_loc(col)
                        tl.iloc[pred_start:pred_start+L,j] = pred[:L]
                        self.series_sim[n].iloc[pred_start:pred_start+L,j] = pred[:L]
            if gap_start<gap_end:
                for n in self.timelines: self.timelines[n].iloc[gap_start:gap_end,:] = np.nan
                for n in self.series_sim: self.series_sim[n].iloc[gap_start:gap_end,:] = df_output.iloc[gap_start:gap_end].values
            pred_start += STEP_TEST + REAL_GAP
        for n,tl in self.timelines.items():
            tl.to_csv(os.path.join(OUTPUT_DIR,f"{n}_preds.csv"))
        logger.info("Execução concluída.")

# ==== MAIN ====
if __name__=="__main__":
    for _st in [50,100,250,500]:
        STEP_TEST = _st; REAL_GAP = 20
        RUN_ID,META = make_run_id()
        OUTPUT_DIR = os.path.join(OUTPUT_ROOT,RUN_ID)
        os.makedirs(OUTPUT_DIR,exist_ok=True)
        with open(os.path.join(OUTPUT_DIR,"run_meta.json"),"w") as f: json.dump(META,f,indent=2)
        logger.info(f"\n=== Rodando STEP_TEST={STEP_TEST} REAL_GAP={REAL_GAP} ===")
        orch = ClosedLoopChronosOnly(); orch.initialize(); orch.run()
