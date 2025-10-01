# run_cosim_multi2.py  safe v17-mix+dashboard  closed-loop multi-chronos dual-outputs stride-2 intercalado
# Padrao temporal:
# - Janela de previsao: [S, S+STEP_TEST)
# - Gap real:           [S+STEP_TEST, S+2*STEP_TEST)
# - Proxima previsao:   S <- S + 2*STEP_TEST
# Co-simulacao:
# - FNO treina com X_hibrido = [X_base[:current], y_sim_FNO[:current], media_series_sim_Chronos[:current]]
# - Chronos usa somente a propria serie_sim acumulada ate current
# Blindagens:
# - Asserts anti vazamento em cada iteracao
# - Verificacoes de janelas vazias antes de escrever

import os
import sys
import time
import json
import math
import hashlib
import logging
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.lines import Line2D

# ========= LOGGING =========
LOG_FILE = "orchestrator_run.log"
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)

# ========= CONFIG =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEP_TRAIN = 0
STEP_TEST  = 100
HORIZON    = STEP_TEST
FNO_EPOCHS = 1000
MIN_TRAIN_SAMPLES =2400

# versoes do Chronos
versions = [
   "chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small",
   "chronos-t5-base", "chronos-t5-large",
   "chronos-bolt-tiny", "chronos-bolt-mini",
   "chronos-bolt-small", "chronos-bolt-base"
]

# ========= CORES FIXAS POR MODELO =========
ALL_MODELS_ORDER = [
    "chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small",
    "chronos-t5-base", "chronos-t5-large",
    "chronos-bolt-tiny", "chronos-bolt-mini",
    "chronos-bolt-small", "chronos-bolt-base",
    "FNO"
]
def _build_model_colors():
    base_colors = plt.rcParams.get("axes.prop_cycle", None)
    if base_colors is not None:
        palette = base_colors.by_key().get("color", [])
    else:
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                   "#9467bd","#8c564b","#e377c2","#7f7f7f",
                   "#bcbd22","#17becf"]
    colors = {}
    for i, name in enumerate(ALL_MODELS_ORDER):
        colors[name] = palette[i % len(palette)]
    return colors

# ========= DADOS =========
DF_INPUT_PATH2 = "/home/jonas/CoSimOrq/00-Data/Exp3/dict_input_temperature_surface.csv"
DF_OUTPUT_PATH2 = "/home/jonas/CoSimOrq/00-Data/Exp3/dict_output_temperature.csv"

DF_INPUT_PATH = "/home/jonas/CoSimOrq/00-Data/01-dict_input_temperature.csv"
DF_OUTPUT_PATH = "/home/jonas/CoSimOrq/00-Data/01-dict_output_temperature.csv"

logger.info("Carregando dados")
df_input  = pd.read_csv(DF_INPUT_PATH)
df_output = pd.read_csv(DF_OUTPUT_PATH)

n_samples = min(len(df_input), len(df_output))
# indice minutely; se sua base tiver outra freq, ajuste aqui
dates = pd.date_range("2023-01-01", periods=n_samples, freq="T")
df_input  = df_input.iloc[:n_samples].copy();  df_input.index  = dates
df_output = df_output.iloc[:n_samples].copy(); df_output.index = dates

assert n_samples > MIN_TRAIN_SAMPLES + 2*STEP_TEST, "Base insuficiente para padrao intercalado"

# ========= OUTPUT DIR =========
def make_run_id():
    meta = dict(
        step_test=STEP_TEST, horizon=HORIZON, fno_epochs=FNO_EPOCHS,
        min_train=MIN_TRAIN_SAMPLES, device=DEVICE,
        in_file=os.path.basename(DF_INPUT_PATH), out_file=os.path.basename(DF_OUTPUT_PATH),
        n_samples=n_samples, n_inputs=int(df_input.shape[1]), n_outputs=int(df_output.shape[1]),
        chronos_versions=versions, ts=datetime.now().strftime("%Y%m%d_%H%M%S"),
        mode="intercalado_stride2",
        selection_rule="prevday_same_clock"
    )
    key = json.dumps(meta, sort_keys=True).encode()
    short = hashlib.md5(key).hexdigest()[:8]
    return f"closedloop_intercalado_ST{STEP_TEST}_HZ{HORIZON}_MIN{MIN_TRAIN_SAMPLES}_E{FNO_EPOCHS}_{short}", meta

RUN_ID, RUN_META = make_run_id()
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(os.getcwd(), "outputs"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
    json.dump(RUN_META, f, indent=2)
logger.info(f"Output: {OUTPUT_DIR}")

# ========= CHRONOS =========
chronos_available = False
CHRONOS_CLASS = None
try:
    from chronos import BaseChronosPipeline as _ChronosPipeline
    CHRONOS_CLASS = _ChronosPipeline
    chronos_available = True
    logger.info("BaseChronosPipeline carregado")
except Exception:
    try:
        from chronos import ChronosPipeline as _ChronosPipeline
        CHRONOS_CLASS = _ChronosPipeline
        chronos_available = True
        logger.info("ChronosPipeline carregado")
    except Exception as e:
        logger.warning(f"Chronos indisponivel: {e}")
        CHRONOS_CLASS = None
        chronos_available = False

class ChronosWrapper:
    def __init__(self, version):
        self.version = version
        self.pipeline = None
        self.loaded = False

    def _dtype_for_device(self):
        if DEVICE == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float32

    def load_model(self):
        if not chronos_available:
            return
        try:
            self.pipeline = CHRONOS_CLASS.from_pretrained(
                f"amazon/{self.version}",
                device_map=DEVICE,
                torch_dtype=self._dtype_for_device(),
            )
            self.loaded = True
            logger.info(f"{self.version} carregado")
        except Exception as e:
            logger.warning(f"Erro ao carregar {self.version}: {e}")
            self.loaded = False

    def predict(self, series: pd.Series, horizon: int) -> np.ndarray:
        if not self.loaded or series is None:
            return np.full(horizon, np.nan, dtype=float)
        series = pd.Series(series).dropna()
        if len(series) < 10:
            return np.full(horizon, np.nan, dtype=float)
        try:
            ctx = torch.tensor(series.values, dtype=torch.float32)
            ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(self.pipeline, 'predict_quantiles'):
                quants, _ = self.pipeline.predict_quantiles(
                    context=ctx, prediction_length=int(horizon),
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                out = quants[0, :, 1].cpu().numpy()
            else:
                pred = self.pipeline.predict(ctx, prediction_length=int(horizon))
                out = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.asarray(pred)
            return np.asarray(out, dtype=float).reshape(-1)
        except Exception as e:
            logger.error(f"Erro na previsao Chronos {self.version}: {e}")
            return np.full(horizon, np.nan, dtype=float)

# ========= FNO =========
class FNOWrapper:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.trained = False
        self.input_features = None
        self.output_features = None

    def train(self, X_train: np.ndarray, Y_train: np.ndarray) -> bool:
        try:
            from sklearn.preprocessing import StandardScaler
            from neuralop.models import FNO
            import torch.nn as nn

            assert X_train.shape[0] == Y_train.shape[0] and X_train.shape[0] > 0, "Treino vazio"
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)

            self.input_features  = X_train.shape[1]
            self.output_features = Y_train.shape[1]

            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            X_scaled = self.scaler_x.fit_transform(X_train)
            Y_scaled = self.scaler_y.fit_transform(Y_train)

            # FNO 1D: [batch, channels, length]
            X_tensor = torch.tensor(X_scaled.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Y_tensor = torch.tensor(Y_scaled.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            self.model = FNO(
                n_modes=[32],
                hidden_channels=64,
                in_channels=self.input_features,
                out_channels=self.output_features
            ).to(DEVICE)

            opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            self.model.train()
            for epoch in range(FNO_EPOCHS):
                opt.zero_grad()
                preds = self.model(X_tensor)
                preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                loss = loss_fn(preds, Y_tensor)
                if not torch.isfinite(loss):
                    logger.warning(f"FNO loss nao finito no epoch {epoch}, pulando atualizacao")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                if epoch % 10 == 0:
                    logger.info(f"FNO epoch {epoch}: loss={float(loss.item()):.6f}")

            self.trained = True
            return True
        except Exception as e:
            logger.error(f"Erro no treino FNO: {e}")
            self.trained = False
            return False

    def predict_step(self, X_context: np.ndarray) -> np.ndarray:
        if not self.trained or self.output_features is None:
            feats = self.output_features if self.output_features else 1
            return np.full((feats,), np.nan, dtype=float)
        try:
            X_context = np.nan_to_num(X_context, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler_x.transform(X_context)
            X_tensor = torch.tensor(X_scaled.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_scaled = self.model(X_tensor)
                pred_scaled = torch.nan_to_num(pred_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                next_scaled = pred_scaled[0, :, -1].cpu().numpy()
            next_inv = self.scaler_y.inverse_transform(next_scaled.reshape(1, -1))[0]
            next_inv = np.nan_to_num(next_inv, nan=0.0, posinf=0.0, neginf=0.0)
            assert next_inv.shape == (self.output_features,)
            return next_inv.astype(float)
        except Exception as e:
            logger.error(f"Erro na previsao step FNO: {e}")
            return np.full((self.output_features,), np.nan, dtype=float)

# ========= UTIL =========
def build_hybrid_inputs(X_base: pd.DataFrame,
                        y_sim_fno: pd.DataFrame,
                        chronos_sim_dict: dict,
                        current: int) -> np.ndarray:
    """
    Concatena X_base[:current], y_sim_fno[:current], media das series_sim de todas as versoes Chronos [:current].
    Nada de futuro.
    """
    Xb = X_base.iloc[:current].copy()
    Xb = Xb.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Ys = y_sim_fno.iloc[:current].copy()
    Ys = Ys.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sims = []
    for _, df_sim in chronos_sim_dict.items():
        s = df_sim.iloc[:current].copy()
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sims.append(s)

    chronos_mean = None
    if sims:
        base_index = Xb.index
        sims_aligned = [s.reindex(base_index) for s in sims]
        chronos_mean = pd.concat(sims_aligned, axis=1, keys=range(len(sims))).groupby(level=0, axis=1).mean()
        chronos_mean = chronos_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Ys = Ys.add_prefix("yFNO_")
    parts = [Xb, Ys]
    if chronos_mean is not None:
        chronos_mean = chronos_mean.add_prefix("yCHRmean_")
        parts.append(chronos_mean)

    X_hybrid = pd.concat(parts, axis=1)
    assert X_hybrid.shape[0] == Xb.shape[0] == Ys.shape[0], "X_hibrid inconsistente"
    return X_hybrid.values

# ========= ORQUESTRADOR =========
class ClosedLoopDualOutputs:
    def __init__(self):
        self.chronos_models = {}
        self.chronos_timelines = {}
        self.chronos_series_sim = {}

        self.fno = FNOWrapper()
        self.timeline_fno = pd.DataFrame(index=df_output.index, columns=df_output.columns, dtype=float)
        self.timeline_fno.iloc[:] = np.nan

        # series simuladas que alimentam os modelos
        self.series_sim_fno = df_output.copy()
        if len(df_output) > MIN_TRAIN_SAMPLES:
            self.series_sim_fno.iloc[MIN_TRAIN_SAMPLES:] = np.nan

        # janelas de previsão
        self.pred_windows = []  # lista de tuplas (start_idx, end_idx)

        # mapa de cores por modelo
        self.model_colors = None

    def initialize_models(self):
        if chronos_available:
            for v in versions:
                m = ChronosWrapper(v)
                m.load_model()
                if m.loaded:
                    self.chronos_models[v] = m
                    tl = pd.DataFrame(index=df_output.index, columns=df_output.columns, dtype=float)
                    tl.iloc[:] = np.nan
                    ss = df_output.copy()
                    if len(df_output) > MIN_TRAIN_SAMPLES:
                        ss.iloc[MIN_TRAIN_SAMPLES:] = np.nan
                    self.chronos_timelines[v] = tl
                    self.chronos_series_sim[v] = ss
        logger.info(f"Chronos carregados: {len(self.chronos_models)}  FNO: 1")
        self.model_colors = _build_model_colors()

    def _make_grid(self, n_plots, ncols=3):
        nrows = int(math.ceil(n_plots / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = np.array([axes])
        for k in range(n_plots, nrows*ncols):
            axes[k].set_visible(False)
        return fig, axes

    # ======== ESTILO DOS PLOTS ========
    def _style_axes(self, ax, title=None, show_legend=False):
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(frameon=False)

    def _plot_real_segmented(self, ax, series_real: pd.Series):
        real_masked = series_real.copy()
        for s, e in self.pred_windows:
            real_masked.iloc[s:e] = np.nan
        ax.plot(series_real.index, real_masked.values, label="Real", linestyle="-", linewidth=1.2, alpha=0.8)

    def _plot_predicted_segmented(self, ax, index, series_pred: pd.Series, label: str, color=None):
        for s, e in self.pred_windows:
            seg = series_pred.iloc[s:e]
            mask = ~pd.isna(seg)
            if mask.any():
                ax.plot(index[s:e][mask], seg[mask].values,
                        linestyle="--", linewidth=1.6, alpha=0.95, label=label, color=color)

    # ========= PREVIOUS-DAY VALIDATION HELPERS =========
    def _infer_day_length_at(self, pos: int) -> int:
        """Retorna quantos timestamps existem na mesma data calendario de idx[pos]."""
        idx = df_output.index
        the_date = idx[pos].date()
        return int(np.sum(idx.date == the_date))

    def _prev_day_slice(self, s: int, e: int):
        """
        Mapeia [s, e) para o mesmo intervalo de relogio do dia anterior.
        Retorna (s_prev, e_prev) ou (None, None) se nao encontrado.
        """
        idx = df_output.index
        if s <= 0:
            return None, None

        # tentativa por deslocamento temporal de 1 dia
        try:
            start_ts_prev = idx[s] - pd.Timedelta(days=1)
            end_ts_prev   = idx[e-1] - pd.Timedelta(days=1)
            s_prev = idx.get_indexer([start_ts_prev])[0]
            e_prev_last = idx.get_indexer([end_ts_prev])[0]
            if s_prev != -1 and e_prev_last != -1:
                return s_prev, e_prev_last + 1
        except Exception:
            pass

        # fallback por comprimento real do dia de inicio
        day_len = self._infer_day_length_at(s)
        s_prev2 = s - day_len
        e_prev2 = e - day_len
        if s_prev2 >= 0 and e_prev2 > s_prev2:
            return s_prev2, e_prev2
        return None, None

    def _compute_best_table_prevday_for_variable(self, col: str, models_map: dict) -> pd.DataFrame:
        """
        Vencedor por menor MAE no mesmo intervalo do dia anterior.
        Fallback para regra antiga apenas se o prev-day nao estiver disponivel.
        """
        idx = df_output.index
        rows = []

        for w, (s, e) in enumerate(self.pred_windows):
            next_idx = e if e < len(df_output) else None
            next_real_val = np.nan
            next_real_time = pd.NaT
            if next_idx is not None:
                next_real_val = df_output.iloc[next_idx][col]
                next_real_time = idx[next_idx]

            s_prev, e_prev = self._prev_day_slice(s, e)

            winner = ""
            best_score = np.inf
            used_prevday = False

            if s_prev is not None and e_prev is not None and e_prev > s_prev:
                y_true = df_output[col].iloc[s_prev:e_prev].astype(float).to_numpy()
                if np.isfinite(y_true).any():
                    for name, tl in models_map.items():
                        y_pred = tl[col].iloc[s_prev:e_prev].astype(float).to_numpy()
                        mask = np.isfinite(y_true) & np.isfinite(y_pred)
                        if mask.any():
                            mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
                            if mae < best_score:
                                best_score = mae
                                winner = name
                                used_prevday = True

            if not used_prevday and next_idx is not None and pd.notna(next_real_val):
                last_t = e - 1
                for name, tl in models_map.items():
                    v = tl.iloc[last_t][col]
                    if pd.isna(v):
                        continue
                    err = abs(float(v) - float(next_real_val))
                    if err < best_score:
                        best_score = err
                        winner = name

            rows.append({
                "window_idx": w,
                "start": idx[s],
                "end": idx[e-1] if e > s else idx[s],
                "next_real_time": next_real_time,
                "next_real_value": float(next_real_val) if pd.notna(next_real_val) else np.nan,
                "winner": winner,
                "winner_score": float(best_score) if np.isfinite(best_score) else np.nan,
                "criterion": "prevday_mae" if used_prevday else "fallback_next_real"
            })
        return pd.DataFrame(rows)

    # ========= REAL OVERRIDES PARA EVITAR DATA LEAK =========
    def _apply_real_overrides_upto(self, current: int):
        """
        Garante que, até current, toda serie_sim use o valor real quando existir.
        Vale para FNO e para cada Chronos. Nao toca a janela de previsao atual.
        """
        if current <= 0:
            return
        real_slice = df_output.iloc[:current]
        mask = ~real_slice.isna()

        # FNO
        self.series_sim_fno.iloc[:current] = np.where(
            mask, real_slice, self.series_sim_fno.iloc[:current]
        )

        # Chronos
        for name in self.chronos_series_sim.keys():
            sim = self.chronos_series_sim[name]
            sim.iloc[:current] = np.where(mask, real_slice, sim.iloc[:current])

    # ========= PLOTS PADRAO =========
    def plot_final_per_chronos(self):
        plt.rcParams.update({"lines.linewidth": 1.3, "lines.markersize": 0})
        for name, tl in self.chronos_timelines.items():
            n_vars = len(df_output.columns)
            fig_c, axes_c = self._make_grid(n_vars, ncols=3)
            for i, col in enumerate(df_output.columns):
                ax = axes_c[i]
                self._plot_real_segmented(ax, df_output[col])
                self._plot_predicted_segmented(ax, df_output.index, tl[col],
                                               label=f"{name} (prev)",
                                               color=self.model_colors.get(name, None))
                self._style_axes(ax, title=f"{col}  {name}", show_legend=(i == 0))
            fig_c.suptitle(f"Closed-loop Chronos  {name}")
            fig_c.tight_layout()
            out_png = os.path.join(OUTPUT_DIR, f"chronos_{name.replace('/', '_')}_timeline.png")
            fig_c.savefig(out_png, dpi=220, bbox_inches="tight")
            plt.close(fig_c)

    def plot_final_fno(self):
        plt.rcParams.update({"lines.linewidth": 1.3, "lines.markersize": 0})
        n_vars = len(df_output.columns)
        fig_f, axes_f = self._make_grid(n_vars, ncols=3)
        for i, col in enumerate(df_output.columns):
            ax = axes_f[i]
            self._plot_real_segmented(ax, df_output[col])
            self._plot_predicted_segmented(ax, df_output.index, self.timeline_fno[col],
                                           label="FNO (prev)",
                                           color=self.model_colors.get("FNO", None))
            self._style_axes(ax, title=f"{col}  FNO", show_legend=(i == 0))
        fig_f.suptitle("Closed-loop FNO")
        fig_f.tight_layout()
        fig_f.savefig(os.path.join(OUTPUT_DIR, "fno_timeline.png"), dpi=220, bbox_inches="tight")
        plt.close(fig_f)

    # ========= COLETA DE TIMELINES =========
    def _collect_all_models_timelines(self):
        models_map = {}
        for name, tl in self.chronos_timelines.items():
            models_map[name] = tl
        models_map["FNO"] = self.timeline_fno
        return models_map

    # ========= VENCEDOR POR JANELA PARA CADA VARIAVEL (prev-day rule) =========
    def plot_final_best_by_window_per_variable(self):
        models_map = self._collect_all_models_timelines()
        idx = df_output.index

        def real_masked(series_real: pd.Series):
            masked = series_real.copy()
            for s, e in self.pred_windows:
                masked.iloc[s:e] = np.nan
            return masked

        for col in df_output.columns:
            best_df = self._compute_best_table_prevday_for_variable(col, models_map)
            out_csv = os.path.join(OUTPUT_DIR, f"best_by_window_{col}.csv")
            best_df.to_csv(out_csv, index=False)

            # Figura por variavel
            fig, ax = plt.subplots(1, 1, figsize=(12, 4.2))
            ax.plot(df_output.index, real_masked(df_output[col]).values,
                    label="Real", linestyle="-", linewidth=1.2, alpha=0.85)

            for _, row in best_df.iterrows():
                s_time = row["start"]; e_time = row["end"]
                s = df_output.index.get_indexer([pd.to_datetime(s_time)])[0]
                e = df_output.index.get_indexer([pd.to_datetime(e_time)])[0] + 1
                if e <= s:
                    continue
                winner = row["winner"]
                if not winner:
                    continue
                tl = models_map.get(winner, None)
                if tl is None:
                    continue
                seg = tl[col].iloc[s:e]
                mask = ~pd.isna(seg)
                color = self.model_colors.get(winner, None)
                label_once = f"{winner} (win, {row['criterion']})" if row["window_idx"] == 0 else None
                if mask.any():
                    ax.plot(df_output.index[s:e][mask],
                            seg[mask].values,
                            linestyle="--",
                            linewidth=1.6,
                            alpha=0.95,
                            color=color,
                            label=label_once)

            ax.set_title(f"{col}  Best model per window (prevday validation)")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            out_png = os.path.join(OUTPUT_DIR, f"best_by_window_{col}.png")
            fig.savefig(out_png, dpi=220, bbox_inches="tight")
            plt.close(fig)

            # Serie composta
            composite = pd.Series(index=df_output.index, dtype=float)
            composite.loc[:] = np.nan
            for _, row in best_df.iterrows():
                s_time = row["start"]; e_time = row["end"]
                s = df_output.index.get_indexer([pd.to_datetime(s_time)])[0]
                e = df_output.index.get_indexer([pd.to_datetime(e_time)])[0] + 1
                if e <= s:
                    continue
                winner = row["winner"]
                if not winner:
                    continue
                tl = models_map.get(winner, None)
                if tl is None:
                    continue
                seg = tl[col].iloc[s:e]
                composite.iloc[s:e] = seg.values

            composite_df = pd.DataFrame({col: composite})
            composite_csv = os.path.join(OUTPUT_DIR, f"composite_{col}.csv")
            composite_df.to_csv(composite_csv)

            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4.2))
            masked_real = df_output[col].copy()
            for s, e in self.pred_windows:
                masked_real.iloc[s:e] = np.nan
            ax2.plot(df_output.index, masked_real.values, label="Real", linestyle="-", linewidth=1.2, alpha=0.85)
            first = True
            for _, row in best_df.iterrows():
                s_time = row["start"]; e_time = row["end"]
                s = df_output.index.get_indexer([pd.to_datetime(s_time)])[0]
                e = df_output.index.get_indexer([pd.to_datetime(e_time)])[0] + 1
                if e <= s:
                    continue
                seg = composite.iloc[s:e]
                mask = ~pd.isna(seg)
                if mask.any():
                    ax2.plot(df_output.index[s:e][mask],
                             seg[mask].values,
                             linestyle="--",
                             linewidth=1.6,
                             alpha=0.95,
                             label="Composite" if first else None)
                    first = False

            ax2.set_title(f"{col}  Composite of window winners (prevday validation)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(frameon=False, loc="best")
            fig2.tight_layout()
            composite_png = os.path.join(OUTPUT_DIR, f"composite_{col}.png")
            fig2.savefig(composite_png, dpi=220, bbox_inches="tight")
            plt.close(fig2)

            logger.info(f"[{col}] best table:  {out_csv}")
            logger.info(f"[{col}] best plot:   {out_png}")
            logger.info(f"[{col}] composite:  {composite_csv}")
            logger.info(f"[{col}] composite plot: {composite_png}")

    # ========= DASHBOARD FINAL: ALL VARIAVEIS + CSV DA FIGURA =========
    def plot_final_dashboard_all_variables(self):
        """
        Gera:
          - best_by_window_ALL.png
          - best_by_window_matrix.csv  vencedores por faixa x variavel
          - best_by_window_ALL_series.csv  dados da figura ALL
        """
        models_map = self._collect_all_models_timelines()

        winners_matrix = {}
        best_tables_cache = {}
        for col in df_output.columns:
            best_df = self._compute_best_table_prevday_for_variable(col, models_map)
            best_tables_cache[col] = best_df
            winners_matrix[col] = best_df.set_index("window_idx")["winner"]

        winners_df = pd.DataFrame(winners_matrix)
        winners_csv = os.path.join(OUTPUT_DIR, "best_by_window_matrix.csv")
        winners_df.to_csv(winners_csv)

        rows_series = []
        for col in df_output.columns:
            best_df = best_tables_cache[col]
            for _, row in best_df.iterrows():
                w_idx = int(row["window_idx"])
                s_time = row["start"]; e_time = row["end"]
                s = df_output.index.get_indexer([pd.to_datetime(s_time)])[0]
                e = df_output.index.get_indexer([pd.to_datetime(e_time)])[0] + 1
                if e <= s:
                    continue
                winner = row["winner"]
                if not winner:
                    continue
                tl = models_map.get(winner, None)
                if tl is None:
                    continue
                seg = tl[col].iloc[s:e]
                mask = ~pd.isna(seg)
                if mask.any():
                    ts = df_output.index[s:e][mask]
                    vals = seg[mask].values
                    for tstamp, val in zip(ts, vals):
                        rows_series.append({
                            "variable": col,
                            "window_idx": w_idx,
                            "model": winner,
                            "timestamp": tstamp,
                            "value": float(val),
                            "criterion": row["criterion"]
                        })
        series_df = pd.DataFrame(rows_series)
        series_csv = os.path.join(OUTPUT_DIR, "best_by_window_ALL_series.csv")
        series_df.to_csv(series_csv, index=False)

        n_vars = len(df_output.columns)
        ncols = 3
        nrows = int(math.ceil(n_vars / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

        def real_masked(series_real: pd.Series):
            masked = series_real.copy()
            for s, e in self.pred_windows:
                masked.iloc[s:e] = np.nan
            return masked

        for i, col in enumerate(df_output.columns):
            ax = axes[i]
            ax.plot(df_output.index, real_masked(df_output[col]).values,
                    label="Real", linestyle="-", linewidth=1.2, alpha=0.85, color="black")

            series_winners = winners_df[col] if col in winners_df.columns else pd.Series(dtype=str)
            for w, (s, e) in enumerate(self.pred_windows):
                if e <= s:
                    continue
                winner = series_winners.get(w, "")
                if not winner:
                    continue
                tl = models_map.get(winner, None)
                if tl is None:
                    continue
                seg = tl[col].iloc[s:e]
                mask = ~pd.isna(seg)
                if mask.any():
                    ax.plot(df_output.index[s:e][mask],
                            seg[mask].values,
                            linestyle="--",
                            linewidth=1.6,
                            alpha=0.95,
                            color=self.model_colors.get(winner, None))

            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        for k in range(n_vars, len(axes)):
            axes[k].set_visible(False)

        handles = []
        labels = []
        available_models = list(self.chronos_timelines.keys()) + ["FNO"]
        ordered = [m for m in ALL_MODELS_ORDER if m in available_models]
        for name in ordered:
            color = self.model_colors.get(name, None)
            h = Line2D([0], [0], linestyle="--", linewidth=2, color=color)
            handles.append(h)
            labels.append(name)

        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("Best model per prediction window (prevday validation) - all variables", y=1.06)
        fig.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, "best_by_window_ALL.png")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Dashboard salvo: {out_png}")
        logger.info(f"Matriz de vencedores: {winners_csv}")
        logger.info(f"Serie da figura ALL: {series_csv}")

    # ======== BLINDAGENS ========
    def _assert_timelines_nan(self, start, end):
        for name, tl in self.chronos_timelines.items():
            assert tl.iloc[start:end].isna().all().all(), f"Timeline Chronos {name} contem valores no gap real [{start}:{end}]"
        assert self.timeline_fno.iloc[start:end].isna().all().all(), f"Timeline FNO contem valores no gap real [{start}:{end}]"

    def _assert_no_future_in_train(self, current, pred_start, pred_end):
        assert self.series_sim_fno.iloc[current:].isna().all().all(), \
            f"series_sim_fno tem valores apos current={current}"
        for name, ss in self.chronos_series_sim.items():
            assert ss.iloc[current:].isna().all().all(), \
                f"series_sim_chronos[{name}] tem valores apos current={current}"
        assert self.series_sim_fno.iloc[pred_start:pred_end].isna().all().all(), \
            f"series_sim_fno tem valores na janela de previsao [{pred_start}:{pred_end}]"
        for name, ss in self.chronos_series_sim.items():
            assert ss.iloc[pred_start:pred_end].isna().all().all(), \
                f"series_sim_chronos[{name}] tem valores na janela de previsao [{pred_start}:{pred_end}]"
        for name, tl in self.chronos_timelines.items():
            assert tl.iloc[pred_start:pred_end].isna().all().all(), \
                f"timeline_chronos[{name}] tem valores na janela de previsao [{pred_start}:{pred_end}]"
        assert self.timeline_fno.iloc[pred_start:pred_end].isna().all().all(), \
            "timeline_fno tem valores na janela de previsao"

    # ======== EXECUCAO ========
    def run(self):
        total_len = len(df_output)
        if total_len <= MIN_TRAIN_SAMPLES:
            logger.warning("Tamanho total insuficiente")
            return self.chronos_timelines, self.timeline_fno

        start_all = time.time()

        pred_start = MIN_TRAIN_SAMPLES
        block_span = 2 * STEP_TEST

        while pred_start < total_len:
            pred_end = min(pred_start + STEP_TEST, total_len)
            gap_start = pred_end
            gap_end = min(pred_start + block_span, total_len)

            if pred_start >= total_len:
                break
            if pred_start < MIN_TRAIN_SAMPLES:
                pred_start = MIN_TRAIN_SAMPLES

            current = pred_start
            logger.info(f"Treino ate {current}, previsao [{pred_start}:{pred_end}], gap real [{gap_start}:{gap_end}]")

            # registra janela
            self.pred_windows.append((pred_start, pred_end))

            # anti vazamento
            self._assert_no_future_in_train(current, pred_start, pred_end)

            # garantir que todo real disponivel ate current sobrescreve series simuladas
            self._apply_real_overrides_upto(current)

            # montar X hibrido ate current
            X_hybrid = build_hybrid_inputs(
                X_base=df_input,
                y_sim_fno=self.series_sim_fno,
                chronos_sim_dict=self.chronos_series_sim,
                current=current
            )
            Y_train = self.series_sim_fno.iloc[:current].values

            ok = self.fno.train(X_hybrid, Y_train)
            if not ok:
                logger.warning("FNO nao treinou nesta janela")

            # previsao Chronos usando somente series simulada acumulada com override de real
            if len(self.chronos_models) > 0:
                for name, m in self.chronos_models.items():
                    if not m.loaded:
                        continue
                    ctx_sim = self.chronos_series_sim[name].iloc[:current]
                    tl = self.chronos_timelines[name]
                    for col in df_output.columns:
                        try:
                            horizon_local = int(pred_end - pred_start)
                            pred = m.predict(ctx_sim[col], horizon_local)
                            L = int(min(horizon_local, len(pred)))
                            if L > 0:
                                j = df_output.columns.get_loc(col)
                                tl.iloc[pred_start:pred_start+L, j] = pred[:L]
                                self.chronos_series_sim[name].iloc[pred_start:pred_start+L, j] = pred[:L]
                        except Exception as e:
                            logger.error(f"Chronos {name} falhou em {col}: {e}")

            # previsao FNO passo a passo com reforco de override de real a cada t
            if self.fno.trained:
                try:
                    for t in range(pred_start, pred_end):
                        self._apply_real_overrides_upto(t)

                        X_ctx = build_hybrid_inputs(
                            X_base=df_input,
                            y_sim_fno=self.series_sim_fno,
                            chronos_sim_dict=self.chronos_series_sim,
                            current=t
                        )
                        next_pred = self.fno.predict_step(X_ctx)
                        for j, col in enumerate(df_output.columns):
                            self.timeline_fno.iloc[t, j] = next_pred[j]
                            self.series_sim_fno.iloc[t, j] = next_pred[j]
                except Exception as e:
                    logger.error(f"Erro na previsao passo a passo FNO: {e}")

            # gap real
            if gap_start < gap_end:
                for name, tl in self.chronos_timelines.items():
                    tl.iloc[gap_start:gap_end, :] = np.nan
                self.timeline_fno.iloc[gap_start:gap_end, :] = np.nan
                self._assert_timelines_nan(gap_start, gap_end)

                seg_real = df_output.iloc[gap_start:gap_end]
                for name in self.chronos_series_sim.keys():
                    self.chronos_series_sim[name].iloc[gap_start:gap_end, :] = seg_real.values
                self.series_sim_fno.iloc[gap_start:gap_end, :] = seg_real.values

            pred_start += block_span

        total_time = time.time() - start_all
        logger.info(f"Tempo total {total_time:.2f}s")

        # salvar CSVs
        for name, tl in self.chronos_timelines.items():
            tl.to_csv(os.path.join(OUTPUT_DIR, f"chronos_{name.replace('/', '_')}_predictions.csv"))
            self.chronos_series_sim[name].to_csv(os.path.join(OUTPUT_DIR, f"series_simulada_chronos_{name.replace('/', '_')}.csv"))
        self.timeline_fno.to_csv(os.path.join(OUTPUT_DIR, "fno_predictions.csv"))
        self.series_sim_fno.to_csv(os.path.join(OUTPUT_DIR, "series_simulada_fno.csv"))
        with open(os.path.join(OUTPUT_DIR, "pred_windows.json"), "w") as f:
            json.dump(self.pred_windows, f, indent=2)

        # plots
        try:
            self.plot_final_per_chronos()
        except Exception as e:
            logger.error(f"Falha plots Chronos: {e}")
        try:
            self.plot_final_fno()
        except Exception as e:
            logger.error(f"Falha plot FNO: {e}")
        try:
            self.plot_final_best_by_window_per_variable()
        except Exception as e:
            logger.error(f"Falha plot Best-by-window por variavel: {e}")
        try:
            self.plot_final_dashboard_all_variables()
        except Exception as e:
            logger.error(f"Falha dashboard ALL: {e}")

        return self.chronos_timelines, self.timeline_fno

# ========= HELPER PARA RODAR VARIACOES DE STEP_TEST =========
def run_full_pipeline_for(step_test_value: int):
    global STEP_TEST, HORIZON, RUN_ID, RUN_META, OUTPUT_DIR
    STEP_TEST = int(step_test_value)
    HORIZON = STEP_TEST

    assert n_samples > MIN_TRAIN_SAMPLES + 2*STEP_TEST, f"Base insuficiente para padrao intercalado com STEP_TEST={STEP_TEST}"

    RUN_ID, RUN_META = make_run_id()
    output_root = os.environ.get("OUTPUT_ROOT", os.path.join(os.getcwd(), "outputs"))
    OUTPUT_DIR = os.path.join(output_root, RUN_ID)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(RUN_META, f, indent=2)

    logger.info("="*60)
    logger.info(f"ORQUESTRADOR v17-mix+dashboard stride-2 intercalado  STEP_TEST={STEP_TEST}")
    logger.info("="*60)

    orch = ClosedLoopDualOutputs()
    orch.initialize_models()
    preds_chronos_dict, preds_fno = orch.run()

    summary = {
        "output_dir": OUTPUT_DIR,
        "n_samples": int(n_samples),
        "n_inputs": int(df_input.shape[1]),
        "n_outputs": int(df_output.shape[1]),
        "step_test": int(STEP_TEST),
        "horizon": int(HORIZON),
        "min_train_samples": int(MIN_TRAIN_SAMPLES),
        "fno_epochs": int(FNO_EPOCHS),
        "device": DEVICE,
        "chronos_versions_loaded": list(preds_chronos_dict.keys()),
        "pattern": "real -> prev -> real -> prev",
        "winner_rule": "prevday_same_clock_mae_with_fallback",
        "no_dataleak_guarantee": [
            "asserts de janela de previsao vazia",
            "override de real ate current antes de treino e previsao",
            "reforco de override a cada t no passo a passo do FNO"
        ]
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Execucao concluida")
    logger.info(f"Resultados em: {OUTPUT_DIR}")

# ========= MAIN =========
if __name__ == "__main__":
    for _st in [50, 100, 250, 500]:
        run_full_pipeline_for(_st)
