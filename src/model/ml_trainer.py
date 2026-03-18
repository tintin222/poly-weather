"""ML model trainer: "Is this the peak hour?" — binary classification.

The question is NOT "at what hour will the peak occur?" (regression).
It is "given everything we observe up to hour H, is H the peak hour
for today's temperature?" (binary classification per checkpoint).

This is more useful for Polymarket because at runtime you call the model
at the CURRENT hour and get P(this is the peak).  High probability → the
peak has likely occurred; act accordingly.

Training design
---------------
1. Each date is expanded into one row per PREDICTION_HOUR (every integer
   hour 6–20, 15 checkpoints).  Each row represents the state of
   knowledge AT that hour.
2. Label = 1 if pred_hour == actual_peak_hour, else 0.
   This is a 1-in-15 positive class (~6.7 % positive rate).
3. Class imbalance handled via class_weight='balanced' (LogisticRegression,
   RandomForest) or sample_weight computed from balanced weights (GradientBoosting).
4. Evaluation: AUC-ROC (robust to imbalance), Precision, Recall.
5. Per-date summary: for each test day, which hour did the model assign
   the highest probability to?  Was it correct?

Features (13, down from 18)
---------------------------
Dropped (redundant or causally weak):
  - month_sin / month_cos  → redundant with doy_sin/cos
  - day_of_week            → no causal link to temperature peak timing
  - morning_humidity       → derived from morning_temp + morning_dew (RH formula)
  - obs_count              → redundant with current_hour_sin/cos

Kept (causal or informative):
  doy_sin, doy_cos           — seasonal cycle
  morning_temp               — daytime max correlates with morning temp
  morning_dew                — atmospheric moisture → cloud formation timing
  morning_wind               — mixing affects peak timing
  morning_pressure           — synoptic situation
  rolling_peak_hour_7d       — climatological baseline for this city
  current_hour_sin/cos       — cyclical time-of-day (critical for classification)
  current_max_temp           — how high has it gotten so far
  current_max_hour           — when was the running max (is peak receding?)
  hours_since_obs_peak       — if >2 h since max, likely past the peak
  temp_trend_2h              — rising (+) or falling (−)? direct peak signal

Data quality
------------
  - Days with peak_hour == 0 excluded (scraper artifact: only midnight obs)
  - Minimum 60 valid days required (hard); warns at < 120
  - Skip date if no morning observations (6–9 AM) — features would be meaningless
  - Skip prediction-hour row if no observations up to that hour
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.db.database import Database

log = get_logger(__name__)

# One training row per (date, prediction_hour): 6 AM through 8 PM
PREDICTION_HOURS = list(range(6, 21))   # 6,7,8,...,20 — 15 checkpoints

MIN_VALID_DAYS = 60     # hard minimum; refuse to train below this
WARN_VALID_DAYS = 120   # warn; suggest more backfill below this

FEATURE_NAMES = [
    "doy_sin", "doy_cos",
    "morning_temp", "morning_dew", "morning_wind", "morning_pressure",
    "rolling_peak_hour_7d",
    "current_hour_sin", "current_hour_cos",
    "current_max_temp", "current_max_hour",
    "hours_since_obs_peak", "temp_trend_2h",
]


@dataclass
class MLModelResult:
    model_name: str
    train_auc: float        # AUC-ROC on training set
    val_auc: float          # AUC-ROC on validation set
    test_auc: float         # AUC-ROC on held-out test set
    train_acc: float        # balanced accuracy on training set (informational)
    test_precision: float   # P(correct | model says "this is the peak")
    test_recall: float      # P(caught | actually the peak hour)
    feature_names: list[str]
    feature_importances: list[float] | None
    predictions_vs_actual: list[dict]  # per date: which hour got highest P?
    trained_at: str
    n_train: int   # training DAYS
    n_val: int
    n_test: int
    n_pos_train: int = 0   # positive rows in training (for imbalance reference)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "train_auc": self.train_auc,
            "val_auc": self.val_auc,
            "test_auc": self.test_auc,
            "train_acc": self.train_acc,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "predictions_vs_actual": self.predictions_vs_actual,
            "trained_at": self.trained_at,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_pos_train": self.n_pos_train,
        }


class MLTrainer:
    """Train classifiers to answer: 'Is this hour the peak hour?'"""

    def __init__(self, db: "Database") -> None:
        self._db = db
        self._cache: dict[str, list[MLModelResult]] = {}
        # Fitted sklearn model objects: city_key → {model_name → sklearn_model}
        self._models: dict[str, dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def _build_dataset(
        self, city_key: str
    ) -> tuple[list[dict], np.ndarray, np.ndarray, dict[str, int]] | None:
        """Build (metadata_list, X, y, peak_by_date).

        Filters out days where peak_hour == 0 (data artifacts).
        Each remaining day is expanded into up to 15 rows (one per PREDICTION_HOUR).
        metadata_list entries: {date, pred_hour}.
        peak_by_date: {date -> actual_peak_hour} for the full dataset.
        """
        import datetime as _dt

        all_peaks = self._db.conn.execute(
            "SELECT date, peak_hour FROM daily_peaks WHERE city_key = ? ORDER BY date",
            (city_key,),
        ).fetchall()

        # --- Filter artifact rows (peak at midnight = scraper captured no daytime data) ---
        peaks_rows = [r for r in all_peaks if r["peak_hour"] > 0]
        n_filtered = len(all_peaks) - len(peaks_rows)

        log.info(
            "ml_build_dataset",
            city=city_key,
            total_peaks=len(all_peaks),
            valid_peaks=len(peaks_rows),
            filtered_zero=n_filtered,
        )

        if len(peaks_rows) < MIN_VALID_DAYS:
            log.warning(
                "ml_insufficient_valid_peaks",
                city=city_key,
                valid=len(peaks_rows),
                minimum=MIN_VALID_DAYS,
            )
            return None

        if len(peaks_rows) < WARN_VALID_DAYS:
            log.warning(
                "ml_low_data_quality",
                city=city_key,
                valid=len(peaks_rows),
                recommend=WARN_VALID_DAYS,
                note="Results may be unreliable. Run a longer backfill.",
            )

        peak_by_date = {r["date"]: r["peak_hour"] for r in peaks_rows}

        obs_rows = self._db.conn.execute(
            """SELECT date, hour, temperature, dew_point, wind_speed, pressure
               FROM hourly_observations WHERE city_key = ? ORDER BY date, hour""",
            (city_key,),
        ).fetchall()

        obs_by_date: dict[str, list] = {}
        for row in obs_rows:
            obs_by_date.setdefault(row["date"], []).append(row)

        # Sequence of valid peak hours (for rolling average)
        peak_hours_seq = [r["peak_hour"] for r in peaks_rows]

        rows_meta: list[dict] = []
        X_rows: list[list[float]] = []
        y_vals: list[float] = []

        def _avg(col: str, obs_list: list) -> float:
            vals = [o[col] for o in obs_list if o[col] is not None]
            return float(np.mean(vals)) if vals else 0.0

        for i, peak_row in enumerate(peaks_rows):
            date = peak_row["date"]
            target_peak_hour = peak_row["peak_hour"]

            if date not in obs_by_date:
                continue

            day_obs = obs_by_date[date]

            # --- Date features ---
            try:
                year, month, day = int(date[:4]), int(date[5:7]), int(date[8:10])
                dt = _dt.date(year, month, day)
                day_of_year = dt.timetuple().tm_yday
            except (ValueError, IndexError):
                continue

            doy_sin = math.sin(2 * math.pi * day_of_year / 365.0)
            doy_cos = math.cos(2 * math.pi * day_of_year / 365.0)

            # --- Morning features (6–9 AM) — skip day if none exist ---
            morning = [o for o in day_obs if 6 <= o["hour"] <= 9]
            if not morning:
                continue   # no morning baseline → features meaningless

            morning_temp = _avg("temperature", morning)
            morning_dew  = _avg("dew_point", morning)
            morning_wind = _avg("wind_speed", morning)
            morning_pres = _avg("pressure", morning)

            # --- Rolling 7-day avg peak_hour (climatological baseline) ---
            past = peak_hours_seq[max(0, i - 7):i]
            rolling_peak_hour = float(np.mean(past)) if past else 14.0

            # --- Per prediction-hour rows ---
            for pred_hour in PREDICTION_HOURS:
                obs_so_far = [o for o in day_obs if o["hour"] <= pred_hour]
                if not obs_so_far:
                    continue

                # Current max temperature & when it occurred
                max_obs = max(obs_so_far, key=lambda o: o["temperature"])
                current_max_temp = float(max_obs["temperature"])
                current_max_hour = int(max_obs["hour"])
                hours_since_obs_peak = float(pred_hour - current_max_hour)

                # Temperature at pred_hour (nearest observation)
                nearest = min(obs_so_far, key=lambda o: abs(o["hour"] - pred_hour))
                current_temp = float(nearest["temperature"])

                # 2-hour trend: rising or falling?
                obs_2h = [o for o in obs_so_far if abs(o["hour"] - (pred_hour - 2)) <= 1]
                if obs_2h:
                    temp_2h = float(min(obs_2h, key=lambda o: abs(o["hour"] - (pred_hour - 2)))["temperature"])
                    trend = current_temp - temp_2h
                else:
                    trend = 0.0

                # Cyclical hour encoding
                hour_sin = math.sin(2 * math.pi * pred_hour / 24.0)
                hour_cos = math.cos(2 * math.pi * pred_hour / 24.0)

                features = [
                    doy_sin, doy_cos,
                    morning_temp, morning_dew, morning_wind, morning_pres,
                    rolling_peak_hour,
                    hour_sin, hour_cos,
                    current_max_temp, float(current_max_hour),
                    hours_since_obs_peak, trend,
                ]

                rows_meta.append({"date": date, "pred_hour": pred_hour})
                X_rows.append(features)
                # Binary label: 1 = this IS the peak hour, 0 = not the peak
                y_vals.append(1.0 if pred_hour == target_peak_hour else 0.0)

        min_rows = MIN_VALID_DAYS * 3   # e.g. 180 rows for 60-day minimum
        if len(rows_meta) < min_rows:
            log.warning(
                "ml_insufficient_rows",
                city=city_key,
                rows=len(rows_meta),
                minimum=min_rows,
                hint="Most dates likely have sparse hourly observations.",
            )
            return None

        log.info(
            "ml_dataset_built",
            city=city_key,
            rows=len(rows_meta),
            positive_rate=f"{100*sum(y_vals)/len(y_vals):.1f}%",
            unique_dates=len(set(m["date"] for m in rows_meta)),
        )
        return rows_meta, np.array(X_rows), np.array(y_vals), peak_by_date

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, city_key: str) -> list[MLModelResult]:
        """Train binary classifiers. Results cached in memory."""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.utils.class_weight import compute_sample_weight

        dataset = self._build_dataset(city_key)
        if dataset is None:
            return []

        rows_meta, X, y, peak_by_date = dataset
        n = len(rows_meta)

        # --- Time-based split by DATE to prevent leakage ---
        all_dates = [m["date"] for m in rows_meta]
        unique_dates = sorted(set(all_dates))
        n_dates = len(unique_dates)

        n_train_d = max(int(n_dates * 0.70), 14)
        n_val_d   = max(int(n_dates * 0.15),  3)

        train_dates = set(unique_dates[:n_train_d])
        val_dates   = set(unique_dates[n_train_d : n_train_d + n_val_d])
        test_dates  = set(unique_dates[n_train_d + n_val_d :])

        if len(test_dates) < 5:
            log.warning(
                "ml_insufficient_test_dates",
                city=city_key,
                test_dates=len(test_dates),
            )
            return []

        train_mask = np.array([all_dates[i] in train_dates for i in range(n)])
        val_mask   = np.array([all_dates[i] in val_dates   for i in range(n)])
        test_mask  = np.array([all_dates[i] in test_dates  for i in range(n)])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        meta_test = [m for i, m in enumerate(rows_meta) if test_mask[i]]

        # Sanity: need positive examples in test set for AUC
        if y_test.sum() < 2:
            log.warning("ml_no_positives_in_test", city=city_key)
            return []

        log.info(
            "ml_training_start",
            city=city_key,
            n_rows=n,
            n_dates=n_dates,
            n_train_dates=n_train_d,
            n_val_dates=n_val_d,
            n_test_dates=len(test_dates),
            n_train_rows=int(train_mask.sum()),
            n_val_rows=int(val_mask.sum()),
            n_test_rows=int(test_mask.sum()),
            pos_rate_train=f"{100*y_train.mean():.1f}%",
        )

        models_to_train = [
            ("LogisticRegression",
             LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1),
             False),
            ("RandomForest",
             RandomForestClassifier(
                 class_weight="balanced", n_estimators=100,
                 max_depth=6, min_samples_leaf=5, random_state=42),
             True),
            ("GradientBoosting",
             GradientBoostingClassifier(
                 n_estimators=100, max_depth=3, learning_rate=0.05,
                 subsample=0.8, random_state=42),
             False),
        ]

        trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        results: list[MLModelResult] = []

        for model_name, model, has_importances in models_to_train:
            try:
                # GradientBoosting doesn't support class_weight; use sample_weight
                if model_name == "GradientBoosting":
                    sw = compute_sample_weight("balanced", y_train)
                    model.fit(X_train, y_train, sample_weight=sw)
                else:
                    model.fit(X_train, y_train)

                # --- Predicted probabilities ---
                proba_train = model.predict_proba(X_train)[:, 1]
                proba_val   = model.predict_proba(X_val)[:, 1]
                proba_test  = model.predict_proba(X_test)[:, 1]

                # --- AUC-ROC (main metric — robust to imbalance) ---
                if len(np.unique(y_val)) < 2:
                    val_auc = float("nan")
                else:
                    val_auc = round(float(roc_auc_score(y_val, proba_val)), 4)

                train_auc = round(float(roc_auc_score(y_train, proba_train)), 4)
                test_auc  = round(float(roc_auc_score(y_test,  proba_test)),  4)

                # --- Precision / Recall at 0.5 threshold on balanced test ---
                pred_labels = (proba_test >= 0.5).astype(int)
                train_acc      = round(float(accuracy_score(y_train, (proba_train >= 0.5).astype(int))), 4)
                test_precision = round(float(precision_score(y_test, pred_labels, zero_division=0)), 4)
                test_recall    = round(float(recall_score(y_test,    pred_labels, zero_division=0)), 4)

                importances = None
                if has_importances and hasattr(model, "feature_importances_"):
                    importances = [round(float(v), 4) for v in model.feature_importances_]

                # --- Per-date summary: which hour got the highest probability? ---
                # Group test rows by date, then pick the hour with max proba
                date_to_hours: dict[str, list[tuple[int, float]]] = defaultdict(list)
                for m, prob in zip(meta_test, proba_test.tolist()):
                    date_to_hours[m["date"]].append((m["pred_hour"], prob))

                preds_vs_actual = []
                for date in sorted(date_to_hours.keys()):
                    actual = peak_by_date.get(date)
                    if actual is None:
                        continue
                    best_hour, best_prob = max(date_to_hours[date], key=lambda x: x[1])
                    preds_vs_actual.append({
                        "date": date,
                        "actual": actual,
                        "predicted": best_hour,
                        "confidence": round(float(best_prob), 3),
                        "correct": best_hour == actual,
                    })

                result = MLModelResult(
                    model_name=model_name,
                    train_auc=train_auc,
                    val_auc=val_auc,
                    test_auc=test_auc,
                    train_acc=train_acc,
                    test_precision=test_precision,
                    test_recall=test_recall,
                    feature_names=FEATURE_NAMES,
                    feature_importances=importances,
                    predictions_vs_actual=preds_vs_actual,
                    trained_at=trained_at,
                    n_train=n_train_d,
                    n_val=n_val_d,
                    n_test=len(test_dates),
                    n_pos_train=int(y_train.sum()),
                )
                results.append(result)
                # Cache the fitted sklearn object for runtime inference
                self._models.setdefault(city_key, {})[model_name] = model

                log.info(
                    "ml_model_trained",
                    city=city_key,
                    model=model_name,
                    train_auc=train_auc,
                    val_auc=val_auc,
                    test_auc=test_auc,
                    test_precision=test_precision,
                    test_recall=test_recall,
                    per_date_accuracy=f"{100*sum(p['correct'] for p in preds_vs_actual)/max(len(preds_vs_actual),1):.0f}%",
                )

            except Exception as exc:
                log.error("ml_model_failed", city=city_key, model=model_name, error=str(exc))

        self._cache[city_key] = results
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def pickle_model(self, city_key: str, model_name: str) -> bytes | None:
        """Return pickle bytes for a trained model (for DB storage)."""
        import pickle
        m = self._models.get(city_key, {}).get(model_name)
        if m is None:
            return None
        return pickle.dumps(m)

    def load_model_from_blob(self, city_key: str, model_name: str, blob: bytes) -> None:
        """Unpickle a model blob and store it in the in-memory cache."""
        import pickle
        model = pickle.loads(blob)
        self._models.setdefault(city_key, {})[model_name] = model

    # ------------------------------------------------------------------
    # Runtime inference
    # ------------------------------------------------------------------

    def predict_is_peak(
        self,
        city_key: str,
        current_hour: int,
        today_obs: list,
        rolling_peak_hour: float = 14.0,
        day_of_year: int = 180,
    ) -> float | None:
        """Return P(current_hour is today's peak hour) using the best cached model.

        Returns None if no model is loaded for the city.

        Args:
            city_key:           City identifier.
            current_hour:       Current hour (0-23).
            today_obs:          List of hourly observation dicts for today so far
                                (each must have: hour, temperature, dew_point,
                                 wind_speed, pressure).
            rolling_peak_hour:  7-day rolling avg peak hour (default 14 if unknown).
            day_of_year:        Day of year (1-365).
        """
        # Ensure model is available in memory
        city_models = self._models.get(city_key)
        if not city_models:
            # Try loading from DB
            from src.db.repositories import ModelResultRepository
            repo = ModelResultRepository(self._db)
            blob_result = repo.get_best_model_blob(city_key)
            if blob_result is None:
                return None
            model_name, blob = blob_result
            self.load_model_from_blob(city_key, model_name, blob)
            city_models = self._models[city_key]

        # Pick best model by test_auc from cached results
        cached = self._cache.get(city_key)
        best_model_name = None
        if cached:
            best = max(cached, key=lambda r: r.test_auc)
            best_model_name = best.model_name
        if best_model_name is None or best_model_name not in city_models:
            best_model_name = next(iter(city_models))

        model = city_models[best_model_name]

        # Build feature vector (same as training, but for a single row)
        obs_so_far = [o for o in today_obs if o.get("hour", 999) <= current_hour]
        if not obs_so_far:
            return None

        morning = [o for o in today_obs if 6 <= o.get("hour", 0) <= 9]
        if not morning:
            morning = obs_so_far[:2]

        def _avg(col: str, obs_list: list) -> float:
            vals = [o[col] for o in obs_list if o.get(col) is not None]
            return float(np.mean(vals)) if vals else 0.0

        morning_temp = _avg("temperature", morning)
        morning_dew  = _avg("dew_point", morning)
        morning_wind = _avg("wind_speed", morning)
        morning_pres = _avg("pressure", morning)

        doy_sin = math.sin(2 * math.pi * day_of_year / 365.0)
        doy_cos = math.cos(2 * math.pi * day_of_year / 365.0)

        max_obs = max(obs_so_far, key=lambda o: o.get("temperature", -999))
        current_max_temp = float(max_obs.get("temperature", 0))
        current_max_hour = int(max_obs.get("hour", current_hour))
        hours_since_obs_peak = float(current_hour - current_max_hour)

        nearest = min(obs_so_far, key=lambda o: abs(o.get("hour", 0) - current_hour))
        current_temp = float(nearest.get("temperature", current_max_temp))

        obs_2h = [o for o in obs_so_far if abs(o.get("hour", 0) - (current_hour - 2)) <= 1]
        if obs_2h:
            temp_2h = float(min(obs_2h, key=lambda o: abs(o.get("hour", 0) - (current_hour - 2))).get("temperature", current_temp))
            trend = current_temp - temp_2h
        else:
            trend = 0.0

        hour_sin = math.sin(2 * math.pi * current_hour / 24.0)
        hour_cos = math.cos(2 * math.pi * current_hour / 24.0)

        X = np.array([[
            doy_sin, doy_cos,
            morning_temp, morning_dew, morning_wind, morning_pres,
            rolling_peak_hour,
            hour_sin, hour_cos,
            current_max_temp, float(current_max_hour),
            hours_since_obs_peak, trend,
        ]])

        try:
            proba = model.predict_proba(X)[0, 1]
            return float(proba)
        except Exception:
            return None

    def get_cached(self, city_key: str) -> list[MLModelResult] | None:
        return self._cache.get(city_key)
