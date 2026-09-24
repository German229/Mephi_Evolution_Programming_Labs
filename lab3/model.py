"""ЛР3: суточная модель водного баланса корнеобитаемого слоя («ведро») и три критерия.

Расписание I (30 суточных норм) задаётся заранее; ET0 известна, осадки — ансамбль из K сценариев.
Состояние θ_t — запас доступной влаги, мм. За сутки t в каждом сценарии:
  θ ← θ + R_t + η·I_t                      (осадки и полив с КПД η)
  θ ← θ − Kc·ET0_t·min(1, θ/θ_low)         (эвапотранспирация, снижается при водном стрессе)
  сток: излишек над θ_sat теряется;  дренаж: доля k_d излишка над полевой влагоёмкостью FC уходит вниз.
Критерии (все минимизируются):
  f1 = Σ I_t                      — расход поливной воды, мм;
  f2 = E_s Σ max(0, θ_low − θ_t)   — дефицит влаги (водный стресс), мм·сут, среднее по сценариям;
  f3 = E_s Σ max(0, θ_t − FC)      — переувлажнение (выше полевой влагоёмкости), мм·сут, среднее по сценариям.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_WEATHER = Path(__file__).parent / "data" / "weather.json"
OBJECTIVES = ["вода, мм", "дефицит, мм·сут", "переувлажнение, мм·сут"]


@dataclass
class Model:
    et0: np.ndarray
    rain: np.ndarray
    fc: float = 100.0         # полевая влагоёмкость слоя (запас доступной влаги), мм
    theta_sat: float = 130.0  # насыщение; выше — поверхностный сток
    theta_low: float = 50.0   # порог водного стресса (легкодоступная влага 50 %)
    theta0: float = 60.0      # начальный запас, мм
    kc: float = 1.0           # коэффициент культуры
    efficiency: float = 0.9   # КПД полива
    drain: float = 0.5        # доля излишка над FC, уходящая дренажом за сутки
    i_max: float = 25.0       # максимальная суточная норма, мм

    @property
    def days(self):
        return len(self.et0)

    @property
    def scenarios(self):
        return self.rain.shape[0]

    def simulate(self, I, trace=False):
        """I: (pop, days) нормы полива. Возвращает F (pop, 3) и, если trace, dict траекторий (pop, K, days)."""
        I = np.atleast_2d(I)[:, None, :]                   # (pop, 1, days) — одно расписание на все сценарии
        shape = (I.shape[0], self.scenarios)
        theta = np.full(shape, self.theta0)
        deficit = np.zeros(shape)
        wet = np.zeros(shape)
        hist = {k: [] for k in ("theta", "et", "runoff", "drainage")} if trace else None
        for t in range(self.days):
            theta = theta + self.rain[:, t] + self.efficiency * I[:, :, t]
            et = self.kc * self.et0[t] * np.minimum(1.0, theta / self.theta_low)
            theta = theta - et
            runoff = np.maximum(0.0, theta - self.theta_sat)
            theta = theta - runoff
            drainage = self.drain * np.maximum(0.0, theta - self.fc)
            theta = np.maximum(0.0, theta - drainage)
            deficit += np.maximum(0.0, self.theta_low - theta)
            wet += np.maximum(0.0, theta - self.fc)
            if trace:
                for k, v in (("theta", theta), ("et", et), ("runoff", runoff), ("drainage", drainage)):
                    hist[k].append(v.copy())
        F = np.column_stack([I[:, 0, :].sum(1), deficit.mean(1), wet.mean(1)])
        if trace:
            return F, {k: np.moveaxis(np.array(v), 0, -1) for k, v in hist.items()}
        return F

    def nadir(self):
        """Худшие значения критериев на крайних стратегиях: нет полива (дефицит) и полив по максимуму."""
        F0 = self.simulate(np.zeros((1, self.days)))[0]
        Fmax = self.simulate(np.full((1, self.days), self.i_max))[0]
        return np.array([Fmax[0], F0[1], Fmax[2]])


def load_model(path=DEFAULT_WEATHER, **params):
    w = json.loads(Path(path).read_text())
    return Model(et0=np.array(w["et0"]), rain=np.atleast_2d(np.array(w["rain"])), **params)
