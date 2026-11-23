"""Request and response schemas (Pydantic v2)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class StressFeatures(BaseModel):
    """Phase-level features. Supply every feature of each modality you want used."""
    # ECG (30-s strip)
    ecg_hr: Optional[float] = Field(None, description="bpm, from R-peaks", examples=[68.0])
    ecg_rmssd: Optional[float] = Field(None, description="ms", examples=[45.0])
    ecg_sdnn: Optional[float] = Field(None, description="ms", examples=[60.0])
    ecg_pnn50: Optional[float] = Field(None, description="%", examples=[25.0])
    ecg_hr_range: Optional[float] = Field(None, description="bpm, max-min instantaneous HR", examples=[20.0])
    # EDA (Fitbit EDA scan)
    eda_scl_mean: Optional[float] = Field(None, examples=[40.0])
    eda_scl_std: Optional[float] = Field(None, examples=[1.0])
    eda_scl_slope: Optional[float] = Field(None, description="units per minute", examples=[-0.3])
    eda_scl_range: Optional[float] = Field(None, examples=[2.5])
    eda_resp_rate: Optional[float] = Field(None, description="responses per minute", examples=[8.0])
    # Fitbit heart rate (8-min window)
    fb_hr_mean: Optional[float] = Field(None, description="bpm", examples=[73.0])
    fb_hr_std: Optional[float] = Field(None, description="bpm", examples=[5.0])
    fb_hr_p05: Optional[float] = Field(None, description="bpm", examples=[65.0])
    fb_hr_p95: Optional[float] = Field(None, description="bpm", examples=[83.0])
    fb_hr_slope: Optional[float] = Field(None, description="bpm per minute", examples=[-0.5])


class StressFeatureRequest(BaseModel):
    features: StressFeatures
    explain: bool = True


class ECGSignal(BaseModel):
    samples: list[float] = Field(..., min_length=2500, description="single-lead ECG samples (>= 10 s)")
    sampling_rate: float = Field(250.0, gt=50, le=2000, description="Hz; the training data were 250 Hz")


class EDASignal(BaseModel):
    scl: list[float] = Field(..., min_length=10, description="skin conductance level, one value per second")
    activation: list[float] = Field(..., min_length=10, description="0/1 EDA response flags, same length")
    valid: Optional[list[bool]] = None
    sampling_rate: float = Field(1.0, gt=0, le=64)

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.activation) != len(self.scl) or (self.valid is not None and len(self.valid) != len(self.scl)):
            raise ValueError("scl, activation (and valid) must have the same length")
        return self


class HeartRateSignal(BaseModel):
    bpm: list[float] = Field(..., min_length=5)
    timestamps_s: list[float] = Field(..., min_length=5, description="seconds, same length as bpm")

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.bpm) != len(self.timestamps_s):
            raise ValueError("bpm and timestamps_s must have the same length")
        return self


class StressRawRequest(BaseModel):
    ecg: Optional[ECGSignal] = None
    eda: Optional[EDASignal] = None
    heart_rate: Optional[HeartRateSignal] = None
    explain: bool = True

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.ecg or self.eda or self.heart_rate):
            raise ValueError("provide at least one of: ecg, eda, heart_rate")
        return self


class SleepFeatureRequest(BaseModel):
    features: dict[str, Optional[float]]
