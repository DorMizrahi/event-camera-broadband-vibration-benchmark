"""Standalone event-camera vibration analysis without learned calibration."""

from .config import AnalysisConfig
from .io import (
    EventData,
    EventFileInfo,
    build_event_accumulation,
    inspect_event_file,
    load_events,
)
from .pipeline import AnalysisResult, run_analysis
from .roi import ROI, auto_roi_from_activity, plot_roi_preview, select_roi_interactive

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "EventData",
    "EventFileInfo",
    "ROI",
    "auto_roi_from_activity",
    "build_event_accumulation",
    "inspect_event_file",
    "load_events",
    "plot_roi_preview",
    "run_analysis",
    "select_roi_interactive",
]

__version__ = "1.0.0"
