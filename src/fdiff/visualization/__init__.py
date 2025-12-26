"""Visualization utilities for diffusion models.

This package provides modules for analyzing and visualizing results from
Time and Frequency domain diffusion models.
"""

from fdiff.visualization.results import (
    process_results,
    plot_sample_quality,
    process_spectral_analysis,
    plot_spectral_density,
    create_summary_table,
    main as results_main,
)

from fdiff.visualization.results_lstm import (
    process_results as process_results_lstm,
    plot_sample_quality as plot_sample_quality_lstm,
    main as results_lstm_main,
)

from fdiff.visualization.spectral_interpretation import (
    process_all_datasets,
    plot_spectral_density as plot_spectral_density_datasets,
    plot_temporal_energy,
    plot_localization,
    plot_localization_joint,
    main as spectral_main,
)

from fdiff.visualization.visualize import (
    visualize_samples,
    load_samples,
    plot_samples,
    heatmap_samples,
    main as visualize_main,
)

__all__ = [
    # Results analysis
    "process_results",
    "plot_sample_quality",
    "process_spectral_analysis",
    "plot_spectral_density",
    "create_summary_table",
    "results_main",
    # LSTM results
    "process_results_lstm",
    "plot_sample_quality_lstm",
    "results_lstm_main",
    # Spectral interpretation
    "process_all_datasets",
    "plot_spectral_density_datasets",
    "plot_temporal_energy",
    "plot_localization",
    "plot_localization_joint",
    "spectral_main",
    # Visualization
    "visualize_samples",
    "load_samples",
    "plot_samples",
    "heatmap_samples",
    "visualize_main",
]

