# Parameter Reference

Every run merges the user block with internal defaults, validates it, and writes
the fully resolved configuration to `parameters.json`.

## `experiment`

| Parameter | Meaning | Default |
|---|---|---:|
| `name` | Directory-safe experiment identifier | `single_sine_500hz_default` |
| `output_root` | Parent directory for experiments and summary | `generated_experiments` |
| `output_policy` | `new`, `error`, or `overwrite` when the name exists | `new` |
| `random_seed` | Frequency, weight, phase, and default V2E seed | `7301` |
| `write_manifest` | Write per-file size and SHA-256 table | `true` |

## `sampling`

| Parameter | Meaning | Default |
|---|---|---:|
| `video_fps` | Video/ground-truth sample rate in Hz | `40000` |
| `duration_mode` | Define duration by `seconds` or `cycles` | `seconds` |
| `duration_sec` | Requested duration when mode is `seconds` | `0.2` |
| `number_of_cycles` | Cycle count when mode is `cycles` | `40` |
| `cycle_reference_frequency_hz` | Reference used to convert cycles to seconds | `500` |
| `width_px`, `height_px` | Output video dimensions | `240`, `180` |
| `save_frames` | Save every source frame as PNG | `true` |
| `recommended_samples_per_cycle` | Warning threshold, not a resampler | `20` |

The frame count is `ceil(video_fps * resolved_duration_sec)`. Nyquist is enforced
against the highest active frequency. The `cycles` duration mode can be used for
all scenarios, but for broadband motion the reference frequency must be chosen
explicitly because there is no single physical period.

## `rendering`

| Parameter | Meaning | Default |
|---|---|---:|
| `object_shape` | `circle`, `rectangle`, or `square` | `circle` |
| `marker_radius_px` | Circle radius | `5` |
| `rectangle_width_px`, `rectangle_height_px` | Rectangle dimensions | `10`, `10` |
| `base_position_x_px`, `base_position_y_px` | Zero-displacement center | `120`, `90` |
| `background_intensity` | 8-bit grayscale background | `5` |
| `marker_contrast` | Fraction of remaining 8-bit intensity range | `0.72` |
| `video_codec` | ImageIO/FFmpeg codec | `libx264` |
| `video_crf` | H.264 quality parameter | `28` |
| `video_preset` | H.264 speed preset | `veryfast` |

The generator rejects a trajectory when the complete object would leave the image.

## `calibration`

| Parameter | Meaning | Default |
|---|---|---:|
| `meters_per_pixel` | Optional physical displacement scale | `null` |
| `standard_gravity_m_s2` | Conversion constant for acceleration in `g` | `9.80665` |

With a null scale, the scientifically valid acceleration unit is `pixel/s^2`.
No physical unit is guessed. With a known scale, SI columns are added while the
original pixel-based columns remain unchanged.

## Common `motion` Parameters

| Parameter | Meaning | Default |
|---|---|---:|
| `type` | One of the six supported scenarios | `single_sine` |
| `frequency_min_hz`, `frequency_max_hz` | Broadband sampling range | `50`, `2000` |
| `normalization_mode` | Motion metric held at the requested value | `peak_displacement_px` |
| `normalization_value` | Target value in the units encoded by the mode | `6` |
| `two_axis_motion` | Enable y-axis motion | `false` |
| `y_amplitude_ratio` | y/x component amplitude ratio | `0.25` |
| `custom_frequencies_hz` | Explicit component list for non-chirp broadband scenarios | `[]` |
| `custom_weights` | Explicit non-negative weights matching custom frequencies | `[]` |
| `custom_phases_x_rad`, `custom_phases_y_rad` | Explicit phases matching components | `[]` |

Accepted normalization modes are:

```text
peak_displacement_px, rms_displacement_px,
peak_velocity_px_s, rms_velocity_px_s,
peak_acceleration_px_s2, rms_acceleration_px_s2,
peak_acceleration_m_s2, rms_acceleration_m_s2
```

The two physical-acceleration modes require `meters_per_pixel`.

## Scenario Sections

### `single_sine`

| Parameter | Meaning | Default |
|---|---|---:|
| `frequency_hz` | Single vibration frequency | `500` |
| `phase_x_rad`, `phase_y_rad` | Initial phases | `0`, `pi/2` |

### `sparse_multi_sine`

| Parameter | Meaning | Default |
|---|---|---:|
| `num_components` | Number of frequencies | `8` |
| `minimum_frequency_separation_hz` | Required pairwise spacing | `35` |
| `weight_min`, `weight_max` | Uniform component weight range | `0.6`, `1.3` |

### `dense_multi_sine`

| Parameter | Meaning | Default |
|---|---|---:|
| `num_components` | Number of random frequencies | `32` |
| `weight_lognormal_mean` | Lognormal mean parameter | `0` |
| `weight_lognormal_sigma` | Lognormal spread parameter | `0.45` |

### `band_limited_random_phase`

| Parameter | Meaning | Default |
|---|---|---:|
| `num_components` | Number of equally spaced components | `80` |
| `spectral_variant` | `flat`, `low_tilt`, or `mid_bump` | `flat` |
| `mid_bump_center_hz` | Gaussian-envelope center | `800` |
| `mid_bump_sigma_hz` | Gaussian-envelope standard deviation | `450` |

`low_tilt` uses weights proportional to `1/sqrt(f)`. Randomness is in phase;
frequencies are equally spaced unless a custom list is supplied.

### `chirp`

| Parameter | Meaning | Default |
|---|---|---:|
| `start_frequency_hz`, `end_frequency_hz` | Sweep endpoints | `50`, `2000` |
| `kind` | `linear` or `log` | `linear` |
| `phase_x_rad`, `phase_y_rad` | Initial phases | `0`, `pi/2` |

Velocity and acceleration are analytical derivatives of the chirp displacement,
including the instantaneous-frequency slope term.

### `mixed_bands`

| Parameter | Meaning | Default |
|---|---|---:|
| `bands_hz` | List of `[low, high]` frequency bands | `[[50,200],[1000,2000]]` |
| `components_per_band` | Number sampled from each band | `[6,6]` |
| `weight_ranges` | Uniform weight range for each band | `[[0.8,1.2],[0.6,1.0]]` |

The three lists must have equal lengths.

## `v2e`

| Parameter | Meaning | Default |
|---|---|---:|
| `enabled` | Attempt V2E conversion | `true` |
| `command` | Optional V2E executable/command | empty |
| `script_path` | Optional path to `v2e.py` | empty |
| `python_executable` | Python used with `script_path` | current Python |
| `disable_slomo` | Pass `--disable_slomo` | `true` |
| `pos_thres`, `neg_thres` | ON/OFF contrast thresholds | `0.2`, `0.2` |
| `sigma_thres` | Pixel threshold variation | `0.03` |
| `refractory_period_sec` | Pixel refractory period | `0.0001` |
| `shot_noise_rate_hz` | Shot-noise rate | `0` |
| `cutoff_hz` | Photoreceptor cutoff; zero disables it | `0` |
| `dvs_emulator_seed` | Positive fixed seed; null uses experiment seed | `null` |
| `timeout_sec` | Conversion timeout; zero means unlimited | `0` |

These parameters are written verbatim, together with the resolved V2E seed and
the complete command. Changing them creates a different event-camera model condition.
