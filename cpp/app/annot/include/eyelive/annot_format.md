# Annotation Persistence Format

The `ExpertLabels` model is saved as a JSON file.

## Structure

```json
{
  "version": 1,
  "n_classes": 3,
  "events": [
    {
      "onset": 100,
      "offset": 120,
      "cls": 1,
      "origin": 1,
      "detector_id": 0,
      "model_version": 0,
      "provisional": false,
      "confidence": 1.0,
      "t_onset": 0.1,
      "t_offset": 0.12,
      "amplitude_deg": 5.0,
      "peak_speed_deg_s": 200.0,
      "direction_rad": 1.57
    }
  ],
  "prelabels": [
    // same structure as events
  ],
  "excluded": [
    [0, 50]
  ],
  "reviewed": [
    [0, 1000]
  ]
}
```

Fields omitted from `Event` if equal to their default values might be allowed by the implementation, but saving writes all fields explicitly.
