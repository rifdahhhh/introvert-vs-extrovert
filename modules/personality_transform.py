
import tensorflow as tf
import tensorflow_transform as tft

# Label 
LABEL_KEY = "Personality"

# Fitur kategorikal
CATEGORICAL_FEATURE_KEYS = [
    "Stage_fear",
    "Drained_after_socializing"
]

# Fitur numerik 
NUMERIC_FEATURE_KEYS = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

def transformed_name(key: str) -> str:
    """Menghasilkan nama fitur setelah transformasi."""
    return key + "_xf"

def preprocessing_fn(inputs: dict) -> dict:
    """
    Fungsi preprocessing untuk mentransformasi fitur (tanpa penanganan missing value).

    Args:
        inputs (dict): Dictionary berisi fitur input (tensors).

    Returns:
        dict: Dictionary berisi fitur output hasil transformasi.
    """
    outputs = {}

    # 1. Proses fitur numerik: langsung normalisasi
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # 2. Proses fitur kategorikal: langsung encoding
    for key in CATEGORICAL_FEATURE_KEYS:
        encoded = tft.compute_and_apply_vocabulary(inputs[key])
        outputs[transformed_name(key)] = encoded

    # 3. Label binarisasi
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], "Extrovert"), tf.int64
    )

    return outputs
