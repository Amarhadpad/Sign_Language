import random
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

from sign_pipeline import load_labels

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
DATA_DIR = BASE_DIR / "Data"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
CANDIDATE_PATH = MODEL_DIR / "keras_model_candidate.h5"
IMAGE_SIZE = 224
BATCH_SIZE = 16
VALIDATION_FRACTION = 0.2
MINIMUM_SAMPLES_PER_CLASS = 20
RANDOM_SEED = 42


def collect_split(labels: list[str]) -> tuple[list[str], list[int], list[str], list[int], dict[str, int]]:
    train_files: list[str] = []
    train_targets: list[int] = []
    validation_files: list[str] = []
    validation_targets: list[int] = []
    counts: dict[str, int] = {}
    random_generator = random.Random(RANDOM_SEED)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_index, label in enumerate(labels):
        class_dir = DATA_DIR / label
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing samples for {label!r}. Collect images first with datacollection.py."
            )

        files = sorted(
            str(path)
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in valid_extensions
        )
        counts[label] = len(files)
        if len(files) < MINIMUM_SAMPLES_PER_CLASS:
            raise ValueError(
                f"{label!r} has {len(files)} images; collect at least "
                f"{MINIMUM_SAMPLES_PER_CLASS} before training."
            )

        random_generator.shuffle(files)
        validation_count = max(1, round(len(files) * VALIDATION_FRACTION))
        validation_count = min(validation_count, len(files) - 1)
        validation_files.extend(files[:validation_count])
        validation_targets.extend([class_index] * validation_count)
        train_files.extend(files[validation_count:])
        train_targets.extend([class_index] * (len(files) - validation_count))

    return train_files, train_targets, validation_files, validation_targets, counts


def decode_and_normalize(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.reverse(image, axis=[-1])
    image = tf.cast(image, tf.float32) / 127.0 - 1.0
    return image, target


def make_dataset(
    files: list[str],
    targets: list[int],
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((files, targets))
    if training:
        dataset = dataset.shuffle(
            len(files),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
    return (
        dataset
        .map(decode_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_training_model(model: tf.keras.Model) -> tuple[tf.keras.Model, tf.keras.Model]:
    feature_extractor = find_nested_model(model, "model1")
    if feature_extractor is None:
        raise ValueError("Could not find the pretrained feature extractor named 'model1'.")
    model.layers[0].trainable = True
    model.layers[1].trainable = True
    feature_extractor.trainable = True
    for layer in feature_extractor.layers:
        layer.trainable = False

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    outputs = model(augmentation(inputs))
    return tf.keras.Model(inputs, outputs), feature_extractor


def find_nested_model(model: tf.keras.Model, name: str) -> tf.keras.Model | None:
    if model.name == name:
        return model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            nested_model = find_nested_model(layer, name)
            if nested_model is not None:
                return nested_model
    return None


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def train_stage(
    model: tf.keras.Model,
    training_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    class_weights: dict[int, float],
    checkpoint_path: Path,
    epochs: int,
) -> tf.keras.Model:
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    best_stage_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    return best_stage_model


def evaluate(model: tf.keras.Model, validation_data: tf.data.Dataset) -> tuple[float, float]:
    if model.optimizer is None:
        compile_model(model, learning_rate=1e-5)
    loss, accuracy = model.evaluate(validation_data, verbose=0)
    return float(loss), float(accuracy)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    labels = load_labels(LABELS_PATH)
    train_files, train_targets, validation_files, validation_targets, counts = collect_split(labels)
    print("Samples per class:")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    if min(counts.values()) < 50:
        print("Tip: validation accuracy will be more reliable with at least 100 varied samples per sign.")

    training_data = make_dataset(train_files, train_targets, training=True)
    validation_data = make_dataset(validation_files, validation_targets, training=False)
    sample_count = len(train_files)
    class_weights = {
        class_index: sample_count / (len(labels) * count)
        for class_index, count in enumerate(
            counts[label] - max(1, round(counts[label] * VALIDATION_FRACTION))
            for label in labels
        )
    }

    original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if original_model.input_shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected model input shape: {original_model.input_shape}")
    if original_model.output_shape[-1] != len(labels):
        raise ValueError(
            f"Model has {original_model.output_shape[-1]} outputs, but labels.txt has {len(labels)} labels."
        )

    baseline_loss, baseline_accuracy = evaluate(original_model, validation_data)
    print(f"Original model validation accuracy: {baseline_accuracy:.3f}")

    training_model, feature_extractor = build_training_model(original_model)
    with tempfile.TemporaryDirectory(dir=MODEL_DIR, prefix="training-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        compile_model(training_model, learning_rate=1e-4)
        head_checkpoint = temporary_path / "head_best.h5"
        head_model = train_stage(
            training_model,
            training_data,
            validation_data,
            class_weights,
            head_checkpoint,
            epochs=20,
        )
        head_loss, head_accuracy = evaluate(head_model, validation_data)
        candidates = [(head_accuracy, head_loss, head_model)]

        fine_tune_model = head_model
        feature_extractor = find_nested_model(fine_tune_model, "model1")
        if feature_extractor is None:
            raise ValueError("Could not find the pretrained feature extractor in the training model.")
        for layer in feature_extractor.layers[:-30]:
            layer.trainable = False
        for layer in feature_extractor.layers[-30:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
        compile_model(fine_tune_model, learning_rate=1e-5)

        fine_tune_checkpoint = temporary_path / "fine_tune_best.h5"
        fine_tuned_model = train_stage(
            fine_tune_model,
            training_data,
            validation_data,
            class_weights,
            fine_tune_checkpoint,
            epochs=10,
        )
        fine_tune_loss, fine_tune_accuracy = evaluate(fine_tuned_model, validation_data)
        candidates.append((fine_tune_accuracy, fine_tune_loss, fine_tuned_model))

        best_accuracy, best_loss, best_model = max(
            candidates,
            key=lambda result: (result[0], -result[1]),
        )
        best_model.save(CANDIDATE_PATH)

    print(f"Original validation accuracy: {baseline_accuracy:.3f} (loss {baseline_loss:.3f})")
    print(f"Candidate validation accuracy: {best_accuracy:.3f} (loss {best_loss:.3f})")
    predictions = best_model.predict(validation_data, verbose=0)
    predicted_targets = np.argmax(predictions, axis=1)
    confusion = tf.math.confusion_matrix(
        validation_targets,
        predicted_targets,
        num_classes=len(labels),
    ).numpy()
    print("Validation accuracy by sign:")
    for class_index, label in enumerate(labels):
        class_total = int(confusion[class_index].sum())
        class_accuracy = confusion[class_index, class_index] / class_total
        print(f"  {label}: {class_accuracy:.3f} ({confusion[class_index, class_index]}/{class_total})")
    print(f"Candidate model saved to: {CANDIDATE_PATH}")
    if best_accuracy < baseline_accuracy:
        print("Candidate did not improve validation accuracy; keep the original model.")
    print("Review the validation results before replacing Model/keras_model.h5.")


if __name__ == "__main__":
    main()
