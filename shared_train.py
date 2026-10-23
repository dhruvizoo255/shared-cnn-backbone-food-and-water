
import os, json, csv, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path.home() / "sharedcnn_project"
DATA, OUT = ROOT / "data", ROOT / "runs"
OUT.mkdir(exist_ok=True)

SEED = 42
IMG = (128, 128)
BATCH = 16
EPOCHS = 50

tf.keras.utils.set_random_seed(SEED)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

FOOD_DIR, WATER_DIR = DATA / "food", DATA / "water"
FOOD_CLASSES = sorted(p.name for p in FOOD_DIR.iterdir() if p.is_dir())
WATER_CLASSES = ["Clear", "Foam", "Highly_Turbid", "Medium_Turbid"]

def collect_food():
    paths, labels = [], []
    for label, cls in enumerate(FOOD_CLASSES):
        for p in (FOOD_DIR / cls).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(str(p))
                labels.append(label)
    return paths, labels

def load_image(path, label):
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG)
    return image, label

def make_food_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 3000), seed=SEED)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(BATCH).prefetch(tf.data.AUTOTUNE)

def make_water_ds(split, training):
    ds = tf.keras.utils.image_dataset_from_directory(
        WATER_DIR / split,
        class_names=WATER_CLASSES,
        image_size=IMG,
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def build_shared(architecture):
    inputs = tf.keras.Input((*IMG, 3), name="image")

    if architecture == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG, 3), include_top=False, weights="imagenet"
        )
        base.trainable = False
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base(x, training=False)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = inputs / 255.0
        for filters in (16, 32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(pooled)
    embedding = tf.keras.layers.Dense(128, activation="relu", name="embedding_128")(x)

    food_head = tf.keras.layers.Dense(
        len(FOOD_CLASSES), activation="softmax", name="food_head"
    )(embedding)
    water_head = tf.keras.layers.Dense(
        len(WATER_CLASSES), activation="softmax", name="water_head"
    )(embedding)

    model = tf.keras.Model(inputs, {"food_head": food_head, "water_head": water_head})
    embedder = tf.keras.Model(inputs, {"embedding": embedding, "pooled": pooled})

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "food_head": "sparse_categorical_crossentropy",
            "water_head": "sparse_categorical_crossentropy",
        },
        metrics={
            "food_head": ["accuracy"],
            "water_head": ["accuracy"],
        },
    )
    return model, embedder

# Build deterministic splits for the quick shared-model run.
# Group-aware deduplication is still required for final evaluation.
food_paths, food_labels = collect_food()
rng = np.random.default_rng(SEED)
food_order = rng.permutation(len(food_paths))
food_n = len(food_order)
food_train_ids = food_order[:int(.7 * food_n)]
food_val_ids = food_order[int(.7 * food_n):int(.85 * food_n)]
food_test_ids = food_order[int(.85 * food_n):]

food_train = make_food_ds(
    [food_paths[i] for i in food_train_ids],
    [food_labels[i] for i in food_train_ids], True
)
food_val = make_food_ds(
    [food_paths[i] for i in food_val_ids],
    [food_labels[i] for i in food_val_ids], False
)

water_train = make_water_ds("train", True)
water_val = make_water_ds("valid", False)


# Correct shared training: alternate domains using explicit task-specific losses.
# Each batch updates the shared trunk and only its relevant classification head.
class SharedTrainer(tf.keras.Model):
    def __init__(self, backbone_model):
        super().__init__()
        self.network = backbone_model

    def train_step(self, data):
        images, targets = data
        with tf.GradientTape() as tape:
            outputs = self.network(images, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    targets["food_head"], outputs["food_head"]
                )
            ) if "food_head" in targets else tf.constant(0.0)

            if "water_head" in targets:
                water_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        targets["water_head"], outputs["water_head"]
                    )
                )
                loss += water_loss

            if self.network.losses:
                loss += tf.add_n(self.network.losses)

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, self.network.trainable_variables)
             if g is not None]
        )
        return {"loss": loss}

def food_targets(images, labels):
    return images, {"food_head": labels}

def water_targets(images, labels):
    return images, {"water_head": labels}

food_train_task = food_train.map(food_targets, num_parallel_calls=tf.data.AUTOTUNE)
water_train_task = water_train.map(water_targets, num_parallel_calls=tf.data.AUTOTUNE)

for architecture in ("mobilenetv2", "customcnn"):
    run_dir = OUT / f"{architecture}_shared"
    run_dir.mkdir(exist_ok=True)

    model, embedder = build_shared(architecture)
    trainer = SharedTrainer(model)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    food_iter = iter(food_train_task.repeat())
    water_iter = iter(water_train_task.repeat())

    for epoch in range(EPOCHS):
        # Equal number of batches from each domain.
        steps = min(
            int(tf.data.experimental.cardinality(food_train).numpy()),
            int(tf.data.experimental.cardinality(water_train).numpy())
        )
        for _ in range(steps):
            trainer.train_step(next(food_iter))
            trainer.train_step(next(water_iter))

        model.save(run_dir / "latest.keras")
        print(f"{architecture}: epoch {epoch+1}/{EPOCHS}", flush=True)

    model.save(run_dir / "final.keras")
    print("Saved:", run_dir, flush=True)

print("Shared training loops completed.", flush=True)

