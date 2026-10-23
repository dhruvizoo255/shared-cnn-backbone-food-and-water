import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import tensorflow as tf

ROOT = Path.home() / "sharedcnn_project"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

IMG = (160, 160)
BATCH = 16
EPOCHS = 3
tf.keras.utils.set_random_seed(42)

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__, "GPUs:", gpus, flush=True)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

def build(n):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = tf.keras.Input((*IMG, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    out = tf.keras.layers.Dense(n, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def ds(path, classes, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path, class_names=classes, image_size=IMG,
        batch_size=BATCH, shuffle=shuffle, seed=42
    ).prefetch(tf.data.AUTOTUNE)

def run(name, train, val, test, classes):
    model = build(len(classes))
    model.fit(train, validation_data=val, epochs=EPOCHS, verbose=1,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                OUT / f"{name}_best.keras", save_best_only=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.CSVLogger(OUT / f"{name}_history.csv")
        ])
    print(name, "TEST:", model.evaluate(test, return_dict=True), flush=True)
    model.save(OUT / f"{name}_final.keras")

# FOOD: map the 12 folders to binary fresh/stale labels.
# Load each folder and relabel its images.
food_root = DATA / "food"
paths, labels = [], []
for folder in sorted(food_root.iterdir()):
    if folder.is_dir() and folder.name.startswith(("fresh_", "stale_")):
        label = 0 if folder.name.startswith("fresh_") else 1
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                paths.append(str(p))
                labels.append(label)

import numpy as np
from sklearn.model_selection import train_test_split

idx = np.arange(len(paths))
tr, hold = train_test_split(idx, test_size=.2, random_state=42, stratify=labels)
va, te = train_test_split(hold, test_size=.5, random_state=42,
                          stratify=np.array(labels)[hold])

def path_ds(ids, shuffle):
    p = [paths[i] for i in ids]
    y = np.array(labels)[ids]
    d = tf.data.Dataset.from_tensor_slices((p, y))
    if shuffle:
        d = d.shuffle(min(len(p), 3000), seed=42)
    def load(path, label):
        im = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        im.set_shape([None, None, 3])
        return tf.image.resize(im, IMG), label
    return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

print("FOOD images:", len(paths), flush=True)
run("food", path_ds(tr, True), path_ds(va, False), path_ds(te, False), ["fresh","stale"])

# WATER: retain the provided splits and exclude Unlabeled.
water_root = DATA / "water"
classes = ["Clear", "Foam", "Highly_Turbid", "Medium_Turbid"]
print("WATER classes:", classes, flush=True)
run("water",
    ds(water_root/"train", classes, True),
    ds(water_root/"valid", classes, False),
    ds(water_root/"test", classes, False),
    classes)

print("Finished. Models and logs:", OUT, flush=True)
