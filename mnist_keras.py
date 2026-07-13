try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow n'est pas installe dans cet environnement. "
        "Installe-le avec `pip install -r requirements-mnist.txt` dans une version "
        "Python compatible TensorFlow."
    ) from exc


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nMNIST test loss={loss:.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
