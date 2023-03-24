import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits2 = load_digits(n_class=2)
X = digits2.data
y = digits2.target
plt.gray()
plt.matshow(digits2.images[1])
plt.show()
model = Sequential([
    tf.keras.Input(shape=(64,)),
    Dense(units=25, activation="sigmoid"),
    Dense(units=15, activation="sigmoid"),
    Dense(units=1, activation="sigmoid")
], name="model1")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer="adam", )
model.fit(X, y, epochs=20)
pred = model.predict(digits2.images[1].reshape(1, 64))
r = pred >= 0.5
print(int(r))
