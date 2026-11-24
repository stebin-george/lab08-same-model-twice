# Which framework should run the weekly job

Both versions are the Lab 07 main network: two hidden layers of 64 ReLU units, mean squared error, Adam at learning rate 0.001, batches of 64, and 80 epochs on the same split. Keras Adam defaults to epsilon 1e-7, so the script sets 1e-8 to match PyTorch. The same seed still does not produce the same starting weights. Keras uses Glorot uniform and zero biases, PyTorch uses its own linear-layer scheme, and each framework shuffles batches with its own generator.

On three seeds the test error averages 3.18 bikes for both. PyTorch test MAE was 3.07, 2.94, and 3.53. Keras was 3.28, 3.14, and 3.11. The averages match, and the PyTorch spread is wider: seed 2028 is about half a bike worse than the other PyTorch seeds, while Keras stays inside 0.17 bikes. Validation means are 3.26 and 3.32. Every run except PyTorch seed 2026 was still improving at epoch 80, and that one stopped at 79. The curves are still falling. This is the small main network. The wider Lab 07 network reached about 2.4 bikes.

The clock is not a tie. PyTorch took 5.6, 3.0, and 3.0 seconds. Keras took 7.9, 7.9, and 8.1. That is the training call only, before the one test score.

The code differs as well. In `train.py`, `run_epoch` calls `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`, and `train_model` copies the best weights by hand. Those two functions are about 45 lines. In `train_keras.py` the update is one `fit` call. `EarlyStopping` uses patience equal to the epoch count, so the run lasts all 80 epochs and then restores the best weights. `ModelCheckpoint` writes `reports/keras_main.keras`, about 81 KB. The PyTorch script saves no model file. Copying the weights with each Dense kernel transposed makes the two networks agree to within 1e-5 before training.

I recommend PyTorch for the weekly job. The test error is the same, the run is faster here, and that script and its tests already exist. Hugging Face models, which the team may try later, are published for PyTorch first, so staying in PyTorch is the shorter path to that work.

What we give up is the smaller training call and the steadier seeds. If the team drops the Hugging Face plan and the Keras developer will own the weekly job, use Keras, so that person keeps the job in the framework they already know.
