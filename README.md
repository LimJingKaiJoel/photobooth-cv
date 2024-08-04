# photobooth-cv

gradient ascent for deepdream explanation:
- normally, we backpropagate the gradient of the loss (cross-entropy(predicted probabilities, actual labels)) wrt each weight
- i.e., we aim to descend the gradient of the loss wrt each weight, thus decreasing the loss (gradient descent)
- in deepdream gradient ascent, we backpropagate the magnitude/norm of an intermediate layer's output wrt each pixel in the image
- we are asking, how can we adjust the value of each pixel in the image to increase the magnitude of the intermediate layer's output,
- i.e., ascending the gradient of the intermediate layer output wrt each pixel of the image (gradient ascent)

