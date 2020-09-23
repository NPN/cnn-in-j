# cnn-in-j

J port of [ashinkarov/cnn-in-apl](https://github.com/ashinkarov/cnn-in-apl), which is described in this paper:

> Artjoms Šinkarovs, Robert Bernecky, and Sven-Bodo Scholz. 2019. Convolutional neural networks in APL. In _Proceedings of the 6th ACM SIGPLAN International Workshop on Libraries, Languages and Compilers for Array Programming (ARRAY 2019)_. Association for Computing Machinery, New York, NY, USA, 69–79. DOI:https://doi.org/10.1145/3315454.3329960

## Usage

Ensure you have J (tested on J901) and the `format/printf` and `stats/base` addons installed. Then, run `download-mnist.sh`. If you can't run the script, download [the MNIST files](http://yann.lecun.com/exdb/mnist/) yourself, extract them, and place them in `input`. Then, run `cnn.ijs` to train and test the CNN.

In `main`, you can customize training by tweaking `epochs`, `trainings` (number of training examples), `tests` (number of test examples), and `rate`. The APL version also has a `batchsize` variable, but it's just for show: the CNN is trained using stochastic gradient descent, not batch gradient descent.

For reproducibility, the RNG seed is explicitly set to `16807`, which appears to be the default as of J901. You can change it if you want to get different results.

## Code notes

This is a mostly faithful translation of the GitHub version (not the paper version, which uses different variable names) with a few enhancements:

* Initialization of weights following [Zhang's paper](http://web.eecs.utk.edu/~zzhang61/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf) (section 1.1)
* Standardization of images
* Shuffling of training data at the start of each epoch

These changes increase the accuracy from 76.23% to 87.51%. There are many other opportunities for improvement, but the code is so slow that it's a drag to test changes. (It takes about 20s per epoch and 40s to test on my laptop.)

The other notable difference is that while the APL version avoids stencil `⌺`, we don't avoid J's subarrays `;._3`: J doesn't add padding like Dyalog does, so there's no performance penalty.
