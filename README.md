# fml

Farcical Machine Learning: A boneless neural network library in Python3.

Inspired by [3blue1brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)'s awesome [video series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) on neural nets.

Motivated by [TensorFlow](https://www.tensorflow.org/), an industry-standard, full-featured machine-learning toolkit which causes my pathetic Pentium to emit `Illegal instruction (core dumped)` errors.

### Project Status:
**Work in progress.**

I have no experience with machine learning. This code is based entirely on knowledge acquired from watching YouTube videos about neural nets. If you wanna hack around on a half-finished implementation, or boost your self-esteem by looking at my sketchy code, please, by all means, do so.

### Features:
- A quick-and-dirty matrix class, to make multiplying huge arrays of numbers (slightly) easier.
- A neural net class that supports an arbitrary number of hidden layers.
- The ability to initialize a neural net with random weights and biases.
- Saveable/loadable models! Dump your neural net to a JSON string to save it for later.
- **Training!** Neural nets can now learn from labelled data! ...Sort of. It doesn't work if you're using a sigmoid transfer function, because the backpropagation algorithm does weird stuff and floating point overflow errors happen. It also doesn't work if you use a ReLU, because a lot of the neurons get stuck at 0 and can't learn anymore. So... yeah. Only the identity function works. And it doesn't work well.

### Problems:
- **Not packaged properly.** You're gonna have to play around with Python's `sys.path` in order to import this. I haven't written installer scripts or setup.py files or whatever yet.
- **Buggy AF.** This library works, but only barely.

### Contributing:
**Pull requests welcome!** If you have an idea that might improve this dumpster fire, if you have some real knowledge of machine learning and my garbage code makes you physically ill, or if you just want a piece of the nightmare, please submit a PR. Seriously, any changes or suggestions at all are welcome.
