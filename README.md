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

### Missing Features:
- **No training yet.** I haven't implemented back-propagation and all that tasty math yet. So... For now, enjoy watching your neural nets produce meaningless garbage, I guess...
- **Not packaged properly.** You're gonna have to play around with Python's `sys.path` in order to import this. I haven't written installer scripts or setup.py files or whatever yet.

### Performance:
"But," I hear you ask, "How does `fml` compare to other ML frameworks?"

`¯\_(ツ)_/¯`
I dunno. Sucks, prob'ly. Nothing works, so there's not really any comparison to be made. I don't know how to use hardware acceleration or CUDA or anything, and it's written in Python, so I don't exactly expect great things.
