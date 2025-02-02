## lazyfox.py

LazyFox is a parallelized implementation of Fox -
a community detection algorithm for undirected graphs with support for overlapping communities

most similar algorithms can only assign a node to a single community, where this one can do multiple

[algorithm paper](https://peerj.com/articles/cs-1291.pdf)
[adapted from c++](https://github.com/timgarrels/LazyFox)

This Python version was created to get an understanding of the algorithm and experiment with improvements.

It's also been useful for testing the Free-threaded CPython 3.13 [non-GIL interpreter](https://docs.python.org/3/whatsnew/3.13.html#free-threaded-cpython) with a highly threaded workload.

The original c++ version is considerably faster.
