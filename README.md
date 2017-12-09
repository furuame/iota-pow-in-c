# Introduction
This project rewrite each implementation of Pow in C referenced from [iota.lib.go](https://github.com/iotaledger/iota.lib.go)

Now we have the following implementation:
* Pure C [PASS]
* SSE [PASS]
* AVX [FAILED]

OpenCL version is still in progress.

## Usage
```$ mkdir build```
```$ make test
