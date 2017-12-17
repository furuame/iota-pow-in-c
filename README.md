# Introduction
This project rewrite each implementation of Pow in C referenced from [iotaledger/iota.lib.go](https://github.com/iotaledger/iota.lib.go)

OpenCL implementation is referenced from [iotaledger/ccurl](https://github.com/iotaledger/ccurl)

Now we have the following implementation:
* Pure C [PASS]
* SSE [PASS]
* AVX [FAILED]
* OpenCL [PASS}

## Usage
You need to modify path of OpenCL shared library in Makefile by yourself.

```$ mkdir build```

```$ make test```

And the following output is expected to see
```
./pow_c
Test pow_c
Trytes: 9HQBCLTYWRBTCAIDIBJJCOSKWZHHNNDLTIGHYUEVQYUOKJKZDIRTI9FLNEZBIQUGHFNSKKUHUBNEZ9999
./pow_sse
Test  pow_sse
Trytes: KJQEILJFZJYJZBNBTYXNBSNCCMHZDYZXTCHXADBMNPKHFOHNLWJLIGTUHPFEKRZEQ9DZHBJIUJRO99999
./pow_avx
Test  pow_avx
Trytes: J9QTUNNMONCMIR9JBNMRC9SC9QTBRKBUVCBYBUITBHEICYVQ9HXEXSPWPU9KACTSDRSQBDOJPOOEAFVMP
./pow_cl
Test pow_cl
Trytes: KJQEILJFZJYJZBNBTYXNBSNCCMHZDYZXTCHXADBMNPKHFOHNLWJLIGTUHPFEKRZEQ9DZHBJIUJRO99999
```
