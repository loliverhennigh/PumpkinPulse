# DensePlasmaFocus
Dense Plasma Focus Fusion Reactor


## Things That are True

* Linear algebra frameworks come and go. Tensorflow 1.0 is long gone and someday PyTorch and JAX will follow. Make any solver resiliant to these changes or your solver will also die someday.
* DSL are really fun to write and use but suck to extend if you go slightly outside the original scope. We have all made this mistake. A truely Multi-Physics framework wont fit into any DSL. Python needs to be your "DSL".
* Seperate Compute from memory if possible. It sucks to use a framework where you cant pull out the compute kernels for some 
