# Inference code with Torch C++. (under construction)

## Installation
### LibTorch
- Download LibTorch from [here](https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip).

- Add path to LibTorch to environment variable `Torch_DIR`.

```bash
echo %Torch_DIR%
> C:\libs\libtorch-win-shared-with-deps-1.4.0\share\cmake\Torch
```

- Add path to *lib and *dll to environment variable `Path`.

```bash
echo %Path%
> foo;bar;C:\libs\libtorch-win-shared-with-deps-1.4.0\lib;
```

### PyTorch BCNNs
- See the https://github.com/yuta-hi/pytorch_bayesian_unet

## Dump model
```bash
python dump_model.py
```

## Build
- CMake
```bash
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
```

- Open `./infer-app.sln`
