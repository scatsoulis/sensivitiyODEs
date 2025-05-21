Sensitivity methods for ODEs - ACM 270 (Caltech)

# Environment setup (cluster)

```
pip3 install poetry
poetry lock
(eventually: poetry env use [Python version])
poetry install
poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
eval $(poetry env activate)
```

# Usage
