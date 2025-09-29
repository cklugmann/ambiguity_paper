# seldon

**seldon** provides ambiguity measures over discrete label distributions and tools for estimating and inferring such distributions from crowdsourced data.  
This package accompanies the paper *Quantifying Ambiguity in Categorical Annotations: A Measure and Statistical Inference Framework*.

---

## 📦 Installation

We recommend installing `seldon` into a fresh virtual environment (e.g., via Conda):

```bash
conda create -n seldon-env python=3.10 -y
conda activate seldon-env
```

Then install the package:

### 🔹 Option 1 – Production install

Use this if you just need the core functionality (e.g. in other scripts or pipelines):

```bash
pip install .
```

### 🔹 Option 2 – Development + notebook install

Use this if you want to explore the codebase, run notebooks, or contribute:

```bash
pip install -e ".[dev,viz]"
```

This installs the package in editable mode, along with optional dependencies:
- `dev`: Jupyter, notebook support
- `viz`: matplotlib, seaborn

---

## 📔 Running notebooks

Make sure your environment is registered as a Jupyter kernel:

```bash
python -m ipykernel install --user --name seldon-env --display-name "Python (seldon)"
```

Then you can run any notebook under the `notebooks/` directory.

---

## 🔧 Structure

```
seldon/         # Python source code
cpp/            # C++ code for amb_pdf module (compiled via pybind11)
notebooks/      # Interactive experiments and demonstrations
pyproject.toml  # Build + dependency configuration
```

---

## 🛠 Development Notes

The `amb_pdf` C++ module is compiled automatically when running `pip install`.  
The resulting `.so` file is placed inside the `seldon/` module and can be imported like a regular Python module:

```python
from seldon import amb_pdf
```