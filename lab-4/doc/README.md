## Usage

### Prerequisites

Under the `codes` folder:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install --file requirements.txt
```

Note that you may need to install specialized versions of certain packages to compute on CUDA. Please comment the line that switches to CUDA if you encounter any trouble.

### Run

#### PyTorch

Under the `codes/PyTorch` folder:

```bash
python exp2.py
```

#### CNN

:

For extracting the feature of `panda.png`, under the `codes/CNN` folder:

```bash
python extract_feature.py
```

For the remaining parts of the lab, also under the `codes/CNN` folder:

```bash
python match.py
```