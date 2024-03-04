# A Distributions-based Approach to Data-consistent Inversion Paper Code Repository

## Overview

Welcome to the GitHub repository for the research paper "**A Distributions-based Approach to Data-consistent Inversion**". This repository is intended to provide a comprehensive set of code and examples to facilitate the reproducibility of the research presented in the paper.

## Paper Information

- **Title:** A Distributions-based Approach to Data-consistent Inversion
- **Authors:** K. O. Bergstrom, T. D. Butler, T. M. Wildey
- **Submitted to:** SIAM Journal on Scientific Computing (SISC)

## Repository Structure

The repository is organized as follows:

- `src`: This directory contains the source code for implementing the algorithms and conducting experiments discussed in the paper. The file `binning` contains high-level routines for the binning-based method discussed in Section 4. The file `weights` contains lower-level routines for computing the naive weighting method discussed in Section 3.

- `examples`: This directory contains usage examples, needed data, and expected outputs to guide users in reproducing the results. The files `heat_eq.py` and `heat_eq.pynb` contain code and results for reproducing the figures and results related to the heat equation illustrative example first introduced in Section 2.4, referenced throughout the paper. The files `heat_eq_convergence.py` and `heat_eq_convergence.pynb`, contain code and resoluts for reproducing the figures and results related to the convergence example in Section 5.1. The files `fluid_flow.py` and `fluid_flow.pynb` contain code and plotting scripts for reproducing the results in Section 5.2. Finally, the `data` subdirectory contains data for the examples in 5.1 and 5.2.

## Reproducing Results

To reproduce the results presented in the paper, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kirana-bergstrom/DCI-distributions.git
   cd DCI-distributions
   ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env create -f environment.yml
    ```

3. **Explore Code and Examples:**

    Navigate to the `src` directory to explore the implementation details.

    Look in the `examples` directory for usage examples, presented in jupyter notebook and `.py` form.

    You can run the examples either as python scripts:

    ```bash
    python heat_eq.py
    ```

    or as Jupyter Notebooks (recommended):

    ```bash
    jupyter notebook heat_eq.py
    ```

## Provide Feedback

If you encounter any issues or have questions, please open an issue on this repository.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{Bergstrom:2024,
  title   = {A Distributions-based Approach to Data-consistent Inversion},
  author  = {K. O. Bergstrom and T. D. Butler and T. M. Wildey},
  journal = {SIAM Journal on Scientific Computing (SISC)},
  year    = {2024}
}
```

## License
This code is released under the MIT License. See the LICENSE file for details.

## Contact
For any inquiries or assistance, please contact Kirana Bergstrom at kirana.bergstrom@ucdenver.edu.
