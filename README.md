# Representation Metrics

This GitHub repository contains the official code for the paper,

> [Transferability of features for neural networks links to adversarial attacks and defences](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266060)\
> Shashank Kotyan, Moe Matsuki and Danilo Vasconcellos Vargas\
> PLOS One (2022).

## Citation

If this work helps your research and/or project in anyway, please cite:

```bibtex
@article{kotyan2022transferability,
  title={Transferability of features for neural networks links to adversarial attacks and defences},
  author={Kotyan, Shashank and Matsuki, Moe and Vargas, Danilo Vasconcellos},
  journal={PloS one},
  volume={17},
  number={4},
  pages={e0266060},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## Testing Environment

The code is tested on Ubuntu 18.04.3 with Python 3.7.4.

## Getting Started

### Requirements

To run the code in the tutorial locally, it is recommended, 
- a dedicated GPU suitable for running, and
- install Anaconda.  

The following python packages are required to run the code. 
- `matplotlib==3.1.1`
- `numpy==1.17.2`
- `seaborn==0.9.0`
- `tensorflow==2.1.0`

---

### Steps

1. Clone the repository.

```bash
git clone https://github.com/shashankkotyan/RepresentationMetrics.git
cd ./RepresentationMetrics
```

2. Create a virtual environment 

```bash
conda create --name rm python=3.7.4
conda activate rm
```

3. Install the python packages in `requirements.txt` if you don't have them already.

```bash
pip install -r ./requirements.txt
```

4. Train and evaluate a normal or a Raw Zero-Shot classifier.

```bash
python -u code/run.py [ARGS] > run.txt
```

<!-- 5. Calculate DBI and Amalgam Metrics for the Raw Zero-Shot architecture including the 2D visualisation of the _'n-1'_ soft labels.

```bash
python -u run_stats.py > run_stats.txt     
```

6. (Optional) Evaluate the DBI and AM metrics with the adversarial examples.

```bash
python -u run_adversarial_stats.py > run_adversarial_stats.txt     
```
-->

## Arguments for run_model.py

TBD

## Notes

- To evaluate the DBI and AM metrics with the adversarial examples. Please generate the adversarial examples using the repository [Dual Quality Assessment](https://github.com/shashankkotyan/DualQualityAssessment/)

## Milestones

- [ ] Tutorials
- [ ] Addition of Comments in the Code
- [ ] Cross Platform Compatibility
- [ ] Description of Method in Readme File

## License

Representation Metrics is licensed under the MIT license. 
Contributors agree to license their contributions under the MIT license.

## Contributors and Acknowledgements

TBD

## Reaching out

You can reach me at shashankkotyan@gmail.com or [\@shashankkotyan](https://twitter.com/shashankkotyan).
If you tweet about Representation Metrics, please use the following tag `#raw_zero_shot`, and/or mention me ([\@shashankkotyan](https://twitter.com/shashankkotyan)) in the tweet.
For bug reports, questions, and suggestions, use [Github issues](https://github.com/shashankkotyan/RepresentationMetrics/issues).
