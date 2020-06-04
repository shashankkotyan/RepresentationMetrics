# Representation Metrics

This GitHub repository contains the official code for the paper,

> [Representation Quality Explains Adversarial Attacks](https://arxiv.org/abs/1906.06627)\
> Danilo Vasconcellos Vargas, Shashank Kotyan, Moe Matsuki\
> _arXiv:1906.06627_.

## Citation

If this work helps your research and/or project in anyway, please cite:

```bibtex
@article{vargas2019representation,
  title   = {Representation Quality Explains Adversarial Attacks},
  author  = {Vargas, Danilo Vasconcellos and Kotyan, Shashank and Matsuki, Moe},
  journal = {arXiv preprint arXiv:1906.06627},
  year    = {2019}
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
- matplotlib==3.1.1
- numpy==1.17.2
- seaborn==0.9.0
- tensorflow==2.1.0

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

4. Train and evaluate a normal architecture.

```bash
python -u run_model.py [ARGS] > run.txt
```

5. Train and evaluate Raw Zero-shot architecture for all CIFAR-10 labels.

```bash
for label in {0..9}
do
    python -u run_model.py -r -xl $label [ARGS] > run.txt
done     
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
