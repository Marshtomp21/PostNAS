# PostNAS

> PostNAS builds on a pre-trained transformer model while enabling flexible exploration of attention block designs, greatly reducing the cost and risk of developing new language model architectures.

This repository implements NVIDIA's proposed PostNAS. At this stage, I have only implemented how to find the optimal position for the full attention layer.

### Paper

You can find this paper in the root directory of the project repository or visit https://arxiv.org/html/2508.15884v1.

### Project Structure

```
./
├── config.py # Config File
├── main.py # Entrance of the Program
├── run.sh # Running Script
├── models/
│ ├── gla.py # Implementation of GLA
│ └── super_network.py # Once-for-all Super Network
├── training/
│ ├── trainer.py # Trainer
│ └── beam_search.py # Beam Search Algorithm
├── utils/
│ └── data_utils.py # Data Loaders
├── data/
├── outputs/
├── checkpoints/
└── logs/
```
