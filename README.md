# TG-CUP Open Source Package: Code Implementation and Evaluation for TG-CUP

## TG-CUP

This report provides a detailed description of the TG-CUP open-source package, demonstrating how to reproduce the key experimental results from the paper. The open-source package includes the code, data, and environment configuration required for training, testing, and evaluating the TG-CUP model, aiming to support the reproducibility verification of the methods proposed in the paper under different experimental settings. This report lists in detail the functions of each file, the steps for environment configuration, the methods for obtaining the datasets, and the steps for reproducing the experiments, so that other researchers can reproduce the results and conduct further experiments.

### Environmental Configuration

The conda_requirements.txt and pip_requirements.txt under the first-level directory contain the environment required for the experiment. By executing the following code in the Linux system command line, the required configuration can be automatically installed.

The hardware environment of the experiment is as described in the paper. We train on **four Nvidia A800 80G** GPUs, the CPU is **Intel (R) Xeon (R) Gold 6348**, the system version is **Ubuntu 18.04.6 LTS**, and the software version is **Python 3.7** and **Pytorch 1.9.0**.

This open-source package has been fully tested on Ubuntu 18.04, and it should also run normally on other Linux distributions (such as CentOS or Debian), but full compatibility is not guaranteed. For Windows or macOS users, it may be necessary to use virtual machines or containerization tools (such as Docker) to simulate a Linux environment.

```shell
conda env create -f conda_requirements.txt
pip install -r pip_requirements.txt
```

### Dataset

We used the same dataset for CUP, HEB-CUP, and HAT-CUP, and removed samples that could not extract AST differences.

Please download the dataset and then place it in the TG_GGNN/data directory.

### Artifacts

| Artifact Name      | Location (Link/DOI)                                          | License | Description                                                  |
| ------------------ | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| TG-CUP Source Code | [GitHub Link](https://github.com/chenyn273/TG-CUP)           | MIT     | The source code for training and evaluation.                 |
| Trained Model      | [Google Drive Link](https://drive.google.com/drive/folders/1Ph5LEJXwMBz89qEIe2KXP_-MGnnHMcOA?usp=share_link) | CC-BY   | Source dataset used in the experiments. Pretrained models for model evaluation. |

### GitHub Open Source Package File Description

1. eval_tools

   ```
   1. eval: storing the evaluation and statistical significance difference analysis of TG-CUP and baseline.
   2. prediction: contains the compiled outputs of CUP, HEB-CUP, HAT-CUP and TG-CUP and some evaluation indicator results.
   3. p_delta.py: code for conducting a statistical significance difference analysis.
   4. utils.py: evaluation script implementation tool.
   ```

2. TG_GGNN

   ```
   1. cup_utils: construct code editing sequence.
   2. data: put the dataset here, with the get_corpus.py file used to build the BPE vocabulary.
   3. experiment: download the trained model and place it in the directory, the directory for model training logs and outputs.
   4. process_ast: script used to process AST to build Diff-AST graph.
   5. tokenizer: BPE model vocabulary and model, tokenize.py is used for BPE tokenization.
   6. beam_decoder.py: for beam_search.
   7. build_data.py: used to construct the output of TG-CUP.
   8. config.py: model training configuration file.
   data_loader.py: used to construct batch data.
   9. gnn.py: GGNN's basic implementation.
   10. graph_encoder.py: Diff-AST Graph Encoder Implementation.
   11. main.py: main function, including model training and model testing. If only model testing is required, please comment out the training part.
   12. model.py: TG-CUP architecture implementation.
   13. train.py: model training code.
   14. utils.py: model Implementation tools.
   15. new.xml、old.xml: temporarily generate intermediate files.
   ```

### Model training and inference

1. Training: After downloading the dataset to the specified location according to the GitHub Open Source Package File Description, run the following code

   ```shell
   python TG_GGNN/main.py
   ```

   

2. Only model inference: After downloading the trained model and dataset to the specified location according to the GitHub Open Source Package File Description, comment out the following code in TG_GGNN/main.py and run main.py

```python
51 train_dataset = MTDataset(config.train_data_path)
52 dev_dataset = MTDataset(config.dev_data_path)

56 train_dataloader = DataLoader(train_dataset, shuffle=True, 	batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
58 dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
78 train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)

```

### Model Evaluation

1. Copy TG_GGNN/experiment/output.txt to the tg directory under eval_tools/prediction (already completed; repeat this step if re-inference is needed).

2. Execute the following code

   ```shell
   python eval_tools/utils.py run
   ```

   
