# UNIT implementation


This code is based on the implementation available at https://github.com/eriklindernoren/PyTorch-GAN

## Running the code

To run the code, the number of the run must be included as an input and the main/test files must be chosen according to the dataset.


For the CBIS-DDSM dataset:
```bash
    python main_cbis.py --run 0
    python unit.py --dataset_name mammo --run 0
    python test_cbis.py --dataset_name mammo  --run 0
```


For the CSAW dataset:
```bash
    python main_csaw.py --run 0
    python unit.py --dataset_name mammo --run 0
    python test_csaw.py --dataset_name mammo  --run 0
```
