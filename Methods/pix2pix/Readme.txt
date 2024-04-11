# Pix2pix implementation

## Running the code

To run the code, the number of the run must be included as an input and the main/test files must be chosen according to the dataset.


For the CBIS-DDSM dataset considering MLO to CC translation:
```bash
    python basis_cbis.py --run 0
    python pix2pix.py -run 0 --confexp MLOtoCC_pix2pix_aligned_200 --dataroot ./datasets/mammo --which_direction AtoB --num_epochs 200 --batchSize 4 --no_resize_or_crop --no_flip
    python pix2pix_test_cbis.py --dataset_name mammo  --run 0 --confexp MLOtoCC_pix2pix_aligned_200 --dataroot ./datasets/mammo --which_direction AtoB --num_epochs 200 --batchSize 4 --no_resize_or_crop --no_flip
```


For the CSAW dataset considering MLO to CC translation:
```bash
    python basis_csaw.py --run 0
    python pix2pix.py -run 0 --confexp MLOtoCC_pix2pix_aligned_200 --dataroot ./datasets/mammo --which_direction AtoB --num_epochs 200 --batchSize 4 --no_resize_or_crop --no_flip
    python pix2pix_test_csaw.py --dataset_name mammo  --run 0 --confexp MLOtoCC_pix2pix_aligned_200 --dataroot ./datasets/mammo --which_direction AtoB --num_epochs 200 --batchSize 4 --no_resize_or_crop --no_flip
```

To change direction of translation, which_direction should be BtoA