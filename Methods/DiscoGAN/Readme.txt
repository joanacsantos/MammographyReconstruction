# DiscoGAN implementation

## Running the code

To run the code, the number of the run must be included as an input.
The run for each dataset is in a separate folder.


For the MLO to CC translation:
```bash
    python main_MLOtoCC.py --run 0
    python discogan.py --task mammo
    python discogan_test.py --task mammo --run 0 --confexp MLOtoCC_DiscoGAN_aligned_200
```

To change direction of translation, main code should be main_CCtoMLO.py