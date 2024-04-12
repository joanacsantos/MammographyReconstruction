# ResViT implementation


This code is based on the implementation available at https://github.com/icon-lab/ResViT

## Running the code

To run the code, the number of the run must be included as an input.
The run for each dataset is in a separate folder.


For the MLO to CC translation:
```bash
    python main_MLOtoCC.py --run 0
    python train.py --dataroot datasets/mammo/ --name pre_trained --which_model_netG res_cnn --which_direction AtoB --niter 50 --niter_decay 50 --lr 0.0002 --conf_exp MLOtoCC_ResViT_200 --run 0
    python train.py --dataroot datasets/mammo/ --name mammo_resvit --which_model_netG resvit --which_direction AtoB --niter 100 --niter_decay 100 --pre_trained_transformer 1 --pre_trained_resnet 1 --pre_trained_path checkpoints/pre_trained/latest_net_G.pth --lr 0.001 --conf_exp MLOtoCC_ResViT_200 --run 0
    python test.py --dataroot datasets/mammo/ --name mammo_resvit --which_model_netG resvit --phase test --serial_batches --conf_exp MLOtoCC_ResViT_200 --run 0
```

To change direction of translation, main code should be main_CCtoMLO.py