# How To

1. Adjust the config.yaml if you want to (this essentially sets the parameter limits)
2. Open a terminal from this folder and and run 'python bicycle_model.py --config config.yaml'
3. Hopefully you will now see the application (image below); there is  
3.1. a phase-space plot (left),  
3.2. plots for the state elements in time domain (middle),  
3.3. and frequency domain (right)
4. Use the slider to play with the parameters and watch how the system behaves!

![image info](./BicycleModelScript.png)

# Known Issues

1. You might face issues with various python versions other than python 3.8
2. The plot limits are not responsive for now and it might be that some information can not be visualized given a certian combination of slider position