# Non-Planar Tactile Methods

## Tools for Original Data
This section is primarily used for processing raw data, which consists of two main components:
1. **LR Data** collected by Xela sensors.
2. **HR Data** captured by GelSight sensors.

Both types of data require processing using the methods provided in this folder.

## NT-SR3
This section employs a vision-based SR3 (Super-Resolution) method for tactile super-resolution tasks.
- **Config**: Used to set parameters that control the model.
- **sr.py**: Script for training the model.
- **Dataset**: Before using, place the datasets separately. The `sr4_40` dataset is derived from `lr_4` through interpolation, which can be accomplished using the tools in the `data` folder.

## NT-SRCNN and NT-SRGAN
- **Utility**: Contains common visualization tools.
- **pth**: Directory for storing model files.

