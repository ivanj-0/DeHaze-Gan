# Project Overview: Dehazing GAN

## Objective
The objective of our project was to develop an enhanced Pix2Pix GAN model capable of effectively dehazing images. Our architecture employs a U-Net styled generator and a PatchGAN styled discriminator. The model uses a combination of adversarial and reconstruction losses to optimize performance.

## Installation

To install the required libraries, you can use pip to install directly from the `requirements.txt` file:

```bash
pip install -r requirements.txt 
```
You can also manually install each library listed in the requirements.txt file.
## Training
Data Preprocessing
Before training the model, preprocess the data to ensure smooth training (Note that this step is not required during testing).
Modify the training data path and run `preprocessing.py`.
Then Run `train.py`.

## Testing 

To test the model:

Ensure all requirements are installed as per the installation guide.
Modify the fisrt two paths in your `testing.py` and run it.
The output images will be saved in the designated output path you specify.


## Initial Setup and Results
- **Architecture**: U-Net styled generator with PatchGAN styled discriminator.
- **Identified Issues**: Initial results showed issues with bright white spots and generally low image quality.

## Experimentation Phase

### Experiment 1: Reducing Learning Rate of Discriminator
- **Motivation**: Discriminator loss was consistently low, suggesting that the discriminator was getting too powerful.
- **Results**: Generator loss: 12.582, Discriminator loss: 0.646, PSNR: 19.843, SSIM: 0.803

### Experiment 2: Introduction of Perceptual Loss
- **Motivation**: To enhance the quality of image reconstruction.
- **Results**: Generator loss: 26.253, Discriminator loss: 0.524, PSNR: 20.757, SSIM: 0.850

### Experiment 3: Weight Demodulation
- **Implementation**: Replacing Conv2d with DemodulatedConv2d.
- **Results**: Generator loss: 13.882, Discriminator loss: 0.600, PSNR: 19.460, SSIM: 0.772

### Experiment 4: Image Augmentation
- **Motivation**: As suggested in the original Pix2Pix paper.
- **Technique**: Random jittering, doubling the dataset.
- **Results**: Generator loss: 65.882, Discriminator loss: 0.600, PSNR: 21.934, SSIM: 0.808

### Experiment 5: Combining Perceptual Loss and Reduced Learning Rate
- **Extended Training to 80 Epochs**:
  - Generator loss: 26.262, Discriminator loss: 0.186, PSNR: 21.142, SSIM: 0.848
- **Outcome**: Best results with significantly reduced spots and improved image quality.

## Conclusion
Through a systematic approach of experimenting and adjusting the parameters, we successfully enhanced the dehazing capabilities of our GAN model. The final model shows marked improvement in both qualitative and quantitative aspects, providing a robust solution for image dehazing.
