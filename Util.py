from tensorflow.keras.preprocessing import image
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

class Util:
    def __init__(self, file):
        self.filename1 = secure_filename(file.filename)
        self.filename2 = 'noisy_'+self.filename1
        self.filename3 = 'denoised_'+self.filename1
        file.save('static/img/'+self.filename1)
    
    def PSNR(self, img1,img2):
        mse = np.mean((img1 - img2) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return round(psnr,2)
    
    def SSIM(self, img1,img2):
        return round(ssim(np.squeeze(img1), np.squeeze(img2)),2)
       
    def getImage(self):
        img = image.load_img('static/img/'+secure_filename(self.filename1), target_size=(128,128), color_mode= 'grayscale')
        img = image.img_to_array(img)
        img = img/255
        img = [img]
        return np.array(img)
    
    def saveImage(self, img, message):
        plt.imshow(img[0])
        plt.set_cmap('gray')
        plt.axis('off')
        plt.savefig('static/img/'+message+self.filename1, bbox_inches='tight', pad_inches=0)
        plt.switch_backend('agg')
        
    def getInfo(self, img1, img2, img3):
        data = {
            'filename1' : self.filename1,
            'filename2' : self.filename2,
            'filename3' : self.filename3,
            'bpsnr'  : self.PSNR(img1,img2),
            'bssim'  : self.SSIM(img1,img2),
            'apsnr'  : self.PSNR(img1,img3),
            'assim'  : self.SSIM(img1,img3),
            'tipe'  : self.tipe
        }
        return data
    
    def GaussianNoise(self, img, mean, sigma):
        self.tipe = 'Gaussian ('+mean+','+sigma+')'
        row,col,ch= 128,128,1
        gauss = np.random.normal(float(mean),float(sigma),(row,col, ch))
        gauss = gauss.reshape(row,col, ch)
        img = img + gauss
        img = np.clip(img, 0,1)
        img = np.array(img)
        self.saveImage(img, 'noisy_')
        return img
    
    def RayleighNoise(self, img, mean):
        self.tipe = 'Rayleigh ('+mean+')'
        row,col,ch= 128,128,1
        mode = np.sqrt(2 / np.pi) * float(mean)
        rey = np.random.rayleigh(mode,(row,col,ch))
        rey = rey.reshape(row,col, ch)
        img = img + rey
        img = np.clip(img, 0,1)
        img = np.array(img)
        self.saveImage(img, 'noisy_')
        return img