from flask import Flask, render_template, redirect, url_for, request, flash
from Util import Util
from Autoencoder import Autoencoder

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return redirect(url_for('index'))

@app.route("/index", methods=['GET','POST'])
def index():
    if request.method == "GET":
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('Tidak ada file terpilih, Harap pilih file dengan format .jpg, .jpeg, atau .png.')
                return redirect(url_for('index'))
            if file and allowed_file((file.filename)):
                derau = request.form.get('option')
                util = Util(file)
                img = util.getImage()
                
                if derau == '2':
                    mean = request.form.get('mean')
                    noisy = util.RayleighNoise(img, mean)
                else:
                    mean = request.form.get('mean')
                    sd = request.form.get('sd')
                    noisy = util.GaussianNoise(img, mean, sd)
                
                cae = Autoencoder()
                denoised = cae.Denoising(img)
                util.saveImage(denoised, "denoised_")
                temp = util.getInfo(img, noisy, denoised)
                return render_template('index.html', temp=temp)  
            else:
                flash('Format file salah, Harap pilih file dengan format .jpg, .jpeg, atau .png.')
                return redirect(url_for('index'))
                           
    return render_template('index.html')   

if __name__ == '__main__':
    app.run(port=8000, debug=True)