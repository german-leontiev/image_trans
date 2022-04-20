import os
from app import app
from flask import flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    return result


def convert_image(filename, filepath):
    img_original = Image.open(filepath)
    images = {}
    # Original
    images['orig'] = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # Grayscale
    images['gray'] = cv2.cvtColor(images['orig'], cv2.COLOR_RGB2GRAY)
    # Black-white
    _, images['bw'] = cv2.threshold(images['gray'], 127, 255, cv2.THRESH_BINARY)
    # Sharpness
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    images['sharp'] = cv2.filter2D(images['orig'], -1, kernel_sharp)
    # Contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    images['cont'] = cv2.convertScaleAbs(images['orig'], alpha=alpha, beta=beta)
    # White Balance
    images['white'] = white_balance_loops(images['orig'])
    # Inversion
    images['inv'] = cv2.bitwise_not(images['orig'])
    list_of_images = []
    for i_key, i_val in images.items():
        save_path = filepath[:-len(filename)] + i_key + "_" + filename
        list_of_images.append(i_key + "_" + filename)
        Image.fromarray(i_val).save(save_path)
    return list_of_images


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        converted = convert_image(filename, filepath)
        filenames = [secure_filename(f) for f in converted]
        print(filename, converted, filenames)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filenames=filenames)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5001")
