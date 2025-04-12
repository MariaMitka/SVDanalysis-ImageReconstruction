#Maria Mitka
#tem2884
#askhsh 1.4


import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

folderpath = r'C:\Users\mituk\PycharmProjects\PythonProject5\.venv\images'
imagefiles = [f for f in os.listdir(folderpath) if f.endswith(('.png', '.jpg', '.jpeg'))]

for imagefile in imagefiles:
    imagepath = os.path.join(folderpath, imagefile)

    img = mpimg.imread(imagepath)
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    print("SVD: ")
    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    Sigma = np.diag(S)
    errors = []

    kval = range(1, min(100, len(S) + 1))
    print("errors for diffetent k:")
    for k in kval:
        Uk = U[:, :k]
        Sigmak = Sigma[:k, :k]
        Vtk = Vt[:k, :]
        Ak = np.dot(Uk, np.dot(Sigmak, Vtk))

        errork = la.norm(img - Ak)
        errors.append(errork)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(kval, errors, 'bo--', label='Error ek')
    plt.title('error ek vs k')
    plt.xlabel('k')
    plt.ylabel('ek')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(S, 'ro--', label='idiazoyses values si')
    plt.title('idiazoyses values si')
    plt.xlabel('i')
    plt.ylabel('si')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    kvalues = [10, 40, 80]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, len(kvalues) + 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    for i, k in enumerate(kvalues):
        Uk = U[:, :k]
        Sigmak = Sigma[:k, :k]
        Vtk = Vt[:k, :]

        Ak = np.dot(Uk, np.dot(Sigmak, Vtk))

        plt.subplot(1, len(kvalues) + 1, i + 2)
        plt.imshow(Ak, cmap='gray')
        plt.title(f'A{k}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    imgM = img.size
    UM = U.size
    SigmaM = Sigma.size
    VtM = Vt.size
    AkM = [U[:, :k].size + Sigma[:k, :k].size + Vt[:k, :].size for k in kvalues]

    print(f"Memory size for original image: {imgM}")
    print(f"Memory size for full matrix A: {UM + SigmaM + VtM}")
    for k, AkMsize in zip(kvalues, AkM):
        print(f"Memory size for A{k}: {AkMsize}")
