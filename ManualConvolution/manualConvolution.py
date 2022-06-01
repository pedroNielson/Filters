import numpy as np
import cv2 as cv 


def convolution (img, kernel):
    m, n = img.shape
    m1, n1 = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))

    image_padding = np.zeros((m + m1*2 - 2, n + n1*2 - 2))
    image_padding[m1//2+1 :- (m1//2+1), n1//2+1 :- (n1//2+1)] = img

    img_out = np.zeros((m + m1 - 1, n + n1 - 1))
    
    m2, n2 = img_out.shape

    for i in range(0, m2):
        for j in range(0, n2):
            img_out[i, j] = np.multiply(kernel, image_padding[i:i+m1, j:j+n1]).sum()

    return img_out.astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('img.png', cv.IMREAD_GRAYSCALE)
    cv.imshow('Imagem inicial', img)

    # Gaussiana 
    k_gaussian = (1/16)*np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]])

    filter_gaussian = convolution(img, k_gaussian)
    cv.imshow('Gaussian', filter_gaussian)

    # Derivada gaussiana 
    k_derivada_gaussian = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
    filter_derivate_gaussian = convolution(img, k_derivada_gaussian)
    cv.imshow('Derivate Gaussian', filter_derivate_gaussian)


    # Laplacian 
    k_laplacian = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
    filter_laplacian = convolution(img, k_laplacian)
    cv.imshow('Laplacian', filter_laplacian)

    # Sobel X 
    k_sobelX = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    filter_sobelX = convolution(img, k_sobelX)
    cv.imshow('Horizontal Sobel', filter_sobelX)

    # Sobel Y 
    k_sobelY = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    filter_sobelY = convolution(img, k_sobelY)
    cv.imshow('Vertical Sobel', filter_sobelY)
    

    cv.waitKey(0)
    cv.destroyAllWindows()