import cv2 as cv



img = cv.imread('img.png')
cv.imshow('Casa', img)

# Grey image
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('cinza', grey_img)

# Gaussian blur

gaussian_img = cv.GaussianBlur(grey_img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Gaussian blur', gaussian_img)

# Sobel X e Y

sobelX_img = cv.Sobel(gaussian_img, cv.CV_64F, 1, 0, 5)
sobelY_img = cv.Sobel(gaussian_img, cv.CV_64F, 0, 1, 5)

cv.imshow('Sobel X', sobelX_img)
cv.imshow('Sobel Y', sobelY_img)

# Laplacian

laplacian = cv.Laplacian(gaussian_img, cv.CV_64F)
cv.imshow('Laplacian', laplacian)

# Canny 
canny = cv.Canny(gaussian_img, 50, 150)
cv.imshow('Canny', canny)


cv.waitKey(0)
cv.destroyAllWindows
