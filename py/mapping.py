import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d, Rbf
from skimage.transform import warp, ProjectiveTransform


def interpolateContours(contour1, contour2, output_shape, input_image):
    # create a meshgrid with the shape of the output image
    y, x = np.meshgrid(np.arange(output_shape[2]), np.arange(output_shape[1]))
    grid = np.column_stack((x.flatten(), y.flatten()))

    # find the corresponding points in contour2 using thin-plate spline interpolation
    rbf = Rbf(contour1[:, 0], contour1[:, 1], contour2[:, 0], function="thin-plate")
    interpolated = rbf(grid[:, 0], grid[:, 1])

    # reshape interpolated values back into the shape of the output image
    warped_interpolated = np.reshape(
        interpolated, (output_shape[1], output_shape[2], 2)
    )

    # calculate the transformation matrix using the spline
    src = contour2.astype(np.float32)
    dst = contour1.astype(np.float32)
    params = ProjectiveTransform().estimate(dst, src)

    # warp the input image using the transformation matrix
    warped_image = np.zeros(output_shape, dtype=np.float32)
    for c in range(output_shape[0]):
        warped_image[c] = warp(
            input_image[c],
            params.inverse,
            output_shape=(output_shape[1], output_shape[2]),
        )

    return warped_image


def estimateAffineTransform(contour1, contour2):
    # Estimate the affine transform between two contours using mean squared error (MSE) as the error function.

    # Define the function that computes the sum of squared errors between the transformed contour1 and contour2.
    def error_func(params):
        a, b, tx, c, d, ty = params
        transform = np.array([[a, b, tx], [c, d, ty], [0, 0, 1]])
        transformed_points = np.array(
            [transform.dot(np.append(point, 1))[:2] for point in contour1]
        )
        return np.mean((transformed_points - contour2) ** 2)

    # Estimate the affine transform using least squares optimization
    initial_guess = np.array([1, 0, 0, 0, 1, 0])
    result = minimize(
        error_func,
        initial_guess,
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 1000},
    )
    transform = np.array(
        [
            [result.x[0], result.x[1], result.x[2]],
            [result.x[3], result.x[4], result.x[5]],
            [0, 0, 1],
        ]
    )

    return transform


if __name__ == "__main__":
    # Test the mapping function
    import matplotlib.pyplot as plt
    import cv2

    image = cv2.imread(
        "C:\\Users\\Alec\\Projects\\belljar-testing\\r71\\dapi\\R71_s02.png"
    )
    atlas = cv2.imread(
        "C:\\Users\\Alec\\Projects\\belljar-testing\\r71_alignment\\atlas\\Atlas_R71_s02.png"
    )

    def getContour(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        largest = largest[:, 0, :]
        return largest

    def resampleContour(contour, N):
        # Compute the total length of the contour
        lengths = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        # Compute the cumulative length along the contour
        cum_lengths = np.concatenate(([0], np.cumsum(lengths)))

        # Compute the fraction of total length at each point
        fractions = cum_lengths / total_length

        # Create a linearly spaced array of fractions with N elements
        new_fractions = np.linspace(0, 1, N)

        # Interpolate the contour coordinates using the new fractions
        interp_func = interp1d(fractions, contour, axis=0, fill_value="extrapolate")
        new_contour = interp_func(new_fractions)

        return new_contour

    atlasContour = getContour(atlas)
    imageContour = getContour(image)

    atlasContour = cv2.convexHull(atlasContour.astype(np.float32))[:, 0, :]
    imageContour = cv2.convexHull(imageContour.astype(np.float32))[:, 0, :]

    if len(atlasContour) != len(imageContour):
        N = min(len(atlasContour), len(imageContour))
        atlasContour = resampleContour(atlasContour, N)
        imageContour = resampleContour(imageContour, N)

    # estimate the affine transform
    # warped = interpolateContours(atlasContour, imageContour, image.shape, image)
    M = cv2.estimateAffine2D(atlasContour, imageContour)[0]
    warped = cv2.warpAffine(
        atlas,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    # recolor warped red
    warped[:, :, 1:] = 0
    # Plot the warped on top of image
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imshow(warped, alpha=0.7)
    plt.show()
