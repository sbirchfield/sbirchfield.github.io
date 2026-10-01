"""Quiz question banks, one list per lesson number. build.py generates quizzes/quizNN.html
from this file, and adds a "Take the Quiz" button to the corresponding lesson page.

Each question is a dict:
    question    - the question text (raw HTML; wrap code/identifiers in <code>...</code>)
    choices     - list of answer strings (same HTML rules)
    correct     - index into `choices` of the right answer
    explanation - one line shown after answering, regardless of right/wrong
"""

QUIZZES = {
    1: [
        {
            "question": "What are the three dimensions of a color image array, in order?",
            "choices": [
                "<code>(height, width, channels)</code>",
                "<code>(channels, height, width)</code>",
                "<code>(width, height, channels)</code>",
                "<code>(channels, width, height)</code>",
            ],
            "correct": 0,
            "explanation": "<code>array.shape</code> for a color image is <code>(height, width, channels)</code> -- rows first, then columns, then color channel.",
        },
        {
            "question": "In NumPy indexing <code>img[row, col]</code>, what does <code>row</code> correspond to?",
            "choices": [
                "x, the horizontal position, since row is just another name for column index",
                "y, the vertical position",
                "The color channel, since row indexes depth in a 3D array",
                "Nothing meaningful -- row and col are interchangeable in NumPy",
            ],
            "correct": 1,
            "explanation": "The first array axis is the row, which moves down the image -- that's y, not x.",
        },
        {
            "question": "<code>cv2.imread</code> loads a color image with channels in what order?",
            "choices": [
                "RGB, the same order Matplotlib and most other libraries use",
                "Grayscale by default, requiring an explicit color flag",
                "BGR",
                "Alphabetical order of the channel names",
            ],
            "correct": 2,
            "explanation": "OpenCV uses BGR order, the opposite of RGB that Matplotlib and most other libraries expect.",
        },
        {
            "question": "Passing a <code>cv2.imread</code>-loaded image straight to <code>plt.imshow</code> without converting typically produces what?",
            "choices": [
                "An identical image to what OpenCV shows, since both libraries assume the same channel order",
                "A grayscale image, since Matplotlib discards color information by default",
                "A runtime error, since Matplotlib refuses arrays it doesn't recognize as RGB",
                "An image with red and blue channels swapped",
            ],
            "correct": 3,
            "explanation": "Matplotlib interprets the array as RGB, so a BGR array renders with red and blue reversed.",
        },
    ],
    2: [
        {
            "question": "With NumPy's default <code>uint8</code> arithmetic, what happens when you add 80 to a pixel of value 220?",
            "choices": [
                "It wraps around to a smaller value (<code>300 mod 256 = 44</code>)",
                "It saturates at 255, the maximum value a <code>uint8</code> can hold",
                "It raises an <code>OverflowError</code> and halts execution",
                "NumPy automatically upgrades the array to a larger integer type",
            ],
            "correct": 0,
            "explanation": "<code>uint8</code> wraps around like an odometer: <code>220 + 80 = 300</code>, which overflows to <code>300 mod 256 = 44</code>.",
        },
        {
            "question": "What does \"saturating arithmetic\" (as used by <code>cv2.add</code>) do differently from plain NumPy <code>uint8</code> addition?",
            "choices": [
                "It wraps around past 255 back to 0, exactly like plain NumPy's <code>uint8</code> addition",
                "It clamps results to the valid range instead of wrapping",
                "It silently converts both inputs to 32-bit floating point before adding",
                "It behaves identically to NumPy's default <code>uint8</code> addition, with no special handling",
            ],
            "correct": 1,
            "explanation": "<code>cv2.add</code> clamps (saturates) at 0 and 255 rather than silently wrapping around.",
        },
        {
            "question": "Why does difference imaging typically take the absolute value of the subtraction?",
            "choices": [
                "Because <code>cv2.subtract</code> would raise an error if it ever produced a negative pixel value",
                "To artificially increase contrast so edges stand out more clearly in the result",
                "So the result is meaningful regardless of which image is brighter at a given pixel",
                "Absolute value isn't actually used anywhere in difference imaging",
            ],
            "correct": 2,
            "explanation": "Without <code>abs()</code>, a pixel that got darker and one that got brighter by the same amount would look different, even though both changed.",
        },
        {
            "question": "What does <code>cv2.addWeighted(a, alpha, b, beta, gamma)</code> compute?",
            "choices": [
                "<code>(a + b) / (alpha + beta) + gamma</code>",
                "<code>alpha*a - beta*b + gamma</code>",
                "<code>max(alpha*a, beta*b) + gamma</code>",
                "<code>a*alpha + b*beta + gamma</code>",
            ],
            "correct": 3,
            "explanation": "It's a saturating weighted sum of the two images plus a constant offset <code>gamma</code>.",
        },
    ],
    3: [
        {
            "question": "What does thresholding do to a grayscale image?",
            "choices": [
                "Converts every pixel above a cutoff to white and at-or-below to black (or vice versa)",
                "Blurs the image with a low-pass filter to remove high-frequency noise, independent of any single pixel value",
                "Converts the image into a different color space such as HSV, preserving all original brightness information",
                "Computes a histogram of pixel intensities across the whole image, without modifying any pixel",
            ],
            "correct": 0,
            "explanation": "Thresholding is a per-pixel comparison against a cutoff value, producing a binary image.",
        },
        {
            "question": "What criterion does Otsu's method use to automatically choose a threshold?",
            "choices": [
                "It picks the single pixel value closest to the image's median intensity",
                "It minimizes within-class variance (equivalently, maximizes between-class variance)",
                "It always uses a fixed threshold of 128, regardless of the image",
                "It searches for the threshold that maximizes the number of white pixels",
            ],
            "correct": 1,
            "explanation": "Otsu searches every possible cutoff for the one that best separates the histogram into two low-variance classes.",
        },
        {
            "question": "What effect does erosion have on a white region in a binary image?",
            "choices": [
                "Grows the white region outward, filling in small holes near its boundary",
                "Leaves the region's size unchanged, only smoothing its edges",
                "Shrinks it, removing small specks",
                "Inverts black and white everywhere in the image",
            ],
            "correct": 2,
            "explanation": "Erosion only keeps a pixel white if the entire structuring element fits inside the white region, so regions shrink.",
        },
        {
            "question": "What is \"opening\" (erosion followed by dilation) typically used for?",
            "choices": [
                "Filling in small black holes that appear inside an otherwise solid white region",
                "Increasing the overall contrast of the image before thresholding",
                "Detecting edges by computing the image gradient magnitude",
                "Removing small white specks while restoring the size of larger regions",
            ],
            "correct": 3,
            "explanation": "Erosion first wipes out small specks entirely; the following dilation grows the surviving (larger) regions back toward their original size.",
        },
    ],

    4: [
        {
            "question": "What is the main difference between flood fill and connected-component labeling?",
            "choices": [
                "Flood fill grows a region from a single seed point; connected components find all separate regions in the image at once",
                "They do exactly the same thing under the hood, just exposed through different function names and documentation",
                "Connected components only work correctly on color images, never grayscale, due to a hard OpenCV restriction",
                "Flood fill labels every blob in the image with a unique ID simultaneously, without a seed, in a single pass",
            ],
            "correct": 0,
            "explanation": "Flood fill needs one seed per region, while <code>cv2.connectedComponentsWithStats</code> scans the whole image and labels every blob at once.",
        },
        {
            "question": "In the classic stack-based <code>flood_fill_stack</code> algorithm, what happens when a pixel is popped from the stack?",
            "choices": [
                "It's colored immediately and pushed back onto the stack, regardless of whether it was visited before",
                "If it's foreground and not yet filled, it's marked filled and its 4-connected neighbors are pushed onto the stack",
                "The algorithm terminates immediately, since popping a pixel always means the fill is complete",
                "It's deleted from the image array and treated as background from then on",
            ],
            "correct": 1,
            "explanation": "Popped pixels that are background or already filled are simply skipped; otherwise they're marked filled and their neighbors are added to the frontier.",
        },
        {
            "question": "What does <code>cv2.connectedComponentsWithStats</code> label as component <code>0</code>?",
            "choices": [
                "The largest blob in the image by pixel area",
                "The smallest blob in the image by pixel area",
                "The background",
                "An error code meaning no blobs were found in the image",
            ],
            "correct": 2,
            "explanation": "Label 0 is always the background, so the number of actual blobs is <code>num_labels - 1</code>.",
        },
        {
            "question": "What extra information does <code>cv2.connectedComponentsWithStats</code> provide alongside the labels, at almost no extra computational cost?",
            "choices": [
                "A trained classifier that predicts each blob's shape category",
                "The Hu moments of each blob, which are rotation- and scale-invariant",
                "The original unthresholded grayscale image, unchanged",
                "Bounding box, area, and centroid for each blob",
            ],
            "correct": 3,
            "explanation": "The lesson notes these stats come essentially free as part of the same labeling pass.",
        },
    ],
    5: [
        {
            "question": "What does the raw moment <code>m00</code> represent for a binary image?",
            "choices": [
                "The area (foreground pixel count)",
                "The x-coordinate of the shape's centroid",
                "The shape's orientation angle in radians",
                "Always zero, by definition, for any binary image",
            ],
            "correct": 0,
            "explanation": "<code>m00 = sum I(x,y)</code> over all pixels, which for a binary image is just a count of foreground pixels.",
        },
        {
            "question": "Why are central moments <code>mu_pq</code> preferred over raw moments <code>m_pq</code> for describing a shape's spread?",
            "choices": [
                "Central moments are always integers, unlike raw moments which can be fractional",
                "Central moments are measured around the shape's own centroid",
                "Central moments are faster to compute than raw moments for large images",
                "There is no real difference between them; the two terms are used interchangeably",
            ],
            "correct": 1,
            "explanation": "Raw moments (except <code>m00</code>) depend on where the shape sits in the image; central moments are translation-invariant.",
        },
        {
            "question": "What property makes Hu moments useful for comparing shapes?",
            "choices": [
                "They uniquely determine the shape's color, independent of its geometry",
                "They only work correctly on convex shapes, failing on concave ones",
                "They are (nearly) invariant to the shape's translation, scale, and rotation",
                "They are always exactly equal to 1, regardless of the shape",
            ],
            "correct": 2,
            "explanation": "The lesson shows a translated and a rotated+scaled version of the same shape produce similar Hu moments, while a different shape does not.",
        },
        {
            "question": "The orientation formula <code>theta = 0.5 * atan2(2*mu11, mu20 - mu02)</code> is computed from which moments?",
            "choices": [
                "The raw moments <code>m10</code>, <code>m01</code>, which locate the centroid",
                "The seven Hu moments, used for shape matching across scale and rotation",
                "<code>m00</code> alone, the total foreground pixel count",
                "The central second-order moments <code>mu20</code>, <code>mu02</code>, <code>mu11</code>",
            ],
            "correct": 3,
            "explanation": "The second-order central moments describe the shape's spread around its centroid, from which the major-axis angle is derived.",
        },
    ],
    6: [
        {
            "question": "For a symmetric 2x2 matrix (like the covariance matrices built in this lesson), what special property do its eigenvectors have?",
            "choices": [
                "They are perpendicular to each other, with real eigenvalues",
                "They are always identical to each other, regardless of the matrix",
                "They are complex numbers whenever the matrix is symmetric",
                "There is only ever one eigenvector for a symmetric 2x2 matrix",
            ],
            "correct": 0,
            "explanation": "A symmetric 2x2 matrix always has two perpendicular (orthogonal) eigenvectors with real eigenvalues.",
        },
        {
            "question": "In the covariance matrix built from a blob's central moments, what do the eigenvectors represent geometrically?",
            "choices": [
                "The blob's three color channel intensities",
                "The blob's major and minor axis directions",
                "The blob's centroid (x, y) coordinates",
                "Random directions with no particular geometric meaning",
            ],
            "correct": 1,
            "explanation": "Diagonalizing the covariance matrix rotates into the frame where the blob's spread in x and y no longer mixes -- exactly its major/minor axes.",
        },
        {
            "question": "What does an eccentricity close to 0 indicate about a shape?",
            "choices": [
                "It's very elongated, like a thin line or needle shape",
                "It has one or more holes somewhere inside the shape",
                "It's nearly circular (major and minor axes about equal length)",
                "It's not a valid binary shape and the moments are undefined",
            ],
            "correct": 2,
            "explanation": "Eccentricity ranges from 0 (a circle) to nearly 1 (a very thin, elongated shape).",
        },
        {
            "question": "This lesson notes that eigendecomposing a covariance matrix of general (not necessarily image) data is known as what technique?",
            "choices": [
                "The Fourier transform, decomposing the data into sinusoids",
                "Otsu's method, applied to the data's covariance matrix",
                "The Hough transform, applied to the data's covariance matrix",
                "Principal component analysis (PCA)",
            ],
            "correct": 3,
            "explanation": "PCA is the same diagonalization operation applied to a covariance matrix of general data, ranking eigenvectors by eigenvalue as principal components.",
        },
    ],
    7: [
        {
            "question": "Which point-to-point distance metric produces diamond-shaped equal-distance contours?",
            "choices": [
                "Manhattan / city-block distance (<code>D4</code>)",
                "Chessboard distance (<code>D8</code>)",
                "Ordinary Euclidean distance",
                "The chamfer distance approximation",
            ],
            "correct": 0,
            "explanation": "Manhattan distance (<code>|dx| + |dy|</code>) forms diamonds; chessboard forms squares; Euclidean forms circles.",
        },
        {
            "question": "Why does the plain Freeman chain-code estimate <code>L = N_e + sqrt(2)*N_o</code> systematically overestimate a smooth curve's true length?",
            "choices": [
                "Because the chain-code formula double-counts every boundary pixel it visits, inflating the total twice over",
                "The digitized boundary zig-zags through many short staircase steps that add up to more than the true arc length",
                "Because the diagonal weight <code>sqrt(2)</code> used in the formula is too small to matter at all",
                "It actually underestimates the true length significantly, not overestimates it as commonly assumed",
            ],
            "correct": 1,
            "explanation": "A digitized circle's boundary alternates through many short zig-zag steps, which sum to more length than the smooth true arc.",
        },
        {
            "question": "What is the key computational advantage of the chamfer distance transform over an exact Euclidean distance transform?",
            "choices": [
                "It only works on already-thresholded color images, never grayscale, due to an internal OpenCV limitation",
                "It ignores background pixels entirely, computing distance purely from the foreground mask alone",
                "It approximates the result with two fast raster-scan passes",
                "It requires no distance function of any kind, relying instead on simple pixel counting heuristics",
            ],
            "correct": 2,
            "explanation": "The chamfer transform propagates local distances in two raster-scan passes, a fast approximation to the more expensive exact Euclidean distance transform.",
        },
        {
            "question": "What does the chessboard distance <code>D8</code> correspond to physically?",
            "choices": [
                "The ordinary straight-line (Euclidean) distance between two points",
                "The area of the smallest square enclosing both points",
                "The number of pawn moves needed to travel between two squares",
                "The number of king moves on a chessboard between two squares",
            ],
            "correct": 3,
            "explanation": "<code>D8 = max(|dx|, |dy|)</code> is the shortest path length when diagonal (8-connected) moves are allowed, like a chess king.",
        },
    ],
    8: [
        {
            "question": "In <code>cv2.flip(img, code)</code>, what does <code>code=1</code> do?",
            "choices": [
                "Flips the image horizontally (left-right)",
                "Flips the image vertically (top-bottom)",
                "Flips the image both horizontally and vertically",
                "Leaves the image completely unchanged",
            ],
            "correct": 0,
            "explanation": "<code>code=1</code> is a horizontal flip, <code>code=0</code> is vertical, and <code>code=-1</code> flips both.",
        },
        {
            "question": "Which of the three transform types (Euclidean, similarity, affine) is capable of shearing a square into a parallelogram?",
            "choices": [
                "Euclidean transforms only, since they include rotation",
                "Affine transforms, since they allow any invertible 2x2 matrix including shear",
                "Similarity transforms only, since they allow non-uniform scaling",
                "None of them can shear a square into a parallelogram",
            ],
            "correct": 1,
            "explanation": "Affine transforms allow any invertible 2x2 matrix, including shear, which neither Euclidean nor similarity transforms permit.",
        },
        {
            "question": "What does a similarity transform preserve that a general affine transform does not necessarily preserve?",
            "choices": [
                "Nothing -- they preserve exactly the same properties as each other",
                "Parallelism of lines, which general affine transforms can destroy",
                "Angles (and ratios of lengths) between corresponding line segments",
                "The exact pixel values at every location in the image",
            ],
            "correct": 2,
            "explanation": "Similarity transforms preserve angles and length ratios; affine transforms give up angle preservation in exchange for allowing shear.",
        },
        {
            "question": "How many point correspondences does <code>cv2.getAffineTransform</code> require to solve for all 6 affine parameters?",
            "choices": ["2 point correspondences", "4 point correspondences", "6 point correspondences", "3 point correspondences"],
            "correct": 3,
            "explanation": "3 point correspondences give exactly 6 equations, matching the 6 free parameters of an affine transform.",
        },
    ],

    9: [
        {
            "question": "What fundamental problem does forward mapping (pushing each source pixel to its transformed location) have?",
            "choices": [
                "It can leave gaps (holes) in the destination image with no assigned value",
                "It is too slow to run on large images in practice",
                "It only works correctly for translations, not rotations",
                "It requires the transform to be invertible, unlike inverse mapping",
            ],
            "correct": 0,
            "explanation": "When a transform enlarges or rotates an image, destination locations spread out and some destination pixels receive no source pixel at all.",
        },
        {
            "question": "Why does inverse mapping avoid the holes problem that forward mapping has?",
            "choices": [
                "It only works correctly on grayscale images, never color, due to how OpenCV handles channels",
                "It iterates over a complete destination grid and looks up each pixel's source location",
                "It rounds all coordinates to the nearest integer before transforming, discarding fractional detail",
                "It skips pixels near the image border entirely, to avoid any out-of-bounds lookups during mapping",
            ],
            "correct": 1,
            "explanation": "Since every destination pixel is visited and mapped back via <code>M_inv</code>, every output pixel is guaranteed to get a value.",
        },
        {
            "question": "Why does inverse mapping need interpolation at all?",
            "choices": [
                "Because the inverse matrix is not always defined for every possible transform matrix",
                "Because OpenCV requires interpolation by default, even when it is not strictly necessary",
                "Because <code>M_inv</code> applied to a destination pixel almost never lands exactly on an integer source coordinate",
                "Because the source image is always smaller in size than the destination image canvas",
            ],
            "correct": 2,
            "explanation": "The mapped-back coordinate is generally fractional, so a value must be estimated from the surrounding source pixels.",
        },
        {
            "question": "What is the key trade-off between nearest-neighbor and bilinear interpolation?",
            "choices": [
                "Nearest-neighbor is smoother but runs slower than bilinear interpolation",
                "Bilinear interpolation is blockier but runs faster than nearest-neighbor",
                "There is no real difference between them for enlarging transforms",
                "Nearest-neighbor is blocky/pixelated; bilinear is smoother but slightly blurrier",
            ],
            "correct": 3,
            "explanation": "Bilinear blends the 4 neighboring pixels, removing the blocky aliasing nearest-neighbor produces at the cost of a softer image.",
        },
    ],
    10: [
        {
            "question": "What is the key difference between true convolution and correlation?",
            "choices": [
                "Convolution flips the kernel 180 degrees before sliding it",
                "Correlation only works correctly on grayscale images, never color",
                "Convolution requires a square kernel; correlation does not",
                "There is no difference; the terms are interchangeable in all cases",
            ],
            "correct": 0,
            "explanation": "Convolution flips the kernel before sliding; for symmetric kernels this makes no difference, but for asymmetric kernels (like Sobel) it does.",
        },
        {
            "question": "Does <code>cv2.filter2D</code> compute convolution or correlation?",
            "choices": [
                "Convolution, flipping the kernel before sliding it across the image",
                "Correlation, despite being casually called \"convolution\" in most image libraries",
                "Neither -- it actually computes a Fourier transform internally",
                "It alternates between the two depending on the kernel's size",
            ],
            "correct": 1,
            "explanation": "Most image libraries, including OpenCV's <code>cv2.filter2D</code>, actually implement correlation despite being casually called \"convolution\".",
        },
        {
            "question": "What does it mean for a 2D kernel to be separable?",
            "choices": [
                "It can only be applied to separate color channels, never to grayscale",
                "It must always be split into two passes, regardless of its internal structure",
                "It can be written as the outer product of two 1D kernels, turning O(n^2) work into O(2n)",
                "It has no measurable effect on the resulting image",
            ],
            "correct": 2,
            "explanation": "A separable kernel like the Gaussian can be applied as a 1D pass along x then a 1D pass along y, which is much cheaper than the full 2D kernel.",
        },
        {
            "question": "In the Sobel edge-detection example, why does the convolution-vs-correlation distinction actually matter?",
            "choices": [
                "Because Sobel only works correctly with correlation, never true convolution",
                "Because the Sobel kernel is symmetric, so flipping it has no visible effect",
                "It never matters for any kernel, symmetric or not",
                "Because the Sobel kernel is asymmetric",
            ],
            "correct": 3,
            "explanation": "For symmetric kernels (box, Gaussian) flipping changes nothing, but asymmetric kernels like Sobel give different results under convolution vs. correlation.",
        },
    ],
    11: [
        {
            "question": "What artifact can occur when an image is downsampled by simply keeping every k-th pixel, without blurring first?",
            "choices": [
                "Aliasing -- fine periodic detail folds into a fake low-frequency pattern",
                "The image becomes larger instead of smaller after the operation",
                "Color channels get swapped unpredictably during subsampling",
                "The image is unaffected as long as the step size k is even",
            ],
            "correct": 0,
            "explanation": "Fine regular detail (like brick mortar lines) that oscillates faster than the new pixel spacing can represent aliases into noisy-looking fake patterns.",
        },
        {
            "question": "Why does blurring before downsampling prevent aliasing?",
            "choices": [
                "It randomly discards pixels instead of following a fixed, predictable sampling pattern",
                "It removes fine detail before subsampling, rather than randomly keeping or dropping it pixel by pixel",
                "It increases the image's overall contrast before subsampling, independent of frequency content",
                "It converts the image to grayscale before subsampling, discarding all color information",
            ],
            "correct": 1,
            "explanation": "Smoothing away fine detail before subsampling avoids the fold-in effect that causes aliasing.",
        },
        {
            "question": "What happens if the Gaussian kernel size is too small for the requested <code>sigma</code>?",
            "choices": [
                "OpenCV raises an error and refuses to construct the kernel at all, halting execution",
                "The image ends up blurred far more strongly than the requested sigma value specifies",
                "The kernel is silently reshaped into something close to a plain box average",
                "Nothing changes -- kernel size has no real relationship to sigma in OpenCV's implementation",
            ],
            "correct": 2,
            "explanation": "A too-small kernel can't represent a wide bell curve, so <code>cv2.getGaussianKernel</code> renormalizes it into something close to a box average.",
        },
        {
            "question": "What is a Gaussian pyramid?",
            "choices": [
                "A single blurred copy of an image, saved at one fixed resolution with no further processing",
                "A histogram of pixel intensities computed across multiple scales of the same image",
                "A stack of images at increasing, rather than decreasing, resolution level by level",
                "A stack of progressively smaller, blurrier versions made by repeated blurring and downsampling",
            ],
            "correct": 3,
            "explanation": "Each pyramid level is a properly anti-aliased, half-size version of the previous level, produced by blur-then-downsample.",
        },
    ],
    12: [
        {
            "question": "Why do Sobel and Prewitt kernels combine a difference in one direction with smoothing in the perpendicular direction?",
            "choices": [
                "To make the operator robust to noise, unlike a plain [-1, 0, 1] difference",
                "To make the resulting kernel square in shape",
                "To increase the image's spatial resolution before differentiating",
                "It's purely a historical convention with no practical effect on noise",
            ],
            "correct": 0,
            "explanation": "Averaging over rows/columns while differencing cuts down the noise response substantially compared to a bare difference kernel.",
        },
        {
            "question": "In the Canny edge detector, what is the purpose of non-maximum suppression?",
            "choices": [
                "To smooth the image with a Gaussian kernel before differentiation, reducing noise sensitivity",
                "To thin wide gradient ridges down to single-pixel-wide lines",
                "To discard every pixel that falls below the high threshold immediately, with no further checks",
                "To convert the image to a binary mask before computing gradients at every pixel",
            ],
            "correct": 1,
            "explanation": "Non-maximum suppression keeps a pixel's gradient magnitude only if it's a local maximum along the gradient direction, producing thin edges.",
        },
        {
            "question": "What does Canny's hysteresis thresholding do with pixels between the low and high threshold?",
            "choices": [
                "Discards them all immediately, regardless of their neighbors or local connectivity",
                "Always keeps them, regardless of connectivity to a strong edge, as genuine edges",
                "Keeps them only if they connect to a definite (above-high-threshold) edge pixel",
                "Converts them to the average of the two threshold values, treating them as uncertain",
            ],
            "correct": 2,
            "explanation": "Hysteresis links up weak-but-real edge segments that connect to strong edges, while suppressing isolated noise responses.",
        },
        {
            "question": "In the Hough transform, what does it mean when several edge points' sinusoids in (rho, theta) space all cross at the same point?",
            "choices": [
                "Those edge points are noise and should be discarded from the accumulator entirely",
                "The image contains a circle rather than a line at that particular location",
                "The accumulator has numerically overflowed at that specific bin location",
                "Those edge points are collinear, and the crossing point gives the shared line's parameters",
            ],
            "correct": 3,
            "explanation": "Each edge point traces a sinusoid of all lines through it; collinear points' sinusoids intersect at the (rho, theta) of the line they share.",
        },
    ],
    13: [
        {
            "question": "Where does an edge appear in the second derivative of image intensity, as opposed to the first derivative?",
            "choices": [
                "At a zero crossing -- where the second derivative swings from positive to negative or vice versa",
                "At a sharp peak, the same location as in the first derivative of intensity",
                "Second derivatives cannot be used to find edges at all, by mathematical definition",
                "At the maximum value of the original image intensity, regardless of derivative",
            ],
            "correct": 0,
            "explanation": "The first derivative peaks at an edge; the second derivative crosses zero exactly at the edge location.",
        },
        {
            "question": "Why does the Marr-Hildreth detector smooth the image with a Gaussian before applying the Laplacian?",
            "choices": [
                "To increase the overall contrast of the image before differentiating it further",
                "Because a second derivative amplifies high-frequency noise even more",
                "Smoothing is not actually part of the Marr-Hildreth method at all, by design",
                "To make the image square before applying the Laplacian operator to it",
            ],
            "correct": 1,
            "explanation": "Marr and Hildreth's fix is to smooth first, then differentiate -- equivalent to convolving with a single combined Laplacian-of-Gaussian kernel.",
        },
        {
            "question": "What does a Difference of Gaussians (DoG) approximate?",
            "choices": [
                "The image histogram, recomputed at two different blur scales",
                "The Sobel gradient magnitude, computed at two different blur scales",
                "A scaled Laplacian of Gaussian (LoG), at much cheaper computational cost",
                "The Hough accumulator, computed from two blurred copies of the image",
            ],
            "correct": 2,
            "explanation": "Subtracting two Gaussian blurs at different scales approximates a scaled LoG using only two blurs and a subtraction, instead of an explicit second-derivative kernel.",
        },
        {
            "question": "What does a Laplacian pyramid store at each level that a Gaussian pyramid does not, allowing exact reconstruction?",
            "choices": [
                "A copy of the original image at full resolution, stored unchanged",
                "A histogram of each pyramid level's intensity distribution",
                "Nothing extra -- Laplacian and Gaussian pyramids store identical information",
                "The difference between a Gaussian level and the coarser level upsampled back to match it",
            ],
            "correct": 3,
            "explanation": "Each Laplacian level captures exactly what blurring/downsampling would discard, so summing back up the pyramid reconstructs the original exactly.",
        },
    ],

    14: [
        {
            "question": "Why is a median filter especially effective against salt-and-pepper noise, compared to Gaussian blur?",
            "choices": [
                "A single extreme outlier barely moves the median, while it drags the mean toward it",
                "The median filter is linear, so it averages noise away more precisely",
                "Median filtering only works correctly on binary images, never grayscale",
                "Gaussian blur cannot process 8-bit images without converting first",
            ],
            "correct": 0,
            "explanation": "A wildly wrong outlier pixel skews a mean a lot but barely affects a median, since the median just picks the middle-ranked value.",
        },
        {
            "question": "On a grayscale image, what operation does erosion correspond to in a neighborhood?",
            "choices": [
                "Taking the maximum value in the neighborhood",
                "Taking the minimum value in the neighborhood",
                "Taking the mean value in the neighborhood",
                "Taking the median value in the neighborhood",
            ],
            "correct": 1,
            "explanation": "Grayscale erosion generalizes binary erosion as a min filter: each pixel becomes the minimum value in its neighborhood.",
        },
        {
            "question": "In the bilateral filter formula, what are the two factors each neighboring pixel is weighted by?",
            "choices": [
                "Its row and column distance from the center pixel, treated independently",
                "Its intensity value and its position in the image (top half vs. bottom half)",
                "How spatially close it is, and how similar its intensity is to the center pixel",
                "Only how similar its intensity is to the center pixel, ignoring position entirely",
            ],
            "correct": 2,
            "explanation": "The bilateral filter combines a spatial Gaussian (<code>G_sigma_s</code>) with a range/intensity Gaussian (<code>G_sigma_r</code>), so only neighbors that are both nearby and similar in intensity get high weight.",
        },
        {
            "question": "Why does the bilateral filter preserve a step edge while Gaussian blur smooths across it?",
            "choices": [
                "The bilateral filter uses a larger spatial radius than Gaussian blur",
                "The bilateral filter only operates correctly on binary images, never grayscale",
                "Gaussian blur always ignores pixel intensity entirely, by design",
                "Pixels across the edge have very different intensity",
            ],
            "correct": 3,
            "explanation": "Even though pixels on the other side of an edge may be spatially close, their intensity difference makes the range weight near zero, so they contribute almost nothing to the output.",
        },
    ],
    15: [
        {
            "question": "What does the 2D Fourier transform decompose an image into?",
            "choices": [
                "A sum of 2D sinusoidal gratings at different frequencies and orientations",
                "A set of non-overlapping rectangular blocks, like JPEG's 8x8 grid",
                "A pyramid of progressively blurred and downsampled copies",
                "A single average brightness value computed per row",
            ],
            "correct": 0,
            "explanation": "The 2D DFT re-expresses the image as a sum of sinusoidal gratings, each with its own frequency, orientation, magnitude, and phase.",
        },
        {
            "question": "Why does a real-valued sinusoidal grating produce two bright spots (not one) in its Fourier spectrum?",
            "choices": [
                "Because of accumulated numerical rounding error in the FFT implementation",
                "A real sine wave is a sum of two complex exponentials at +f and -f, by Euler's formula",
                "Because the image was captured using two separate cameras",
                "<code>cv2</code> always duplicates spectrum peaks purely for display purposes",
            ],
            "correct": 1,
            "explanation": "By Euler's formula, a real sinusoid decomposes into two complex exponentials at frequencies +f and -f, so its spectrum shows two symmetric spikes.",
        },
        {
            "question": "What causes the visible ringing artifacts when applying an ideal (sharp-cutoff) low-pass filter in the frequency domain?",
            "choices": [
                "The image was not converted to grayscale before transforming",
                "<code>np.fft.fft2</code> always introduces significant rounding artifacts",
                "A sharp frequency cutoff corresponds to convolving with a spatial sinc kernel",
                "The frequency-domain mask was not centered correctly before multiplying",
            ],
            "correct": 2,
            "explanation": "By the convolution theorem, an ideal sharp cutoff in frequency corresponds to a sinc function in space, whose infinite ripples cause the Gibbs phenomenon.",
        },
        {
            "question": "According to the convolution theorem, multiplying two images' Fourier transforms elementwise and inverse-transforming is equivalent to what operation in the spatial domain?",
            "choices": [
                "Cropping the two images down to the same shared size",
                "Subtracting the two images from each other, pixel by pixel",
                "Stacking the two images together as separate channels",
                "Convolving the two images in the spatial domain",
            ],
            "correct": 3,
            "explanation": "The convolution theorem states that convolution in the spatial domain equals elementwise multiplication in the frequency domain, and vice versa.",
        },
    ],
    16: [
        {
            "question": "What does the Fourier transform (Lesson 15) fail to tell you, that wavelets and Gabor filters are designed to capture?",
            "choices": [
                "Which frequencies are present, without knowing where in the image they occur",
                "The overall average brightness of the image, summed across every pixel",
                "The image's color channel values at each individual pixel location",
                "The size of the image in pixels, both its width and its height",
            ],
            "correct": 0,
            "explanation": "Sine wave basis functions extend across the whole image, so the Fourier transform loses spatial localization; wavelets and Gabor filters use spatially localized oscillations instead.",
        },
        {
            "question": "In the Haar wavelet transform, what do the approximation (<code>a_k</code>) and detail (<code>d_k</code>) coefficients represent?",
            "choices": [
                "The maximum and minimum values of each pair of samples",
                "A local average and a local difference of each pair of samples",
                "The real and imaginary parts of a single Fourier coefficient",
                "Two unrelated random projections of the input signal",
            ],
            "correct": 1,
            "explanation": "<code>a_k</code> is the (scaled) sum of a sample pair, i.e. a local average; <code>d_k</code> is the (scaled) difference, i.e. a local detail.",
        },
        {
            "question": "In a one-level 2D wavelet decomposition of an image, what does the LH subband respond to?",
            "choices": [
                "A half-resolution blurred copy of the whole image",
                "Vertical edges, since it is high-pass along rows and low-pass along columns",
                "Horizontal edges, since it is low-pass along rows and high-pass along columns",
                "Diagonal detail and corners, since it is high-pass in both directions",
            ],
            "correct": 2,
            "explanation": "LH means low-pass along rows, high-pass along columns, which makes it sensitive to horizontal edges.",
        },
        {
            "question": "What does a Gabor filter's response tell you when the filter is tuned to one orientation and applied to bars drawn at several different orientations?",
            "choices": [
                "It responds equally to all orientations, regardless of its tuning",
                "It only responds to bars drawn in color, never in grayscale",
                "It responds most strongly to the orientation perpendicular to its own tuning",
                "It responds most strongly to the bar whose orientation matches the filter's tuning",
            ],
            "correct": 3,
            "explanation": "The response matrix in the lesson is diagonal-dominant: each filter's peak response lands on the bar matching its own tuned orientation, demonstrating orientation selectivity.",
        },
    ],
    17: [
        {
            "question": "What kind of image content does run-length encoding (RLE) compress well?",
            "choices": [
                "Images with large flat regions of constant value",
                "Photographs with fine-grained noise throughout the frame",
                "Any image whatsoever, regardless of its content",
                "Only images that have already been JPEG-compressed",
            ],
            "correct": 0,
            "explanation": "RLE stores (value, run length) pairs for consecutive identical pixels, so it shines on flat regions but does nothing useful when pixel values rarely repeat, as in noisy photographs.",
        },
        {
            "question": "What does Shannon entropy represent in the context of lossless image compression?",
            "choices": [
                "The exact number of bits Huffman coding will always achieve, with no exceptions",
                "The theoretical floor on average bits/pixel for any code based only on the value distribution",
                "The maximum possible file size the image could have after compression",
                "The number of unique pixel values present anywhere in the image",
            ],
            "correct": 1,
            "explanation": "Entropy H = -sum(p_i * log2(p_i)) is the theoretical minimum average bits/pixel achievable by any prefix code based on the symbol probabilities; Huffman coding gets close to, but does not always exactly reach, this bound.",
        },
        {
            "question": "In JPEG compression, what is the main effect of quantizing (dividing and rounding) the DCT coefficients of an 8x8 block?",
            "choices": [
                "It converts the image to grayscale before any further JPEG processing occurs",
                "It increases the spatial resolution of the image block beyond its original size",
                "It sets many coefficients, especially high-frequency ones, to exactly zero",
                "It removes the need for any Huffman coding step afterward entirely",
            ],
            "correct": 2,
            "explanation": "Quantization divides coefficients by values from a quantization table and rounds, zeroing out many low-energy (mostly high-frequency) coefficients so they cost almost nothing to store.",
        },
        {
            "question": "Why is JPEG generally a poor choice for compressing graphics like screenshots or text, compared to PNG?",
            "choices": [
                "JPEG cannot encode images containing any sharp edges at all",
                "PNG is actually always a lossy format, which is why it looks better",
                "JPEG only supports grayscale images, never full color",
                "Sharp edges spread energy across all DCT frequencies",
            ],
            "correct": 3,
            "explanation": "A sharp edge's energy is spread across every DCT frequency (like the cross pattern in a Fourier spectrum), so the same quantization that photos tolerate produces visible ringing and blocking artifacts around graphics' edges and text.",
        },
    ],
    18: [
        {
            "question": "Why does the lesson prefer <code>Y = 0.299 R + 0.587 G + 0.114 B</code> over a plain average <code>(R + G + B)/3</code> for grayscale conversion?",
            "choices": [
                "The human visual system is more sensitive to green",
                "The plain average is not mathematically valid for 8-bit image arrays",
                "The plain average only produces correct results on already-grayscale images",
                "There is no real difference between the two formulas in practice",
            ],
            "correct": 0,
            "explanation": "A plain average can't tell green and blue apart perceptually; weighting toward green (as human vision does) gives a much better brightness estimate.",
        },
        {
            "question": "In HSV, which channel should you threshold on to segment an object by color robustly, even under a lighting gradient across its surface?",
            "choices": [
                "Value, since it captures brightness directly",
                "Hue, since it captures which color something is, independent of brightness",
                "Saturation, since it captures how vivid or washed-out the color is",
                "All three channels equally, averaged together",
            ],
            "correct": 1,
            "explanation": "Hue captures which color something is, independent of how bright or washed-out it appears, so it stays consistent even as brightness varies across the object.",
        },
        {
            "question": "Why does video/JPEG compression discard resolution from the chroma channels (<code>Cb</code>, <code>Cr</code>) more aggressively than from luma (<code>Y</code>)?",
            "choices": [
                "Chroma channels take up more storage space than luma by default, before any subsampling",
                "Luma cannot be compressed at all under any circumstances, unlike chroma",
                "The human visual system resolves fine spatial detail in luma far better than in chroma",
                "<code>Cb</code> and <code>Cr</code> are simply redundant copies of <code>Y</code>, carrying no new information",
            ],
            "correct": 2,
            "explanation": "Chroma subsampling exploits the fact that losing the same amount of detail is far less objectionable in color than in brightness.",
        },
        {
            "question": "Why was the L*a*b* color space designed the way it was?",
            "choices": [
                "So that it uses noticeably less memory per pixel than RGB requires",
                "So that converting to grayscale is faster computationally than from RGB",
                "So that hue and brightness channels are always numerically identical to each other",
                "So that Euclidean distance between two Lab colors approximates perceived color difference",
            ],
            "correct": 3,
            "explanation": "Equal RGB distances can correspond to wildly different perceived differences depending on where they fall in color space; Lab was designed so Euclidean distance tracks perception much more consistently.",
        },
    ],
    19: [
        {
            "question": "What is the key difference between k-means and a Gaussian mixture model (GMM)?",
            "choices": [
                "GMM allows elliptical clusters and soft assignment; k-means only allows round, hard-assigned clusters",
                "k-means always outperforms GMM on every dataset, regardless of cluster shape or size",
                "GMM cannot handle more than 2 clusters at once, unlike k-means which has no such limit",
                "There is no meaningful difference; they are the same algorithm under two different names",
            ],
            "correct": 0,
            "explanation": "GMM generalizes k-means by giving each cluster its own covariance shape and computing a probability of membership for every point, rather than a single hard label.",
        },
        {
            "question": "Why does k-means fail on two concentric rings, landing at chance accuracy?",
            "choices": [
                "k-means can only handle 3D data, not 2D data like the rings dataset used here",
                "k-means's update rule can only carve space into convex, center-based regions",
                "The rings dataset simply has too many points for k-means to process efficiently",
                "k-means requires labeled training data, which the rings dataset entirely lacks",
            ],
            "correct": 1,
            "explanation": "Assigning each point to its nearest center can only produce straight-line boundaries between clusters, which cannot separate one ring from another.",
        },
        {
            "question": "In DBSCAN, what makes a point a <code>core point</code>?",
            "choices": [
                "It is the single point closest to the true cluster center",
                "It has the highest density of any point in the entire dataset",
                "At least <code>min_samples</code> other points lie within distance <code>eps</code> of it",
                "It was simply the first point visited by the algorithm",
            ],
            "correct": 2,
            "explanation": "A core point is defined purely by local density: enough neighbors within a fixed radius, with no notion of a cluster center at all.",
        },
        {
            "question": "What can DBSCAN do that neither k-means nor GMM can, when clutter/noise points are added to a dataset?",
            "choices": [
                "Run strictly faster than k-means on any dataset, clutter or not",
                "Automatically discover the correct number of clusters every single time",
                "Guarantee a globally optimal clustering regardless of initialization",
                "Label a point as \"noise,\" rather than being forced to assign it to some cluster",
            ],
            "correct": 3,
            "explanation": "k-means and GMM assign every point to some cluster no matter what; DBSCAN has an explicit noise category for points that aren't part of any sufficiently dense region.",
        },
    ],

    20: [
        {
            "question": "The structure tensor <code>M</code> built from image gradients is exactly analogous to which earlier concept?",
            "choices": [
                "The moment covariance matrix from Lesson 6, built from gradients",
                "The Hough accumulator from Lesson 12, applied here to local gradient directions",
                "The DCT quantization table from Lesson 17, applied here to a local window",
                "The Laplacian pyramid from Lesson 13, applied here to a local window",
            ],
            "correct": 0,
            "explanation": "Both are 2x2 covariance-style matrices; Lesson 6 built one from pixel coordinates, this lesson builds the same kind of matrix from local image gradients.",
        },
        {
            "question": "What do both eigenvalues of the structure tensor being large indicate about a point?",
            "choices": [
                "It's in a flat, low-texture region with little gradient in any direction",
                "It's a corner -- gradient strong in every direction",
                "It's on a straight edge, with gradient strong in one direction only",
                "It's outside the valid bounds of the image entirely",
            ],
            "correct": 1,
            "explanation": "A corner has strong gradient in every direction, which is exactly what two large eigenvalues of M mean.",
        },
        {
            "question": "Why are Harris and Shi-Tomasi corner detectors NOT scale-invariant, motivating SIFT?",
            "choices": [
                "They only work correctly on grayscale images, never color",
                "They require manually labeled training data to operate at all",
                "They operate at a single, fixed window size",
                "They cannot run on images larger than 100x100 pixels",
            ],
            "correct": 2,
            "explanation": "A sharp corner becomes a gentle curve at a fixed window size when the image is zoomed out enough, so a fixed-scale detector misses it.",
        },
        {
            "question": "What does Lowe's ratio test do when matching SIFT descriptors?",
            "choices": [
                "It discards all matches farther than a fixed pixel distance from each other entirely",
                "It only keeps matches that land within the exact same color channel",
                "It requires the two images to be exactly the same pixel dimensions throughout",
                "It keeps a match only if the best candidate is meaningfully closer than the second-best candidate",
            ],
            "correct": 3,
            "explanation": "Comparing the best match's distance to the second-best's rejects ambiguous matches where two candidates are nearly equally good.",
        },
    ],
    21: [
        {
            "question": "What is the aperture problem?",
            "choices": [
                "Motion along an edge (rather than perpendicular to it) produces no visible local change",
                "A camera lens defect that specifically blurs fast-moving objects in low light",
                "The fact that optical flow only works correctly on grayscale video, never color",
                "A limitation that only appears in Horn-Schunck, never in Lucas-Kanade's formulation",
            ],
            "correct": 0,
            "explanation": "Looking through a small window at a moving edge, only the motion perpendicular to the edge is visible; motion along the edge is invisible locally.",
        },
        {
            "question": "In Lucas-Kanade, why does the flow estimate at a corner tend to be reliable while at an edge it is not?",
            "choices": [
                "Corners are always brighter in intensity than edges, regardless of gradient direction or local contrast",
                "The same matrix M (the structure tensor) must be well-conditioned to solve the flow system",
                "Edges genuinely move faster than corners in real video sequences, due to their one-dimensional structure",
                "Lucas-Kanade cannot process edge pixels at all, only true corners, by explicit design in the algorithm",
            ],
            "correct": 1,
            "explanation": "Solving the Lucas-Kanade system requires M to be invertible; a corner (two large eigenvalues) is well-conditioned, while an edge (one small eigenvalue) is ill-conditioned -- the aperture problem again.",
        },
        {
            "question": "Why does <code>cv2.calcOpticalFlowPyrLK</code> use an image pyramid and iterate, rather than solving once?",
            "choices": [
                "To save memory only, with no accuracy benefit whatsoever",
                "Because color images require multiple passes to process correctly",
                "Because the underlying linear (Taylor) approximation is only valid for small motions",
                "Pyramids are only used for display purposes, not for the actual flow computation",
            ],
            "correct": 2,
            "explanation": "Estimating coarsely on a small, blurry pyramid level first, then refining level by level, lets large motions be recovered even though the linear approximation only holds locally, one small step at a time.",
        },
        {
            "question": "What does the Horn-Schunck smoothness term let flow estimates do that Lucas-Kanade's purely local windows cannot?",
            "choices": [
                "Run strictly faster than Lucas-Kanade in every situation, regardless of image content",
                "Avoid the aperture problem entirely, even for a single isolated edge pixel with no neighbors",
                "Work without relying on any brightness constancy assumption at all, unlike every other method here",
                "Propagate reliable flow from edges into flat, textureless regions with no local information of their own",
            ],
            "correct": 3,
            "explanation": "The smoothness term couples every pixel's flow to its neighbors', letting information spread from regions with strong gradients into flat regions that have no data term of their own.",
        },
    ],
    22: [
        {
            "question": "Why does stereo matching on a rectified image pair only need a 1D search instead of a full 2D search?",
            "choices": [
                "Because rectification puts the corresponding point on the same row in both images",
                "Because stereo images are always captured in grayscale, never color, by camera design",
                "Because disparity is always exactly zero for a rectified pair, by definition of rectification",
                "Because block matching is inherently limited to searching in 1D, regardless of rectification",
            ],
            "correct": 0,
            "explanation": "Epipolar geometry for a rectified stereo pair guarantees the match lies on the same row, collapsing the search from 2D to a 1D scan.",
        },
        {
            "question": "According to <code>Z = f * B / d</code>, what happens to estimated depth as disparity <code>d</code> increases?",
            "choices": [
                "Depth increases proportionally as disparity increases, following a direct linear relationship",
                "Depth decreases -- nearby objects have large disparity, distant objects have small disparity",
                "Depth is completely unrelated to the disparity value, depending only on focal length",
                "Depth becomes negative once disparity exceeds the camera baseline distance",
            ],
            "correct": 1,
            "explanation": "Depth is inversely proportional to disparity: nearer surfaces shift more (larger d) and so have smaller Z.",
        },
        {
            "question": "What does the left-right consistency check do in stereo matching?",
            "choices": [
                "It converts the disparity map into color purely for visualization purposes, with no effect on accuracy",
                "It blurs the disparity map afterward to remove residual noise, smoothing over fine detail",
                "It matches both left-to-right and right-to-left, then keeps only pixels where the two results agree",
                "It simply doubles the disparity search range used for matching, without checking consistency",
            ],
            "correct": 2,
            "explanation": "A pixel that disagrees between the two matching directions by more than about a pixel is flagged as unreliable, which is especially effective at catching occlusions.",
        },
        {
            "question": "What is the trade-off of increasing the block-matching window size in stereo matching?",
            "choices": [
                "Larger windows are always strictly better, with no real downside, regardless of scene content",
                "Window size has essentially no effect on the final disparity result, only on runtime",
                "Larger windows only affect runtime speed, never the resulting disparity values at any pixel",
                "Larger windows are more robust to noise but blur across depth discontinuities",
            ],
            "correct": 3,
            "explanation": "A bigger window averages over more pixels (robust to noise) but mixes pixels from two different true depths near object edges (blurred boundaries); a smaller window has the opposite trade-off.",
        },
    ],
    23: [
        {
            "question": "Why does ordinary least squares (OLS) give a biased line fit when both x and y coordinates are noisy?",
            "choices": [
                "OLS measures error only vertically",
                "OLS can only be used with integer-valued pixel coordinates",
                "OLS requires at least 100 data points to produce a valid fit",
                "OLS always overestimates the slope, regardless of the noise direction",
            ],
            "correct": 0,
            "explanation": "Total least squares (TLS) measures perpendicular distance and treats both coordinates symmetrically, which is more appropriate when both x and y carry comparable noise.",
        },
        {
            "question": "Why does ordinary least-squares fitting break catastrophically under outliers?",
            "choices": [
                "Least squares cannot be computed at all once outliers are present -- it simply errors out immediately",
                "It minimizes squared residuals, so one far-away outlier dominates the total error",
                "Outliers only ever affect the intercept term, never the slope, regardless of their position",
                "Squaring residuals actually makes outliers contribute less, not more, by design",
            ],
            "correct": 1,
            "explanation": "Squaring a large residual makes it dominate the total cost, so the fit moves away from the correct points to reduce that one huge squared term.",
        },
        {
            "question": "What is the core strategy RANSAC uses to fit a model in the presence of outliers?",
            "choices": [
                "It averages every possible candidate model over the entire dataset at once, weighting by confidence",
                "It removes only the single most extreme data point and reruns least squares repeatedly",
                "It repeatedly picks the smallest possible random subset needed to define a candidate model",
                "It requires the user to manually label which points are outliers beforehand, before fitting",
            ],
            "correct": 2,
            "explanation": "RANSAC fits from minimal random samples and picks whichever candidate has the largest consensus set, then does a final least-squares refit on just the inliers.",
        },
        {
            "question": "According to the RANSAC iteration-count formula, why does fitting a homography (minimal sample size 4) require many more iterations than fitting a line (minimal sample size 2) at the same outlier rate?",
            "choices": [
                "Homographies are inherently slower to evaluate on a computer than lines",
                "Lines actually require more iterations than homographies, not fewer",
                "The iteration count does not depend on the minimal sample size at all",
                "The probability of drawing an all-inlier sample is <code>w^n</code>",
            ],
            "correct": 3,
            "explanation": "Since the chance of an all-inlier sample is w raised to the sample size, a larger minimal sample (4 vs. 2) makes an all-inlier draw far less likely, requiring many more trials for the same confidence.",
        },
    ],
    24: [
        {
            "question": "Why can a homography (3x3 matrix on homogeneous coordinates) represent translation, while a plain 2x2 matrix cannot?",
            "choices": [
                "Homogeneous coordinates add a third coordinate, letting the matrix shift points via the extra row/column",
                "3x3 matrices are always invertible, while 2x2 matrices are not, by their larger dimension",
                "Translation is fundamentally impossible to represent with any matrix, homogeneous or not, ever",
                "A homography only works correctly on images that have already been translated beforehand",
            ],
            "correct": 0,
            "explanation": "Representing a 2D point as (x, y, 1) lets a 3x3 matrix encode translation directly, unlike a 2x2 matrix acting on (x, y) alone.",
        },
        {
            "question": "What specifically causes a homography to destroy parallelism, unlike an affine transform?",
            "choices": [
                "Using floating-point coordinates instead of integer ones",
                "A nonzero bottom row <code>(g, h)</code> in the 3x3 matrix",
                "Applying the transform to the image more than once in sequence",
                "The mere presence of a nonzero translation term in the matrix",
            ],
            "correct": 1,
            "explanation": "An affine transform's bottom row is always (0, 0, 1); a homography's nonzero (g, h) introduces the division effect that makes parallel lines converge.",
        },
        {
            "question": "How does the point-at-infinity trick locate a vanishing point?",
            "choices": [
                "By averaging the endpoints of all visible line segments in the image, weighted equally",
                "By computing the centroid of the whole image, ignoring the lines entirely",
                "By applying the homography to the homogeneous point representing the lines' shared direction",
                "It only works for vertical lines, never for horizontal ones, due to a coordinate convention",
            ],
            "correct": 2,
            "explanation": "A point with zero third homogeneous coordinate represents a direction at infinity; transforming it by H lands exactly where the two parallel lines' images converge.",
        },
        {
            "question": "In what two situations does a single homography exactly relate two photographs of a scene?",
            "choices": [
                "Any two photos taken from any two camera positions, of any scene whatsoever, without restriction",
                "Only when the two photos are taken with the exact same camera settings and exposure",
                "Only when the photographed scene contains no straight lines at all, by construction",
                "Photographing a flat surface, or photographing any scene from a camera that only rotates in place",
            ],
            "correct": 3,
            "explanation": "A homography exactly models plane-to-plane relationships (a flat scene) or pure camera rotation about a fixed center; a translating camera viewing a genuinely 3D scene has no single exact homography relating the views.",
        },
    ],

    25: [
        {
            "question": "What four steps does the stitching pipeline in this lesson combine, in order?",
            "choices": [
                "Detect/match features, estimate a robust homography, warp, composite/blend",
                "Threshold, morphology, connected components, moments, in that specific order",
                "Calibrate the camera, undistort, rectify, match rows, then triangulate",
                "Convert to grayscale, blur, edge-detect, Hough transform, then overlay lines",
            ],
            "correct": 0,
            "explanation": "The pipeline is: SIFT + ratio-test matching (Lesson 20), RANSAC homography (Lessons 23-24), <code>cv2.warpPerspective</code> (Lesson 9), then feathered blending.",
        },
        {
            "question": "In the final composite, what does the linear feather blend do across the overlap region?",
            "choices": [
                "Picks whichever image has higher average brightness for the whole region",
                "Ramps the blend weight smoothly from fully view 1 to fully view 2 across the overlap",
                "Averages the two images everywhere, including non-overlapping regions",
                "Discards the overlap region entirely, leaving a visible black gap",
            ],
            "correct": 1,
            "explanation": "A spatially-varying alpha ramp (same weighted-sum idea as <code>cv2.addWeighted</code>, Lesson 2) is used only within the overlap, so the seam fades smoothly rather than showing a hard cut.",
        },
        {
            "question": "With no synthetic ground truth available for a real photo pair, how does the lesson measure how good the estimated homography's fit is?",
            "choices": [
                "It cannot be measured at all for real, unlabeled photo pairs without extra equipment",
                "By comparing the recovered homography's determinant to exactly 1, as a sanity check",
                "By computing the reprojection error of the RANSAC inliers themselves under the fitted homography",
                "By simply counting the number of SIFT keypoints detected in each image",
            ],
            "correct": 2,
            "explanation": "Projecting each inlier through <code>H</code> and measuring distance to its matched point (reprojection error) gives a direct, ground-truth-free measure of geometric fit quality.",
        },
        {
            "question": "According to the lesson, why might OpenCV's default <code>cv2.Stitcher</code> <code>PANORAMA</code> mode bow straight lines when stitching photos of a flat building facade?",
            "choices": [
                "It always fails completely on any outdoor scene, regardless of lighting",
                "It requires at least 3 input images to work correctly, unlike two-view stitching",
                "It only works correctly in grayscale, never full color, due to a format limitation",
                "It assumes the camera rotated about its optical center and warps onto a sphere",
            ],
            "correct": 3,
            "explanation": "<code>PANORAMA</code> mode's spherical warp is suited to pure camera rotation; <code>SCANS</code> mode instead composites with planar homographies directly, matching this lesson's approach.",
        },
    ],
    26: [
        {
            "question": "In the pinhole projection formula <code>x = f*X/Z</code>, <code>y = f*Y/Z</code>, what determines how much a 3D point's image position changes as its depth <code>Z</code> increases (with <code>X</code>, <code>Y</code> fixed)?",
            "choices": [
                "Increasing <code>Z</code> shrinks the projected coordinates toward the center",
                "The image position is completely independent of the depth <code>Z</code>",
                "Increasing <code>Z</code> moves the projected point further from the image center",
                "Depth <code>Z</code> only affects the pixel's color, never its position",
            ],
            "correct": 0,
            "explanation": "Both <code>x</code> and <code>y</code> are divided by <code>Z</code>, so farther points (larger <code>Z</code>) project closer to the image center -- the usual perspective effect.",
        },
        {
            "question": "Why do objects nearer or farther than the focal plane appear blurred (the circle of confusion), according to this lesson?",
            "choices": [
                "Because the sensor's Bayer color filter fails specifically at those distances, not the lens",
                "Because a lens focuses sharply only for one particular distance",
                "Because gamma correction systematically distorts out-of-focus regions, not the lens itself",
                "Because JPEG compression blurs distant objects more than nearby ones, during encoding",
            ],
            "correct": 1,
            "explanation": "Only points at the focal plane converge to a sharp point on the sensor; nearer/farther points spread over a disk whose size grows with distance from the focal plane.",
        },
        {
            "question": "What happens if a raw Bayer mosaic is demosaicked assuming the wrong filter pattern (e.g. BGGR treated as RGGB)?",
            "choices": [
                "OpenCV raises an error and refuses to proceed with demosaicking under a wrong pattern",
                "The result is identical either way, since demosaicking is pattern-independent by design",
                "It fails silently: every pixel still gets a value, but colors are badly and systematically wrong",
                "Only the brightness channel is affected; the color channels stay correct regardless",
            ],
            "correct": 2,
            "explanation": "Since every pixel gets *some* value regardless of the assumed pattern, a wrong pattern produces a plausible-looking but strongly color-shifted image rather than an obvious failure.",
        },
        {
            "question": "Why is naively averaging two gamma-encoded pixel values (e.g. for blending or blurring) physically wrong?",
            "choices": [
                "Gamma encoding only affects color images, never grayscale ones, by definition",
                "It isn't actually wrong; encoded values can always be averaged directly without loss",
                "Gamma correction has been removed entirely from all modern image formats and sensors",
                "Gamma encoding is nonlinear, so averaging encoded values isn't the same as averaging true light intensity",
            ],
            "correct": 3,
            "explanation": "The correct approach is to decode to linear light, average there, then re-encode; averaging directly in encoded space under-represents the true blended brightness, as the lesson's black/white example shows.",
        },
    ],
    27: [
        {
            "question": "What two categories of parameters does camera calibration recover, according to this lesson?",
            "choices": [
                "Intrinsics (K) and distortion coefficients",
                "Focal length and image resolution only, nothing else",
                "The checkerboard's rotation and translation only",
                "Color balance and exposure settings used by the camera",
            ],
            "correct": 0,
            "explanation": "Calibration recovers the ideal pinhole intrinsics matrix <code>K</code> plus the radial/tangential distortion coefficients that describe how a real lens deviates from that ideal.",
        },
        {
            "question": "Why does the lesson generate many synthetic checkerboard views from different angles rather than just one?",
            "choices": [
                "A single view is always sufficient to recover K and distortion exactly, with zero noise",
                "More views from a wider variety of angles provide more constraints, averaging out detector noise",
                "OpenCV's <code>calibrateCamera</code> function requires exactly 40 views to run at all",
                "Multiple views are only needed to estimate the checkerboard's physical size, nothing else",
            ],
            "correct": 1,
            "explanation": "With noisy corner detections, a single view underconstrains the problem; many diverse views average out noise and better pin down both K and distortion.",
        },
        {
            "question": "What does <code>k1</code> and <code>k2</code> radial distortion do to straight lines in a scene, when captured through a real lens?",
            "choices": [
                "Nothing -- radial distortion only affects color, never geometry",
                "It always makes lines longer, without introducing any curve",
                "It bows them into curves (barrel or pincushion, depending on sign)",
                "It removes straight lines entirely from the captured image",
            ],
            "correct": 2,
            "explanation": "Radial distortion terms in the distortion model curve straight lines; the sign of <code>k1</code>/<code>k2</code> determines whether they bow outward (barrel) or inward (pincushion).",
        },
        {
            "question": "After applying <code>cv2.undistortPoints</code> with the *estimated* (not true) calibration, why don't the grid lines straighten out perfectly?",
            "choices": [
                "Because <code>undistortPoints</code> only works correctly on color images, never grayscale",
                "Because <code>undistortPoints</code> intentionally leaves a small amount of distortion for aesthetic reasons",
                "Because the grid lines were never actually distorted in the first place, by construction",
                "Because the estimated K and distortion coefficients are themselves only approximately correct",
            ],
            "correct": 3,
            "explanation": "Calibration from noisy corner detections only approximates the true K/distortion, so undoing distortion with the estimate leaves a faint residual bow.",
        },
    ],
    28: [
        {
            "question": "What does the epipolar constraint <code>x2^T F x1 = 0</code> tell you about where a point's match must lie in the second image?",
            "choices": [
                "The match is constrained to lie on a specific 1D line",
                "The match could be anywhere in the second image with equal probability",
                "The match must be at the exact same pixel coordinates as in image 1",
                "The match is constrained only if the images have already been rectified",
            ],
            "correct": 0,
            "explanation": "Given <code>x1</code>, the line <code>l2 = F x1</code> is the epipolar line in image 2 that the true match is guaranteed to lie on, collapsing the search from 2D to 1D.",
        },
        {
            "question": "Geometrically, why does a point's match always fall on a single line rather than anywhere in the second image?",
            "choices": [
                "Because both cameras must have exactly identical intrinsic parameters, which is rarely true",
                "Because every candidate 3D point on camera 1's ray lies in one epipolar plane with both camera centers",
                "Because feature descriptors can only ever match along horizontal lines, by convention",
                "This fact is only true after the two images have been rectified, not in general",
            ],
            "correct": 1,
            "explanation": "All points along the ray from C1 through x1 share the same epipolar plane with C1 and C2; camera 2 sees that plane edge-on, so every candidate projects onto the same line.",
        },
        {
            "question": "Why does the 8-point algorithm normalize pixel coordinates (centering and rescaling) before solving for F via SVD?",
            "choices": [
                "Normalization is purely cosmetic and has no effect on the actual numerical result",
                "OpenCV requires normalized coordinates as a strict API constraint on all inputs",
                "Raw pixel coordinates (in the hundreds) make the linear system badly conditioned numerically",
                "Normalization is needed only in the rare case where <code>F</code> has rank 3, not rank 2",
            ],
            "correct": 2,
            "explanation": "This is the Hartley normalization step: solving in centered, rescaled coordinates avoids numerical ill-conditioning that raw large pixel values would cause.",
        },
        {
            "question": "Why can <code>cv2.recoverPose</code> only recover the *direction* of the translation between the two cameras, not its magnitude?",
            "choices": [
                "Because of a known implementation bug in that specific OpenCV function, now fixed",
                "Because the essential matrix only exists when the cameras are calibrated in advance",
                "Because rotation and translation can never both be recovered from two views at once",
                "Because doubling scene size and baseline together produces identical images",
            ],
            "correct": 3,
            "explanation": "The scale ambiguity is fundamental to monocular two-view geometry: uniformly scaling the scene and baseline together produces identical images, so absolute scale can't be recovered from correspondences alone.",
        },
    ],
    29: [
        {
            "question": "What does triangulation compute, given two camera projection matrices and a matched 2D point pair?",
            "choices": [
                "The 3D position of the point that projects to both observed 2D locations",
                "The camera's intrinsic calibration matrix K, recovered from checkerboard images",
                "The fundamental matrix F relating the two views, up to an unknown scale",
                "The optimal feature descriptor for the matched point, used for re-matching",
            ],
            "correct": 0,
            "explanation": "Triangulation intersects the two rays (one per camera) that could have produced the matched 2D observations, recovering the 3D point via DLT/SVD.",
        },
        {
            "question": "What is the purpose of the cheirality check (requiring positive depth in both cameras) after triangulation?",
            "choices": [
                "To speed up the SVD computation used during triangulation, nothing more",
                "To catch mismatches that pass RANSAC but triangulate behind a camera",
                "To convert the reconstruction from a sparse to a dense point cloud directly",
                "To separately estimate the camera's unknown focal length from the matches",
            ],
            "correct": 1,
            "explanation": "The epipolar constraint doesn't rule out points that triangulate behind a camera; requiring positive depth in both views is a free extra filter that catches such bad matches.",
        },
        {
            "question": "Why does the reconstruction from *estimated* poses need to be multiplied by the true baseline length before it matches the true-pose reconstruction?",
            "choices": [
                "Because the estimated poses use a different coordinate system entirely",
                "Because triangulation always underestimates distances by a fixed known factor",
                "Because <code>recoverPose</code> only returns a translation *direction*",
                "This scaling step is unnecessary and is only done here for illustration",
            ],
            "correct": 2,
            "explanation": "Since <code>t</code> from <code>recoverPose</code> is unit-length, every triangulated point comes out a fixed factor smaller than reality; rescaling by the true baseline fixes the metric scale (using outside knowledge, since two views alone can't supply it).",
        },
        {
            "question": "What does bundle adjustment do that distinguishes a real multi-view SfM system from the simple two-view pipeline in this lesson?",
            "choices": [
                "It replaces feature matching entirely with a neural network trained end-to-end",
                "It performs demosaicking on each input photo before matching begins",
                "It converts the sparse point cloud directly into a dense mesh with no further computation",
                "It refines all camera poses and 3D points together in one large nonlinear least-squares optimization",
            ],
            "correct": 3,
            "explanation": "As more views are added, per-view pose errors compound; bundle adjustment jointly refines every camera pose and 3D point at once to minimize the total reprojection error, correcting for that compounding.",
        },
    ],

    30: [
        {
            "question": "According to this lesson, what do PCA (Lesson 6) and camera projection (P = K[R|t]) have in common?",
            "choices": [
                "Both are examples of the same underlying operation: y = w^T x + b, a linear projection",
                "Both require labeled training data in order to be computed",
                "Neither one involves any matrix multiplication at any step",
                "PCA and camera projection are unrelated operations with no shared structure",
            ],
            "correct": 0,
            "explanation": "Both compute a projection via a dot product between a fixed direction and the input coordinates; they differ only in how that direction was chosen.",
        },
        {
            "question": "What single operation does the lesson show convolution, the Fourier transform, and a homography all secretly are?",
            "choices": [
                "Nonlinear activation functions applied elementwise to each output",
                "A matrix multiplication, i.e. a bank of simultaneous linear projections",
                "Gradient descent updates applied to each learnable parameter",
                "Random sampling procedures used to initialize the weights",
            ],
            "correct": 1,
            "explanation": "Each output value in all three is a dot product between the input and a fixed row/kernel/basis function, which is exactly what one row of a matrix multiply computes.",
        },
        {
            "question": "What can a single linear projection classifier never do, regardless of how w and b are chosen?",
            "choices": [
                "Separate two classes that are already linearly separable",
                "Compute a simple dot product between two vectors",
                "Separate a ring of points from a disk of points it surrounds",
                "Be visualized in 1D after projecting the data down",
            ],
            "correct": 2,
            "explanation": "A single projection can only produce a straight-line (hyperplane) decision boundary, and no straight line can separate a ring from the disk it encloses.",
        },
        {
            "question": "In the classification example, what determines which side of the decision boundary a point falls on?",
            "choices": [
                "The point's distance from the image origin alone",
                "Whichever class happens to have more training examples",
                "The point's raw color channel values, unrelated to w or b",
                "The sign of the projected score <code>w^T x + b</code>",
            ],
            "correct": 3,
            "explanation": "Points are classified by thresholding the projected score at zero: positive score means one class, negative means the other.",
        },
    ],
    31: [
        {
            "question": "Why does the lesson use the sigmoid function instead of directly thresholding the raw score z = w^T x + b for training?",
            "choices": [
                "Raw accuracy is flat almost everywhere and has no useful gradient",
                "Sigmoid makes the model run measurably faster during training, by skipping the derivative step",
                "Sigmoid is strictly required in order to compute a dot product between weights and inputs",
                "Thresholding directly would make every prediction always equal to 1, regardless of the input",
            ],
            "correct": 0,
            "explanation": "Hard accuracy jumps discontinuously at the boundary with zero gradient almost everywhere, so gradient descent has nothing to follow; sigmoid squashes the score into a smooth probability instead.",
        },
        {
            "question": "In binary cross-entropy, what happens to the loss when the true label is 1 but the predicted probability p is close to 0?",
            "choices": [
                "The loss approaches zero, rewarding the confident wrong guess",
                "The loss explodes toward infinity -- confidently wrong is punished severely",
                "The loss is completely unaffected by how confident the wrong prediction was",
                "The loss becomes negative, canceling out other terms in the batch",
            ],
            "correct": 1,
            "explanation": "The penalty for true label 1 is -log(p), which grows without bound as p approaches 0.",
        },
        {
            "question": "What two independent methods does the lesson use to verify the hand-derived backprop gradient formulas are correct?",
            "choices": [
                "Running the model twice in a row and comparing the two outputs for consistency",
                "Checking only that the computed loss is always positive throughout training",
                "Numerical finite-difference gradient checking, and comparing against PyTorch's autograd",
                "Comparing training accuracy against a random chance baseline at every epoch",
            ],
            "correct": 2,
            "explanation": "The lesson perturbs each parameter by a tiny amount to estimate the gradient numerically, and separately lets PyTorch's .backward() compute it automatically -- both agree with the hand-derived formula.",
        },
        {
            "question": "Why does gradient descent do *worse* than exhaustive search on the ring-vs-disk dataset, converging to near-chance accuracy?",
            "choices": [
                "Gradient descent always fails on any dataset with fewer than 1000 points",
                "The learning rate was simply too small to move the weights at all",
                "Sigmoid activations cannot be used with more than 60 data points",
                "The dataset is roughly symmetric around the origin",
            ],
            "correct": 3,
            "explanation": "Because the ring and disk are both centered at the origin, the training signal from all points nearly cancels in every direction, so the optimizer settles near w=0 instead of finding an asymmetric corner-case solution.",
        },
    ],
    32: [
        {
            "question": "Why does stacking two purely linear layers (no nonlinearity in between) fail to add any representational power over one layer?",
            "choices": [
                "The composition of two linear projections is algebraically just one combined linear projection",
                "Two linear layers always train measurably slower than a single one, due to extra matrix multiplies",
                "PyTorch does not technically allow stacking two linear layers without an activation between them",
                "Linear layers can only be stacked if they share the exact same width at every layer",
            ],
            "correct": 0,
            "explanation": "Without a nonlinearity, z = w2^T(W1 x + b1) + b2 simplifies algebraically into a single w'^T x + b', exactly Lesson 31's single neuron again.",
        },
        {
            "question": "What role does the ReLU nonlinearity play in an MLP's hidden layer?",
            "choices": [
                "It has no real effect and is included only by convention, not necessity",
                "It breaks the collapse that would reduce stacked linear layers into one projection",
                "It converts the entire network into an unsupervised learning model with no labels",
                "It only affects the bias terms, never the weight matrices themselves, during backprop",
            ],
            "correct": 1,
            "explanation": "Because ReLU is not a linear function, w2^T ReLU(W1 x + b1) + b2 cannot be rewritten as any single linear projection, unlike the all-linear case.",
        },
        {
            "question": "In the capacity sweep, what does H=1 (a single hidden unit) achieve on the ring-vs-disk problem, no matter how long it trains?",
            "choices": [
                "Consistently 100% accuracy, no matter the random seed",
                "Exactly 0% accuracy every single time it is trained",
                "A ceiling near Lesson 30's ~74% linear limit",
                "Accuracy that improves without bound given enough training epochs",
            ],
            "correct": 2,
            "explanation": "One hidden unit provides only a single straight-line cut, so it inherits the same representational ceiling a lone linear projection has -- a capacity limit, not an optimization failure.",
        },
        {
            "question": "When the trained hidden layer's 4D output is projected to 2D with PCA for visualization, what does the lesson observe about the two classes?",
            "choices": [
                "They remain exactly as tangled as they were in the original input space, unchanged",
                "They collapse together into a single overlapping cluster, losing all class structure",
                "The visualization turns out identical to the raw input-space plot, pixel for pixel",
                "They are pulled apart into something close to linearly separable, even in this lossy 2D snapshot",
            ],
            "correct": 3,
            "explanation": "The hidden layer reshapes the space so the classes become nearly linearly separable, illustrating that each layer's job is to make the next layer's job easier.",
        },
    ],
    33: [
        {
            "question": "On the narrow, steep-in-y bowl loss surface, why does plain gradient descent zigzag instead of converging smoothly?",
            "choices": [
                "A step size that makes progress along the shallow x-axis overshoots along the steep y-axis",
                "Plain gradient descent cannot handle any 2D loss surface at all, by mathematical necessity",
                "The loss function being minimized has no minimum to converge to anywhere",
                "The learning rate was mistakenly set to exactly zero for this particular run",
            ],
            "correct": 0,
            "explanation": "A single shared step size can't be simultaneously right for a shallow and a steep direction at once, causing the characteristic zigzag.",
        },
        {
            "question": "What is the key mechanism behind momentum's benefit on the narrow bowl?",
            "choices": [
                "It sets the learning rate to exactly zero after a fixed number of steps",
                "It accumulates a running velocity so consistent gradients",
                "It removes the need for a loss function entirely during training",
                "It only ever slows down training progress, never speeds it up",
            ],
            "correct": 1,
            "explanation": "Momentum's real benefit is acceleration along consistently-pointing directions, not damping oscillation, since a single beta can't be tuned per-axis.",
        },
        {
            "question": "How does Adam differ from momentum in how it handles the steep vs. shallow axes of the bowl?",
            "choices": [
                "Adam ignores gradient direction entirely, using only its magnitude at each step",
                "Adam is functionally identical to momentum, just under a different name and notation",
                "Adam divides the step by a running estimate of each parameter's squared gradient",
                "Adam only works correctly on strictly 1-dimensional loss surfaces, unlike momentum",
            ],
            "correct": 2,
            "explanation": "Dividing by sqrt(v) per-parameter automatically shrinks steps on the steep axis and boosts them on the shallow one, unlike momentum's single shared coefficient.",
        },
        {
            "question": "Why does initializing all weights to exactly zero fail catastrophically for a hidden layer with more than one unit?",
            "choices": [
                "Zero weights cause an immediate runtime crash in PyTorch",
                "Zero-initialized networks actually train faster than randomly-initialized ones",
                "Zero initialization only fails specifically when using the Adam optimizer",
                "Every hidden unit computes the identical function and receives the identical gradient",
            ],
            "correct": 3,
            "explanation": "With identical weights and identical gradients, every unit updates identically forever, so the model is stuck computing a single trivial function no matter how long it trains.",
        },
    ],
    34: [
        {
            "question": "What two problems does a convolutional layer's weight-sharing fix, compared to a fully-connected layer applied to a flattened image?",
            "choices": [
                "It reduces parameter count by reusing the same small filter everywhere, and gives translation equivariance",
                "It makes training completely deterministic and removes the need for random initialization",
                "It discards spatial structure and requires re-learning the same feature at every position independently",
                "It eliminates the need for any loss function during convolutional training",
            ],
            "correct": 0,
            "explanation": "Restricting each output to a local patch and sharing that patch's weights across positions both shrinks the parameter count and makes detections shift consistently with the input.",
        },
        {
            "question": "For a 3x3 convolution filter applied to a 3-channel RGB image, how many weights does that single filter actually have?",
            "choices": [
                "9, the same number of weights as for a grayscale image of the same kernel size",
                "27, since the filter spans 3x3 spatially across all 3 input channels at once",
                "3, exactly one weight per color channel, ignoring the spatial kernel entirely",
                "It depends entirely on the image's spatial resolution, not on channel count",
            ],
            "correct": 1,
            "explanation": "A conv filter is really 3x3xC_in, so on 3 input channels a '3x3 filter' actually has 3x3x3=27 weights, summed into one output value per position.",
        },
        {
            "question": "What does max pooling add to a CNN, beyond shrinking the spatial size of the feature map?",
            "choices": [
                "Exact invariance to any transformation of the input whatsoever",
                "The ability to skip the convolution step entirely, saving computation",
                "A small amount of local translation invariance",
                "A guarantee that the resulting network cannot possibly overfit",
            ],
            "correct": 2,
            "explanation": "Keeping only the max value in each window means small position shifts within that window don't change the pooled output.",
        },
        {
            "question": "In the unseen-position generalization test, why does the flatten-based MLP perform near chance on shapes placed at corner positions never seen in training, while the CNN does not?",
            "choices": [
                "The MLP simply has more trainable parameters than the CNN, causing it to overfit faster",
                "The CNN happened to be trained for more epochs than the MLP in this comparison",
                "The MLP only works correctly on grayscale images, never color, due to its input layer",
                "The MLP memorized which specific pixels tend to be on for each class at the training positions",
            ],
            "correct": 3,
            "explanation": "Flattening ties the MLP's decision to specific pixel positions, while the CNN's global pooling collapses spatial position, letting it recognize shapes regardless of where they appear.",
        },
    ],
    35: [
        {
            "question": "In the overfitting demonstration with only 12 training images, what pattern do the training and validation loss curves show?",
            "choices": [
                "Training loss marches to zero while validation loss bottoms out partway through training",
                "Both curves decrease together and never diverge from each other at any point",
                "Both curves increase steadily throughout the entire training run, without exception",
                "Validation loss is always lower than training loss at every single epoch",
            ],
            "correct": 0,
            "explanation": "The network perfectly memorizes the 12 training images while validation loss reaches a minimum then gets worse as the model overfits further.",
        },
        {
            "question": "How does data augmentation help fight overfitting on a small dataset?",
            "choices": [
                "It deletes the hardest training examples from the dataset entirely, before training starts",
                "It manufactures more training variety via label-preserving transforms like flips and rotations",
                "It automatically increases the optimizer's learning rate as training proceeds",
                "It removes the need for a held-out validation set entirely during training",
            ],
            "correct": 1,
            "explanation": "Applying random label-preserving transforms to the existing images gives the model many more effective examples to learn from, rather than memorizing a handful of exact images.",
        },
        {
            "question": "What does weight decay do to fight overfitting?",
            "choices": [
                "It deletes a random fraction of the training images on each epoch of training",
                "It increases the overall capacity of the model being trained, adding more layers",
                "It adds a penalty proportional to weight magnitude, discouraging large weights",
                "It forces every weight in the network to become exactly zero at initialization",
            ],
            "correct": 2,
            "explanation": "Penalizing large weight magnitudes makes memorizing noisy specifics of a small dataset more costly relative to finding a simpler, smoother function.",
        },
        {
            "question": "In inverted dropout, what happens to the surviving (non-zeroed) activations during training, and why?",
            "choices": [
                "They are left completely unchanged, exactly as computed, with no rescaling",
                "They are set to exactly 1, regardless of their original computed value",
                "They are doubled in value, regardless of the dropout probability used",
                "They are rescaled by 1/(1-p) to keep the layer's expected output the same scale",
            ],
            "correct": 3,
            "explanation": "Rescaling by 1/(1-p) keeps the expected activation magnitude consistent between training (with dropout) and evaluation (without it).",
        },
    ],

    36: [
        {
            "question": "Why does an overall accuracy of just over 50% on this lesson's 4-class CIFAR-10 task hide something important?",
            "choices": [
                "Because it doesn't reveal that the model is not equally good at all four classes",
                "Because 50% accuracy is actually below chance level for 4 classes",
                "Because accuracy can only ever be computed for binary classifiers",
                "Because the test set itself turns out to be imbalanced",
            ],
            "correct": 0,
            "explanation": "A single overall-accuracy number averages away per-class differences; here it hides that the truck class (starved to 90 training images) is rarely predicted correctly.",
        },
        {
            "question": "In a confusion matrix where row <code>i</code>, column <code>j</code> counts examples of true class <code>i</code> predicted as class <code>j</code>, what does a perfect classifier's matrix look like?",
            "choices": [
                "All zeros, with no predictions recorded anywhere",
                "All the mass on the diagonal (row i, column i for every i)",
                "Spread perfectly uniformly across every cell of the matrix",
                "All the mass concentrated in the top-right corner of the matrix",
            ],
            "correct": 1,
            "explanation": "A correct prediction means predicted class equals true class, i.e. j = i, which is exactly the diagonal.",
        },
        {
            "question": "What does <code>recall</code> measure for a given class?",
            "choices": [
                "Of everything the model called this class, what fraction actually was",
                "The total number of training examples available for this class",
                "Of everything that really was this class, what fraction the model caught",
                "How fast the model runs inference on this class's images",
            ],
            "correct": 2,
            "explanation": "Recall = TP / (TP + FN): low recall means the model misses that class often, regardless of what it says about other classes.",
        },
        {
            "question": "Why does the starved <code>truck</code> class end up with high precision but very low recall?",
            "choices": [
                "Because trucks are visually identical to automobiles in every single image, pixel for pixel",
                "Because the test set has no truck images in it at all, unlike the training set",
                "Precision and recall are mathematically always equal for any class, by definition",
                "It fails to recognize most actual trucks, defaulting to well-represented classes like automobile instead",
            ],
            "correct": 3,
            "explanation": "This is the standard signature of class imbalance: the model rarely guesses the rare class, but when it does, it's confident and usually correct.",
        },
    ],
    37: [
        {
            "question": "What made the ILSVRC benchmark (built on a subset of ImageNet) so important for comparing architectures?",
            "choices": [
                "Every group trained on the same data and was scored on the same held-out test set",
                "It was the first dataset ever to include color images, unlike earlier benchmarks",
                "It only ever allowed one submission per architecture, forever, with no resubmission",
                "It required all models to have exactly the same number of parameters, by rule",
            ],
            "correct": 0,
            "explanation": "A shared, standardized competition -- not just a big labeled dataset -- is what let different architectures be compared fairly and drove rapid progress.",
        },
        {
            "question": "According to the lesson, why does stacking more layers make a *plain* (non-residual) deep network worse, not better?",
            "choices": [
                "More layers always increase training time only, with no effect on accuracy whatsoever",
                "Backprop multiplies the gradient by each layer's Jacobian, so small factors shrink it to nothing",
                "Deeper networks always overfit, regardless of the dataset's size, by construction",
                "GPUs are physically incapable of executing more than 20 sequential layers in sequence",
            ],
            "correct": 1,
            "explanation": "This is the vanishing-gradient problem: repeated multiplication by small per-layer factors drives the gradient toward zero, so early layers stop learning.",
        },
        {
            "question": "Structurally, why does a residual connection (<code>x_{l+1} = x_l + F(x_l)</code>) prevent the gradient from vanishing across many layers?",
            "choices": [
                "It removes the need for any activation function whatsoever, anywhere in the network",
                "It automatically doubles the learning rate during training, independent of the optimizer",
                "Its Jacobian is I + dF/dx_l, so an identity term always survives the multiplication chain",
                "It replaces backpropagation entirely with a different optimization algorithm, like RANSAC",
            ],
            "correct": 2,
            "explanation": "The identity matrix in the Jacobian is never shrunk by a small activation derivative, unlike a plain network where every layer's Jacobian is purely <code>dF/dx_l</code>.",
        },
        {
            "question": "What do the two learnable parameters <code>gamma</code> and <code>beta</code> do in batch normalization, after a channel's activations are normalized to mean 0 and standard deviation 1?",
            "choices": [
                "They discard the normalized values entirely and recompute them from scratch",
                "They control the mini-batch size used during training",
                "They convert the normalized activations back into 8-bit integers",
                "They let the network rescale and shift the normalized activations",
            ],
            "correct": 3,
            "explanation": "gamma (scale) and beta (shift) are applied on top of the 0/1-normalized activations, so the network can still recover a different mean/scale if that's actually useful.",
        },
    ],
    38: [
        {
            "question": "What is the key idea behind transfer learning, as motivated in this lesson?",
            "choices": [
                "Reusing a network already trained on a different, data-rich source task",
                "Training a brand new network from scratch on the target task, but with more epochs",
                "Always using the largest possible batch size for the target task",
                "Manually copying weights between two unrelated tasks without any pretraining",
            ],
            "correct": 0,
            "explanation": "Transfer learning sidesteps data-starved target tasks by reusing generic low-level features learned on a data-rich source task, rather than learning everything from scratch.",
        },
        {
            "question": "What is the key difference between the 'frozen backbone' and 'fine-tuning' strategies in this lesson?",
            "choices": [
                "Fine-tuning never actually uses the pretrained weights at all, starting from scratch instead",
                "Frozen backbone trains a classifier on fixed features; fine-tuning also updates the backbone",
                "Frozen backbone requires strictly more target training images than fine-tuning does",
                "There is no real difference between the two strategies in practice, despite the names",
            ],
            "correct": 1,
            "explanation": "Frozen backbone only trains the new head; fine-tuning also updates the backbone, but carefully, with a small backbone learning rate to avoid wrecking what it already learned.",
        },
        {
            "question": "Why does fine-tuning use a backbone learning rate roughly 30x smaller than the classifier head's learning rate?",
            "choices": [
                "To make the overall training process run measurably faster, regardless of dataset size",
                "Because PyTorch requires different learning rates for different layers by default settings",
                "A large backbone update from just 30 target examples would overfit those few images",
                "Because the backbone simply has more parameters than the classifier head, by architecture",
            ],
            "correct": 2,
            "explanation": "A much smaller backbone learning rate protects the pretrained features from being wiped out by noisy updates from a tiny target dataset.",
        },
        {
            "question": "Why is the comparison between the toy frozen backbone and the real ImageNet-pretrained ResNet-18 not a perfectly controlled experiment, according to the lesson's caveat?",
            "choices": [
                "ResNet-18 was never actually trained on ImageNet in the first place",
                "The toy backbone turns out to be actually larger than ResNet-18",
                "The comparison secretly uses different target images for each strategy",
                "ImageNet's 1,000 classes already include cats and several vehicle types",
            ],
            "correct": 3,
            "explanation": "Since ImageNet already contains cat and vehicle photos, some of ResNet-18's advantage is direct class overlap, not purely richer general-purpose pretraining.",
        },
    ],
    39: [
        {
            "question": "What does a saliency map visualize?",
            "choices": [
                "The gradient of the predicted class score with respect to every input pixel",
                "The raw pixel values of the input image, left completely unchanged",
                "The convolutional kernel weights themselves, visualized directly",
                "The class label the model predicts, overlaid as text on the image",
            ],
            "correct": 0,
            "explanation": "A pixel with large gradient magnitude is one the network's prediction is most sensitive to -- a pixel it's 'looking at'.",
        },
        {
            "question": "Why are Grad-CAM heatmaps coarser (lower-resolution) than saliency maps?",
            "choices": [
                "Grad-CAM only works correctly on grayscale images, never color",
                "Grad-CAM operates on a convolutional layer's feature map",
                "Grad-CAM intentionally blurs its own output for aesthetic reasons",
                "Saliency maps are actually the coarser of the two, not Grad-CAM",
            ],
            "correct": 1,
            "explanation": "Grad-CAM's resolution matches whichever conv layer it uses, which is coarser than the full input resolution a saliency map operates at.",
        },
        {
            "question": "On *clean* test images (no marker present), why does Grad-CAM still catch <code>model_shortcut</code>'s reliance on the marker far more reliably than raw saliency does?",
            "choices": [
                "Saliency maps cannot be computed at all on clean, unmarked images, only on marked ones",
                "Grad-CAM was specifically trained in advance to detect markers, unlike saliency maps",
                "Grad-CAM pools gradients over a whole feature-map channel, smoothing out pixel-level noise",
                "There is actually no measurable difference between the two methods on clean images at all",
            ],
            "correct": 2,
            "explanation": "The lesson reports Grad-CAM's peak lands in the marker region 92% of the time on clean images versus only 56% for saliency, attributed to Grad-CAM's channel-level pooling smoothing out pixel noise.",
        },
        {
            "question": "What does t-SNE do differently from a linear projection like PCA (Lesson 6)?",
            "choices": [
                "t-SNE requires labeled training data, while PCA does not need any labels at all",
                "PCA can only ever be applied to images, never to general feature vectors directly",
                "t-SNE and PCA are mathematically identical operations under the hood, just renamed",
                "t-SNE explicitly optimizes to preserve local neighborhoods in the projection",
            ],
            "correct": 3,
            "explanation": "t-SNE emphasizes preserving which points are close to which other points locally, which is why same-class images can end up visibly clustered even without ever seeing their labels.",
        },
    ],
    40: [
        {
            "question": "What is the core idea of a sliding-window detector, as built in this lesson?",
            "choices": [
                "Run a classifier at every location and scale, treating each window as its own classification problem",
                "Directly regress bounding box coordinates from the whole image in a single forward pass",
                "Cluster pixels together by color alone to find object boundaries directly",
                "Use only the image's frequency-domain representation to locate objects precisely",
            ],
            "correct": 0,
            "explanation": "A window classifier answers 'face or not, at this exact window,' and sliding it across every position turns that per-window classifier into a detector.",
        },
        {
            "question": "In this lesson's box convention, how is each detected face represented?",
            "choices": [
                "By its top-left corner coordinates and width/height",
                "By its center coordinates <code>(cx, cy)</code> and width/height",
                "By the pixel coordinates of all four of its corners",
                "By a single pixel location, with no size information at all",
            ],
            "correct": 1,
            "explanation": "Boxes here are stored as (cx, cy, w, h), with the center reported directly rather than a top-left corner.",
        },
        {
            "question": "After thresholding the sliding-window score map, why does a single true face typically produce a *cluster* of detections rather than one?",
            "choices": [
                "Because the window classifier is applied only a single time per image",
                "Because faces are always detected exactly twice, due to a scoring bug",
                "Because every window that overlaps a face heavily enough scores above threshold",
                "Because thresholding always produces exactly one detection per object",
            ],
            "correct": 2,
            "explanation": "Many nearby, overlapping windows all score above threshold near a real face, producing a cluster of near-duplicate detections.",
        },
        {
            "question": "What does non-maximum suppression (NMS) do to resolve duplicate detections?",
            "choices": [
                "It averages all overlapping detections together into a single box, blending their scores",
                "It discards every single detection that lies above the score threshold immediately",
                "It requires the user to manually click on the correct detection every single time",
                "It keeps the highest-scoring detection and discards overlapping ones above an IoU threshold",
            ],
            "correct": 3,
            "explanation": "NMS greedily keeps the best-scoring box in each cluster and suppresses its highly-overlapping neighbors, collapsing duplicates down to one detection per object.",
        },
    ],
    41: [
        {
            "question": "How does this lesson's detector represent the box it predicts?",
            "choices": [
                "Four numbers (cx, cy, width, height), normalized to [0, 1] by image size",
                "A pixel-wise segmentation mask covering the whole image",
                "A class label only, with no location information whatsoever",
                "Four corner coordinates expressed in absolute pixel units",
            ],
            "correct": 0,
            "explanation": "The network's regression head outputs (cx, cy, w, h) through a sigmoid, keeping every value in the valid normalized [0, 1] range.",
        },
        {
            "question": "Why is IoU used to evaluate the detector instead of just looking at how close the four predicted numbers are to the true ones?",
            "choices": [
                "IoU is faster to compute than any plain numeric comparison",
                "IoU directly measures how much the predicted and true boxes overlap",
                "IoU is only ever used during training, never for evaluation",
                "The four box numbers cannot be meaningfully compared directly, for any reason",
            ],
            "correct": 1,
            "explanation": "Two boxes can have similar-looking coordinates but very different overlap (or vice versa), so IoU is the metric that reflects what actually matters for localization.",
        },
        {
            "question": "Why can this lesson's simple regression detector only ever predict exactly one object per image?",
            "choices": [
                "PyTorch is technically incapable of outputting more than 4 numbers",
                "It was simply never trained on any multi-object scenes",
                "Its output is a fixed-size vector of exactly 4 numbers",
                "The sigmoid activation itself limits the network to one prediction",
            ],
            "correct": 2,
            "explanation": "A fixed 4-number output has no way to represent a variable number of objects; real multi-object detectors need a different architecture (grid cells, region proposals, etc.).",
        },
        {
            "question": "What is the key architectural difference between two-stage (R-CNN family) and single-stage (YOLO, SSD) detectors?",
            "choices": [
                "Single-stage detectors always use strictly more parameters than two-stage ones, by design",
                "Two-stage detectors are structurally incapable of using convolutional networks at all",
                "There is no real architectural difference; the two terms are interchangeable in practice",
                "Two-stage detectors first generate region proposals and then classify/regress each one",
            ],
            "correct": 3,
            "explanation": "Two-stage methods (e.g. Faster R-CNN) propose-then-classify; single-stage methods (YOLO, SSD) predict boxes and classes directly per grid cell in a single pass, trading some accuracy for speed.",
        },
    ],
    42: [
        {
            "question": "What is the fundamental difference between semantic segmentation and detection (Lesson 41)?",
            "choices": [
                "Segmentation labels every pixel with a class; detection only draws boxes around a handful of objects",
                "Segmentation only works correctly on grayscale images, never color, due to its design",
                "Detection can find more object classes than segmentation ever possibly could, by nature",
                "There is no real difference; the two terms are used interchangeably in the literature",
            ],
            "correct": 0,
            "explanation": "Detection localizes objects with boxes; semantic segmentation goes further and assigns a class label to every single pixel in the image.",
        },
        {
            "question": "In this lesson's minimal FCN-style network, what does the decoder see when producing its output?",
            "choices": [
                "The original input image, passed through completely unchanged, with no processing",
                "Only the low-resolution bottleneck features, upsampled back to full resolution",
                "A random subset of encoder features sampled at every resolution, chosen arbitrarily",
                "Only the ground-truth mask, visible exclusively during training, never at test time",
            ],
            "correct": 1,
            "explanation": "A plain FCN just downsamples then upsamples: the decoder has no access to the encoder's higher-resolution intermediate features, only the coarse bottleneck.",
        },
        {
            "question": "How do U-Net's skip connections fix the boundary-detail loss of a plain FCN?",
            "choices": [
                "By simply training the same network for twice as long, with no architecture change",
                "By replacing max pooling throughout with average pooling instead, everywhere",
                "By concatenating the encoder's higher-resolution features directly into the matching decoder stage",
                "By adding more convolutional layers directly to the bottleneck itself, making it deeper",
            ],
            "correct": 2,
            "explanation": "Skip connections concatenate encoder features into the decoder at each matching resolution, so fine spatial detail never has to survive the lossy bottleneck in the first place.",
        },
        {
            "question": "When the real pretrained FCN is run on a photo containing a tram, it labels part of the tram `train` and part `bus`. What does this reveal?",
            "choices": [
                "The model is completely broken, so none of its other predictions can be trusted",
                "The photo was somehow corrupted while it was being loaded",
                "FCN can only ever label one class per image, so this is expected of every photo",
                "Pascal VOC's 20-class vocabulary has no `tram` class",
            ],
            "correct": 3,
            "explanation": "The split roughly tracks a real visual seam (windows vs. body) and reflects VOC's limited vocabulary, not a random or nonsensical failure — a reminder that a model's answers are only as good as its label set.",
        },
    ],
    43: [
        {
            "question": "What can semantic segmentation not distinguish, that instance segmentation adds?",
            "choices": [
                "Which pixels belong to *this specific* object versus another object of the same class",
                "The precise color of each object in the scene, down to the exact RGB value",
                "Whether a given object is moving or stationary, frame to frame",
                "The exact class name assigned to each pixel, with no instance information",
            ],
            "correct": 0,
            "explanation": "Semantic segmentation only assigns a class per pixel; it has no notion of separate identity, so two touching objects of the same class merge into one labeled region.",
        },
        {
            "question": "How does this lesson turn Lesson 42's segmentation network into an instance segmentation network?",
            "choices": [
                "By training a completely new, entirely unrelated architecture from scratch, with no reuse",
                "By adding a second head predicting, per pixel, an offset toward its instance's center",
                "By simply running the same semantic network twice in a row, on the same input",
                "By increasing the total number of semantic classes the network predicts, nothing more",
            ],
            "correct": 1,
            "explanation": "The network keeps its original semantic head and gains a second head trained to regress a per-pixel offset to the instance center; clustering the resulting votes (Hough-style) recovers individual instances.",
        },
        {
            "question": "In the toy mask head experiment, why can the mask head reach high per-instance IoU even though every test crop contains a visible neighboring circle?",
            "choices": [
                "Because the crops used are simply too small to ever contain a neighbor, by design choice",
                "Because the neighboring circle is always erased before the crop is taken, in preprocessing",
                "Because detection already solved 'where' — the mask head only answers a simpler, local question",
                "Because the mask head secretly sees the ground-truth instance mask at test time, as a shortcut",
            ],
            "correct": 2,
            "explanation": "Given a box already centered on one instance, the mask head's job narrows to a local same-box question, which is why Mask R-CNN's real mask branch can be small and fast — a good box already does most of the work.",
        },
        {
            "question": "What is the one new piece Mask R-CNN adds on top of Faster R-CNN (Lesson 41), and what other change does it require?",
            "choices": [
                "A completely new region proposal network, plus a substantially bigger backbone, from scratch",
                "A third detection stage, plus doubling the total number of anchor boxes used",
                "Nothing at all — Mask R-CNN is just Faster R-CNN with a different loss function, unchanged otherwise",
                "A parallel mask-prediction branch on every detected box, plus replacing RoIPool with RoIAlign",
            ],
            "correct": 3,
            "explanation": "Mask R-CNN keeps Faster R-CNN's RPN, classifier, and box regression unchanged, adds a mask branch per box, and swaps RoIPool for RoIAlign because coordinate quantization that barely hurts classification visibly misaligns pixel-accurate masks.",
        },
    ],
    44: [
        {
            "question": "Why does casting a very small gradient value (e.g. around `1e-11`) from fp32 to fp16 cause training to fail, rather than just losing some precision?",
            "choices": [
                "fp16's limited exponent range means tiny values round to exactly zero instead of something small",
                "fp16 rounds every value to the nearest power of two, which is always a large relative error regardless of magnitude",
                "fp16 is fundamentally incapable of storing negative numbers at all, by its format",
                "PyTorch raises an error and halts training the instant this happens, every single time",
            ],
            "correct": 0,
            "explanation": "Below fp16's representable floor, underflow means total information loss (exactly 0.0), not just reduced precision — and gradients in deep sigmoid networks routinely fall in that underflowing range.",
        },
        {
            "question": "What are the two safeguards mixed-precision training uses to keep the speed of fp16 without hitting the underflow problem?",
            "choices": [
                "Training for many more epochs while using a smaller learning rate, with no other change",
                "Keeping fp32 master weights that absorb small updates, and loss scaling to keep gradients in range",
                "Switching over to double precision (fp64) whenever a gradient underflows, automatically",
                "Rounding every gradient up to the nearest nonzero representable fp16 value, always",
            ],
            "correct": 1,
            "explanation": "fp32 master weights avoid updates being rounded away, and loss scaling exploits gradients being linear in the loss to push otherwise-underflowing values back into fp16's representable range.",
        },
        {
            "question": "In the data-parallelism demonstration, splitting a batch into 4 shards, computing gradients independently per shard, and averaging them afterward produces a result that:",
            "choices": [
                "Is only a rough approximation of the single-device gradient, useful but not exact in practice",
                "Is always larger than the single-device gradient by a factor of exactly 4, every time",
                "Matches the single-device, whole-batch gradient to within ordinary floating-point roundoff",
                "Only works correctly if the model has no nonlinear activations at all, anywhere",
            ],
            "correct": 2,
            "explanation": "This is the whole correctness argument for data parallelism: per-shard gradients averaged after the fact equal the gradient computed over the whole batch at once, up to ~1e-8 roundoff.",
        },
        {
            "question": "What is the key difference between what limits data parallelism versus what limits (pipeline) model parallelism at scale?",
            "choices": [
                "Both are limited by exactly the same bottleneck: available GPU memory, in every case",
                "Data parallelism has no real bottleneck whatsoever, at any scale, no matter how large",
                "Model parallelism is limited by loss-scaling overflow, the same issue as fp16 training",
                "Data parallelism is limited by communication, the all-reduce that averages gradients",
            ],
            "correct": 3,
            "explanation": "Data parallelism's cost is the all-reduce communication step; model (pipeline) parallelism's cost is devices sitting idle unless multiple microbatches keep every stage busy.",
        },
    ],
}
