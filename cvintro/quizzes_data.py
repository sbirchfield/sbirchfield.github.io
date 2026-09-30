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
                "<code>(channels, height, width)</code>",
                "<code>(height, width, channels)</code>",
                "<code>(width, height, channels)</code>",
                "<code>(channels, width, height)</code>",
            ],
            "correct": 1,
            "explanation": "<code>array.shape</code> for a color image is <code>(height, width, channels)</code> -- rows first, then columns, then color channel.",
        },
        {
            "question": "In NumPy indexing <code>img[row, col]</code>, what does <code>row</code> correspond to?",
            "choices": [
                "x, the horizontal position",
                "y, the vertical position",
                "the color channel",
                "Nothing -- row indexes color depth",
            ],
            "correct": 1,
            "explanation": "The first array axis is the row, which moves down the image -- that's y, not x.",
        },
        {
            "question": "<code>cv2.imread</code> loads a color image with channels in what order?",
            "choices": ["RGB", "BGR", "Grayscale by default", "Alphabetical"],
            "correct": 1,
            "explanation": "OpenCV uses BGR order, the opposite of RGB that Matplotlib and most other libraries expect.",
        },
        {
            "question": "Passing a <code>cv2.imread</code>-loaded image straight to <code>plt.imshow</code> without converting typically produces what?",
            "choices": [
                "An identical image to what OpenCV shows",
                "A grayscale image",
                "An image with red and blue channels swapped",
                "An error, since Matplotlib can't read OpenCV arrays",
            ],
            "correct": 2,
            "explanation": "Matplotlib interprets the array as RGB, so a BGR array renders with red and blue reversed.",
        },
    ],
    2: [
        {
            "question": "With NumPy's default <code>uint8</code> arithmetic, what happens when you add 80 to a pixel of value 220?",
            "choices": [
                "It saturates at 255",
                "It wraps around to a smaller value (<code>300 mod 256 = 44</code>)",
                "It raises an <code>OverflowError</code>",
                "NumPy automatically upgrades to a larger integer type",
            ],
            "correct": 1,
            "explanation": "<code>uint8</code> wraps around like an odometer: <code>220 + 80 = 300</code>, which overflows to <code>300 mod 256 = 44</code>.",
        },
        {
            "question": "What does \"saturating arithmetic\" (as used by <code>cv2.add</code>) do differently from plain NumPy <code>uint8</code> addition?",
            "choices": [
                "It wraps around instead of clamping",
                "It clamps results to the valid range instead of wrapping",
                "It converts to floating point automatically",
                "It's identical to NumPy's behavior",
            ],
            "correct": 1,
            "explanation": "<code>cv2.add</code> clamps (saturates) at 0 and 255 rather than silently wrapping around.",
        },
        {
            "question": "Why does difference imaging typically take the absolute value of the subtraction?",
            "choices": [
                "So the result is meaningful regardless of which image is brighter at a given pixel",
                "Because <code>cv2.subtract</code> requires positive inputs",
                "To artificially increase contrast",
                "Absolute value isn't actually used in difference imaging",
            ],
            "correct": 0,
            "explanation": "Without <code>abs()</code>, a pixel that got darker and one that got brighter by the same amount would look different, even though both changed.",
        },
        {
            "question": "What does <code>cv2.addWeighted(a, alpha, b, beta, gamma)</code> compute?",
            "choices": [
                "<code>a*alpha + b*beta + gamma</code>",
                "<code>(a + b) / (alpha + beta) + gamma</code>",
                "<code>alpha*a - beta*b + gamma</code>",
                "<code>max(alpha*a, beta*b) + gamma</code>",
            ],
            "correct": 0,
            "explanation": "It's a saturating weighted sum of the two images plus a constant offset <code>gamma</code>.",
        },
    ],
    3: [
        {
            "question": "What does thresholding do to a grayscale image?",
            "choices": [
                "Converts every pixel above a cutoff to white and at-or-below to black (or vice versa)",
                "Blurs the image to remove noise",
                "Converts the image to a different color space",
                "Computes the image's histogram",
            ],
            "correct": 0,
            "explanation": "Thresholding is a per-pixel comparison against a cutoff value, producing a binary image.",
        },
        {
            "question": "What criterion does Otsu's method use to automatically choose a threshold?",
            "choices": [
                "It picks the median pixel value",
                "It minimizes within-class variance (equivalently, maximizes between-class variance)",
                "It always picks 128",
                "It maximizes the number of white pixels",
            ],
            "correct": 1,
            "explanation": "Otsu searches every possible cutoff for the one that best separates the histogram into two low-variance classes.",
        },
        {
            "question": "What effect does erosion have on a white region in a binary image?",
            "choices": [
                "Grows it, filling small holes",
                "Shrinks it, removing small specks",
                "Leaves it unchanged but smooths edges",
                "Inverts black and white",
            ],
            "correct": 1,
            "explanation": "Erosion only keeps a pixel white if the entire structuring element fits inside the white region, so regions shrink.",
        },
        {
            "question": "What is \"opening\" (erosion followed by dilation) typically used for?",
            "choices": [
                "Filling small black holes inside a white region",
                "Removing small white specks while restoring the size of larger regions",
                "Increasing image contrast",
                "Detecting edges",
            ],
            "correct": 1,
            "explanation": "Erosion first wipes out small specks entirely; the following dilation grows the surviving (larger) regions back toward their original size.",
        },
    ],

    4: [
        {
            "question": "What is the main difference between flood fill and connected-component labeling?",
            "choices": [
                "Flood fill grows a region from a single seed point; connected components find all separate regions in the image at once",
                "They do exactly the same thing, just with different function names",
                "Connected components only work on color images",
                "Flood fill labels every blob with a unique ID simultaneously",
            ],
            "correct": 0,
            "explanation": "Flood fill needs one seed per region, while <code>cv2.connectedComponentsWithStats</code> scans the whole image and labels every blob at once.",
        },
        {
            "question": "In the classic stack-based <code>flood_fill_stack</code> algorithm, what happens when a pixel is popped from the stack?",
            "choices": [
                "It's colored immediately, regardless of whether it was visited before",
                "If it's foreground and not yet filled, it's marked filled and its 4-connected neighbors are pushed onto the stack",
                "The algorithm terminates immediately",
                "It's deleted from the image array",
            ],
            "correct": 1,
            "explanation": "Popped pixels that are background or already filled are simply skipped; otherwise they're marked filled and their neighbors are added to the frontier.",
        },
        {
            "question": "What does <code>cv2.connectedComponentsWithStats</code> label as component <code>0</code>?",
            "choices": [
                "The largest blob",
                "The background",
                "The smallest blob",
                "An error code meaning no blobs were found",
            ],
            "correct": 1,
            "explanation": "Label 0 is always the background, so the number of actual blobs is <code>num_labels - 1</code>.",
        },
        {
            "question": "What extra information does <code>cv2.connectedComponentsWithStats</code> provide alongside the labels, at almost no extra computational cost?",
            "choices": [
                "Bounding box, area, and centroid for each blob",
                "A trained classifier for each blob's shape",
                "The Hu moments of each blob",
                "The original unthresholded grayscale image",
            ],
            "correct": 0,
            "explanation": "The lesson notes these stats come essentially free as part of the same labeling pass.",
        },
    ],
    5: [
        {
            "question": "What does the raw moment <code>m00</code> represent for a binary image?",
            "choices": [
                "The area (foreground pixel count)",
                "The x-coordinate of the centroid",
                "The shape's orientation angle",
                "Always zero, by definition",
            ],
            "correct": 0,
            "explanation": "<code>m00 = sum I(x,y)</code> over all pixels, which for a binary image is just a count of foreground pixels.",
        },
        {
            "question": "Why are central moments <code>mu_pq</code> preferred over raw moments <code>m_pq</code> for describing a shape's spread?",
            "choices": [
                "Central moments are measured around the shape's own centroid, so they don't change if the shape is translated",
                "Central moments are always integers",
                "Central moments are faster to compute than raw moments",
                "There is no real difference between them",
            ],
            "correct": 0,
            "explanation": "Raw moments (except <code>m00</code>) depend on where the shape sits in the image; central moments are translation-invariant.",
        },
        {
            "question": "What property makes Hu moments useful for comparing shapes?",
            "choices": [
                "They are (nearly) invariant to the shape's translation, scale, and rotation",
                "They uniquely determine the shape's color",
                "They only work on convex shapes",
                "They are always exactly equal to 1",
            ],
            "correct": 0,
            "explanation": "The lesson shows a translated and a rotated+scaled version of the same shape produce similar Hu moments, while a different shape does not.",
        },
        {
            "question": "The orientation formula <code>theta = 0.5 * atan2(2*mu11, mu20 - mu02)</code> is computed from which moments?",
            "choices": [
                "The raw moments <code>m10</code>, <code>m01</code>",
                "The central second-order moments <code>mu20</code>, <code>mu02</code>, <code>mu11</code>",
                "The Hu moments",
                "<code>m00</code> alone",
            ],
            "correct": 1,
            "explanation": "The second-order central moments describe the shape's spread around its centroid, from which the major-axis angle is derived.",
        },
    ],
    6: [
        {
            "question": "For a symmetric 2x2 matrix (like the covariance matrices built in this lesson), what special property do its eigenvectors have?",
            "choices": [
                "They are always identical to each other",
                "They are perpendicular to each other, with real eigenvalues",
                "They are complex numbers",
                "There is only ever one eigenvector",
            ],
            "correct": 1,
            "explanation": "A symmetric 2x2 matrix always has two perpendicular (orthogonal) eigenvectors with real eigenvalues.",
        },
        {
            "question": "In the covariance matrix built from a blob's central moments, what do the eigenvectors represent geometrically?",
            "choices": [
                "The blob's major and minor axis directions",
                "The blob's color channels",
                "The blob's centroid coordinates",
                "Random directions with no particular meaning",
            ],
            "correct": 0,
            "explanation": "Diagonalizing the covariance matrix rotates into the frame where the blob's spread in x and y no longer mixes -- exactly its major/minor axes.",
        },
        {
            "question": "What does an eccentricity close to 0 indicate about a shape?",
            "choices": [
                "It's nearly circular (major and minor axes about equal length)",
                "It's very elongated, like a thin line",
                "It has a hole in it",
                "It's not a valid binary shape",
            ],
            "correct": 0,
            "explanation": "Eccentricity ranges from 0 (a circle) to nearly 1 (a very thin, elongated shape).",
        },
        {
            "question": "This lesson notes that eigendecomposing a covariance matrix of general (not necessarily image) data is known as what technique?",
            "choices": [
                "The Fourier transform",
                "Principal component analysis (PCA)",
                "Otsu's method",
                "The Hough transform",
            ],
            "correct": 1,
            "explanation": "PCA is the same diagonalization operation applied to a covariance matrix of general data, ranking eigenvectors by eigenvalue as principal components.",
        },
    ],
    7: [
        {
            "question": "Which point-to-point distance metric produces diamond-shaped equal-distance contours?",
            "choices": [
                "Manhattan / city-block (<code>D4</code>)",
                "Chessboard (<code>D8</code>)",
                "Euclidean",
                "Chamfer",
            ],
            "correct": 0,
            "explanation": "Manhattan distance (<code>|dx| + |dy|</code>) forms diamonds; chessboard forms squares; Euclidean forms circles.",
        },
        {
            "question": "Why does the plain Freeman chain-code estimate <code>L = N_e + sqrt(2)*N_o</code> systematically overestimate a smooth curve's true length?",
            "choices": [
                "The digitized boundary zig-zags through many short staircase steps that add up to more than the true arc length",
                "Because it double-counts every boundary pixel",
                "Because <code>sqrt(2)</code> is too small a weight",
                "It actually underestimates, not overestimates",
            ],
            "correct": 0,
            "explanation": "A digitized circle's boundary alternates through many short zig-zag steps, which sum to more length than the smooth true arc.",
        },
        {
            "question": "What is the key computational advantage of the chamfer distance transform over an exact Euclidean distance transform?",
            "choices": [
                "It approximates the result with two fast raster-scan passes instead of comparing every pixel against every foreground pixel",
                "It only works on already-thresholded color images",
                "It ignores background pixels entirely",
                "It requires no distance function of any kind",
            ],
            "correct": 0,
            "explanation": "The chamfer transform propagates local distances in two raster-scan passes, a fast approximation to the more expensive exact Euclidean distance transform.",
        },
        {
            "question": "What does the chessboard distance <code>D8</code> correspond to physically?",
            "choices": [
                "The number of king moves on a chessboard between two squares",
                "The ordinary straight-line distance",
                "The area of the smallest enclosing square",
                "The number of pawn moves between two squares",
            ],
            "correct": 0,
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
                "Leaves the image unchanged",
            ],
            "correct": 0,
            "explanation": "<code>code=1</code> is a horizontal flip, <code>code=0</code> is vertical, and <code>code=-1</code> flips both.",
        },
        {
            "question": "Which of the three transform types (Euclidean, similarity, affine) is capable of shearing a square into a parallelogram?",
            "choices": [
                "Euclidean only",
                "Similarity only",
                "Affine",
                "None of them can shear a square",
            ],
            "correct": 2,
            "explanation": "Affine transforms allow any invertible 2x2 matrix, including shear, which neither Euclidean nor similarity transforms permit.",
        },
        {
            "question": "What does a similarity transform preserve that a general affine transform does not necessarily preserve?",
            "choices": [
                "Angles (and ratios of lengths)",
                "Nothing -- they preserve exactly the same properties",
                "Parallelism of lines",
                "Exact pixel values",
            ],
            "correct": 0,
            "explanation": "Similarity transforms preserve angles and length ratios; affine transforms give up angle preservation in exchange for allowing shear.",
        },
        {
            "question": "How many point correspondences does <code>cv2.getAffineTransform</code> require to solve for all 6 affine parameters?",
            "choices": ["2", "3", "4", "6"],
            "correct": 1,
            "explanation": "3 point correspondences give exactly 6 equations, matching the 6 free parameters of an affine transform.",
        },
    ],

    9: [
        {
            "question": "What fundamental problem does forward mapping (pushing each source pixel to its transformed location) have?",
            "choices": [
                "It is too slow to run on large images",
                "It can leave gaps (holes) in the destination image with no assigned value",
                "It only works for translations, not rotations",
                "It requires the transform to be invertible",
            ],
            "correct": 1,
            "explanation": "When a transform enlarges or rotates an image, destination locations spread out and some destination pixels receive no source pixel at all.",
        },
        {
            "question": "Why does inverse mapping avoid the holes problem that forward mapping has?",
            "choices": [
                "It iterates over a complete destination grid and looks up each pixel's source location",
                "It only works on grayscale images",
                "It rounds all coordinates to integers before transforming",
                "It skips pixels near the image border",
            ],
            "correct": 0,
            "explanation": "Since every destination pixel is visited and mapped back via <code>M_inv</code>, every output pixel is guaranteed to get a value.",
        },
        {
            "question": "Why does inverse mapping need interpolation at all?",
            "choices": [
                "Because the inverse matrix is not always defined",
                "Because <code>M_inv</code> applied to a destination pixel almost never lands exactly on an integer source coordinate",
                "Because OpenCV requires interpolation by default",
                "Because the source image is always smaller than the destination",
            ],
            "correct": 1,
            "explanation": "The mapped-back coordinate is generally fractional, so a value must be estimated from the surrounding source pixels.",
        },
        {
            "question": "What is the key trade-off between nearest-neighbor and bilinear interpolation?",
            "choices": [
                "Nearest-neighbor is smoother but slower",
                "Bilinear is blockier but faster",
                "Nearest-neighbor is blocky/pixelated; bilinear is smoother but slightly blurrier",
                "There is no difference for enlarging transforms",
            ],
            "correct": 2,
            "explanation": "Bilinear blends the 4 neighboring pixels, removing the blocky aliasing nearest-neighbor produces at the cost of a softer image.",
        },
    ],
    10: [
        {
            "question": "What is the key difference between true convolution and correlation?",
            "choices": [
                "Convolution flips the kernel 180 degrees before sliding it; correlation does not",
                "Correlation only works on grayscale images",
                "Convolution requires a square kernel; correlation does not",
                "There is no difference; the terms are interchangeable in all cases",
            ],
            "correct": 0,
            "explanation": "Convolution flips the kernel before sliding; for symmetric kernels this makes no difference, but for asymmetric kernels (like Sobel) it does.",
        },
        {
            "question": "Does <code>cv2.filter2D</code> compute convolution or correlation?",
            "choices": [
                "Convolution",
                "Correlation",
                "Neither -- it computes a Fourier transform",
                "It alternates depending on kernel size",
            ],
            "correct": 1,
            "explanation": "Most image libraries, including OpenCV's <code>cv2.filter2D</code>, actually implement correlation despite being casually called \"convolution\".",
        },
        {
            "question": "What does it mean for a 2D kernel to be separable?",
            "choices": [
                "It can only be applied to separate color channels",
                "It can be written as the outer product of two 1D kernels, turning O(n^2) work into O(2n)",
                "It must be split into two passes regardless of its structure",
                "It has no effect on the image",
            ],
            "correct": 1,
            "explanation": "A separable kernel like the Gaussian can be applied as a 1D pass along x then a 1D pass along y, which is much cheaper than the full 2D kernel.",
        },
        {
            "question": "In the Sobel edge-detection example, why does the convolution-vs-correlation distinction actually matter?",
            "choices": [
                "Because Sobel only works with correlation",
                "Because the Sobel kernel is symmetric",
                "Because the Sobel kernel is asymmetric, so flipping it changes the result (typically a sign flip)",
                "It never matters for any kernel",
            ],
            "correct": 2,
            "explanation": "For symmetric kernels (box, Gaussian) flipping changes nothing, but asymmetric kernels like Sobel give different results under convolution vs. correlation.",
        },
    ],
    11: [
        {
            "question": "What artifact can occur when an image is downsampled by simply keeping every k-th pixel, without blurring first?",
            "choices": [
                "Aliasing -- fine periodic detail folds into a fake low-frequency pattern",
                "The image becomes larger instead of smaller",
                "Color channels get swapped",
                "The image is unaffected as long as k is even",
            ],
            "correct": 0,
            "explanation": "Fine regular detail (like brick mortar lines) that oscillates faster than the new pixel spacing can represent aliases into noisy-looking fake patterns.",
        },
        {
            "question": "Why does blurring before downsampling prevent aliasing?",
            "choices": [
                "It randomly discards pixels instead of a fixed pattern",
                "It removes fine detail before subsampling, rather than randomly keeping or dropping it pixel by pixel",
                "It increases image contrast",
                "It converts the image to grayscale",
            ],
            "correct": 1,
            "explanation": "Smoothing away fine detail before subsampling avoids the fold-in effect that causes aliasing.",
        },
        {
            "question": "What happens if the Gaussian kernel size is too small for the requested <code>sigma</code>?",
            "choices": [
                "OpenCV raises an error",
                "The kernel is silently reshaped into something close to a plain box average, barely blurring at the requested strength",
                "The image is blurred more strongly than requested",
                "Nothing changes -- kernel size is irrelevant to sigma",
            ],
            "correct": 1,
            "explanation": "A too-small kernel can't represent a wide bell curve, so <code>cv2.getGaussianKernel</code> renormalizes it into something close to a box average.",
        },
        {
            "question": "What is a Gaussian pyramid?",
            "choices": [
                "A single blurred copy of an image",
                "A stack of progressively smaller, blurrier versions of an image, made by repeatedly blurring and downsampling by 2",
                "A histogram of pixel intensities",
                "A stack of images at increasing resolution",
            ],
            "correct": 1,
            "explanation": "Each pyramid level is a properly anti-aliased, half-size version of the previous level, produced by blur-then-downsample.",
        },
    ],
    12: [
        {
            "question": "Why do Sobel and Prewitt kernels combine a difference in one direction with smoothing in the perpendicular direction?",
            "choices": [
                "To make the kernel square",
                "To make the operator robust to noise, unlike a plain [-1, 0, 1] difference",
                "To increase the image's resolution",
                "It's purely a historical convention with no practical effect",
            ],
            "correct": 1,
            "explanation": "Averaging over rows/columns while differencing cuts down the noise response substantially compared to a bare difference kernel.",
        },
        {
            "question": "In the Canny edge detector, what is the purpose of non-maximum suppression?",
            "choices": [
                "To smooth the image before differentiation",
                "To thin wide gradient ridges down to single-pixel-wide lines by keeping only local maxima along the gradient direction",
                "To discard all pixels below the high threshold",
                "To convert the image to binary before computing gradients",
            ],
            "correct": 1,
            "explanation": "Non-maximum suppression keeps a pixel's gradient magnitude only if it's a local maximum along the gradient direction, producing thin edges.",
        },
        {
            "question": "What does Canny's hysteresis thresholding do with pixels between the low and high threshold?",
            "choices": [
                "Discards them all immediately",
                "Keeps them only if they connect to a definite (above-high-threshold) edge pixel",
                "Always keeps them regardless of connectivity",
                "Converts them to the average of the two thresholds",
            ],
            "correct": 1,
            "explanation": "Hysteresis links up weak-but-real edge segments that connect to strong edges, while suppressing isolated noise responses.",
        },
        {
            "question": "In the Hough transform, what does it mean when several edge points' sinusoids in (rho, theta) space all cross at the same point?",
            "choices": [
                "Those edge points are noise and should be discarded",
                "Those edge points are colinear, and the crossing point gives the shared line's parameters",
                "The image contains a circle, not a line",
                "The accumulator has overflowed",
            ],
            "correct": 1,
            "explanation": "Each edge point traces a sinusoid of all lines through it; colinear points' sinusoids intersect at the (rho, theta) of the line they share.",
        },
    ],
    13: [
        {
            "question": "Where does an edge appear in the second derivative of image intensity, as opposed to the first derivative?",
            "choices": [
                "At a peak, same as the first derivative",
                "At a zero crossing -- where the second derivative swings from positive to negative or vice versa",
                "Second derivatives cannot be used to find edges",
                "At the maximum of the original intensity",
            ],
            "correct": 1,
            "explanation": "The first derivative peaks at an edge; the second derivative crosses zero exactly at the edge location.",
        },
        {
            "question": "Why does the Marr-Hildreth detector smooth the image with a Gaussian before applying the Laplacian?",
            "choices": [
                "To increase contrast",
                "Because a second derivative amplifies high-frequency noise even more than a first derivative does",
                "Smoothing is not actually part of the Marr-Hildreth method",
                "To make the image square",
            ],
            "correct": 1,
            "explanation": "Marr and Hildreth's fix is to smooth first, then differentiate -- equivalent to convolving with a single combined Laplacian-of-Gaussian kernel.",
        },
        {
            "question": "What does a Difference of Gaussians (DoG) approximate?",
            "choices": [
                "The image histogram",
                "A scaled Laplacian of Gaussian (LoG), at much cheaper computational cost",
                "The Sobel gradient magnitude",
                "The Hough accumulator",
            ],
            "correct": 1,
            "explanation": "Subtracting two Gaussian blurs at different scales approximates a scaled LoG using only two blurs and a subtraction, instead of an explicit second-derivative kernel.",
        },
        {
            "question": "What does a Laplacian pyramid store at each level that a Gaussian pyramid does not, allowing exact reconstruction?",
            "choices": [
                "A copy of the original image at full resolution",
                "The difference between a Gaussian level and the coarser level upsampled back to match it (the detail otherwise lost)",
                "A histogram of each level",
                "Nothing extra -- Laplacian and Gaussian pyramids store identical information",
            ],
            "correct": 1,
            "explanation": "Each Laplacian level captures exactly what blurring/downsampling would discard, so summing back up the pyramid reconstructs the original exactly.",
        },
    ],

    14: [
        {
            "question": "Why is a median filter especially effective against salt-and-pepper noise, compared to Gaussian blur?",
            "choices": [
                "A single extreme outlier barely moves the median, while it drags the mean toward it",
                "The median filter is linear, so it averages noise away more precisely",
                "Median filtering only works on binary images",
                "Gaussian blur cannot process 8-bit images",
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
                "Its row and column distance from the center, independently",
                "How spatially close it is, and how similar its intensity is to the center pixel",
                "Its intensity value and its position in the image (top vs. bottom)",
                "Only how similar its intensity is to the center pixel",
            ],
            "correct": 1,
            "explanation": "The bilateral filter combines a spatial Gaussian (<code>G_sigma_s</code>) with a range/intensity Gaussian (<code>G_sigma_r</code>), so only neighbors that are both nearby and similar in intensity get high weight.",
        },
        {
            "question": "Why does the bilateral filter preserve a step edge while Gaussian blur smooths across it?",
            "choices": [
                "The bilateral filter has a larger spatial radius than Gaussian blur",
                "Pixels across the edge have very different intensity, so the range weight down-weights them almost to zero",
                "The bilateral filter only operates on binary images",
                "Gaussian blur always ignores pixel intensity entirely",
            ],
            "correct": 1,
            "explanation": "Even though pixels on the other side of an edge may be spatially close, their intensity difference makes the range weight near zero, so they contribute almost nothing to the output.",
        },
    ],
    15: [
        {
            "question": "What does the 2D Fourier transform decompose an image into?",
            "choices": [
                "A sum of 2D sinusoidal gratings at different frequencies and orientations",
                "A set of non-overlapping rectangular blocks",
                "A pyramid of progressively blurred and downsampled copies",
                "A single average brightness value per row",
            ],
            "correct": 0,
            "explanation": "The 2D DFT re-expresses the image as a sum of sinusoidal gratings, each with its own frequency, orientation, magnitude, and phase.",
        },
        {
            "question": "Why does a real-valued sinusoidal grating produce two bright spots (not one) in its Fourier spectrum?",
            "choices": [
                "Because of numerical rounding error in the FFT",
                "A real sine wave is a sum of two complex exponentials at +f and -f, by Euler's formula",
                "Because the image was captured with two cameras",
                "cv2 always duplicates spectrum peaks for display purposes",
            ],
            "correct": 1,
            "explanation": "By Euler's formula, a real sinusoid decomposes into two complex exponentials at frequencies +f and -f, so its spectrum shows two symmetric spikes.",
        },
        {
            "question": "What causes the visible ringing artifacts when applying an ideal (sharp-cutoff) low-pass filter in the frequency domain?",
            "choices": [
                "The image was not converted to grayscale first",
                "A sharp frequency cutoff corresponds to convolving with a spatial sinc kernel, which has infinite ripples (the Gibbs phenomenon)",
                "np.fft.fft2 always introduces rounding artifacts",
                "The mask was not centered correctly",
            ],
            "correct": 1,
            "explanation": "By the convolution theorem, an ideal sharp cutoff in frequency corresponds to a sinc function in space, whose infinite ripples cause the Gibbs phenomenon.",
        },
        {
            "question": "According to the convolution theorem, multiplying two images' Fourier transforms elementwise and inverse-transforming is equivalent to what operation in the spatial domain?",
            "choices": [
                "Cropping the two images to the same size",
                "Convolving the two images",
                "Subtracting the two images",
                "Stacking the two images as separate channels",
            ],
            "correct": 1,
            "explanation": "The convolution theorem states that convolution in the spatial domain equals elementwise multiplication in the frequency domain, and vice versa.",
        },
    ],
    16: [
        {
            "question": "What does the Fourier transform (Lesson 15) fail to tell you, that wavelets and Gabor filters are designed to capture?",
            "choices": [
                "The overall brightness of the image",
                "Which frequencies are present, without knowing where in the image they occur",
                "The image's color channels",
                "The size of the image in pixels",
            ],
            "correct": 1,
            "explanation": "Sine wave basis functions extend across the whole image, so the Fourier transform loses spatial localization; wavelets and Gabor filters use spatially localized oscillations instead.",
        },
        {
            "question": "In the Haar wavelet transform, what do the approximation (<code>a_k</code>) and detail (<code>d_k</code>) coefficients represent?",
            "choices": [
                "The maximum and minimum of each pair of samples",
                "A local average and a local difference of each pair of samples",
                "The real and imaginary parts of a Fourier coefficient",
                "Two unrelated random projections of the signal",
            ],
            "correct": 1,
            "explanation": "<code>a_k</code> is the (scaled) sum of a sample pair, i.e. a local average; <code>d_k</code> is the (scaled) difference, i.e. a local detail.",
        },
        {
            "question": "In a one-level 2D wavelet decomposition of an image, what does the LH subband respond to?",
            "choices": [
                "A half-resolution blurred copy of the whole image",
                "Horizontal edges",
                "Vertical edges",
                "Diagonal detail and corners",
            ],
            "correct": 1,
            "explanation": "LH means low-pass along rows, high-pass along columns, which makes it sensitive to horizontal edges.",
        },
        {
            "question": "What does a Gabor filter's response tell you when the filter is tuned to one orientation and applied to bars drawn at several different orientations?",
            "choices": [
                "It responds equally to all orientations regardless of tuning",
                "It responds most strongly to the bar whose orientation matches the filter's tuning",
                "It only responds to bars drawn in color, not grayscale",
                "It responds most strongly to the orientation perpendicular to its tuning, with no exceptions",
            ],
            "correct": 1,
            "explanation": "The response matrix in the lesson is diagonal-dominant: each filter's peak response lands on the bar matching its own tuned orientation, demonstrating orientation selectivity.",
        },
    ],
    17: [
        {
            "question": "What kind of image content does run-length encoding (RLE) compress well?",
            "choices": [
                "Images with large flat regions of constant value",
                "Photographs with fine-grained noise throughout",
                "Any image, regardless of content",
                "Only images that are already JPEG-compressed",
            ],
            "correct": 0,
            "explanation": "RLE stores (value, run length) pairs for consecutive identical pixels, so it shines on flat regions but does nothing useful when pixel values rarely repeat, as in noisy photographs.",
        },
        {
            "question": "What does Shannon entropy represent in the context of lossless image compression?",
            "choices": [
                "The exact number of bits Huffman coding will always achieve",
                "The theoretical floor on average bits/pixel for any code based only on the value distribution",
                "The maximum possible file size after compression",
                "The number of unique pixel values in the image",
            ],
            "correct": 1,
            "explanation": "Entropy H = -sum(p_i * log2(p_i)) is the theoretical minimum average bits/pixel achievable by any prefix code based on the symbol probabilities; Huffman coding gets close to, but does not always exactly reach, this bound.",
        },
        {
            "question": "In JPEG compression, what is the main effect of quantizing (dividing and rounding) the DCT coefficients of an 8x8 block?",
            "choices": [
                "It converts the image to grayscale",
                "It sets many coefficients, especially high-frequency ones, to exactly zero, discarding detail with little energy",
                "It increases the resolution of the image",
                "It removes the need for Huffman coding afterward",
            ],
            "correct": 1,
            "explanation": "Quantization divides coefficients by values from a quantization table and rounds, zeroing out many low-energy (mostly high-frequency) coefficients so they cost almost nothing to store.",
        },
        {
            "question": "Why is JPEG generally a poor choice for compressing graphics like screenshots or text, compared to PNG?",
            "choices": [
                "JPEG cannot encode images with sharp edges at all",
                "Sharp edges spread energy across all DCT frequencies, so quantization causes visible ringing/blocking around edges and text",
                "PNG is always a lossy format, so it looks better",
                "JPEG only supports grayscale images",
            ],
            "correct": 1,
            "explanation": "A sharp edge's energy is spread across every DCT frequency (like the cross pattern in a Fourier spectrum), so the same quantization that photos tolerate produces visible ringing and blocking artifacts around graphics' edges and text.",
        },
    ],
    18: [
        {
            "question": "Why does the lesson prefer <code>Y = 0.299 R + 0.587 G + 0.114 B</code> over a plain average <code>(R + G + B)/3</code> for grayscale conversion?",
            "choices": [
                "The plain average is not mathematically valid for 8-bit images",
                "The human visual system is more sensitive to green, so weighting it more matches perceived brightness",
                "The plain average only works on already-grayscale images",
                "There is no real difference between the two formulas",
            ],
            "correct": 1,
            "explanation": "A plain average can't tell green and blue apart perceptually; weighting toward green (as human vision does) gives a much better brightness estimate.",
        },
        {
            "question": "In HSV, which channel should you threshold on to segment an object by color robustly, even under a lighting gradient across its surface?",
            "choices": [
                "Value", "Saturation", "Hue", "All three channels equally",
            ],
            "correct": 2,
            "explanation": "Hue captures which color something is, independent of how bright or washed-out it appears, so it stays consistent even as brightness varies across the object.",
        },
        {
            "question": "Why does video/JPEG compression discard resolution from the chroma channels (<code>Cb</code>, <code>Cr</code>) more aggressively than from luma (<code>Y</code>)?",
            "choices": [
                "Chroma channels take up more storage than luma by default",
                "The human visual system resolves fine spatial detail in luma far better than in chroma, so chroma loss is less noticeable",
                "Luma cannot be compressed at all",
                "Cb and Cr are redundant copies of Y",
            ],
            "correct": 1,
            "explanation": "Chroma subsampling exploits the fact that losing the same amount of detail is far less objectionable in color than in brightness.",
        },
        {
            "question": "Why was the L*a*b* color space designed the way it was?",
            "choices": [
                "So that Euclidean distance between two Lab colors approximates perceived color difference consistently",
                "So that it uses less memory per pixel than RGB",
                "So that it is faster to convert to grayscale than RGB",
                "So that hue and brightness are always identical",
            ],
            "correct": 0,
            "explanation": "Equal RGB distances can correspond to wildly different perceived differences depending on where they fall in color space; Lab was designed so Euclidean distance tracks perception much more consistently.",
        },
    ],
    19: [
        {
            "question": "What is the key difference between k-means and a Gaussian mixture model (GMM)?",
            "choices": [
                "GMM allows elliptical clusters and soft (probabilistic) assignment; k-means only allows round clusters and hard assignment",
                "k-means always outperforms GMM on every dataset",
                "GMM cannot handle more than 2 clusters",
                "There is no meaningful difference; they are the same algorithm",
            ],
            "correct": 0,
            "explanation": "GMM generalizes k-means by giving each cluster its own covariance shape and computing a probability of membership for every point, rather than a single hard label.",
        },
        {
            "question": "Why does k-means fail on two concentric rings, landing at chance accuracy?",
            "choices": [
                "k-means can only handle 3D data, not 2D",
                "k-means's update rule can only carve space into convex, center-based regions, and rings aren't convex",
                "The rings dataset has too many points for k-means to process",
                "k-means requires labeled training data",
            ],
            "correct": 1,
            "explanation": "Assigning each point to its nearest center can only produce straight-line boundaries between clusters, which cannot separate one ring from another.",
        },
        {
            "question": "In DBSCAN, what makes a point a <code>core point</code>?",
            "choices": [
                "It is the single point closest to the true cluster center",
                "At least <code>min_samples</code> other points lie within distance <code>eps</code> of it",
                "It has the highest density of any point in the dataset",
                "It was the first point visited by the algorithm",
            ],
            "correct": 1,
            "explanation": "A core point is defined purely by local density: enough neighbors within a fixed radius, with no notion of a cluster center at all.",
        },
        {
            "question": "What can DBSCAN do that neither k-means nor GMM can, when clutter/noise points are added to a dataset?",
            "choices": [
                "Run faster than k-means on any dataset",
                "Automatically discover the correct number of clusters every time",
                "Label a point as \"noise,\" rather than being forced to assign it to some cluster",
                "Guarantee a globally optimal clustering regardless of initialization",
            ],
            "correct": 2,
            "explanation": "k-means and GMM assign every point to some cluster no matter what; DBSCAN has an explicit noise category for points that aren't part of any sufficiently dense region.",
        },
    ],

    20: [
        {
            "question": "The structure tensor <code>M</code> built from image gradients is exactly analogous to which earlier concept?",
            "choices": [
                "The moment covariance matrix from Lesson 6, but built from gradients instead of pixel coordinates",
                "The Hough accumulator from Lesson 12",
                "The DCT quantization table from Lesson 17",
                "The Laplacian pyramid from Lesson 13",
            ],
            "correct": 0,
            "explanation": "Both are 2x2 covariance-style matrices; Lesson 6 built one from pixel coordinates, this lesson builds the same kind of matrix from local image gradients.",
        },
        {
            "question": "What do both eigenvalues of the structure tensor being large indicate about a point?",
            "choices": [
                "It's in a flat region",
                "It's on an edge",
                "It's a corner -- gradient strong in every direction",
                "It's outside the image bounds",
            ],
            "correct": 2,
            "explanation": "A corner has strong gradient in every direction, which is exactly what two large eigenvalues of M mean.",
        },
        {
            "question": "Why are Harris and Shi-Tomasi corner detectors NOT scale-invariant, motivating SIFT?",
            "choices": [
                "They only work on grayscale images",
                "They operate at a single, fixed window size, so a corner can stop looking like a corner when zoomed out",
                "They require manually labeled training data",
                "They cannot run on images larger than 100x100",
            ],
            "correct": 1,
            "explanation": "A sharp corner becomes a gentle curve at a fixed window size when the image is zoomed out enough, so a fixed-scale detector misses it.",
        },
        {
            "question": "What does Lowe's ratio test do when matching SIFT descriptors?",
            "choices": [
                "It keeps a match only if the best candidate is meaningfully closer than the second-best candidate",
                "It discards all matches farther than a fixed pixel distance",
                "It only keeps matches within the same color channel",
                "It requires the two images to be exactly the same size",
            ],
            "correct": 0,
            "explanation": "Comparing the best match's distance to the second-best's rejects ambiguous matches where two candidates are nearly equally good.",
        },
    ],
    21: [
        {
            "question": "What is the aperture problem?",
            "choices": [
                "A camera lens defect that blurs moving objects",
                "Motion along an edge (rather than perpendicular to it) produces no visible local change, so it can't be recovered from a small local window",
                "The fact that optical flow only works on grayscale video",
                "A limitation that only appears in Horn-Schunck, not Lucas-Kanade",
            ],
            "correct": 1,
            "explanation": "Looking through a small window at a moving edge, only the motion perpendicular to the edge is visible; motion along the edge is invisible locally.",
        },
        {
            "question": "In Lucas-Kanade, why does the flow estimate at a corner tend to be reliable while at an edge it is not?",
            "choices": [
                "Corners are always brighter than edges",
                "The same matrix M (the structure tensor) must be well-conditioned to solve the flow system, and a corner has two large eigenvalues while an edge has only one",
                "Edges move faster than corners in real video",
                "Lucas-Kanade cannot process edges at all, only corners",
            ],
            "correct": 1,
            "explanation": "Solving the Lucas-Kanade system requires M to be invertible; a corner (two large eigenvalues) is well-conditioned, while an edge (one small eigenvalue) is ill-conditioned -- the aperture problem again.",
        },
        {
            "question": "Why does <code>cv2.calcOpticalFlowPyrLK</code> use an image pyramid and iterate, rather than solving once?",
            "choices": [
                "To save memory only, with no accuracy benefit",
                "Because the underlying linear (Taylor) approximation is only valid for small motions, so large motions are estimated coarse-to-fine and refined",
                "Because color images require multiple passes",
                "Pyramids are only used for display purposes, not for the actual flow computation",
            ],
            "correct": 1,
            "explanation": "Estimating coarsely on a small, blurry pyramid level first, then refining level by level, lets large motions be recovered even though the linear approximation only holds locally, one small step at a time.",
        },
        {
            "question": "What does the Horn-Schunck smoothness term let flow estimates do that Lucas-Kanade's purely local windows cannot?",
            "choices": [
                "Run faster than Lucas-Kanade in every case",
                "Propagate reliable flow from edges into flat, textureless regions with no local information of their own",
                "Avoid the aperture problem entirely, even at a single edge pixel",
                "Work without any brightness constancy assumption",
            ],
            "correct": 1,
            "explanation": "The smoothness term couples every pixel's flow to its neighbors', letting information spread from regions with strong gradients into flat regions that have no data term of their own.",
        },
    ],
    22: [
        {
            "question": "Why does stereo matching on a rectified image pair only need a 1D search instead of a full 2D search?",
            "choices": [
                "Because stereo images are always grayscale",
                "Because for a rectified pair, the corresponding point for any pixel in the left image lies on the same row in the right image",
                "Because disparity is always zero for rectified pairs",
                "Because block matching is inherently limited to 1D",
            ],
            "correct": 1,
            "explanation": "Epipolar geometry for a rectified stereo pair guarantees the match lies on the same row, collapsing the search from 2D to a 1D scan.",
        },
        {
            "question": "According to <code>Z = f * B / d</code>, what happens to estimated depth as disparity <code>d</code> increases?",
            "choices": [
                "Depth increases proportionally",
                "Depth decreases -- nearby objects have large disparity, distant objects have small disparity",
                "Depth is unrelated to disparity",
                "Depth becomes negative",
            ],
            "correct": 1,
            "explanation": "Depth is inversely proportional to disparity: nearer surfaces shift more (larger d) and so have smaller Z.",
        },
        {
            "question": "What does the left-right consistency check do in stereo matching?",
            "choices": [
                "It converts the disparity map to color for visualization",
                "It matches both left-to-right and right-to-left, then keeps only pixels where the two results agree, discarding the rest as unreliable",
                "It blurs the disparity map to remove noise",
                "It doubles the disparity search range",
            ],
            "correct": 1,
            "explanation": "A pixel that disagrees between the two matching directions by more than about a pixel is flagged as unreliable, which is especially effective at catching occlusions.",
        },
        {
            "question": "What is the trade-off of increasing the block-matching window size in stereo matching?",
            "choices": [
                "Larger windows are always strictly better with no downside",
                "Larger windows are more robust to noise but blur across depth discontinuities; smaller windows preserve sharp boundaries but are more easily fooled by noise",
                "Window size has no effect on the result",
                "Larger windows only affect runtime, never the resulting disparity values",
            ],
            "correct": 1,
            "explanation": "A bigger window averages over more pixels (robust to noise) but mixes pixels from two different true depths near object edges (blurred boundaries); a smaller window has the opposite trade-off.",
        },
    ],
    23: [
        {
            "question": "Why does ordinary least squares (OLS) give a biased line fit when both x and y coordinates are noisy?",
            "choices": [
                "OLS can only be used with integer coordinates",
                "OLS measures error only vertically, which is the right choice only when x is known exactly and y alone is noisy",
                "OLS requires at least 100 data points to work correctly",
                "OLS always overestimates the slope regardless of noise direction",
            ],
            "correct": 1,
            "explanation": "Total least squares (TLS) measures perpendicular distance and treats both coordinates symmetrically, which is more appropriate when both x and y carry comparable noise.",
        },
        {
            "question": "Why does ordinary least-squares fitting break catastrophically under outliers?",
            "choices": [
                "It minimizes the sum of squared residuals, so a single far-away outlier contributes an enormous amount to the total error and drags the whole fit toward it",
                "Least squares cannot be computed when outliers are present -- it simply errors out",
                "Outliers only affect the intercept, never the slope",
                "Squaring residuals makes outliers contribute less, not more",
            ],
            "correct": 0,
            "explanation": "Squaring a large residual makes it dominate the total cost, so the fit moves away from the correct points to reduce that one huge squared term.",
        },
        {
            "question": "What is the core strategy RANSAC uses to fit a model in the presence of outliers?",
            "choices": [
                "It averages every possible model over all the data",
                "It repeatedly picks the smallest possible random subset needed to define a candidate model, counts how many other points agree, and keeps the model with the most agreement",
                "It removes the single most extreme data point and reruns least squares",
                "It requires the user to manually label which points are outliers",
            ],
            "correct": 1,
            "explanation": "RANSAC fits from minimal random samples and picks whichever candidate has the largest consensus set, then does a final least-squares refit on just the inliers.",
        },
        {
            "question": "According to the RANSAC iteration-count formula, why does fitting a homography (minimal sample size 4) require many more iterations than fitting a line (minimal sample size 2) at the same outlier rate?",
            "choices": [
                "Homographies are inherently slower to evaluate on a computer",
                "The probability of drawing an all-inlier sample is <code>w^n</code>, which shrinks rapidly as the minimal sample size n grows",
                "Lines require more iterations than homographies, not fewer",
                "The iteration count does not depend on the minimal sample size at all",
            ],
            "correct": 1,
            "explanation": "Since the chance of an all-inlier sample is w raised to the sample size, a larger minimal sample (4 vs. 2) makes an all-inlier draw far less likely, requiring many more trials for the same confidence.",
        },
    ],
    24: [
        {
            "question": "Why can a homography (3x3 matrix on homogeneous coordinates) represent translation, while a plain 2x2 matrix cannot?",
            "choices": [
                "Homogeneous coordinates add a third coordinate, letting the matrix shift points via the extra row/column instead of needing a separate additive vector",
                "3x3 matrices are always invertible while 2x2 matrices are not",
                "Translation is impossible to represent with any matrix, homogeneous or not",
                "A homography only works on already-translated images",
            ],
            "correct": 0,
            "explanation": "Representing a 2D point as (x, y, 1) lets a 3x3 matrix encode translation directly, unlike a 2x2 matrix acting on (x, y) alone.",
        },
        {
            "question": "What specifically causes a homography to destroy parallelism, unlike an affine transform?",
            "choices": [
                "A nonzero bottom row (g, h) in the 3x3 matrix",
                "Using floating-point instead of integer coordinates",
                "Applying the transform more than once",
                "The presence of a nonzero translation term",
            ],
            "correct": 0,
            "explanation": "An affine transform's bottom row is always (0, 0, 1); a homography's nonzero (g, h) introduces the division effect that makes parallel lines converge.",
        },
        {
            "question": "How does the point-at-infinity trick locate a vanishing point?",
            "choices": [
                "By averaging the endpoints of all visible line segments",
                "By applying the homography to the homogeneous point representing the lines' shared direction (e.g. (1, 0, 0)), which lands exactly on the vanishing point",
                "By computing the image's centroid",
                "It only works for vertical lines, never horizontal ones",
            ],
            "correct": 1,
            "explanation": "A point with zero third homogeneous coordinate represents a direction at infinity; transforming it by H lands exactly where the two parallel lines' images converge.",
        },
        {
            "question": "In what two situations does a single homography exactly relate two photographs of a scene?",
            "choices": [
                "Any two photos taken from any two camera positions of any scene",
                "Photographing a flat surface from two different positions, or photographing any 3D scene from the same camera position while only rotating between shots",
                "Only when the two photos are taken with the exact same camera settings",
                "Only when the scene contains no straight lines",
            ],
            "correct": 1,
            "explanation": "A homography exactly models plane-to-plane relationships (a flat scene) or pure camera rotation about a fixed center; a translating camera viewing a genuinely 3D scene has no single exact homography relating the views.",
        },
    ],

    25: [
        {
            "question": "What four steps does the stitching pipeline in this lesson combine, in order?",
            "choices": [
                "Threshold, morphology, connected components, moments",
                "Detect/match features, estimate a robust homography, warp, composite/blend",
                "Calibrate the camera, undistort, rectify, match rows",
                "Convert to grayscale, blur, edge-detect, Hough transform",
            ],
            "correct": 1,
            "explanation": "The pipeline is: SIFT + ratio-test matching (Lesson 20), RANSAC homography (Lessons 23-24), <code>cv2.warpPerspective</code> (Lesson 9), then feathered blending.",
        },
        {
            "question": "In the final composite, what does the linear feather blend do across the overlap region?",
            "choices": [
                "Picks whichever image has higher average brightness for the whole region",
                "Ramps the blend weight smoothly from fully view 1 to fully view 2 across the overlap",
                "Averages the two images everywhere, including non-overlapping regions",
                "Discards the overlap region entirely, leaving a black gap",
            ],
            "correct": 1,
            "explanation": "A spatially-varying alpha ramp (same weighted-sum idea as <code>cv2.addWeighted</code>, Lesson 2) is used only within the overlap, so the seam fades smoothly rather than showing a hard cut.",
        },
        {
            "question": "With no synthetic ground truth available for a real photo pair, how does the lesson measure how good the estimated homography's fit is?",
            "choices": [
                "It cannot be measured at all for real photos",
                "By comparing the recovered homography's determinant to 1",
                "By computing the reprojection error of the RANSAC inliers themselves under the fitted homography",
                "By counting the number of SIFT keypoints detected",
            ],
            "correct": 2,
            "explanation": "Projecting each inlier through <code>H</code> and measuring distance to its matched point (reprojection error) gives a direct, ground-truth-free measure of geometric fit quality.",
        },
        {
            "question": "According to the lesson, why might OpenCV's default <code>cv2.Stitcher</code> <code>PANORAMA</code> mode bow straight lines when stitching photos of a flat building facade?",
            "choices": [
                "It always fails on outdoor scenes",
                "It assumes the camera rotated about its optical center and warps onto a sphere, which distorts flat rectilinear scenes when the camera actually translated",
                "It requires at least 3 images to work correctly",
                "It only works in grayscale",
            ],
            "correct": 1,
            "explanation": "<code>PANORAMA</code> mode's spherical warp is suited to pure camera rotation; <code>SCANS</code> mode instead composites with planar homographies directly, matching this lesson's approach.",
        },
    ],
    26: [
        {
            "question": "In the pinhole projection formula <code>x = f*X/Z</code>, <code>y = f*Y/Z</code>, what determines how much a 3D point's image position changes as its depth <code>Z</code> increases (with <code>X</code>, <code>Y</code> fixed)?",
            "choices": [
                "The image position is completely independent of <code>Z</code>",
                "Increasing <code>Z</code> moves the projected point further from the image center",
                "Increasing <code>Z</code> shrinks the projected coordinates toward the center, since they scale as <code>1/Z</code>",
                "Depth <code>Z</code> only affects color, not position",
            ],
            "correct": 2,
            "explanation": "Both <code>x</code> and <code>y</code> are divided by <code>Z</code>, so farther points (larger <code>Z</code>) project closer to the image center -- the usual perspective effect.",
        },
        {
            "question": "Why do objects nearer or farther than the focal plane appear blurred (the circle of confusion), according to this lesson?",
            "choices": [
                "Because the sensor's Bayer filter fails at those distances",
                "Because a lens focuses sharply only for one particular distance; other distances spread their light over a small disk on the sensor instead of a point",
                "Because gamma correction distorts out-of-focus regions",
                "Because JPEG compression blurs distant objects more than near ones",
            ],
            "correct": 1,
            "explanation": "Only points at the focal plane converge to a sharp point on the sensor; nearer/farther points spread over a disk whose size grows with distance from the focal plane.",
        },
        {
            "question": "What happens if a raw Bayer mosaic is demosaicked assuming the wrong filter pattern (e.g. BGGR treated as RGGB)?",
            "choices": [
                "OpenCV raises an error and refuses to proceed",
                "The result is identical either way, since demosaicking is pattern-independent",
                "It fails silently: every pixel still gets a value, but colors are badly and systematically wrong",
                "Only the brightness channel is affected, not color",
            ],
            "correct": 2,
            "explanation": "Since every pixel gets *some* value regardless of the assumed pattern, a wrong pattern produces a plausible-looking but strongly color-shifted image rather than an obvious failure.",
        },
        {
            "question": "Why is naively averaging two gamma-encoded pixel values (e.g. for blending or blurring) physically wrong?",
            "choices": [
                "Gamma encoding is nonlinear, so averaging encoded values does not equal averaging the true linear light intensity they represent",
                "Gamma encoding only affects color images, not grayscale",
                "It isn't wrong; encoded values can always be averaged directly",
                "Gamma correction has been removed from all modern image formats",
            ],
            "correct": 0,
            "explanation": "The correct approach is to decode to linear light, average there, then re-encode; averaging directly in encoded space under-represents the true blended brightness, as the lesson's black/white example shows.",
        },
    ],
    27: [
        {
            "question": "What two categories of parameters does camera calibration recover, according to this lesson?",
            "choices": [
                "Focal length and image resolution only",
                "Intrinsics (K) and distortion coefficients",
                "Rotation and translation of the checkerboard only",
                "Color balance and exposure settings",
            ],
            "correct": 1,
            "explanation": "Calibration recovers the ideal pinhole intrinsics matrix <code>K</code> plus the radial/tangential distortion coefficients that describe how a real lens deviates from that ideal.",
        },
        {
            "question": "Why does the lesson generate many synthetic checkerboard views from different angles rather than just one?",
            "choices": [
                "A single view is always sufficient to recover K and distortion exactly",
                "More views from a wider variety of angles provide more constraints, averaging out detector noise and better constraining the estimate",
                "OpenCV's calibrateCamera function requires exactly 40 views to run",
                "Multiple views are only needed to estimate the checkerboard's physical size",
            ],
            "correct": 1,
            "explanation": "With noisy corner detections, a single view underconstrains the problem; many diverse views average out noise and better pin down both K and distortion.",
        },
        {
            "question": "What does <code>k1</code> and <code>k2</code> radial distortion do to straight lines in a scene, when captured through a real lens?",
            "choices": [
                "Nothing -- radial distortion only affects color, not geometry",
                "It bows them into curves (barrel or pincushion, depending on sign)",
                "It always makes them longer without curving them",
                "It removes them entirely from the image",
            ],
            "correct": 1,
            "explanation": "Radial distortion terms in the distortion model curve straight lines; the sign of <code>k1</code>/<code>k2</code> determines whether they bow outward (barrel) or inward (pincushion).",
        },
        {
            "question": "After applying <code>cv2.undistortPoints</code> with the *estimated* (not true) calibration, why don't the grid lines straighten out perfectly?",
            "choices": [
                "Because undistortPoints only works on color images",
                "Because the estimated K and distortion coefficients are themselves only approximately correct, due to detector noise",
                "Because undistortPoints intentionally leaves a small amount of distortion for aesthetic reasons",
                "Because the grid lines were never actually distorted in the first place",
            ],
            "correct": 1,
            "explanation": "Calibration from noisy corner detections only approximates the true K/distortion, so undoing distortion with the estimate leaves a faint residual bow.",
        },
    ],
    28: [
        {
            "question": "What does the epipolar constraint <code>x2^T F x1 = 0</code> tell you about where a point's match must lie in the second image?",
            "choices": [
                "The match could be anywhere in the second image with equal probability",
                "The match is constrained to lie on a specific 1D line (the epipolar line), not just anywhere in 2D",
                "The match must be at the exact same pixel coordinates as in image 1",
                "The match is constrained only if the images have already been rectified",
            ],
            "correct": 1,
            "explanation": "Given <code>x1</code>, the line <code>l2 = F x1</code> is the epipolar line in image 2 that the true match is guaranteed to lie on, collapsing the search from 2D to 1D.",
        },
        {
            "question": "Geometrically, why does a point's match always fall on a single line rather than anywhere in the second image?",
            "choices": [
                "Because both cameras must have identical intrinsics",
                "Because every candidate 3D point along camera 1's ray, together with both camera centers, lies in one epipolar plane, which camera 2 sees edge-on as a line",
                "Because feature descriptors can only match along horizontal lines",
                "This is only true after image rectification",
            ],
            "correct": 1,
            "explanation": "All points along the ray from C1 through x1 share the same epipolar plane with C1 and C2; camera 2 sees that plane edge-on, so every candidate projects onto the same line.",
        },
        {
            "question": "Why does the 8-point algorithm normalize pixel coordinates (centering and rescaling) before solving for F via SVD?",
            "choices": [
                "Normalization is purely cosmetic and has no effect on the result",
                "Raw pixel coordinates (in the hundreds) make the linear system badly conditioned numerically; normalizing dramatically improves the result",
                "OpenCV requires normalized coordinates as an API constraint",
                "Normalization is needed only when F has rank 3",
            ],
            "correct": 1,
            "explanation": "This is the Hartley normalization step: solving in centered, rescaled coordinates avoids numerical ill-conditioning that raw large pixel values would cause.",
        },
        {
            "question": "Why can <code>cv2.recoverPose</code> only recover the *direction* of the translation between the two cameras, not its magnitude?",
            "choices": [
                "Because OpenCV has a bug in that function",
                "Because a scene twice as large viewed by cameras with twice the baseline produces identical images -- an inherent scale ambiguity in two-view geometry",
                "Because the essential matrix only exists for calibrated cameras",
                "Because rotation and translation cannot both be recovered from two views",
            ],
            "correct": 1,
            "explanation": "The scale ambiguity is fundamental to monocular two-view geometry: uniformly scaling the scene and baseline together produces identical images, so absolute scale can't be recovered from correspondences alone.",
        },
    ],
    29: [
        {
            "question": "What does triangulation compute, given two camera projection matrices and a matched 2D point pair?",
            "choices": [
                "The camera's intrinsic calibration matrix K",
                "The 3D position of the point that projects to both observed 2D locations",
                "The fundamental matrix F relating the two views",
                "The optimal feature descriptor for the matched point",
            ],
            "correct": 1,
            "explanation": "Triangulation intersects the two rays (one per camera) that could have produced the matched 2D observations, recovering the 3D point via DLT/SVD.",
        },
        {
            "question": "What is the purpose of the cheirality check (requiring positive depth in both cameras) after triangulation?",
            "choices": [
                "To speed up the SVD computation",
                "To catch mismatched correspondences that pass the epipolar/RANSAC test but triangulate behind a camera, which the epipolar constraint alone doesn't rule out",
                "To convert the reconstruction from sparse to dense",
                "To estimate the camera's focal length",
            ],
            "correct": 1,
            "explanation": "The epipolar constraint doesn't rule out points that triangulate behind a camera; requiring positive depth in both views is a free extra filter that catches such bad matches.",
        },
        {
            "question": "Why does the reconstruction from *estimated* poses need to be multiplied by the true baseline length before it matches the true-pose reconstruction?",
            "choices": [
                "Because recoverPose only returns a translation *direction*, so triangulation from it is correct only up to an unknown overall scale",
                "Because the estimated poses use a different coordinate system entirely",
                "Because triangulation always underestimates distances by a fixed known factor",
                "This scaling step is unnecessary and only done for illustration",
            ],
            "correct": 0,
            "explanation": "Since <code>t</code> from <code>recoverPose</code> is unit-length, every triangulated point comes out a fixed factor smaller than reality; rescaling by the true baseline fixes the metric scale (using outside knowledge, since two views alone can't supply it).",
        },
        {
            "question": "What does bundle adjustment do that distinguishes a real multi-view SfM system from the simple two-view pipeline in this lesson?",
            "choices": [
                "It replaces feature matching with a neural network",
                "It refines all camera poses and 3D points together in one large nonlinear least-squares optimization minimizing total reprojection error across every view",
                "It performs demosaicking on each input photo",
                "It converts the sparse point cloud directly into a dense mesh with no further computation",
            ],
            "correct": 1,
            "explanation": "As more views are added, per-view pose errors compound; bundle adjustment jointly refines every camera pose and 3D point at once to minimize the total reprojection error, correcting for that compounding.",
        },
    ],

    30: [
        {
            "question": "According to this lesson, what do PCA (Lesson 6) and camera projection (P = K[R|t]) have in common?",
            "choices": [
                "Both are examples of the same underlying operation: y = w^T x + b, a linear projection",
                "Both require labeled training data to compute",
                "Neither one involves any matrix multiplication",
                "PCA and camera projection are unrelated operations with no shared structure",
            ],
            "correct": 0,
            "explanation": "Both compute a projection via a dot product between a fixed direction and the input coordinates; they differ only in how that direction was chosen.",
        },
        {
            "question": "What single operation does the lesson show convolution, the Fourier transform, and a homography all secretly are?",
            "choices": [
                "Nonlinear activation functions",
                "A matrix multiplication, i.e. a bank of simultaneous linear projections",
                "Gradient descent updates",
                "Random sampling procedures",
            ],
            "correct": 1,
            "explanation": "Each output value in all three is a dot product between the input and a fixed row/kernel/basis function, which is exactly what one row of a matrix multiply computes.",
        },
        {
            "question": "What can a single linear projection classifier never do, regardless of how w and b are chosen?",
            "choices": [
                "Separate two classes that are linearly separable",
                "Separate a ring of points from a disk of points it surrounds",
                "Compute a dot product",
                "Be visualized in 1D after projection",
            ],
            "correct": 1,
            "explanation": "A single projection can only produce a straight-line (hyperplane) decision boundary, and no straight line can separate a ring from the disk it encloses.",
        },
        {
            "question": "In the classification example, what determines which side of the decision boundary a point falls on?",
            "choices": [
                "The point's distance from the image origin only",
                "The sign of the projected score w^T x + b",
                "Which class has more training examples",
                "The point's color channel values",
            ],
            "correct": 1,
            "explanation": "Points are classified by thresholding the projected score at zero: positive score means one class, negative means the other.",
        },
    ],
    31: [
        {
            "question": "Why does the lesson use the sigmoid function instead of directly thresholding the raw score z = w^T x + b for training?",
            "choices": [
                "Sigmoid makes the model run faster",
                "Raw accuracy is flat almost everywhere and has no useful gradient; sigmoid gives a smooth, differentiable output gradient descent can follow",
                "Sigmoid is required to compute a dot product",
                "Thresholding directly would make predictions always equal to 1",
            ],
            "correct": 1,
            "explanation": "Hard accuracy jumps discontinuously at the boundary with zero gradient almost everywhere, so gradient descent has nothing to follow; sigmoid squashes the score into a smooth probability instead.",
        },
        {
            "question": "In binary cross-entropy, what happens to the loss when the true label is 1 but the predicted probability p is close to 0?",
            "choices": [
                "The loss approaches zero",
                "The loss explodes toward infinity -- confidently wrong is punished severely",
                "The loss is unaffected by how confident the wrong prediction was",
                "The loss becomes negative",
            ],
            "correct": 1,
            "explanation": "The penalty for true label 1 is -log(p), which grows without bound as p approaches 0.",
        },
        {
            "question": "What two independent methods does the lesson use to verify the hand-derived backprop gradient formulas are correct?",
            "choices": [
                "Running the model twice and comparing outputs",
                "Numerical finite-difference gradient checking, and comparing against PyTorch's autograd",
                "Checking that the loss is always positive",
                "Comparing training accuracy to a random baseline",
            ],
            "correct": 1,
            "explanation": "The lesson perturbs each parameter by a tiny amount to estimate the gradient numerically, and separately lets PyTorch's .backward() compute it automatically -- both agree with the hand-derived formula.",
        },
        {
            "question": "Why does gradient descent do *worse* than exhaustive search on the ring-vs-disk dataset, converging to near-chance accuracy?",
            "choices": [
                "Gradient descent always fails on datasets with fewer than 1000 points",
                "The dataset is roughly symmetric around the origin, so the average gradient pull nearly cancels out in every direction, settling the weights near zero",
                "The learning rate was too small to move the weights at all",
                "Sigmoid cannot be used with more than 60 data points",
            ],
            "correct": 1,
            "explanation": "Because the ring and disk are both centered at the origin, the training signal from all points nearly cancels in every direction, so the optimizer settles near w=0 instead of finding an asymmetric corner-case solution.",
        },
    ],
    32: [
        {
            "question": "Why does stacking two purely linear layers (no nonlinearity in between) fail to add any representational power over one layer?",
            "choices": [
                "Two linear layers always train slower than one",
                "The composition of two linear projections is algebraically just one combined linear projection (w2^T W1 collapses to a single matrix)",
                "PyTorch does not allow stacking linear layers",
                "Linear layers can only be stacked if they have the same width",
            ],
            "correct": 1,
            "explanation": "Without a nonlinearity, z = w2^T(W1 x + b1) + b2 simplifies algebraically into a single w'^T x + b', exactly Lesson 31's single neuron again.",
        },
        {
            "question": "What role does the ReLU nonlinearity play in an MLP's hidden layer?",
            "choices": [
                "It has no real effect and is included only by convention",
                "It breaks the algebraic collapse that would otherwise reduce stacked linear layers back into one linear projection",
                "It converts the network into an unsupervised model",
                "It only affects the bias terms, not the weights",
            ],
            "correct": 1,
            "explanation": "Because ReLU is not a linear function, w2^T ReLU(W1 x + b1) + b2 cannot be rewritten as any single linear projection, unlike the all-linear case.",
        },
        {
            "question": "In the capacity sweep, what does H=1 (a single hidden unit) achieve on the ring-vs-disk problem, no matter how long it trains?",
            "choices": [
                "Consistently 100% accuracy",
                "A ceiling near Lesson 30's ~74% linear limit, since one hidden unit is just a single cut",
                "Exactly 0% accuracy every time",
                "Accuracy that improves without bound given enough epochs",
            ],
            "correct": 1,
            "explanation": "One hidden unit provides only a single straight-line cut, so it inherits the same representational ceiling a lone linear projection has -- a capacity limit, not an optimization failure.",
        },
        {
            "question": "When the trained hidden layer's 4D output is projected to 2D with PCA for visualization, what does the lesson observe about the two classes?",
            "choices": [
                "They remain just as tangled as in the original input space",
                "They are pulled apart into something close to linearly separable, even in this lossy 2D snapshot",
                "They collapse into a single overlapping cluster",
                "The visualization is identical to the raw input space plot",
            ],
            "correct": 1,
            "explanation": "The hidden layer reshapes the space so the classes become nearly linearly separable, illustrating that each layer's job is to make the next layer's job easier.",
        },
    ],
    33: [
        {
            "question": "On the narrow, steep-in-y bowl loss surface, why does plain gradient descent zigzag instead of converging smoothly?",
            "choices": [
                "A step size that makes progress along the shallow x-axis overshoots along the steep y-axis, and vice versa",
                "Plain gradient descent cannot handle 2D loss surfaces at all",
                "The loss function has no minimum",
                "The learning rate was set to exactly zero",
            ],
            "correct": 0,
            "explanation": "A single shared step size can't be simultaneously right for a shallow and a steep direction at once, causing the characteristic zigzag.",
        },
        {
            "question": "What is the key mechanism behind momentum's benefit on the narrow bowl?",
            "choices": [
                "It sets the learning rate to zero after a fixed number of steps",
                "It accumulates a running velocity so consistent gradients (the shallow axis) compound into a larger effective step, accelerating progress",
                "It removes the need for a loss function entirely",
                "It only ever slows down training, never speeds it up",
            ],
            "correct": 1,
            "explanation": "Momentum's real benefit is acceleration along consistently-pointing directions, not damping oscillation, since a single beta can't be tuned per-axis.",
        },
        {
            "question": "How does Adam differ from momentum in how it handles the steep vs. shallow axes of the bowl?",
            "choices": [
                "Adam ignores gradient direction entirely and only uses magnitude",
                "Adam divides the step by a running estimate of each parameter's squared gradient, independently rescaling each axis instead of sharing one coefficient",
                "Adam is identical to momentum with a different name",
                "Adam only works on 1-dimensional loss surfaces",
            ],
            "correct": 1,
            "explanation": "Dividing by sqrt(v) per-parameter automatically shrinks steps on the steep axis and boosts them on the shallow one, unlike momentum's single shared coefficient.",
        },
        {
            "question": "Why does initializing all weights to exactly zero fail catastrophically for a hidden layer with more than one unit?",
            "choices": [
                "Zero weights cause a runtime crash in PyTorch",
                "Every hidden unit computes the identical function and receives the identical gradient, so they never differentiate from each other -- the symmetry never breaks",
                "Zero-initialized networks train faster than randomly-initialized ones",
                "Zero initialization only fails when using the Adam optimizer",
            ],
            "correct": 1,
            "explanation": "With identical weights and identical gradients, every unit updates identically forever, so the model is stuck computing a single trivial function no matter how long it trains.",
        },
    ],
    34: [
        {
            "question": "What two problems does a convolutional layer's weight-sharing fix, compared to a fully-connected layer applied to a flattened image?",
            "choices": [
                "It makes training completely deterministic and removes the need for random initialization",
                "It discards spatial structure and requires re-learning the same feature at every position independently",
                "It reduces parameter count by reusing the same small filter everywhere, and gives translation equivariance -- a feature learned at one position is detected everywhere",
                "It eliminates the need for a loss function",
            ],
            "correct": 2,
            "explanation": "Restricting each output to a local patch and sharing that patch's weights across positions both shrinks the parameter count and makes detections shift consistently with the input.",
        },
        {
            "question": "For a 3x3 convolution filter applied to a 3-channel RGB image, how many weights does that single filter actually have?",
            "choices": [
                "9, the same as for a grayscale image",
                "3, one per color channel",
                "27, since the filter spans 3x3 spatially across all 3 input channels at once",
                "It depends on the image's resolution",
            ],
            "correct": 2,
            "explanation": "A conv filter is really 3x3xC_in, so on 3 input channels a '3x3 filter' actually has 3x3x3=27 weights, summed into one output value per position.",
        },
        {
            "question": "What does max pooling add to a CNN, beyond shrinking the spatial size of the feature map?",
            "choices": [
                "A small amount of local translation invariance -- a feature detected at slightly different positions within one pooling window still produces the same output",
                "Exact invariance to any transformation of the input",
                "The ability to skip the convolution step entirely",
                "A guarantee that the network cannot overfit",
            ],
            "correct": 0,
            "explanation": "Keeping only the max value in each window means small position shifts within that window don't change the pooled output.",
        },
        {
            "question": "In the unseen-position generalization test, why does the flatten-based MLP perform near chance on shapes placed at corner positions never seen in training, while the CNN does not?",
            "choices": [
                "The MLP has more parameters than the CNN",
                "The MLP memorized which specific pixels tend to be on for each class at the training positions, which doesn't transfer to new positions; the CNN's global max pooling makes the decision depend on what fired, not where",
                "The CNN was trained for more epochs than the MLP",
                "The MLP only works on grayscale images",
            ],
            "correct": 1,
            "explanation": "Flattening ties the MLP's decision to specific pixel positions, while the CNN's global pooling collapses spatial position, letting it recognize shapes regardless of where they appear.",
        },
    ],
    35: [
        {
            "question": "In the overfitting demonstration with only 12 training images, what pattern do the training and validation loss curves show?",
            "choices": [
                "Both curves decrease together and never diverge",
                "Training loss marches to zero while validation loss bottoms out partway through training, then climbs back up",
                "Both curves increase throughout training",
                "Validation loss is always lower than training loss",
            ],
            "correct": 1,
            "explanation": "The network perfectly memorizes the 12 training images while validation loss reaches a minimum then gets worse as the model overfits further.",
        },
        {
            "question": "How does data augmentation help fight overfitting on a small dataset?",
            "choices": [
                "It deletes the hardest training examples",
                "It manufactures more training variety from the same images via label-preserving transforms (flips, rotations), so the model sees a wider variety of what the class can look like",
                "It increases the learning rate automatically",
                "It removes the need for a validation set",
            ],
            "correct": 1,
            "explanation": "Applying random label-preserving transforms to the existing images gives the model many more effective examples to learn from, rather than memorizing a handful of exact images.",
        },
        {
            "question": "What does weight decay do to fight overfitting?",
            "choices": [
                "It adds a penalty proportional to weight magnitude, discouraging the large, highly-tuned weights that memorization requires",
                "It deletes a random fraction of the training images each epoch",
                "It increases the model's capacity",
                "It forces all weights to become exactly zero",
            ],
            "correct": 0,
            "explanation": "Penalizing large weight magnitudes makes memorizing noisy specifics of a small dataset more costly relative to finding a simpler, smoother function.",
        },
        {
            "question": "In inverted dropout, what happens to the surviving (non-zeroed) activations during training, and why?",
            "choices": [
                "They are left completely unchanged",
                "They are rescaled by 1/(1-p) so the layer's expected output stays at the same overall scale whether or not dropout is active",
                "They are set to exactly 1 regardless of their original value",
                "They are doubled regardless of the dropout probability",
            ],
            "correct": 1,
            "explanation": "Rescaling by 1/(1-p) keeps the expected activation magnitude consistent between training (with dropout) and evaluation (without it).",
        },
    ],

    36: [
        {
            "question": "Why does an overall accuracy of just over 50% on this lesson's 4-class CIFAR-10 task hide something important?",
            "choices": [
                "Because 50% is actually below chance for 4 classes",
                "Because it doesn't reveal that the model is not equally good at all four classes -- truck, in particular, is barely recognized",
                "Because accuracy can only be computed for binary classifiers",
                "Because the test set itself is imbalanced",
            ],
            "correct": 1,
            "explanation": "A single overall-accuracy number averages away per-class differences; here it hides that the truck class (starved to 90 training images) is rarely predicted correctly.",
        },
        {
            "question": "In a confusion matrix where row <code>i</code>, column <code>j</code> counts examples of true class <code>i</code> predicted as class <code>j</code>, what does a perfect classifier's matrix look like?",
            "choices": [
                "All zeros",
                "Uniform across every cell",
                "All the mass on the diagonal (row i, column i for every i)",
                "All the mass in the top-right corner",
            ],
            "correct": 2,
            "explanation": "A correct prediction means predicted class equals true class, i.e. j = i, which is exactly the diagonal.",
        },
        {
            "question": "What does <code>recall</code> measure for a given class?",
            "choices": [
                "Of everything the model called this class, what fraction actually was",
                "Of everything that really was this class, what fraction the model caught",
                "The total number of training examples for this class",
                "How fast the model runs on this class's images",
            ],
            "correct": 1,
            "explanation": "Recall = TP / (TP + FN): low recall means the model misses that class often, regardless of what it says about other classes.",
        },
        {
            "question": "Why does the starved <code>truck</code> class end up with high precision but very low recall?",
            "choices": [
                "Because trucks are visually identical to automobiles in every image",
                "When the model does say 'truck' it's usually right, but it fails to recognize most actual trucks, defaulting to well-represented classes like automobile instead",
                "Because the test set has no truck images at all",
                "Precision and recall are always equal for any class",
            ],
            "correct": 1,
            "explanation": "This is the standard signature of class imbalance: the model rarely guesses the rare class, but when it does, it's confident and usually correct.",
        },
    ],
    37: [
        {
            "question": "What made the ILSVRC benchmark (built on a subset of ImageNet) so important for comparing architectures?",
            "choices": [
                "It was the first dataset to include color images",
                "Every group trained on the same data and was scored on the same held-out test set, making results directly, objectively comparable at unprecedented scale",
                "It only allowed one submission per architecture, ever",
                "It required all models to have exactly the same number of parameters",
            ],
            "correct": 1,
            "explanation": "A shared, standardized competition -- not just a big labeled dataset -- is what let different architectures be compared fairly and drove rapid progress.",
        },
        {
            "question": "According to the lesson, why does stacking more layers make a *plain* (non-residual) deep network worse, not better?",
            "choices": [
                "More layers always increase training time only, with no effect on accuracy",
                "Backprop multiplies the gradient by every layer's local Jacobian on the way back, and if those factors are consistently smaller than 1 (as with saturating activations), the gradient reaching early layers vanishes",
                "Deeper networks always overfit regardless of the dataset size",
                "GPUs cannot execute more than 20 sequential layers",
            ],
            "correct": 1,
            "explanation": "This is the vanishing-gradient problem: repeated multiplication by small per-layer factors drives the gradient toward zero, so early layers stop learning.",
        },
        {
            "question": "Structurally, why does a residual connection (<code>x_{l+1} = x_l + F(x_l)</code>) prevent the gradient from vanishing across many layers?",
            "choices": [
                "It removes the need for any activation function",
                "Its Jacobian is <code>I + dF/dx_l</code>, so an identity term always survives the multiplication chain, giving the gradient an unobstructed shortcut back to the input",
                "It doubles the learning rate automatically",
                "It replaces backpropagation with a different optimization algorithm entirely",
            ],
            "correct": 1,
            "explanation": "The identity matrix in the Jacobian is never shrunk by a small activation derivative, unlike a plain network where every layer's Jacobian is purely <code>dF/dx_l</code>.",
        },
        {
            "question": "What do the two learnable parameters <code>gamma</code> and <code>beta</code> do in batch normalization, after a channel's activations are normalized to mean 0 and standard deviation 1?",
            "choices": [
                "They discard the normalized values and recompute them from scratch",
                "They let the network rescale and shift the normalized activations, so normalization is a starting point the network can undo if useful, not a hard constraint",
                "They control the batch size used during training",
                "They convert the activations back into 8-bit integers",
            ],
            "correct": 1,
            "explanation": "gamma (scale) and beta (shift) are applied on top of the 0/1-normalized activations, so the network can still recover a different mean/scale if that's actually useful.",
        },
    ],
    38: [
        {
            "question": "What is the key idea behind transfer learning, as motivated in this lesson?",
            "choices": [
                "Training a brand new network from scratch on the target task, but with more epochs",
                "Reusing a network already trained on a different, data-rich source task, since its early-layer features (edges, blobs, textures) are useful for almost any visual task",
                "Always using the largest possible batch size for the target task",
                "Manually copying weights between two unrelated tasks without any pretraining",
            ],
            "correct": 1,
            "explanation": "Transfer learning sidesteps data-starved target tasks by reusing generic low-level features learned on a data-rich source task, rather than learning everything from scratch.",
        },
        {
            "question": "What is the key difference between the 'frozen backbone' and 'fine-tuning' strategies in this lesson?",
            "choices": [
                "Frozen backbone trains a new classifier on fixed pretrained features; fine-tuning also updates the backbone itself, using a much smaller learning rate than the classifier head",
                "Fine-tuning never uses the pretrained weights at all",
                "Frozen backbone requires more target training images than fine-tuning",
                "There is no real difference between the two strategies",
            ],
            "correct": 0,
            "explanation": "Frozen backbone only trains the new head; fine-tuning also updates the backbone, but carefully, with a small backbone learning rate to avoid wrecking what it already learned.",
        },
        {
            "question": "Why does fine-tuning use a backbone learning rate roughly 30x smaller than the classifier head's learning rate?",
            "choices": [
                "To make training run faster",
                "A large backbone update from just 30 target examples would overfit those few images, discarding useful structure learned from the much larger source task -- the same catastrophic-forgetting failure as Lesson 35's overfitting curve",
                "Because PyTorch requires different learning rates for different layers by default",
                "Because the backbone has more parameters than the head",
            ],
            "correct": 1,
            "explanation": "A much smaller backbone learning rate protects the pretrained features from being wiped out by noisy updates from a tiny target dataset.",
        },
        {
            "question": "Why is the comparison between the toy frozen backbone and the real ImageNet-pretrained ResNet-18 not a perfectly controlled experiment, according to the lesson's caveat?",
            "choices": [
                "ResNet-18 was never actually trained on ImageNet",
                "ImageNet's 1,000 classes already include cats and several vehicle types, unlike the toy source task (only airplanes/automobiles), so part of ResNet-18's edge comes from having directly seen the target classes",
                "The toy backbone is actually larger than ResNet-18",
                "The comparison uses different target images for each strategy",
            ],
            "correct": 1,
            "explanation": "Since ImageNet already contains cat and vehicle photos, some of ResNet-18's advantage is direct class overlap, not purely richer general-purpose pretraining.",
        },
    ],
    39: [
        {
            "question": "What does a saliency map visualize?",
            "choices": [
                "The gradient of the predicted class score with respect to every input pixel -- large-magnitude pixels are ones a small change to would most affect the prediction",
                "The raw pixel values of the input image, unchanged",
                "The convolutional kernel weights themselves",
                "The class label the model predicts, overlaid as text",
            ],
            "correct": 0,
            "explanation": "A pixel with large gradient magnitude is one the network's prediction is most sensitive to -- a pixel it's 'looking at'.",
        },
        {
            "question": "Why are Grad-CAM heatmaps coarser (lower-resolution) than saliency maps?",
            "choices": [
                "Grad-CAM operates on a convolutional layer's feature map, which has already been downsampled by pooling layers, rather than on individual input pixels",
                "Grad-CAM only works on grayscale images",
                "Grad-CAM intentionally blurs its output for aesthetic reasons",
                "Saliency maps are actually coarser, not Grad-CAM",
            ],
            "correct": 0,
            "explanation": "Grad-CAM's resolution matches whichever conv layer it uses, which is coarser than the full input resolution a saliency map operates at.",
        },
        {
            "question": "On *clean* test images (no marker present), why does Grad-CAM still catch <code>model_shortcut</code>'s reliance on the marker far more reliably than raw saliency does?",
            "choices": [
                "Grad-CAM pools gradients over an entire feature-map channel before localizing, smoothing out pixel-level noise, while saliency is exactly as noisy as the raw per-pixel gradient",
                "Saliency maps cannot be computed on clean images at all",
                "Grad-CAM was specifically trained to detect markers",
                "There is actually no difference between the two methods on clean images",
            ],
            "correct": 0,
            "explanation": "The lesson reports Grad-CAM's peak lands in the marker region 92% of the time on clean images versus only 56% for saliency, attributed to Grad-CAM's channel-level pooling smoothing out pixel noise.",
        },
        {
            "question": "What does t-SNE do differently from a linear projection like PCA (Lesson 6)?",
            "choices": [
                "t-SNE requires labeled data while PCA does not",
                "t-SNE explicitly optimizes to preserve local neighborhoods in the projection, rather than using a single linear projection",
                "PCA can only be applied to images, never to feature vectors",
                "t-SNE and PCA are mathematically identical operations",
            ],
            "correct": 1,
            "explanation": "t-SNE emphasizes preserving which points are close to which other points locally, which is why same-class images can end up visibly clustered even without ever seeing their labels.",
        },
    ],
    40: [
        {
            "question": "What is the core idea of a sliding-window detector, as built in this lesson?",
            "choices": [
                "Run a classifier at every location (and scale) in the image, treating each window as its own yes/no classification problem",
                "Directly regress bounding box coordinates from the whole image in one forward pass",
                "Cluster pixels by color to find object boundaries",
                "Use only the image's frequency-domain representation to locate objects",
            ],
            "correct": 0,
            "explanation": "A window classifier answers 'face or not, at this exact window,' and sliding it across every position turns that per-window classifier into a detector.",
        },
        {
            "question": "In this lesson's box convention, how is each detected face represented?",
            "choices": [
                "By its top-left corner and width/height",
                "By its center coordinates (cx, cy) and width/height",
                "By the coordinates of all four corners",
                "By a single pixel with no size information",
            ],
            "correct": 1,
            "explanation": "Boxes here are stored as (cx, cy, w, h), with the center reported directly rather than a top-left corner.",
        },
        {
            "question": "After thresholding the sliding-window score map, why does a single true face typically produce a *cluster* of detections rather than one?",
            "choices": [
                "Because the window classifier is applied only once per image",
                "Because every window that overlaps a face heavily enough scores above threshold, not just the single best-centered one",
                "Because faces are always detected twice due to a bug in the scoring function",
                "Because thresholding always produces exactly one detection per object",
            ],
            "correct": 1,
            "explanation": "Many nearby, overlapping windows all score above threshold near a real face, producing a cluster of near-duplicate detections.",
        },
        {
            "question": "What does non-maximum suppression (NMS) do to resolve duplicate detections?",
            "choices": [
                "It averages all overlapping detections into one box",
                "It repeatedly keeps the highest-scoring remaining detection and discards every other detection that overlaps it by more than an IoU threshold",
                "It discards every detection above the score threshold",
                "It requires the user to manually click the correct detection",
            ],
            "correct": 1,
            "explanation": "NMS greedily keeps the best-scoring box in each cluster and suppresses its highly-overlapping neighbors, collapsing duplicates down to one detection per object.",
        },
    ],
    41: [
        {
            "question": "How does this lesson's detector represent the box it predicts?",
            "choices": [
                "Four numbers (cx, cy, width, height), normalized to [0, 1] by image size",
                "A pixel-wise segmentation mask",
                "A class label only, with no location information",
                "Four corner coordinates in absolute pixel units",
            ],
            "correct": 0,
            "explanation": "The network's regression head outputs (cx, cy, w, h) through a sigmoid, keeping every value in the valid normalized [0, 1] range.",
        },
        {
            "question": "Why is IoU used to evaluate the detector instead of just looking at how close the four predicted numbers are to the true ones?",
            "choices": [
                "IoU is faster to compute than any numeric comparison",
                "IoU directly measures how much the predicted and true boxes overlap, which is the metric that actually matters for detection quality, not raw numeric closeness",
                "IoU is only used during training, never for evaluation",
                "The four box numbers cannot be compared directly for any reason",
            ],
            "correct": 1,
            "explanation": "Two boxes can have similar-looking coordinates but very different overlap (or vice versa), so IoU is the metric that reflects what actually matters for localization.",
        },
        {
            "question": "Why can this lesson's simple regression detector only ever predict exactly one object per image?",
            "choices": [
                "PyTorch cannot output more than 4 numbers from a single network",
                "Its output is a fixed-size vector of exactly 4 numbers, which can only describe one box, unlike real scenes with a variable, unknown number of objects",
                "It was never trained on multi-object scenes",
                "The sigmoid activation limits the network to one prediction",
            ],
            "correct": 1,
            "explanation": "A fixed 4-number output has no way to represent a variable number of objects; real multi-object detectors need a different architecture (grid cells, region proposals, etc.).",
        },
        {
            "question": "What is the key architectural difference between two-stage (R-CNN family) and single-stage (YOLO, SSD) detectors?",
            "choices": [
                "Two-stage detectors first generate region proposals and then classify/regress each one; single-stage detectors skip proposals and predict boxes directly from a grid in one forward pass",
                "Single-stage detectors always use more parameters than two-stage detectors",
                "Two-stage detectors cannot use convolutional networks",
                "There is no real architectural difference; the terms are interchangeable",
            ],
            "correct": 0,
            "explanation": "Two-stage methods (e.g. Faster R-CNN) propose-then-classify; single-stage methods (YOLO, SSD) predict boxes and classes directly per grid cell in a single pass, trading some accuracy for speed.",
        },
    ],
}
