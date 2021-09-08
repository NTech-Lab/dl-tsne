import cv2
import numpy as np
import insightface
import skimage
from insightface.app import FaceAnalysis
import argparse
import os



SRC = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)
SRC[:, 0] += 8.0


def detect_all(img):
    faces = app.get(img)
    res = []
    for face in faces:
        st = skimage.transform.SimilarityTransform()
        st.estimate(face["kps"], SRC)
        imgf = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        res.append(imgf)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", help="source directory", default="")
    parser.add_argument("-L", help="source relative paths", default="/dev/stdin")
    parser.add_argument("-T", help="target directory", default="")
    args = parser.parse_args()
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    with open(args.L) as f:
        for path in f:
            path = path.strip()
            image = cv2.imread(os.path.join(args.S, path))
            faces = detect_all(image)
            os.makedirs(os.path.join(args.T, path), exist_ok=True)
            for i, face in enumerate(faces):
                cv2.imwrite(os.path.join(args.T, path, f"{i}.png"), face)
