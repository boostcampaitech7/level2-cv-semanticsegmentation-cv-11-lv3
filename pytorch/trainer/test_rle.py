import numpy as np

def encode_mask_to_rle(mask):
    """
    바이너리 마스크를 RLE(Run-Length Encoding) 형식으로 인코딩.

    Args:
        mask (numpy.ndarray): 2D 바이너리 마스크 (0과 1로 구성된 numpy 배열)

    Returns:
        str: RLE 형식으로 인코딩된 문자열
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    """
    RLE(Run-Length Encoding) 문자열을 바이너리 마스크로 변환.

    Args:
        rle (str): RLE 형식으로 인코딩된 문자열
        height (int): 복원할 마스크의 높이
        width (int): 복원할 마스크의 너비

    Returns:
        numpy.ndarray: 디코딩된 2D 바이너리 마스크
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)