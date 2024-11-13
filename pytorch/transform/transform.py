import albumentations as A

def get_transform(arg:list) -> A.Compose:
    '''
    리스트를 받아서 Albumentation Compose 객체를 반환
    
    Args:
        arg (list) : Albumentation Augmentation 리스트
    
    Return:
        A.Compose 객체
    '''
    tf = A.Compose([
        *arg
    ])
    return tf
