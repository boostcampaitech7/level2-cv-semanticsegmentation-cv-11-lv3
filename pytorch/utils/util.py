def set_seed(seed):
    '''
    랜덤 시드 설정
    
    Args:
        seed (int): 랜덤 시드 값
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def label2rgb(label) -> np.ndarray:
    '''
    클래스 마스크를 RGB 이미지로 변환
    
    Args:
        label (np.ndarray): 클래스별 바이너리 마스크 배열
    
    Returns:
        np.ndarray: RGB 이미지 배열
    '''
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
    return image

def find_file(root_path, extension: str) -> set:
    '''
    특정 확장자 파일 검색
    
    Args:
        root_path (str): 루트 디렉토리 경로
        extension (str): 파일 확장자 (예: '.png', '.json')
    
    Returns:
        set: 파일 경로 집합
    '''
    result = {
        os.path.relpath(os.path.join(root, fname), start=root_path)
        for root, _dirs, files in os.walk(root_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == extension
    }
    return result

def save_best(model, save_dir, cur_fold, file_name='best.pt'):
    '''
    모델 저장
    
    Args:
        model (torch.nn.Module): PyTorch 모델
        save_dir (str): 저장할 디렉토리 경로
        cur_fold (int): Fold 번호
        file_name (str): 파일명
    '''
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'{cur_fold}_' + file_name
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)

def save_ckpt(model, save_dir, epoch):
    '''
    가중치 저장
    
    Args:
        model (torch.nn.Module): PyTorch 모델
        save_dir (str): 저장할 디렉토리 경로
        epoch (int): 에폭 번호
    '''
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{epoch}_checkpoint.pth")
    torch.save(model.state_dict(), output_path)

def read_annotations(json_path) -> dict:
    '''
    annotation 데이터 읽기
    
    Args:
        json_path (str): JSON 파일 경로
    
    Returns:
        dict: annotation 데이터
    '''
    with open(json_path, 'r') as f:
        return json.load(f)['annotations']

def inference_save(filename_and_class, image_root, result_df, save_dir="inference_results", num_samples=10):
    '''
    인퍼런스 결과 저장
    
    Args:
        filename_and_class (list): 파일명 및 클래스 리스트
        image_root (str): 원본 이미지 경로
        result_df (pd.DataFrame): 결과 데이터프레임
        save_dir (str): 저장 디렉토리
        num_samples (int): 샘플 개수
    '''
    os.makedirs(save_dir, exist_ok=True)
    classes = get_classes()
    image_names = list(set([x.split("_", 1)[1] for x in filename_and_class]))
    sample_indices = list(range(min(num_samples, len(image_names))))

    for idx in tqdm(sample_indices, desc="Saving inference results"):
        try:
            image_name = image_names[idx]
            image_path = os.path.join(image_root, image_name)
            name_only = list(image_name.split('/'))[-1]
            rles = result_df[result_df['image_name'] == name_only]['rle'].tolist()
            image = cv2.imread(image_path)

            if image is None:
                continue

            orig_h, orig_w = image.shape[:2]
            preds = [decode_rle_to_mask(rle, orig_h, orig_w) for rle in rles[:len(get_classes())]]
            preds = np.stack(preds, axis=0)
            pred_rgb = label2rgb(preds)
            viz = np.hstack((image, pred_rgb))

            if viz.dtype != np.uint8:
                viz = viz.astype(np.uint8)

            image_base_name = os.path.splitext(os.path.basename(image_name))[0]
            save_path = os.path.join(save_dir, f"{image_base_name}_infer.png")
            cv2.imwrite(save_path, viz)
        except Exception as e:
            print(f"오류 발생: {e}")

def inference_to_csv(filename_and_class, rles, path='./results', output_name="output.csv"):
    '''
    결과를 CSV로 저장
    
    Args:
        filename_and_class (list): 파일명 및 클래스 리스트
        rles (list): RLE 리스트
        path (str): 저장 경로
        output_name (str): CSV 파일명
    '''
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({"image_name": image_name, "class": classes, 'rle': rles})
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, output_name), index=False)
    return df

def get_classes():
    '''
    클래스 목록 반환
    
    Returns:
        list: 클래스 이름 리스트
    '''
    return [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

def get_CLASS2IND():
    '''
    클래스 이름 -> 인덱스 맵 반환
    
    Returns:
        dict: 클래스 이름에서 인덱스로의 매핑
    '''
    return {cls: i for i, cls in enumerate(get_classes())}

def get_IND2CLASS():
    '''
    인덱스 -> 클래스 이름 맵 반환
    
    Returns:
        dict: 인덱스에서 클래스 이름으로의 매핑
    '''
    return {i: cls for i, cls in enumerate(get_classes())}
