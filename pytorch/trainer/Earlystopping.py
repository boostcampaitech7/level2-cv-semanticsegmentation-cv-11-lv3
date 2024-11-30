class EarlyStopper:
    def __init__(self, patience=7, delta=0, verbose=False):
        """
        Args:
            patience (int): 성능 향상을 기다리는 에폭 수
            delta (float): 성능 향상의 최소 변화량
            verbose (bool): True일 경우 매번 성능 향상 시 메세지 출력
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score):
        """
        모델 성능 점수를 기반으로 Early Stopping 여부를 결정.

        Args:
            score (float): 현재 에폭에서의 성능 지표 (예: 검증 손실, 정확도 등)

        동작:
            - 만약 첫 호출인 경우, 주어진 점수를 기준으로 초기 최적 점수를 설정.
            - 새로운 점수가 이전 최적 점수보다 개선되지 않았다고 판단되는 경우, 카운터를 증가.
                - 개선 판단 기준: (현재 점수) < (최적 점수 + delta)
            - 카운터가 patience 값을 초과하면 early_stop 플래그를 True로 설정.
            - 점수가 개선된 경우, 최적 점수를 갱신하고 카운터를 0으로 초기화.

        Returns:
            - self.early_stop 플래그를 통해 Early Stopping 여부를 알 수 있음.
            - verbose 옵션이 활성화된 경우, 개선 여부와 카운터 상태를 출력.
        """

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0