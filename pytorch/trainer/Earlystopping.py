
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